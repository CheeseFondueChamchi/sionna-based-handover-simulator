"""
RRC State Machine and Handover Controller

Implements UE RRC procedures according to 3GPP TS 38.331:
- RRC State Machine (IDLE, CONNECTED, INACTIVE)
- Handover execution (Section 5.3.5.4)
- RLF detection and recovery (Section 5.3.10)
- RACH procedure modeling

Integrates with:
- PHY Abstraction for BLER-based RLF detection
- Measurement Manager for event evaluation
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Tuple
import logging
import os
import math
import random

import sys
from pathlib import Path
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from core.event_scheduler import EventScheduler, EventType, Timer
from rrc.measurement import MeasConfig, MeasResult, MeasurementManager, MeasEventType
from phy.phy_abstraction import RLFDetector, RLFState, PHYAbstraction

logger = logging.getLogger(__name__)


class RRCState(Enum):
    """RRC States from 3GPP TS 38.331"""
    IDLE = auto()          # RRC_IDLE
    CONNECTED = auto()     # RRC_CONNECTED
    INACTIVE = auto()      # RRC_INACTIVE (NR specific)


class HOState(Enum):
    """Handover execution states"""
    NONE = auto()          # No HO in progress
    PREPARING = auto()     # Source gNB preparing HO
    EXECUTING = auto()     # UE executing HO (received RRCReconfiguration)
    COMPLETING = auto()    # Completing HO at target


class RACHState(Enum):
    """Random Access procedure states"""
    IDLE = auto()
    PREAMBLE_TX = auto()   # Transmitting preamble
    WAITING_RAR = auto()   # Waiting for RAR
    MSG3_TX = auto()       # Transmitting Msg3
    CONTENTION_RES = auto() # Waiting for contention resolution
    SUCCESS = auto()
    FAILURE = auto()


class ReestablishmentState(Enum):
    """RRC Re-establishment states per 3GPP TS 38.331 Section 5.3.7"""
    NONE = auto()              # Not in re-establishment
    INITIATED = auto()         # RLF detected, T311 started
    CELL_SELECTION = auto()    # Performing cell selection
    RACH_TO_TARGET = auto()    # RACH to selected cell
    WAITING_RESPONSE = auto()  # Waiting for RRCReestablishment
    SUCCESS = auto()           # Re-establishment successful
    FAILURE = auto()           # Re-establishment failed (T311 expired)


@dataclass
class RACHConfig:
    """RACH configuration for handover"""
    preamble_tx_max: int = 10             # Maximum preamble transmissions
    preamble_initial_power_dbm: float = -104  # Initial preamble power
    power_ramping_step_db: float = 2      # Power ramping step
    ra_response_window_ms: float = 10     # RAR window
    contention_resolution_timer_ms: float = 64  # Contention resolution


@dataclass
class HandoverResult:
    """Result of a handover attempt"""
    success: bool
    source_cell: int
    target_cell: int
    start_time: float
    end_time: float
    duration_ms: float
    failure_reason: Optional[str] = None
    rach_attempts: int = 0


@dataclass
class CellSelectionConfig:
    """Cell selection configuration per 3GPP TS 38.304"""
    s_intra_search_p: float = -140.0   # RSRP threshold for intra-freq search (per-RE RSRP)
    s_non_intra_search_p: float = -140.0  # RSRP threshold for inter-freq search (per-RE RSRP)
    q_rxlev_min: float = -140.0        # Minimum required RSRP (per-RE RSRP)
    q_qual_min: float = -20.0          # Minimum required RSRQ (dB)
    t_reselection_s: float = 1.0       # Reselection timer
    cell_selection_timeout_ms: float = 1000.0  # Max time for cell selection


@dataclass
class CellCandidate:
    """Candidate cell for re-establishment"""
    cell_id: int
    rsrp_dbm: float
    rsrq_db: float = -20.0
    sinr_db: float = -10.0
    s_criterion_met: bool = False
    rank: int = 0


@dataclass
class HODelayConfig:
    """Handover delay configuration for realistic timing.

    Tuned 2026-05-08 against KTX field log: median sim-vs-field HO bias was
    -0.30s (sim early). Bumping the sub-component defaults to typical vendor
    UE values pushes the realized HO completion ~150 ms later, eliminating
    most of the early-HO bias. Source-side guidance:
      - RRC HO Cmd processing: 30-50 ms (UE parsing + L1 commit)
      - Target SSB sync: 40-80 ms (SSB period 20 ms × 2-4 attempts at HST)
      - TA estimation: 5-10 ms
    Total realistic: 100-200 ms (vs old default 40-100 ms).
    """
    rrc_reconfiguration_processing_ms: float = 40.0  # UE processing time
    target_cell_sync_ms: float = 60.0                # Target cell synchronization
    timing_advance_estimation_ms: float = 10.0       # TA estimation
    rach_preamble_period_ms: float = 1.0             # RACH period
    total_min_delay_ms: float = 100.0                # Minimum total delay
    total_max_delay_ms: float = 200.0                # Maximum total delay


@dataclass
class UEContext:
    """UE context information"""
    ue_id: int
    serving_cell_id: Optional[int] = None
    target_cell_id: Optional[int] = None
    
    # Position
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_mps: float = 0.0
    
    # RRC state
    rrc_state: RRCState = RRCState.IDLE
    ho_state: HOState = HOState.NONE
    rach_state: RACHState = RACHState.IDLE
    
    # Measurements (latest)
    serving_rsrp_dbm: float = -140.0
    serving_sinr_db: float = -20.0
    serving_bler: float = 1.0
    
    # Statistics
    total_handovers: int = 0
    successful_handovers: int = 0
    failed_handovers: int = 0
    rlf_count: int = 0
    ping_pong_count: int = 0  # HO back to previous cell within threshold


class UERRCController:
    """
    UE RRC Controller implementing 3GPP TS 38.331.
    
    Manages:
    - RRC state transitions
    - Measurement processing and reporting
    - Handover execution
    - RLF detection and recovery (BLER-based)
    - Timer management
    
    Integration with PHY:
    - Receives SINR/BLER updates from channel calculator
    - Uses RLFDetector for out-of-sync/in-sync indication
    - Triggers RLF when T310 expires
    """
    
    # 3GPP Timer values (ms)
    VALID_T300 = [100, 200, 300, 400, 600, 1000, 1500, 2000]
    VALID_T301 = [100, 200, 300, 400, 600, 1000, 1500, 2000]
    VALID_T304 = [50, 100, 150, 200, 500, 1000, 2000]
    VALID_T310 = [0, 50, 100, 200, 500, 1000, 2000]
    VALID_T311 = [1000, 3000, 5000, 10000, 15000, 20000, 30000]
    
    def __init__(self, 
                 ue_id: int,
                 scheduler: EventScheduler,
                 meas_configs: List[MeasConfig],
                 rlf_config: Dict,
                 timer_config: Dict,
                 rach_config: Optional[RACHConfig] = None):
        """
        Initialize UE RRC Controller.
        
        Args:
            ue_id: UE identifier
            scheduler: Event scheduler
            meas_configs: List of measurement configurations
            rlf_config: RLF detection configuration
            timer_config: RRC timer configuration
            rach_config: RACH configuration
        """
        self.ue_id = ue_id
        self.scheduler = scheduler
        self.logger = logging.getLogger(f"RRC.UE{ue_id}")
        
        # UE context
        self.context = UEContext(ue_id=ue_id)
        
        # Measurement manager
        self.meas_manager = MeasurementManager(meas_configs, scheduler)
        self.meas_manager.on_measurement_report = self._on_measurement_event
        
        # RLF detector with BLER-based detection
        self.rlf_detector = RLFDetector(
            n310=rlf_config.get('N310', 1),
            n311=rlf_config.get('N311', 2),  # HST default, see UEStateMachineConfig
            t310_ms=timer_config.get('T310_ms', 1000),
            qout_bler=rlf_config.get('Qout_bler', 0.10),
            qin_bler=rlf_config.get('Qin_bler', 0.02),
            scheduler=scheduler
        )
        self.rlf_detector.on_t310_start = self._on_t310_start
        self.rlf_detector.on_t310_stop = self._on_t310_stop
        self.rlf_detector.on_rlf_detected = self._on_rlf_detected
        
        # Timers
        self._init_timers(timer_config)
        
        # RACH
        self.rach_config = rach_config or RACHConfig()
        self.rach_attempt_count = 0
        self.rach_power_dbm = self.rach_config.preamble_initial_power_dbm
        
        # Handover tracking
        self._ho_start_time: Optional[float] = None
        self._previous_cells: List[int] = []  # For ping-pong detection
        self._ping_pong_threshold_s = 5.0  # Ping-pong if HO back within this time
        
        # Event callbacks (set by simulator)
        self.on_handover_start: Optional[Callable] = None
        self.on_handover_complete: Optional[Callable] = None
        self.on_handover_failure: Optional[Callable] = None
        self.on_rlf: Optional[Callable] = None
        self.on_rrc_state_change: Optional[Callable] = None
        self.on_reestablishment_complete: Optional[Callable] = None
        self.on_reestablishment_failure: Optional[Callable] = None

        # Results tracking
        self.handover_results: List[HandoverResult] = []
        self.rlf_events: List[Dict] = []

        # Per 3GPP: A2 must fire before A5 is evaluated (inter-freq gating)
        self._a2_gate_active: bool = False

        # Re-establishment state tracking
        self._reestablishment_state = ReestablishmentState.NONE
        self._reestablishment_cause: Optional[str] = None
        self._reestablishment_source_cell: Optional[int] = None
        self._reestablishment_target_cell: Optional[int] = None
        self._cell_selection_start_time: Optional[float] = None
        self.reestablishment_events: List[Dict] = []

        # Last neighbor measurements (updated by process_channel_update)
        self._last_neighbor_measurements: List[Dict] = []

        # Cell selection configuration
        self._cell_selection_config = CellSelectionConfig()

        # HO delay configuration
        self._ho_delay_config = HODelayConfig()

        # ── HOF Classification (3GPP TS 38.300 §15.5 MRO) ──
        self._tstore_ue_cntxt_ms: float = 1000.0  # "short time" threshold
        self._tpp_ms: float = 1000.0               # Ping-pong window
        self._ho_history: List[Dict] = []           # Rolling HO history
        self._ho_history_max: int = 20
        self.hof_classifications: List[Dict] = []   # Classification results
        self._rlf_context: Optional[Dict] = None    # Stored at RLF for deferred classification

        self.logger.info(f"UE RRC Controller initialized with realistic HO/RLF recovery")
    
    def _init_timers(self, timer_config: Dict):
        """Initialize RRC timers"""
        self.timers: Dict[str, Timer] = {}
        
        # T300: RRC Connection Request
        self.timers['T300'] = Timer(
            'T300', timer_config.get('T300_ms', 1000),
            self.scheduler, self._on_t300_expire
        )
        
        # T301: RRC Connection Re-establishment Request
        self.timers['T301'] = Timer(
            'T301', timer_config.get('T301_ms', 1000),
            self.scheduler, self._on_t301_expire
        )
        
        # T304: Handover execution
        self.timers['T304'] = Timer(
            'T304', timer_config.get('T304_ms', 200),
            self.scheduler, self._on_t304_expire
        )
        
        # T311: RRC re-establishment
        self.timers['T311'] = Timer(
            'T311', timer_config.get('T311_ms', 10000),
            self.scheduler, self._on_t311_expire
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # Measurement Processing
    # ═══════════════════════════════════════════════════════════════════
    
    def process_channel_update(self, 
                               serving_rsrp_dbm: float,
                               serving_sinr_db: float,
                               serving_bler: float,
                               neighbors: List[Dict],
                               current_time: float,
                               mcs_index: int = 10):
        """
        Process channel update from PHY layer.
        
        This is the main entry point called every measurement period.
        
        Args:
            serving_rsrp_dbm: Serving cell RSRP
            serving_sinr_db: Serving cell SINR
            serving_bler: Serving cell BLER
            neighbors: List of neighbor measurements [{cell_id, rsrp, sinr, bler}, ...]
            current_time: Current simulation time
            mcs_index: Current MCS index for BLER calculation
        """
        if self.context.rrc_state != RRCState.CONNECTED:
            return
        
        # Update context
        self.context.serving_rsrp_dbm = serving_rsrp_dbm
        self.context.serving_sinr_db = serving_sinr_db
        self.context.serving_bler = serving_bler

        # Cache neighbor measurements for cell selection during re-establishment
        self._last_neighbor_measurements = neighbors.copy() if neighbors else []
        
        # === Step 1: RLF Detection (BLER-based) ===
        # Per 3GPP TS 38.331: T310 must NOT start while T304 (HO), T300, T301,
        # T311, or T316 is running. OOS/IS counting still proceeds.
        ho_in_progress = (self.context.ho_state != HOState.NONE)
        self.rlf_detector.process_measurement(
            serving_sinr_db, mcs_index, suppress_t310=ho_in_progress,
            current_time_s=current_time
        )
        
        # === Step 2: Measurement Event Evaluation (frequency-aware) ===
        # Per 3GPP: A3 = intra-freq (same freq NR only), A5 = inter-freq (diff freq NR only)
        # LTE cells are excluded from HO candidates in all events.
        if self.context.ho_state == HOState.NONE and not self.rlf_detector.state.rlf_detected:
            serving_meas = MeasResult(
                cell_id=self.context.serving_cell_id,
                rsrp_dbm=serving_rsrp_dbm,
                sinr_db=serving_sinr_db,
                timestamp=current_time
            )

            serving_freq = getattr(self, '_serving_frequency_ghz', 3.5)
            triggered_reports = []

            # Per 3GPP TS 38.133 §8.1: during OOS/Gray, measurements are
            # unreliable — do NOT cancel existing TTTs.
            from rrc.rrc_types import RadioLinkStatus
            rl_status = self.rlf_detector.state.radio_link_status
            freeze_cancel = (rl_status != RadioLinkStatus.IN_SYNC)

            for meas_id, config in self.meas_manager.configs.items():
                # Filter neighbors by event type and frequency
                if config.event_type == MeasEventType.A3:
                    # Intra-freq: same-frequency NR neighbors only
                    freq_neighbors = [n for n in neighbors
                                     if abs(n.get('frequency_ghz', 3.5) - serving_freq) < 0.01
                                     and not n.get('is_lte', False)]
                elif config.event_type == MeasEventType.A2:
                    # A2: serving-cell-only event per 3GPP TS 38.331 sec 5.5.4.2
                    # Pass serving as single TTT target (no per-neighbor dimension)
                    neighbor_meas_a2 = [serving_meas]
                    triggered = self.meas_manager.process_measurements_single(
                        meas_id, serving_meas, neighbor_meas_a2, current_time,
                        freeze_cancel=freeze_cancel
                    )
                    if triggered:
                        triggered_reports.extend(triggered)
                        self._a2_gate_active = True

                    # Check A2 leaving condition: Ms - Hys > Thresh
                    a2_leaving = (serving_rsrp_dbm - config.hysteresis > config.threshold1)
                    if a2_leaving and self._a2_gate_active:
                        self._a2_gate_active = False
                        # Reset A5 TTT states since inter-freq measurement stops
                        for mid, cfg in self.meas_manager.configs.items():
                            if cfg.event_type == MeasEventType.A5:
                                self.meas_manager.ttt_states.get(mid, {}).clear()
                    continue  # A2 handled completely
                elif config.event_type == MeasEventType.A5:
                    # Inter-freq: GATED by A2 per 3GPP TS 38.331
                    if not self._a2_gate_active:
                        # A5 not yet active - clear any stale TTT and skip
                        self.meas_manager.ttt_states.get(meas_id, {}).clear()
                        continue
                    # Different-frequency NR neighbors only
                    freq_neighbors = [n for n in neighbors
                                     if abs(n.get('frequency_ghz', 3.5) - serving_freq) >= 0.01
                                     and not n.get('is_lte', False)]
                else:
                    freq_neighbors = [n for n in neighbors if not n.get('is_lte', False)]

                neighbor_meas = [
                    MeasResult(
                        cell_id=n['cell_id'],
                        rsrp_dbm=n.get('rsrp', -140),
                        sinr_db=n.get('sinr', -20),
                        timestamp=current_time
                    )
                    for n in freq_neighbors
                ]

                triggered = self.meas_manager.process_measurements_single(
                    meas_id, serving_meas, neighbor_meas, current_time,
                    freeze_cancel=freeze_cancel
                )
                if triggered:
                    triggered_reports.extend(triggered)

            # === Update measurement_events from meas_manager (per-cell TTT) ===
            from rrc.rrc_types import EventState
            updated_events = {}
            for meas_id, config in self.meas_manager.configs.items():
                evt_name = config.event_type.name  # "A2", "A3", "A5", etc.
                ttt_ms = config.time_to_trigger_ms
                ttt_states = self.meas_manager.ttt_states.get(meas_id, {})

                # Find the cell with most TTT progress (closest to report)
                best_cell_id = None
                best_elapsed = -1.0
                for cell_id, ttt_state in ttt_states.items():
                    elapsed = current_time - ttt_state.start_time
                    if elapsed > best_elapsed:
                        best_elapsed = elapsed
                        best_cell_id = cell_id

                es = EventState(event_type=evt_name)
                if best_cell_id is not None:
                    es.triggered = True
                    es.target_cell_id = best_cell_id
                    remaining = max(0.0, ttt_ms - best_elapsed * 1000.0)
                    es.time_to_trigger_remaining = remaining
                    es.report_sent = False
                else:
                    es.triggered = False

                # Check if any cell just triggered a report this step
                for _mid, t_cells in triggered_reports:
                    if _mid == meas_id and t_cells:
                        es.triggered = True
                        es.report_sent = True
                        es.target_cell_id = t_cells[0]
                        es.time_to_trigger_remaining = 0.0

                updated_events[evt_name] = es

            self.rlf_detector.state.measurement_events = updated_events

            # Handle triggered measurement reports
            for meas_id, triggered_cells in triggered_reports:
                # Build full neighbor_meas for _handle_measurement_report (NR only)
                all_nr_neighbor_meas = [
                    MeasResult(
                        cell_id=n['cell_id'],
                        rsrp_dbm=n.get('rsrp', -140),
                        sinr_db=n.get('sinr', -20),
                        timestamp=current_time
                    )
                    for n in neighbors if not n.get('is_lte', False)
                ]
                self._handle_measurement_report(meas_id, triggered_cells,
                                               serving_meas, all_nr_neighbor_meas)
    
    def _on_measurement_event(self, meas_id: int, cell_ids: List[int]):
        """Callback from MeasurementManager when event is triggered"""
        self.logger.debug(f"Measurement event {meas_id} triggered for cells {cell_ids}")
    
    def _handle_measurement_report(self, meas_id: int, triggered_cells: List[int],
                                   serving: MeasResult, neighbors: List[MeasResult]):
        """
        Handle measurement report (TTT expired).
        
        In a real system, this would send MeasurementReport to gNB.
        Here, we simulate gNB's handover decision.
        """
        if not triggered_cells:
            return
        
        config = self.meas_manager.configs.get(meas_id)
        if not config:
            return
        
        # Find best target cell
        best_cell = triggered_cells[0]
        best_neighbor = next((n for n in neighbors if n.cell_id == best_cell), None)
        
        if best_neighbor is None:
            return
        
        self.logger.info(f"Measurement Report: serving={self.context.serving_cell_id} "
                        f"(RSRP={serving.rsrp_dbm:.1f}dBm), "
                        f"target={best_cell} (RSRP={best_neighbor.rsrp_dbm:.1f}dBm)")
        
        # Simulate gNB handover decision (always accept in this simulation)
        # In real system, gNB would evaluate and send RRCReconfiguration
        self._initiate_handover(best_cell)
    
    # ═══════════════════════════════════════════════════════════════════
    # Handover Execution (3GPP TS 38.331 Section 5.3.5.4)
    # ═══════════════════════════════════════════════════════════════════
    
    def _initiate_handover(self, target_cell_id: int):
        """
        Initiate handover to target cell.

        This simulates receiving RRCReconfiguration with reconfigurationWithSync.
        Includes realistic delays for:
        - RRCReconfiguration message processing
        - Target cell synchronization
        - Timing advance estimation
        """
        if self.context.ho_state != HOState.NONE:
            self.logger.warning(f"HO already in progress, ignoring HO to {target_cell_id}")
            return
        
        if self.context.serving_cell_id == target_cell_id:
            self.logger.warning(f"Target cell same as serving, ignoring")
            return
        
        source_cell = self.context.serving_cell_id
        
        self.logger.info(f"=== HANDOVER START: {source_cell} -> {target_cell_id} ===")
        
        # Update state
        self.context.ho_state = HOState.EXECUTING
        self.context.target_cell_id = target_cell_id
        self._ho_start_time = self.scheduler.current_time
        
        # Reset RLF detector (per 3GPP spec)
        self.rlf_detector.reset()
        
        # Start T304 (handover execution timer)
        self.timers['T304'].start()
        
        # Callback
        if self.on_handover_start:
            self.on_handover_start(self.ue_id, source_cell, target_cell_id)

        # === NEW: Realistic HO delay before RACH ===
        ho_delay_config = getattr(self, '_ho_delay_config', HODelayConfig())

        # Calculate total delay (with some randomness)
        base_delay = (ho_delay_config.rrc_reconfiguration_processing_ms +
                      ho_delay_config.target_cell_sync_ms +
                      ho_delay_config.timing_advance_estimation_ms)

        # Add randomness within bounds
        total_delay = random.uniform(
            max(base_delay, ho_delay_config.total_min_delay_ms),
            ho_delay_config.total_max_delay_ms
        )

        self.logger.debug(f"HO processing delay: {total_delay:.1f}ms before RACH")

        # Schedule RACH start after processing delay
        self.scheduler.schedule(
            delay=total_delay / 1000.0,
            event_type=EventType.HO_EXECUTION,
            callback=self._start_rach_to_target,
            description=f"HO RACH start after {total_delay:.1f}ms delay"
        )
    
    def _start_rach_to_target(self):
        """Start Random Access procedure to target cell"""
        self.context.rach_state = RACHState.PREAMBLE_TX
        self.rach_attempt_count = 0
        self.rach_power_dbm = self.rach_config.preamble_initial_power_dbm
        
        # Schedule RACH attempt
        self._attempt_rach()
    
    def _attempt_rach(self):
        """Attempt RACH preamble transmission"""
        self.rach_attempt_count += 1
        
        if self.rach_attempt_count > self.rach_config.preamble_tx_max:
            # Max attempts exceeded - RACH failure
            self._on_rach_failure()
            return
        
        self.logger.debug(f"RACH attempt {self.rach_attempt_count}/{self.rach_config.preamble_tx_max} "
                         f"at power {self.rach_power_dbm:.1f}dBm")
        
        # Simulate RACH success probability based on SINR
        # In real simulation, this would depend on target cell channel quality
        rach_success_prob = self._calculate_rach_success_probability()

        if random.random() < rach_success_prob:
            # RACH success
            self._on_rach_success()
        else:
            # RACH failed - ramp up power and retry
            self.rach_power_dbm += self.rach_config.power_ramping_step_db
            
            # Schedule next attempt after RAR window
            self.scheduler.schedule(
                delay=self.rach_config.ra_response_window_ms / 1000.0,
                event_type=EventType.RACH_PREAMBLE_TX,
                callback=self._attempt_rach,
                description=f"RACH retry {self.rach_attempt_count + 1}"
            )
    
    def _calculate_rach_success_probability(self) -> float:
        """
        Calculate RACH success probability based on target cell SINR.

        Per 3GPP TS 38.141-1 Section 8.4:
        - PRACH preamble detection Pd >= 99% at SNR = -14 dB (Format 0)
        - Model: sigmoid function calibrated to this requirement

        Uses actual target cell SINR from last neighbor measurements.
        Power ramping improves effective SNR on each retry.
        """
        # Get target cell SINR from last neighbor measurements
        target_sinr_db = -20.0  # Default fallback (poor conditions)
        target_cell = self.context.target_cell_id
        if target_cell is not None and hasattr(self, '_last_neighbor_measurements'):
            for n in self._last_neighbor_measurements:
                if n.get('cell_id') == target_cell:
                    target_sinr_db = n.get('sinr', -20.0)
                    break

        # Power ramping benefit: each retry adds power_ramping_step_db
        ramp_gain = max(0, self.rach_attempt_count - 1) * self.rach_config.power_ramping_step_db
        effective_sinr = target_sinr_db + ramp_gain

        # Sigmoid model: P = 1 / (1 + exp(-k * (SINR - threshold)))
        # Calibration: Pd = 99% at SINR = -14 dB, Pd = 50% at -20 dB
        sinr_threshold = -20.0  # 50% detection point (dB)
        steepness = 0.77        # Gives Pd ~99% at -14 dB

        p_detection = 1.0 / (1.0 + math.exp(-steepness * (effective_sinr - sinr_threshold)))

        # Collision probability (simplified, low-load assumption)
        p_no_collision = 0.99

        success_prob = p_detection * p_no_collision

        self.logger.debug(
            f"RACH prob: target_sinr={target_sinr_db:.1f}dB, "
            f"ramp={ramp_gain:.1f}dB, eff_sinr={effective_sinr:.1f}dB, "
            f"P={success_prob:.3f}"
        )

        return success_prob
    
    def _on_rach_success(self):
        """RACH succeeded - complete handover"""
        self.context.rach_state = RACHState.SUCCESS
        
        self.logger.info(f"RACH success after {self.rach_attempt_count} attempts")
        
        # Stop T304
        self.timers['T304'].stop()
        
        # Complete handover
        self._complete_handover()
    
    def _on_rach_failure(self):
        """RACH failed after max attempts"""
        self.context.rach_state = RACHState.FAILURE
        
        self.logger.warning(f"RACH failed after {self.rach_attempt_count} attempts")
        
        # This will lead to T304 expiry and HO failure
        # (T304 is still running)
    
    def _complete_handover(self):
        """Complete successful handover"""
        # Guard against double-completion (e.g., RACH success + manual call)
        if self.context.ho_state == HOState.NONE:
            return
        source_cell = self.context.serving_cell_id
        target_cell = self.context.target_cell_id
        current_time = self.scheduler.current_time
        
        # Calculate HO duration
        ho_duration_ms = (current_time - self._ho_start_time) * 1000
        
        # Update context
        self.context.serving_cell_id = target_cell
        self.context.target_cell_id = None
        self.context.ho_state = HOState.NONE
        self.context.rach_state = RACHState.IDLE
        self.context.total_handovers += 1
        self.context.successful_handovers += 1
        
        # Track cell history (legacy)
        self._previous_cells.append(source_cell)
        if len(self._previous_cells) > 10:
            self._previous_cells.pop(0)
        
        # Reset RLF detector for new cell
        self.rlf_detector.reset()
        
        # Record result
        result = HandoverResult(
            success=True,
            source_cell=source_cell,
            target_cell=target_cell,
            start_time=self._ho_start_time,
            end_time=current_time,
            duration_ms=ho_duration_ms,
            rach_attempts=self.rach_attempt_count
        )
        self.handover_results.append(result)

        # ── HOF: Record HO history & check ping-pong (Case 4) ──
        self._record_ho_history_cb(
            timestamp=current_time,
            source=source_cell,
            target=target_cell,
            success=True,
            rach_attempts=self.rach_attempt_count,
            duration_ms=ho_duration_ms
        )
        self._check_ping_pong_cb(current_time, source_cell, target_cell)
        
        self.logger.info(f"=== HANDOVER COMPLETE: {source_cell} -> {target_cell} "
                        f"(duration={ho_duration_ms:.1f}ms, RACH attempts={self.rach_attempt_count}) ===")

        # ── Vendor gNB MRO (TS 38.473 / TS 28.541 NRM): register a post-HO
        # blacklist entry (target → source) on the measurement engine so the
        # UE will not immediately ping-pong back to `source_cell` while
        # serving=`target_cell`. No-op when the duration is 0.
        if hasattr(self.measurement_engine, "add_post_ho_blacklist"):
            try:
                self.measurement_engine.add_post_ho_blacklist(
                    source_cell, target_cell, current_time
                )
            except Exception:  # never let MRO bookkeeping abort an HO
                self.logger.debug("post-HO blacklist add failed", exc_info=True)

        # Callback
        if self.on_handover_complete:
            self.on_handover_complete(self.ue_id, source_cell, target_cell, result)
    
    def _on_t304_expire(self):
        """
        T304 expired - Handover failure per 3GPP TS 38.331 Section 5.3.5.6.

        Actions:
        1. Consider handover to have failed
        2. Trigger Radio Link Failure (RLF)
        3. Initiate RRC re-establishment procedure
        """
        source_cell = self.context.serving_cell_id
        target_cell = self.context.target_cell_id
        current_time = self.scheduler.current_time

        self.logger.error(f"=== T304 EXPIRED: HO FAILURE {source_cell} -> {target_cell} ===")
        self.logger.error(f"    Triggering RLF per 3GPP TS 38.331 Section 5.3.5.6")

        # ── HOF: Store RLF context BEFORE resetting HO state ──
        self._store_rlf_context_cb()

        # ── HOF: Record failed HO in history ──
        self._record_ho_history_cb(
            timestamp=current_time,
            source=source_cell,
            target=target_cell,
            success=False,
            rach_attempts=self.rach_attempt_count,
            duration_ms=(current_time - self._ho_start_time) * 1000,
            failure_reason="T304_EXPIRE"
        )

        # Record handover failure result
        result = HandoverResult(
            success=False,
            source_cell=source_cell,
            target_cell=target_cell,
            start_time=self._ho_start_time,
            end_time=current_time,
            duration_ms=(current_time - self._ho_start_time) * 1000,
            failure_reason="T304_EXPIRE_RLF",
            rach_attempts=self.rach_attempt_count
        )
        self.handover_results.append(result)

        # Update statistics
        self.context.total_handovers += 1
        self.context.failed_handovers += 1

        # Reset HO state
        self.context.ho_state = HOState.NONE
        self.context.target_cell_id = None
        self.context.rach_state = RACHState.IDLE

        # Record RLF event (T304 expiry triggers RLF)
        rlf_event = {
            'time': current_time,
            'serving_cell': source_cell,
            'target_cell': target_cell,
            'cause': 'T304_EXPIRE',
            'sinr_db': self.context.serving_sinr_db,
            'bler': self.context.serving_bler
        }
        self.rlf_events.append(rlf_event)
        self.context.rlf_count += 1

        # Callback for HO failure
        if self.on_handover_failure:
            self.on_handover_failure(self.ue_id, source_cell, target_cell, "T304_EXPIRE_RLF")

        # Callback for RLF
        if self.on_rlf:
            self.on_rlf(self.ue_id, source_cell, rlf_event)

        # Per 3GPP: T304 expiry triggers RLF -> initiate re-establishment
        self._initiate_reestablishment(cause="T304_EXPIRE")
    
    # ═══════════════════════════════════════════════════════════════════
    # RLF Detection and Recovery (3GPP TS 38.331 Section 5.3.10)
    # ═══════════════════════════════════════════════════════════════════
    
    def _on_t310_start(self):
        """T310 started - potential RLF"""
        self.logger.warning(f"T310 started - potential RLF detected "
                          f"(SINR={self.rlf_detector.state.last_sinr_db:.1f}dB, "
                          f"BLER={self.rlf_detector.state.last_bler:.2%})")
    
    def _on_t310_stop(self):
        """T310 stopped - recovered from potential RLF"""
        self.logger.info(f"T310 stopped - recovered from potential RLF "
                        f"(SINR={self.rlf_detector.state.last_sinr_db:.1f}dB)")
    
    def _on_rlf_detected(self):
        """
        RLF detected (T310 expired).
        
        Actions per 3GPP TS 38.331:
        1. Store UE AS context
        2. Stop all timers except T311
        3. Initiate RRC re-establishment
        """
        self.logger.error(f"=== RLF DETECTED === "
                         f"(SINR={self.rlf_detector.state.last_sinr_db:.1f}dB, "
                         f"BLER={self.rlf_detector.state.last_bler:.2%})")
        
        # ── HOF: Store RLF context for deferred classification ──
        self._store_rlf_context_cb()

        # Record RLF event
        rlf_event = {
            'time': self.scheduler.current_time,
            'serving_cell': self.context.serving_cell_id,
            'cause': 'T310_EXPIRE',
            'sinr_db': self.rlf_detector.state.last_sinr_db,
            'bler': self.rlf_detector.state.last_bler,
            'n310_counter': self.rlf_detector.state.n310_counter,
            'consecutive_oos': self.rlf_detector.state.consecutive_oos
        }
        self.rlf_events.append(rlf_event)
        
        # Update statistics
        self.context.rlf_count += 1
        
        # Callback
        if self.on_rlf:
            self.on_rlf(self.ue_id, self.context.serving_cell_id, rlf_event)
        
        # Stop ongoing handover if any
        if self.context.ho_state != HOState.NONE:
            self.timers['T304'].stop()
            self.context.ho_state = HOState.NONE
            self.context.target_cell_id = None

        # Initiate RRC re-establishment
        self._initiate_reestablishment(cause="T310_EXPIRE")
    
    def _initiate_reestablishment(self, cause: str = "RLF"):
        """
        Initiate RRC connection re-establishment per 3GPP TS 38.331 Section 5.3.7.

        Args:
            cause: Cause of re-establishment (T304_EXPIRE, T310_EXPIRE, OTHER)

        Actions:
        1. Store UE AS context (for security context)
        2. Stop all timers except T311
        3. Start T311
        4. Initiate cell selection
        5. If suitable cell found, send RRCReestablishmentRequest
        6. If T311 expires, go to IDLE
        """
        self.logger.info(f"=== RRC RE-ESTABLISHMENT INITIATED (cause={cause}) ===")

        # Store re-establishment cause
        self._reestablishment_cause = cause
        self._reestablishment_source_cell = self.context.serving_cell_id

        # Initialize re-establishment state
        self._reestablishment_state = ReestablishmentState.INITIATED

        # Stop all RRC timers except T311 (per 3GPP)
        for timer_name, timer in self.timers.items():
            if timer_name != 'T311' and timer.running:
                timer.stop()
                self.logger.debug(f"Stopped timer {timer_name} for re-establishment")

        # Reset RLF detector
        self.rlf_detector.reset()

        # Start T311 (RRC re-establishment timer)
        self.timers['T311'].start()
        self.logger.info(f"T311 started ({self.timers['T311'].duration_ms}ms) - searching for suitable cell")

        # Begin cell selection
        self._reestablishment_state = ReestablishmentState.CELL_SELECTION
        self._cell_selection_start_time = self.scheduler.current_time

        # Schedule periodic cell selection attempts
        self._schedule_cell_selection_attempt()

    def _schedule_cell_selection_attempt(self):
        """Schedule a cell selection attempt"""
        # Attempt cell selection every 100ms during T311
        self.scheduler.schedule(
            delay=0.1,  # 100ms between attempts
            event_type=EventType.RLF_RECOVERY,
            callback=self._attempt_cell_selection,
            description="Cell selection attempt"
        )

    def _attempt_cell_selection(self):
        """
        Attempt cell selection per 3GPP TS 38.304.

        Uses S-criterion: Srxlev = Qrxlevmeas - Qrxlevmin > 0
        """
        if self._reestablishment_state != ReestablishmentState.CELL_SELECTION:
            return

        # Check if T311 is still running
        if not self.timers['T311'].running:
            return

        # Get available cell measurements (from measurement manager cache)
        candidates = self._get_cell_candidates()

        if not candidates:
            self.logger.debug("No suitable cells found, will retry...")
            self._schedule_cell_selection_attempt()
            return

        # Find best cell meeting S-criterion
        best_cell = self._select_best_cell(candidates)

        if best_cell is not None:
            self.logger.info(f"Suitable cell found: {best_cell.cell_id} "
                            f"(RSRP={best_cell.rsrp_dbm:.1f}dBm)")
            self._send_rrc_reestablishment_request(best_cell)
        else:
            self.logger.debug("No cell meets S-criterion, will retry...")
            self._schedule_cell_selection_attempt()

    def _get_cell_candidates(self) -> List[CellCandidate]:
        """
        Get candidate cells for re-establishment.

        Returns list of cells with their measurements.
        """
        candidates = []

        # Get neighbor measurements from measurement manager
        # Note: In real implementation, UE would measure cells during T311
        # Here we use cached measurements from the last measurement period

        # Add serving cell as candidate (may have recovered)
        if self.context.serving_cell_id is not None:
            candidates.append(CellCandidate(
                cell_id=self.context.serving_cell_id,
                rsrp_dbm=self.context.serving_rsrp_dbm,
                sinr_db=self.context.serving_sinr_db
            ))

        # Add any neighbor cells that were measured
        # This would be populated by the simulator with actual neighbor measurements
        if hasattr(self, '_last_neighbor_measurements'):
            for n in self._last_neighbor_measurements:
                candidates.append(CellCandidate(
                    cell_id=n.get('cell_id'),
                    rsrp_dbm=n.get('rsrp', -140.0),
                    sinr_db=n.get('sinr', -20.0)
                ))

        return candidates

    def _select_best_cell(self, candidates: List[CellCandidate]) -> Optional[CellCandidate]:
        """
        Select best cell meeting S-criterion.

        S-criterion: Srxlev = Qrxlevmeas - Qrxlevmin > 0

        Args:
            candidates: List of candidate cells

        Returns:
            Best cell meeting S-criterion, or None
        """
        # Get cell selection config
        q_rxlev_min = getattr(self, '_cell_selection_config', CellSelectionConfig()).q_rxlev_min

        # Filter cells meeting S-criterion
        suitable = []
        for cell in candidates:
            s_rxlev = cell.rsrp_dbm - q_rxlev_min
            if s_rxlev > 0:
                cell.s_criterion_met = True
                cell.rank = int(cell.rsrp_dbm + 140)  # Simple ranking by RSRP
                suitable.append(cell)

        if not suitable:
            return None

        # Sort by RSRP (best first)
        suitable.sort(key=lambda c: c.rsrp_dbm, reverse=True)

        return suitable[0]

    def _send_rrc_reestablishment_request(self, target_cell: CellCandidate):
        """
        Send RRCReestablishmentRequest to target cell.

        Per 3GPP TS 38.331 Section 5.3.7.3:
        1. Perform RACH to target cell
        2. Send RRCReestablishmentRequest in Msg3
        3. Wait for RRCReestablishment or RRCReject

        Args:
            target_cell: Selected target cell
        """
        self._reestablishment_state = ReestablishmentState.RACH_TO_TARGET
        self._reestablishment_target_cell = target_cell.cell_id

        self.logger.info(f"Sending RRCReestablishmentRequest to cell {target_cell.cell_id}")

        # Initialize RACH for re-establishment
        self.context.rach_state = RACHState.PREAMBLE_TX
        self.rach_attempt_count = 0
        self.rach_power_dbm = self.rach_config.preamble_initial_power_dbm

        # Start RACH procedure
        self._attempt_reestablishment_rach()

    def _attempt_reestablishment_rach(self):
        """Attempt RACH for re-establishment"""
        self.rach_attempt_count += 1

        # Check T311 still running
        if not self.timers['T311'].running:
            self.logger.warning("T311 expired during RACH for re-establishment")
            return

        if self.rach_attempt_count > self.rach_config.preamble_tx_max:
            # RACH failed - try another cell
            self.logger.warning(f"RACH failed after {self.rach_attempt_count - 1} attempts, "
                               "trying another cell")
            self._reestablishment_state = ReestablishmentState.CELL_SELECTION
            self._schedule_cell_selection_attempt()
            return

        self.logger.debug(f"Re-establishment RACH attempt {self.rach_attempt_count}/"
                         f"{self.rach_config.preamble_tx_max}")

        # Simulate RACH success probability
        rach_success_prob = self._calculate_rach_success_probability()

        if random.random() < rach_success_prob:
            self._on_reestablishment_rach_success()
        else:
            # Retry after RAR window
            self.rach_power_dbm += self.rach_config.power_ramping_step_db
            self.scheduler.schedule(
                delay=self.rach_config.ra_response_window_ms / 1000.0,
                event_type=EventType.RACH_PREAMBLE_TX,
                callback=self._attempt_reestablishment_rach,
                description=f"Re-establishment RACH retry {self.rach_attempt_count + 1}"
            )

    def _on_reestablishment_rach_success(self):
        """RACH succeeded for re-establishment"""
        self._reestablishment_state = ReestablishmentState.WAITING_RESPONSE
        self.context.rach_state = RACHState.SUCCESS

        self.logger.info(f"RACH success for re-establishment after {self.rach_attempt_count} attempts")

        # Simulate network response delay (typically 10-50ms)
        response_delay_ms = 20.0

        self.scheduler.schedule(
            delay=response_delay_ms / 1000.0,
            event_type=EventType.RLF_RECOVERY,
            callback=self._on_reestablishment_response,
            description="RRCReestablishment response"
        )

    def _on_reestablishment_response(self):
        """Handle network response to RRCReestablishmentRequest"""
        # Check T311 still running
        if not self.timers['T311'].running:
            self.logger.warning("T311 expired waiting for re-establishment response")
            return

        # Simulate acceptance (could add rejection logic based on network conditions)
        accept_probability = 0.95

        if random.random() < accept_probability:
            self._complete_reestablishment_success()
        else:
            self.logger.warning("RRCReestablishment rejected, trying another cell")
            self._reestablishment_state = ReestablishmentState.CELL_SELECTION
            self._schedule_cell_selection_attempt()

    def _complete_reestablishment_success(self):
        """Complete successful RRC re-establishment"""
        # Stop T311
        self.timers['T311'].stop()

        # Update serving cell to target
        old_cell = self.context.serving_cell_id
        self.context.serving_cell_id = self._reestablishment_target_cell

        # Reset states
        self._reestablishment_state = ReestablishmentState.SUCCESS
        self.context.rach_state = RACHState.IDLE
        self.rlf_detector.reset()

        # Calculate re-establishment time
        reest_duration_ms = (self.scheduler.current_time - self._cell_selection_start_time) * 1000

        self.logger.info(f"=== RRC RE-ESTABLISHMENT COMPLETE ===")
        self.logger.info(f"    Source cell: {self._reestablishment_source_cell}")
        self.logger.info(f"    Target cell: {self._reestablishment_target_cell}")
        self.logger.info(f"    Duration: {reest_duration_ms:.1f}ms")
        self.logger.info(f"    Cause: {self._reestablishment_cause}")

        # Record re-establishment event
        reest_event = {
            'time': self.scheduler.current_time,
            'source_cell': self._reestablishment_source_cell,
            'target_cell': self._reestablishment_target_cell,
            'cause': self._reestablishment_cause,
            'duration_ms': reest_duration_ms,
            'rach_attempts': self.rach_attempt_count,
            'success': True
        }

        if not hasattr(self, 'reestablishment_events'):
            self.reestablishment_events = []
        self.reestablishment_events.append(reest_event)

        # ── HOF: Final classification (Cases 1-5) ──
        if self._reestablishment_target_cell is not None:
            self._classify_hof_on_reestablishment_cb(
                reest_cell_id=self._reestablishment_target_cell)

        # Invoke callback for deferred HO failure classification
        if self.on_reestablishment_complete:
            self.on_reestablishment_complete(
                self.ue_id,
                self._reestablishment_source_cell,
                self._reestablishment_target_cell,
                self._reestablishment_cause
            )

    def _on_t300_expire(self):
        """T300 expired - RRC Connection Request failed"""
        self.logger.warning("T300 expired - RRC Connection Request failed")
        self._set_rrc_state(RRCState.IDLE)
    
    def _on_t301_expire(self):
        """T301 expired - RRC Re-establishment Request failed"""
        self.logger.warning("T301 expired - RRC Re-establishment Request failed")
    
    def _on_t311_expire(self):
        """
        T311 expired - RRC re-establishment failed per 3GPP TS 38.331 Section 5.3.7.8.

        Actions:
        1. Abort re-establishment procedure
        2. Transition to RRC_IDLE
        3. Perform cell selection for camping
        4. Start connection setup from scratch (RACH, RRC Setup)
        """
        self.timers['T311'].stop()
        self.logger.error(f"=== T311 EXPIRED: RE-ESTABLISHMENT FAILED ===")
        self.logger.error(f"    UE transitioning to RRC_IDLE")

        # Record failed re-establishment
        reest_event = {
            'time': self.scheduler.current_time,
            'source_cell': getattr(self, '_reestablishment_source_cell', None),
            'target_cell': getattr(self, '_reestablishment_target_cell', None),
            'cause': getattr(self, '_reestablishment_cause', 'UNKNOWN'),
            'duration_ms': (self.scheduler.current_time -
                           getattr(self, '_cell_selection_start_time', self.scheduler.current_time)) * 1000,
            'rach_attempts': self.rach_attempt_count,
            'success': False
        }

        if not hasattr(self, 'reestablishment_events'):
            self.reestablishment_events = []
        self.reestablishment_events.append(reest_event)

        # ── HOF: T311 expiry = re-establishment failed entirely ──
        rlf_ctx = self._rlf_context or {}
        rlf_cause_orig = rlf_ctx.get('ho_in_progress', False)
        if rlf_cause_orig:
            hof_type = "T304_EXPIRY"
            cause = "T304 expiry + T311 expiry (re-estab failed entirely)"
        else:
            hof_type = "TOO_LATE"
            cause = "T310 expiry + T311 expiry (no cell found for re-estab)"
        result = {
            'hof_type': hof_type,
            'timestamp': self.scheduler.current_time,
            'rlf_cell_id': rlf_ctx.get('cell_id', -1),
            'reestablishment_cell_id': -1,
            'last_ho_source': -1,
            'last_ho_target': -1,
            'time_since_last_ho_ms': -1,
            'cause': cause
        }
        self.hof_classifications.append(result)
        self.logger.warning(f"*** HOF CLASSIFIED: {hof_type} *** | {cause}")
        self._rlf_context = None

        # Update state
        self._reestablishment_state = ReestablishmentState.FAILURE
        self.context.rach_state = RACHState.IDLE

        # Invoke callback so simulator can clear pending deferred classification
        if self.on_reestablishment_failure:
            self.on_reestablishment_failure(
                self.ue_id,
                getattr(self, '_reestablishment_source_cell', None),
                getattr(self, '_reestablishment_cause', 'T311_EXPIRE')
            )

        # Transition to IDLE
        self._set_rrc_state(RRCState.IDLE)

        # In IDLE, UE would perform cell selection and start new connection
        # This is simplified - in real system would trigger T300 for RRC Setup
        self.logger.info("UE in IDLE - would start RRC Connection Setup to camp on cell")

    # ═══════════════════════════════════════════════════════════════════
    # HOF Classification (3GPP TS 38.300 §15.5 MRO)
    # ═══════════════════════════════════════════════════════════════════

    def _record_ho_history_cb(self, timestamp: float, source: int,
                               target: int, success: bool,
                               rach_attempts: int = 0,
                               duration_ms: float = 0.0,
                               failure_reason: str = ""):
        """Record a HO event in rolling history buffer (callback mode)."""
        entry = {
            'timestamp': timestamp,
            'source_cell_id': source,
            'target_cell_id': target,
            'success': success,
            'rach_attempts': rach_attempts,
            'duration_ms': duration_ms,
            'failure_reason': failure_reason
        }
        self._ho_history.append(entry)
        if len(self._ho_history) > self._ho_history_max:
            self._ho_history = self._ho_history[-self._ho_history_max:]

    def _store_rlf_context_cb(self):
        """Store RLF context for deferred HOF classification."""
        self._rlf_context = {
            'time': self.scheduler.current_time,
            'cell_id': self.context.serving_cell_id,
            'ho_in_progress': self.context.ho_state != HOState.NONE,
            'target_cell_id': self.context.target_cell_id
        }

    def _classify_hof_on_reestablishment_cb(self, reest_cell_id: int):
        """
        Final HOF classification when re-establishment completes (callback mode).

        Implements the 3GPP TS 38.300 §15.5 decision tree.
        """
        current_time = self.scheduler.current_time
        rlf_ctx = self._rlf_context or {}
        rlf_time = rlf_ctx.get('time', current_time)
        rlf_cell = rlf_ctx.get('cell_id', -1)
        ho_was_in_progress = rlf_ctx.get('ho_in_progress', False)

        # Find last successful HO
        last_ho = None
        for entry in reversed(self._ho_history):
            if entry['success']:
                last_ho = entry
                break

        hof_type = "NONE"
        cause = ""

        # Case 5: T304 expiry
        if ho_was_in_progress:
            hof_type = "T304_EXPIRY"
            cause = (f"T304 expired during HO {rlf_cell}->"
                     f"{rlf_ctx.get('target_cell_id', '?')}, "
                     f"re-estab at {reest_cell_id}")
        # Cases 1-3
        elif last_ho:
            dt_ms = (rlf_time - last_ho['timestamp']) * 1000.0
            src = last_ho['source_cell_id']
            tgt = last_ho['target_cell_id']

            if dt_ms <= self._tstore_ue_cntxt_ms:
                if reest_cell_id == src:
                    hof_type = "TOO_EARLY"
                    cause = (f"RLF {dt_ms:.0f}ms after HO {src}->{tgt}, "
                             f"re-estab back to source {reest_cell_id}")
                elif reest_cell_id == tgt:
                    hof_type = "NONE"
                    cause = f"Re-estab at target (recovered)"
                else:
                    hof_type = "WRONG_CELL"
                    cause = (f"RLF {dt_ms:.0f}ms after HO {src}->{tgt}, "
                             f"re-estab at 3rd cell {reest_cell_id}")
            else:
                hof_type = "TOO_LATE"
                cause = (f"RLF {dt_ms:.0f}ms after last HO "
                         f"(>{self._tstore_ue_cntxt_ms:.0f}ms), "
                         f"re-estab at {reest_cell_id}")
        else:
            hof_type = "TOO_LATE"
            cause = f"RLF with no prior HO, re-estab at {reest_cell_id}"

        result = {
            'hof_type': hof_type,
            'timestamp': current_time,
            'rlf_cell_id': rlf_cell,
            'reestablishment_cell_id': reest_cell_id,
            'last_ho_source': last_ho['source_cell_id'] if last_ho else -1,
            'last_ho_target': last_ho['target_cell_id'] if last_ho else -1,
            'time_since_last_ho_ms': (
                (rlf_time - last_ho['timestamp']) * 1000.0 if last_ho else -1),
            'cause': cause
        }
        self.hof_classifications.append(result)
        self.logger.warning(f"*** HOF CLASSIFIED: {hof_type} *** | {cause}")

        self._rlf_context = None
        return result

    def _check_ping_pong_cb(self, current_time: float,
                             source: int, target: int):
        """Check for ping-pong (Case 4) with time-based Tpp (callback mode)."""
        tpp_s = self._tpp_ms / 1000.0
        for entry in reversed(self._ho_history):
            if not entry['success']:
                continue
            dt = current_time - entry['timestamp']
            if dt > tpp_s:
                break
            if (entry['source_cell_id'] == target and
                    entry['target_cell_id'] == source):
                result = {
                    'hof_type': 'PING_PONG',
                    'timestamp': current_time,
                    'rlf_cell_id': -1,
                    'reestablishment_cell_id': -1,
                    'last_ho_source': source,
                    'last_ho_target': target,
                    'time_since_last_ho_ms': dt * 1000.0,
                    'cause': (f"Ping-pong: {entry['source_cell_id']}->"
                              f"{entry['target_cell_id']} then {source}->"
                              f"{target} within {dt*1000:.0f}ms")
                }
                self.hof_classifications.append(result)
                self.context.ping_pong_count += 1
                self.logger.warning(
                    f"*** HOF CLASSIFIED: PING_PONG *** | {result['cause']}")
                return result
        return None

    # ═══════════════════════════════════════════════════════════════════
    # Utility Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def attach_to_cell(self, cell_id: int):
        """Initial cell attachment"""
        self.context.serving_cell_id = cell_id
        self._set_rrc_state(RRCState.CONNECTED)
        self.logger.info(f"Attached to cell {cell_id}")
    
    def _set_rrc_state(self, new_state: RRCState):
        """Set RRC state with logging"""
        old_state = self.context.rrc_state
        if old_state != new_state:
            self.context.rrc_state = new_state
            self.logger.info(f"RRC state: {old_state.name} -> {new_state.name}")
            
            if self.on_rrc_state_change:
                self.on_rrc_state_change(self.ue_id, old_state, new_state)
    
    def get_state(self) -> Dict:
        """Get comprehensive UE state"""
        return {
            'ue_id': self.ue_id,
            'rrc_state': self.context.rrc_state.name,
            'ho_state': self.context.ho_state.name,
            'serving_cell': self.context.serving_cell_id,
            'target_cell': self.context.target_cell_id,
            'serving_rsrp_dbm': self.context.serving_rsrp_dbm,
            'serving_sinr_db': self.context.serving_sinr_db,
            'serving_bler': self.context.serving_bler,
            'rlf_state': self.rlf_detector.get_state(),
            'statistics': {
                'total_handovers': self.context.total_handovers,
                'successful_handovers': self.context.successful_handovers,
                'failed_handovers': self.context.failed_handovers,
                'rlf_count': self.context.rlf_count,
                'ping_pong_count': self.context.ping_pong_count
            }
        }
    
    def get_statistics(self) -> Dict:
        """Get UE statistics summary"""
        ho_success_rate = (
            self.context.successful_handovers / max(1, self.context.total_handovers)
        )
        
        avg_ho_duration = 0
        if self.handover_results:
            successful_hos = [r for r in self.handover_results if r.success]
            if successful_hos:
                avg_ho_duration = np.mean([r.duration_ms for r in successful_hos])
        
        return {
            'total_handovers': self.context.total_handovers,
            'successful_handovers': self.context.successful_handovers,
            'failed_handovers': self.context.failed_handovers,
            'handover_success_rate': ho_success_rate,
            'avg_handover_duration_ms': avg_ho_duration,
            'rlf_count': self.context.rlf_count,
            'ping_pong_count': self.context.ping_pong_count,
            'handover_results': self.handover_results,
            'rlf_events': self.rlf_events,
            # HOF Classification (3GPP TS 38.300 §15.5)
            'hof_classifications': self.hof_classifications,
            'hof_summary': self._get_hof_summary()
        }

    def _get_hof_summary(self) -> Dict:
        """Summarize HOF classifications by type."""
        from collections import Counter
        counts = Counter(c.get('hof_type', 'NONE') if isinstance(c, dict)
                         else getattr(c, 'hof_type', 'NONE')
                         for c in self.hof_classifications)
        return {
            'too_late': counts.get('TOO_LATE', 0),
            'too_early': counts.get('TOO_EARLY', 0),
            'wrong_cell': counts.get('WRONG_CELL', 0),
            'ping_pong': counts.get('PING_PONG', 0),
            't304_expiry': counts.get('T304_EXPIRY', 0),
            'total_hof': sum(v for k, v in counts.items() if k != 'NONE'),
        }


# For backwards compatibility
import numpy as np


# ============================================================================
# NEW: Time-Continuous State Machine UE Controller
# ============================================================================

class UEStateMachineConfig:
    """Configuration for UEStateMachine"""
    def __init__(self,
                 # Measurement Config
                 a3_offset_db: float = 3.0,
                 a2_threshold_dbm: float = -118.0,
                 a5_threshold1_dbm: float = -125.0,
                 a5_threshold2_dbm: float = -115.0,
                 hysteresis_db: float = 2.0,
                 a2_hysteresis_db: float = 3.0,
                 ttt_ms: float = 256.0,
                 a2_ttt_ms: float = 100.0,
                 a3_report_interval_ms: float = 480.0,
                 a2_report_interval_ms: float = 1024.0,
                 a5_ttt_ms: float = 256.0,
                 filter_coef: int = 4,
                 # RLF Config
                 n310: int = 10,
                 # N311 default = 2 for HST (High-Speed Train) scenarios.
                 # At v≈250–350 km/h and f_c=3.5 GHz, Doppler-induced fast
                 # fading causes rapid IS/OOS oscillation around Qin/Qout.
                 # N311=1 over-reacts to a single constructive-fading step
                 # and aborts T310 prematurely (false recovery). Requiring
                 # two consecutive IS indications (≈80 ms at 40 ms step)
                 # filters these blips while staying well inside T310.
                 # References: 3GPP TR 38.854 (HST NR), Chen et al. "HO
                 # optimization for HSR", Tavares et al. HST mobility.
                 n311: int = 2,
                 t310_ms: float = 1000.0,
                 # T312 (3GPP TS 38.331 §5.3.5.5.2 / §5.5.4 + §7.1). A SHORTER
                 # fast-RLF timer that is started for a measId whose reportConfig
                 # has useT312=true WHEN that report is triggered AND T310 is
                 # ALREADY running. We model a single global useT312. Once T312
                 # expires the UE declares RLF — i.e. when the link is degrading
                 # enough that T310 is counting AND the UE has just reported a
                 # better neighbour (a HO is needed) but the HO cannot complete,
                 # the UE gives up FASTER than the full T310 (= aligns the sim RLF
                 # with the field failure time). T312 is bound INSIDE the T310
                 # window: it is stopped whenever T310 stops (N311 recovery), when
                 # a HO is triggered (T304 starts), at re-establishment, and at RLF.
                 # Valid values per §7.1: 0,50,100,200,300,400,500,600,1000 ms.
                 # Default for this twin: 200 ms (< t310 → fires earlier).
                 t312_ms: float = 200.0,
                 # DEFAULT OFF (2026-06-11): T312 default-on over-fires the
                 # fast-RLF path on replayed field data and tanked HOF/RLF
                 # precision 53%→35%. The full T312 logic + USE_T312 env override
                 # are kept INTACT (still available via USE_T312=1); only the
                 # default reverts to OFF (legacy = no T312).
                 use_t312: bool = False,
                 t304_ms: float = 200.0,
                 t311_ms: float = 5000.0,
                 t301_ms: float = 400.0,
                 # HO finalization delay (post-RACH-success → serving-cell-switch).
                 # ALWAYS 0 — simulation and validation must run on identical
                 # state-machine timing. A non-zero value locks each HO for
                 # `delay_ms / step_ms` timesteps and the sim cannot keep up
                 # with KTX rapid cell churn → spurious T304 expiry RLFs
                 # (validated 2026-05-08: BO baseline RLF 4 → 9 at 450 ms).
                 # If frame-match needs sub-200 ms HO timing realism, address
                 # it via measurement-period downsampling, NOT this knob.
                 ho_processing_delay_ms: float = 0.0,
                 # RACH Config
                 preamble_tx_max: int = 10,
                 preamble_initial_power_dbm: float = -104.0,
                 reest_preamble_initial_power_dbm: float = -104.0,
                 power_ramping_step_db: float = 2.0,
                 ra_response_window_ms: float = 10.0,       # Not used in polling model; RAR resolves within one 40ms timestep
                 contention_resolution_timer_ms: float = 64.0,
                 # Sync thresholds
                 qout_rsrp: float = -150.0,  # Out-of-sync threshold (per-RE RSRP)
                 qin_rsrp: float = -140.0,   # In-sync threshold (per-RE RSRP)
                 # SINR-based RLF (3GPP TS 38.133 §8.1)
                 use_sinr_for_rlf: bool = False,     # False = legacy RSRP mode
                 qout_bler: float = 0.10,            # Qout: hypothetical PDCCH BLER > 10% (3GPP TS 38.133 §8.1)
                 qin_bler: float = 0.02,             # Qin: hypothetical PDCCH BLER < 2%
                 rlf_mcs_index: int = 0,             # PDSCH MCS 0 fallback (QPSK, rate 120/1024, -6.7 dB).
                                                     # Used for RLM only when rlf_use_pdcch_bler=False.
                 # Hypothetical-PDCCH RLM curve (TS 38.133 §8.1.2.1). When True
                 # (default) Qout/Qin are evaluated on a DCI-1_0 QPSK+Polar
                 # PDCCH BLER curve using ASYMMETRIC aggregation levels and
                 # energy boosts per spec:
                 #   Qout: AL8 + +4 dB PDCCH RE energy (Table 8.1.2.1-1)
                 #   Qin:  AL4 +  0 dB                 (Table 8.1.2.1-2)
                 # NOT on the PDSCH MCS0 curve. Decoupled from the PDSCH MCS
                 # table. env overrides: RLF_USE_PDCCH_BLER=0 reverts to MCS0.
                 rlf_use_pdcch_bler: bool = True,
                 rlf_pdcch_aggregation_level: int = 4,  # legacy single-AL knob (unused when rlm_qout/qin_al set)
                 # Spec-asymmetric RLM AL + energy boost knobs (TS 38.133 §8.1.2.1).
                 # The STRICT-SPEC reference is Qout = AL8 + 4 dB RE-energy boost,
                 # Qin = AL4 + 0 dB (Tables 8.1.2.1-1/2) → Qout cell-SINR ≈ -12 dB.
                 # On replayed FIELD SINR that reference makes pure-RLM T310-RLF
                 # almost never fire (field RLF is dominated by RACH/HO/SIB, not
                 # RLM) and DROPS RLF-timing recall 54%→14% — a regression. So the
                 # DEFAULT is calibrated to the accuracy-preserving knee instead:
                 #   Qout = AL4 + 0 dB → cell-SINR ≈ -5 dB  (recall 54%, prec 53%,
                 #   median|Δt| 0.55s — matches/improves the prior best).
                 # To run the STRICT TS 38.133 reference (compliance studies) set
                 #   env RLM_QOUT_AL=8 RLM_QOUT_BOOST_DB=4.
                 # HST (high-speed train) leaves the 10%/2% BLER + 200/100ms
                 # T_Evaluate unchanged per Rel-17 (only cell-reselection periods
                 # change at HST, not RLM) — verified spec-correct at 300+ km/h.
                 rlm_qout_al: int = 4,                      # env RLM_QOUT_AL  (strict spec = 8)
                 rlm_qin_al: int = 4,                       # env RLM_QIN_AL
                 rlm_qout_energy_boost_db: float = 0.0,     # env RLM_QOUT_BOOST_DB  (strict spec = 4.0)
                 rlm_qin_energy_boost_db: float = 0.0,      # env RLM_QIN_BOOST_DB   ( 0 dB per Table 8.1.2.1-2)
                 # UL SINR-based RACH
                 use_ul_sinr_for_rach: bool = True,  # default ON (TS 38.141-1 §8.4)
                 # RSRQ-based RACH penalty (abstraction layer for unmodeled
                 # msg2/3/4 and control-channel delivery — see memory
                 # `rsrq_rach_penalty_theory.md`). RSRQ captures wideband
                 # interference loading that single-SINR PHY abstraction
                 # cannot represent. When serving RSRQ degrades, the full
                 # 4-step RACH procedure (msg3 RRC Reconfiguration Complete
                 # on UE UL) and HO command delivery (PDCCH on serving DL)
                 # become unreliable beyond what target UL_SINR sigmoid
                 # predicts. Penalty is multiplied with msg1 detection
                 # probability. Disabled when penalty_enabled=False.
                 rsrq_rach_penalty_enabled: bool = True,
                 # Thresholds in 3GPP TS 36.214/38.215 reporting range
                 # ([-19.5, -3] dB). Standard UEs saturate at -19.5 so
                 # observing "near -19.5" is the worst-case the reporting
                 # protocol can express. Penalty fires across [-18, -19.5].
                 # UL msg3 / RACH delivery block: joint (RSRQ, RSRP) criterion.
                 # MEMORY v3 values (rsrq_rach_penalty_theory.md, 2026-05-18): RACH
                 # delivery penalty = FULL block (floor 0.0) when BOTH RSRQ < -20.0 dB
                 # AND RSRP < -100.0 dBm. FIELD 18-case: 12/12 UL-cluster failures fall
                 # in this region (100% recall); RLM-only RLF (RSRP -97~-99) left to
                 # N310/T310. (Restored from drift -19.0/-93.0/floor 0.5 → memory v3.)
                 rsrq_rach_penalty_full_db: float = -20.0,   # RSRQ >= this → no block
                 rsrq_rach_penalty_floor_db: float = -20.0,  # RSRQ <  this → eligible for block
                 rsrq_rach_penalty_floor: float = 0.0,       # full block when criteria met
                 rsrq_rach_penalty_rsrp_gate_dbm: float = -100.0,  # joint: also require RSRP < this
                 # Two-path UL msg3 delivery-failure gate. Path A = RSRP < -115
                 # AND RSRQ < -18.5; Path B = RSRQ <= -19.3 (alone). Path B raised
                 # -20.5 → -19.3 and made INCLUSIVE (2026-06-11, user request:
                 # "rsrq가 -19여도 msg3이 성공안되게", tuned to -19.3) so an RSRQ at
                 # or below -19.3 dB blocks msg3 delivery / RACH instead of
                 # succeeding. This is the same vendor abstraction layer as
                 # _rsrq_rach_penalty (NOT a 3GPP UE-spec timer) — it widens the
                 # RACH_PROBLEM recall but also fires on any cell whose filtered
                 # RSRQ reaches -19.3 dB.
                 ul_block_path_a_rsrp_dbm: float = -115.0,
                 ul_block_path_a_rsrq_db: float = -18.5,
                 ul_block_path_b_rsrq_db: float = -19.3,
                 # UL-block release hysteresis (2026-06-11): consecutive good
                 # ticks required before the latched UL block releases.
                 # SYMMETRIC with the 3-tick entry (N_ENTER_TICKS): once the
                 # gate reads L1 (raw) RSRQ (2026-06-12), the release uses the
                 # same 3-consecutive-good-tick rule as entry. Re-swept on the
                 # L1 baseline: N=3 gives the best aggregate F1 (63.7) and the
                 # largest weak-region (maesong) gain (F1 44→53), both regions
                 # improved vs the old N=10 (60.4). Was 10 under the L3 gate.
                 # Env override UL_BLOCK_RELEASE_TICKS for sweeps.
                 ul_block_release_good_ticks: int = 3,
                 # Pre-RACH admission gate (TS 38.304 §5.2.3.2 / TS 38.213 §8.1):
                 # the UE can only initiate RACH on a target whose DL RSRP is
                 # at or above Q_RxLevMin (typical LTE / NR cell selection
                 # threshold ≈ −115 dBm). When target RSRP is below this,
                 # PRACH preamble cannot be acquired (no SSB) and the RACH
                 # attempt fails with probability 1. This blocks sim's
                 # phantom HO success to a cell with un-readable DL coverage.
                 q_rxlevmin_dbm: float = -140.0,
                 # Strong-target RACH shortcut (vendor calibration, 2026-05-19).
                 # If target DL signal is clearly healthy, bypass the UL SINR
                 # sigmoid. Rationale: CSV channel mode does not decouple UL
                 # vs DL SINR, so non-serving cells' UL SINR is identical to
                 # DL SINR; the probabilistic sigmoid then under-estimates
                 # RACH success and produces T304 expiry on otherwise viable
                 # targets. Spec basis: TS 38.304 §5.2.3.2 (cell selection
                 # on DL RSRP/RSRQ); TS 38.214 (UL/DL coupling bounded).
                 # Knob: same vendor abstraction layer as _rsrq_rach_penalty.
                 rach_strong_target_enabled: bool = True,
                 rach_strong_target_rsrp_dbm: float = -95.0,
                 rach_strong_target_sinr_db: float = -5.0,
                 rach_strong_target_probability: float = 0.95,
                 # B1 Inter-RAT (3GPP TS 38.331 §5.5.4.7)
                 b1_threshold_dbm: float = -125.0,
                 b1_ttt_ms: float = 256.0,
                 b1_offset_db: float = 0.0,
                 # B2 Inter-RAT (3GPP TS 38.331 §5.5.4.8)
                 b2_threshold1_dbm: float = -130.0,
                 b2_threshold2_dbm: float = -125.0,
                 b2_ttt_ms: float = 256.0,
                 b2_offset_db: float = 0.0,
                 # RRC Connection Setup (3GPP TS 38.331 §5.3.3)
                 t300_ms: float = 1000.0,
                 setup_preamble_initial_power_dbm: float = -104.0,
                 # CIO Ocn (3GPP TS 38.331 §6.3.2 cellIndividualOffsets)
                 # per-(serving, neighbor_pci) offset map for the CURRENT serving
                 # cell. Empty dict ⇒ Ocn=0 ⇒ A3 byte-identical to pre-CIO.
                 cio_table: dict = None,
                 # S-Measure gate (3GPP TS 38.331 §5.5.4 / measObjectNR.s-Measure).
                 # When set and serving RSRP > s_measure_dbm, A3 evaluation is
                 # suppressed for the current serving cell. Per-cell: usually
                 # set to -97 dBm only for HSR cells. None = disabled (legacy).
                 s_measure_dbm: Optional[float] = None,
                 # L3 filter passthrough (CSV channel mode).
                 # When True, MeasurementEngine.filter_l3 skips IIR averaging
                 # and returns raw inputs (cadence-gated).
                 # ⚠️ 2026-06-09 (ops-confirmed): the wide PCI RSRP/RSRQ CSV is
                 # NOT L3-filtered — it is RAW L1 from the DM-tool log. The real
                 # network applies fc4 (3GPP filterCoefficient k=4, α=0.5) in the
                 # UE BEFORE A3/RLM, and that fc4 output is NOT in the logged data.
                 # So passthrough=True (skip L3) is NOT operationally faithful;
                 # to match ops the sim must APPLY fc4 (passthrough=False +
                 # filter_coef=4). The old "already field-L3-filtered → passthrough"
                 # rationale was wrong. Default False preserves legacy stats/RT.
                 l3_filter_passthrough: bool = False,
                 # Vendor gNB-side HO Decision Algorithm delay (ms).
                 # 3GPP TS 38.331 §5.5 standardises only UE-side measurement
                 # reporting; the actual HO command (RRCReconfiguration with
                 # reconfigurationWithSync) is gNB implementation. Real
                 # operator gNBs buffer A3/A5/B1/B2 reports and re-validate
                 # at +delay before issuing the HO command. This value
                 # controls how long after a UE-side report fires the sim
                 # waits before committing the HO. During the window:
                 #   - if target is no longer the strongest non-serving
                 #     (by ≥ 0.5 dB margin), the report is withdrawn (no HO)
                 #   - if delay elapsed and target still strongest, fire HO
                 # Default 100ms ≈ typical vendor coherence window. Set to 0
                 # to disable (legacy byte-identical behaviour).
                 gnb_ho_decision_delay_ms: float = 0.0,
                 # Stochastic vendor gNB HO decision delay. When std > 0, the
                 # per-buffer commit delay is drawn at buffer-open time from
                 # N(mean, std), clipped to [0, ∞). `mean` defaults to
                 # `gnb_ho_decision_delay_ms` (back-compat with the fixed
                 # delay knob). When `adapt_to_rf` is True, mean & std are
                 # further scaled by the recent serving-cell RF state at
                 # the moment the report arrives:
                 #   mean *= clip((sinr_smoothed + 5) / 15, 0.3, 2.0)
                 #     → strong SINR → longer deliberation; weak SINR →
                 #     fast commit (no time to dither when serving is bad)
                 #   std  *= clip(sinr_rolling_std / 3.0, 0.5, 2.0)
                 #     → high recent volatility → wider distribution
                 # The window for SINR mean+std is the most recent ~500 ms
                 # (50 samples at 10 ms tick). Default 0/0/False = legacy
                 # deterministic behaviour controlled by the fixed knob.
                 gnb_ho_decision_delay_std_ms: float = 0.0,
                 gnb_ho_decision_delay_adapt_to_rf: bool = False,
                 # DL HO-command delivery gate (2026-06-11). The HO command
                 # (RRCReconfiguration w/ reconfigurationWithSync) is a DL
                 # message on PDCCH; model whether the UE can actually DECODE
                 # it at the moment of commit, using a hypothetical-PDCCH BLER
                 # at a robust aggregation level (AL16 default — important RRC
                 # signalling is sent robustly) on the *serving* DL SINR, with
                 # `ho_command_harq_max` HARQ retransmissions:
                 #   P(miss) = pdcch_bler(serving_sinr, AL16) ** harq_max
                 # On miss the HO command is lost → HO does NOT start → the UE
                 # stays on the collapsing source (→ T310 keeps running →
                 # likely Too-Late RLF); a fresh measurement report retries on
                 # the next reportInterval. At normal SINR (e.g. -5 dB → AL16
                 # BLER ~0.002) P(miss)~0 so it passes through, exactly as
                 # before; the gate only bites in deep fade (≲ -12 dB). env:
                 # HO_CMD_DELIVERY_GATE=0 disables; HO_CMD_PDCCH_AL / HO_CMD_HARQ_MAX.
                 ho_command_delivery_gate: bool = True,
                 ho_command_pdcch_al: int = 16,
                 ho_command_harq_max: int = 4,
                 # gNB-side HO target-serviceability gate DURING T310 (2026-06-11).
                 # Vendor gNB-side HO ADMISSION decision (NOT a 3GPP UE-spec
                 # change — A3/A5/B1/B2 entry/leave, T310/T304/T311/N310/N311,
                 # RLM Qout/Qin are all untouched). When the radio link is in
                 # trouble (T310 running) the gNB only commits a "rescue" HO if
                 # the chosen target is itself SERVICEABLE — a HO into another
                 # coverage hole cannot actually rescue a failing link, it just
                 # ping-pongs laterally and (in sim) spuriously stops T310 before
                 # expiry, masking the RLF the field really experienced.
                 # LOG EVIDENCE (ue6 godeok_900M downside): a 275→33 HO at
                 # t≈18.21 to a target with SINR −7.9 dB stopped T310 (16.54→
                 # 18.20) right before its ~18.5 expiry, so the field RLF@18.74
                 # was missed. Of 7 T310-rescue HOs in that UE, 5 had target
                 # SINR<0 (invalid); the 2 genuine recoveries had +5/+11.7 dB.
                 # The gate ONLY fires while T310 is running; normal HOs are
                 # never affected. This is a VENDOR gNB-side HO-decision
                 # heuristic (NOT a 3GPP standard feature) — HO target admission
                 # is gNB-implementation territory, but the SINR floor is an
                 # empirical knob, NOT spec.
                 # DEFAULT ON at −3 dB since 2026-06-12 (= Qout −5 dB + 2 dB
                 # serviceability margin). History: the first evaluation
                 # (2026-06-11, PRE reportInterval-rollback baseline) showed the
                 # gain OVER-FIT to godeok (Δmatched godeok +4 / maesong −1) and
                 # the knob shipped OFF (−100). RE-EVALUATED on the post-fix
                 # baseline (reportInterval rollback + UL-block release
                 # hysteresis changed the rescue dynamics): floor −3 now improves
                 # BOTH regions (godeok F1 58→65, maesong 44→46; 900M F1 56→62;
                 # aggregate rec 50→58% / prec 55.6→58% / PCI 94.7%) — the
                 # cross-region guard passes. Sweep −6…+3 in
                 # harness/reports/t310_gate_sweep_2026-06-12.md. Env
                 # HO_T310_TGT_MIN_SINR overrides; −100 restores the old OFF.
                 ho_t310_target_min_sinr_db: float = -3.0,
                 # HO target-quality admission floor for ALL HOs (filtered
                 # RSRQ, dB). Vendor admission control — refuse HO to a
                 # load/interference-junk target even outside T310. DEFAULT ON
                 # since 2026-06-12; floor re-tuned −18 → −19 the same day
                 # after the A3 TTT-tracker ZOMBIE fix (measurement.py)
                 # removed the stale-report storm the gate was partly
                 # masking. On the zombie-fixed 36-case baseline, −19 (and
                 # −18.5, identical) dominates: rec 62% / prec 57.4% /
                 # godeok F1 67 / maesong 46 — both regions tied-best across
                 # the sweep table. Normal HOs have target RSRQ p10 −16.7,
                 # so −19 sits well below the legitimate population. Env
                 # HO_TARGET_MIN_RSRQ (−100 = OFF). See
                 # harness/reports/ho_target_rsrq_sweep_2026-06-12.md.
                 ho_target_min_rsrq_db: float = -19.0,
                 # DL TARGET-cell RRC-config delivery gate (S4, 2026-06-11).
                 # After RACH to the target succeeds (S3) and BEFORE the cell
                 # switch completes, model whether the UE can DECODE the
                 # target-cell RRC config delivery (RRCReconfigurationComplete
                 # ack / target SIB+RRC config) on the *target* DL PDCCH, as a
                 # hypothetical-PDCCH BLER at a robust aggregation level
                 # (AL16 default) on the target DL SINR, with
                 # `target_rrc_harq_max` HARQ retransmissions:
                 #   P(miss) = pdcch_bler(target_sinr, AL16) ** harq_max
                 # On miss the HO is NOT completed this tick (no cell switch);
                 # T304 keeps running → on T304 expiry the HO fails → RLF
                 # (HOF Case 5, T304 Expiry). At healthy target SINR (≳ -5 dB →
                 # AL16 BLER ~0.002) P(miss)~0 so the HO completes exactly as
                 # before; the gate only bites in deep target fade. This is a
                 # vendor delivery abstraction (NOT a 3GPP UE-spec change), the
                 # DL mirror of S2's serving-side HO-command gate. env:
                 # TARGET_RRC_DELIVERY_GATE=0 disables; TARGET_RRC_PDCCH_AL /
                 # TARGET_RRC_HARQ_MAX.
                 target_rrc_delivery_gate: bool = True,
                 target_rrc_pdcch_al: int = 16,
                 target_rrc_harq_max: int = 4,
                 # Vendor gNB MRO (Mobility Robustness Optimization) post-HO
                 # blacklist duration (seconds). After HO source→target,
                 # (target, source) is blacklisted on the measurement engine
                 # so the A3 entering condition for `source` is ignored
                 # while serving=target for this duration. Standardised in
                 # TS 38.473 / TS 28.541 NRM as gNB-side MRO behavior, NOT
                 # part of the UE FSM (TS 38.331). Default 0.0 disables
                 # (legacy byte-identical behaviour).
                 post_ho_blacklist_s: float = 0.0):
        """Store every kwarg as an instance attribute.

        This constructor is intentionally mechanical: each keyword argument
        maps 1:1 to `self.<name>`. Defaults follow 3GPP TS 38.331 / TS 38.133
        typical operational values and are overridden per-cell from the gNB
        CSV in `NRSimulation._build_cell_configs`.

        Side effects:
            Allocates no engine state. The full FSM is constructed by
            `UEStateMachine.__init__`, which consumes this config object.
        """
        self.a3_offset_db = a3_offset_db
        self.a2_threshold_dbm = a2_threshold_dbm
        self.a5_threshold1_dbm = a5_threshold1_dbm
        self.a5_threshold2_dbm = a5_threshold2_dbm
        self.hysteresis_db = hysteresis_db
        self.a2_hysteresis_db = a2_hysteresis_db
        self.ttt_ms = ttt_ms
        self.a2_ttt_ms = a2_ttt_ms
        self.a3_report_interval_ms = a3_report_interval_ms
        self.a2_report_interval_ms = a2_report_interval_ms
        self.a5_ttt_ms = a5_ttt_ms
        self.filter_coef = filter_coef
        self.n310 = n310
        self.n311 = n311
        self.t310_ms = t310_ms
        # T312 (TS 38.331 §7.1) — env overrides T312_MS / USE_T312, mirroring
        # the other env-toggleable knobs. USE_T312=0 reverts to legacy (no T312).
        self.t312_ms = float(os.environ.get("T312_MS", str(t312_ms)))
        self.use_t312 = (
            os.environ.get("USE_T312", "1" if use_t312 else "0")
            not in ("0", "false", "False", ""))
        # Validate t312_ms against the 3GPP TS 38.331 §7.1 value set (warn only,
        # never abort — mirrors permissive timer handling elsewhere).
        _VALID_T312 = (0, 50, 100, 200, 300, 400, 500, 600, 1000)
        if self.use_t312 and int(round(self.t312_ms)) not in _VALID_T312:
            logger.warning(
                "t312_ms=%s is not a 3GPP TS 38.331 §7.1 value %s; using as-is",
                self.t312_ms, _VALID_T312)
        self.t304_ms = t304_ms
        self.ho_processing_delay_ms = ho_processing_delay_ms
        self.t311_ms = t311_ms
        self.t301_ms = t301_ms
        self.preamble_tx_max = preamble_tx_max
        self.preamble_initial_power_dbm = preamble_initial_power_dbm
        self.reest_preamble_initial_power_dbm = reest_preamble_initial_power_dbm
        self.power_ramping_step_db = power_ramping_step_db
        self.ra_response_window_ms = ra_response_window_ms
        self.contention_resolution_timer_ms = contention_resolution_timer_ms
        self.qout_rsrp = qout_rsrp
        self.qin_rsrp = qin_rsrp
        self.use_sinr_for_rlf = use_sinr_for_rlf
        self.qout_bler = qout_bler
        self.qin_bler = qin_bler
        self.rlf_mcs_index = rlf_mcs_index
        # Hypothetical-PDCCH RLM curve (default AL4) with env overrides.
        self.rlf_use_pdcch_bler = (
            os.environ.get("RLF_USE_PDCCH_BLER",
                           "1" if rlf_use_pdcch_bler else "0")
            not in ("0", "false", "False", ""))
        self.rlf_pdcch_aggregation_level = int(
            os.environ.get("RLF_PDCCH_AL", str(rlf_pdcch_aggregation_level)))
        # Asymmetric RLM AL + energy boost (TS 38.133 §8.1.2.1)
        self.rlm_qout_al = int(
            os.environ.get("RLM_QOUT_AL", str(rlm_qout_al)))
        self.rlm_qin_al = int(
            os.environ.get("RLM_QIN_AL", str(rlm_qin_al)))
        self.rlm_qout_energy_boost_db = float(
            os.environ.get("RLM_QOUT_BOOST_DB", str(rlm_qout_energy_boost_db)))
        self.rlm_qin_energy_boost_db = float(
            os.environ.get("RLM_QIN_BOOST_DB", str(rlm_qin_energy_boost_db)))
        self.use_ul_sinr_for_rach = use_ul_sinr_for_rach
        self.rsrq_rach_penalty_enabled = rsrq_rach_penalty_enabled
        self.rsrq_rach_penalty_full_db = rsrq_rach_penalty_full_db
        self.rsrq_rach_penalty_floor_db = rsrq_rach_penalty_floor_db
        self.rsrq_rach_penalty_floor = rsrq_rach_penalty_floor
        self.rsrq_rach_penalty_rsrp_gate_dbm = rsrq_rach_penalty_rsrp_gate_dbm
        self.ul_block_path_a_rsrp_dbm = ul_block_path_a_rsrp_dbm
        self.ul_block_path_a_rsrq_db = ul_block_path_a_rsrq_db
        # Env override for sweep experiments (default keeps the field-tuned
        # value passed in by config; see UL msg3 two-path gate in update()).
        self.ul_block_path_b_rsrq_db = float(
            os.environ.get("UL_BLOCK_PATHB_RSRQ", str(ul_block_path_b_rsrq_db)))
        self.ul_block_release_good_ticks = int(
            os.environ.get("UL_BLOCK_RELEASE_TICKS",
                           str(ul_block_release_good_ticks)))
        self.q_rxlevmin_dbm = q_rxlevmin_dbm
        self.rach_strong_target_enabled = rach_strong_target_enabled
        self.rach_strong_target_rsrp_dbm = rach_strong_target_rsrp_dbm
        self.rach_strong_target_sinr_db = rach_strong_target_sinr_db
        self.rach_strong_target_probability = rach_strong_target_probability
        self.b1_threshold_dbm = b1_threshold_dbm
        self.b1_ttt_ms = b1_ttt_ms
        self.b1_offset_db = b1_offset_db
        self.b2_threshold1_dbm = b2_threshold1_dbm
        self.b2_threshold2_dbm = b2_threshold2_dbm
        self.b2_ttt_ms = b2_ttt_ms
        self.b2_offset_db = b2_offset_db
        self.t300_ms = t300_ms
        self.setup_preamble_initial_power_dbm = setup_preamble_initial_power_dbm
        self.cio_table: dict = cio_table if cio_table is not None else {}
        self.s_measure_dbm: Optional[float] = s_measure_dbm
        self.l3_filter_passthrough = l3_filter_passthrough
        self.gnb_ho_decision_delay_ms = gnb_ho_decision_delay_ms
        self.gnb_ho_decision_delay_std_ms = gnb_ho_decision_delay_std_ms
        self.gnb_ho_decision_delay_adapt_to_rf = gnb_ho_decision_delay_adapt_to_rf
        # DL HO-command delivery gate (AL16 PDCCH + HARQ) with env overrides.
        self.ho_command_delivery_gate = (
            os.environ.get("HO_CMD_DELIVERY_GATE",
                           "1" if ho_command_delivery_gate else "0")
            not in ("0", "false", "False", ""))
        self.ho_command_pdcch_al = int(
            os.environ.get("HO_CMD_PDCCH_AL", str(ho_command_pdcch_al)))
        self.ho_command_harq_max = int(
            os.environ.get("HO_CMD_HARQ_MAX", str(ho_command_harq_max)))
        # gNB-side HO target-serviceability floor during T310 (env override).
        # Default −100.0 dB = effective no-op; orchestrator sweeps a real floor.
        self.ho_t310_target_min_sinr_db = float(
            os.environ.get("HO_T310_TGT_MIN_SINR", str(ho_t310_target_min_sinr_db)))
        # HO target-quality admission floor (ALL HOs, filtered RSRQ).
        self.ho_target_min_rsrq_db = float(
            os.environ.get("HO_TARGET_MIN_RSRQ", str(ho_target_min_rsrq_db)))
        # DL TARGET-cell RRC-config delivery gate (S4, AL16 PDCCH + HARQ) with
        # env overrides. Mirrors the ho_command_* gate but on the TARGET DL.
        self.target_rrc_delivery_gate = (
            os.environ.get("TARGET_RRC_DELIVERY_GATE",
                           "1" if target_rrc_delivery_gate else "0")
            not in ("0", "false", "False", ""))
        self.target_rrc_pdcch_al = int(
            os.environ.get("TARGET_RRC_PDCCH_AL", str(target_rrc_pdcch_al)))
        self.target_rrc_harq_max = int(
            os.environ.get("TARGET_RRC_HARQ_MAX", str(target_rrc_harq_max)))
        self.post_ho_blacklist_s = post_ho_blacklist_s


class UEStateMachine:
    """
    Time-Continuous State Machine for UE RRC.

    This class implements a polling-based state machine where:
    - update() is called every timestep with measurements
    - State (timers, counters, procedures) persists across calls
    - Returns UEState object showing current state

    Unlike UERRCController (event-based), this provides:
    - Explicit state object returned each step
    - No callbacks - caller inspects returned state
    - Clear separation of concerns

    3GPP Reference: TS 38.331 (RRC Protocol)
    """

    def __init__(self, ue_id: int, config: UEStateMachineConfig):
        """Construct the time-continuous UE RRC state machine.

        Args:
            ue_id: integer UE identifier (used in logs and event metadata).
            config: `UEStateMachineConfig` carrying every per-cell HO
                parameter (A2/A3/A5/B1/B2 thresholds + TTTs, T310/T311/T304,
                N310/N311, RACH knobs, L3 filter settings, etc.).

        Side effects:
            - Allocates `self.state` (UEState) and `self.measurement_engine`
              (MeasurementEngine + MeasConfig2 derived from `config`).
            - Wires per-event callbacks (HO history recorder, ping-pong
              detector) and initialises RACH / RLM smoothing scratch space.
            - Does NOT start any timer; `update()` drives all state changes.
        """
        from .rrc_types import (
            UEState, RRCState, RadioLinkStatus, SignalingState
        )
        from .measurement import MeasurementEngine, MeasConfig2

        self.ue_id = ue_id
        self.config = config

        # Initialize State
        self.state = UEState(ue_id=ue_id)

        # Initialize Measurement Engine
        meas_conf = MeasConfig2(
            a3_offset=config.a3_offset_db,
            a2_threshold=config.a2_threshold_dbm,
            a5_threshold1=config.a5_threshold1_dbm,
            a5_threshold2=config.a5_threshold2_dbm,
            hysteresis=config.hysteresis_db,
            a2_hysteresis=config.a2_hysteresis_db,
            time_to_trigger_ms=config.ttt_ms,
            a2_ttt_ms=config.a2_ttt_ms,
            a5_ttt_ms=config.a5_ttt_ms,
            filter_coefficient=config.filter_coef,
            b1_threshold=config.b1_threshold_dbm,
            b1_ttt_ms=config.b1_ttt_ms,
            b1_offset=config.b1_offset_db,
            b2_threshold1=config.b2_threshold1_dbm,
            b2_threshold2=config.b2_threshold2_dbm,
            b2_ttt_ms=config.b2_ttt_ms,
            b2_offset=config.b2_offset_db,
            a3_report_interval_ms=config.a3_report_interval_ms,
            a2_report_interval_ms=config.a2_report_interval_ms,
            l3_filter_passthrough=getattr(config, "l3_filter_passthrough", False),
            post_ho_blacklist_s=float(getattr(config, "post_ho_blacklist_s", 0.0) or 0.0),
            s_measure_dbm=getattr(config, "s_measure_dbm", None),
        )
        self.measurement_engine = MeasurementEngine(meas_conf)

        # Set Counter Thresholds
        self.state.counters["N310"].threshold = config.n310
        self.state.counters["N311"].threshold = config.n311

        # Initial State
        self.state.rrc_state = RRCState.RRC_CONNECTED
        self.state.serving_cell_id = -1

        self._logger = logging.getLogger(f"UEStateMachine.{ue_id}")

        # Dedup set for one-shot CONFIG APPLY log per (ue_id, cell_id)
        self._logged_configs: set = set()

        self._logger.debug(
            f"[INIT CONFIG] UE={ue_id} "
            f"n310={config.n310} n311={config.n311} "
            f"t310={config.t310_ms}ms t312={config.t312_ms}ms(use={config.use_t312}) "
            f"t304={config.t304_ms}ms t311={config.t311_ms}ms "
            f"a3_off={config.a3_offset_db}dB hys={config.hysteresis_db}dB "
            f"ttt={config.ttt_ms}ms"
        )

        # BLER calculator for SINR-based RLF
        from phy.bler_calculator import BLERCalculator
        self._bler_calculator = BLERCalculator()

        # gNB HO Decision Algorithm — pending report buffer.
        # When _pending_ho_target is set, an A3/A5/B1/B2 report is being
        # validated by the gNB before being committed as a HO command.
        # See gnb_ho_decision_delay_ms in UEStateMachineConfig for spec.
        self._pending_ho_target: Optional[int] = None
        self._pending_ho_t_report: Optional[float] = None
        self._pending_ho_type: str = ""
        self._pending_ho_rsrp: Optional[float] = None
        # Per-buffer commit delay sampled at buffer-open time (Gaussian
        # mode). When std_ms == 0 this stays None and the legacy fixed
        # `gnb_ho_decision_delay_ms` knob is used directly.
        self._pending_ho_decision_delay_s: Optional[float] = None
        # Rolling window of serving SINR samples used to derive the
        # adaptive mean/std of the Gaussian decision delay. ~500 ms at a
        # 10 ms step (so 50 samples). Filled in update() each tick when
        # a serving SINR is available.
        from collections import deque as _deque
        self._sinr_window: _deque = _deque(maxlen=50)
        # Deterministic-per-UE RNG. Seeded from the cell_id (or 0) so
        # repeat runs of the same case produce identical samples.
        import numpy as _np
        _seed = int(getattr(self.config, "cell_id", 0)) & 0xFFFF
        self._ho_decision_rng: _np.random.Generator = _np.random.default_rng(_seed)
        # Withdraw threshold: if target drops below the new best non-serving
        # by more than this many dB during the decision window, discard.
        self._pending_ho_withdraw_db: float = 0.5

    def attach_to_cell(self, cell_id: int):
        """Initial cell attachment"""
        from .rrc_types import RRCState
        self.state.serving_cell_id = cell_id
        self.state.rrc_state = RRCState.RRC_CONNECTED
        self.state.rrc_connected = True

    def apply_config(self, new_config: 'UEStateMachineConfig'):
        """Hot-swap HO parameters (e.g., after serving cell change).

        Updates config + measurement engine config without resetting
        timers, counters, or FSM state. Per 3GPP, RRC reconfiguration
        updates parameters but doesn't restart ongoing procedures.
        """
        self.config = new_config

        # Update measurement engine config
        mc = self.measurement_engine.config
        mc.a3_offset = new_config.a3_offset_db
        mc.a2_threshold = new_config.a2_threshold_dbm
        mc.a5_threshold1 = new_config.a5_threshold1_dbm
        mc.a5_threshold2 = new_config.a5_threshold2_dbm
        mc.hysteresis = new_config.hysteresis_db
        mc.time_to_trigger_ms = new_config.ttt_ms
        mc.a2_ttt_ms = new_config.a2_ttt_ms
        mc.a5_ttt_ms = new_config.a5_ttt_ms
        mc.a3_report_interval_ms = new_config.a3_report_interval_ms
        mc.filter_coefficient = new_config.filter_coef
        mc.b1_threshold = new_config.b1_threshold_dbm
        mc.b1_ttt_ms = new_config.b1_ttt_ms
        mc.b1_offset = new_config.b1_offset_db
        mc.b2_threshold1 = new_config.b2_threshold1_dbm
        mc.b2_threshold2 = new_config.b2_threshold2_dbm
        mc.b2_ttt_ms = new_config.b2_ttt_ms
        mc.b2_offset = new_config.b2_offset_db
        # CIO Ocn — swap to NEW serving cell's outgoing CIO map
        mc.cio_table = getattr(new_config, "cio_table", None) or {}
        # S-Measure: per-cell (HSR cells set this; regular cells leave None).
        mc.s_measure_dbm = getattr(new_config, "s_measure_dbm", None)
        # L3 passthrough flag survives per-cell config swap (CSV mode stays CSV mode)
        mc.l3_filter_passthrough = getattr(new_config, "l3_filter_passthrough", False)
        # Vendor gNB MRO post-HO blacklist duration. Mirror the (possibly
        # per-cell) config value to BOTH the dataclass field (for debugging
        # and snapshotting) and the engine's runtime cache so blacklist
        # checks pick up the latest value without restarting the engine.
        new_blacklist_s = float(
            getattr(new_config, "post_ho_blacklist_s", 0.0) or 0.0
        )
        mc.post_ho_blacklist_s = new_blacklist_s
        self.measurement_engine._blacklist_duration_s = new_blacklist_s

        # Update counter thresholds (don't reset counts)
        self.state.counters["N310"].threshold = new_config.n310
        self.state.counters["N311"].threshold = new_config.n311

        # One-shot INFO log per (ue_id, cell_id) — deduped to avoid spam.
        # Includes A3 trigger_quantity ("rsrp" for HSR / "rsrq" for regular)
        # so log readers can verify HO profile + quantity routing.
        _cid = getattr(new_config, '_cell_id', None)
        _key = (self.ue_id, _cid if _cid is not None else id(new_config))
        if _key not in self._logged_configs:
            self._logged_configs.add(_key)
            _q = self.measurement_engine.quantity_for_a3.get(
                int(_cid), "rsrp") if _cid is not None else "rsrp"
            self._logger.info(
                f"[CONFIG APPLY] UE={self.ue_id} "
                f"cell={_cid if _cid is not None else '?'} "
                f"a3_q={_q} "
                f"n310={new_config.n310} n311={new_config.n311} "
                f"t310={new_config.t310_ms}ms t304={new_config.t304_ms}ms "
                f"t311={new_config.t311_ms}ms "
                f"a3_off={new_config.a3_offset_db}dB hys={new_config.hysteresis_db}dB "
                f"ttt={new_config.ttt_ms}ms"
            )

        self._logger.debug(
            f"Config swapped: a3_offset={new_config.a3_offset_db}, "
            f"ttt={new_config.ttt_ms}, n310={new_config.n310}")

    def _rlm_band_speed_params(self):
        """Return per-(serving-band, speed-bucket) RLM Qout/Qin AL+boost, or
        None to fall back to the global config (the ~240 km/h single fit).

        Driven by three attributes installed by the sim driver (all optional):
          * ``self._rlm_band_speed_table``: {band: {"<speed>": {qout_al,
            qin_al, qout_boost_db, qin_boost_db}}} — a SPARSE override table;
            absent / empty ⇒ None ⇒ legacy behaviour (byte-identical).
          * ``self._cell_band``: {cell_id: band_label ("900M"/"1.8G"/"2.1G"/…)}.
          * ``self._current_velocity_kmh``: this UE's speed this tick.
        The speed-bucket is the NEAREST configured bucket to the live speed.
        Per-key fields default to the global config when a table entry omits
        them, so an override can set just qout while keeping qin at the fit.

        DEAD-END — do NOT retune 1.8G Qout to fix the 1.8G FALSE RLFs
        (verified 2026-06-15, full bidirectional 36-case sweep):

            1.8G Qout   1.8G m/mi/f   aggregate F1   maesong F1
            AL1 +1dB    14/ 2/14      64.6           53
            AL2 -2dB    14/ 2/14      64.6           53
            AL4 -5dB *  13/ 3/10      65.9 (peak)    56   <- current default
            AL8 -8dB     7/ 9/ 5      60.0           52

        F1 is a single peak at the current AL4. Stricter (AL1/AL2) adds
        matched +1 but false +4 (precision −5.8); more lenient (AL8) drops
        false −5 but matched −6 (recall 60→48). 1.8G matched and false RLFs
        are the SAME RLM/T310 OUT_OF_SYNC mechanism (the 1.8G fades are real
        downlink failures at SINR −7…−13 dB), so any threshold move shifts
        them together — they are NOT separable by Qout in either direction.
        900M is untouched at every setting (m17/mi17/f1) — 1.8G Qout never
        affects 900M. The clean 900M lever is the UL-deliverability RLF
        (IN_SYNC scope, see docs/UL_DELIVERABILITY_RLF.md), NOT this table.
        Reducing 1.8G FALSE further needs input-data augmentation (UL SINR /
        RACH counters), not a parameter. The ONLY adopted entry here is the
        2.1G Qout-relax (AL8+4) — 2.1G has 0 field RLFs so relaxing only
        removes false at zero recall cost.
        """
        table = getattr(self, "_rlm_band_speed_table", None)
        if not table:
            return None
        sid = self.state.serving_cell_id
        if sid is None or sid < 0:
            return None
        band = getattr(self, "_cell_band", {}).get(int(sid))
        band_tbl = table.get(band)
        if not band_tbl:
            return None
        v = getattr(self, "_current_velocity_kmh", None)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        try:
            buckets = sorted(int(b) for b in band_tbl.keys())
        except (TypeError, ValueError):
            return None
        if not buckets:
            return None
        sb = min(buckets, key=lambda b: abs(b - v))
        e = band_tbl.get(str(sb)) or band_tbl.get(sb) or {}
        return {
            "qout_al": int(e.get("qout_al", getattr(self.config, "rlm_qout_al", 4))),
            "qin_al": int(e.get("qin_al", getattr(self.config, "rlm_qin_al", 4))),
            "qout_boost_db": float(e.get("qout_boost_db",
                                         getattr(self.config, "rlm_qout_energy_boost_db", 0.0))),
            "qin_boost_db": float(e.get("qin_boost_db",
                                        getattr(self.config, "rlm_qin_energy_boost_db", 0.0))),
        }

    def update(self, current_time: float, dt: float,
               raw_measurements: Dict[int, float],
               inter_freq_cells: Optional[List[int]] = None,
               inter_rat_cells: Optional[List[int]] = None,
               sinr_measurements: Optional[Dict[int, float]] = None,
               ul_sinr_measurements: Optional[Dict[int, float]] = None,
               raw_rsrq_measurements: Optional[Dict[int, float]] = None,
               quantity_for_a3: Optional[Dict[int, str]] = None) -> 'UEState':
        """
        Main State Machine Update Loop.

        Call this every simulation timestep with current measurements.

        Args:
            current_time: Current simulation time in seconds
            dt: Time delta since last update in seconds
            raw_measurements: {cell_id: rsrp_dbm}
            inter_freq_cells: Optional list of inter-frequency cell IDs
            raw_rsrq_measurements: Optional {cell_id: rsrq_db} for RSRQ-based
                A3 evaluation on regular (non-HSR) cells. Default None ⇒
                RSRP-only A3 (legacy, byte-identical baseline behavior).
            quantity_for_a3: Optional {cell_id: "rsrp"|"rsrq"} per-cell A3
                quantity selector. When set, the measurement engine compares
                the chosen quantity for each cell. Default None ⇒ all cells
                use RSRP.

        Returns:
            Updated UEState object
        """
        # Staged HO flow: clear a settled "COMPLETE" marker at the START of the
        # tick AFTER the completing tick. The success block leaves
        # ho_stage="COMPLETE" (+flags) so the completing tick is observable in
        # detailed_log; here, one tick later (HO no longer in progress), we
        # reset to the idle state so the marker does not leak into subsequent
        # rows. (RLF declare / re-establishment reset independently.)
        if self.state.ho_stage == "COMPLETE" and not self.state.ho_in_progress:
            self.state.ho_stage = ""
            self.state.ho_cmd_decoded = False
            self.state.target_rach_ok = False
            self.state.target_rrc_ok = False

        # Store SINR measurements for SINR-based procedures
        self._sinr_measurements = sinr_measurements or {}
        self._ul_sinr_measurements = ul_sinr_measurements or {}
        # Cache the raw DL RSRP dict for cross-method use (RACH admission
        # gate, A3 candidate freshness gate). The measurement engine
        # consumes a filtered copy; this stores the unfiltered input keyed
        # by cell_id for direct lookup elsewhere in this tick's processing.
        self._raw_rsrp_measurements = raw_measurements or {}
        # Raw (L1) RSRQ kept for the UL-block gate, which models instantaneous
        # msg3/UL deliverability — a physical-layer condition, so it reads the
        # L1 measurement, NOT the L3-filtered RSRQ used for A3 event evaluation.
        self._raw_rsrq_measurements = raw_rsrq_measurements or {}

        # Apply per-cell A3 quantity selector (RSRQ for non-HSR cells when
        # --use-rsrq-for-regular-cells is on). When None / empty, defaults
        # to all-RSRP — byte-identical to pre-flag behavior.
        self.measurement_engine.set_quantity_for_a3(quantity_for_a3)

        from .rrc_types import (
            RRCState, RadioLinkStatus, SignalingState
        )

        if inter_freq_cells is None:
            inter_freq_cells = []

        self.state.current_time = current_time
        self.state.signaling = SignalingState()  # Reset signaling for this step

        # Auto-attach if needed
        if self.state.serving_cell_id == -1 and raw_measurements:
            best = max(raw_measurements, key=raw_measurements.get)
            self.attach_to_cell(best)
            self._logger.info(f"[{current_time:.2f}s] Auto-attached to cell {best}")
            return self.state

        # =========================================================
        # Step 1: Measurement & L3 Filtering
        # =========================================================
        # Pass RSRQ measurements alongside RSRP. When raw_rsrq_measurements is
        # None, MeasurementEngine.filter_l3 skips RSRQ tracking entirely,
        # preserving byte-identical baseline behavior.
        filtered_meas = self.measurement_engine.filter_l3(
            raw_measurements, rsrq_measurements=raw_rsrq_measurements,
            current_time_s=current_time)

        # =========================================================
        # Step 2: Evaluate Events (A2, A3, A5)
        # =========================================================
        # freeze_cancel during OOS and GRAY_ZONE: prevents TTT cancellation
        # when channel measurements are unreliable. Removing GRAY_ZONE from freeze
        # caused 44 unrecoverable mismatches — too aggressive cancellation.
        from rrc.rrc_types import RadioLinkStatus
        _fc = (self.state.radio_link_status != RadioLinkStatus.IN_SYNC)
        events = self.measurement_engine.evaluate(
            current_time, dt, self.state.serving_cell_id,
            filtered_meas, inter_freq_cells,
            inter_rat_cells=inter_rat_cells,
            freeze_cancel=_fc,
        )
        self.state.measurement_events = events

        # Handle A2 (Inter-Freq Measurement Indication)
        if events["A2"].report_sent:
            if not self.state.pending_context.get("inter_meas_active"):
                self._logger.info(f"[{current_time:.2f}s] Event A2 Triggered -> Start Inter-Freq Meas")
                self.state.pending_context["inter_meas_active"] = True

        # =========================================================
        # Step 3-pre: 3GPP TS 38.133 §8.5.2.2 RLM quality smoothing
        # Single-pole IIR (τ=200 ms) on serving SINR & RSRP. The L1
        # quality estimate fed to Q_out/Q_in is averaged over ~200 ms,
        # NOT the instantaneous channel sample. This is INDEPENDENT of
        # the L3 RRM filter (TS 38.331 §5.5.3.2) which already gates
        # at 200 ms in MeasurementEngine.filter_l3.
        # =========================================================
        import math as _math_rlm
        if not hasattr(self, "_rlm_smoothed_sinr"):
            self._rlm_smoothed_sinr: Optional[float] = None
            self._rlm_smoothed_rsrp: Optional[float] = None
            self._rlm_last_smooth_t: Optional[float] = None
            self._rlm_tau_s: float = float(getattr(
                self.config, "rlm_smoothing_tau_s", 0.200))
            # 3GPP TS 38.133 §8.5.2.2 post-(re)config warm-up tracker:
            # records when the EMA window starts filling on the current
            # serving cell. None = no samples yet (full warm-up pending).
            self._rlm_warmup_started_t: Optional[float] = None
            # Tracks when the serving cell first went un-measurable in the
            # current "missing" run. None = serving currently observable.
            # Used by the sync check to force OOS once the absence exceeds
            # T_evaluate_out_DL (TS 38.133 §8.5.2.2 — L1 must indicate OOS
            # when the cell becomes un-measurable for >200ms FR1 no-DRX).
            self._rlm_serving_missing_since: Optional[float] = None
            # OOS evaluation-PERIOD (sample-and-hold) — TS 38.133 §8.5.2.2.
            # While T310 is NOT running, re-evaluate the OOS/IS decision only
            # once per this period (default 0 = legacy every-tick). After T310
            # starts, evaluate every tick for responsive N311 recovery. Set
            # RLM_OOS_PERIOD_MS=200 to restore the 200ms T_Evaluate_out_DL
            # cadence (a brief sub-period quality blip then cannot flip OOS/IS
            # or reset N310). The 200ms SINR EMA is unchanged; this gates only
            # the DECISION cadence, not the smoothing.
            self._rlm_oos_period_ms: float = float(
                os.environ.get("RLM_OOS_PERIOD_MS", "0"))
            self._rlm_last_oos_eval_t: Optional[float] = None
        # Begin warm-up clock only when the FIRST L1 indication of the
        # serving cell actually arrives (TS 38.133 §8.5.2.2: the
        # T_evaluate_out_DL=200ms accumulation starts on the first
        # measurable sample, not on the first sim tick where serving
        # may still be missing from the L1 input). Latched below
        # inside the "_has_serving" branch.
        if self._rlm_last_smooth_t is None:
            _dt_rlm_s = 0.0
        else:
            _dt_rlm_s = max(0.0, current_time - self._rlm_last_smooth_t)
        self._rlm_last_smooth_t = current_time
        if self._rlm_tau_s <= 0.0 or _dt_rlm_s <= 0.0:
            _alpha_rlm = 1.0
        else:
            _alpha_rlm = 1.0 - _math_rlm.exp(-_dt_rlm_s / self._rlm_tau_s)

        # 3GPP TS 38.133 §8.5.2.2: RLM uses L1 measurement indications.
        # When no L1 indication is generated for the serving cell this period
        # (the input dict has no entry for it — common in CSV channel mode
        # when the field UE briefly lost the cell from its measurement
        # report), do NOT advance the smoothed RSRP toward −140 dBm. Holding
        # the previous smoothed value is the spec-compliant interpretation
        # ("no indication" ≠ "indication of −140 dBm"). This prevents the
        # serving filter from drifting to the floor and tripping a false
        # T310 RLF when the cell is healthy in the field.
        # 3GPP TS 38.133 §8.5.2.2: the RLM EMA may ONLY be seeded from
        # an actual L1 measurement of the serving cell. If the serving
        # cell is missing from this tick's measurement dict, the L1
        # never produced an indication this period — neither seed the
        # filter from a sentinel (-140 dBm / -20 dB) nor advance it.
        # The downstream sync check treats `None` as "no L1 indication"
        # → GRAY_ZONE, matching the spec ("UE shall not generate any
        # indication when the evaluation conditions are not met").
        _has_serving = self.state.serving_cell_id in filtered_meas
        if _has_serving:
            _raw_serving_rsrp = float(filtered_meas[self.state.serving_cell_id])
            if self._rlm_smoothed_rsrp is None:
                # First L1 indication after (re)config / cell change.
                self._rlm_smoothed_rsrp = _raw_serving_rsrp
                # Start the T_evaluate_out_DL warm-up clock now (first
                # measurable serving-cell sample arrived).
                if self._rlm_warmup_started_t is None:
                    self._rlm_warmup_started_t = current_time
            else:
                self._rlm_smoothed_rsrp = (
                    _alpha_rlm * _raw_serving_rsrp
                    + (1.0 - _alpha_rlm) * self._rlm_smoothed_rsrp
                )
            # Cell is observable this tick — clear the missing-since marker.
            self._rlm_serving_missing_since = None
        else:
            # 3GPP TS 38.133 §8.5.2.2: if the L1 cannot measure the serving
            # cell for more than T_evaluate_out_DL (200ms FR1 no-DRX), it
            # must generate an out-of-sync indication. Track when the
            # missing-period started so the sync check below can force OOS
            # once the threshold is exceeded. Hold the smoothed value
            # otherwise (per TS 38.331 §5.5.3.2 L3 filter hold rule).
            if self._rlm_serving_missing_since is None:
                self._rlm_serving_missing_since = current_time

        if self._sinr_measurements:
            # Same spec rule as RSRP path above (TS 38.133 §8.5.2.2).
            _has_serving_sinr = self.state.serving_cell_id in self._sinr_measurements
            if _has_serving_sinr:
                _raw_serving_sinr = float(
                    self._sinr_measurements[self.state.serving_cell_id])
                if self._rlm_smoothed_sinr is None:
                    self._rlm_smoothed_sinr = _raw_serving_sinr
                else:
                    self._rlm_smoothed_sinr = (
                        _alpha_rlm * _raw_serving_sinr
                        + (1.0 - _alpha_rlm) * self._rlm_smoothed_sinr
                    )
                # Feed the rolling window used by the adaptive Gaussian
                # gNB HO decision delay (sampled at buffer-open time).
                self._sinr_window.append(_raw_serving_sinr)
            # else: hold previous smoothed value (None or last good).

        # =========================================================
        # Step 3: Physical Layer Sync Check (uses smoothed quality)
        # =========================================================
        # 3GPP TS 38.133 §8.5.2.2: when no L1 measurement of the
        # serving cell has been received yet (smoothed_* still None),
        # the L1 has produced no indication this period — that means
        # neither IN_SYNC nor OUT_OF_SYNC. Emit GRAY_ZONE so neither
        # N310 nor N311 advance, matching the spec literal text:
        # "UE shall not generate any indication when the evaluation
        #  conditions are not met."
        serving_rsrp = self._rlm_smoothed_rsrp

        # 3GPP TS 38.133 §8.5.2.2 OOS-on-prolonged-missing: if the L1 has been
        # unable to measure the serving cell for more than T_evaluate_out_DL
        # (200ms FR1 no-DRX), the L1 SHALL generate an out-of-sync
        # indication. This is what eventually drives N310 → T310 → RLF →
        # reestablishment when the field UE has clearly switched to a cell
        # the sim cannot pin to its current serving choice. Without this,
        # holding the previous L3 value forever leaves the sim stuck on a
        # cell that is no longer detectable, and PCI/RLF metrics diverge
        # from field. Apply the override only AFTER warm-up has started
        # (a serving sample arrived at least once on this cell).
        _t_eval_out_dl_s = self._rlm_tau_s  # 200 ms by default
        _serving_stale = (
            self._rlm_warmup_started_t is not None
            and self._rlm_serving_missing_since is not None
            and (current_time - self._rlm_serving_missing_since) > _t_eval_out_dl_s
        )

        # OOS evaluation-period gate: snapshot the prior decision; the block
        # below always recomputes radio_link_status, but if we are NOT at an
        # evaluation boundary (pre-T310, period > 0) the recomputed value is
        # DISCARDED at the end of the block (sample-and-hold). Observability
        # fields (rlm_bler_qout/qin/smoothed_sinr) still refresh every tick.
        _prev_rls = self.state.radio_link_status
        _oos_period_s = self._rlm_oos_period_ms / 1000.0
        _do_rlm_eval = (
            _oos_period_s <= 0.0
            or self.state.timers["T310"].running
            or self._rlm_last_oos_eval_t is None
            or (current_time - self._rlm_last_oos_eval_t) >= _oos_period_s - 1e-9
        )
        if _do_rlm_eval:
            self._rlm_last_oos_eval_t = current_time

        if _serving_stale:
            self.state.radio_link_status = RadioLinkStatus.OUT_OF_SYNC
        elif self.config.use_sinr_for_rlf and self._sinr_measurements:
            if self._rlm_smoothed_sinr is None:
                self.state.radio_link_status = RadioLinkStatus.GRAY_ZONE
            else:
                serving_sinr = self._rlm_smoothed_sinr
                if getattr(self.config, "rlf_use_pdcch_bler", False):
                    # -------------------------------------------------------
                    # Spec-faithful asymmetric RLM (TS 38.133 V16.4.0 §8.1):
                    #
                    # §8.1.1 BLER thresholds (unchanged for HST per Rel-17):
                    #   Qout: BLER > 10%  (BLERout, Table 8.1.1-1) → OUT_OF_SYNC
                    #   Qin:  BLER <  2%  (BLERin)                 → IN_SYNC
                    #
                    # §8.1.2.1 Hypothetical PDCCH (DCI format 1_0, 2 OFDM
                    #   symbols, REG bundle 6, distributed mapping):
                    #   Qout eval: AL8 + PDCCH RE energy +4 dB over SSS
                    #              (Table 8.1.2.1-1)
                    #   Qin  eval: AL4 + PDCCH RE energy  0 dB
                    #              (Table 8.1.2.1-2)
                    #
                    # The energy boost is added to the cell SINR before
                    # looking up the BLER curve; pdcch_sinr_to_bler() takes
                    # PDCCH-SINR (= cell_SINR + boost_dB).
                    # -------------------------------------------------------
                    # Dynamic per-band / per-speed RLM Qin/Qout (2026-06-12).
                    # If a (band, speed) table is installed, the serving cell's
                    # band + the current UE speed-bucket select the Qout/Qin
                    # AL+boost; else fall back to the global config values
                    # (which are fitted at ~240 km/h). Absent table → byte-
                    # identical to the prior single-fit behaviour.
                    _bs = self._rlm_band_speed_params()
                    if _bs is not None:
                        rlm_qout_al, rlm_qin_al = _bs["qout_al"], _bs["qin_al"]
                        rlm_qout_boost, rlm_qin_boost = _bs["qout_boost_db"], _bs["qin_boost_db"]
                    else:
                        rlm_qout_al = getattr(self.config, "rlm_qout_al", 8)
                        rlm_qin_al  = getattr(self.config, "rlm_qin_al",  4)
                        rlm_qout_boost = getattr(self.config, "rlm_qout_energy_boost_db", 4.0)
                        rlm_qin_boost  = getattr(self.config, "rlm_qin_energy_boost_db",  0.0)

                    bler_qout = self._bler_calculator.pdcch_sinr_to_bler(
                        serving_sinr + rlm_qout_boost, rlm_qout_al)
                    bler_qin  = self._bler_calculator.pdcch_sinr_to_bler(
                        serving_sinr + rlm_qin_boost,  rlm_qin_al)

                    # Observability read-out for OOS/IS reconstruction.
                    self.state.rlm_smoothed_sinr_db = float(serving_sinr)
                    self.state.rlm_bler_qout = float(bler_qout)
                    self.state.rlm_bler_qin = float(bler_qin)

                    if bler_qin < self.config.qin_bler:
                        self.state.radio_link_status = RadioLinkStatus.IN_SYNC
                    elif bler_qout > self.config.qout_bler:
                        self.state.radio_link_status = RadioLinkStatus.OUT_OF_SYNC
                    else:
                        self.state.radio_link_status = RadioLinkStatus.GRAY_ZONE
                elif self.config.rlf_mcs_index < 0:
                    # Legacy MCS-adaptive fallback (rlf_use_pdcch_bler=False)
                    pdcch_bler, _ = self._bler_calculator.sinr_to_bler_adaptive(serving_sinr)
                    if pdcch_bler < self.config.qin_bler:
                        self.state.radio_link_status = RadioLinkStatus.IN_SYNC
                    elif pdcch_bler > self.config.qout_bler:
                        self.state.radio_link_status = RadioLinkStatus.OUT_OF_SYNC
                    else:
                        self.state.radio_link_status = RadioLinkStatus.GRAY_ZONE
                else:
                    # Legacy MCS0 PDSCH fallback (rlf_use_pdcch_bler=False)
                    pdcch_bler = self._bler_calculator.sinr_to_bler(
                        serving_sinr, self.config.rlf_mcs_index, fading=False)
                    if pdcch_bler < self.config.qin_bler:
                        self.state.radio_link_status = RadioLinkStatus.IN_SYNC
                    elif pdcch_bler > self.config.qout_bler:
                        self.state.radio_link_status = RadioLinkStatus.OUT_OF_SYNC
                    else:
                        self.state.radio_link_status = RadioLinkStatus.GRAY_ZONE
        else:
            if serving_rsrp is None:
                self.state.radio_link_status = RadioLinkStatus.GRAY_ZONE
            elif serving_rsrp > self.config.qin_rsrp:
                self.state.radio_link_status = RadioLinkStatus.IN_SYNC
            elif serving_rsrp < self.config.qout_rsrp:
                self.state.radio_link_status = RadioLinkStatus.OUT_OF_SYNC
            else:
                self.state.radio_link_status = RadioLinkStatus.GRAY_ZONE

        # OOS evaluation-period sample-and-hold: outside an evaluation boundary
        # (pre-T310, RLM_OOS_PERIOD_MS>0) hold the previous OOS/IS decision so a
        # sub-period blip cannot flip it / reset N310. After T310 starts the
        # gate above forces every-tick evaluation, so recovery stays responsive.
        if not _do_rlm_eval:
            self.state.radio_link_status = _prev_rls

        # =========================================================
        # Step 4: Timer Updates
        # =========================================================
        for timer in self.state.timers.values():
            timer.check(current_time)

        # =========================================================
        # Step 5: Procedure Logic
        # =========================================================

        # 5-1. RLF Detection
        if (self.state.rrc_state == RRCState.RRC_CONNECTED
            and not self.state.rlf_declared):
            self._handle_rlf_detection(current_time)

        # 5-2. Handover Execution (T304 Monitoring)
        if self.state.ho_in_progress:
            self._handle_handover_execution(current_time, filtered_meas)

        # 5-3. Re-establishment (RLF Recovery)
        if self.state.rlf_declared:
            self._handle_reestablishment(current_time, filtered_meas)

        # 5-3b. RRC Connection Setup (IDLE recovery, 3GPP TS 38.331 §5.3.3)
        if (self.state.rrc_state == RRCState.RRC_IDLE
                and not self.state.rlf_declared):
            self._handle_rrc_setup(current_time, filtered_meas)

        # Observability read-out (NEVER gates anything): reset every tick, then
        # set below to record WHY a viable HO was not triggered this step.
        self.state.ho_suppress_reason = ""

        # 5-4a. T304 in-flight A3 discard (OBSERVABILITY ONLY; phenomenon G).
        # While a HO is executing (ho_in_progress / T304 running) the decision
        # block below is skipped (gated on `not ho_in_progress`), so a fresh A3
        # report toward a viable target is silently dropped. Surface the drop so
        # detailed_log self-attributes it. This is a pure read-out + INFO log —
        # the FSM still does NOT reroute (that would need the off-by-default
        # t304_reroute knob; see plan Task 3). events.csv stays byte-identical.
        if (self.state.rrc_state == RRCState.RRC_CONNECTED
                and self.state.ho_in_progress
                and not self.state.rlf_declared
                and events.get("A3") is not None
                and getattr(events["A3"], "report_sent", False)
                and getattr(events["A3"], "target_cell_id", None) is not None):
            self.state.ho_suppress_reason = "HO_IN_PROGRESS"
            _t304 = self.state.timers.get("T304")
            _rem = (_t304.remaining(current_time) * 1000.0
                    if (_t304 is not None and _t304.running) else 0.0)
            self._logger.info(
                f"[{current_time:.2f}s] A3 report to "
                f"{events['A3'].target_cell_id} discarded — HO in progress "
                f"(T304 rem={_rem:.0f}ms)")

        # 5-4. Handover Triggering (Decision)
        if (self.state.rrc_state == RRCState.RRC_CONNECTED
            and not self.state.ho_in_progress
            and not self.state.rlf_declared):

            target = None
            ho_type = ""
            # `_ul_block_kind` records which gate raised ul_report_blocked so the
            # detailed_log can self-attribute the block (no INFO re-run needed).
            _ul_block_kind = ""

            # === UL Report Transmission Gate (vendor SIB/RACH model) ===
            # When T310 is running, the serving cell's DL is already below
            # Qout (that's what started T310). The UE's measurement report
            # must traverse UL (PUCCH/PUSCH) to reach the serving gNB; if
            # UL is also degraded, the gNB never receives the report → no
            # RRCReconfiguration is issued → T310 expires → RLF.
            #
            # Approximation: T310 running + serving UL_SINR < -8 dB (or NaN)
            # → block the report this tick. Field log's `SIB_READ_FAILURE`
            # is the bidirectional-signaling-failure class that this models.
            ul_report_blocked = False
            self.state.sib_block_blocked_target = None
            _serving_id = self.state.serving_cell_id
            if self.state.timers["T310"].running:
                if _serving_id is not None and _serving_id >= 0:
                    _ul = self._ul_sinr_measurements.get(int(_serving_id))
                    if (_ul is None or not math.isfinite(float(_ul))
                            or float(_ul) < -8.0):
                        ul_report_blocked = True
                        _ul_block_kind = "SIB_BLOCK"  # T310+UL_SINR<-8 path

            # === Joint (RSRQ, RSRP) UL report block ===
            # Spec-aligned (TS 38.133 §8.5.2.2 T_evaluate_out_DL=200ms averaging):
            #   - Enter:   N=3 consecutive ticks (30ms) of (RSRQ_filt < full_db AND
            #              RSRP_filt < rsrp_gate_dbm). Single-tick noise never latches.
            #   - Auto-release: radio_link_status == IN_SYNC continuously for >= 200ms
            #     (matches T_evaluate_in_DL_SYNC). Real UE HARQ cycle is ~10 ms; the
            #     200ms window is the L1 hysteresis the spec already mandates.
            #   - Hard release: HO_COMPLETE, RLF declare, re-establishment (existing).
            N_ENTER_TICKS = 3
            T_INSYNC_RELEASE_S = 0.20

            if (not ul_report_blocked
                    and _serving_id is not None and _serving_id >= 0
                    and getattr(self.config, "rsrq_rach_penalty_enabled", False)):
                # UL-block reads the L1 (raw) serving RSRQ/RSRP — msg3/UL
                # deliverability is an instantaneous physical-layer condition,
                # NOT the L3-filtered quantity used for A3 event evaluation
                # (2026-06-12, per user instruction). The N_ENTER_TICKS=3
                # consecutive-tick debounce already filters single-tick L1
                # noise (≈ the L1 averaging the spec assumes), so no L3 smoothing
                # is needed here. Env UL_BLOCK_RSRQ_SOURCE=l3 restores the old
                # filtered behaviour for reproducing pre-2026-06-12 baselines.
                if os.environ.get("UL_BLOCK_RSRQ_SOURCE", "l1").lower() == "l3":
                    _rsrq = getattr(self.measurement_engine,
                                    "filtered_rsrq", {}).get(int(_serving_id))
                    _rsrp = getattr(self.measurement_engine,
                                    "filtered_rsrp", {}).get(int(_serving_id))
                else:
                    _rsrq = self._raw_rsrq_measurements.get(int(_serving_id))
                    _rsrp = self._raw_rsrp_measurements.get(int(_serving_id))
                # Two-path UL msg3 delivery-failure gate (Fix I+M — 2026-05-19):
                #   Path A: (RSRP < -115 dBm) AND (RSRQ < -18.5 dB)
                #     — very weak RSRP coupled with bad RSRQ.
                #   Path B: (RSRQ <= -19 dB)
                #     — bad RSRQ alone (any RSRP). INCLUSIVE so RSRQ == -19
                #       blocks (2026-06-11 user request, see config default).
                # If either path holds for N_ENTER_TICKS consecutive ticks,
                # the A3 measurement report cannot be delivered → UL block
                # latches and HO is suppressed until T310/RLF/recovery.
                _rsrp_path_a_dbm = float(getattr(self.config, "ul_block_path_a_rsrp_dbm", -115.0))
                _rsrq_path_a_db = float(getattr(self.config, "ul_block_path_a_rsrq_db", -18.5))
                _rsrq_path_b_db = float(getattr(self.config, "ul_block_path_b_rsrq_db", -19.3))

                rq_finite = (_rsrq is not None and math.isfinite(float(_rsrq)))
                rp_finite = (_rsrp is not None and math.isfinite(float(_rsrp)))
                path_a = (rp_finite and rq_finite
                          and float(_rsrp) < _rsrp_path_a_dbm
                          and float(_rsrq) < _rsrq_path_a_db)
                path_b = (rq_finite and float(_rsrq) <= _rsrq_path_b_db)
                joint_bad = path_a or path_b

                # Observability read-out: the EXACT values the gate compared.
                # `ul_block_applied_rsrq` is the L1 (raw) serving RSRQ by default
                # (== `serving_rsrq`), or the L3-filtered value when
                # UL_BLOCK_RSRQ_SOURCE=l3.
                self.state.ul_block_applied_rsrq = float(_rsrq) if rq_finite else None
                self.state.ul_block_applied_rsrp = float(_rsrp) if rp_finite else None
                self.state.ul_block_threshold_rsrq_db = _rsrq_path_b_db

                if not self.state.ul_block_active:
                    # Counting toward entry
                    if joint_bad:
                        self.state.ul_block_pending_ticks += 1
                        if self.state.ul_block_pending_ticks >= N_ENTER_TICKS:
                            self.state.ul_block_active = True
                            self.state.ul_block_pending_ticks = 0
                            self.state.ul_block_in_sync_start_t = None
                            _which = "A" if path_a else "B"
                            self.state.ul_block_path = _which  # diagnostics read-out
                            self._logger.info(
                                f"[{current_time:.2f}s] UL block ENTERED after "
                                f"{N_ENTER_TICKS} consecutive bad ticks "
                                f"(path {_which}, RSRP={float(_rsrp) if rp_finite else float('nan'):.2f}, "
                                f"RSRQ={float(_rsrq) if rq_finite else float('nan'):.2f})")
                    else:
                        # Reset consecutive counter on any good tick
                        self.state.ul_block_pending_ticks = 0
                else:
                    # Already blocked: TWO release paths (Fix J — 2026-05-19):
                    #   (1) IMMEDIATE: if `joint_bad` is currently False, the
                    #       entry condition no longer holds → release this
                    #       tick. This prevents the block from latching past
                    #       a recovered serving condition during T310 (where
                    #       radio_link_status=OOS would otherwise keep the
                    #       block alive until RLF).
                    #   (2) IN_SYNC sustained ≥200ms (legacy Fix A path) —
                    #       handles the case where joint_bad fluctuates but
                    #       radio_link_status has stabilised.
                    from .rrc_types import RadioLinkStatus
                    # Release path (1) with HYSTERESIS (2026-06-11): require
                    # the entry condition to stay cleared for N consecutive
                    # ticks before releasing. A single good tick at 300 km/h
                    # is not UL recovery — after the reportInterval rollback
                    # fix, the old instant release let one flap tick deliver
                    # the rescue report mid-outage. N=1 keeps the legacy
                    # instant-release (Fix J) semantics.
                    _rel_ticks = int(os.environ.get(
                        "UL_BLOCK_RELEASE_TICKS",
                        str(getattr(self.config, "ul_block_release_good_ticks", 1))))
                    # NOTE (2026-06-12): a LEAKY release counter (decay the
                    # good-tick count on a bad tick instead of resetting to 0)
                    # was prototyped to tolerate RSRQ chatter around the −19.3
                    # threshold, then REJECTED. The filtered serving RSRQ in the
                    # user-spotted godeok_1.8G_upside 00#5 case was good for only
                    # 9 of the needed 10 ticks in a single burst, so the leaky
                    # counter helped only when _rel_ticks was also lowered to ≤8
                    # — and that over-released in maesong (matched 8→7), failing
                    # the cross-region guard. Hard reset retained.
                    if not joint_bad:
                        self.state.ul_block_good_ticks = (
                            getattr(self.state, "ul_block_good_ticks", 0) + 1)
                        if self.state.ul_block_good_ticks >= _rel_ticks:
                            self.state.ul_block_active = False
                            self.state.ul_block_pending_ticks = 0
                            self.state.ul_block_in_sync_start_t = None
                            self.state.ul_block_good_ticks = 0
                            self._logger.info(
                                f"[{current_time:.2f}s] UL block RELEASED "
                                f"(entry cleared ≥{_rel_ticks} good ticks: RSRP="
                                f"{float(_rsrp) if rp_finite else float('nan'):.2f}, "
                                f"RSRQ={float(_rsrq) if rq_finite else float('nan'):.2f})")
                    else:
                        self.state.ul_block_good_ticks = 0
                        is_in_sync = self.state.radio_link_status == RadioLinkStatus.IN_SYNC
                        # Release path (2) releases the block on RLM IN_SYNC even
                        # though `joint_bad` (dead RSRQ, the ENTRY condition) still
                        # holds. UL/msg3 deliverability is an RSRQ condition, not a
                        # SINR one — so a block entered on dead RSRQ should NOT be
                        # released by healthy SINR. This SINR-release is exactly
                        # how the SINR-good/RSRQ-dead "too-late HO" escapes the
                        # field RLF (the held A3 report delivers on the IN_SYNC
                        # release). UL_BLOCK_INSYNC_RELEASE=0 disables path (2) so
                        # the block latches until RSRQ recovers (path 1) or the
                        # UL-deliverability RLF fires. DEFAULT "0" since 2026-06-14
                        # (UL deliverability tracks RSRQ, not DL SINR — UE-side,
                        # spec-aligned with the random-access-problem / RLC-max-retx
                        # RLF triggers of TS 38.331 §5.3.10.3, NOT a vendor knob).
                        # UL_BLOCK_INSYNC_RELEASE=1 restores the pre-2026-06-14
                        # SINR-release behaviour.
                        _insync_release = os.environ.get(
                            "UL_BLOCK_INSYNC_RELEASE", "0") != "0"
                        if is_in_sync and _insync_release:
                            if self.state.ul_block_in_sync_start_t is None:
                                self.state.ul_block_in_sync_start_t = current_time
                            elif (current_time - self.state.ul_block_in_sync_start_t) >= T_INSYNC_RELEASE_S:
                                self.state.ul_block_active = False
                                self.state.ul_block_pending_ticks = 0
                                self.state.ul_block_in_sync_start_t = None
                                self._logger.info(
                                    f"[{current_time:.2f}s] UL block RELEASED "
                                    f"(IN_SYNC sustained ≥{T_INSYNC_RELEASE_S*1000:.0f}ms)")
                        else:
                            # OOS or transitional — reset the in-sync timer
                            self.state.ul_block_in_sync_start_t = None

                if self.state.ul_block_active:
                    ul_report_blocked = True
                    _ul_block_kind = "UL_BLOCK_PATH" + (self.state.ul_block_path or "B")

                # ---- UL-deliverability RLF (2026-06-13) ----
                # Models the FIELD's RACH_PROBLEM at a SINR-good / RSRQ-dead
                # source: the DL is decodable (RLM IN_SYNC, qout_BLER≈0, so
                # RLM/T310 never fires) but the UL cannot carry msg3/RRC because
                # RSRQ has collapsed. Without this the UE sits UL-blocked at good
                # SINR and escapes the HO on a noise-bounce release, MISSING the
                # field RLF (the dominant 900M recall gap — verified: rescue HO at
                # rlm_sinr +0.6…+7.3 / qout_BLER 0.00). When the UL-block stays
                # active while RLM is IN_SYNC for ≥ UL_DELIV_RLF_MS, declare RLF.
                # The IN_SYNC scope is the clean discriminator: the 1.8G FALSEs are
                # RLM-OOS (qout_BLER 0.43…1.00, T310-driven) so they are NEVER in
                # this branch. Pairs with UL_BLOCK_INSYNC_RELEASE=0 (else the block
                # is released by IN_SYNC before the timer matures). DEFAULT 1000ms
                # since 2026-06-14 (spec-aligned UE-side RLF, TS 38.331 §5.3.10.3
                # random-access-problem / RLC-max-retx; band selectivity is by RLM
                # state physics, not a per-band threshold — 1.8G FALSEs are OOS so
                # untouched). UL_DELIV_RLF_MS=0 disables (pre-2026-06-14 baseline).
                _deliv_ms = float(os.environ.get("UL_DELIV_RLF_MS", "1000") or "1000")
                if _deliv_ms > 0 and not self.state.rlf_declared:
                    _in_sync_now = (self.state.ul_block_active
                                    and self.state.radio_link_status
                                    == RadioLinkStatus.IN_SYNC)
                    if _in_sync_now:
                        if getattr(self, "_ul_deliv_dead_start_t", None) is None:
                            self._ul_deliv_dead_start_t = current_time
                        elif (current_time - self._ul_deliv_dead_start_t) * 1000.0 \
                                >= _deliv_ms:
                            self._logger.warning(
                                f"[{current_time:.2f}s] UL-deliverability RLF "
                                f"(UL-block active + RLM IN_SYNC ≥{_deliv_ms:.0f}ms; "
                                f"DL decodable, UL msg3 undeliverable → RACH_PROBLEM)")
                            self.state.pending_context["rlf_cause_hint"] = "UL_DELIVERY"
                            self._ul_deliv_dead_start_t = None
                            self._declare_rlf(current_time)
                    else:
                        self._ul_deliv_dead_start_t = None

            # Priority: A3 (Intra) > A5 (Inter-Freq) > B2 (Inter-RAT dual) > B1 (Inter-RAT single)
            if events["A3"].report_sent:
                target = events["A3"].target_cell_id
                ho_type = "Intra-Freq"
            elif events["A5"].report_sent:
                target = events["A5"].target_cell_id
                ho_type = "Inter-Freq"
            elif events.get("B2") and events["B2"].report_sent:
                target = events["B2"].target_cell_id
                ho_type = "Inter-RAT-B2"
            elif events.get("B1") and events["B1"].report_sent:
                target = events["B1"].target_cell_id
                ho_type = "Inter-RAT-B1"

            # ---- UL-blocked A3 HO: pending delivery (2026-06-12) ----
            # reportInterval (TS 38.331 §5.5.5) governs report GENERATION. A
            # report generated but not deliverable (UL blocked) must NOT be
            # regenerated by the measurement engine every 10 ms — the previous
            # reportInterval-rollback did exactly that (re-fired the A3 report
            # every tick, 884× per fade: spec-violating spam the user flagged).
            # Instead the generated report is HELD as a pending HO target and
            # the delivery layer retries it every tick, so the HO fires the
            # instant UL recovers. This preserves the prompt escape-HO delivery
            # that protects precision (rollback-OFF crashes precision 57→42%)
            # WITHOUT the per-tick report spam. A3 (intra) only — the old
            # rollback was A3-scoped too.
            _a3_expired = (events["A3"].triggered
                           and events["A3"].target_cell_id is not None
                           and events["A3"].time_to_trigger_remaining <= 0.0)
            if ul_report_blocked:
                # Track the current best expired A3 candidate while blocked
                # (mirrors the per-tick candidate re-selection of the rollback
                # path). Keep any prior pending if the trigger momentarily lapses.
                if _a3_expired:
                    self.state.ul_pending_ho_target = int(events["A3"].target_cell_id)
                    self.state.ul_pending_ho_type = "Intra-Freq"
            else:
                # UL clear: deliver a still-valid pending HO when the measurement
                # engine did not regenerate the report this tick (reportInterval
                # not yet elapsed). A fresh report this tick (target already set)
                # takes precedence; either way the pending slot is consumed.
                if (target is None
                        and self.state.ul_pending_ho_target is not None
                        and _a3_expired
                        and events["A3"].target_cell_id == self.state.ul_pending_ho_target):
                    target = int(self.state.ul_pending_ho_target)
                    ho_type = self.state.ul_pending_ho_type or "Intra-Freq"
                self.state.ul_pending_ho_target = None
                self.state.ul_pending_ho_type = ""

            # === Start T312 (TS 38.331 §5.3.5.5.2 / §5.5.4) ===
            # A measurement report has just been sent (any of A3/A5/B1/B2
            # report_sent above set `target`). If useT312 is configured AND
            # T310 is ALREADY running AND T312 is not already running, arm T312
            # for t312_ms. This is evaluated BEFORE the UL-block suppression
            # below, so T312 arms even when the HO is UL-blocked/suppressed —
            # that is exactly the target pattern: a report was triggered (HO is
            # needed) but the link keeps degrading / the HO cannot complete, so
            # the UE declares RLF faster than full T310. T312 can NEVER fire
            # without T310 already running + a report → precision is protected.
            if (self.config.use_t312
                    and target is not None
                    and self.config.t312_ms > 0.0
                    and self.state.timers["T310"].running
                    and not self.state.timers["T312"].running):
                self.state.timers["T312"].start(
                    current_time, self.config.t312_ms / 1000.0)
                self._logger.warning(
                    f"[{current_time:.2f}s] T312 started ({self.config.t312_ms:.0f}ms) "
                    f"— report sent (target={target}) while T310 running")

            # NOTE (2026-06-12): a TARGET-AWARE UL-block bypass (let the HO
            # proceed when the A3 target SINR ≥ floor) was prototyped here and
            # REJECTED. It removed the godeok_1.8G_upside 00#5 FALSE RLF but
            # collapsed recall (62%→30% at floor +6, →6% at 0): most FIELD RLFs
            # are escape-HOs that FAILED despite an available target
            # (RACH_PROBLEM/HO_FAILURE), which the sim's UL-block correctly
            # reproduces. With only radio signals the sim cannot tell the rare
            # "field succeeded the HO" case from the many "field failed" ones,
            # so the bypass deletes far more matched RLFs than false ones. See
            # the UL-block FALSE diagnosis in the session memory.
            # A HO target is UL-blocked this tick if either a fresh report was
            # generated (target set) OR an A3 report is pending delivery (held
            # between reportIntervals). Both must accrue the per-tick SIB count
            # — the old rollback re-fired the A3 report each tick, so the count
            # ran per-tick; with pending delivery we count the held target too.
            _blocked_tgt = None
            if ul_report_blocked:
                if target is not None:
                    _blocked_tgt = int(target)
                elif _a3_expired and self.state.ul_pending_ho_target is not None:
                    # A3 still TTT-expired but the report was interval-gated this
                    # tick (not regenerated). The old rollback re-fired it, so it
                    # counted here; match that by counting only while still
                    # expired (NOT merely while a stale pending lingers).
                    _blocked_tgt = int(self.state.ul_pending_ho_target)
            if _blocked_tgt is not None:
                # Suppress HO trigger. If T310 is currently running, set the
                # sticky flag so that when T310 expires → RLF, the cause is
                # classified as SIB_READ_FAILURE (single event per failure,
                # matching field log's rlf_cause column semantics).
                self.state.sib_block_blocked_target = _blocked_tgt
                # Diagnostics read-out: record WHY this viable HO was suppressed.
                self.state.ho_suppress_reason = _ul_block_kind or "SIB_BLOCK"
                if self.state.timers["T310"].running:
                    # Count this tick toward sustained-block threshold (Fix E).
                    # SIB classification requires ≥SIB_TICK_THRESHOLD (50ms)
                    # of sustained UL block with a viable HO target suppressed.
                    self.state.sib_block_ticks_during_t310 += 1
                if target is not None:
                    self._logger.info(
                        f"[{current_time:.2f}s] UL report BLOCKED "
                        f"(T310={self.state.timers['T310'].running}, "
                        f"RSRQ-driven) — suppressed HO to {target}")
                    target = None
                    ho_type = ""

            # === Vendor gNB-side HO Decision Algorithm ===
            # 3GPP only standardises UE-side reporting. The gNB buffers the
            # report and revalidates after gnb_ho_decision_delay_ms before
            # issuing RRCReconfiguration. When mean+std==0 we keep legacy
            # byte-identical behaviour (immediate fire on report).
            base_mean_ms = float(getattr(self.config, "gnb_ho_decision_delay_ms", 0.0))
            base_std_ms = float(getattr(self.config, "gnb_ho_decision_delay_std_ms", 0.0))
            decision_enabled = (base_mean_ms > 0.0) or (base_std_ms > 0.0)

            commit_target = None
            commit_ho_type = ""

            if not decision_enabled:
                # Legacy path — fire immediately
                if target is not None:
                    # S1 A3_REPORTED: a viable report passed the UL gate and a
                    # HO target is pending (immediate-fire path has no buffer).
                    self.state.ho_stage = "A3_REPORTED"
                    commit_target = target
                    commit_ho_type = ho_type
            else:
                # === Buffered gNB decision path ===
                # Update / refresh the buffer when a new fresh report arrives
                # (target changes or no buffer yet).
                if target is not None and target != self.state.serving_cell_id:
                    if self._pending_ho_target is None:
                        # New report: open the decision window.
                        # S1 A3_REPORTED: report passed UL gate, pending_ho set.
                        self.state.ho_stage = "A3_REPORTED"
                        self._pending_ho_target = int(target)
                        self._pending_ho_t_report = current_time
                        self._pending_ho_type = ho_type
                        self._pending_ho_rsrp = float(filtered_meas.get(int(target), -140.0))
                        self._pending_ho_decision_delay_s = self._sample_ho_decision_delay_s(
                            base_mean_ms, base_std_ms)
                        self._logger.debug(
                            f"[{current_time:.2f}s] gNB HO Decision: buffer report "
                            f"target={target} ({ho_type}) for "
                            f"{self._pending_ho_decision_delay_s*1000:.0f}ms "
                            f"(N(mean={base_mean_ms:.0f},std={base_std_ms:.0f})ms"
                            f"{', RF-adapted' if self.config.gnb_ho_decision_delay_adapt_to_rf else ''})")
                    elif int(target) != self._pending_ho_target:
                        # Different target arrived → withdraw old, restart with new.
                        self._logger.debug(
                            f"[{current_time:.2f}s] gNB HO Decision: target switched "
                            f"{self._pending_ho_target}->{target}, restart window")
                        self._pending_ho_target = int(target)
                        self._pending_ho_t_report = current_time
                        self._pending_ho_type = ho_type
                        self._pending_ho_rsrp = float(filtered_meas.get(int(target), -140.0))
                        self._pending_ho_decision_delay_s = self._sample_ho_decision_delay_s(
                            base_mean_ms, base_std_ms)
                    else:
                        # Same target re-confirmed: refresh stored RSRP.
                        self._pending_ho_rsrp = float(filtered_meas.get(int(target), self._pending_ho_rsrp or -140.0))

                # Process the active buffer (if any).
                if self._pending_ho_target is not None and self._pending_ho_t_report is not None:
                    pending_t = self._pending_ho_target

                    # Discard if the pending target became the serving cell
                    # somehow (e.g., re-establishment to it) — no HO needed.
                    if pending_t == self.state.serving_cell_id:
                        self._pending_ho_target = None
                        self._pending_ho_t_report = None
                        self._pending_ho_type = ""
                        self._pending_ho_rsrp = None
                        self._pending_ho_decision_delay_s = None
                    else:
                        # Coherence check — withdraw if target dropped below
                        # the strongest non-serving cell by ≥ withdraw_db.
                        cur_target_rsrp = filtered_meas.get(pending_t, None)
                        best_other_rsrp = None
                        for cid, rsrp in filtered_meas.items():
                            if cid == self.state.serving_cell_id or cid == pending_t:
                                continue
                            if best_other_rsrp is None or rsrp > best_other_rsrp:
                                best_other_rsrp = rsrp
                        if (cur_target_rsrp is not None and best_other_rsrp is not None
                                and (best_other_rsrp - cur_target_rsrp) > self._pending_ho_withdraw_db):
                            self._logger.debug(
                                f"[{current_time:.2f}s] gNB HO Decision: withdraw "
                                f"target={pending_t} (rsrp={cur_target_rsrp:.1f} < "
                                f"best_other={best_other_rsrp:.1f} by "
                                f"{best_other_rsrp - cur_target_rsrp:.1f} dB)")
                            self._pending_ho_target = None
                            self._pending_ho_t_report = None
                            self._pending_ho_type = ""
                            self._pending_ho_rsrp = None
                            self._pending_ho_decision_delay_s = None
                        else:
                            elapsed_s = current_time - self._pending_ho_t_report
                            this_delay_s = (self._pending_ho_decision_delay_s
                                            if self._pending_ho_decision_delay_s is not None
                                            else base_mean_ms / 1000.0)
                            if elapsed_s >= this_delay_s and \
                                    not self._ho_command_decoded():
                                # DL HO-command (RRCReconfiguration) could NOT
                                # be decoded on the serving PDCCH (AL16 + HARQ)
                                # → command lost. Drop this attempt; the UE
                                # stays on the source and a fresh report retries
                                # on the next reportInterval. In deep fade this
                                # repeats → T310 runs to RLF (Too-Late).
                                self.state.ho_suppress_reason = "HO_CMD_DL_MISS"
                                self._logger.info(
                                    f"[{current_time:.2f}s] HO command NOT decoded "
                                    f"(AL{self.config.ho_command_pdcch_al} PDCCH miss) "
                                    f"→ HO to {pending_t} dropped, await next report")
                                self._pending_ho_target = None
                                self._pending_ho_t_report = None
                                self._pending_ho_type = ""
                                self._pending_ho_rsrp = None
                                self._pending_ho_decision_delay_s = None
                            elif elapsed_s >= this_delay_s:
                                # Delay elapsed AND target still coherent AND HO
                                # command decoded on serving DL → commit.
                                commit_target = pending_t
                                commit_ho_type = self._pending_ho_type or "Intra-Freq"
                                self._logger.debug(
                                    f"[{current_time:.2f}s] gNB HO Decision: commit "
                                    f"target={pending_t} after {elapsed_s*1000:.0f}ms "
                                    f"(sampled delay={this_delay_s*1000:.0f}ms)")
                                self._pending_ho_target = None
                                self._pending_ho_t_report = None
                                self._pending_ho_type = ""
                                self._pending_ho_rsrp = None
                                self._pending_ho_decision_delay_s = None

            if commit_target is not None:
                # gNB-side HO target-serviceability gate DURING T310 (vendor
                # admission decision; NOT a 3GPP UE-spec change). While T310 is
                # running (radio-link problem), withhold a "rescue" HO if the
                # commit target is ITSELF sub-floor on its current instantaneous
                # SINR — a HO into another coverage hole cannot rescue a failing
                # link; it only ping-pongs laterally and (in sim) stops T310
                # before expiry, masking the field's real RLF. Default floor
                # −100 dB ⇒ never fires (no-op). Gate is scoped to T310-running
                # only, so normal HOs are NEVER affected.
                if self.state.timers["T310"].running:
                    _tgt_sinr = self._sinr_measurements.get(int(commit_target))
                    _floor = float(getattr(self.config, "ho_t310_target_min_sinr_db", -100.0))
                    if (_tgt_sinr is not None and math.isfinite(float(_tgt_sinr))
                            and float(_tgt_sinr) < _floor):
                        # Withhold: leave serving unchanged, leave T310 RUNNING
                        # (do NOT stop it) so it proceeds to expiry → RLF (the
                        # field outcome). Drop the pending-HO buffer (mirrors the
                        # HO_CMD_DL_MISS drop path) so A3 re-evaluates next tick.
                        self.state.ho_suppress_reason = "T310_TGT_WEAK"
                        self._logger.info(
                            f"[{current_time:.2f}s] HO to {commit_target} withheld "
                            f"(T310 running, target SINR={float(_tgt_sinr):.1f} dB "
                            f"< floor {_floor:.1f} dB) — T310 continues to expiry")
                        self._pending_ho_target = None
                        self._pending_ho_t_report = None
                        self._pending_ho_type = ""
                        self._pending_ho_rsrp = None
                        self._pending_ho_decision_delay_s = None
                        commit_target = None

            if commit_target is not None:
                # gNB-side HO target-quality ADMISSION floor for ALL HOs
                # (vendor admission control, 2026-06-12; same family as the
                # T310 gate above but unconditional). Refuse a HO whose commit
                # target's filtered RSRQ is below the floor: a target that
                # edges past A3 on RSRP while its RSRQ is load/interference-
                # junk cannot complete RACH+RRC at 300 km/h (observed: maesong
                # 900M 390→164, target RSRQ −19.9 / SINR −1…−14 — the field
                # network never attempted it; the sim T304-expired into a
                # FALSE RLF). Default −100 ⇒ OFF. Env HO_TARGET_MIN_RSRQ.
                _floor_q = float(getattr(self.config, "ho_target_min_rsrq_db", -100.0))
                if _floor_q > -99.0:
                    _tgt_rsrq = getattr(self.measurement_engine,
                                        "filtered_rsrq", {}).get(int(commit_target))
                    if (_tgt_rsrq is not None and math.isfinite(float(_tgt_rsrq))
                            and float(_tgt_rsrq) < _floor_q):
                        self.state.ho_suppress_reason = "TGT_RSRQ_WEAK"
                        self._logger.info(
                            f"[{current_time:.2f}s] HO to {commit_target} refused "
                            f"(target RSRQ={float(_tgt_rsrq):.1f} dB < admission "
                            f"floor {_floor_q:.1f} dB)")
                        self._pending_ho_target = None
                        self._pending_ho_t_report = None
                        self._pending_ho_type = ""
                        self._pending_ho_rsrp = None
                        self._pending_ho_decision_delay_s = None
                        commit_target = None

            if commit_target is not None:
                target = commit_target
                ho_type = commit_ho_type
                # Prevent self-HO (target same as serving)
                if target == self.state.serving_cell_id:
                    self._logger.warning(
                        f"[{current_time:.2f}s] {ho_type} HO to cell {target} ignored (same as serving)")
                else:
                    self._logger.info(f"[{current_time:.2f}s] {ho_type} HO Triggered -> Target: {target}")
                    # S2 HO_CMD_RX: the serving-DL HO command (decoded via
                    # _ho_command_decoded() in the buffered path above; legacy
                    # immediate-fire path commits unconditionally) has been
                    # received → HO starts. Stamp the explicit stage.
                    self.state.ho_stage = "HO_CMD_RX"
                    self.state.ho_cmd_decoded = True
                    self.state.target_rach_ok = False
                    self.state.target_rrc_ok = False
                    self.state.signaling.rrc_reconfiguration = True
                    self.state.ho_in_progress = True
                    self.state.target_cell_id = target
                    self.state.ho_type = ho_type
                    # Clear stale RACH snapshots from prior failed HO/T304
                    # expiry (Fix K side-effect). Otherwise rach_state would
                    # still be "exhausted" and rach_attempt_count > max from
                    # the previous HO, causing the new HO to never run RACH
                    # → T304 expiry → spurious RLF without RACH attempts.
                    # Diagnostic 2026-05-19: confirmed via HHE458 probe that
                    # godeok_1.8G_upside_ue1 t=85.66 HO→458 inherited stale
                    # exhausted/attempt=11 from t=74.89 HO→458 failure.
                    for _k in ["rach_state", "rach_attempt_count", "rach_power_dbm"]:
                        self.state.pending_context.pop(_k, None)
                    # Cache target RSRP at HO trigger (prefer raw, fall back
                    # to filtered L3) so the RACH admission gate can recover
                    # from raw-dict NaN gaps in the wide CSV channel during
                    # RACH retries. Default -140 if neither has the target.
                    _raw = self._raw_rsrp_measurements.get(int(target))
                    if _raw is not None and math.isfinite(float(_raw)):
                        self._ho_start_target_rsrp = float(_raw)
                    else:
                        _filt = filtered_meas.get(int(target))
                        if _filt is not None and math.isfinite(float(_filt)):
                            self._ho_start_target_rsrp = float(_filt)
                        else:
                            self._ho_start_target_rsrp = -140.0

                    # Per 3GPP TS 38.331: Stop T310 when HO starts (T304 takes over)
                    if self.state.timers["T310"].running:
                        self._logger.info(f"[{current_time:.2f}s] T310 stopped (HO started, T304 takes over)")
                        self.state.timers["T310"].stop()
                        self.state.counters["N310"].reset()
                        self.state.counters["N311"].reset()
                        # Fix E: T310 epoch ends → reset sustained UL-block tick count
                        self.state.sib_block_ticks_during_t310 = 0
                    # T312 is bound inside the T310 window: HO triggered (T304
                    # takes over) → stop T312 too (TS 38.331 §5.3.5.5.2 stop).
                    if self.state.timers["T312"].running:
                        self._logger.info(f"[{current_time:.2f}s] T312 stopped (HO started)")
                        self.state.timers["T312"].stop()

                    # Start T304
                    self.state.timers["T304"].start(current_time, self.config.t304_ms / 1000.0)

        return self.state

    def _ho_command_decoded(self) -> bool:
        """Decide whether the UE decodes the DL HO command this commit.

        Models RRCReconfiguration(reconfigurationWithSync) delivery on the
        SERVING-cell PDCCH as a hypothetical-PDCCH BLER decode at a robust
        aggregation level (AL16 default) with `ho_command_harq_max` HARQ
        retransmissions:  P(miss) = pdcch_bler(serving_sinr, AL) ** harq_max.

        Returns True (decoded → HO may commit) when the gate is disabled, when
        no serving SINR is available (fail-open — never block on missing data),
        or when the Bernoulli draw succeeds. At normal SINR (e.g. -5 dB → AL16
        BLER ~0.002) P(miss)~0, so this is effectively a no-op; it only bites
        in deep fade (≲ -12 dB) where the command is genuinely lost.
        """
        if not getattr(self.config, "ho_command_delivery_gate", False):
            return True
        serving = self.state.serving_cell_id
        # Prefer raw per-tick serving SINR (instantaneous decode), fall back to
        # the RLM-smoothed value, then fail-open.
        sinr = None
        if serving is not None and hasattr(self, "_sinr_measurements"):
            _s = self._sinr_measurements.get(serving)
            if _s is not None and math.isfinite(float(_s)):
                sinr = float(_s)
        if sinr is None:
            _rs = getattr(self, "_rlm_smoothed_sinr", None)
            if _rs is not None and math.isfinite(float(_rs)):
                sinr = float(_rs)
        if sinr is None:
            return True  # no data → don't block the HO command
        al = int(getattr(self.config, "ho_command_pdcch_al", 16))
        harq = max(1, int(getattr(self.config, "ho_command_harq_max", 4)))
        bler = self._bler_calculator.pdcch_sinr_to_bler(sinr, al)
        p_miss = bler ** harq
        return float(self._ho_decision_rng.random()) >= p_miss

    def _target_rrc_decoded(self, target_cell_id: Optional[int]) -> bool:
        """Decide whether the UE decodes the TARGET-cell RRC config (S4).

        DL mirror of `_ho_command_decoded()`: after RACH to the target
        succeeds (S3), the UE must still receive the target-cell RRC config
        delivery (RRCReconfigurationComplete ack / target SIB+RRC config) on
        the TARGET DL PDCCH before the handover can complete. Modelled as a
        hypothetical-PDCCH BLER decode at a robust aggregation level (AL16
        default) on the TARGET DL SINR with `target_rrc_harq_max` HARQ
        retransmissions:  P(miss) = pdcch_bler(target_sinr, AL) ** harq_max.

        Returns True (config delivered → HO may complete) when the gate is
        disabled, when no target SINR is available (fail-open — never block on
        missing data), or when the Bernoulli draw succeeds. At healthy target
        SINR (e.g. -5 dB → AL16 BLER ~0.002) P(miss)~0, so this is effectively
        a no-op; it only bites in deep target fade where the config delivery
        is genuinely lost (→ HO not completed → T304 expiry → RLF, HOF Case 5).
        """
        if not getattr(self.config, "target_rrc_delivery_gate", False):
            return True
        if target_cell_id is None:
            return True  # no target → don't block (defensive)
        # Prefer raw per-tick target SINR (instantaneous decode), fall back to
        # the RLM-smoothed value, then fail-open.
        sinr = None
        if hasattr(self, "_sinr_measurements"):
            _s = self._sinr_measurements.get(int(target_cell_id))
            if _s is not None and math.isfinite(float(_s)):
                sinr = float(_s)
        if sinr is None:
            _rs = getattr(self, "_rlm_smoothed_sinr", None)
            if _rs is not None and math.isfinite(float(_rs)):
                sinr = float(_rs)
        if sinr is None:
            return True  # no data → don't block target RRC delivery
        al = int(getattr(self.config, "target_rrc_pdcch_al", 16))
        harq = max(1, int(getattr(self.config, "target_rrc_harq_max", 4)))
        bler = self._bler_calculator.pdcch_sinr_to_bler(sinr, al)
        p_miss = bler ** harq
        return float(self._ho_decision_rng.random()) >= p_miss

    def _sample_ho_decision_delay_s(self, base_mean_ms: float,
                                    base_std_ms: float) -> float:
        """Draw a single gNB HO-decision delay (seconds) for a fresh buffer.

        Mean & std are scaled by the recent serving SINR when
        gnb_ho_decision_delay_adapt_to_rf is True:
          - mean *= clip((sinr_smoothed + 5) / 15, 0.3, 2.0)
              SINR=10 → 1.0× ; SINR=−5 → 0× (clip 0.3) ; SINR=25 → 2.0×
              Strong serving → gNB has time to deliberate; weak serving
              → commit fast (no point waiting if serving is collapsing).
          - std  *= clip(window_std / 3.0, 0.5, 2.0)
              Bigger recent SINR variability → wider distribution
              (gNB more uncertain about whether the report is transient).
        Sample is clipped to ≥0 ms.
        """
        import numpy as _np
        mean_ms = float(base_mean_ms)
        std_ms = float(base_std_ms)
        if getattr(self.config, "gnb_ho_decision_delay_adapt_to_rf", False):
            sinr_s = self._rlm_smoothed_sinr if self._rlm_smoothed_sinr is not None else 10.0
            mean_factor = max(0.3, min(2.0, (float(sinr_s) + 5.0) / 15.0))
            mean_ms *= mean_factor
            if len(self._sinr_window) >= 5:
                window_std = float(_np.std(list(self._sinr_window)))
            else:
                window_std = 1.0
            std_factor = max(0.5, min(2.0, window_std / 3.0))
            std_ms *= std_factor
        if std_ms <= 0.0:
            return max(0.0, mean_ms) / 1000.0
        sampled_ms = float(self._ho_decision_rng.normal(mean_ms, std_ms))
        return max(0.0, sampled_ms) / 1000.0

    def _handle_rlf_detection(self, current_time: float):
        """Handle RLF detection per 3GPP TS 38.331 §5.3.10.3.

        Canonical sequence:
          OOS → N310 consecutive (and no guard running) → start T310
          IS while T310 running → N311 consecutive → stop T310
          T310 expiry → declare RLF (deterministic, not rescuable)
        """
        from .rrc_types import RadioLinkStatus

        # ─────────────────────────────────────────────────────────────
        # (A) Consume T310 expiry BEFORE any IS/OOS handling.
        # Rationale: Step 4 (timer.check) sets expired=True while leaving
        # running=True. Without this early consumption, an in-sync
        # indication in the same step could increment N311 to threshold
        # and call T310.stop(), which clears the `expired` flag
        # (TimerState.stop erases it) → RLF would never fire.
        # T310 expiry is a deterministic RLF trigger per spec; nothing
        # must rescue it once the timer has reached expiry.
        # ─────────────────────────────────────────────────────────────
        if self.state.timers["T310"].expired:
            self._logger.error(f"[{current_time:.2f}s] T310 Expired -> RLF DECLARED")
            self._declare_rlf(current_time)
            return

        # (A') T312 expiry (TS 38.331 §5.3.5.5.2): a report was sent while T310
        # was running and T312 reached its (shorter) duration before T310 →
        # declare RLF NOW, faster than the full T310 expiry. T312 can only be
        # running if T310 was running and a report fired, so this is impossible
        # on a healthy cell (precision protection). Checked right after T310 so
        # the fast-RLF takes effect before any IS/N311 rescue this step. The
        # T312_EXPIRY sub-cause is recorded; under RLF_UNIFY it still counts as
        # a single RLF event (details=RLF|cause=T312_EXPIRY).
        if self.state.timers["T312"].expired:
            self._logger.error(f"[{current_time:.2f}s] T312 Expired -> RLF DECLARED (fast RLF)")
            self.state.pending_context["rlf_cause_hint"] = "T312_EXPIRY"
            self._declare_rlf(current_time)
            return

        # ─────────────────────────────────────────────────────────────
        # (B) Strict guard-timer gating per TS 38.331 §5.3.10.3:
        # "Upon receiving N310 consecutive out-of-sync indications while
        #  none of T300, T301, T304, T311, T319 are running: start T310."
        # We read this strictly: while a guard is running, the OOS/IS
        # counting machinery is suspended entirely. N310/N311 are already
        # reset at each guard-start site (T304: `_handle_handover` start,
        # T311: `_declare_rlf`, T300/T301: reached only from non-CONNECTED
        # states where this method does not run), so simply early-return.
        # ─────────────────────────────────────────────────────────────
        guard_timers_running = any(
            self.state.timers.get(t, None) and self.state.timers[t].running
            for t in ("T300", "T301", "T304", "T311", "T319")
        )
        if guard_timers_running:
            return

        # ─────────────────────────────────────────────────────────────
        # (C) Per-frame OOS/IS counting per 3GPP TS 38.133 §8.5.2.2
        # The L1 emits ONE indication per radio frame (10 ms cadence).
        # Our sim step IS the frame cadence (dt_step=10 ms in csv mode).
        # The 200 ms quality smoothing (Step 3-pre) ensures the OOS/IS
        # decision is over a 200 ms-averaged quality estimate, NOT the
        # instantaneous channel sample. No per-step vote needed here.
        #
        # Post-HO warm-up: per TS 38.133 §8.5.2.2 the L1 must accumulate
        # at least T_evaluate_out_DL=200 ms of samples before producing
        # the first OOS/IS indication. While the EMA window is still
        # filling on a freshly-reset serving cell, we suppress all
        # indications (treat as GRAY_ZONE) so a single bad instantaneous
        # sample does not immediately push N310 to threshold.
        # ─────────────────────────────────────────────────────────────
        _rlm_warmup_started = getattr(self, "_rlm_warmup_started_t", None)
        _rlm_tau_ms = float(getattr(self, "_rlm_tau_s", 0.200)) * 1000.0
        if _rlm_warmup_started is None:
            # No measurable serving-cell sample has arrived yet → L1
            # has produced no IS/OOS indication. Suppress counting per
            # TS 38.133 §8.5.2.2.
            return
        if (current_time - _rlm_warmup_started) * 1000.0 < _rlm_tau_ms:
            return  # GRAY_ZONE during warm-up — no N310/N311 change

        if self.state.radio_link_status == RadioLinkStatus.OUT_OF_SYNC:
            # OoS breaks the consecutive-IS run → reset N311.
            if (not self.state.timers["T310"].running
                    and not self.state.counters["N310"].reached):
                self.state.counters["N310"].increment()
            self.state.counters["N311"].reset()

            if (self.state.counters["N310"].reached
                    and not self.state.timers["T310"].running):
                self._logger.warning(f"[{current_time:.2f}s] N310 Reached -> Start T310")
                self.state.timers["T310"].start(current_time, self.config.t310_ms / 1000.0)

        elif self.state.radio_link_status == RadioLinkStatus.IN_SYNC:
            # IS breaks the consecutive-OoS run → reset N310.
            self.state.counters["N310"].reset()
            if self.state.timers["T310"].running:
                self.state.counters["N311"].increment()
                if self.state.counters["N311"].reached:
                    self._logger.info(f"[{current_time:.2f}s] N311 Reached -> Stop T310")
                    self.state.timers["T310"].stop()
                    self.state.counters["N311"].reset()
                    # T312 lives inside T310: N311 recovery stops T310 → stop
                    # T312 too (TS 38.331 §5.3.5.5.2 stop condition).
                    if self.state.timers["T312"].running:
                        self._logger.info(f"[{current_time:.2f}s] T312 stopped (N311 recovery)")
                        self.state.timers["T312"].stop()
                    # Fix E: clean N311 recovery → fresh epoch for next T310.
                    # Without this, sib_block ticks accumulated in the just-stopped
                    # epoch would leak into the NEXT T310 epoch and miscls as SIB.
                    self.state.sib_block_ticks_during_t310 = 0
        # GRAY_ZONE → no L1 indication (TS 38.133 §8.1)

    def _handle_handover_execution(self, current_time: float,
                                   measurements: Dict[int, float]):
        """Handle handover execution with dedicated RACH procedure.

        Models multi-timestep RACH with power ramping per 3GPP TS 38.321.
        RAR wait phase eliminated for polling model (see plan Timing Analysis).
        States: preamble_tx -> success | exhausted -> T304 expiry -> RLF.
        """
        ctx = self.state.pending_context

        # Initialize RACH on first entry
        if "rach_state" not in ctx:
            ctx["rach_state"] = "preamble_tx"
            ctx["rach_attempt_count"] = 0
            ctx["rach_power_dbm"] = self.config.preamble_initial_power_dbm

        # T304 expiry check (always first)
        if self.state.timers["T304"].expired:
            self._logger.error(f"[{current_time:.2f}s] HO Failed (T304 Expired) -> RLF")
            self.state.ho_in_progress = False
            # Fix K (2026-05-19): preserve rach_state snapshot BEFORE pop so
            # `_classify_hof_on_rlf` (inside `_declare_rlf`) can distinguish
            # RACH_PROBLEM vs T304_EXPIRE. Previously the cleanup pop ran
            # first and the classifier saw rach_state=None → always classified
            # as T304_EXPIRE even when preamble_tx_max was reached.
            _rach_state_snapshot = ctx.get("rach_state")
            _rach_attempts_snapshot = int(ctx.get("rach_attempt_count", 0))
            # Clean up RACH context
            for k in ["rach_state", "rach_attempt_count", "rach_power_dbm"]:
                ctx.pop(k, None)
            # Re-publish snapshots under stable keys consumed by the classifier
            ctx["rach_state"] = _rach_state_snapshot
            ctx["rach_attempt_count"] = _rach_attempts_snapshot
            self._declare_rlf(current_time)
            return

        rach_state = ctx["rach_state"]

        # State: preamble_tx - attempt RACH this timestep
        if rach_state == "preamble_tx":
            ctx["rach_attempt_count"] += 1
            attempt = ctx["rach_attempt_count"]

            if attempt > self.config.preamble_tx_max:
                # All attempts exhausted -> terminal state (logs once here)
                self._logger.warning(
                    f"[{current_time:.2f}s] RACH exhausted after {self.config.preamble_tx_max} "
                    f"attempts, awaiting T304 expiry")
                ctx["rach_state"] = "exhausted"
                return

            self.state.signaling.rach_preamble_sent = True

            power = ctx["rach_power_dbm"]
            success_prob = self._calculate_rach_success_probability(power)

            if random.random() < success_prob:
                # S3 TARGET_RACH: RACH to the target succeeded.
                self.state.target_rach_ok = True
                self.state.ho_stage = "TARGET_RACH"
                # RACH success — but delay the actual cell switch to model
                # realistic HO finalization (target SSB resync + RRC
                # Reconfiguration Complete + L1 commit). Without this delay
                # sim was ~0.30 s ahead of field on HO completion.
                proc_delay_s = max(0.0, self.config.ho_processing_delay_ms / 1000.0)
                if proc_delay_s > 0.0:
                    # CRITICAL: stop T304 here — RACH execution is done; T304
                    # exists to bound RACH execution, not the post-RACH UE-side
                    # finalization. Leaving it running causes T304 expiry mid-
                    # finalize and false RLF declarations (default T304=200 ms
                    # < typical proc_delay 450 ms).
                    self.state.timers["T304"].stop()
                    ctx["rach_state"] = "finalizing"
                    ctx["ho_finalize_at"] = current_time + proc_delay_s
                    self._logger.debug(
                        f"[{current_time:.2f}s] RACH success; entering "
                        f"HO_FINALIZING for {proc_delay_s*1000:.0f}ms")
                    return
                ctx["rach_state"] = "success"
                # Fall through to success handling below
            else:
                # Ramp power for next attempt (no RAR wait - polling model)
                ctx["rach_power_dbm"] += self.config.power_ramping_step_db
                self._logger.debug(
                    f"[{current_time:.2f}s] RACH attempt {attempt} failed "
                    f"(power={power:.1f}dBm, prob={success_prob:.2f}), retrying next timestep")
                return

        # State: finalizing — wait for HO processing delay to elapse, then
        # promote to "success" so the cell-switch block below executes.
        if ctx.get("rach_state") == "finalizing":
            if current_time + 1e-9 >= ctx.get("ho_finalize_at", current_time):
                ctx["rach_state"] = "success"
                ctx.pop("ho_finalize_at", None)
                # Fall through to success handling below
            else:
                return

        # State: success - complete handover
        if ctx.get("rach_state") == "success":
            target_id = self.state.target_cell_id
            source_id = self.state.serving_cell_id  # Capture before switch
            attempt = ctx["rach_attempt_count"]

            # === S4 TARGET_RRC_CFG: target-DL RRC-config delivery gate ===
            # After RACH success (S3) and BEFORE the cell switch, the UE must
            # decode the target-cell RRC config delivery on the target DL.
            # Vendor delivery abstraction (mirror of S2); default ON, no-op at
            # healthy target SINR. On miss: do NOT complete the HO this tick —
            # leave ho_in_progress and let T304 run to expiry → RLF (Case 5).
            if not self._target_rrc_decoded(target_id):
                self.state.target_rrc_ok = False
                self.state.ho_stage = "TARGET_RACH"  # stalled at S3/S4 boundary
                self._logger.warning(
                    f"[{current_time:.2f}s] target RRC config NOT decoded "
                    f"(AL{getattr(self.config, 'target_rrc_pdcch_al', 16)} "
                    f"target-DL PDCCH miss, cell {target_id}) → HO NOT "
                    f"completed; awaiting T304 expiry → RLF (HOF Case 5)")
                # Per 3GPP TS 38.331: T304 runs until RRCReconfigurationComplete.
                # The finalize path stops T304 at RACH success to bound only the
                # RACH execution; if config delivery then fails, T304 must run
                # again to its deadline so expiry declares the HO failure. Do
                # not change the T304 *value* — only ensure it is running so the
                # standard expiry → RLF path fires next ticks.
                if not self.state.timers["T304"].running:
                    self.state.timers["T304"].start(
                        current_time, self.config.t304_ms / 1000.0)
                # Park in "rrc_wait": a terminal hold (like "exhausted") that
                # does nothing each tick except let the top-of-handler T304
                # expiry check fire → RLF. We do NOT re-enter "success" (no
                # S4 retry) and do NOT use "finalizing" (whose promotion could
                # race T304 expiry). This guarantees the miss → T304 → RLF path.
                ctx["rach_state"] = "rrc_wait"
                return
            # S4 passed → config delivered. Complete HO as before.
            self.state.target_rrc_ok = True
            self.state.ho_stage = "COMPLETE"

            self._logger.info(
                f"[{current_time:.2f}s] HO Success to Cell {target_id} "
                f"(RACH attempts: {attempt})")
            self.state.ho_in_progress = False
            self.state.timers["T304"].stop()
            self.state.signaling.rrc_reconfiguration_complete = True

            # Switch Cell
            self.state.serving_cell_id = target_id
            self.state.target_cell_id = None
            self.state.ho_count += 1

            # Staged HO flow: HO complete. Leave ho_stage="COMPLETE" and the
            # per-attempt flags (cmd/rach/rrc all True) set so the completing
            # tick is OBSERVABLE in detailed_log (with proc_delay=0 the whole
            # S2→S3→S4 chain collapses into ~1 tick; resetting here would erase
            # the COMPLETE marker before the row is written). They are reset to
            # ""/False at the next HO start (S1 A3_REPORTED / S2 HO_CMD_RX) and
            # on RLF declare / re-establishment.
            self.state.ho_stage = "COMPLETE"
            self.state.ho_cmd_decoded = True
            self.state.target_rach_ok = True
            self.state.target_rrc_ok = True

            # Reset RLF State (new serving cell per 3GPP)
            self.state.timers["T310"].stop()
            self.state.counters["N310"].reset()
            self.state.counters["N311"].reset()
            # New serving cell ⇒ start a fresh T310 epoch tracking window
            self.state.sib_block_ticks_during_t310 = 0
            # Sticky UL block: HO complete → new serving cell, fresh UL state
            self.state.ul_block_active = False
            self.state.ul_block_pending_ticks = 0
            self.state.ul_block_in_sync_start_t = None
            # ─── RLM smoothing reset on cell change (TS 38.133 §8.5.2.2) ───
            # New serving cell → previous SINR/RSRP averages are stale.
            # Re-arm warm-up clock so the next 200 ms suppresses OOS/IS
            # indications (the L1 needs ≥T_evaluate_out_DL of samples
            # before it can produce a valid first indication).
            self._rlm_smoothed_sinr = None
            self._rlm_smoothed_rsrp = None
            self._rlm_last_smooth_t = None
            self._rlm_warmup_started_t = None

            # Reset measConfig-derived state per TS 38.331 §5.3.5.5: new serving
            # cell → cellsTriggeredList invalidated. PRESERVE _last_report_time
            # (reportInterval gating) across HO_COMPLETE — operationally the
            # measId is unchanged through RRCReconfiguration, so reportInterval
            # state should survive HO to prevent sub-reportInterval A3 ping-pong
            # (e.g., 14.88→15.01 = 130ms gap < 480ms reportInterval, observed
            # before this fix). L3 filter/RSRQ caches preserved.
            self.measurement_engine.reset_for_meas_config_change(preserve_report_intervals=True)

            # Vendor gNB MRO post-HO blacklist (TS 38.473 / TS 28.541 NRM):
            # after HO source→target, suppress A3 reports for `source` while
            # serving=target for post_ho_blacklist_s seconds. No-op when 0.
            if hasattr(self.measurement_engine, "add_post_ho_blacklist"):
                try:
                    self.measurement_engine.add_post_ho_blacklist(
                        source_id, target_id, current_time
                    )
                except Exception:
                    self._logger.debug("post-HO blacklist add failed", exc_info=True)

            # ── HOF: Record HO history & check ping-pong (Case 4) ──
            ho_duration_ms = 0.0
            if self.state.timers["T304"].start_time > 0:
                ho_duration_ms = (current_time - self.state.timers["T304"].start_time) * 1000.0
            self._record_ho_history(
                current_time=current_time,
                source=source_id,
                target=target_id,
                success=True,
                ho_type=self.state.ho_type or "A3_intra",
                rach_attempts=attempt,
                duration_ms=ho_duration_ms
            )
            self._check_ping_pong(current_time, source_id, target_id)

            # Clean up RACH context
            for k in ["rach_state", "rach_attempt_count", "rach_power_dbm"]:
                ctx.pop(k, None)

        # State: exhausted - terminal, do nothing (T304 expiry checked above)
        elif ctx.get("rach_state") == "exhausted":
            pass

    def _handle_reestablishment(self, current_time: float,
                                measurements: Dict[int, float]):
        """Handle RRC re-establishment with contention-based RACH.

        Three phases per 3GPP TS 38.331 §5.3.7:
        1. cell_selection: T311 running, search for suitable cell (RSRP > -100)
        2. rach: Contention-based RACH to selected cell (one attempt per timestep)
        3. waiting_response: T301 running, waiting for network accept/reject

        RACH failure -> retry cell selection.
        T311 expiry at any phase -> RRC_IDLE.
        """
        from .rrc_types import RRCState
        ctx = self.state.pending_context

        # Initialize phase if not set
        if "reest_phase" not in ctx:
            ctx["reest_phase"] = "cell_selection"

        # T311 expiry check (any phase)
        if self.state.timers["T311"].expired:
            self.state.timers["T311"].stop()
            self._logger.error(f"[{current_time:.2f}s] T311 Expired -> IDLE")

            # ── HOF: T311 expiry means re-establishment failed entirely.
            # Classify based on the original RLF cause (stored by _classify_hof_on_rlf).
            # Re-establishment cell = -1 (none found).
            from .rrc_types import HOFType, HOFClassificationResult
            rlf_cause = ctx.get("rlf_cause", "UNKNOWN")
            if rlf_cause == "T304_EXPIRE":
                hof_type = HOFType.T304_EXPIRY
                cause_str = "T304 expiry + T311 expiry (re-estab failed entirely)"
            else:
                hof_type = HOFType.TOO_LATE
                cause_str = "T310 expiry + T311 expiry (no cell found for re-estab)"

            result = HOFClassificationResult(
                hof_type=hof_type,
                timestamp=current_time,
                rlf_cell_id=ctx.get("rlf_cell_id", -1),
                reestablishment_cell_id=-1,
                last_ho_source=-1,
                last_ho_target=-1,
                time_since_last_ho_ms=-1,
                cause=cause_str
            )
            self.state.hof_classifications.append(result)
            self.state.last_hof_classification = result
            self._logger.warning(
                f"[{current_time:.2f}s] *** HOF CLASSIFIED: {hof_type.value} *** "
                f"| {cause_str}")

            self.state.rrc_state = RRCState.RRC_IDLE
            self.state.rrc_connected = False
            self.state.rlf_declared = False
            # Clean up ALL reest keys AND rlf context keys
            for k in list(ctx.keys()):
                if k.startswith("reest_") or k.startswith("rlf_"):
                    ctx.pop(k, None)
            return

        reest_phase = ctx["reest_phase"]

        # Phase 1: Cell Selection (T311 running)
        if reest_phase == "cell_selection":
            if not self.state.timers["T311"].running:
                return  # guard

            best_cell = None
            best_rsrp = -999
            for cid, rsrp in measurements.items():
                if rsrp > -100 and rsrp > best_rsrp:
                    best_rsrp = rsrp
                    best_cell = cid

            if best_cell is not None:
                self._logger.info(
                    f"[{current_time:.2f}s] Cell {best_cell} found -> "
                    f"SSB measurement (1 timestep delay before RACH)")
                ctx["reest_target"] = best_cell
                ctx["reest_phase"] = "cell_found_pending"
            # else: keep searching (T311 still ticking)

        # Phase 1b: Cell found, SSB measurement delay (1 timestep)
        elif reest_phase == "cell_found_pending":
            self._logger.info(
                f"[{current_time:.2f}s] SSB measured on Cell {ctx.get('reest_target')} "
                f"-> start contention-based RACH")
            ctx["reest_phase"] = "rach"
            ctx["reest_rach_attempt"] = 0
            ctx["reest_rach_power"] = self.config.reest_preamble_initial_power_dbm

        # Phase 2: Contention-Based RACH (one attempt per timestep, no RAR wait)
        elif reest_phase == "rach":
            ctx["reest_rach_attempt"] += 1
            attempt = ctx["reest_rach_attempt"]

            if attempt > self.config.preamble_tx_max:
                # RACH failed -> retry cell selection (3GPP TS 38.331 §5.3.7)
                self._logger.warning(
                    f"[{current_time:.2f}s] Contention-based RACH failed after "
                    f"{self.config.preamble_tx_max} attempts -> retry cell selection")
                ctx["reest_phase"] = "cell_selection"
                for k in ["reest_rach_attempt", "reest_rach_power", "reest_target"]:
                    ctx.pop(k, None)
                return

            self.state.signaling.rach_preamble_sent = True

            power = ctx["reest_rach_power"]
            success_prob = self._calculate_rach_success_probability(power)

            if random.random() < success_prob:
                # RACH success -> RRCReestablishmentRequest
                self._logger.info(
                    f"[{current_time:.2f}s] Contention-based RACH success "
                    f"(attempt {attempt}) -> RRCReestablishmentRequest")
                self.state.timers["T311"].stop()
                self.state.timers["T301"].start(current_time, self.config.t301_ms / 1000.0)
                self.state.signaling.rrc_reestablishment_request = True
                ctx["reest_phase"] = "waiting_response"
                # Clean RACH sub-state keys (keep reest_target for event detection)
                for k in ["reest_rach_attempt", "reest_rach_power"]:
                    ctx.pop(k, None)
            else:
                # Ramp power for next attempt (no RAR wait - polling model)
                ctx["reest_rach_power"] += self.config.power_ramping_step_db
                self._logger.debug(
                    f"[{current_time:.2f}s] Contention RACH attempt {attempt} failed "
                    f"(power={power:.1f}dBm), retrying next timestep")

        # Phase 3: Waiting for Network Response (T301 running)
        elif reest_phase == "waiting_response":
            if self.state.timers["T301"].expired:
                self.state.timers["T301"].stop()
                # 3GPP TS 38.331 §5.3.7.7: T301 expiry → RRC_IDLE
                self._logger.warning(
                    f"[{current_time:.2f}s] T301 Expired -> RRC_IDLE (3GPP §5.3.7.7)")

                # HOF classification (same logic as T311 expiry)
                from .rrc_types import HOFType, HOFClassificationResult
                rlf_cause = ctx.get("rlf_cause", "UNKNOWN")
                if rlf_cause == "T304_EXPIRE":
                    hof_type = HOFType.T304_EXPIRY
                    cause_str = "T304 expiry + T301 expiry (network response timeout)"
                else:
                    hof_type = HOFType.TOO_LATE
                    cause_str = "T310 expiry + T301 expiry (network response timeout)"

                result = HOFClassificationResult(
                    hof_type=hof_type,
                    timestamp=current_time,
                    rlf_cell_id=ctx.get("rlf_cell_id", -1),
                    reestablishment_cell_id=-1,
                    last_ho_source=-1,
                    last_ho_target=-1,
                    time_since_last_ho_ms=-1,
                    cause=cause_str
                )
                self.state.hof_classifications.append(result)
                self.state.last_hof_classification = result
                self._logger.warning(
                    f"[{current_time:.2f}s] *** HOF CLASSIFIED: {hof_type.value} *** "
                    f"| {cause_str}")

                self.state.rrc_state = RRCState.RRC_IDLE
                self.state.rrc_connected = False
                self.state.rlf_declared = False
                for k in list(ctx.keys()):
                    if k.startswith("reest_") or k.startswith("rlf_"):
                        ctx.pop(k, None)
                return

            # Simulate network response (20ms delay after T301 start)
            elapsed = current_time - self.state.timers["T301"].start_time
            if elapsed >= 0.02:
                accept_probability = 0.95
                if random.random() < accept_probability:
                    # Re-establishment SUCCESS
                    target = ctx.get("reest_target")
                    self._logger.info(
                        f"[{current_time:.2f}s] Re-establishment Success to Cell {target}")
                    self.state.timers["T301"].stop()
                    self.state.signaling.rrc_reestablishment = True
                    self.state.signaling.rrc_reestablishment_complete = True
                    self.state.rlf_declared = False
                    self.state.rrc_state = RRCState.RRC_CONNECTED
                    self.state.rrc_connected = True
                    self.state.serving_cell_id = target
                    self.state.reestablishment_count += 1
                    # Staged HO flow: re-establishment → reset stage + flags.
                    self.state.ho_stage = ""
                    self.state.ho_cmd_decoded = False
                    self.state.target_rach_ok = False
                    self.state.target_rrc_ok = False

                    # Reset N310/N311 after re-establishment (3GPP)
                    self.state.counters["N310"].reset()
                    self.state.counters["N311"].reset()
                    self.state.timers["T310"].stop()
                    # Fix E: re-establishment → fresh serving cell, fresh T310 epoch
                    self.state.sib_block_ticks_during_t310 = 0
                    # ─── RLM smoothing reset on cell change (TS 38.133 §8.5.2.2) ───
                    self._rlm_smoothed_sinr = None
                    self._rlm_smoothed_rsrp = None
                    self._rlm_last_smooth_t = None
                    self._rlm_warmup_started_t = None

                    # Reset measConfig-derived state per TS 38.331 §5.3.5.5
                    # (re-establishment success → new RSC → measConfig refreshed).
                    self.measurement_engine.reset_for_meas_config_change()

                    # ── HOF: Final classification (Cases 1-5) ──
                    # Must be called BEFORE cleaning up rlf_* context keys
                    if target is not None:
                        self._classify_hof_on_reestablishment(
                            current_time, reest_cell_id=target)

                    # Clean up ALL reest keys
                    for k in list(ctx.keys()):
                        if k.startswith("reest_"):
                            ctx.pop(k, None)
                else:
                    # 3GPP TS 38.331 §5.3.7.6: RRCReject received → RRC_IDLE
                    self._logger.warning(
                        f"[{current_time:.2f}s] RRC Re-establishment rejected "
                        f"-> RRC_IDLE (3GPP §5.3.7.6)")
                    self.state.timers["T301"].stop()

                    # HOF classification
                    from .rrc_types import HOFType, HOFClassificationResult
                    rlf_cause = ctx.get("rlf_cause", "UNKNOWN")
                    if rlf_cause == "T304_EXPIRE":
                        hof_type = HOFType.T304_EXPIRY
                        cause_str = "T304 expiry + network reject (re-estab rejected)"
                    else:
                        hof_type = HOFType.TOO_LATE
                        cause_str = "T310 expiry + network reject (re-estab rejected)"

                    result = HOFClassificationResult(
                        hof_type=hof_type,
                        timestamp=current_time,
                        rlf_cell_id=ctx.get("rlf_cell_id", -1),
                        reestablishment_cell_id=-1,
                        last_ho_source=-1,
                        last_ho_target=-1,
                        time_since_last_ho_ms=-1,
                        cause=cause_str
                    )
                    self.state.hof_classifications.append(result)
                    self.state.last_hof_classification = result
                    self._logger.warning(
                        f"[{current_time:.2f}s] *** HOF CLASSIFIED: {hof_type.value} *** "
                        f"| {cause_str}")

                    self.state.rrc_state = RRCState.RRC_IDLE
                    self.state.rrc_connected = False
                    self.state.rlf_declared = False
                    for k in list(ctx.keys()):
                        if k.startswith("reest_") or k.startswith("rlf_"):
                            ctx.pop(k, None)

    def _handle_rrc_setup(self, current_time: float,
                          measurements: Dict[int, float]):
        """Handle RRC Connection Setup for IDLE recovery (3GPP TS 38.331 §5.3.3).

        Called when UE is in RRC_IDLE after re-establishment failure (T311/T301
        expiry or network reject). Performs a fresh connection setup:

        Four phases:
        1. cell_selection: Search for cell with RSRP > -100 dBm (infinite retry)
        2. cell_found_pending: SSB measurement delay (1 timestep)
        3. rach: Contention-based RACH (power ramping, reuses _calculate_rach_success_probability)
        4. waiting_response: T300 guard timer, 20ms network response delay, 100% accept

        Key differences from re-establishment:
        - No T311 (cell selection retries indefinitely)
        - 100% accept (fresh connection, no security context to verify)
        - No HOF classification (new connection, not recovery)
        - pending_context key prefix: "setup_" (not "reest_")
        """
        from .rrc_types import RRCState
        ctx = self.state.pending_context

        # Initialize phase if not set
        if "setup_phase" not in ctx:
            ctx["setup_phase"] = "cell_selection"

        setup_phase = ctx["setup_phase"]

        # Phase 1: Cell Selection (no T311 — infinite retry)
        if setup_phase == "cell_selection":
            best_cell = None
            best_rsrp = -999
            for cid, rsrp in measurements.items():
                if rsrp > -100 and rsrp > best_rsrp:
                    best_rsrp = rsrp
                    best_cell = cid

            if best_cell is not None:
                self._logger.info(
                    f"[{current_time:.2f}s] RRC Setup: Cell {best_cell} found "
                    f"(RSRP={best_rsrp:.1f}) -> SSB measurement (1 timestep delay)")
                ctx["setup_target"] = best_cell
                ctx["setup_phase"] = "cell_found_pending"
            # else: keep searching next timestep

        # Phase 1b: Cell found, SSB measurement delay (1 timestep)
        elif setup_phase == "cell_found_pending":
            self._logger.info(
                f"[{current_time:.2f}s] RRC Setup: SSB measured on Cell "
                f"{ctx.get('setup_target')} -> start contention-based RACH")
            ctx["setup_phase"] = "rach"
            ctx["setup_rach_attempt"] = 0
            ctx["setup_rach_power"] = self.config.setup_preamble_initial_power_dbm

        # Phase 2: Contention-Based RACH (one attempt per timestep)
        elif setup_phase == "rach":
            ctx["setup_rach_attempt"] += 1
            attempt = ctx["setup_rach_attempt"]

            if attempt > self.config.preamble_tx_max:
                # RACH failed -> retry cell selection
                self._logger.warning(
                    f"[{current_time:.2f}s] RRC Setup: RACH failed after "
                    f"{self.config.preamble_tx_max} attempts -> retry cell selection")
                ctx["setup_phase"] = "cell_selection"
                for k in ["setup_rach_attempt", "setup_rach_power", "setup_target"]:
                    ctx.pop(k, None)
                return

            self.state.signaling.rach_preamble_sent = True

            power = ctx["setup_rach_power"]
            # Temporarily set target_cell_id for UL SINR RACH lookup
            orig_target = self.state.target_cell_id
            self.state.target_cell_id = ctx.get("setup_target")
            success_prob = self._calculate_rach_success_probability(power)
            self.state.target_cell_id = orig_target

            if random.random() < success_prob:
                # RACH success -> RRCSetupRequest (3GPP TS 38.331 §5.3.3)
                self._logger.info(
                    f"[{current_time:.2f}s] RRC Setup: RACH success "
                    f"(attempt {attempt}) -> RRCSetupRequest")
                self.state.signaling.rrc_setup_request = True
                self.state.timers["T300"].start(current_time, self.config.t300_ms / 1000.0)
                ctx["setup_phase"] = "waiting_response"
                # Clean RACH sub-state keys (keep setup_target)
                for k in ["setup_rach_attempt", "setup_rach_power"]:
                    ctx.pop(k, None)
            else:
                # Ramp power for next attempt
                ctx["setup_rach_power"] += self.config.power_ramping_step_db
                self._logger.debug(
                    f"[{current_time:.2f}s] RRC Setup: RACH attempt {attempt} failed "
                    f"(power={power:.1f}dBm), retrying next timestep")

        # Phase 3: Waiting for Network Response (T300 running)
        elif setup_phase == "waiting_response":
            if self.state.timers["T300"].expired:
                self.state.timers["T300"].stop()
                # T300 expiry -> retry cell selection (3GPP TS 38.331 §5.3.3.6)
                self._logger.warning(
                    f"[{current_time:.2f}s] RRC Setup: T300 expired -> retry cell selection")
                for k in list(ctx.keys()):
                    if k.startswith("setup_"):
                        ctx.pop(k, None)
                ctx["setup_phase"] = "cell_selection"
                return

            # Simulate network response (20ms delay after T300 start)
            elapsed = current_time - self.state.timers["T300"].start_time
            if elapsed >= 0.02:
                # 100% accept (fresh connection, no security context)
                target = ctx.get("setup_target")
                self._logger.info(
                    f"[{current_time:.2f}s] RRC Setup: Connection established to Cell {target}")
                self.state.timers["T300"].stop()
                self.state.signaling.rrc_setup = True
                self.state.signaling.rrc_setup_complete = True
                self.state.rrc_state = RRCState.RRC_CONNECTED
                self.state.rrc_connected = True
                self.state.serving_cell_id = target
                self.state.rrc_setup_count += 1

                # Reset N310/N311 (fresh connection)
                self.state.counters["N310"].reset()
                self.state.counters["N311"].reset()
                self.state.timers["T310"].stop()
                # Fix E: RRC Setup → fresh CONNECTED state, no T310 epoch carried over
                self.state.sib_block_ticks_during_t310 = 0
                # ─── RLM smoothing reset on cell change (TS 38.133 §8.5.2.2) ───
                self._rlm_smoothed_sinr = None
                self._rlm_smoothed_rsrp = None
                self._rlm_last_smooth_t = None
                self._rlm_warmup_started_t = None

                # Reset measConfig-derived state per TS 38.331 §5.3.5.5 (RRC
                # setup IDLE→CONNECTED → fresh measConfig applied).
                self.measurement_engine.reset_for_meas_config_change()

                # Clean up ALL setup keys
                for k in list(ctx.keys()):
                    if k.startswith("setup_"):
                        ctx.pop(k, None)

    def _declare_rlf(self, current_time: float):
        """Declare Radio Link Failure per 3GPP TS 38.331 §5.3.10"""
        self._logger.warning(f"[{current_time:.2f}s] RLF DECLARED")

        # ── HOF: Pre-classify before modifying state ──
        # Must be called BEFORE clearing ho_in_progress / target_cell_id
        self._classify_hof_on_rlf(current_time)

        # If T304 was running (HO failure), record failed HO in history
        if self.state.ho_in_progress and self.state.target_cell_id is not None:
            self._record_ho_history(
                current_time=current_time,
                source=self.state.serving_cell_id,
                target=self.state.target_cell_id,
                success=False,
                failure_reason="T304_EXPIRE",
                rach_attempts=self.state.pending_context.get("rach_attempt_count", 0)
            )

        self.state.rlf_declared = True
        self.state.rrc_connected = False
        self.state.rlf_count += 1

        # Stop all procedure timers
        self.state.timers["T310"].stop()
        self.state.timers["T304"].stop()
        # T312 is bound inside the T310 window — stop it on RLF declare too
        # (TS 38.331 §5.3.5.5.2). Whether RLF was caused by T310 OR T312
        # expiry, both timers are cleared here.
        self.state.timers["T312"].stop()
        # Fix E: RLF declared → reset sustained UL-block tick counter.
        # IMPORTANT: this runs AFTER _classify_hof_on_rlf (called above),
        # so the classifier already consumed the counter to decide
        # SIB vs RLF cause. Reset is for the NEXT T310 epoch.
        self.state.sib_block_ticks_during_t310 = 0

        # Reset HO state (if HO was in progress)
        self.state.ho_in_progress = False
        self.state.target_cell_id = None
        # Staged HO flow: RLF declare → reset stage + per-attempt flags.
        self.state.ho_stage = ""
        self.state.ho_cmd_decoded = False
        self.state.target_rach_ok = False
        self.state.target_rrc_ok = False

        # Sticky UL block: RLF ends current cycle, fresh state for re-establishment
        self.state.ul_block_active = False
        self.state.ul_block_pending_ticks = 0
        self.state.ul_block_in_sync_start_t = None

        # Reset RLF counters
        self.state.counters["N310"].reset()
        self.state.counters["N311"].reset()

        # Reset measConfig-derived state on RLF (cellsTriggeredList +
        # VarMeasReportList invalid; re-establishment will rebuild them).
        self.measurement_engine.reset_for_meas_config_change()

        # Start T311 for re-establishment
        self.state.timers["T311"].start(current_time, self.config.t311_ms / 1000.0)

    def _calculate_rach_success_probability(self, rach_power_dbm: float) -> float:
        """
        Calculate RACH success probability.

        Two modes:
        1. UL SINR-based (use_ul_sinr_for_rach=True): Sigmoid detection model
           calibrated per 3GPP TS 38.141-1 §8.4 (Pd>=99% at SNR=-14dB).
        2. Legacy power-based (default): preamble TX power vs sensitivity.

        Args:
            rach_power_dbm: Current RACH preamble transmit power in dBm.

        Returns:
            Probability in [0, 1].
        """
        target = self.state.target_cell_id

        # ── Pre-RACH admission gate (Fix L+M — 2026-05-19) ───────────────
        # Unified with the UL msg3 block criterion (Fix I+M) so the same
        # (RSRP, RSRQ) joint condition that suppresses A3 report delivery
        # also blocks RACH on the same kind of target.
        #   Path A:  RSRP < -115 dBm AND RSRQ < -18.5 dB
        #   Path B:  RSRQ <= -19 dB (any RSRP) — INCLUSIVE, reads the same
        #            config knob as the UL report-block gate (2026-06-11 user
        #            request: "rsrq가 -19여도 msg3이 성공안되게").
        # When either path holds → RACH cannot succeed (return 0.0).
        # The legacy Q_RxLevMin check (TS 38.304 §5.2.3.2) remains as a
        # weak fallback for the RSRP-only catastrophe (target invisible).
        if target is not None:
            target_rsrp = self._raw_rsrp_measurements.get(target)
            target_rsrq = None
            if hasattr(self, "measurement_engine"):
                target_rsrq = getattr(self.measurement_engine, "filtered_rsrq", {}).get(int(target))
            qmin = float(getattr(self.config, "q_rxlevmin_dbm", -140.0))
            if target_rsrp is None or not math.isfinite(float(target_rsrp)):
                # Fallback chain for transient NaN in wide CSV at RACH tick.
                # Field-equivalent UE would carry the last L3-filtered sample
                # through brief measurement gaps; raw-dict NaN at the RACH-
                # execution tick is a data artifact, not signal loss.
                # Step 1: try filtered (L3) RSRP from measurement engine
                if hasattr(self, "measurement_engine"):
                    _filtered = getattr(self.measurement_engine,
                                        "filtered_rsrp", {}).get(int(target))
                    if (_filtered is not None and
                            math.isfinite(float(_filtered))):
                        target_rsrp = _filtered
                # Step 2: fall back to RSRP captured at HO_START
                if target_rsrp is None or not math.isfinite(float(target_rsrp)):
                    _cached = getattr(self, "_ho_start_target_rsrp", None)
                    if _cached is not None and math.isfinite(float(_cached)):
                        target_rsrp = _cached
                if target_rsrp is None or not math.isfinite(float(target_rsrp)):
                    return 0.0
            # Joint (RSRP, RSRQ) admission gate — two paths
            rp = float(target_rsrp)
            rq = float(target_rsrq) if (target_rsrq is not None and math.isfinite(float(target_rsrq))) else None
            _pa_rsrp = float(getattr(self.config, "ul_block_path_a_rsrp_dbm", -115.0))
            _pa_rsrq = float(getattr(self.config, "ul_block_path_a_rsrq_db", -18.5))
            _pb_rsrq = float(getattr(self.config, "ul_block_path_b_rsrq_db", -19.3))
            path_a = (rq is not None and rp < _pa_rsrp and rq < _pa_rsrq)
            path_b = (rq is not None and rq <= _pb_rsrq)
            if path_a or path_b:
                self._logger.debug(
                    f"RACH admission gate: target={target} "
                    f"(path {'A' if path_a else 'B'}, RSRP={rp:.1f} "
                    f"RSRQ={rq:.1f}) → fail")
                return 0.0
            # Weak fallback (legacy Q_RxLevMin): still apply for the
            # RSRP-only catastrophe (e.g., -130 dBm with NaN RSRQ).
            if rp < qmin:
                self._logger.debug(
                    f"RACH admission gate: target={target} RSRP="
                    f"{rp:.1f} dBm < Q_RxLevMin={qmin:.1f} → fail")
                return 0.0
            # Strong-target shortcut (2026-05-19): if target DL signal is
            # clearly healthy (RSRP ≥ thr AND SINR ≥ thr), bypass the UL
            # SINR sigmoid. Diagnostic finding (2026-05-19): in CSV channel
            # mode, _ul_sinr_measurements for non-serving cells equals DL
            # SINR (no UL/DL decoupling), so the sigmoid spuriously fails
            # over T304 window for targets that DL-wise are fully viable.
            # TS 38.304 §5.2.3.2 cell selection is DL-based; TS 38.214
            # bounds UL/DL coupling. Vendor calibration layer, same kind
            # of knob as _rsrq_rach_penalty.
            if getattr(self.config, "rach_strong_target_enabled", False):
                _sinr = (self._sinr_measurements.get(target)
                         if hasattr(self, "_sinr_measurements") else None)
                _rp_thr = float(self.config.rach_strong_target_rsrp_dbm)
                _si_thr = float(self.config.rach_strong_target_sinr_db)
                if (_sinr is not None and math.isfinite(float(_sinr)) and
                        rp >= _rp_thr and float(_sinr) >= _si_thr):
                    return float(self.config.rach_strong_target_probability)

        if self.config.use_ul_sinr_for_rach and self._ul_sinr_measurements:
            if target is not None and target in self._ul_sinr_measurements:
                p_msg1 = self._ul_sinr_rach_probability(
                    self._ul_sinr_measurements[target], rach_power_dbm)
                # TS 38.321 §5.1: RACH targets the target cell — use target's
                # (RSRQ, RSRP) for the delivery-block penalty, not serving's.
                return p_msg1 * self._rsrq_rach_penalty(self.state.target_cell_id)
        # Fallback: existing power-based formula
        base_success_prob = 0.90
        power_factor = max(0.0, min(1.0, (rach_power_dbm + 110) / 20))
        # TS 38.321 §5.1: RACH targets the target cell — use target's
        # (RSRQ, RSRP) for the delivery-block penalty, not serving's.
        return base_success_prob * power_factor * self._rsrq_rach_penalty(self.state.target_cell_id)

    def _rsrq_rach_penalty(self, cell_id: Optional[int] = None) -> float:
        """
        Joint (RSRQ, RSRP) gate for UL msg3 / RACH delivery block.

        Args:
            cell_id: which cell's filtered RSRQ/RSRP to use.
                     - None (default): self.state.serving_cell_id (UL-block path,
                       models PUCCH/PUSCH delivery failure on the source cell)
                     - explicit int: that cell (RACH-on-target path per
                       TS 38.321 §5.1 — RACH targets the target cell during HO)

        Both criteria must be met to apply a penalty:
          - cell RSRQ < rsrq_rach_penalty_full_db  (default -20.0 dB)
          - cell RSRP < rsrq_rach_penalty_rsrp_gate_dbm  (default -100.0 dBm)

        FIELD 18-case calibration (see `rsrq_rach_penalty_theory.md` v2):
        all 12 UL-cluster failures satisfy RSRQ < -20 AND RSRP < -100.
        RLM-driven pure RLF (RSRP -97~-99 with bad SINR) is excluded here —
        it is handled separately by N310/T310 SINR-based RLM.

        NaN/missing RSRQ or RSRP → 1.0 (no penalty; avoids double-penalizing
        the existing NaN-target admission gate). If only one criterion is met
        (RSRQ bad but RSRP good, or vice versa) → 1.0 (no block).

        When full_db == floor_db (step mode, current default): returns floor
        (0.0 = full block) when both criteria met. When full_db > floor_db:
        linearly interpolates from 1.0 at full_db down to floor at floor_db
        (preserves soft-transition semantics for future tuning).
        """
        if not getattr(self.config, "rsrq_rach_penalty_enabled", False):
            return 1.0
        serv = cell_id if cell_id is not None else self.state.serving_cell_id
        if serv is None or serv < 0:
            return 1.0
        rsrq = getattr(self.measurement_engine, "filtered_rsrq", {}).get(int(serv))
        if rsrq is None or not math.isfinite(float(rsrq)):
            return 1.0
        rsrp = getattr(self.measurement_engine, "filtered_rsrp", {}).get(int(serv))
        if rsrp is None or not math.isfinite(float(rsrp)):
            return 1.0
        full = float(self.config.rsrq_rach_penalty_full_db)
        floor_db = float(self.config.rsrq_rach_penalty_floor_db)
        floor = float(self.config.rsrq_rach_penalty_floor)
        rsrp_gate = float(getattr(self.config, "rsrq_rach_penalty_rsrp_gate_dbm", -100.0))
        r = float(rsrq)
        # Joint gate: BOTH criteria must be met to apply any penalty
        if r >= full or float(rsrp) >= rsrp_gate:
            return 1.0
        # Both criteria met: RSRQ < full_db AND RSRP < rsrp_gate
        if r <= floor_db:
            return floor
        # linear interpolate from (full, 1.0) to (floor_db, floor)
        frac = (full - r) / (full - floor_db)
        return 1.0 - frac * (1.0 - floor)

    def _ul_sinr_rach_probability(self, ul_sinr_db: float, rach_power_dbm: float) -> float:
        """
        RACH success probability from UL SINR with power ramping.

        3GPP TS 38.141-1 §8.4: PRACH detection Pd >= 99% at SNR = -14 dB.
        Sigmoid model calibrated: Pd=99% at -14dB, Pd=50% at -20dB.
        Power ramping improves effective SNR relative to initial power.

        Args:
            ul_sinr_db: UL SINR at gNB receiver
            rach_power_dbm: Current preamble TX power (after ramping)

        Returns:
            Detection probability * no-collision probability
        """
        ramp_gain = max(0.0, rach_power_dbm - self.config.preamble_initial_power_dbm)
        effective_sinr = ul_sinr_db + ramp_gain
        threshold, steepness = -20.0, 0.77
        p_detect = 1.0 / (1.0 + math.exp(-steepness * (effective_sinr - threshold)))
        return p_detect * 0.99  # 0.99 = approximate no-collision probability

    # ══════════════════════════════════════════════════════════════════
    # HOF Classification (3GPP TS 38.300 §15.5 MRO)
    # ══════════════════════════════════════════════════════════════════

    def _record_ho_history(self, current_time: float, source: int,
                           target: int, success: bool,
                           ho_type: str = "", rach_attempts: int = 0,
                           duration_ms: float = 0.0,
                           failure_reason: str = ""):
        """
        Record a handover event in the rolling history buffer.

        Called after every HO completion (success or failure) so the
        classifier can look back when an RLF occurs.
        """
        from .rrc_types import HOHistoryEntry

        entry = HOHistoryEntry(
            timestamp=current_time,
            source_cell_id=source,
            target_cell_id=target,
            success=success,
            ho_type=ho_type,
            rach_attempts=rach_attempts,
            duration_ms=duration_ms,
            failure_reason=failure_reason
        )
        self.state.ho_history.append(entry)

        # Trim to max size
        if len(self.state.ho_history) > self.state.ho_history_max:
            self.state.ho_history = self.state.ho_history[-self.state.ho_history_max:]

        self._logger.debug(
            f"[{current_time:.2f}s] HO history recorded: "
            f"{source}->{target} success={success} reason={failure_reason}")

    def _classify_hof_on_rlf(self, current_time: float):
        """
        Initial HOF classification when RLF is declared.

        Case 5 (T304 Expiry) is classified immediately here because
        we know T304 was running.  Cases 1-3 require knowing where
        the UE re-establishes, so they are deferred to
        _classify_hof_on_reestablishment().

        Stores the RLF context in pending_context for later use.
        """
        ctx = self.state.pending_context

        # Consume the one-shot T312 fast-RLF hint (set by _handle_rlf_detection
        # right before _declare_rlf). Popped here so it can never leak into a
        # later, unrelated RLF classification.
        _t312_hint = ctx.pop("rlf_cause_hint", None)

        # Store RLF context for deferred classification
        ctx["rlf_time"] = current_time
        ctx["rlf_cell_id"] = self.state.serving_cell_id
        ctx["rlf_ho_was_in_progress"] = self.state.ho_in_progress
        ctx["rlf_target_cell_id"] = self.state.target_cell_id
        # Snapshot sib_block_ticks BEFORE _declare_rlf resets the counter,
        # so downstream event emission can include it in details.
        ctx["rlf_sib_ticks"] = getattr(self.state, "sib_block_ticks_during_t310", 0)

        # Fix E — Spec-aligned RLF cause classification per
        # 3GPP TS 38.331 §5.3.10 + field rlf_cause schema:
        #   - "RACH_PROBLEM": RACH exhausted during HO (msg1/msg3 max retx)
        #   - "T304_EXPIRE": HO timer expired without RACH-exhaust (other timing)
        #   - "SIB_READ_FAILURE": sustained UL block during T310 (≥5 ticks = 50ms)
        #     with a viable HO target suppressed — bidirectional signaling fail
        #   - "RLF" (RLM-driven): T310 expire on pure DL Qout, no UL-block dominance
        # SIB_READ_FAILURE classification: sustained UL block with a viable
        # HO target suppressed during T310. Empirical distribution across
        # 36 cases (87 RLFs) is bimodal: 6 events at 0 ticks, ~60 events
        # at 8 ticks (transient ~80ms UL blip), ~19 events at 16 ticks
        # (sustained ~160ms UL block). Threshold 10 (100ms = half of
        # T_evaluate_out_DL=200ms) cleanly separates the 8-cluster from
        # the 16-cluster, yielding ~22% SIB which matches FIELD's 29%
        # SIB_READ_FAILURE rate. Semantically: only UL blocks sustained
        # for >80ms during T310 qualify as bidirectional signaling fail.
        SIB_TICK_THRESHOLD = 10
        # `preamble_tx_max` is the actual config attribute on UEStateMachineConfig
        # (TS 38.321: maximum preamble transmissions; default 10).
        _preamble_max = int(getattr(self.config, "preamble_tx_max", 10))
        rach_exhausted = (
            self.state.pending_context.get("rach_state") == "exhausted"
            or self.state.pending_context.get("rach_attempt_count", 0) >= _preamble_max
        )
        if self.state.timers["T304"].expired or self.state.ho_in_progress:
            if rach_exhausted:
                ctx["rlf_cause"] = "RACH_PROBLEM"
                self._logger.info(
                    f"[{current_time:.2f}s] HOF pre-classification: RACH_PROBLEM "
                    f"(HO {self.state.serving_cell_id}->{self.state.target_cell_id}, "
                    f"attempts={self.state.pending_context.get('rach_attempt_count', 0)}"
                    f"/{_preamble_max})")
            else:
                ctx["rlf_cause"] = "T304_EXPIRE"
                self._logger.info(
                    f"[{current_time:.2f}s] HOF pre-classification: T304_EXPIRY "
                    f"(HO {self.state.serving_cell_id}->{self.state.target_cell_id})")
        elif getattr(self.state, "sib_block_ticks_during_t310", 0) >= SIB_TICK_THRESHOLD:
            # Bidirectional signaling failure during T310 → matches field
            # log's SIB_READ_FAILURE class (single event per failure).
            ctx["rlf_cause"] = "SIB_READ_FAILURE"
            self._logger.info(
                f"[{current_time:.2f}s] RLF cause: SIB_READ_FAILURE "
                f"(sustained report block ticks="
                f"{self.state.sib_block_ticks_during_t310} ≥ {SIB_TICK_THRESHOLD})")
        elif _t312_hint == "T312_EXPIRY":
            # T312 (fast-RLF) expiry: a report was sent while T310 was running
            # and T312 (the shorter timer) reached its duration first. Recorded
            # as the T312_EXPIRY sub-cause; under RLF_UNIFY it is still a single
            # RLF event (TS 38.331 §5.3.5.5.2 — same outcome as T310 expiry).
            ctx["rlf_cause"] = "T312_EXPIRY"
            self._logger.info(
                f"[{current_time:.2f}s] RLF cause: T312_EXPIRY "
                f"(fast RLF — report sent during T310, T312 fired first)")
        else:
            # Pure RLM RLF — DL Qout countdown won, no dominant UL-block path.
            # Renamed from legacy "T310_EXPIRE" to match field rlf_cause schema.
            ctx["rlf_cause"] = "RLF"
            self._logger.info(
                f"[{current_time:.2f}s] RLF cause: RLF "
                f"(T310 expiry, RLM-driven, sib_ticks="
                f"{getattr(self.state, 'sib_block_ticks_during_t310', 0)})")

    def classify_pending_hof(self):
        """Classify any pending HOF at end of simulation.

        Called when simulation ends with UE still in RLF state.
        If RLF was caused by T304 expiry (HO failure), classify as T304_EXPIRY.
        Otherwise classify as TOO_LATE.
        """
        if not self.state.rlf_declared:
            return

        from .rrc_types import HOFType, HOFClassificationResult

        ctx = self.state.pending_context
        rlf_cause = ctx.get("rlf_cause", "")

        if "T304" in rlf_cause or ctx.get("rlf_ho_was_in_progress", False):
            hof_type = HOFType.T304_EXPIRY
        else:
            hof_type = HOFType.TOO_LATE

        # Only add if not already classified
        if not any(h.hof_type == hof_type for h in self.state.hof_classifications):
            classification = HOFClassificationResult(
                hof_type=hof_type,
                timestamp=ctx.get("rlf_time", 0.0),
                cause=f"End-of-simulation pending RLF ({rlf_cause})",
                rlf_cell_id=self.state.serving_cell_id,
                last_ho_source=ctx.get("last_ho_source", -1),
                last_ho_target=ctx.get("last_ho_target", -1),
                reestablishment_cell_id=-1
            )
            self.state.hof_classifications.append(classification)
            self.state.last_hof_classification = classification
            self._logger.warning(
                f"End-of-simulation HOF classification: {hof_type.value} "
                f"(pending RLF: {rlf_cause})")

    def _classify_hof_on_reestablishment(self, current_time: float,
                                          reest_cell_id: int):
        """
        Final HOF classification when re-establishment completes.

        Implements the 3GPP TS 38.300 §15.5 decision tree:

        1. Was T304 running when RLF occurred?
           → Yes: Case 5 (T304 Expiry) — already pre-classified
           → No: continue to step 2

        2. Was there a recent successful HO (within Tstore_UE_cntxt)?
           → No: Case 1 (Too Late)
           → Yes: Where did UE re-establish?
             - Source cell of last HO → Case 2 (Too Early)
             - Target cell of last HO → (rare, coverage issue)
             - Third cell             → Case 3 (Wrong Cell)

        Args:
            current_time: Current simulation time
            reest_cell_id: Cell where UE re-established
        """
        from .rrc_types import HOFType, HOFClassificationResult

        ctx = self.state.pending_context
        rlf_time = ctx.get("rlf_time", current_time)
        rlf_cell = ctx.get("rlf_cell_id", -1)
        rlf_cause = ctx.get("rlf_cause", "UNKNOWN")

        # Find last successful HO in history
        last_successful_ho = None
        for entry in reversed(self.state.ho_history):
            if entry.success:
                last_successful_ho = entry
                break

        # Default classification
        hof_type = HOFType.NONE
        cause_str = ""

        # ── Case 5: T304 Expiry (already pre-classified) ──
        if rlf_cause == "T304_EXPIRE":
            hof_type = HOFType.T304_EXPIRY
            last_ho_src = ctx.get("rlf_cell_id", -1)
            last_ho_tgt = ctx.get("rlf_target_cell_id", -1)
            time_since_ho_ms = (
                (rlf_time - last_successful_ho.timestamp) * 1000.0
                if last_successful_ho else 0.0
            )
            cause_str = (f"T304 expired during HO {last_ho_src}->{last_ho_tgt}, "
                         f"re-established at cell {reest_cell_id}")

        # ── Cases 1-3: RLF from T310 expiry ──
        elif last_successful_ho is not None:
            time_since_ho_ms = (rlf_time - last_successful_ho.timestamp) * 1000.0
            last_ho_src = last_successful_ho.source_cell_id
            last_ho_tgt = last_successful_ho.target_cell_id

            if time_since_ho_ms <= self.state.tstore_ue_cntxt_ms:
                # "Short time" after last HO — Case 2 or 3
                if reest_cell_id == last_ho_src:
                    # Re-established back at source → Too Early
                    hof_type = HOFType.TOO_EARLY
                    cause_str = (
                        f"RLF {time_since_ho_ms:.0f}ms after HO "
                        f"{last_ho_src}->{last_ho_tgt}, "
                        f"re-estab back to source {reest_cell_id}")
                elif reest_cell_id == last_ho_tgt:
                    # Re-established at target (rare, coverage recovered)
                    hof_type = HOFType.NONE
                    cause_str = (
                        f"RLF {time_since_ho_ms:.0f}ms after HO but "
                        f"re-estab at same target {reest_cell_id} (recovered)")
                else:
                    # Re-established at third cell → Wrong Cell
                    hof_type = HOFType.WRONG_CELL
                    cause_str = (
                        f"RLF {time_since_ho_ms:.0f}ms after HO "
                        f"{last_ho_src}->{last_ho_tgt}, "
                        f"re-estab at 3rd cell {reest_cell_id}")
            else:
                # Long time after last HO → Too Late
                hof_type = HOFType.TOO_LATE
                cause_str = (
                    f"RLF {time_since_ho_ms:.0f}ms after last HO "
                    f"(>{self.state.tstore_ue_cntxt_ms:.0f}ms threshold), "
                    f"re-estab at cell {reest_cell_id}")
        else:
            # No HO history at all → Too Late (never handed over)
            hof_type = HOFType.TOO_LATE
            time_since_ho_ms = -1
            last_ho_src = -1
            last_ho_tgt = -1
            cause_str = (
                f"RLF with no prior HO, "
                f"re-estab at cell {reest_cell_id}")

        # Build classification result
        result = HOFClassificationResult(
            hof_type=hof_type,
            timestamp=current_time,
            rlf_cell_id=rlf_cell,
            reestablishment_cell_id=reest_cell_id,
            last_ho_source=last_ho_src if last_successful_ho else -1,
            last_ho_target=last_ho_tgt if last_successful_ho else -1,
            time_since_last_ho_ms=(
                time_since_ho_ms if last_successful_ho else -1),
            cause=cause_str
        )

        # Store result
        self.state.hof_classifications.append(result)
        self.state.last_hof_classification = result

        # Log
        self._logger.warning(
            f"[{current_time:.2f}s] *** HOF CLASSIFIED: {hof_type.value} *** "
            f"| {cause_str}")

        # Clean up RLF context
        for k in ["rlf_time", "rlf_cell_id", "rlf_cause",
                   "rlf_ho_was_in_progress", "rlf_target_cell_id"]:
            ctx.pop(k, None)

        return result

    def _check_ping_pong(self, current_time: float, source: int, target: int):
        """
        Check for ping-pong handover (Case 4).

        Ping-pong: two successive successful HOs A→B then B→A
        within Tpp window.  Does NOT require RLF — both HOs succeed.

        Args:
            current_time: Time of the just-completed HO
            source: Source cell of current HO
            target: Target cell of current HO (now serving)
        """
        from .rrc_types import HOFType, HOFClassificationResult

        tpp_s = self.state.tpp_ms / 1000.0

        # Look for previous HO: target→source (i.e., reverse of current)
        for entry in reversed(self.state.ho_history):
            if not entry.success:
                continue
            time_gap = current_time - entry.timestamp
            if time_gap > tpp_s:
                break  # Too old, stop searching

            # Check if this was the reverse HO (B→A before current A→B...
            # actually current is source→target, previous should be target→source)
            if (entry.source_cell_id == target and
                    entry.target_cell_id == source):
                # Ping-pong detected!
                result = HOFClassificationResult(
                    hof_type=HOFType.PING_PONG,
                    timestamp=current_time,
                    rlf_cell_id=-1,  # No RLF in ping-pong
                    reestablishment_cell_id=-1,
                    last_ho_source=source,
                    last_ho_target=target,
                    time_since_last_ho_ms=time_gap * 1000.0,
                    cause=(
                        f"Ping-pong: {entry.source_cell_id}->{entry.target_cell_id} "
                        f"then {source}->{target} within "
                        f"{time_gap*1000:.0f}ms (Tpp={self.state.tpp_ms:.0f}ms)")
                )
                self.state.hof_classifications.append(result)
                self.state.last_hof_classification = result

                self._logger.warning(
                    f"[{current_time:.2f}s] *** HOF CLASSIFIED: PING_PONG *** "
                    f"| {result.cause}")
                return result

        return None
