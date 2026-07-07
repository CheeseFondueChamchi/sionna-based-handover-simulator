"""
3GPP Measurement Events for NR Handover

Implements measurement event evaluation according to 3GPP TS 38.331 Section 5.5.4.

Event Types:
- A1: Serving becomes better than threshold
- A2: Serving becomes worse than threshold  
- A3: Neighbour becomes offset better than serving
- A4: Neighbour becomes better than threshold
- A5: Serving becomes worse than threshold1 AND neighbour becomes better than threshold2

Each event has:
- Entering condition (when event is triggered)
- Leaving condition (when event is cancelled)
- Time-to-Trigger (TTT) that must be satisfied before reporting
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import math
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MeasEventType(Enum):
    """Measurement event types from 3GPP TS 38.331"""
    A1 = "a1"  # Serving > threshold
    A2 = "a2"  # Serving < threshold
    A3 = "a3"  # Neighbor > Serving + offset
    A4 = "a4"  # Neighbor > threshold
    A5 = "a5"  # Serving < th1 AND Neighbor > th2


class MeasQuantity(Enum):
    """Measurement quantities"""
    RSRP = "rsrp"  # Reference Signal Received Power (dBm)
    RSRQ = "rsrq"  # Reference Signal Received Quality (dB)
    SINR = "sinr"  # Signal to Interference plus Noise Ratio (dB)


@dataclass
class MeasConfig:
    """
    Measurement Configuration from 3GPP TS 38.331.
    
    This represents MeasConfig IE sent in RRCReconfiguration.
    """
    # Event type
    event_type: MeasEventType = MeasEventType.A3
    
    # Measurement quantity
    quantity: MeasQuantity = MeasQuantity.RSRP

    # Thresholds (dBm for RSRP, dB for RSRQ/SINR)
    threshold1: float = -140.0  # A1/A2 threshold, A5 serving threshold (per-RE RSRP)
    threshold2: float = -140.0  # A5 neighbor threshold (per-RE RSRP)
    
    # A3 specific
    a3_offset: float = 3.0  # dB, range: [-30, +30], step 0.5dB
    
    # Common parameters
    hysteresis: float = 2.0  # dB, range: [0, 30], step 0.5dB
    time_to_trigger_ms: float = 256  # ms
    
    # Layer 3 filtering (k parameter from 3GPP TS 38.331)
    # F_n = (1 - a) * F_{n-1} + a * M_n, where a = 1/2^(k/4)
    filter_coefficient: int = 4
    
    # Reporting configuration
    report_interval_ms: float = 480  # ms
    report_amount: int = 1  # Number of reports (1 = single shot)
    max_report_cells: int = 8  # Maximum cells in report
    
    # Measurement ID
    meas_id: int = 1
    
    def __post_init__(self):
        """Validate configuration against 3GPP spec"""
        # Validate TTT
        valid_ttt = [0, 40, 64, 80, 100, 128, 160, 256, 
                   320, 480, 512, 640, 1024, 1280, 2560, 5120]
        if self.time_to_trigger_ms not in valid_ttt:
            logger.warning(f"TTT {self.time_to_trigger_ms}ms not in 3GPP valid set")
        
        # Validate A3 offset
        if not -30 <= self.a3_offset <= 30:
            raise ValueError(f"A3 offset must be in [-30, 30] dB, got {self.a3_offset}")
        
        # Validate hysteresis
        if not 0 <= self.hysteresis <= 30:
            raise ValueError(f"Hysteresis must be in [0, 30] dB, got {self.hysteresis}")
        
        # Validate filter coefficient
        if not 0 <= self.filter_coefficient <= 20:
            raise ValueError(f"Filter coefficient must be in [0, 20], got {self.filter_coefficient}")
    
    @property
    def time_to_trigger_s(self) -> float:
        """TTT in seconds"""
        return self.time_to_trigger_ms / 1000.0


@dataclass
class MeasResult:
    """
    Measurement result for a single cell.
    
    Represents the filtered measurement after Layer 3 filtering.
    """
    cell_id: int
    pci: int = 0  # Physical Cell ID
    
    # Measurements (after L3 filtering)
    rsrp_dbm: float = -140.0
    rsrq_db: float = -20.0
    sinr_db: float = -10.0
    
    # Raw measurements (before filtering)
    rsrp_raw: float = -140.0
    rsrq_raw: float = -20.0
    sinr_raw: float = -10.0
    
    # Timing
    timestamp: float = 0.0
    
    def get_quantity(self, quantity: MeasQuantity) -> float:
        """Get measurement value for specified quantity"""
        if quantity == MeasQuantity.RSRP:
            return self.rsrp_dbm
        elif quantity == MeasQuantity.RSRQ:
            return self.rsrq_db
        elif quantity == MeasQuantity.SINR:
            return self.sinr_db
        return self.rsrp_dbm


class Layer3Filter:
    """
    Layer 3 measurement filtering from 3GPP TS 38.331.
    
    IIR filter: F_n = (1 - a) * F_{n-1} + a * M_n
    where a = 1/2^(k/4), k = filterCoefficient
    
    - k=0: No filtering (a=1)
    - k=4: a = 0.5 (moderate filtering)
    - k=8: a = 0.25 (heavy filtering)
    """
    
    def __init__(self, filter_coefficient: int = 4):
        """Initialise an L3 IIR filter (legacy 40ms-stepped variant).

        Args:
            filter_coefficient: k value per TS 38.331 §5.5.3.2. Sets the
                IIR coefficient `a = 1/2^(k/4)`. Larger k → heavier
                smoothing. Default k=4 → a=0.5 (one-sample memory).

        Side effects:
            Allocates an empty `_filtered` dict; per-cell state is created
            lazily on first sample.
        """
        self.k = filter_coefficient
        self.a = 1.0 / (2 ** (self.k / 4))
        
        # Filtered values per cell
        self._filtered: Dict[int, Dict[str, float]] = {}
        
        logger.debug(f"L3 Filter initialized: k={self.k}, a={self.a:.4f}")
    
    def filter(self, cell_id: int, quantity: str, raw_value: float) -> float:
        """
        Apply L3 filtering to a measurement.
        
        Args:
            cell_id: Cell identifier
            quantity: Measurement quantity name
            raw_value: Raw measurement value
            
        Returns:
            Filtered measurement value
        """
        key = f"{cell_id}_{quantity}"
        
        if cell_id not in self._filtered:
            self._filtered[cell_id] = {}
        
        if quantity not in self._filtered[cell_id]:
            # First measurement - no filtering
            self._filtered[cell_id][quantity] = raw_value
        else:
            # Apply IIR filter
            prev = self._filtered[cell_id][quantity]
            self._filtered[cell_id][quantity] = (1 - self.a) * prev + self.a * raw_value
        
        return self._filtered[cell_id][quantity]
    
    def reset(self, cell_id: Optional[int] = None):
        """Reset filter state"""
        if cell_id is None:
            self._filtered.clear()
        elif cell_id in self._filtered:
            del self._filtered[cell_id]


class MeasurementEventEvaluator:
    """
    Measurement Event Evaluator from 3GPP TS 38.331 Section 5.5.4.
    
    Evaluates entering and leaving conditions for measurement events.
    
    Entering conditions must be met for the duration of TTT before
    a measurement report is triggered.
    
    Key equations (using RSRP as example):
    - A1 entering: Ms - Hys > Thresh
    - A1 leaving:  Ms + Hys < Thresh
    - A2 entering: Ms + Hys < Thresh
    - A2 leaving:  Ms - Hys > Thresh
    - A3 entering: Mn - Hys > Ms + Ofs
    - A3 leaving:  Mn + Hys < Ms + Ofs
    - A4 entering: Mn - Hys > Thresh
    - A4 leaving:  Mn + Hys < Thresh
    - A5 entering: Ms + Hys < Thresh1 AND Mn - Hys > Thresh2
    - A5 leaving:  Ms - Hys > Thresh1 OR Mn + Hys < Thresh2
    
    Where:
    - Ms = Serving cell measurement
    - Mn = Neighbor cell measurement
    - Hys = Hysteresis
    - Ofs = Offset (A3 only)
    - Thresh = Threshold
    """
    
    def __init__(self, config: MeasConfig):
        """Bind a legacy `MeasConfig` and validate 3GPP-spec field values.

        Args:
            config: `MeasConfig` dataclass with A2/A3/A5 thresholds, TTT,
                hysteresis, filterCoefficient. Used by legacy
                `UERRCController`; the active `UEStateMachine` uses
                `MeasConfig2` + `MeasurementEngine` instead.

        Side effects:
            Calls `_validate_config()` which raises ValueError if any
            field is outside the 3GPP enum (e.g., TTT not in valid set).
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration"""
        # TTT validation with all 3GPP valid values
        valid_ttt_ms = [0, 40, 64, 80, 100, 128, 160, 256, 
                       320, 480, 512, 640, 1024, 1280, 2560, 5120]
        if self.config.time_to_trigger_ms not in valid_ttt_ms:
            logger.warning(f"TTT {self.config.time_to_trigger_ms}ms "
                         f"not in 3GPP valid set: {valid_ttt_ms}")
    
    def evaluate_entering(self, serving: MeasResult,
                         neighbor: Optional[MeasResult] = None) -> bool:
        """
        Evaluate if entering condition is met.
        
        Args:
            serving: Serving cell measurement
            neighbor: Neighbor cell measurement (required for A3/A4/A5)
            
        Returns:
            True if entering condition is satisfied
        """
        ms = serving.get_quantity(self.config.quantity)
        hys = self.config.hysteresis
        
        if self.config.event_type == MeasEventType.A1:
            # A1: Ms - Hys > Thresh
            return ms - hys > self.config.threshold1
        
        elif self.config.event_type == MeasEventType.A2:
            # A2: Ms + Hys < Thresh
            return ms + hys < self.config.threshold1
        
        elif self.config.event_type == MeasEventType.A3:
            if neighbor is None:
                return False
            mn = neighbor.get_quantity(self.config.quantity)
            ofs = self.config.a3_offset
            # A3: Mn - Hys > Ms + Ofs
            return mn - hys > ms + ofs
        
        elif self.config.event_type == MeasEventType.A4:
            if neighbor is None:
                return False
            mn = neighbor.get_quantity(self.config.quantity)
            # A4: Mn - Hys > Thresh
            return mn - hys > self.config.threshold1
        
        elif self.config.event_type == MeasEventType.A5:
            if neighbor is None:
                return False
            mn = neighbor.get_quantity(self.config.quantity)
            # A5: Ms + Hys < Thresh1 AND Mn - Hys > Thresh2
            cond1 = ms + hys < self.config.threshold1
            cond2 = mn - hys > self.config.threshold2
            return cond1 and cond2
        
        return False
    
    def evaluate_leaving(self, serving: MeasResult,
                        neighbor: Optional[MeasResult] = None) -> bool:
        """
        Evaluate if leaving condition is met.
        
        Args:
            serving: Serving cell measurement
            neighbor: Neighbor cell measurement
            
        Returns:
            True if leaving condition is satisfied
        """
        ms = serving.get_quantity(self.config.quantity)
        hys = self.config.hysteresis
        
        if self.config.event_type == MeasEventType.A1:
            # A1 leaving: Ms + Hys < Thresh
            return ms + hys < self.config.threshold1
        
        elif self.config.event_type == MeasEventType.A2:
            # A2 leaving: Ms - Hys > Thresh
            return ms - hys > self.config.threshold1
        
        elif self.config.event_type == MeasEventType.A3:
            if neighbor is None:
                return True  # No neighbor = leave
            mn = neighbor.get_quantity(self.config.quantity)
            ofs = self.config.a3_offset
            # A3 leaving: Mn + Hys < Ms + Ofs
            return mn + hys < ms + ofs
        
        elif self.config.event_type == MeasEventType.A4:
            if neighbor is None:
                return True
            mn = neighbor.get_quantity(self.config.quantity)
            # A4 leaving: Mn + Hys < Thresh
            return mn + hys < self.config.threshold1
        
        elif self.config.event_type == MeasEventType.A5:
            if neighbor is None:
                return True
            mn = neighbor.get_quantity(self.config.quantity)
            # A5 leaving: Ms - Hys > Thresh1 OR Mn + Hys < Thresh2
            cond1 = ms - hys > self.config.threshold1
            cond2 = mn + hys < self.config.threshold2
            return cond1 or cond2
        
        return True
    
    def find_triggering_cells(self, serving: MeasResult,
                              neighbors: List[MeasResult]) -> List[MeasResult]:
        """
        Find all neighbor cells that trigger the event.
        
        For A3/A4/A5, multiple neighbors may satisfy the condition.
        Returns cells sorted by measurement value (best first).
        
        Args:
            serving: Serving cell measurement
            neighbors: List of neighbor measurements
            
        Returns:
            List of triggering neighbor cells (sorted by quality)
        """
        triggering = []
        
        for neighbor in neighbors:
            if self.evaluate_entering(serving, neighbor):
                triggering.append(neighbor)
        
        # Sort by measurement quantity (best first)
        triggering.sort(
            key=lambda x: x.get_quantity(self.config.quantity),
            reverse=True
        )
        
        # Limit to max_report_cells
        return triggering[:self.config.max_report_cells]


@dataclass
class TTTState:
    """Time-to-Trigger state for a cell"""
    cell_id: int
    start_time: float
    entering_met: bool = True
    timer_event: Optional[object] = None  # Reference to scheduled event


class MeasurementManager:
    """
    Manages measurements, filtering, and event evaluation for a UE.
    
    Responsibilities:
    1. Apply L3 filtering to raw measurements
    2. Evaluate measurement events (A1-A5)
    3. Manage TTT timers
    4. Generate measurement reports
    """
    
    def __init__(self, 
                 meas_configs: List[MeasConfig],
                 scheduler: 'EventScheduler'):
        """
        Initialize measurement manager.
        
        Args:
            meas_configs: List of measurement configurations
            scheduler: Event scheduler for TTT timers
        """
        self.scheduler = scheduler
        
        # Measurement configurations and evaluators
        self.configs: Dict[int, MeasConfig] = {}
        self.evaluators: Dict[int, MeasurementEventEvaluator] = {}
        
        for config in meas_configs:
            self.configs[config.meas_id] = config
            self.evaluators[config.meas_id] = MeasurementEventEvaluator(config)
        
        # L3 filters per measurement config
        self.filters: Dict[int, Layer3Filter] = {}
        for meas_id, config in self.configs.items():
            self.filters[meas_id] = Layer3Filter(config.filter_coefficient)
        
        # TTT states: {meas_id: {cell_id: TTTState}}
        self.ttt_states: Dict[int, Dict[int, TTTState]] = {
            meas_id: {} for meas_id in self.configs
        }
        
        # Pending measurement events (cells that passed TTT)
        self.pending_reports: Dict[int, List[int]] = {
            meas_id: [] for meas_id in self.configs
        }

        # 3GPP L1 measurement period (SSB-based intra-freq RSRP, no DRX,
        # FR1) — TS 38.133 §9.2. The L1 layer reports one filtered RSRP/
        # RSRQ/SINR per period; L3 filter (above) operates on those L1
        # outputs. Gating per-call evaluation by elapsed time makes A3
        # trigger behaviour invariant to the simulator step size.
        # L1 measurement period (TS 38.133 §9.2.5). Default 20ms (per-SSB
        # cadence; finer sample-and-hold so a3_serv_filt tracks the field RSRP/
        # SINR more tightly). env MEAS_EVAL_PERIOD_MS overrides for experiments
        # (e.g. 200ms = the prior FR1/no-DRX baseline period).
        self._meas_eval_period_ms: float = float(os.environ.get("MEAS_EVAL_PERIOD_MS", "20"))
        self._last_meas_eval_s: Dict[int, float] = {}

        # Callbacks
        self.on_measurement_report: Optional[callable] = None

        logger.info(f"MeasurementManager initialized with {len(meas_configs)} configs")
    
    def process_measurements(self, 
                            serving_raw: MeasResult,
                            neighbors_raw: List[MeasResult],
                            current_time: float) -> List[Tuple[int, List[int]]]:
        """
        Process raw measurements and evaluate events.
        
        Args:
            serving_raw: Raw serving cell measurement
            neighbors_raw: Raw neighbor cell measurements
            current_time: Current simulation time
            
        Returns:
            List of (meas_id, [triggered_cell_ids]) for cells that passed TTT
        """
        triggered_reports = []
        
        for meas_id, config in self.configs.items():
            l3_filter = self.filters[meas_id]
            evaluator = self.evaluators[meas_id]
            
            # Apply L3 filtering to serving cell
            serving = self._apply_filter(serving_raw, l3_filter, current_time)
            
            # Apply L3 filtering to neighbors
            neighbors = [
                self._apply_filter(n, l3_filter, current_time) 
                for n in neighbors_raw
            ]
            
            # Find cells that satisfy entering condition
            triggering_cells = evaluator.find_triggering_cells(serving, neighbors)
            
            # Update TTT states
            triggered = self._update_ttt_states(
                meas_id, config, evaluator, serving, 
                neighbors, triggering_cells, current_time
            )
            
            if triggered:
                triggered_reports.append((meas_id, triggered))
        
        return triggered_reports

    def process_measurements_single(self, meas_id: int,
                                     serving_raw: MeasResult,
                                     neighbors_raw: List[MeasResult],
                                     current_time: float,
                                     freeze_cancel: bool = False) -> List[Tuple[int, List[int]]]:
        """Process measurements for a single config with pre-filtered neighbors.

        Used by frequency-aware event evaluation where neighbors are pre-filtered
        by frequency group (intra-freq for A3, inter-freq for A5, etc.).

        Args:
            meas_id: Measurement configuration ID
            serving_raw: Raw serving cell measurement
            neighbors_raw: Pre-filtered neighbor measurements for this config
            current_time: Current simulation time
            freeze_cancel: If True, don't cancel existing TTTs (OOS/Gray)

        Returns:
            List of (meas_id, [triggered_cell_ids]) for cells that passed TTT
        """
        config = self.configs.get(meas_id)
        if config is None:
            return []

        # 3GPP L1 measurement-period gate: throttle filter+A3 evaluation
        # to one update per `_meas_eval_period_ms` so behaviour is
        # invariant to simulator step (TS 38.133 §9.2 / TS 38.331
        # §5.5.3.2 — L3 filter operates on L1 measurements, not on every
        # PHY tick).
        last_eval = self._last_meas_eval_s.get(meas_id)
        if last_eval is not None and \
                (current_time - last_eval) * 1000.0 < self._meas_eval_period_ms:
            return []
        self._last_meas_eval_s[meas_id] = current_time

        l3_filter = self.filters[meas_id]
        evaluator = self.evaluators[meas_id]

        serving = self._apply_filter(serving_raw, l3_filter, current_time)
        neighbors = [self._apply_filter(n, l3_filter, current_time) for n in neighbors_raw]

        triggering_cells = evaluator.find_triggering_cells(serving, neighbors)
        triggered = self._update_ttt_states(
            meas_id, config, evaluator, serving,
            neighbors, triggering_cells, current_time,
            freeze_cancel=freeze_cancel
        )
        if triggered:
            return [(meas_id, triggered)]
        return []

    def _apply_filter(self, raw: MeasResult, l3_filter: Layer3Filter,
                      timestamp: float) -> MeasResult:
        """Apply L3 filtering to a measurement result"""
        return MeasResult(
            cell_id=raw.cell_id,
            pci=raw.pci,
            rsrp_dbm=l3_filter.filter(raw.cell_id, 'rsrp', raw.rsrp_dbm),
            rsrq_db=l3_filter.filter(raw.cell_id, 'rsrq', raw.rsrq_db),
            sinr_db=l3_filter.filter(raw.cell_id, 'sinr', raw.sinr_db),
            rsrp_raw=raw.rsrp_dbm,
            rsrq_raw=raw.rsrq_db,
            sinr_raw=raw.sinr_db,
            timestamp=timestamp
        )
    
    def _update_ttt_states(self, meas_id: int, config: MeasConfig,
                          evaluator: MeasurementEventEvaluator,
                          serving: MeasResult, neighbors: List[MeasResult],
                          triggering_cells: List[MeasResult],
                          current_time: float,
                          freeze_cancel: bool = False) -> List[int]:
        """
        Update TTT states and return cells that completed TTT.

        Args:
            meas_id: Measurement configuration ID
            config: Measurement configuration
            evaluator: Event evaluator
            serving: Filtered serving measurement
            neighbors: Filtered neighbor measurements
            triggering_cells: Cells satisfying entering condition
            current_time: Current time
            freeze_cancel: If True, do NOT cancel existing TTTs even if
                entering condition is no longer met (used during OOS/Gray
                per 3GPP TS 38.133 §8.1 — unreliable measurements should
                not reset accumulated TTT progress)

        Returns:
            List of cell IDs that completed TTT
        """
        ttt_states = self.ttt_states[meas_id]
        triggered_cells = []

        triggering_ids = {c.cell_id for c in triggering_cells}

        # Start new TTT timers for cells that just entered
        for cell in triggering_cells:
            if cell.cell_id not in ttt_states:
                # New entering condition - start TTT
                ttt_states[cell.cell_id] = TTTState(
                    cell_id=cell.cell_id,
                    start_time=current_time,
                    entering_met=True
                )
                logger.debug(f"TTT started for cell {cell.cell_id} "
                           f"(meas_id={meas_id}, TTT={config.time_to_trigger_ms}ms)")

        # Check existing TTT states
        cells_to_remove = []
        for cell_id, state in ttt_states.items():
            if cell_id not in triggering_ids:
                if not freeze_cancel:
                    # IN_SYNC: entering condition no longer met - cancel TTT
                    cells_to_remove.append(cell_id)
                    logger.debug(f"TTT cancelled for cell {cell_id} "
                               f"(entering condition no longer met)")
                # else: OOS/Gray — keep TTT alive, don't cancel
            else:
                # Check if TTT has elapsed
                elapsed = current_time - state.start_time
                if elapsed >= config.time_to_trigger_s:
                    triggered_cells.append(cell_id)
                    cells_to_remove.append(cell_id)
                    logger.info(f"TTT expired for cell {cell_id} - triggering report")

        # Remove completed/cancelled TTT states
        for cell_id in cells_to_remove:
            del ttt_states[cell_id]

        return triggered_cells
    
    def get_pending_cells(self, meas_id: int) -> List[int]:
        """Get cells with pending TTT timers"""
        return list(self.ttt_states.get(meas_id, {}).keys())
    
    def reset(self, meas_id: Optional[int] = None):
        """Reset measurement state"""
        if meas_id is None:
            for mid in self.configs:
                self.ttt_states[mid].clear()
                self.filters[mid].reset()
        elif meas_id in self.configs:
            self.ttt_states[meas_id].clear()
            self.filters[meas_id].reset()


# ============================================================================
# NEW: Time-Continuous State Machine Support
# ============================================================================

@dataclass
class MeasConfig2:
    """Configuration for MeasurementEngine (new polling-based API)"""
    # A3 (Intra-Freq HO)
    a3_offset: float = 3.0

    # A2 (Start Inter-Freq Meas) - must be >= A5 thresh1 so A2 fires first
    a2_threshold: float = -118.0

    # A5 (Inter-Freq HO)
    a5_threshold1: float = -125.0  # Serving Threshold (per-RE RSRP)
    a5_threshold2: float = -115.0   # Neighbor Threshold (per-RE RSRP)

    hysteresis: float = 2.0
    a2_hysteresis: float = 3.0  # A2-specific hysteresis (3GPP TS 38.331 §5.5.4.2)
    time_to_trigger_ms: float = 256.0  # Default TTT (used for A3)
    a2_ttt_ms: float = 0.0   # 0 = use time_to_trigger_ms
    a5_ttt_ms: float = 0.0   # 0 = use time_to_trigger_ms
    # reportInterval (TS 38.331 §5.5.5). LAST-RESORT FALLBACK ONLY — overridden
    # per-cell from the operational enb CSV (240/120) via _g(gnb,...) in
    # _build_cell_configs. Verified runtime gate = 240ms (CSV), not 480.
    a3_report_interval_ms: float = 480.0
    a2_report_interval_ms: float = 1024.0
    filter_coefficient: int = 4

    # B1 (Inter-RAT single threshold) - 3GPP TS 38.331 §5.5.4.7
    # Entering: Mn + Ofn - Hys > Thresh
    b1_threshold: float = -125.0
    b1_ttt_ms: float = 256.0
    b1_offset: float = 0.0            # Ofn (frequency offset)

    # B2 (Inter-RAT dual threshold) - 3GPP TS 38.331 §5.5.4.8
    # Entering: (Ms + Hys < Thresh1) AND (Mn + Ofn - Hys > Thresh2)
    b2_threshold1: float = -130.0     # Serving threshold (per-RE RSRP)
    b2_threshold2: float = -125.0      # Inter-RAT neighbor threshold (per-RE RSRP)
    b2_ttt_ms: float = 256.0
    b2_offset: float = 0.0

    # CIO Ocn (3GPP TS 38.331 §6.3.2 cellIndividualOffsets) — per-(serving, neighbor) PCI
    # offset added to neighbor measurement in A3 entering condition. Empty dict ⇒ Ocn=0
    # for all neighbors ⇒ byte-identical to pre-CIO behavior. Populated from per-cell
    # GnbSectorInfo.cio_table at config swap time (apply_config).
    cio_table: dict = field(default_factory=dict)

    # L3 filter passthrough (CSV channel mode).
    # ⚠️ 2026-06-09 (ops-confirmed): the wide PCI RSRP/RSRQ CSV is NOT
    # L3-filtered — it is RAW L1 from the DM-tool log. The "already
    # field-L3-filtered → skip in-sim L3 to avoid double-smoothing" premise
    # below is WRONG. The operational UE applies fc4 (filterCoefficient k=4,
    # α=0.5) before A3/RLM and that output is absent from the log, so the
    # faithful config is passthrough=False + filter_coef=4 (apply fc4 in-sim),
    # NOT passthrough. Kept as a toggle; True returns the raw input dict while
    # the 200ms cadence gating
    # is preserved (TS 38.133 §9.2 L1 measurement period). HO logic and the
    # A3/A5/B1/B2 entering/leaving math are unchanged. Default False keeps
    # legacy statistical / Sionna-RT behavior.
    l3_filter_passthrough: bool = False

    # Vendor gNB MRO (Mobility Robustness Optimization) — post-HO blacklist.
    # After HO source→target, the (target, source) pair is blacklisted for
    # `post_ho_blacklist_s` seconds: while serving=target, the A3 entering
    # condition for `source` is ignored. Prevents immediate ping-pong back
    # to the cell we just left. NOT in 3GPP UE FSM (TS 38.331); this is
    # gNB implementation behavior (TS 38.473 / TS 28.541 NRM). Default 0.0
    # disables (legacy byte-identical).
    post_ho_blacklist_s: float = 0.0

    # S-Measure (3GPP TS 38.331 §5.5.4 measObjectNR / s-Measure equivalent).
    # When set, the UE only evaluates A3 entering conditions while the
    # serving cell's RSRP <= ``s_measure_dbm``. Above this threshold, A3
    # neighbor measurement is suppressed (no TTT accumulation, no triggers).
    # Used by HSR (high-speed-rail) cells to suppress unnecessary HOs while
    # serving signal is strong (-97 dBm typical). None = disabled.
    s_measure_dbm: Optional[float] = None


class MeasurementEngine:
    """
    Polling-based Measurement Engine for Time-Continuous State Machine.

    Unlike MeasurementManager (callback-based), this class:
    - Uses evaluate() to return EventState objects each timestep
    - Maintains internal state for L3 filtering and TTT tracking
    - Returns measurement event results instead of triggering callbacks

    3GPP Reference: TS 38.331 §5.5.4 (Measurement Events)
    """

    def __init__(self, config: MeasConfig2):
        """Construct the active-FSM measurement engine.

        Args:
            config: `MeasConfig2` dataclass with per-event thresholds, TTT,
                hysteresis, filter coefficient, CIO table, S-Measure, and
                CSV-mode `l3_filter_passthrough` flag.

        Side effects:
            Allocates all per-engine state: TTT trackers per event
            (A2/A3/A5/B1/B2), A3 grace-tick counters, L3 filter caches
            (`filtered_rsrp`, `filtered_rsrq`), per-cell quantity selector
            (RSRP vs RSRQ), `_last_report_time` reportInterval ledger,
            candidate freshness ages, and post-HO blacklist dict.
            Computes IIR coefficient `alpha = 0.5 ** (k/4)`.
        """
        self.config = config

        # TTT Trackers: {event_type: {key: time_accumulated}}
        self.ttt_trackers: Dict[str, Dict[str, float]] = {
            "A2": {}, "A3": {}, "A5": {}, "B1": {}, "B2": {}
        }

        # A3 one-tick grace counters per-cell (TS 38.331 §5.5.4 leaving
        # hysteresis). When an A3 candidate misses the entering condition
        # but already accumulated >=50% of TTT, grant one tick of grace
        # before deleting its tracker. {key_str: remaining_grace_ticks}.
        self._a3_grace_ticks: Dict[str, int] = {}

        # Per 3GPP: A2 must fire before A5 is evaluated (inter-freq gating)
        self.a2_gate_active: bool = False

        # Report interval tracking: {(event_type, cell_id): last_report_timestamp}
        # 3GPP TS 38.331 §5.5.5 — reportInterval: minimum time between CONSECUTIVE reports
        # for the SAME triggered cell. Keyed per-cell so first reports for newly-triggered
        # cells are never suppressed by stale timestamps from other cells.
        self._last_report_time: Dict[Tuple[str, Optional[int]], float] = {}
        # Observability-only read-out: why the latest A3 report was withheld
        # (currently only "REPORT_INTERVAL") and the gap (ms) since the last A3
        # report at that tick. Surfaced via a3_diagnostics_snapshot. Never gates.
        self._last_report_block_reason: str = ""
        self._last_report_gap_ms: Optional[float] = None

        # L3 Filtering State
        self.filtered_rsrp: Dict[int, float] = {}
        # Held last measured serving RSRP — used as Ms when serving cell is
        # absent from the current tick's measurements (TS 38.331 §5.5.3.2 L3
        # filter HOLDS its state on missing samples). Replaces the old
        # `measurements.get(serving, -140.0)` sentinel that drove phantom HOs.
        self._last_serving_rsrp: Optional[float] = None
        # RSRQ filter state — parallel to filtered_rsrp. Populated only when
        # filter_l3() is called with rsrq_measurements (RSRQ-A3 mode). When
        # not provided, this stays empty and behavior is byte-identical to
        # pre-RSRQ baseline.
        self.filtered_rsrq: Dict[int, float] = {}
        # Per-cell A3 measurement quantity selector. Default empty ⇒ all cells
        # use RSRP (legacy). Populated by caller via set_quantity_for_a3() to
        # mark non-HSR cells as "rsrq" when --use-rsrq-for-regular-cells is on.
        self.quantity_for_a3: Dict[int, str] = {}
        self.alpha = 0.5 ** (self.config.filter_coefficient / 4)

        # 3GPP L1 measurement-period gate (TS 38.133 §9.2 — SSB-based
        # intra-freq RSRP, no DRX, FR1). The L3 filter operates on L1
        # measurement outputs that arrive once per period, NOT per PHY
        # tick. Gating filter_l3() makes the smoothing time-constant
        # invariant to the simulator step size.
        # L1/L3 measurement period (TS 38.133 §9.2.5 / TS 38.331 §5.5.3.2) —
        # the cadence of the entering-DETECTION (pre-TTT) phase (see filter_l3).
        # Default 20ms: a 36-case sweep showed a 50ms pre-TTT period improves
        # serving-PCI tracking (94.8→95.6%) but costs RLF-timing F1 (63.7→56.5),
        # so 20ms is retained (user decision 2026-06-12). Once a TTT runs the
        # filter switches to every-tick fc4 regardless of this period (the
        # spec continuous-satisfaction phase). env MEAS_EVAL_PERIOD_MS overrides
        # (50ms = PCI-leaning; 200ms = the prior FR1/no-DRX baseline).
        self._meas_eval_period_ms: float = float(os.environ.get("MEAS_EVAL_PERIOD_MS", "20"))
        self._last_filter_time_s: Optional[float] = None

        # Per-cell A3 candidate age counter (consecutive evaluate() ticks
        # that the neighbor has appeared with a finite RSRP). Used as a
        # freshness gate: a phantom or just-emerged neighbor must persist
        # for >= _A3_CANDIDATE_MIN_AGE ticks before its TTT is allowed to
        # start. Allowed by 3GPP TS 38.331 §5.5.4.4 (UE may apply
        # implementation-specific candidate validation).
        self._candidate_age: Dict[int, int] = {}
        self._A3_CANDIDATE_MIN_AGE: int = 2

        # Vendor gNB MRO post-HO blacklist:
        #   key=(serving_cell_id, banned_neighbor_id), value=expiry_time_s.
        # After HO source→target, (target, source) is added so that while
        # serving=target the A3 entering condition for `source` is ignored
        # until the expiry passes. Standardised in TS 38.473 / TS 28.541
        # NRM as gNB-side MRO (Mobility Robustness Optimization); not part
        # of the UE FSM. Duration is taken from `_blacklist_duration_s`,
        # which is mirrored from MeasConfig2.post_ho_blacklist_s via
        # apply_config(). Empty / 0.0 → byte-identical to pre-MRO baseline.
        self._post_ho_blacklist: Dict[Tuple[int, int], float] = {}
        self._blacklist_duration_s: float = float(
            getattr(config, "post_ho_blacklist_s", 0.0) or 0.0
        )

    def rollback_report_interval(self, event_type: str = "A3") -> None:
        """DEPRECATED / UNUSED (2026-06-12). Un-consume the reportInterval slot
        for a report suppressed before transmission.

        This was the original fix for the UL-block deadlock (a TTT-expired
        rescue HO that the interval gate + a flapping UL block could stall for
        seconds → spurious RLF). It worked by letting the measurement engine
        REGENERATE the report every tick while blocked — which re-fired the A3
        report every 10 ms (884× per fade, violating reportInterval spacing).
        Superseded by the controller's pending-delivery model (UEState
        ul_pending_ho_target): the report is generated once per reportInterval
        and HELD; the delivery layer retries it every tick, delivering the HO
        the instant UL recovers WITHOUT regenerating the report. The controller
        no longer calls this method; kept only for reference / back-compat.
        """
        self._last_report_time.pop((event_type, None), None)

    def reset_for_meas_config_change(self, preserve_report_intervals: bool = False) -> None:
        """Reset cellsTriggeredList + VarMeasReportList state.

        Called when measConfig changes per TS 38.331 §5.3.5.5 (HO complete with
        reconfiguration) and §5.5.2.1 (measConfig update). Trigger points in
        UEStateMachine: HO_COMPLETE, re-establishment success, RRC setup
        (IDLE→CONNECTED), RLF declaration.

        Cleared:
          - `ttt_trackers` — cellsTriggeredList per event (A2/A3/A5/B1/B2)
          - `_last_report_time` — VarMeasReportList per-event reportInterval state
            (UNLESS preserve_report_intervals=True; HO_COMPLETE caller passes True
            because the same measId carries over and reportInterval gating should
            survive HO to prevent ping-pong reports within < reportInterval after
            HO completes, TS 38.331 §5.5.5.1 reportInterval semantics).

        Preserved (intentional):
          - `filtered_rsrp`/`filtered_rsrq` — L3 filter state is per physical
            cell, survives measConfig change (TS 38.331 §5.5.3.2)
          - `quantity_for_a3` — per-cell quantity selector, orthogonal
          - `_post_ho_blacklist` — time-based expiry runs independently
          - `_candidate_age` — A3 candidate freshness gate, neighbor-tied
        """
        for ev in ("A2", "A3", "A5", "B1", "B2"):
            self.ttt_trackers.setdefault(ev, {}).clear()
        if not preserve_report_intervals:
            self._last_report_time.clear()
        # Clear A3 one-tick grace counters — measConfig change invalidates
        # cellsTriggeredList, so any pending grace from the previous config
        # is irrelevant under the new measurement objects.
        if hasattr(self, "_a3_grace_ticks"):
            self._a3_grace_ticks.clear()

    def add_post_ho_blacklist(self, source_cell: int, target_cell: int,
                              current_time: float) -> None:
        """Register a post-HO MRO blacklist entry.

        After a successful HO source_cell→target_cell at `current_time`,
        suppress A3 reports for `source_cell` while serving=`target_cell`
        for the duration `_blacklist_duration_s`. No-op when the duration
        is non-positive.
        """
        if self._blacklist_duration_s > 0.0:
            self._post_ho_blacklist[(int(target_cell), int(source_cell))] = (
                float(current_time) + self._blacklist_duration_s
            )

    def set_quantity_for_a3(self, quantity_map: Optional[Dict[int, str]]) -> None:
        """Replace the per-cell A3 quantity map.

        Args:
            quantity_map: {cell_id: "rsrp"|"rsrq"}; cells not in the map default
                to "rsrp". Pass None to clear (revert to all-RSRP).
        """
        self.quantity_for_a3 = dict(quantity_map) if quantity_map else {}

    def filter_l3(self, raw_measurements: Dict[int, float],
                  rsrq_measurements: Optional[Dict[int, float]] = None,
                  current_time_s: Optional[float] = None
                  ) -> Dict[int, float]:
        """
        Apply L3 Filtering: F_n = (1-a)*F_{n-1} + a*M_n

        Args:
            raw_measurements: {cell_id: raw_rsrp} or {cell_id: (rsrp, sinr)}
            rsrq_measurements: optional {cell_id: raw_rsrq_db}. When provided,
                filtered RSRQ is tracked in self.filtered_rsrq parallel to
                filtered_rsrp. Default None ⇒ no RSRQ tracking (legacy
                byte-identical behavior).
            current_time_s: Current simulator wall-clock in seconds. When
                provided, the L3 filter is updated at most once per
                `_meas_eval_period_ms` (3GPP TS 38.133 §9.2). Between
                L1 measurement points the cached filtered values are
                returned unchanged. When None, legacy per-call behaviour.

        Returns:
            {cell_id: filtered_rsrp}
        """
        # L3-filter / entering-evaluation cadence (2026-06-12, per spec model):
        #   * BEFORE any HO-event TTT starts (entering DETECTION phase): the
        #     L3 filter steps once per `_meas_eval_period_ms` (the L1/L3
        #     measurement period, default 50ms). evaluate() runs every tick but
        #     sees the cached value between steps, so entering detection is
        #     effectively at the 50ms cadence — like the RLM OOS counter.
        #   * ONCE a TTT is running (continuous-satisfaction phase): the filter
        #     steps EVERY tick (no period gate) so the entering condition is
        #     re-checked at full resolution against the fc4-filtered value
        #     (current vs previous, alpha). A real dip resets the TTT promptly.
        # Applies to BOTH the RSRP and RSRQ filters (stepped together below),
        # so HSR (RSRP-A3) and regular (RSRQ-A3) cells are handled uniformly.
        # DEFAULT OFF (2026-06-13 user decision): the every-tick-during-TTT
        # model is spec-realistic but costs RLF-timing fidelity (F1 64.4→62.5,
        # precision 72.5→65.2%); the uniform-cadence default maximises the gate
        # metric. Set MEAS_TTT_CONTINUOUS=1 to re-enable the spec-realistic
        # continuous-satisfaction phase for measurement-model studies.
        _ttt_continuous = os.environ.get("MEAS_TTT_CONTINUOUS", "0") != "0"
        _ttt_active = _ttt_continuous and any(
            self.ttt_trackers.get(_ev) for _ev in ("A3", "A5", "B1", "B2"))
        if current_time_s is not None:
            if not _ttt_active:
                if self._last_filter_time_s is not None and \
                        (current_time_s - self._last_filter_time_s) * 1000.0 < self._meas_eval_period_ms:
                    return self.filtered_rsrp  # cached — no L1 update this tick
            # Stamp every step (period step OR every-tick TTT step) so the
            # 50ms cadence resumes cleanly the tick after a TTT clears.
            self._last_filter_time_s = current_time_s

        # CSV passthrough: return raw inputs, skipping the IIR averaging.
        # ⚠️ The wide CSV is RAW L1 (NOT field-L3-filtered) — see config note
        # above. Passthrough therefore means NO L3 anywhere, which is NOT the
        # operational fc4 path; it is only an approximation. The 200ms cadence
        # gating above is preserved so the eval rhythm matches TS 38.133 §9.2.
        if getattr(self.config, "l3_filter_passthrough", False):
            new_filtered = {}
            for cell_id, raw_val in raw_measurements.items():
                if isinstance(raw_val, (tuple, list)):
                    raw_val = float(raw_val[0])
                else:
                    raw_val = float(raw_val)
                new_filtered[cell_id] = raw_val
            self.filtered_rsrp = new_filtered
            if rsrq_measurements:
                new_rsrq = {}
                for cell_id, raw_q in rsrq_measurements.items():
                    try:
                        new_rsrq[cell_id] = float(raw_q)
                    except (TypeError, ValueError):
                        continue
                self.filtered_rsrq = new_rsrq
            return new_filtered

        new_filtered = {}
        for cell_id, raw_val in raw_measurements.items():
            # Handle tuple format (rsrp, sinr) from channel calculator
            if isinstance(raw_val, (tuple, list)):
                raw_val = float(raw_val[0])  # Extract RSRP
            else:
                raw_val = float(raw_val)

            prev_val = self.filtered_rsrp.get(cell_id, raw_val)
            new_filtered[cell_id] = (1 - self.alpha) * prev_val + self.alpha * raw_val
        self.filtered_rsrp = new_filtered

        # Parallel RSRQ filter — only when caller provides RSRQ measurements.
        if rsrq_measurements:
            new_rsrq = {}
            for cell_id, raw_q in rsrq_measurements.items():
                try:
                    raw_q = float(raw_q)
                except (TypeError, ValueError):
                    continue
                prev_q = self.filtered_rsrq.get(cell_id, raw_q)
                new_rsrq[cell_id] = (1 - self.alpha) * prev_q + self.alpha * raw_q
            self.filtered_rsrq = new_rsrq

        return new_filtered

    def _update_ttt(self, event_type: str, key: str, dt: float,
                    entering_condition: bool,
                    current_time: float = 0.0,
                    serving_cell_id: Optional[int] = None) -> 'EventState':
        """
        Process Time-To-Trigger logic for a specific event.

        Args:
            event_type: "A2", "A3", or "A5"
            key: Identifier for the condition (e.g., cell_id as string)
            dt: Time delta in seconds
            entering_condition: Whether the entering condition is currently met
            current_time: Current simulation time (used for reportInterval suppression)
            serving_cell_id: Current serving cell ID (used for A2 per-cell report key)

        Returns:
            EventState with trigger status and remaining TTT
        """
        from .rrc_types import EventState

        tracker = self.ttt_trackers[event_type]
        state = EventState(event_type=event_type)

        if entering_condition:
            if key not in tracker:
                tracker[key] = 0.0
            tracker[key] += dt

            state.triggered = True

            # Per-event TTT: use event-specific value if set, else default
            if event_type == "A2" and self.config.a2_ttt_ms > 0:
                ttt_ms = self.config.a2_ttt_ms
            elif event_type == "A5" and self.config.a5_ttt_ms > 0:
                ttt_ms = self.config.a5_ttt_ms
            elif event_type == "B1" and self.config.b1_ttt_ms > 0:
                ttt_ms = self.config.b1_ttt_ms
            elif event_type == "B2" and self.config.b2_ttt_ms > 0:
                ttt_ms = self.config.b2_ttt_ms
            else:
                ttt_ms = self.config.time_to_trigger_ms
            ttt_sec = ttt_ms / 1000.0
            if tracker[key] >= ttt_sec:
                # TTT expired — check reportInterval before sending report
                # 3GPP TS 38.331 §5.5.5: suppress consecutive reports within reportInterval
                if event_type == "A2":
                    interval_s = self.config.a2_report_interval_ms / 1000.0
                    report_key: Tuple[str, Optional[int]] = ("A2", serving_cell_id)
                    last_t = self._last_report_time.get(report_key, -9999.0)
                    if (current_time - last_t) < interval_s:
                        # Within reportInterval — suppress report, keep TTT alive
                        state.report_sent = False
                        state.time_to_trigger_remaining = 0.0
                    else:
                        state.report_sent = True
                        state.time_to_trigger_remaining = 0.0
                        self._last_report_time[report_key] = current_time
                else:
                    state.report_sent = True
                    state.time_to_trigger_remaining = 0.0
            else:
                state.report_sent = False
                state.time_to_trigger_remaining = (ttt_sec - tracker[key]) * 1000.0
        else:
            if key in tracker and not self._freeze_cancel:
                del tracker[key]
            state.triggered = key in tracker
            state.report_sent = False

        return state

    def a3_diagnostics_snapshot(
        self,
        serving_cell_id: int,
        current_time_s: Optional[float] = None,
    ) -> Dict[str, object]:
        """Read-only A3 evaluation diagnostics for downstream logging.

        Surfaces the internal state used by ``evaluate()`` so that
        detailed_log_ue*.csv can show what the engine actually compared
        (filtered RSRP/RSRQ, candidate ages, per-cell TTT elapsed, the
        serving-driven A3 metric, post-HO blacklist + s-Measure + freshness
        gating) — not just the per-tick TTT remaining already in meas_events.

        This method MUST NOT mutate any engine state. It mirrors the
        entering-condition arithmetic in ``evaluate()`` (around the
        ``threshold_a3_enter = serving_metric + a3_offset + hys`` block) to
        recompute the entering set from the *currently held* filtered values.

        Args:
            serving_cell_id: Current serving cell (int).
            current_time_s: Wall sim-clock; used only to derive
                ``filter_age_ms = (current - _last_filter_time_s) * 1000``.
                If None or ``_last_filter_time_s`` is None, age is None.

        Returns:
            Dict with keys:
              - serving_cell_id: int
              - serving_filt_rsrq: Optional[float]
              - serving_filt_rsrp: Optional[float]
              - serving_a3_quantity: "rsrp" | "rsrq"
              - filter_age_ms: Optional[float]
              - entering_now: List[int]  (cell_ids meeting entering cond.)
              - candidates: List[dict]    (top-3 by metric DESC; each has
                  cell_id, metric, ttt_elapsed_ms, age, entering,
                  filt_rsrq, filt_rsrp)
              - tracker_n: int           (len(ttt_trackers["A3"]))
        """
        # Mirror the serving-driven metric selection in evaluate() (~L1063).
        serving_a3_quantity: str = self.quantity_for_a3.get(
            int(serving_cell_id), "rsrp"
        )

        serving_filt_rsrq: Optional[float] = self.filtered_rsrq.get(
            int(serving_cell_id)
        ) if self.filtered_rsrq else None
        serving_filt_rsrp: Optional[float] = self.filtered_rsrp.get(
            int(serving_cell_id)
        ) if self.filtered_rsrp else None

        # Determine the serving metric used for the A3 entering threshold.
        # When RSRQ-A3 is active for this serving and we have a filtered
        # RSRQ for the serving cell, use it; otherwise fall back to
        # filtered RSRP (or the held last-serving RSRP). If neither is
        # available, leave serving_metric = None and skip entering-set
        # recomputation (candidates will still surface their own metrics).
        serving_metric: Optional[float] = None
        if serving_a3_quantity == "rsrq" and self.filtered_rsrq and \
                serving_filt_rsrq is not None:
            serving_metric = serving_filt_rsrq
        elif serving_filt_rsrp is not None:
            serving_metric = serving_filt_rsrp
        elif self._last_serving_rsrp is not None:
            serving_metric = float(self._last_serving_rsrp)

        # Filter age (ms) since last L3 filter update.
        filter_age_ms: Optional[float] = None
        if current_time_s is not None and self._last_filter_time_s is not None:
            filter_age_ms = (
                float(current_time_s) - float(self._last_filter_time_s)
            ) * 1000.0

        # S-Measure gate (same rule as evaluate(): if serving RSRP exceeds
        # s_measure_dbm, all A3 evaluation is suppressed).
        s_meas = getattr(self.config, "s_measure_dbm", None)
        s_measure_blocked: bool = False
        if s_meas is not None and self._last_serving_rsrp is not None:
            s_measure_blocked = float(self._last_serving_rsrp) > float(s_meas)

        a3_offset: float = float(self.config.a3_offset)
        hys: float = float(self.config.hysteresis)
        cio_table = self.config.cio_table or {}

        # Build the candidate universe from filtered_rsrp keys (all currently
        # tracked neighbors) — these are the cells the engine "sees".
        candidate_cells: List[int] = []
        for cid in self.filtered_rsrp.keys():
            try:
                cid_i = int(cid)
            except (TypeError, ValueError):
                continue
            if cid_i == int(serving_cell_id):
                continue
            candidate_cells.append(cid_i)

        # Per-cell metric + entering check.
        cand_records: List[Dict[str, object]] = []
        entering_now: List[int] = []

        for cid_i in candidate_cells:
            # Mirror neighbor_metric rule from evaluate() (~L1138-1141).
            if serving_a3_quantity == "rsrq" and self.filtered_rsrq:
                base = self.filtered_rsrq.get(cid_i)
                if base is None:
                    # Engine uses -20.0 fallback in evaluate(); mirror that
                    # so the surfaced metric matches what evaluate() would
                    # compare. The "filt_rsrq" field stays None to signal
                    # absence.
                    base_metric: float = -20.0
                else:
                    base_metric = float(base)
            else:
                base = self.filtered_rsrp.get(cid_i)
                if base is None:
                    base_metric = -140.0
                else:
                    base_metric = float(base)

            ocn = float(cio_table.get(int(cid_i), 0.0)) if cio_table else 0.0
            metric = base_metric + ocn

            ttt_elapsed_s: float = float(
                self.ttt_trackers["A3"].get(str(cid_i), 0.0)
            )
            ttt_elapsed_ms: float = ttt_elapsed_s * 1000.0
            age: int = int(self._candidate_age.get(int(cid_i), 0))

            # Entering check — must mirror evaluate() exactly:
            #   metric > serving_metric + a3_offset + hys
            # AND not s_measure_blocked
            # AND not post-HO blacklisted for (serving, cid)
            # AND age >= _A3_CANDIDATE_MIN_AGE (freshness gate)
            entering = False
            if not s_measure_blocked and serving_metric is not None:
                # Post-HO blacklist filter.
                blacklisted = False
                if self._post_ho_blacklist:
                    bl_key = (int(serving_cell_id), int(cid_i))
                    expiry = self._post_ho_blacklist.get(bl_key)
                    if expiry is not None and current_time_s is not None:
                        if float(current_time_s) < float(expiry):
                            blacklisted = True
                if not blacklisted and age >= self._A3_CANDIDATE_MIN_AGE:
                    threshold_a3_enter = serving_metric + a3_offset + hys
                    if metric > threshold_a3_enter:
                        entering = True

            if entering:
                entering_now.append(int(cid_i))

            cand_records.append({
                "cell_id": int(cid_i),
                "metric": float(metric),
                "ttt_elapsed_ms": float(ttt_elapsed_ms),
                "age": int(age),
                "entering": bool(entering),
                "filt_rsrq": (
                    float(self.filtered_rsrq[cid_i])
                    if self.filtered_rsrq and cid_i in self.filtered_rsrq
                    else None
                ),
                "filt_rsrp": (
                    float(self.filtered_rsrp[cid_i])
                    if self.filtered_rsrp and cid_i in self.filtered_rsrp
                    else None
                ),
            })

        # Top-3 by metric DESC.
        cand_records.sort(
            key=lambda r: float(r["metric"]) if r["metric"] is not None else -float("inf"),
            reverse=True,
        )
        top3 = cand_records[:3]

        return {
            "serving_cell_id": int(serving_cell_id),
            "serving_filt_rsrq": (
                float(serving_filt_rsrq) if serving_filt_rsrq is not None else None
            ),
            "serving_filt_rsrp": (
                float(serving_filt_rsrp) if serving_filt_rsrp is not None else None
            ),
            "serving_a3_quantity": str(serving_a3_quantity),
            "filter_age_ms": (
                float(filter_age_ms) if filter_age_ms is not None else None
            ),
            "entering_now": entering_now,
            "candidates": top3,
            "tracker_n": int(len(self.ttt_trackers.get("A3", {}))),
            # Observability read-outs (do not gate anything):
            "report_block_reason": self._last_report_block_reason,
            "report_gap_ms": self._last_report_gap_ms,
        }

    def evaluate(self, current_time: float, dt: float,
                 serving_cell_id: int,
                 measurements: Dict[int, float],
                 inter_freq_cells: Optional[List[int]] = None,
                 inter_rat_cells: Optional[List[int]] = None,
                 freeze_cancel: bool = False) -> Dict[str, 'EventState']:
        """
        Evaluate A2, A3, A5 measurement events.

        Args:
            current_time: Current simulation time
            dt: Time delta since last evaluation
            serving_cell_id: Current serving cell ID
            measurements: {cell_id: rsrp_dbm} - filtered or raw
            inter_freq_cells: List of inter-frequency cell IDs

        Returns:
            Dict mapping event type to EventState
        """
        from .rrc_types import EventState

        if inter_freq_cells is None:
            inter_freq_cells = []
        if inter_rat_cells is None:
            inter_rat_cells = []

        self._freeze_cancel = freeze_cancel
        results = {}

        # 3GPP TS 38.331 §5.5.3.2 (Layer-3 filtering) + §5.5.4.1 (event
        # evaluation): the entering condition references Ms, the L3-filtered
        # serving measurement. When the L1 produced no indication for the
        # serving cell this period (cell absent from the measurements dict),
        # the spec calls for the UE to use the previously stored filtered
        # value (the L3 filter HOLDS its state) — not to fabricate a sentinel
        # like -140 dBm. The original code did `measurements.get(serving, -140)`
        # which made the A3/A5/B1/B2 entering condition `Mn > Ms + offset + hys`
        # trivially true for any visible neighbor (-100 > -140+5+4), generating
        # phantom HO chains → downstream false RLFs.
        #
        # Spec-compliant action: maintain `_last_serving_rsrp` as a held value
        # of the most recent measured serving RSRP, and use it as Ms when the
        # current tick has no fresh serving sample. This lets A3 still fire
        # when a neighbor is genuinely much stronger than the held serving
        # value (legitimate HO away from a fading cell), but does not
        # fabricate a noise-floor Ms that always trips the entering condition.
        # If we have no held value either (very first ticks before any
        # measurement), skip evaluation cleanly with empty triggered states.
        if serving_cell_id in measurements:
            serving_rsrp = float(measurements[serving_cell_id])
            self._last_serving_rsrp = serving_rsrp
        elif self._last_serving_rsrp is not None:
            # L3 filter hold per TS 38.331 §5.5.3.2 — use last stored value.
            serving_rsrp = self._last_serving_rsrp
        else:
            empty: Dict[str, EventState] = {}
            for _ev in ("A2", "A3", "A5", "B1", "B2"):
                _s = EventState(event_type=_ev)
                _s.triggered = False
                _s.report_sent = False
                _s.target_cell_id = None
                _s.quantity = float("-inf")
                _s.time_to_trigger_remaining = 0.0
                empty[_ev] = _s
            return empty

        hys = self.config.hysteresis

        # --- 1. Evaluate A2 (Serving < Threshold) ---
        # Entering: Ms + Hys < Thresh  (uses A2-specific hysteresis per TS 38.331 §5.5.4.2)
        a2_hys = self.config.a2_hysteresis
        a2_entering = (serving_rsrp + a2_hys < self.config.a2_threshold)
        results["A2"] = self._update_ttt("A2", "serving", dt, a2_entering, current_time=current_time,
                                          serving_cell_id=serving_cell_id)

        # A2 gate: Per 3GPP, A5 inter-freq measurement only starts after A2 fires
        if results["A2"].report_sent:
            self.a2_gate_active = True
        elif not a2_entering:
            # A2 leaving condition met (serving recovered) -> close gate
            self.a2_gate_active = False

        # --- 2. Evaluate A3 (Intra-Freq Neighbors) — Per-Cell Independent TTT ---
        # 3GPP TS 38.331 §5.5.4.6: each neighbor independently evaluated
        # Entering: Mn > Ms + Offset + Hys
        # Leaving:  Mn + Hys < Ms + Offset  (i.e. Mn < Ms + Offset - Hys)
        # RSRQ-A3 (regular cells): when self.quantity_for_a3[serving] == "rsrq",
        # the comparison is performed on filtered RSRQ instead of RSRP. The
        # a3_offset/hys parameters retain their dB scaling — same threshold
        # arithmetic, different signal quantity.
        serving_a3_quantity = self.quantity_for_a3.get(int(serving_cell_id), "rsrp")
        if serving_a3_quantity == "rsrq" and self.filtered_rsrq:
            serving_metric = self.filtered_rsrq.get(serving_cell_id, -20.0)
            metric_source = self.filtered_rsrq
        else:
            serving_metric = serving_rsrp
            metric_source = measurements
        threshold_a3_enter = serving_metric + self.config.a3_offset + hys
        threshold_a3_leave = serving_metric + self.config.a3_offset - hys

        # Per 3GPP TS 38.331 §5.5.4.1: "entry condition is fulfilled for all
        # measurements after layer 3 filtering taken during timeToTrigger"
        # → entering condition must be CONTINUOUSLY satisfied every timestep.
        # If entering condition fails at any step, TTT resets for that cell.

        # Reverted Fix G (L3 cadence quantization): T_evaluate=200ms is the
        # spec's *integration window*, NOT the output cadence. L3 filter
        # outputs per L1 input sample (~40ms for FR1 SSB) and A3 is evaluated
        # continuously. Gating to 200ms cadence was an over-interpretation.
        do_l3_step = True
        a3_eval_dt = dt

        # Per-PCI A3 candidate freshness gate (TS 38.331 §5.5.4.4 implementation
        # latitude). For every cell that appears in measurements with a finite,
        # non-NaN RSRP this tick, increment its age counter. Cells absent (or
        # with NaN/inf RSRP) are reset to 0. Only cells with age >=
        # _A3_CANDIDATE_MIN_AGE are eligible to enter the A3 candidate set,
        # which suppresses single-tick phantom blips from interpolation/decay.
        if not do_l3_step:
            # Skip A3 evaluation between L3 outputs; trackers/entering set held.
            pass
        else:
            seen_this_tick = set()
            for _cell_id, _rsrp in measurements.items():
                if _rsrp is None:
                    continue
                try:
                    _v = float(_rsrp)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(_v):
                    continue
                seen_this_tick.add(int(_cell_id))
                self._candidate_age[int(_cell_id)] = self._candidate_age.get(int(_cell_id), 0) + 1
            # Reset age for cells not seen this tick
            for _cid in list(self._candidate_age.keys()):
                if _cid not in seen_this_tick:
                    self._candidate_age[_cid] = 0

            # Build set of cells currently meeting entering condition
            # 3GPP TS 38.331 §5.5.4.4: include per-(serving,neighbor) Ocn from cio_table
            a3_entering_now = set()
            # S-Measure gate (3GPP TS 38.331 §5.5.4 measObjectNR / s-Measure):
            # if serving RSRP exceeds the threshold, suppress all A3 evaluation.
            # Used by HSR cells to skip neighbor measurement while serving is strong.
            s_meas = getattr(self.config, "s_measure_dbm", None)
            s_measure_blocked = s_meas is not None and serving_rsrp > s_meas
            if s_measure_blocked:
                # Clear in-flight TTT trackers; the downstream loop is skipped via
                # the empty `a3_entering_now` so no entering set is built.
                self.ttt_trackers["A3"].clear()
            for cell_id, rsrp in measurements.items():
                if s_measure_blocked:
                    break
                if cell_id == serving_cell_id or cell_id in inter_freq_cells or cell_id in inter_rat_cells:
                    continue
                # Vendor gNB MRO post-HO blacklist (TS 38.473 / TS 28.541 NRM):
                # if (serving_cell, cell_id) is on the blacklist and not yet
                # expired, skip this neighbor for A3. Lazily evict expired
                # entries so the dict does not grow unbounded.
                if self._post_ho_blacklist:
                    bl_key = (int(serving_cell_id), int(cell_id))
                    expiry = self._post_ho_blacklist.get(bl_key)
                    if expiry is not None:
                        if current_time < expiry:
                            continue
                        else:
                            del self._post_ho_blacklist[bl_key]
                # Per 3GPP TS 38.331 §5.5.4.4: A3 compares neighbour and serving
                # in the SAME quantity. The SERVING cell's class decides which
                # metric. When serving is regular (RSRQ-A3), evaluate every
                # neighbour by its filtered RSRQ regardless of the neighbour's
                # own HSR/regular classification — otherwise an HSR neighbour
                # gets its RSRP (~-100 dBm) compared against an RSRQ threshold
                # (~-20 dB) and never triggers, silently blocking non-HSR →
                # HSR A3 handovers.
                if serving_a3_quantity == "rsrq" and self.filtered_rsrq:
                    neighbor_metric = self.filtered_rsrq.get(cell_id, -20.0)
                else:
                    neighbor_metric = rsrp
                ocn = self.config.cio_table.get(int(cell_id), 0.0) if self.config.cio_table else 0.0
                if neighbor_metric + ocn > threshold_a3_enter:
                    # Freshness gate: require N consecutive ticks with finite RSRP
                    # before allowing the TTT to start.
                    if self._candidate_age.get(int(cell_id), 0) < self._A3_CANDIDATE_MIN_AGE:
                        continue
                    a3_entering_now.add(str(cell_id))

            # Start new TTTs for cells that just entered
            for key in a3_entering_now:
                if key not in self.ttt_trackers["A3"]:
                    self.ttt_trackers["A3"][key] = 0.0

            # Cancel TTT for cells NOT currently meeting entering condition,
            # EXCEPT: candidates with >=50% TTT accumulated get a one-tick grace
            # period (consistent with §5.5.4 leaving-hysteresis intent; mitigates
            # single-sample noise during TTT — does NOT extend trigger duration).
            # Track grace state per-cell to prevent zombies.
            if not hasattr(self, "_a3_grace_ticks"):
                self._a3_grace_ticks = {}

            ttt_a3_sec = self.config.time_to_trigger_ms / 1000.0
            half_ttt = ttt_a3_sec * 0.5
            stale = [key for key in self.ttt_trackers["A3"] if key not in a3_entering_now]
            for key in stale:
                elapsed = self.ttt_trackers["A3"].get(key, 0.0)
                # Grace bookkeeping (ZOMBIE FIX 2026-06-12): `key in
                # _a3_grace_ticks` means this cell already spent its single
                # grace evaluation while stale. The old code reset the marker
                # to 0 in the accumulation loop below, so the `== 0` check
                # here re-granted grace EVERY eval → the tracker never died:
                # a TTT-expired candidate kept firing A3 reports at every
                # reportInterval long after its entering condition cleared
                # (user-observed; the source of junk-target HOs / FALSE
                # RACH_PROBLEM RLFs). The marker is now left in place while
                # stale and only removed on re-entry or deletion.
                if elapsed >= half_ttt and key not in self._a3_grace_ticks:
                    # Grant ONE evaluation of grace; tracker keeps its
                    # elapsed value (accumulation skipped below).
                    self._a3_grace_ticks[key] = 1
                else:
                    # Either no grace earned OR grace already used → delete tracker
                    del self.ttt_trackers["A3"][key]
                    self._a3_grace_ticks.pop(key, None)

            # Reset grace for cells re-entering
            for key in a3_entering_now:
                self._a3_grace_ticks.pop(key, None)

            # Accumulate L3 period for cells meeting entering condition
            # (grace-period cells skip accumulation — their tracker freezes
            # for this step; the grace marker itself stays until re-entry
            # or deletion, see ZOMBIE FIX above)
            for key in list(self.ttt_trackers["A3"].keys()):
                if key not in self._a3_grace_ticks:
                    self.ttt_trackers["A3"][key] += a3_eval_dt

        # Find cell with most TTT progress → report candidate
        # 3GPP TS 38.331 §5.5.5.1: "the UE shall include in measResults all
        # neighbour cells fulfilling the entry condition, ordered by descending
        # order of [reportQuantity]". When multiple A3-eligible cells have
        # passed TTT in the same evaluation tick, the strongest must be the
        # report's primary target — NOT whichever was inserted first into the
        # dict. Sort by per-neighbor metric (RSRP, including Ocn) descending
        # so the strongest TTT-expired candidate wins.
        ttt_a3_sec = self.config.time_to_trigger_ms / 1000.0

        def _a3_metric_for(key_str: str) -> float:
            try:
                cid = int(key_str)
            except (TypeError, ValueError):
                return -float("inf")
            # Same serving-driven metric rule as the entering check above
            # so the report ordering uses the comparable quantity.
            if serving_a3_quantity == "rsrq" and self.filtered_rsrq:
                base = self.filtered_rsrq.get(cid, -20.0)
            else:
                base = measurements.get(cid, -140.0)
            try:
                base = float(base)
            except (TypeError, ValueError):
                return -float("inf")
            ocn = self.config.cio_table.get(cid, 0.0) if self.config.cio_table else 0.0
            return base + ocn

        ordered_a3 = sorted(
            self.ttt_trackers["A3"].items(),
            key=lambda kv: _a3_metric_for(kv[0]),
            reverse=True,
        )
        best_a3_cell = None
        best_a3_elapsed = -1.0
        a3_report_sent = False
        # Observability read-out: reset each evaluation; set in the else-branch
        # below if a TTT-expired A3 report is withheld by the reportInterval gate.
        self._last_report_block_reason = ""
        self._last_report_gap_ms = None
        # First pass: among TTT-expired cells, pick the STRONGEST (TS 38.331 §5.5.5.1).
        # Per 3GPP cellsTriggeredList semantics: a cell stays in the list as long
        # as its entering condition holds; only leaving condition + TTT removes it.
        # We therefore keep the tracker alive after report (no delete) so that the
        # SAME strongest cell remains the consistent target across consecutive ticks,
        # rather than rotating to the next-best after each report (sim artifact that
        # caused HO Decision delay buffer to thrash on multiple A3 candidates).
        for key, elapsed in ordered_a3:
            if elapsed >= ttt_a3_sec:
                # TTT expired — check reportInterval before sending report
                # 3GPP TS 38.331 §5.5.5: reportInterval is PER-EVENT (one
                # measConfig owns one timestamp), not per-cell. After any A3
                # report is sent, no further A3 reports for ANY cell until
                # interval_s elapses; meanwhile cellsTriggeredList can keep
                # growing. Key `("A3", None)` is the per-event slot.
                interval_s = self.config.a3_report_interval_ms / 1000.0
                cid_int = int(key) if key.isdigit() else None
                best_a3_cell = cid_int
                best_a3_elapsed = elapsed
                report_key: Tuple[str, Optional[int]] = ("A3", None)
                last_t = self._last_report_time.get(report_key, -9999.0)
                if (current_time - last_t) >= interval_s:
                    a3_report_sent = True
                    self._last_report_time[report_key] = current_time
                else:
                    # Suppressed by reportInterval — diagnostics read-out only,
                    # does NOT change the gate (TS 38.331 §5.5.5 behavior intact).
                    self._last_report_block_reason = "REPORT_INTERVAL"
                    self._last_report_gap_ms = (current_time - last_t) * 1000.0
                # NOTE: do NOT delete the TTT tracker — cell remains in
                # cellsTriggeredList until its leaving condition is satisfied.
                break
        # If no TTT-expired cell, surface the strongest in-progress candidate
        # (informational `target_cell_id` for downstream visibility).
        if best_a3_cell is None:
            for key, elapsed in ordered_a3:
                if elapsed > best_a3_elapsed:
                    best_a3_elapsed = elapsed
                    try:
                        best_a3_cell = int(key)
                    except (TypeError, ValueError):
                        best_a3_cell = None

        a3_state = EventState(event_type="A3")
        a3_state.target_cell_id = best_a3_cell
        if a3_report_sent:
            a3_state.triggered = True
            a3_state.report_sent = True
            a3_state.time_to_trigger_remaining = 0.0
            a3_state.quantity = measurements.get(best_a3_cell, -140.0) if best_a3_cell else -140.0
        elif best_a3_cell is not None:
            a3_state.triggered = True
            a3_state.report_sent = False
            a3_state.time_to_trigger_remaining = max(0, (ttt_a3_sec - best_a3_elapsed) * 1000.0)
            a3_state.quantity = measurements.get(best_a3_cell, -140.0) if best_a3_cell else -140.0
        else:
            a3_state.triggered = False
            a3_state.quantity = -140.0
        results["A3"] = a3_state

        # --- 3. Evaluate A5 (Inter-Freq Neighbors) — Per-Cell, GATED by A2 ---
        if self.a2_gate_active:
            threshold_a5_enter_serving = self.config.a5_threshold1
            serving_bad = (serving_rsrp + hys < threshold_a5_enter_serving)

            # Per 3GPP TS 38.331 §5.5.4.1: continuous satisfaction required.
            # Build set of cells currently meeting A5 entering condition
            a5_entering_now = set()
            if serving_bad:
                for cell_id in inter_freq_cells:
                    rsrp = measurements.get(cell_id, -140.0)
                    if rsrp - hys > self.config.a5_threshold2:
                        a5_entering_now.add(str(cell_id))

            # Start new TTTs
            for key in a5_entering_now:
                if key not in self.ttt_trackers["A5"]:
                    self.ttt_trackers["A5"][key] = 0.0

            # Cancel TTT for cells NOT currently meeting entering condition
            stale = [k for k in self.ttt_trackers["A5"] if k not in a5_entering_now]
            for k in stale:
                del self.ttt_trackers["A5"][k]

            # Accumulate dt for cells meeting entering condition this step
            for key in list(self.ttt_trackers["A5"].keys()):
                self.ttt_trackers["A5"][key] += dt

            ttt_a5_sec = (self.config.a5_ttt_ms if self.config.a5_ttt_ms > 0
                          else self.config.time_to_trigger_ms) / 1000.0
            # 3GPP TS 38.331 §5.5.5.1: report cells ordered by descending
            # reportQuantity. When multiple A5-eligible inter-freq cells expire
            # TTT in the same tick, pick the strongest as the report target.
            def _a5_metric_for(key_str: str) -> float:
                try:
                    cid = int(key_str)
                except (TypeError, ValueError):
                    return -float("inf")
                v = measurements.get(cid, -140.0)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return -float("inf")
            ordered_a5 = sorted(
                self.ttt_trackers["A5"].items(),
                key=lambda kv: _a5_metric_for(kv[0]),
                reverse=True,
            )
            best_a5_cell = None
            best_a5_elapsed = -1.0
            a5_report_sent = False
            for key, elapsed in ordered_a5:
                if elapsed >= ttt_a5_sec:
                    best_a5_cell = int(key) if key.isdigit() else None
                    a5_report_sent = True
                    del self.ttt_trackers["A5"][key]
                    break
            if best_a5_cell is None:
                for key, elapsed in ordered_a5:
                    if elapsed > best_a5_elapsed:
                        best_a5_elapsed = elapsed
                        best_a5_cell = int(key) if key.isdigit() else None

            a5_state = EventState(event_type="A5")
            a5_state.target_cell_id = best_a5_cell
            if a5_report_sent:
                a5_state.triggered = True
                a5_state.report_sent = True
                a5_state.time_to_trigger_remaining = 0.0
            elif best_a5_cell is not None:
                a5_state.triggered = True
                a5_state.report_sent = False
                a5_state.time_to_trigger_remaining = max(0, (ttt_a5_sec - best_a5_elapsed) * 1000.0)
            else:
                a5_state.triggered = False
            a5_state.quantity = measurements.get(best_a5_cell, -140.0) if best_a5_cell else -140.0
        else:
            if not freeze_cancel:
                self.ttt_trackers["A5"].clear()
            a5_state = EventState(event_type="A5")
            a5_state.target_cell_id = None
            a5_state.quantity = -140.0
        results["A5"] = a5_state

        # --- 4. Evaluate B2 (Inter-RAT dual threshold) --- GATED by A2
        # 3GPP TS 38.331 §5.5.4.8
        # Entering: (Ms + Hys < Thresh1) AND (Mn + Ofn - Hys > Thresh2)
        if self.a2_gate_active and inter_rat_cells:
            serving_bad_b2 = (serving_rsrp + hys < self.config.b2_threshold1)
            best_rat_b2 = None
            max_rat_rsrp_b2 = -999.0

            if serving_bad_b2:
                for cell_id in inter_rat_cells:
                    rsrp = measurements.get(cell_id, -140.0)
                    if rsrp + self.config.b2_offset - hys > self.config.b2_threshold2:
                        if rsrp > max_rat_rsrp_b2:
                            max_rat_rsrp_b2 = rsrp
                            best_rat_b2 = cell_id

            b2_key = str(best_rat_b2) if best_rat_b2 else "none"
            if best_rat_b2 is not None:
                stale = [k for k in self.ttt_trackers["B2"] if k != b2_key]
                for k in stale:
                    del self.ttt_trackers["B2"][k]
            else:
                self.ttt_trackers["B2"].clear()
            b2_state = self._update_ttt("B2", b2_key, dt, best_rat_b2 is not None)
            b2_state.target_cell_id = best_rat_b2
            b2_state.quantity = max_rat_rsrp_b2 if best_rat_b2 else -140.0
        else:
            self.ttt_trackers["B2"].clear()
            b2_state = EventState(event_type="B2")
            b2_state.target_cell_id = None
            b2_state.quantity = -140.0
        results["B2"] = b2_state

        # --- 5. Evaluate B1 (Inter-RAT single threshold) --- GATED by A2
        # 3GPP TS 38.331 §5.5.4.7
        # Entering: Mn + Ofn - Hys > Thresh
        if self.a2_gate_active and inter_rat_cells:
            best_rat_b1 = None
            max_rat_rsrp_b1 = -999.0

            for cell_id in inter_rat_cells:
                rsrp = measurements.get(cell_id, -140.0)
                if rsrp + self.config.b1_offset - hys > self.config.b1_threshold:
                    if rsrp > max_rat_rsrp_b1:
                        max_rat_rsrp_b1 = rsrp
                        best_rat_b1 = cell_id

            b1_key = str(best_rat_b1) if best_rat_b1 else "none"
            if best_rat_b1 is not None:
                stale = [k for k in self.ttt_trackers["B1"] if k != b1_key]
                for k in stale:
                    del self.ttt_trackers["B1"][k]
            else:
                self.ttt_trackers["B1"].clear()
            b1_state = self._update_ttt("B1", b1_key, dt, best_rat_b1 is not None)
            b1_state.target_cell_id = best_rat_b1
            b1_state.quantity = max_rat_rsrp_b1 if best_rat_b1 else -140.0
        else:
            self.ttt_trackers["B1"].clear()
            b1_state = EventState(event_type="B1")
            b1_state.target_cell_id = None
            b1_state.quantity = -140.0
        results["B1"] = b1_state

        return results

    def reset(self):
        """Reset all internal state"""
        self.ttt_trackers = {"A2": {}, "A3": {}, "A5": {}, "B1": {}, "B2": {}}
        self._a3_grace_ticks = {}
        self.filtered_rsrp = {}
        self.filtered_rsrq = {}
        self.a2_gate_active = False
