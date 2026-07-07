"""
NR Handover Digital Twin Simulator

Main simulation orchestrator that integrates:
- Sionna RT-based channel calculation
- PHY abstraction (SINR→BLER)
- RRC controller (RLF/HO procedures)
- Event scheduler

Supports 3GPP handover failure case detection:
- Too Late: RLF at source before HO complete
- Too Early: RLF at target shortly after HO
- Wrong Cell: Re-establishment to third cell after HO failure
- Weak Cell / Ping-Pong: Short stay at target cell

Based on 3GPP TS 38.331, TS 38.133.
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from pathlib import Path
import json
import csv
from datetime import datetime

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path for imports
_src_dir = Path(__file__).parent
_project_dir = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

# Import local modules - using absolute imports
try:
    from src.core.event_scheduler import EventScheduler, EventType
    from src.phy.phy_abstraction import PHYAbstraction, RLFDetector
    from src.rrc.measurement import MeasConfig, MeasResult, MeasEventType, MeasQuantity
    from src.rrc.rrc_controller import UERRCController, RRCState, HOState, HandoverResult
    from src.channel.channel_calculator import SionnaChannelCalculator, ChannelState
    from src.channel.trajectory_channel_map import TrajectoryChannelMap, TrajectoryChannelMapConfig
except ImportError:
    # Fallback for direct execution
    from core.event_scheduler import EventScheduler, EventType
    from phy.phy_abstraction import PHYAbstraction, RLFDetector
    from rrc.measurement import MeasConfig, MeasResult, MeasEventType, MeasQuantity
    from rrc.rrc_controller import UERRCController, RRCState, HOState, HandoverResult
    from channel.channel_calculator import SionnaChannelCalculator, ChannelState
    from channel.trajectory_channel_map import TrajectoryChannelMap, TrajectoryChannelMapConfig


class HOFailureType(Enum):
    """3GPP Handover Failure Classification"""
    NONE = auto()
    TOO_LATE = auto()        # RLF at source, reconnect to target
    TOO_EARLY = auto()       # RLF at target shortly after HO, back to source
    WRONG_CELL = auto()      # RLF at target, reconnect to third cell
    WEAK_CELL = auto()       # Short stay at target (no RLF)
    PING_PONG = auto()       # Back to source within threshold


@dataclass
class HOEvent:
    """Handover event record with failure classification"""
    timestamp: float
    ue_id: int
    source_cell: int
    target_cell: int
    event_type: str  # 'START', 'COMPLETE', 'FAILURE', 'RLF'

    # HO timing
    ho_start_time: Optional[float] = None
    ho_complete_time: Optional[float] = None
    ho_duration_ms: Optional[float] = None

    # Failure info
    failure_type: Optional[HOFailureType] = None
    failure_reason: Optional[str] = None
    reestablishment_cell: Optional[int] = None
    time_since_ho_complete_ms: Optional[float] = None

    # Radio conditions
    source_rsrp_dbm: Optional[float] = None
    target_rsrp_dbm: Optional[float] = None
    source_sinr_db: Optional[float] = None
    target_sinr_db: Optional[float] = None
    bler: Optional[float] = None

    # 3GPP timers (for verification)
    t310_started: bool = False
    t310_expired: bool = False
    t304_expired: bool = False
    n310_count: int = 0
    n311_count: int = 0


@dataclass
class RLFEvent:
    """RLF event record"""
    timestamp: float
    ue_id: int
    cell_id: int
    sinr_db: float
    bler: float
    n310_count: int
    consecutive_oos: int

    # Context
    previous_cell: Optional[int] = None
    time_since_ho_ms: Optional[float] = None
    ho_state_at_rlf: Optional[str] = None
    cause: Optional[str] = None


@dataclass
class MeasurementTrace:
    """Single measurement trace point"""
    timestamp: float
    ue_id: int
    position: Tuple[float, float, float]
    serving_cell: int
    serving_rsrp_dbm: float
    serving_sinr_db: float
    bler: float

    # All cell measurements
    cell_measurements: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    # Duration
    duration_s: float = 30.0
    measurement_period_ms: float = 20.0  # 20ms as requested

    # Channel
    frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 46.0

    # Channel model selection
    channel_model_type: str = "sionna_rt"  # "statistical" or "sionna_rt"
    channel_scenario: str = "UMa"  # For statistical model: UMa, UMi, RMa

    # Measurement
    a3_offset_db: float = 3.0
    hysteresis_db: float = 2.0
    ttt_ms: float = 256.0
    filter_coefficient: int = 4

    # RLF
    n310: int = 1
    n311: int = 1
    t310_ms: float = 1000.0
    qout_bler: float = 0.10
    qin_bler: float = 0.02

    # Timers
    t304_ms: float = 200.0
    t311_ms: float = 10000.0

    # MCS - use low MCS to avoid immediate RLF at cell edge
    default_mcs: int = 0  # QPSK with lowest code rate - tolerates low SINR

    # Failure classification thresholds
    too_early_threshold_ms: float = 1000.0  # Time since HO for Too Early
    short_stay_threshold_ms: float = 1000.0  # Min stay for Weak Cell


class NRHandoverSimulator:
    """
    NR Handover Digital Twin Simulator.

    Orchestrates:
    1. UE trajectory-based mobility
    2. Sionna RT channel calculation
    3. PHY abstraction (SINR→BLER)
    4. RRC state machine (RLF, HO)
    5. 3GPP timer processing
    6. HO failure case detection

    Usage:
        sim = NRHandoverSimulator(config)
        sim.add_gnb(...)
        sim.add_ue(...)
        sim.run()
        sim.analyze_results()
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize simulator.

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self.logger = logging.getLogger("Simulator")

        # Event scheduler
        self.scheduler = EventScheduler()

        # Channel calculator using factory pattern
        from src.channel import ChannelModelFactory, ChannelConfig, ChannelModelType

        channel_config = ChannelConfig(
            model_type=ChannelModelType(self.config.channel_model_type),
            frequency_hz=self.config.frequency_hz,
            tx_power_dbm=self.config.tx_power_dbm,
            bandwidth_hz=self.config.bandwidth_hz,
            scenario=self.config.channel_scenario,
        )

        self.channel_calc = ChannelModelFactory.create(channel_config)
        self.logger.info(f"Using channel model: {self.channel_calc.model_type.value}")

        # PHY abstraction
        self.phy_abstraction = PHYAbstraction()

        # gNB configurations: {gnb_id: {position, ...}}
        self.gnb_configs: Dict[int, Dict] = {}

        # UE controllers: {ue_id: UERRCController}
        self.ue_controllers: Dict[int, UERRCController] = {}

        # UE trajectories: {ue_id: [(timestamp, x, y, z), ...]}
        self.ue_trajectories: Dict[int, List[Tuple[float, float, float, float]]] = {}

        # UE current positions
        self.ue_positions: Dict[int, Tuple[float, float, float]] = {}

        # Event tracking
        self.ho_events: List[HOEvent] = []
        self.rlf_events: List[RLFEvent] = []
        self.measurement_traces: List[MeasurementTrace] = []

        # HO state tracking for failure classification
        self._ue_ho_history: Dict[int, Dict] = {}  # {ue_id: {last_ho_time, last_source, ...}}
        self._pending_rlf_classification: Dict[int, Dict] = {}

        # Simulation state
        self._running = False
        self._current_time = 0.0

        # Pre-computed channel map for trajectory-aware optimization
        self._channel_map: Optional[TrajectoryChannelMap] = None
        self._use_precomputed = True  # Enable pre-computation by default

        self.logger.info("NRHandoverSimulator initialized")

    def add_gnb(
        self,
        gnb_id: int,
        position: Tuple[float, float, float],
        azimuth_deg: float = 0.0,
        tx_power_dbm: Optional[float] = None
    ):
        """Add a gNB to the simulation"""
        self.gnb_configs[gnb_id] = {
            'position': position,
            'azimuth_deg': azimuth_deg,
            'tx_power_dbm': tx_power_dbm or self.config.tx_power_dbm
        }

        self.channel_calc.add_gnb(
            gnb_id=gnb_id,
            position=position,
            sector_id=1,
            azimuth_deg=azimuth_deg,
            tx_power_dbm=tx_power_dbm or self.config.tx_power_dbm
        )

        self.logger.info(f"Added gNB_{gnb_id} at {position}")

    def add_ue(
        self,
        ue_id: int,
        initial_position: Tuple[float, float, float],
        serving_cell: int,
        trajectory: Optional[List[Tuple[float, float, float, float]]] = None
    ):
        """
        Add a UE to the simulation.

        Args:
            ue_id: UE identifier
            initial_position: (x, y, z) starting position
            serving_cell: Initial serving cell gNB ID
            trajectory: Optional list of (timestamp, x, y, z) waypoints
        """
        # Create measurement config (A3 event)
        meas_config = MeasConfig(
            meas_id=1,
            event_type=MeasEventType.A3,
            quantity=MeasQuantity.RSRP,
            a3_offset=self.config.a3_offset_db,
            hysteresis=self.config.hysteresis_db,
            time_to_trigger_ms=self.config.ttt_ms,
            filter_coefficient=self.config.filter_coefficient
        )

        # RLF configuration
        rlf_config = {
            'N310': self.config.n310,
            'N311': self.config.n311,
            'Qout_bler': self.config.qout_bler,
            'Qin_bler': self.config.qin_bler
        }

        # Timer configuration
        timer_config = {
            'T304_ms': self.config.t304_ms,
            'T310_ms': self.config.t310_ms,
            'T311_ms': self.config.t311_ms
        }

        # Create RRC controller
        controller = UERRCController(
            ue_id=ue_id,
            scheduler=self.scheduler,
            meas_configs=[meas_config],
            rlf_config=rlf_config,
            timer_config=timer_config
        )

        # Set callbacks
        controller.on_handover_start = self._on_ho_start
        controller.on_handover_complete = self._on_ho_complete
        controller.on_handover_failure = self._on_ho_failure
        controller.on_rlf = self._on_rlf
        controller.on_reestablishment_complete = self._on_reestablishment_complete
        controller.on_reestablishment_failure = self._on_reestablishment_failure

        # Attach to serving cell
        controller.attach_to_cell(serving_cell)

        self.ue_controllers[ue_id] = controller
        self.ue_positions[ue_id] = initial_position
        self.ue_trajectories[ue_id] = trajectory or []

        # Initialize HO history
        self._ue_ho_history[ue_id] = {
            'last_ho_complete_time': None,
            'last_source_cell': None,
            'last_target_cell': None,
            'cell_entry_time': 0.0,
            'previous_cells': []
        }

        # Add to channel calculator
        self.channel_calc.add_ue(ue_id=ue_id, position=initial_position)

        self.logger.info(f"Added UE_{ue_id} at {initial_position}, serving cell {serving_cell}")

    def set_linear_trajectory(
        self,
        ue_id: int,
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        speed_mps: float,
        start_time: float = 0.0
    ):
        """
        Set a linear trajectory for a UE.

        Args:
            ue_id: UE identifier
            start_pos: Starting position (x, y, z)
            end_pos: Ending position (x, y, z)
            speed_mps: Speed in m/s
            start_time: Start time in seconds
        """
        # Calculate distance and duration
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dz = end_pos[2] - start_pos[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        duration = distance / speed_mps

        # Generate trajectory points at measurement period intervals
        period_s = self.config.measurement_period_ms / 1000.0
        num_points = int(duration / period_s) + 1

        trajectory = []
        for i in range(num_points):
            t = start_time + i * period_s
            progress = i * period_s / duration if duration > 0 else 0
            progress = min(1.0, progress)

            x = start_pos[0] + progress * dx
            y = start_pos[1] + progress * dy
            z = start_pos[2] + progress * dz

            trajectory.append((t, x, y, z))

        self.ue_trajectories[ue_id] = trajectory
        self.logger.info(f"UE_{ue_id} trajectory: {len(trajectory)} points, "
                        f"{distance:.1f}m at {speed_mps:.1f}m/s ({speed_mps*3.6:.1f}km/h)")

    # ═══════════════════════════════════════════════════════════════════
    # Pre-Computation for Trajectory-Aware Optimization
    # ═══════════════════════════════════════════════════════════════════

    def _precompute_channels(self) -> None:
        """
        Pre-computation phase before simulation loop.

        This method:
        1. Checks for cached HDF5 file
        2. If not found, pre-computes channels for all trajectory positions
        3. Saves to HDF5 for future runs

        For high-speed railway scenarios with known trajectory, this provides
        100% cache hit rate during simulation (vs ~0% with on-demand caching).
        """
        if not self._use_precomputed:
            return

        if not self.ue_trajectories:
            self.logger.warning("No UE trajectories defined, skipping pre-computation")
            return

        import hashlib

        # Determine cache file path
        cache_dir = Path("output/channel_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache filename based on trajectory hash
        trajectory_hash = self._compute_trajectory_hash()
        cache_file = cache_dir / f"channel_map_{trajectory_hash}.h5"

        # Create channel map
        config = TrajectoryChannelMapConfig(
            sample_interval_m=5.0,
            interpolation_enabled=True,
            cache_file=str(cache_file)
        )
        self._channel_map = TrajectoryChannelMap(config)

        # Try to load from cache
        if self._channel_map.load_from_hdf5(str(cache_file)):
            self.logger.info(f"Loaded pre-computed channel map from {cache_file}")
            return

        # Extract positions from trajectory
        positions = self._extract_trajectory_positions()

        if not positions:
            self.logger.warning("No trajectory positions extracted, skipping pre-computation")
            self._channel_map = None
            return

        self.logger.info(f"Pre-computing channels for {len(positions)} positions...")

        # Pre-compute (this is the slow part - only runs once)
        ue_id = list(self.ue_controllers.keys())[0]  # First UE
        num_computed = self._channel_map.precompute(
            positions,
            self.channel_calc,
            ue_id=ue_id
        )

        # Save for future runs
        try:
            self._channel_map.save_to_hdf5(str(cache_file))
            self.logger.info(f"Saved channel map ({num_computed} positions) to {cache_file}")
        except Exception as e:
            self.logger.warning(f"Could not save channel map to HDF5: {e}")

    def _extract_trajectory_positions(self) -> List[Tuple[float, float, float]]:
        """Extract all unique positions from UE trajectories."""
        positions = []
        for ue_id, trajectory in self.ue_trajectories.items():
            for (t, x, y, z) in trajectory:
                positions.append((x, y, z))
        return positions

    def _compute_trajectory_hash(self) -> str:
        """Compute hash of trajectory for cache invalidation."""
        import hashlib

        # Include trajectory data and gNB configuration in hash
        data_parts = []

        # Trajectory info
        for ue_id in sorted(self.ue_trajectories.keys()):
            trajectory = self.ue_trajectories[ue_id]
            if trajectory:
                # Include first, middle, last points and count
                data_parts.append(f"ue{ue_id}:{len(trajectory)}")
                data_parts.append(f"start:{trajectory[0]}")
                data_parts.append(f"end:{trajectory[-1]}")

        # gNB configuration
        for gnb_id in sorted(self.gnb_configs.keys()):
            pos = self.gnb_configs[gnb_id]['position']
            data_parts.append(f"gnb{gnb_id}:{pos}")

        # Channel config
        data_parts.append(f"freq:{self.config.frequency_hz}")

        data = "|".join(data_parts)
        return hashlib.md5(data.encode()).hexdigest()[:8]

    def set_precomputation_enabled(self, enabled: bool) -> None:
        """
        Enable or disable trajectory-aware pre-computation.

        Args:
            enabled: True to enable pre-computation, False to use on-demand
        """
        self._use_precomputed = enabled
        if not enabled:
            self._channel_map = None

    def run(self, duration_s: Optional[float] = None) -> Dict:
        """
        Run the simulation.

        Args:
            duration_s: Override simulation duration

        Returns:
            Simulation results dictionary
        """
        duration = duration_s or self.config.duration_s
        period_s = self.config.measurement_period_ms / 1000.0

        self.logger.info(f"Starting simulation: duration={duration}s, "
                        f"period={self.config.measurement_period_ms}ms")

        # Pre-compute channels if enabled (one-time, before loop)
        self._precompute_channels()

        self._running = True
        self._current_time = 0.0
        step_count = 0

        while self._running and self._current_time < duration:
            step_count += 1

            # === Step 1: Update UE positions ===
            self._update_ue_positions(self._current_time)

            # === Step 2: Compute channel states ===
            # Use pre-computed map if available, otherwise compute on-demand
            if self._channel_map and self._channel_map.is_precomputed:
                # O(1) lookup from pre-computed map
                first_ue_id = list(self.ue_controllers.keys())[0]
                ue_pos = self.ue_positions.get(first_ue_id)
                if ue_pos:
                    channel_states = self._channel_map.lookup(ue_pos, ue_id=first_ue_id)
                else:
                    channel_states = self.channel_calc.compute_all(timestamp=self._current_time)
            else:
                # On-demand computation (fallback)
                channel_states = self.channel_calc.compute_all(timestamp=self._current_time)

            # === Step 3 & 4: Process each UE ===
            for ue_id, controller in self.ue_controllers.items():
                self._process_ue_step(ue_id, controller, channel_states)

            # === Step 5: Process scheduled events ===
            next_time = self._current_time + period_s
            self.scheduler.run_until(next_time)

            # === Step 6: Logging (every 1 second) ===
            if step_count % 50 == 0:  # ~1 second at 20ms period
                self._log_progress()

            self._current_time = next_time

        self._running = False
        self.logger.info(f"Simulation complete: {step_count} steps, "
                        f"{len(self.ho_events)} HO events, "
                        f"{len(self.rlf_events)} RLF events")

        return self._compile_results()

    def _update_ue_positions(self, current_time: float):
        """Update UE positions based on trajectories"""
        for ue_id, trajectory in self.ue_trajectories.items():
            if not trajectory:
                continue

            # Find appropriate trajectory point
            pos = None
            for i, (t, x, y, z) in enumerate(trajectory):
                if t <= current_time:
                    pos = (x, y, z)
                else:
                    # Interpolate between points
                    if i > 0:
                        t_prev, x_prev, y_prev, z_prev = trajectory[i-1]
                        if t > t_prev:
                            alpha = (current_time - t_prev) / (t - t_prev)
                            pos = (
                                x_prev + alpha * (x - x_prev),
                                y_prev + alpha * (y - y_prev),
                                z_prev + alpha * (z - z_prev)
                            )
                    break

            if pos is None and trajectory:
                # Use last point if past trajectory end
                _, x, y, z = trajectory[-1]
                pos = (x, y, z)

            if pos:
                self.ue_positions[ue_id] = pos
                self.channel_calc.update_ue_position(ue_id, pos, current_time)

    def _process_ue_step(
        self,
        ue_id: int,
        controller: UERRCController,
        channel_states: Dict[Tuple[int, int, int], ChannelState]
    ):
        """Process single simulation step for a UE"""
        serving_cell = controller.context.serving_cell_id
        if serving_cell is None:
            return

        # Get serving cell state
        serving_key = (ue_id, serving_cell, 1)  # sector_id=1
        serving_state = channel_states.get(serving_key)

        if serving_state is None:
            self.logger.warning(f"No channel state for UE_{ue_id} serving cell {serving_cell}")
            return

        # Calculate BLER from SINR
        bler = self.phy_abstraction.sinr_to_bler(
            serving_state.sinr_db,
            self.config.default_mcs
        )

        # Get neighbor measurements
        neighbors = []
        for gnb_id in self.gnb_configs:
            if gnb_id != serving_cell:
                key = (ue_id, gnb_id, 1)
                state = channel_states.get(key)
                if state:
                    neighbors.append({
                        'cell_id': gnb_id,
                        'rsrp': state.rsrp_dbm,
                        'sinr': state.sinr_db,
                        'bler': self.phy_abstraction.sinr_to_bler(
                            state.sinr_db, self.config.default_mcs
                        )
                    })

        # Update RRC controller
        controller.process_channel_update(
            serving_rsrp_dbm=serving_state.rsrp_dbm,
            serving_sinr_db=serving_state.sinr_db,
            serving_bler=bler,
            neighbors=neighbors,
            current_time=self._current_time,
            mcs_index=self.config.default_mcs
        )

        # Record measurement trace
        trace = MeasurementTrace(
            timestamp=self._current_time,
            ue_id=ue_id,
            position=self.ue_positions.get(ue_id, (0, 0, 0)),
            serving_cell=serving_cell,
            serving_rsrp_dbm=serving_state.rsrp_dbm,
            serving_sinr_db=serving_state.sinr_db,
            bler=bler,
            cell_measurements={
                gnb_id: {
                    'rsrp': channel_states.get((ue_id, gnb_id, 1), ChannelState(ue_id, gnb_id, 1, -140)).rsrp_dbm,
                    'sinr': channel_states.get((ue_id, gnb_id, 1), ChannelState(ue_id, gnb_id, 1, -140)).sinr_db
                }
                for gnb_id in self.gnb_configs
            }
        )
        self.measurement_traces.append(trace)

    def _log_progress(self):
        """Log simulation progress"""
        for ue_id, controller in self.ue_controllers.items():
            state = controller.get_state()
            pos = self.ue_positions.get(ue_id, (0, 0, 0))
            self.logger.info(
                f"t={self._current_time:.2f}s UE_{ue_id}: "
                f"pos=({pos[0]:.0f},{pos[1]:.0f}), "
                f"cell={state['serving_cell']}, "
                f"RSRP={state['serving_rsrp_dbm']:.1f}dBm, "
                f"SINR={state['serving_sinr_db']:.1f}dB, "
                f"BLER={state['serving_bler']:.2%}"
            )

    # ═══════════════════════════════════════════════════════════════════
    # Event Callbacks
    # ═══════════════════════════════════════════════════════════════════

    def _on_ho_start(self, ue_id: int, source_cell: int, target_cell: int):
        """Callback when handover starts"""
        # Check for weak-cell / ping-pong condition on PREVIOUS cell
        weak_cell_type = self._check_weak_cell(ue_id, target_cell)
        if weak_cell_type is not None:
            weak_event = HOEvent(
                timestamp=self._current_time,
                ue_id=ue_id,
                source_cell=source_cell,
                target_cell=target_cell,
                event_type='WEAK_CELL',
                failure_type=weak_cell_type
            )
            self.ho_events.append(weak_event)

        event = HOEvent(
            timestamp=self._current_time,
            ue_id=ue_id,
            source_cell=source_cell,
            target_cell=target_cell,
            event_type='START',
            ho_start_time=self._current_time
        )

        # Get current radio conditions
        controller = self.ue_controllers.get(ue_id)
        if controller:
            event.source_rsrp_dbm = controller.context.serving_rsrp_dbm
            event.source_sinr_db = controller.context.serving_sinr_db
            event.bler = controller.context.serving_bler

        self.ho_events.append(event)

        self.logger.info(f"[HO_START] UE_{ue_id}: {source_cell} -> {target_cell} "
                        f"(RSRP={event.source_rsrp_dbm:.1f}dBm)")

    def _on_ho_complete(
        self,
        ue_id: int,
        source_cell: int,
        target_cell: int,
        result: HandoverResult
    ):
        """Callback when handover completes successfully"""
        event = HOEvent(
            timestamp=self._current_time,
            ue_id=ue_id,
            source_cell=source_cell,
            target_cell=target_cell,
            event_type='COMPLETE',
            ho_start_time=result.start_time,
            ho_complete_time=result.end_time,
            ho_duration_ms=result.duration_ms
        )

        self.ho_events.append(event)

        # Update HO history for failure classification
        history = self._ue_ho_history[ue_id]
        history['last_ho_complete_time'] = self._current_time
        history['last_source_cell'] = source_cell
        history['last_target_cell'] = target_cell
        history['cell_entry_time'] = self._current_time
        history['previous_cells'].append(source_cell)
        if len(history['previous_cells']) > 10:
            history['previous_cells'].pop(0)

        self.logger.info(f"[HO_COMPLETE] UE_{ue_id}: {source_cell} -> {target_cell} "
                        f"(duration={result.duration_ms:.1f}ms)")

    def _on_ho_failure(
        self,
        ue_id: int,
        source_cell: int,
        target_cell: int,
        reason: str
    ):
        """Callback when handover fails"""
        event = HOEvent(
            timestamp=self._current_time,
            ue_id=ue_id,
            source_cell=source_cell,
            target_cell=target_cell,
            event_type='FAILURE',
            failure_reason=reason,
            t304_expired=(reason == 'T304_EXPIRE')
        )

        self.ho_events.append(event)

        self.logger.warning(f"[HO_FAILURE] UE_{ue_id}: {source_cell} -> {target_cell} "
                          f"(reason={reason})")

    def _on_rlf(self, ue_id: int, cell_id: int, rlf_info: Dict):
        """
        Callback when RLF is detected.

        Split classification model per 3GPP TS 38.423 MRO:
        - Too-Late: classified IMMEDIATELY (RLF at source or during HO-in-progress)
        - Too-Early/Wrong-Cell: DEFERRED until re-establishment completes
        """
        controller = self.ue_controllers.get(ue_id)
        history = self._ue_ho_history.get(ue_id, {})

        # === T304 duplicate prevention ===
        # When T304 expires, rrc_controller fires both on_handover_failure and on_rlf.
        # Check if a T304 FAILURE event already exists for this UE at the same timestamp.
        if rlf_info.get('cause') == 'T304_EXPIRE':
            for i in range(len(self.ho_events) - 1, max(len(self.ho_events) - 5, -1), -1):
                existing = self.ho_events[i]
                if (existing.ue_id == ue_id and
                    existing.event_type == 'FAILURE' and
                    existing.t304_expired and
                    abs(existing.timestamp - self._current_time) < 0.001):
                    # Update existing T304 FAILURE event with RLF info
                    existing.source_sinr_db = rlf_info.get('sinr_db')
                    existing.bler = rlf_info.get('bler')
                    existing.t310_started = True
                    existing.t310_expired = True
                    self.logger.info(
                        f"[T304_DEDUP] UE_{ue_id}: Updated existing T304 FAILURE event "
                        f"instead of creating duplicate RLF event"
                    )
                    return  # Skip creating duplicate event

        # Create RLF event
        rlf_event = RLFEvent(
            timestamp=self._current_time,
            ue_id=ue_id,
            cell_id=cell_id,
            sinr_db=rlf_info.get('sinr_db', -20),
            bler=rlf_info.get('bler', 1.0),
            n310_count=rlf_info.get('n310_counter', 0),
            consecutive_oos=rlf_info.get('consecutive_oos', 0),
            previous_cell=history.get('last_source_cell'),
            ho_state_at_rlf=controller.context.ho_state.name if controller else None,
            cause=rlf_info.get('cause', 'UNKNOWN')
        )

        # Calculate time since last HO
        last_ho_time = history.get('last_ho_complete_time')
        if last_ho_time is not None:
            rlf_event.time_since_ho_ms = (self._current_time - last_ho_time) * 1000

        self.rlf_events.append(rlf_event)

        # === Split classification: immediate vs deferred ===
        last_source = history.get('last_source_cell')
        last_target = history.get('last_target_cell')

        # HO-in-progress detection
        ho_state_at_rlf = rlf_event.ho_state_at_rlf
        current_ho_target = None
        if controller and ho_state_at_rlf in ('PREPARING', 'EXECUTING', 'COMPLETING'):
            current_ho_target = controller.context.target_cell_id

        failure_type = None
        needs_deferred = False
        expected_target_for_verification = None

        # --- Priority 1: HO was in progress when RLF fired ---
        if ho_state_at_rlf in ('PREPARING', 'EXECUTING', 'COMPLETING'):
            failure_type = HOFailureType.TOO_LATE
            expected_target_for_verification = current_ho_target
            self.logger.info(
                f"[CLASSIFICATION] Too Late (HO in progress): RLF at cell {cell_id} "
                f"while HO to {current_ho_target} was in {ho_state_at_rlf} state"
            )

        # --- Priority 2: No HO ever completed ---
        elif last_ho_time is None:
            failure_type = HOFailureType.TOO_LATE
            self.logger.info(
                f"[CLASSIFICATION] Too Late (no prior HO): RLF at cell {cell_id}"
            )

        # --- Priority 3: RLF at source cell ---
        elif cell_id == last_source:
            failure_type = HOFailureType.TOO_LATE
            expected_target_for_verification = last_target
            self.logger.info(
                f"[CLASSIFICATION] Too Late (at source): RLF at source {last_source}, "
                f"expected target was {last_target}"
            )

        # --- Priority 4: RLF at target cell shortly after HO ---
        elif cell_id == last_target:
            time_since_ho_ms = rlf_event.time_since_ho_ms or 0
            if time_since_ho_ms < self.config.too_early_threshold_ms:
                needs_deferred = True

        # Create HOEvent
        ho_event = HOEvent(
            timestamp=self._current_time,
            ue_id=ue_id,
            source_cell=last_source if last_source is not None else -1,
            target_cell=cell_id,
            event_type='RLF',
            failure_type=failure_type,
            time_since_ho_complete_ms=rlf_event.time_since_ho_ms,
            t310_started=True,
            t310_expired=True,
            n310_count=rlf_event.n310_count,
            bler=rlf_event.bler
        )
        ho_event_index = len(self.ho_events)
        self.ho_events.append(ho_event)

        # Store pending info for deferred classification or Too-Late verification
        if needs_deferred:
            self._pending_rlf_classification[ue_id] = {
                'ho_event_index': ho_event_index,
                'failed_cell': cell_id,
                'last_source': last_source,
                'last_target': last_target,
                'classification': 'PENDING_DEFERRED',
                'timestamp': self._current_time,
                'time_since_ho_ms': rlf_event.time_since_ho_ms or 0
            }
        elif failure_type == HOFailureType.TOO_LATE and expected_target_for_verification is not None:
            self._pending_rlf_classification[ue_id] = {
                'ho_event_index': ho_event_index,
                'failed_cell': cell_id,
                'last_source': last_source,
                'last_target': last_target,
                'expected_target': expected_target_for_verification,
                'classification': 'TOO_LATE',
                'timestamp': self._current_time,
                'time_since_ho_ms': rlf_event.time_since_ho_ms or 0
            }

        self.logger.error(
            f"[RLF] UE_{ue_id} at cell {cell_id}: "
            f"SINR={rlf_event.sinr_db:.1f}dB, BLER={rlf_event.bler:.2%}, "
            f"N310={rlf_event.n310_count}, "
            f"failure_type={failure_type.name if failure_type else 'PENDING'}"
        )

    def _on_reestablishment_complete(self, ue_id: int, source_cell: int,
                                     reest_cell: int, cause: str):
        """Handle successful re-establishment - triggers deferred HO failure classification."""
        pending = self._pending_rlf_classification.pop(ue_id, None)
        if pending is None:
            self.logger.info(f"[REEST_COMPLETE] UE_{ue_id}: source={source_cell}, "
                            f"reest_cell={reest_cell}, cause={cause} (no pending classification)")
            return

        classification = pending.get('classification')
        idx = pending['ho_event_index']

        # === Case A: Deferred classification (Too-Early or Wrong-Cell) ===
        if classification == 'PENDING_DEFERRED':
            last_source = pending['last_source']
            last_target = pending['last_target']

            if reest_cell == last_source:
                failure_type = HOFailureType.TOO_EARLY
            elif reest_cell != last_target:
                failure_type = HOFailureType.WRONG_CELL
            else:
                failure_type = None  # Re-established back to same target

            # Update the stored HOEvent
            if 0 <= idx < len(self.ho_events):
                self.ho_events[idx].failure_type = failure_type
                self.ho_events[idx].reestablishment_cell = reest_cell

            self.logger.info(f"[DEFERRED_CLASSIFICATION] UE_{ue_id}: "
                            f"reest_cell={reest_cell}, failure_type="
                            f"{failure_type.name if failure_type else 'NONE'}")

        # === Case B: Too-Late post-re-establishment verification ===
        elif classification == 'TOO_LATE':
            expected_target = pending.get('expected_target')
            if 0 <= idx < len(self.ho_events):
                self.ho_events[idx].reestablishment_cell = reest_cell

            if expected_target is not None and reest_cell != expected_target:
                self.logger.warning(
                    f"[TOO_LATE_VERIFICATION] UE_{ue_id}: Expected re-establishment "
                    f"to target cell {expected_target}, but UE re-established to "
                    f"cell {reest_cell}. Classification remains TOO_LATE but "
                    f"re-establishment cell mismatch detected."
                )
            else:
                self.logger.info(
                    f"[TOO_LATE_VERIFICATION] UE_{ue_id}: Confirmed re-establishment "
                    f"to expected target cell {reest_cell}."
                )

    def _on_reestablishment_failure(self, ue_id: int, source_cell: Optional[int],
                                    cause: str):
        """Handle failed re-establishment (T311 expiry) - clears pending classification."""
        pending = self._pending_rlf_classification.pop(ue_id, None)
        if pending is not None:
            classification = pending.get('classification')
            if classification == 'PENDING_DEFERRED':
                self.logger.warning(
                    f"[DEFERRED_CLASSIFICATION] UE_{ue_id}: re-establishment failed "
                    f"(cause={cause}), clearing pending classification. "
                    f"Cannot determine Too-Early vs Wrong-Cell."
                )
            elif classification == 'TOO_LATE':
                self.logger.warning(
                    f"[TOO_LATE_VERIFICATION] UE_{ue_id}: re-establishment failed "
                    f"(cause={cause}), cannot verify Too-Late re-establishment cell. "
                    f"Classification remains TOO_LATE."
                )
        else:
            self.logger.warning(f"[REEST_FAILURE] UE_{ue_id}: source={source_cell}, cause={cause}")

    def _check_weak_cell(self, ue_id: int, new_cell: int) -> Optional[HOFailureType]:
        """
        Check for Weak Cell / Ping-Pong condition.

        Called when a new HO is initiated to classify the PREVIOUS cell stay.
        Returns HOFailureType.WEAK_CELL or PING_PONG if short stay detected.
        """
        history = self._ue_ho_history.get(ue_id, {})
        cell_entry_time = history.get('cell_entry_time')
        previous_cells = history.get('previous_cells', [])

        if cell_entry_time is None:
            return None

        time_of_stay_ms = (self._current_time - cell_entry_time) * 1000

        if time_of_stay_ms < self.config.short_stay_threshold_ms:
            # Short stay detected
            if previous_cells and new_cell == previous_cells[-1]:
                # Ping-pong: going back to previous cell
                self.logger.warning(f"[PING_PONG] UE_{ue_id}: "
                                  f"stay={time_of_stay_ms:.0f}ms < threshold")
                return HOFailureType.PING_PONG
            else:
                # Weak cell: short stay then move to different cell
                self.logger.warning(f"[WEAK_CELL] UE_{ue_id}: "
                                  f"stay={time_of_stay_ms:.0f}ms < threshold")
                return HOFailureType.WEAK_CELL

        return None

    # ═══════════════════════════════════════════════════════════════════
    # Results and Analysis
    # ═══════════════════════════════════════════════════════════════════

    def _compile_results(self) -> Dict:
        """Compile simulation results"""
        results = {
            'config': {
                'duration_s': self.config.duration_s,
                'measurement_period_ms': self.config.measurement_period_ms,
                'a3_offset_db': self.config.a3_offset_db,
                'ttt_ms': self.config.ttt_ms,
                'n310': self.config.n310,
                't310_ms': self.config.t310_ms
            },
            'statistics': self._compute_statistics(),
            'ho_events': [self._event_to_dict(e) for e in self.ho_events],
            'rlf_events': [self._rlf_to_dict(e) for e in self.rlf_events],
            'failure_classification': self._classify_all_failures()
        }

        return results

    def _compute_statistics(self) -> Dict:
        """Compute simulation statistics"""
        total_ho = sum(1 for e in self.ho_events if e.event_type == 'COMPLETE')
        failed_ho = sum(1 for e in self.ho_events if e.event_type == 'FAILURE')
        rlf_count = len(self.rlf_events)

        # HO durations
        ho_durations = [e.ho_duration_ms for e in self.ho_events
                       if e.event_type == 'COMPLETE' and e.ho_duration_ms]

        # Failure type counts
        failure_counts = {}
        for e in self.ho_events:
            if e.failure_type:
                name = e.failure_type.name
                failure_counts[name] = failure_counts.get(name, 0) + 1

        return {
            'total_handovers': total_ho,
            'successful_handovers': total_ho,
            'failed_handovers': failed_ho,
            'rlf_count': rlf_count,
            'ho_success_rate': total_ho / max(1, total_ho + failed_ho),
            'avg_ho_duration_ms': np.mean(ho_durations) if ho_durations else 0,
            'failure_classification': failure_counts
        }

    def _classify_all_failures(self) -> Dict:
        """Classify all failures with details"""
        classification = {
            'too_late': [],
            'too_early': [],
            'wrong_cell': [],
            'weak_cell': [],
            'ping_pong': []
        }

        for event in self.ho_events:
            if event.failure_type:
                entry = {
                    'timestamp': event.timestamp,
                    'ue_id': event.ue_id,
                    'source_cell': event.source_cell,
                    'target_cell': event.target_cell,
                    'time_since_ho_ms': event.time_since_ho_complete_ms
                }

                if event.failure_type == HOFailureType.TOO_LATE:
                    classification['too_late'].append(entry)
                elif event.failure_type == HOFailureType.TOO_EARLY:
                    classification['too_early'].append(entry)
                elif event.failure_type == HOFailureType.WRONG_CELL:
                    classification['wrong_cell'].append(entry)
                elif event.failure_type == HOFailureType.WEAK_CELL:
                    classification['weak_cell'].append(entry)
                elif event.failure_type == HOFailureType.PING_PONG:
                    classification['ping_pong'].append(entry)

        return classification

    def _event_to_dict(self, event: HOEvent) -> Dict:
        """Convert HOEvent to dictionary"""
        return {
            'timestamp': event.timestamp,
            'ue_id': event.ue_id,
            'source_cell': event.source_cell,
            'target_cell': event.target_cell,
            'event_type': event.event_type,
            'ho_duration_ms': event.ho_duration_ms,
            'failure_type': event.failure_type.name if event.failure_type else None,
            'failure_reason': event.failure_reason,
            't310_expired': event.t310_expired,
            't304_expired': event.t304_expired,
            'n310_count': event.n310_count,
            'bler': event.bler
        }

    def _rlf_to_dict(self, event: RLFEvent) -> Dict:
        """Convert RLFEvent to dictionary"""
        return {
            'timestamp': event.timestamp,
            'ue_id': event.ue_id,
            'cell_id': event.cell_id,
            'sinr_db': event.sinr_db,
            'bler': event.bler,
            'n310_count': event.n310_count,
            'time_since_ho_ms': event.time_since_ho_ms
        }

    def save_results(self, output_dir: str = 'output'):
        """Save simulation results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save HO events
        ho_file = output_path / f'ho_events_{timestamp}.csv'
        with open(ho_file, 'w', newline='') as f:
            if self.ho_events:
                writer = csv.DictWriter(f, fieldnames=self._event_to_dict(self.ho_events[0]).keys())
                writer.writeheader()
                for event in self.ho_events:
                    writer.writerow(self._event_to_dict(event))

        # Save RLF events
        rlf_file = output_path / f'rlf_events_{timestamp}.csv'
        with open(rlf_file, 'w', newline='') as f:
            if self.rlf_events:
                writer = csv.DictWriter(f, fieldnames=self._rlf_to_dict(self.rlf_events[0]).keys())
                writer.writeheader()
                for event in self.rlf_events:
                    writer.writerow(self._rlf_to_dict(event))

        # Save measurement traces
        trace_file = output_path / f'measurements_{timestamp}.csv'
        with open(trace_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'ue_id', 'x', 'y', 'z',
                           'serving_cell', 'rsrp_dbm', 'sinr_db', 'bler'])
            for trace in self.measurement_traces:
                writer.writerow([
                    trace.timestamp, trace.ue_id,
                    trace.position[0], trace.position[1], trace.position[2],
                    trace.serving_cell, trace.serving_rsrp_dbm,
                    trace.serving_sinr_db, trace.bler
                ])

        # Save summary JSON
        summary_file = output_path / f'summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(self._compile_results(), f, indent=2, default=str)

        self.logger.info(f"Results saved to {output_path}")

        return {
            'ho_events': str(ho_file),
            'rlf_events': str(rlf_file),
            'measurements': str(trace_file),
            'summary': str(summary_file)
        }

    def plot_results(self, output_dir: str = 'output'):
        """Generate visualization plots"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available, skipping plots")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if not self.measurement_traces:
            self.logger.warning("No measurement traces to plot")
            return

        # Extract data
        times = [t.timestamp for t in self.measurement_traces]
        sinr = [t.serving_sinr_db for t in self.measurement_traces]
        bler = [t.bler for t in self.measurement_traces]
        rsrp = [t.serving_rsrp_dbm for t in self.measurement_traces]
        cells = [t.serving_cell for t in self.measurement_traces]

        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Plot RSRP
        ax1 = axes[0]
        ax1.plot(times, rsrp, 'b-', linewidth=0.8)
        ax1.set_ylabel('RSRP (dBm)')
        ax1.set_title('NR Handover Simulation Results')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=-110, color='r', linestyle='--', alpha=0.5, label='Threshold')

        # Plot SINR
        ax2 = axes[1]
        ax2.plot(times, sinr, 'g-', linewidth=0.8)
        ax2.set_ylabel('SINR (dB)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=-8, color='r', linestyle='--', alpha=0.5, label='Qout')
        ax2.axhline(y=-6, color='orange', linestyle='--', alpha=0.5, label='Qin')

        # Plot BLER
        ax3 = axes[2]
        ax3.semilogy(times, [max(b, 1e-4) for b in bler], 'r-', linewidth=0.8)
        ax3.set_ylabel('BLER')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0.10, color='r', linestyle='--', alpha=0.5, label='Qout (10%)')
        ax3.axhline(y=0.02, color='orange', linestyle='--', alpha=0.5, label='Qin (2%)')
        ax3.set_ylim([1e-4, 1])

        # Plot serving cell
        ax4 = axes[3]
        ax4.step(times, cells, 'k-', linewidth=1.0, where='post')
        ax4.set_ylabel('Serving Cell')
        ax4.set_xlabel('Time (s)')
        ax4.grid(True, alpha=0.3)

        # Mark HO events
        for event in self.ho_events:
            if event.event_type == 'COMPLETE':
                for ax in axes:
                    ax.axvline(x=event.timestamp, color='green', linestyle='-',
                              alpha=0.3, linewidth=2)
            elif event.event_type == 'FAILURE' or event.failure_type:
                for ax in axes:
                    ax.axvline(x=event.timestamp, color='red', linestyle='-',
                              alpha=0.5, linewidth=2)

        plt.tight_layout()

        plot_file = output_path / f'sinr_bler_timeline_{timestamp}.png'
        plt.savefig(plot_file, dpi=150)
        plt.close()

        self.logger.info(f"Plot saved to {plot_file}")

        return str(plot_file)

    def print_summary(self):
        """Print simulation summary to console"""
        stats = self._compute_statistics()
        failures = self._classify_all_failures()

        print("\n" + "=" * 60)
        print("NR Handover Simulation Summary")
        print("=" * 60)

        print(f"\n[Configuration]")
        print(f"  Duration: {self.config.duration_s}s")
        print(f"  Measurement Period: {self.config.measurement_period_ms}ms")
        print(f"  A3 Offset: {self.config.a3_offset_db}dB")
        print(f"  TTT: {self.config.ttt_ms}ms")
        print(f"  N310/N311: {self.config.n310}/{self.config.n311}")
        print(f"  T310: {self.config.t310_ms}ms")

        print(f"\n[Statistics]")
        print(f"  Total Handovers: {stats['total_handovers']}")
        print(f"  Failed Handovers: {stats['failed_handovers']}")
        print(f"  RLF Count: {stats['rlf_count']}")
        print(f"  HO Success Rate: {stats['ho_success_rate']:.1%}")
        print(f"  Avg HO Duration: {stats['avg_ho_duration_ms']:.1f}ms")

        print(f"\n[Failure Classification]")
        print(f"  Too Late: {len(failures['too_late'])}")
        print(f"  Too Early: {len(failures['too_early'])}")
        print(f"  Wrong Cell: {len(failures['wrong_cell'])}")
        print(f"  Weak Cell: {len(failures['weak_cell'])}")
        print(f"  Ping-Pong: {len(failures['ping_pong'])}")

        print(f"\n[HO Events]")
        for event in self.ho_events[-10:]:  # Last 10 events
            failure_str = f" [{event.failure_type.name}]" if event.failure_type else ""
            print(f"  t={event.timestamp:.2f}s: {event.event_type} "
                  f"{event.source_cell}->{event.target_cell}{failure_str}")

        print("\n" + "=" * 60)


def run_test_simulation():
    """Run test simulation with specified parameters"""
    print("\n" + "=" * 60)
    print("NR Handover Simulation Test")
    print("=" * 60)

    # Configuration
    config = SimulationConfig(
        duration_s=30.0,
        measurement_period_ms=20.0,
        a3_offset_db=3.0,
        hysteresis_db=2.0,
        ttt_ms=256.0,
        n310=1,
        n311=1,
        t310_ms=1000.0
    )

    # Create simulator
    sim = NRHandoverSimulator(config)

    # Add gNBs: (0,100,30), (500,100,30), (1000,100,30)
    gnb_positions = [
        (0, 100, 30),
        (500, 100, 30),
        (1000, 100, 30)
    ]

    for i, pos in enumerate(gnb_positions, 1):
        sim.add_gnb(gnb_id=i, position=pos, azimuth_deg=180)

    # Add UE: (0,0,1.5) → (1200,0,1.5), 83.3 m/s (300 km/h)
    sim.add_ue(
        ue_id=0,
        initial_position=(0, 0, 1.5),
        serving_cell=1
    )

    # Set linear trajectory
    sim.set_linear_trajectory(
        ue_id=0,
        start_pos=(0, 0, 1.5),
        end_pos=(1200, 0, 1.5),
        speed_mps=83.33,  # 300 km/h
        start_time=0.0
    )

    # Run simulation
    print("\nRunning simulation...")
    results = sim.run()

    # Print summary
    sim.print_summary()

    # Save results
    output_files = sim.save_results('output')
    print(f"\nResults saved to: {output_files}")

    # Generate plots
    plot_file = sim.plot_results('output')
    if plot_file:
        print(f"Plot saved to: {plot_file}")

    return sim, results


if __name__ == "__main__":
    run_test_simulation()
