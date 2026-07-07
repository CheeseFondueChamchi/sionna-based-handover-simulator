# src/rrc/rrc_types.py
"""
State and Data Structure Definitions for Time-Continuous RRC State Machine

This module defines all state-related dataclasses and enums used by the
UE RRC Controller for tracking timers, counters, signaling, and procedures.

3GPP Reference: TS 38.331 (RRC Protocol)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class RRCState(Enum):
    """RRC Connection States per 3GPP TS 38.331"""
    RRC_IDLE = "RRC_IDLE"
    RRC_INACTIVE = "RRC_INACTIVE"
    RRC_CONNECTED = "RRC_CONNECTED"


class RadioLinkStatus(Enum):
    """Physical Layer Sync Status (3GPP TS 38.133 §8.1)"""
    IN_SYNC = "IN_SYNC"
    OUT_OF_SYNC = "OUT_OF_SYNC"
    GRAY_ZONE = "GRAY_ZONE"  # Qin < BLER < Qout: no L1 indication


class HOFType(Enum):
    """
    Handover Failure Classification per 3GPP TS 38.300 §15.5 (MRO).

    Used by network and simulation to classify connection failures
    for Mobility Robustness Optimization.
    """
    NONE = "NONE"                          # No failure
    TOO_LATE = "TOO_LATE"                  # Case 1: RLF before HO, re-estab to different cell
    TOO_EARLY = "TOO_EARLY"                # Case 2: RLF shortly after HO, re-estab to source
    WRONG_CELL = "WRONG_CELL"              # Case 3: RLF shortly after HO, re-estab to 3rd cell
    PING_PONG = "PING_PONG"               # Case 4: A→B→A within Tpp
    T304_EXPIRY = "T304_EXPIRY"            # Case 5: RACH failure, T304 expires


@dataclass
class TimerState:
    """
    Timer State for 3GPP timers (T300, T301, T304, T310, T311, T319)

    Tracks whether a timer is running, when it started, and when it expires.
    """
    running: bool = False
    start_time: float = 0.0
    expiry_time: float = 0.0
    duration: float = 0.0
    expired: bool = False

    def start(self, current_time: float, duration: float):
        """Start the timer with given duration"""
        self.running = True
        self.start_time = current_time
        self.duration = duration
        self.expiry_time = current_time + duration
        self.expired = False

    def stop(self):
        """Stop the timer"""
        self.running = False
        self.expired = False

    def check(self, current_time: float) -> bool:
        """
        Check if timer has expired.
        Returns True if expired, False otherwise.
        Note: Does not auto-stop; let the calling logic handle expiry.
        """
        if self.running and current_time >= self.expiry_time:
            self.expired = True
            return True
        return False

    def remaining(self, current_time: float) -> float:
        """Get remaining time in seconds, or 0 if not running/expired"""
        if not self.running or self.expired:
            return 0.0
        return max(0.0, self.expiry_time - current_time)


@dataclass
class CounterState:
    """
    Counter State for 3GPP counters (N310, N311)

    Tracks count and threshold, with reached flag.
    """
    count: int = 0
    threshold: int = 0
    reached: bool = False

    def increment(self):
        """Increment counter and check threshold"""
        self.count += 1
        if self.count >= self.threshold:
            self.reached = True

    def reset(self):
        """Reset counter to zero"""
        self.count = 0
        self.reached = False


@dataclass
class HOHistoryEntry:
    """
    Record of a single handover event for HOF classification.

    Stored in a rolling buffer so the classifier can look back at
    recent HO activity when an RLF occurs.

    3GPP Reference: TS 38.300 §15.5 — Tstore_UE_cntxt determines
    whether a failure is "shortly after" a previous HO.
    """
    timestamp: float = 0.0          # Time HO completed (or failed)
    source_cell_id: int = -1        # Cell the UE was leaving
    target_cell_id: int = -1        # Cell the UE was going to
    success: bool = True            # Whether HO completed successfully
    ho_type: str = ""               # "A3_intra", "A5_inter", etc.
    rach_attempts: int = 0          # Number of RACH attempts
    duration_ms: float = 0.0        # Total HO duration
    failure_reason: str = ""        # "T304_EXPIRE", etc. (if failed)


@dataclass
class HOFClassificationResult:
    """
    Result of HOF classification for a single failure event.

    Produced by the classifier when an RLF or connection failure
    is detected.  Attached to the event log for MRO analysis.
    """
    hof_type: 'HOFType' = None          # Classification result
    timestamp: float = 0.0              # When the failure was classified
    rlf_cell_id: int = -1               # Cell where RLF occurred
    reestablishment_cell_id: int = -1   # Cell where UE re-established
    last_ho_source: int = -1            # Source of last HO (if any)
    last_ho_target: int = -1            # Target of last HO (if any)
    time_since_last_ho_ms: float = -1   # Time between last HO and RLF
    cause: str = ""                     # Human-readable cause string

    def __post_init__(self):
        """Default `hof_type` to HOFType.NONE when caller passed None.

        Side effects:
            Mutates `self.hof_type` from None → HOFType.NONE so downstream
            classifier code can safely compare against enum values without
            None-guards.
        """
        if self.hof_type is None:
            self.hof_type = HOFType.NONE


@dataclass
class SignalingState:
    """
    Signaling State for the current timestep.

    Flags indicating if a message was sent/received in the current step.
    Reset at the beginning of each update cycle.
    """
    # Initial Access
    rach_preamble_sent: bool = False
    rar_received: bool = False

    # RRC Setup
    rrc_setup_request: bool = False
    rrc_setup: bool = False
    rrc_setup_complete: bool = False

    # Handover (RRC Reconfiguration)
    rrc_reconfiguration: bool = False  # HO Command received
    rrc_reconfiguration_complete: bool = False  # HO Complete sent

    # Re-establishment
    rrc_reestablishment_request: bool = False
    rrc_reestablishment: bool = False
    rrc_reestablishment_complete: bool = False

    # Release
    rrc_release: bool = False


@dataclass
class EventState:
    """
    State for a specific measurement event (A2, A3, A5)

    Tracks whether event condition is met, TTT progress, and report status.
    """
    event_type: str = ""
    triggered: bool = False  # Entering condition met
    time_to_trigger_remaining: float = 0.0  # Remaining TTT in ms
    report_sent: bool = False  # TTT expired, report generated
    target_cell_id: Optional[int] = None  # For A3/A5: target cell
    quantity: float = -140.0  # RSRP/RSRQ value


@dataclass
class UEState:
    """
    Complete UE State for Time-Continuous State Machine.

    This dataclass holds all state information needed to track a UE's
    RRC procedures, timers, counters, and pending operations across
    simulation timesteps.
    """
    # 1. Context & RRC State
    ue_id: int = 0
    current_time: float = 0.0
    rrc_state: RRCState = RRCState.RRC_CONNECTED
    rrc_connected: bool = True
    serving_cell_id: int = -1
    target_cell_id: Optional[int] = None

    # 2. Physical Layer Status
    radio_link_status: RadioLinkStatus = RadioLinkStatus.IN_SYNC
    rlf_declared: bool = False
    ho_in_progress: bool = False

    # 3. Signaling (Instantaneous for this step)
    signaling: SignalingState = field(default_factory=SignalingState)

    # 4. Measurement Events Status
    measurement_events: Dict[str, EventState] = field(default_factory=dict)

    # 5. Timers (3GPP TS 38.331)
    timers: Dict[str, TimerState] = field(default_factory=lambda: {
        "T300": TimerState(),  # RRC Setup
        "T301": TimerState(),  # RRC Re-establishment
        "T304": TimerState(),  # Handover execution
        "T310": TimerState(),  # RLF detection
        "T312": TimerState(),  # Fast RLF when report triggered while T310 running (TS 38.331 §5.5.4)
        "T311": TimerState(),  # RLF recovery (cell selection)
        "T319": TimerState(),  # RRC Resume
    })

    # 6. Counters (3GPP TS 38.331)
    counters: Dict[str, CounterState] = field(default_factory=lambda: {
        "N310": CounterState(threshold=10),  # Out-of-sync -> T310 start
        "N311": CounterState(threshold=1),   # In-sync -> T310 stop
    })

    # 7. Internal / Pending Procedures
    pending_procedure: Optional[str] = None
    pending_context: Dict[str, Any] = field(default_factory=dict)

    # 8. Statistics
    ho_count: int = 0

    # 9. Per-tick visualization/event flags (cleared each update())
    # When T310 is running and serving UL_SINR fails (< -8 dB or NaN), the
    # UE's measurement report cannot reach the serving gNB. UEStateMachine
    # sets this to the suppressed HO target's cell_id so the sim driver
    # (run_simulation.py) can emit a SIB_READ_FAILURE event (matching the
    # field log's bidirectional-signaling-failure class).
    sib_block_blocked_target: Optional[int] = None
    # Counter (10ms ticks) of sustained UL block ticks during current T310
    # epoch where a viable HO target was suppressed. Used to differentiate
    # true SIB_READ_FAILURE (bidirectional UL signaling fail) from pure
    # RLM RLF (DL Qout countdown). Reset on HO_COMPLETE / T310 stop /
    # RLF declare. Threshold for SIB classification = 5 ticks (50ms).
    sib_block_ticks_during_t310: int = 0
    # Sticky UL msg3 block state. Once UE enters (RSRQ < -17.8 AND RSRP < -93),
    # this stays True until an external event resets it (HO_COMPLETE, RLF
    # declare, re-establishment). Per spec: UL channel degradation does not
    # auto-recover from a single-tick RSRQ improvement. N310/T310 progresses
    # naturally in parallel; T310 expiry → RLF (classified as SIB_READ_FAILURE).
    ul_block_active: bool = False
    # N-consecutive-tick entry counter (TS 38.133 §8.5.2.2 spec-aligned).
    # Counts consecutive ticks meeting the joint (RSRQ, RSRP) entry condition.
    # Resets to 0 on any good tick OR when ul_block_active flips True. The
    # 3-tick (=30 ms) requirement filters single-sample noise; real L1 averages
    # over T_evaluate_out_DL=200 ms before reacting.
    ul_block_pending_ticks: int = 0
    # Timestamp (s) when radio_link_status became IN_SYNC while ul_block_active.
    # Used to release the block after sustained IN_SYNC ≥ T_evaluate_in_DL_SYNC
    # (~200 ms). None when not blocked OR not currently IN_SYNC.
    ul_block_in_sync_start_t: Optional[float] = None
    # Consecutive good ticks toward UL-block release (hysteresis). Reset to 0
    # on a bad tick. Set/read dynamically by UEStateMachine.
    ul_block_good_ticks: int = 0
    # Pending UL-blocked HO report (2026-06-12). When an A3 report is generated
    # but the UL is blocked, the measurement layer must NOT regenerate it every
    # tick (TS 38.331 §5.5.5 reportInterval governs report generation). Instead
    # the generated report is held here and the DELIVERY layer retries it every
    # tick — delivering the HO the instant UL recovers, with NO per-tick report
    # spam. Tracks the current best A3 target while blocked; None when not
    # pending. Cleared on delivery / serving change / target no longer valid.
    ul_pending_ho_target: Optional[int] = None
    ul_pending_ho_type: str = ""
    # --- Observability-only diagnostics (NEVER gate any FSM decision) ---
    # Which UL-block path latched (\"A\"/\"B\"), for the gate-reason read-out.
    ul_block_path: str = ""
    # UL-block gate read-out (2026-06-12): the RSRQ value the gate actually
    # evaluated (the L3-FILTERED serving RSRQ — NOT the raw per-tick RSRQ) and
    # the path-B threshold it was compared against. Logged so analysts can see
    # exactly why the block engaged when the RAW serving RSRQ looks healthy.
    ul_block_applied_rsrq: Optional[float] = None
    ul_block_applied_rsrp: Optional[float] = None
    ul_block_threshold_rsrq_db: Optional[float] = None
    # RLM (radio-link monitoring) OOS/IS read-out (2026-06-12): the smoothed
    # serving SINR fed to the hypothetical-PDCCH BLER curve and the resulting
    # Qout/Qin BLERs, so the IN_SYNC/OUT_OF_SYNC verdict (which arms/clears
    # T310) is fully reconstructable from the log.
    rlm_smoothed_sinr_db: Optional[float] = None
    rlm_bler_qout: Optional[float] = None
    rlm_bler_qin: Optional[float] = None
    # UE ground speed (km/h) at this tick — for L3-filter-vs-speed analysis
    # (a fast UE crossing cells can trip A3 before the L3 filter settles).
    ue_velocity_kmh: Optional[float] = None
    # Per-tick read-out of WHY a viable HO was not triggered this step:
    # \"\"=not suppressed, UL_BLOCK_PATHA/UL_BLOCK_PATHB, SIB_BLOCK, T310_RUNNING,
    # HO_IN_PROGRESS. Set after the HO-decision is made; surfaced into
    # detailed_log so RLF-window analysis no longer needs an INFO re-run.
    ho_suppress_reason: str = ""
    rlf_count: int = 0
    reestablishment_count: int = 0
    rrc_setup_count: int = 0  # Fresh RRC Connection Setup count (IDLE recovery)
    ho_type: str = ""  # Last HO type: "Intra-Freq", "Inter-Freq", "Inter-RAT-B2", "Inter-RAT-B1"

    # --- Explicit staged HO signaling flow (2026-06-11; observability + S4 gate) ---
    # Tracks the per-message HO chain as explicit, verifiable stages so each
    # handover step in detailed_log can be inspected. These NEVER alter any
    # 3GPP FSM decision other than the S4 target-DL RRC-config delivery gate
    # (a vendor delivery abstraction like the S2 HO-command gate; default ON,
    # no-op at healthy target SINR). Stage transitions:
    #   "" → A3_REPORTED (S1: report passed UL gate, pending_ho set)
    #      → HO_CMD_RX   (S2: serving-DL HO command decoded, commit)
    #      → TARGET_RACH (S3: RACH to target succeeded)
    #      → COMPLETE     (S4: target-DL RRC config delivered → cell switch)
    # Reset to ""/False on HO complete, RLF declare, and re-establishment.
    ho_stage: str = ""
    ho_cmd_decoded: bool = False   # S2 result this attempt
    target_rach_ok: bool = False   # S3 result this attempt
    target_rrc_ok: bool = False    # S4 result this attempt

    # 9. HOF Classification (3GPP TS 38.300 §15.5 MRO)
    #    Tstore_UE_cntxt: network-configured threshold (ms) to determine
    #    whether a failure happened "shortly after" a HO.
    #    Tpp: ping-pong threshold (ms) for detecting A→B→A oscillation.
    tstore_ue_cntxt_ms: float = 1000.0   # Default 1s per 3GPP
    tpp_ms: float = 1000.0               # Ping-pong window, typically 1s

    # Rolling HO history (last N handovers for classification)
    ho_history: List[Any] = field(default_factory=list)       # List[HOHistoryEntry]
    ho_history_max: int = 20

    # HOF classification results for this UE
    hof_classifications: List[Any] = field(default_factory=list)  # List[HOFClassificationResult]

    # Last classified HOF (for signaling in current step)
    last_hof_classification: Optional[Any] = None  # HOFClassificationResult or None
