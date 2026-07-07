"""
PHY Abstraction Layer for NR Handover Simulation

This module provides SINR to BLER mapping using 3GPP link-level curves.
The BLER is critical for:
1. Link Adaptation (MCS selection)
2. RLF Detection (Out-of-sync / In-sync indication)
3. HARQ performance modeling

Based on 3GPP TS 38.214, 38.133
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MCSTable(Enum):
    """MCS Tables from 3GPP TS 38.214"""
    TABLE1 = "table1"  # Table 5.1.3.1-1 (64QAM)
    TABLE2 = "table2"  # Table 5.1.3.1-2 (256QAM)
    TABLE3 = "table3"  # Table 5.1.3.1-3 (64QAM, low SE)


@dataclass
class MCSEntry:
    """MCS table entry"""
    index: int
    modulation_order: int  # Qm: 2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM
    target_code_rate: float  # R x 1024
    spectral_efficiency: float  # bits/s/Hz
    
    @property
    def modulation_name(self) -> str:
        """Return the human-readable modulation label for this MCS entry.

        Returns:
            One of "QPSK", "16QAM", "64QAM", "256QAM", or "Unknown" derived
            from `self.modulation_order` (bits/symbol).
        """
        mapping = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}
        return mapping.get(self.modulation_order, "Unknown")


# 3GPP TS 38.214 Table 5.1.3.1-1: MCS index table 1 for PDSCH
MCS_TABLE_1 = [
    MCSEntry(0,  2, 120,  0.2344),
    MCSEntry(1,  2, 157,  0.3066),
    MCSEntry(2,  2, 193,  0.3770),
    MCSEntry(3,  2, 251,  0.4902),
    MCSEntry(4,  2, 308,  0.6016),
    MCSEntry(5,  2, 379,  0.7402),
    MCSEntry(6,  2, 449,  0.8770),
    MCSEntry(7,  2, 526,  1.0273),
    MCSEntry(8,  2, 602,  1.1758),
    MCSEntry(9,  2, 679,  1.3262),
    MCSEntry(10, 4, 340,  1.3281),
    MCSEntry(11, 4, 378,  1.4766),
    MCSEntry(12, 4, 434,  1.6953),
    MCSEntry(13, 4, 490,  1.9141),
    MCSEntry(14, 4, 553,  2.1602),
    MCSEntry(15, 4, 616,  2.4063),
    MCSEntry(16, 4, 658,  2.5703),
    MCSEntry(17, 6, 438,  2.5664),
    MCSEntry(18, 6, 466,  2.7305),
    MCSEntry(19, 6, 517,  3.0293),
    MCSEntry(20, 6, 567,  3.3223),
    MCSEntry(21, 6, 616,  3.6094),
    MCSEntry(22, 6, 666,  3.9023),
    MCSEntry(23, 6, 719,  4.2129),
    MCSEntry(24, 6, 772,  4.5234),
    MCSEntry(25, 6, 822,  4.8164),
    MCSEntry(26, 6, 873,  5.1152),
    MCSEntry(27, 6, 910,  5.3320),
    MCSEntry(28, 6, 948,  5.5547),
]


class PHYAbstraction:
    """
    PHY Abstraction for SINR to BLER mapping.

    Delegates to BLERCalculator (single source of truth) while
    preserving the existing API for backward compatibility.

    Uses AWGN BLER curves approximated by sigmoid functions:
    BLER(SINR) = 1 / (1 + exp(a * (SINR - SINR_10%)))

    Where SINR_10% is the SINR required for 10% BLER.
    """

    # Keep class-level constants for backward compatibility
    SINR_10_PERCENT_BLER = None  # Populated from BLERCalculator
    BLER_SLOPE = 0.7  # Reference only; actual computation in BLERCalculator

    def __init__(self, mcs_table: MCSTable = MCSTable.TABLE1):
        """Initialise PHY abstraction with an MCS table and BLER calculator.

        Args:
            mcs_table: which 3GPP TS 38.214 MCS table to use; default
                MCSTable.TABLE1 (PDSCH index table 1).

        Side effects:
            - Imports and instantiates `BLERCalculator`.
            - Backfills class-level `PHYAbstraction.SINR_10_PERCENT_BLER`
              from the calculator for legacy callers.
        """
        from .bler_calculator import BLERCalculator
        self.mcs_table = mcs_table
        self.mcs_entries = MCS_TABLE_1 if mcs_table == MCSTable.TABLE1 else MCS_TABLE_1
        self._bler_calculator = BLERCalculator()

        # Expose the canonical SINR table for backward compatibility
        PHYAbstraction.SINR_10_PERCENT_BLER = self._bler_calculator.SINR_10_PERCENT_BLER

        logger.info(f"PHY Abstraction initialized with {mcs_table.value}")

    def sinr_to_bler(self, sinr_db: float, mcs_index: int) -> float:
        """
        Map SINR to BLER for a given MCS.

        Delegates to BLERCalculator (single source of truth).

        Args:
            sinr_db: Effective SINR in dB
            mcs_index: MCS index (0-28)

        Returns:
            BLER (0.0 to 1.0)
        """
        return self._bler_calculator.sinr_to_bler(sinr_db, mcs_index)
    
    def sinr_to_bler_adaptive(self, sinr_db: float) -> Tuple[float, int]:
        """
        BLER with adaptive MCS (AMC): pick highest MCS where BLER <= 10%.

        Returns:
            (bler, selected_mcs) tuple
        """
        return self._bler_calculator.sinr_to_bler_adaptive(sinr_db)

    def get_max_mcs_for_target_bler(self, sinr_db: float,
                                     target_bler: float = 0.1) -> int:
        """
        Find maximum MCS that achieves target BLER.
        
        This is the core of Link Adaptation.
        
        Args:
            sinr_db: Effective SINR in dB
            target_bler: Target BLER (default 10%)
            
        Returns:
            Maximum MCS index
        """
        max_mcs = 0
        
        for mcs_index in range(len(self.mcs_entries)):
            bler = self.sinr_to_bler(sinr_db, mcs_index)
            if bler <= target_bler:
                max_mcs = mcs_index
            else:
                break
        
        return max_mcs
    
    def get_throughput(self, sinr_db: float, mcs_index: int,
                       bandwidth_hz: float = 20e6,
                       num_prbs: int = 273) -> float:
        """
        Calculate expected throughput considering BLER.
        
        Throughput = SE * BW * (1 - BLER)
        
        Args:
            sinr_db: Effective SINR in dB
            mcs_index: MCS index
            bandwidth_hz: System bandwidth in Hz
            num_prbs: Number of PRBs
            
        Returns:
            Throughput in bits/s
        """
        if mcs_index >= len(self.mcs_entries):
            mcs_index = len(self.mcs_entries) - 1
        
        mcs = self.mcs_entries[mcs_index]
        bler = self.sinr_to_bler(sinr_db, mcs_index)
        
        # Effective throughput
        throughput = mcs.spectral_efficiency * bandwidth_hz * (1 - bler)
        
        return throughput


class SyncIndicator:
    """
    In-sync / Out-of-sync Indicator for RLF Detection.
    
    Based on 3GPP TS 38.133 Section 8.1.
    
    The physical layer provides the following indications to higher layers:
    - Out-of-sync: When the estimated quality is worse than Qout
    - In-sync: When the estimated quality is better than Qin
    
    Qout and Qin can be defined in terms of:
    - BLER (typical: Qout = 10% BLER, Qin = 2% BLER)
    - SINR (alternative threshold-based approach)
    """
    
    def __init__(self, 
                 qout_bler: float = 0.10,
                 qin_bler: float = 0.02,
                 qout_sinr_db: float = -8.0,
                 qin_sinr_db: float = -6.0,
                 use_bler: bool = True):
        """
        Initialize sync indicator.
        
        Args:
            qout_bler: Out-of-sync BLER threshold (default 10%)
            qin_bler: In-sync BLER threshold (default 2%)
            qout_sinr_db: Out-of-sync SINR threshold
            qin_sinr_db: In-sync SINR threshold
            use_bler: Use BLER-based detection (True) or SINR-based (False)
        """
        self.qout_bler = qout_bler
        self.qin_bler = qin_bler
        self.qout_sinr_db = qout_sinr_db
        self.qin_sinr_db = qin_sinr_db
        self.use_bler = use_bler
        
        self.phy_abstraction = PHYAbstraction()
        
        logger.info(f"SyncIndicator initialized: "
                   f"Qout_BLER={qout_bler}, Qin_BLER={qin_bler}, "
                   f"use_bler={use_bler}")
    
    def evaluate(self, sinr_db: float, mcs_index: int = 0) -> Tuple[bool, bool, float]:
        """
        Evaluate one per-sample sync verdict (3GPP TS 38.133 §8.5.2.2 Qout/Qin).

        Qout / Qin are the radio-link quality thresholds the RLM watches on the
        SERVING cell's downlink CONTROL channel (PDCCH): if you cannot decode
        PDCCH you cannot be scheduled, so the link is effectively down. The spec
        defines them on a *hypothetical* PDCCH: Qout = the quality giving 10%
        BLER (link unusable), Qin = the quality giving 2% BLER (link usably
        recovered). The 10%>2% gap is hysteresis to stop OOS/IS chattering.

        PDCCH is NOT modelled structurally here (no CCE/aggregation/DCI/blind
        decode). It is approximated by a PDCCH-like BLER curve: mcs_index=0
        (QPSK, lowest code rate ≈ rate 120/1024) fed through PHYAbstraction's
        AWGN sigmoid BLER mapping. On that curve Qout(10%) ≈ SINR -6.7 dB and
        Qin(2%) ≈ -3 dB.

        OPERATIONAL DEFAULTS (use_bler path — the production/CSV mode): spec
        literals qout_bler=0.10, qin_bler=0.02, fed by the field-measured
        serving SINR. The SINR-threshold path below (qout_sinr/qin_sinr) is the
        legacy RSRP/SINR-only alternative; CSV mode uses BLER (use_sinr_for_rlf
        → use_bler=True), so keep qout_bler/qin_bler at the spec 10%/2%.

        NOTE: this returns the PER-SAMPLE (per 10 ms tick) verdict only. It is
        NOT an OOS/IS indication on its own — RLFDetector.update() accumulates
        these samples into the 200 ms (Qout) / 100 ms (Qin) T_Evaluate windows
        and emits ONE indication per window by majority vote.

        Args:
            sinr_db: Current SINR in dB
            mcs_index: Current MCS index (for BLER calculation; RLM uses 0)

        Returns:
            Tuple of (is_out_of_sync, is_in_sync, bler)
            Note: Both can be False (in "gray zone")
        """
        if self.use_bler:
            # BLER-based evaluation (spec-aligned; production default). OOS when
            # PDCCH-proxy BLER > Qout(10%), IS when < Qin(2%).
            bler = self.phy_abstraction.sinr_to_bler(sinr_db, mcs_index)

            is_out_of_sync = bler > self.qout_bler
            is_in_sync = bler < self.qin_bler

        else:
            # SINR-based evaluation
            bler = self.phy_abstraction.sinr_to_bler(sinr_db, mcs_index)
            
            is_out_of_sync = sinr_db < self.qout_sinr_db
            is_in_sync = sinr_db > self.qin_sinr_db
        
        return is_out_of_sync, is_in_sync, bler


@dataclass
class RLFState:
    """Radio Link Failure detection state"""
    n310_counter: int = 0          # Out-of-sync counter
    n311_counter: int = 0          # In-sync counter
    t310_running: bool = False     # T310 timer status
    rlf_detected: bool = False     # RLF has been detected
    last_bler: float = 0.0         # Last BLER value
    last_sinr_db: float = 0.0      # Last SINR value
    consecutive_oos: int = 0       # Consecutive out-of-sync indications
    consecutive_is: int = 0        # Consecutive in-sync indications


class RLFDetector:
    """
    Radio Link Failure Detector.
    
    Implements 3GPP TS 38.331 Section 5.3.10 (RLF detection and recovery)
    
    RLF Detection Procedure:
    1. PHY provides out-of-sync (OOS) indications when BLER > Qout
    2. When N310 consecutive OOS indications received -> Start T310
    3. If N311 in-sync indications received before T310 expires -> Stop T310, reset N310
    4. If T310 expires -> Declare RLF
    
    The N310/N311 counters and T310 timer work together:
    - N310: Number of OOS indications to start T310
    - N311: Number of IS indications to stop T310
    - T310: RLF detection timer
    """
    
    def __init__(self,
                 n310: int = 1,
                 n311: int = 1,
                 t310_ms: float = 1000,
                 qout_bler: float = 0.10,
                 qin_bler: float = 0.02,
                 scheduler: Optional['EventScheduler'] = None,
                 t_evaluate_out_ms: float = 200.0,
                 t_evaluate_in_ms: float = 100.0):
        """
        Initialize RLF detector.

        Args:
            n310: Out-of-sync indication threshold [1,2,3,4,6,8,10,20]
            n311: In-sync indication threshold [1,2,3,4,5,6,8,10]
            t310_ms: RLF detection timer duration [0,50,100,200,500,1000,2000]ms
            qout_bler: Out-of-sync BLER threshold
            qin_bler: In-sync BLER threshold
            scheduler: Event scheduler for T310 timer
            t_evaluate_out_ms: OOS evaluation window per 3GPP TS 38.133 §8.5.2.2
                (default 200 ms for FR1 SSB-based RLM, no DRX). One OOS
                indication is emitted per window when windowed quality < Qout.
            t_evaluate_in_ms: IS evaluation window (default 100 ms).
        """
        # Validate 3GPP values
        valid_n310 = [1, 2, 3, 4, 6, 8, 10, 20]
        valid_n311 = [1, 2, 3, 4, 5, 6, 8, 10]
        valid_t310 = [0, 50, 100, 200, 500, 1000, 2000]
        
        if n310 not in valid_n310:
            logger.warning(f"N310={n310} not in 3GPP valid set {valid_n310}")
        if n311 not in valid_n311:
            logger.warning(f"N311={n311} not in 3GPP valid set {valid_n311}")
        if t310_ms not in valid_t310:
            logger.warning(f"T310={t310_ms}ms not in 3GPP valid set {valid_t310}")
        
        self.n310_threshold = n310
        self.n311_threshold = n311
        self.t310_ms = t310_ms
        self.scheduler = scheduler

        # 3GPP RLM evaluation windows. The L1 emits AT MOST one IS/OOS
        # indication per evaluation period — N310/N311 increment per
        # indication, NOT per measurement sample. Enforcing this here
        # makes the RLM behaviour invariant to the simulator's step size.
        self.t_evaluate_out_ms = t_evaluate_out_ms
        self.t_evaluate_in_ms = t_evaluate_in_ms
        self._oos_window_n_oos = 0
        self._oos_window_n_total = 0
        self._oos_window_elapsed_ms = 0.0
        self._is_window_n_is = 0
        self._is_window_n_total = 0
        self._is_window_elapsed_ms = 0.0
        self._last_eval_time_s: Optional[float] = None
        
        # Sync indicator
        self.sync_indicator = SyncIndicator(
            qout_bler=qout_bler,
            qin_bler=qin_bler,
            use_bler=True
        )
        
        # State
        self.state = RLFState()
        
        # Timer
        self._t310_event: Optional['Event'] = None
        
        # Callbacks
        self.on_t310_start: Optional[callable] = None
        self.on_t310_stop: Optional[callable] = None
        self.on_rlf_detected: Optional[callable] = None
        
        logger.info(f"RLFDetector initialized: N310={n310}, N311={n311}, "
                   f"T310={t310_ms}ms, Qout={qout_bler}, Qin={qin_bler}")
    
    def process_measurement(self, sinr_db: float, mcs_index: int = 0,
                            suppress_t310: bool = False,
                            current_time_s: Optional[float] = None) -> RLFState:
        """
        Process a single measurement and update RLF state.

        Called once per simulator step. Per 3GPP TS 38.133 §8.5.2.2 the L1
        emits an OOS indication once per `t_evaluate_out_ms` window (default
        200 ms) and an IS indication once per `t_evaluate_in_ms` (default
        100 ms). Per-step samples are accumulated; an indication is emitted
        only when its window completes, decided by the majority of OOS/IS
        samples observed in that window.

        Args:
            sinr_db: Current SINR in dB
            mcs_index: Current MCS index
            suppress_t310: If True, T310 will not be started even if N310 is reached.
                Per 3GPP TS 38.331: T310 must not start while T304/T300/T301/T311 is running.
            current_time_s: Current simulator wall-clock in seconds. Used to
                compute the elapsed time accumulated into the OOS / IS
                evaluation windows. When None, dt is assumed 20 ms (legacy).

        Returns:
            Updated RLF state
        """
        # Skip if RLF already detected
        if self.state.rlf_detected:
            return self.state

        # Evaluate sync status (per-sample)
        is_oos, is_is, bler = self.sync_indicator.evaluate(sinr_db, mcs_index)

        # Update state
        self.state.last_sinr_db = sinr_db
        self.state.last_bler = bler

        # Compute elapsed since last call
        if current_time_s is not None and self._last_eval_time_s is not None:
            dt_ms = max(0.0, (current_time_s - self._last_eval_time_s) * 1000.0)
        else:
            dt_ms = 20.0  # legacy default
        if current_time_s is not None:
            self._last_eval_time_s = current_time_s

        # Accumulate sample into OOS evaluation window (3GPP TS 38.133 §8.5.2.2)
        self._oos_window_n_total += 1
        if is_oos:
            self._oos_window_n_oos += 1
        self._oos_window_elapsed_ms += dt_ms
        if self._oos_window_elapsed_ms >= self.t_evaluate_out_ms:
            # Windowed quality < Qout if majority of samples were OOS
            if self._oos_window_n_total > 0 and \
                    self._oos_window_n_oos * 2 >= self._oos_window_n_total:
                self._handle_out_of_sync(suppress_t310=suppress_t310)
            self._oos_window_n_oos = 0
            self._oos_window_n_total = 0
            self._oos_window_elapsed_ms = 0.0

        # Accumulate sample into IS evaluation window
        self._is_window_n_total += 1
        if is_is:
            self._is_window_n_is += 1
        self._is_window_elapsed_ms += dt_ms
        if self._is_window_elapsed_ms >= self.t_evaluate_in_ms:
            if self._is_window_n_total > 0 and \
                    self._is_window_n_is * 2 >= self._is_window_n_total:
                self._handle_in_sync()
            self._is_window_n_is = 0
            self._is_window_n_total = 0
            self._is_window_elapsed_ms = 0.0

        return self.state
    
    def _handle_out_of_sync(self, suppress_t310: bool = False):
        """Handle out-of-sync indication.

        Args:
            suppress_t310: If True, T310 will not be started even if N310 threshold
                is reached. Per 3GPP TS 38.331: T310 must not start while
                T304/T300/T301/T311/T316 is running.
        """
        self.state.consecutive_oos += 1
        self.state.consecutive_is = 0

        if not self.state.t310_running:
            # T310 not running - count towards N310
            self.state.n310_counter += 1

            logger.debug(f"OOS indication: N310 counter = {self.state.n310_counter}/{self.n310_threshold}")

            if self.state.n310_counter >= self.n310_threshold:
                if suppress_t310:
                    logger.debug("T310 start suppressed (guard timer running, e.g. T304)")
                else:
                    # Start T310
                    self._start_t310()
        else:
            # T310 running - reset N311 counter
            self.state.n311_counter = 0
    
    def _handle_in_sync(self):
        """Handle in-sync indication"""
        self.state.consecutive_is += 1
        self.state.consecutive_oos = 0
        
        if self.state.t310_running:
            # T310 running - count towards N311
            self.state.n311_counter += 1
            
            logger.debug(f"IS indication: N311 counter = {self.state.n311_counter}/{self.n311_threshold}")
            
            if self.state.n311_counter >= self.n311_threshold:
                # Stop T310, reset counters
                self._stop_t310()
                self.state.n310_counter = 0
                self.state.n311_counter = 0
    
    def _start_t310(self):
        """Start T310 timer"""
        if self.state.t310_running:
            return
        
        self.state.t310_running = True
        self.state.n311_counter = 0
        
        logger.info(f"T310 started ({self.t310_ms}ms) - potential RLF")
        
        if self.on_t310_start:
            self.on_t310_start()
        
        # Schedule T310 expiry
        if self.scheduler:
            import sys
            from pathlib import Path
            _src = Path(__file__).parent.parent
            if str(_src) not in sys.path:
                sys.path.insert(0, str(_src))
            from core.event_scheduler import EventType
            self._t310_event = self.scheduler.schedule(
                delay=self.t310_ms / 1000.0,
                event_type=EventType.T310_EXPIRE,
                callback=self._on_t310_expire,
                description="T310 RLF detection"
            )
    
    def _stop_t310(self):
        """Stop T310 timer (recovered from potential RLF)"""
        if not self.state.t310_running:
            return
        
        self.state.t310_running = False
        
        logger.info("T310 stopped - recovered from potential RLF")
        
        if self.on_t310_stop:
            self.on_t310_stop()
        
        # Cancel T310 event
        if self._t310_event and self.scheduler:
            self.scheduler.cancel(self._t310_event)
            self._t310_event = None
    
    def _on_t310_expire(self):
        """T310 timer expired - RLF detected"""
        self.state.t310_running = False
        self.state.rlf_detected = True
        self._t310_event = None
        
        logger.warning(f"T310 expired - RLF DETECTED! "
                      f"Last SINR={self.state.last_sinr_db:.1f}dB, "
                      f"Last BLER={self.state.last_bler:.2%}")
        
        if self.on_rlf_detected:
            self.on_rlf_detected()
    
    def reset(self):
        """Reset RLF detector state (e.g., after HO or re-establishment)"""
        if self._t310_event and self.scheduler:
            self.scheduler.cancel(self._t310_event)

        self.state = RLFState()
        self._t310_event = None

        # Reset 3GPP evaluation windows so the next IS/OOS indication is
        # emitted exactly t_evaluate_*_ms after the reset, never sooner.
        self._oos_window_n_oos = 0
        self._oos_window_n_total = 0
        self._oos_window_elapsed_ms = 0.0
        self._is_window_n_is = 0
        self._is_window_n_total = 0
        self._is_window_elapsed_ms = 0.0
        self._last_eval_time_s = None

        logger.debug("RLF detector reset")
    
    def get_state(self) -> Dict:
        """Get current RLF state as dictionary"""
        return {
            'n310_counter': self.state.n310_counter,
            'n311_counter': self.state.n311_counter,
            't310_running': self.state.t310_running,
            'rlf_detected': self.state.rlf_detected,
            'last_bler': self.state.last_bler,
            'last_sinr_db': self.state.last_sinr_db,
            'consecutive_oos': self.state.consecutive_oos,
            'consecutive_is': self.state.consecutive_is
        }


class EESM:
    """
    Exponential Effective SINR Mapping (EESM)
    
    Maps per-subcarrier SINRs to a single effective SINR.
    Used for PHY abstraction in multi-carrier systems.
    
    SINR_eff = -β * ln(1/N * Σ exp(-SINR_n / β))
    
    Where β is the tuning parameter that depends on MCS.
    """
    
    # Beta values per modulation order (approximate)
    BETA_VALUES = {
        2: 1.49,   # QPSK
        4: 3.36,   # 16QAM
        6: 8.30,   # 64QAM
        8: 17.6,   # 256QAM
    }
    
    def __init__(self):
        """Stateless EESM SINR-effective-mapper. All beta values are class-level."""
        pass
    
    def compute(self, sinr_per_rb: np.ndarray, modulation_order: int = 2) -> float:
        """
        Compute effective SINR from per-RB SINRs.
        
        Args:
            sinr_per_rb: SINR values per RB (linear scale)
            modulation_order: Qm (2, 4, 6, or 8)
            
        Returns:
            Effective SINR in dB
        """
        beta = self.BETA_VALUES.get(modulation_order, 1.49)
        
        # Ensure linear scale
        sinr_linear = sinr_per_rb
        if np.any(sinr_per_rb < 0):
            # Assume dB scale if negative values present
            sinr_linear = 10 ** (sinr_per_rb / 10)
        
        # EESM formula
        n = len(sinr_linear)
        exp_sum = np.mean(np.exp(-sinr_linear / beta))
        
        sinr_eff_linear = -beta * np.log(exp_sum)
        sinr_eff_db = 10 * np.log10(sinr_eff_linear)
        
        return float(sinr_eff_db)
