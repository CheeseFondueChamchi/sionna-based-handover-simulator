"""
Unified BLER Calculator for NR Handover Simulation

Single source of truth for SINR-to-BLER mapping:
- AWGN BLER (sigmoid model, slope=0.7 for static/AWGN)
- Fading BLER (slope=0.5 for Doppler-affected channels)

Doppler ICI post-correction is applied in SionnaRTChannelModel._apply_doppler_correction()
using the Russell & Stuber ICI model. This calculator provides the sigmoid BLER mapping
that is called with corrected SINR values.

3GPP Reference:
    TS 38.214 Section 5.1.3 (MCS determination)
    TS 38.133 Section 8.1 (Sync indication)

Author: Claude Code
Date: 2026-02-27
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BLERCalculator:
    """
    Unified BLER calculator for NR.

    SINR thresholds for 10% BLER per MCS index are from 3GPP link-level
    simulations. The AWGN BLER curve uses a sigmoid approximation:

        BLER(SINR) = 1 / (1 + exp(slope * (SINR - threshold)))

    where threshold is shifted so that BLER=10% at SINR_10%.
    """

    # 3GPP TS 38.214 Table 5.1.3.1-1: MCS index table for PDSCH
    # Modulation order per MCS: 2=QPSK(4QAM), 4=16QAM, 6=64QAM, 8=256QAM
    MCS_MODULATION_ORDER = {
        0: 2,  1: 2,  2: 2,  3: 2,  4: 2,    # QPSK (4QAM)
        5: 2,  6: 2,  7: 2,  8: 2,  9: 2,    # QPSK (4QAM)
        10: 4, 11: 4, 12: 4, 13: 4, 14: 4,   # 16QAM
        15: 4, 16: 4,                          # 16QAM
        17: 6, 18: 6, 19: 6, 20: 6, 21: 6,   # 64QAM
        22: 6, 23: 6, 24: 6, 25: 6, 26: 6,   # 64QAM
        27: 6, 28: 6,                          # 64QAM
    }

    # SINR thresholds for 10% BLER per MCS (dB)
    # 3GPP link-level simulations (AWGN channel)
    # MCS 0-9: QPSK (4QAM), MCS 10-16: 16QAM, MCS 17-28: 64QAM
    SINR_10_PERCENT_BLER = {
        # QPSK (4QAM) - code rates 120/1024 to 679/1024
        0: -6.7,   1: -4.7,   2: -2.3,   3: 0.2,    4: 2.2,
        5: 3.7,    6: 5.4,    7: 6.9,    8: 8.4,    9: 10.4,
        # 16QAM - code rates 340/1024 to 666/1024
        10: 11.2,  11: 12.0,  12: 13.4,  13: 14.8,  14: 16.2,
        15: 17.6,  16: 18.8,
        # 64QAM - code rates 466/1024 to 948/1024
        17: 19.3,  18: 20.0,  19: 21.2,
        20: 22.5,  21: 23.8,  22: 25.0,  23: 26.4,  24: 27.6,
        25: 28.8,  26: 30.0,  27: 31.0,  28: 32.0,
    }

    # QPSK MCS range (MCS 0-9)
    QPSK_MCS_MIN = 0
    QPSK_MCS_MAX = 9

    # AWGN BLER slope (steepness of sigmoid transition)
    # 0.7 is the standard value for NR AWGN channels
    AWGN_SLOPE = 0.7

    # ---- Hypothetical PDCCH BLER model (for RLM Qout/Qin, TS 38.133 §8.1.2.1) ----
    # The RLM (radio-link monitoring) watches a *hypothetical PDCCH*, NOT a
    # PDSCH MCS. PDCCH is QPSK + Polar coding (TS 38.212), carries a DCI
    # format 1_0 (2 OFDM symbols, REG bundle 6, distributed mapping) mapped
    # onto an aggregation level (AL = 1/2/4/8/16 CCEs; 1 CCE = 6 REGs).
    # Higher AL = lower effective code rate = more coding gain (~3 dB per AL
    # doubling — i.e. AL8 is 3 dB more robust than AL4, AL16 3 dB more than AL8).
    #
    # Spec refs:
    #   TS 38.133 V16.4.0 §8.1.2.1, Tables 8.1.2.1-1/2:
    #     - Qout uses AL8 + PDCCH RE energy +4 dB over SSS
    #     - Qin  uses AL4 + PDCCH RE energy  0 dB
    #   RAN4-typical PDCCH-SINR operating points (not cell-SINR; the +4 dB /
    #   0 dB energy boosts are applied by the RLM caller in rrc_controller.py):
    #     Chen et al. arXiv:2001.02757 — ~3 dB coding gain per AL doubling;
    #     AL8 ~10% BLER at −7 to −9 dB PDCCH SINR,
    #     AL4 ~10% BLER at −5 to −6 dB PDCCH SINR.
    #
    # Values below are PDCCH-SINR thresholds for 10% BLER, monotonically
    # decreasing (AL16 most robust = lowest threshold). MUST remain monotonic.
    # NOTE: decoupled from SINR_10_PERCENT_BLER (PDSCH/LDPC) — do not reuse.
    PDCCH_SINR_10PCT_BLER_BY_AL = {
        1:  +1.0,  # AL1  (1 CCE)  — least robust
        2:  -2.0,  # AL2  (2 CCE)
        4:  -5.0,  # AL4  (4 CCE)  — Qin reference (0 dB boost → cell SINR ≈ -5 dB)
        8:  -8.0,  # AL8  (8 CCE)  — Qout reference (+4 dB boost → cell SINR ≈ -12 dB)
        16: -11.0, # AL16 (16 CCE) — most robust (used for HO-command gate)
    }
    # Default aggregation level the RLM assumes for the hypothetical PDCCH.
    PDCCH_AL_DEFAULT = 4
    # Sigmoid steepness for the short Polar block — steeper than the PDSCH AWGN
    # 0.7 (so 10% -> 2% BLER spans ~1.9 dB, matching observed PDCCH curves).
    PDCCH_BLER_SLOPE = 0.9

    @classmethod
    def get_modulation_name(cls, mcs_index: int) -> str:
        """Get modulation scheme name for an MCS index."""
        order = cls.MCS_MODULATION_ORDER.get(mcs_index, 2)
        return {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}.get(order, "?")

    @classmethod
    def clamp_to_qpsk(cls, mcs_index: int) -> int:
        """Clamp MCS index to QPSK (4QAM) range (0-9)."""
        return min(max(mcs_index, cls.QPSK_MCS_MIN), cls.QPSK_MCS_MAX)

    def sinr_to_bler(
        self,
        sinr_db: float,
        mcs_index: int,
        fading: bool = False
    ) -> float:
        """
        Map SINR to BLER for a given MCS.

        Uses sigmoid approximation:
            BLER = 1 / (1 + exp(slope * (SINR - SINR_threshold)))

        The threshold is offset so that BLER = 10% at SINR_10%.

        Args:
            sinr_db: Effective SINR in dB
            mcs_index: MCS index (0-28)
            fading: If True, use gentler slope (0.5) for fading/Doppler channels.

        Returns:
            BLER (0.0 to 1.0)
        """
        if mcs_index not in self.SINR_10_PERCENT_BLER:
            logger.warning(f"Invalid MCS index: {mcs_index}, using MCS 0")
            mcs_index = 0

        sinr_threshold = self.SINR_10_PERCENT_BLER[mcs_index]
        slope = 0.5 if fading else self.AWGN_SLOPE

        # Offset to anchor sigmoid at 10% BLER point
        offset = np.log(9.0) / slope
        exponent = np.clip(slope * (sinr_db - (sinr_threshold - offset)), -50, 50)
        bler = 1.0 / (1.0 + np.exp(exponent))

        return float(bler)

    def pdcch_sinr_to_bler(
        self,
        sinr_db: float,
        aggregation_level: int = PDCCH_AL_DEFAULT,
    ) -> float:
        """Map SINR to hypothetical-PDCCH BLER for RLM (TS 38.133 §8.5.2).

        Models the DCI 1_0 (QPSK + Polar) control channel at a given
        aggregation level, NOT a PDSCH MCS. Use this for Qout/Qin radio-link
        monitoring instead of ``sinr_to_bler(.., mcs_index=0)`` — the PDCCH is
        more robust than PDSCH MCS0 (coding gain from the AL), so its 10% BLER
        point sits at a different (AL-dependent) SINR.

        Sigmoid anchored at 10% BLER = ``PDCCH_SINR_10PCT_BLER_BY_AL[AL]``:
            BLER = 1 / (1 + exp(slope * (SINR - (thr - ln9/slope))))

        Args:
            sinr_db: Effective serving-cell SINR in dB.
            aggregation_level: PDCCH AL (1/2/4/8/16). Default 4. Unknown values
                fall back to the nearest defined AL.

        Returns:
            BLER in [0, 1].
        """
        table = self.PDCCH_SINR_10PCT_BLER_BY_AL
        if aggregation_level not in table:
            # snap to nearest defined AL
            aggregation_level = min(table.keys(),
                                    key=lambda al: abs(al - aggregation_level))
        sinr_threshold = table[aggregation_level]
        slope = self.PDCCH_BLER_SLOPE
        offset = np.log(9.0) / slope
        exponent = np.clip(slope * (sinr_db - (sinr_threshold - offset)), -50, 50)
        return float(1.0 / (1.0 + np.exp(exponent)))

    def select_mcs(self, sinr_db: float, target_bler: float = 0.1) -> int:
        """
        Adaptive MCS selection (AMC): pick the highest MCS with BLER <= target.

        This mimics 3GPP CQI-based link adaptation. The scheduler selects the
        highest MCS index where the expected BLER does not exceed target_bler
        (typically 10%).

        Args:
            sinr_db: Effective SINR in dB
            target_bler: Target BLER threshold (default 10%)

        Returns:
            Selected MCS index (0-28), or -1 if even MCS 0 exceeds target
        """
        best_mcs = -1
        for mcs in sorted(self.SINR_10_PERCENT_BLER.keys()):
            bler = self.sinr_to_bler(sinr_db, mcs)
            if bler <= target_bler:
                best_mcs = mcs
            else:
                break  # Higher MCS will only be worse
        return best_mcs

    def sinr_to_bler_adaptive(self, sinr_db: float) -> Tuple[float, int]:
        """
        Compute BLER with adaptive MCS selection (AMC).

        Returns the BLER at the highest supportable MCS, mimicking real
        3GPP link adaptation. If SINR is too low for even MCS 0, returns
        BLER=1.0 with MCS 0.

        Args:
            sinr_db: Effective SINR in dB

        Returns:
            (bler, selected_mcs) tuple
        """
        mcs = self.select_mcs(sinr_db)
        if mcs < 0:
            # Even MCS 0 can't achieve 10% BLER - return actual BLER at MCS 0
            return self.sinr_to_bler(sinr_db, 0), 0
        return self.sinr_to_bler(sinr_db, mcs), mcs


# Module-level singleton for convenience
_default_calculator = None


def get_bler_calculator() -> BLERCalculator:
    """Get the module-level BLERCalculator singleton."""
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = BLERCalculator()
    return _default_calculator
