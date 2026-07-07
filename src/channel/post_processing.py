"""
Shared Channel Post-Processing for NR Handover Simulation

Single source of truth for:
- Noise floor calculation
- Propagation loss application (penetration + surface distortion)
- SINR computation with beam isolation
- RSRQ computation

Both SionnaRTChannelModel and StatisticalChannelModel delegate to these
functions, eliminating duplicated formulas and inconsistent parameters.

3GPP Reference:
    TR 38.901 Section 7.4.3 (Additional losses)
    TS 38.215 Section 5.1 (RSRP/RSRQ definitions)

Author: Claude Code
Date: 2026-02-27
"""

import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Physical constants
THERMAL_NOISE_DBM_HZ = -174.0  # kTB reference at 290K


def default_bandwidth_mhz(freq_ghz: float) -> float:
    """
    Frequency-based default bandwidth (MHz) for LTE HST / NR scenarios.

    LTE HST: SCS = 15 kHz fixed.
        0.9 GHz (Band 8)  → 10 MHz  (N_SC=600,  PRB=50)
        1.8 GHz (Band 3)  → 20 MHz  (N_SC=1200, PRB=100)
        2.1 GHz (Band 1)  → 20 MHz  (N_SC=1200, PRB=100)
    NR sub-6: SCS = 30 kHz.
        3.5 GHz (n78)     → 20 MHz  (N_SC=588,  PRB=49)

    Override per-gNB via CSV column ``bandwidth_MHz`` or ``add_gnb(bandwidth_mhz=...)``.

    Args:
        freq_ghz: Carrier frequency in GHz

    Returns:
        Default bandwidth in MHz
    """
    if freq_ghz < 1.0:
        return 10.0   # LTE 900 (Band 8)
    elif freq_ghz < 3.0:
        return 20.0   # LTE 1.8/2.1 (Band 3/Band 1)
    else:
        return 20.0   # NR n78 (3.5 GHz)


def default_scs_hz(freq_ghz: float) -> float:
    """
    Frequency-based default subcarrier spacing (Hz).

    LTE (< 3 GHz): 15 kHz (fixed for all LTE bands)
    NR sub-6 (>= 3 GHz): 30 kHz

    Args:
        freq_ghz: Carrier frequency in GHz

    Returns:
        Default SCS in Hz
    """
    return 15e3 if freq_ghz < 3.0 else 30e3


def compute_noise_floor(bandwidth_hz: float, noise_figure_db: float) -> float:
    """
    Calculate thermal noise floor.

    Noise = kTB + NF = -174 dBm/Hz + 10*log10(BW) + NF

    Args:
        bandwidth_hz: System bandwidth in Hz
        noise_figure_db: UE noise figure in dB

    Returns:
        Noise floor in dBm

    3GPP Reference:
        TS 38.104 Section 7 (Receiver characteristics)
    """
    return (
        THERMAL_NOISE_DBM_HZ +
        10 * np.log10(bandwidth_hz) +
        noise_figure_db
    )


def _fft_size_from_bw_scs(bandwidth_mhz: float, scs_hz: float) -> int:
    """
    Number of usable subcarriers from bandwidth and SCS (3GPP TS 38.211).

    Args:
        bandwidth_mhz: Bandwidth in MHz
        scs_hz: Subcarrier spacing in Hz

    Returns:
        Number of usable subcarriers (aligned to PRB boundary)

    3GPP Reference:
        TS 38.211 Section 4.4.2 (Transmission bandwidth configuration)
    """
    bw_hz = bandwidth_mhz * 1e6
    effective_bw = bw_hz * 0.9  # 90% usable bandwidth
    n_sc = int(effective_bw / scs_hz)
    n_sc = max(12, (n_sc // 12) * 12)  # PRB boundary (12 subcarriers per PRB)
    return n_sc


def compute_noise_floor_per_re(
    bandwidth_mhz: float,
    scs_hz: float,
    noise_figure_db: float
) -> float:
    """
    Per-RE noise floor (matches per-RE RSRP definition).

    N_per_RE = N_total - 10*log10(N_SC)

    This matches 3GPP TS 38.215 Section 5.1.1 where RSRP is defined
    as the linear average over the power contributions of resource
    elements that carry reference signals.

    Args:
        bandwidth_mhz: Bandwidth in MHz
        scs_hz: Subcarrier spacing in Hz
        noise_figure_db: Noise figure in dB

    Returns:
        Per-RE noise floor in dBm

    3GPP Reference:
        TS 38.215 Section 5.1.1 (RSRP definition)
    """
    n_sc = _fft_size_from_bw_scs(bandwidth_mhz, scs_hz)
    total = compute_noise_floor(bandwidth_mhz * 1e6, noise_figure_db)
    return total - 10 * np.log10(n_sc)


def apply_propagation_losses(
    rsrp_raw_dbm: float,
    penetration_loss_db: float = 0.0,
    surface_mean_db: float = 0.0,
    surface_std_db: float = 0.0,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[float, float]:
    """
    Apply propagation losses to raw RSRP.

    Losses applied:
    1. Penetration loss (deterministic): vehicle/building body loss
    2. Surface distortion (stochastic): random signal reduction from
       train body reflections/scattering, modeled as log-normal

    Args:
        rsrp_raw_dbm: Raw RSRP before additional losses (dBm)
        penetration_loss_db: Deterministic penetration loss (dB)
        surface_mean_db: Mean of surface distortion loss (dB)
        surface_std_db: Std deviation of surface distortion (dB)
        rng: Optional random state for reproducibility

    Returns:
        Tuple of (rsrp_dbm, total_loss_db) where:
            rsrp_dbm = rsrp_raw - penetration - surface_distortion
            total_loss_db = penetration + surface_distortion

    3GPP Reference:
        TR 38.901 Section 7.4.3 (O2I penetration loss)
    """
    # Deterministic component
    total_loss = penetration_loss_db

    # Stochastic surface distortion (log-normal)
    if surface_std_db > 0.0:
        if rng is not None:
            surface_loss = max(0.0, rng.normal(surface_mean_db, surface_std_db))
        else:
            surface_loss = max(0.0, np.random.normal(surface_mean_db, surface_std_db))
        total_loss += surface_loss

    rsrp_dbm = rsrp_raw_dbm - total_loss
    return rsrp_dbm, total_loss


def compute_ul_sinr(
    dl_rsrp_dbm: float,
    gnb_ref_tx_power_dbm: float,
    ue_tx_power_dbm: float,
    ul_noise_floor_dbm: float,
) -> float:
    """
    Compute UL SINR using TDD channel reciprocity.

    In TDD (same freq DL/UL), the total channel loss is identical in both
    directions. We derive it from the DL measurement:

        total_channel_loss = gnb_ref_tx_power - dl_rsrp
        ul_rx_power_at_gnb = ue_tx_power - total_channel_loss
        ul_sinr = ul_rx_power_at_gnb - ul_noise_floor

    Args:
        dl_rsrp_dbm: DL RSRP at UE (after ALL losses: PL + SF + penetration + surface)
        gnb_ref_tx_power_dbm:
            - RT model: conducted_power (tx_power - antenna_gain), because Sionna
              adds antenna gain via ray-tracing, and reciprocity preserves it on UL RX
            - Statistical model: tx_power_dbm (antenna gain implicit in PL model)
        ue_tx_power_dbm: UE transmit power (23 dBm for PC3)
        ul_noise_floor_dbm: gNB noise floor = kTB + gNB_NF

    Returns:
        UL SINR in dB (single-cell, no UL inter-cell interference)

    3GPP Reference:
        TS 38.104 Section 7.4 (gNB receiver)
        TS 38.101-1 Section 6.2 (UE TX power)
    """
    total_channel_loss = gnb_ref_tx_power_dbm - dl_rsrp_dbm
    ul_rx_power_dbm = ue_tx_power_dbm - total_channel_loss
    return ul_rx_power_dbm - ul_noise_floor_dbm


def compute_sinr(
    signal_rsrp_dbm: float,
    interferer_rsrps_dbm: list,
    noise_floor_dbm: float,
    beam_isolation_db: float = 15.0
) -> float:
    """
    Compute SINR with multi-cell interference and beam isolation.

    SINR = P_signal / (sum(P_interference * isolation_factor) + P_noise)

    Beam isolation models the beamforming gain difference between
    the serving beam and interfering beams. Typical value: 15 dB.

    Args:
        signal_rsrp_dbm: Serving cell RSRP in dBm
        interferer_rsrps_dbm: List of interferer RSRP values in dBm
        noise_floor_dbm: Noise floor in dBm
        beam_isolation_db: Beam isolation factor in dB (default: 15 dB)

    Returns:
        SINR in dB

    3GPP Reference:
        TS 38.214 Section 5.1 (CSI framework)
    """
    signal_power = 10 ** (signal_rsrp_dbm / 10)
    noise_power = 10 ** (noise_floor_dbm / 10)

    isolation_factor = 10 ** (-beam_isolation_db / 10)

    interference_power = 0.0
    for rsrp in interferer_rsrps_dbm:
        if rsrp > -200.0:  # Skip invalid entries
            interference_power += (10 ** (rsrp / 10)) * isolation_factor

    sinr_linear = signal_power / (interference_power + noise_power + 1e-30)
    return float(10 * np.log10(sinr_linear + 1e-30))


def compute_sinr_linear_components(
    signal_rsrp_dbm: float,
    interferer_rsrps_dbm: list,
    noise_floor_dbm: float,
    beam_isolation_db: float = 15.0
) -> Tuple[float, float, float, float]:
    """
    Compute SINR and return linear-scale components for BLER computation.

    Returns signal, interference, noise in linear scale (mW) plus SINR in dB.
    This is needed by the Doppler BLER calculator which operates on linear
    power values.

    Args:
        signal_rsrp_dbm: Serving cell RSRP in dBm
        interferer_rsrps_dbm: List of interferer RSRP values in dBm
        noise_floor_dbm: Noise floor in dBm
        beam_isolation_db: Beam isolation factor in dB

    Returns:
        Tuple of (sinr_db, signal_linear, interference_linear, noise_linear)
    """
    signal_power = 10 ** (signal_rsrp_dbm / 10)
    noise_power = 10 ** (noise_floor_dbm / 10)

    isolation_factor = 10 ** (-beam_isolation_db / 10)

    interference_power = 0.0
    for rsrp in interferer_rsrps_dbm:
        if rsrp > -200.0:
            interference_power += (10 ** (rsrp / 10)) * isolation_factor

    sinr_linear = signal_power / (interference_power + noise_power + 1e-30)
    sinr_db = float(10 * np.log10(sinr_linear + 1e-30))

    return sinr_db, signal_power, interference_power, noise_power


def compute_rsrq(
    signal_rsrp_dbm: float,
    interferer_rsrps_dbm: list,
    noise_floor_dbm: float,
    n_rb: int = 100,
    beam_isolation_db: float = 15.0
) -> float:
    """
    Compute RSRQ (Reference Signal Received Quality).

    RSRQ = 10*log10(N_RB * RSRP / RSSI)
    where RSSI = signal + interference + noise

    Args:
        signal_rsrp_dbm: Serving cell RSRP in dBm
        interferer_rsrps_dbm: List of interferer RSRPs in dBm
        noise_floor_dbm: Noise floor in dBm
        n_rb: Number of resource blocks (default: 100)
        beam_isolation_db: Beam isolation factor in dB

    Returns:
        RSRQ in dB

    3GPP Reference:
        TS 38.215 Section 5.1.3 (RSRQ definition)
    """
    signal_power = 10 ** (signal_rsrp_dbm / 10)
    noise_power = 10 ** (noise_floor_dbm / 10)

    isolation_factor = 10 ** (-beam_isolation_db / 10)

    interference_power = 0.0
    for rsrp in interferer_rsrps_dbm:
        if rsrp > -200.0:
            interference_power += (10 ** (rsrp / 10)) * isolation_factor

    rssi_linear = signal_power + interference_power + noise_power
    return float(10 * np.log10((n_rb * signal_power) / (rssi_linear + 1e-30)))
