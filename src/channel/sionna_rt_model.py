"""
Sionna Ray-Tracing Channel Model for NR/LTE Handover Simulation (v2)

Changes from v1:
  1. Per-gNB frequency_ghz, bandwidth_mhz, scs_hz (not global)
  2. Per-link noise floor from gnb's bandwidth
  3. Multi-frequency RT: path computation grouped by frequency
  4. compute_all() accepts serving_cells dict → BLER for actual serving cell
  5. compute_link_bler() includes interference
  6. PHY chain uses gnb's SCS for subcarrier frequencies
  7. precompute SINR map bandwidth from CSV

SINR: precomputed RadioMap lookup (precompute_sinr_map.py)
BLER: RT CIR → full PHY chain (Sionna phy)
RSRP: RT CIR path power summation (핸드오버 측정용)
"""

import os
import glob
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from .channel_model import (
    ChannelModel,
    ChannelConfig,
    ChannelModelType,
    DopplerInfo,
)
from .channel_calculator import (
    ChannelState,
    GnbConfig,
    UeConfig,
    SIONNA_AVAILABLE,
)
from .post_processing import (
    compute_noise_floor,
    compute_noise_floor_per_re,
    _fft_size_from_bw_scs,
    default_bandwidth_mhz,
    default_scs_hz,
    apply_propagation_losses,
    compute_sinr_linear_components,
    compute_rsrq,
    compute_ul_sinr,
)

logger = logging.getLogger(__name__)

if SIONNA_AVAILABLE:
    import tensorflow as tf
    import sionna
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
    from sionna.rt.path_solvers import PathSolver

    try:
        from sionna.phy.mapping import BinarySource, Mapper, Demapper
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        _PHY_AVAILABLE = True
    except ImportError:
        _PHY_AVAILABLE = False

    try:
        from sionna.sys import PHYAbstraction as SionnaPHYAbstraction
        from sionna.sys import EESM
        _SYS_AVAILABLE = True
    except ImportError:
        _SYS_AVAILABLE = False

    # ★ Extend ITU materials to support 0.5-1.0 GHz (LTE 0.9/0.8 GHz bands)
    # ITU-R P2040 Table 3 defines most materials from 1.0 GHz.
    # Extrapolation is physically valid (< 0.5 GHz below range).
    try:
        from sionna.rt.radio_materials.itu import ITU_MATERIALS_PROPERTIES
        _itu_extensions = {
            "metal":            {(0.5, 100.): (1.0, 0.0, 1e7, 0.0)},
            "concrete":         {(0.5, 100.): (5.24, 0.0, 0.0462, 0.7822)},
            "brick":            {(0.5, 40.):  (3.91, 0.0, 0.0238, 0.16)},
            "medium_dry_ground":{(0.5, 10.):  (15., -0.1, 0.035, 1.63)},
            "very_dry_ground":  {(0.5, 10.):  (3.0, 0.0, 0.00015, 2.52)},
            "wet_ground":       {(0.5, 10.):  (30., -0.4, 0.15, 1.30)},
            "plasterboard":     {(0.5, 100.): (2.73, 0.0, 0.0085, 0.9395)},
            "ceiling_board":    {(0.5, 100.): (1.48, 0.0, 0.0011, 1.0750),
                                 (220., 450.): (1.52, 0.0, 0.0029, 1.029)},
            "chipboard":        {(0.5, 100.): (2.58, 0.0, 0.0217, 0.7800)},
            "plywood":          {(0.5, 40.):  (2.71, 0.0, 0.33, 0.0)},
            "marble":           {(0.5, 60.):  (7.074, 0.0, 0.0055, 0.9262)},
        }
        for mat_name, ranges in _itu_extensions.items():
            if mat_name in ITU_MATERIALS_PROPERTIES:
                ITU_MATERIALS_PROPERTIES[mat_name] = ranges
        logger.info("Extended ITU material frequency ranges to 0.5 GHz")
    except ImportError:
        pass


# =========================================================================
# LTE / NR MCS Tables
# =========================================================================
LTE_MCS_TABLE = {
    0:  (2, 0.12),  1:  (2, 0.15),  2:  (2, 0.19),  3:  (2, 0.25),
    4:  (2, 0.30),  5:  (2, 0.37),  6:  (2, 0.44),  7:  (2, 0.51),
    8:  (2, 0.59),  9:  (2, 0.66),
    10: (4, 0.33),  11: (4, 0.37),  12: (4, 0.42),  13: (4, 0.48),
    14: (4, 0.54),  15: (4, 0.60),  16: (4, 0.64),
    17: (6, 0.43),  18: (6, 0.46),  19: (6, 0.50),  20: (6, 0.55),
    21: (6, 0.60),  22: (6, 0.65),  23: (6, 0.70),  24: (6, 0.75),
    25: (6, 0.80),  26: (6, 0.85),  27: (6, 0.89),  28: (6, 0.93),
}

NR_MCS_TABLE = {
    0:  (2, 0.12),  1:  (2, 0.15),  2:  (2, 0.19),  3:  (2, 0.25),
    4:  (2, 0.31),  5:  (2, 0.38),  6:  (2, 0.44),  7:  (2, 0.51),
    8:  (2, 0.59),  9:  (2, 0.66),
    10: (4, 0.33),  11: (4, 0.37),  12: (4, 0.42),  13: (4, 0.48),
    14: (4, 0.54),  15: (4, 0.60),  16: (4, 0.64),
    17: (6, 0.46),  18: (6, 0.50),  19: (6, 0.55),  20: (6, 0.60),
    21: (6, 0.65),  22: (6, 0.70),  23: (6, 0.75),  24: (6, 0.80),
    25: (6, 0.85),  26: (6, 0.89),  27: (6, 0.93),  28: (6, 0.95),
}


# =========================================================================
# Per-gNB RF Parameters (parallel storage for fields not in GnbConfig)
# =========================================================================
# default_bandwidth_mhz, default_scs_hz, _fft_size_from_bw_scs are
# imported from post_processing.py (single source of truth)


# =========================================================================
# SINRMapLookup — precomputed SINR map lookup (unchanged from v1)
# =========================================================================
class SINRMapLookup:
    """Precomputed .npz SINR map → position-dependent Doppler penalty.

    Loads 2D arrays (sinr_db, sinr_doppler_db) per frequency band.
    Provides get_doppler_penalty(x, y, freq) which returns the Doppler
    degradation at the UE position: (sinr_doppler - sinr_static).

    Usage in compute_all():
        1. Always compute RSRP-ratio SINR (per-serving-cell, accurate)
        2. Look up Doppler penalty from map (position-dependent, ~-6 dB mean @300km/h)
        3. Apply: sinr_final = sinr_rsrp_ratio + doppler_penalty

    This decouples the map from serving-cell identity — any cell's SINR
    benefits from position-dependent Doppler correction.
    """

    _INVALID_THRESHOLD = -150.0  # SINR below this = no RT coverage

    def __init__(self):
        """Initialize empty SINR map storage and penalty grid cache."""
        self._maps: Dict[float, dict] = {}
        # Precomputed penalty grids (sinr_doppler - sinr_static), NaN where invalid
        self._penalty_grids: Dict[float, np.ndarray] = {}

    @property
    def loaded(self) -> bool:
        """Return True if at least one SINR map has been loaded."""
        return len(self._maps) > 0

    @property
    def frequencies(self) -> List[float]:
        """Return sorted list of loaded carrier frequencies in GHz."""
        return sorted(self._maps.keys())

    def load_all(self, map_dir: str) -> int:
        """Load all sinr_map_*.npz files from a directory.

        Args:
            map_dir: Directory containing precomputed .npz SINR map files.

        Returns:
            Number of maps successfully loaded.

        Side effects:
            - Populates self._maps and self._penalty_grids.
            - Logs a warning for each file that fails to load.
        """
        pattern = os.path.join(map_dir, "sinr_map_*.npz")
        for fpath in sorted(glob.glob(pattern)):
            try:
                self.load(fpath)
            except Exception as e:
                logger.warning(f"SINRMapLookup: skipped {os.path.basename(fpath)} ({e})")
        logger.info(f"SINRMapLookup: loaded {len(self._maps)} maps "
                     f"({', '.join(f'{f:.1f}GHz' for f in sorted(self._maps))}) from {map_dir}")
        return len(self._maps)

    def load(self, npz_path: str) -> None:
        """Load a single .npz SINR map file and register it by frequency.

        Args:
            npz_path: Path to a .npz file produced by precompute_sinr_map.py.
                Expected keys: frequency_ghz, sinr_db, center_x, center_y,
                size_x, size_y, cell_size. Optional: sinr_doppler_db, velocity_kmh.

        Side effects:
            - Adds an entry to self._maps keyed by frequency in GHz.
            - If sinr_doppler_db is present, pre-computes the penalty grid
              (sinr_doppler - sinr_static) and stores it in self._penalty_grids.
        """
        data = np.load(npz_path, allow_pickle=True)
        freq = float(data["frequency_ghz"])
        sinr_static = np.array(data["sinr_db"])  # (H, W) float32
        m = dict(
            sinr_db=sinr_static,
            center_x=float(data["center_x"]),
            center_y=float(data["center_y"]),
            size_x=float(data["size_x"]),
            size_y=float(data["size_y"]),
            cell_size=float(data["cell_size"]),
        )
        v_kmh = 0.0
        if "sinr_doppler_db" in data:
            sinr_doppler = np.array(data["sinr_doppler_db"])
            m["sinr_doppler_db"] = sinr_doppler
            v_kmh = float(data["velocity_kmh"]) if "velocity_kmh" in data else 0.0
            # Precompute penalty grid: NaN where either is invalid
            valid = (sinr_static > self._INVALID_THRESHOLD) & \
                    (sinr_doppler > self._INVALID_THRESHOLD)
            penalty = np.full_like(sinr_static, np.nan)
            penalty[valid] = sinr_doppler[valid] - sinr_static[valid]
            self._penalty_grids[freq] = penalty
            n_valid = int(valid.sum())
            pct = 100 * n_valid / valid.size
            mean_pen = float(np.nanmean(penalty[valid])) if n_valid > 0 else 0.0
            logger.info(f"  {freq:.1f}GHz: {sinr_static.shape}, "
                         f"Doppler@{v_kmh:.0f}km/h, "
                         f"coverage={pct:.0f}%, "
                         f"mean_penalty={mean_pen:.1f}dB")
        else:
            logger.info(f"  {freq:.1f}GHz: {sinr_static.shape} (static only, no Doppler)")
        self._maps[freq] = m

    # Max physically plausible Doppler penalty (dB).
    # At 300km/h sub-6GHz: fd/SCS ≈ 3%, degradation ≈ 3-10 dB.
    # Values beyond -20 dB are coverage-edge artifacts, not real Doppler.
    _MAX_PENALTY_DB = -20.0

    def get_doppler_penalty(
        self,
        x: float, y: float,
        freq_ghz: float,
        search_radius: int = 20,
        min_valid_points: int = 4,
    ) -> Optional[float]:
        """Position-dependent Doppler penalty (dB, always <= 0).

        Returns median(sinr_doppler - sinr_static) at nearby valid grid points.
        Filters out coverage-edge artifacts (penalty < -20 dB) and requires
        min_valid_points for statistical reliability.

        Args:
            x, y: UE position (world coordinates)
            freq_ghz: Carrier frequency
            search_radius: Max grid cells to search (default 20 = 100m @5m grid)
            min_valid_points: Minimum valid points for reliable estimate (default 4)
        """
        m = self._find_map(freq_ghz)
        if m is None:
            return None
        map_key = self._find_map_key(freq_ghz)
        if map_key is None:
            return None
        penalty = self._penalty_grids.get(map_key)
        if penalty is None:
            return None  # No Doppler data for this frequency

        ix, iy = self._world_to_grid(x, y, m)
        H, W = penalty.shape

        iy_int = int(np.clip(round(iy), 0, H - 1))
        ix_int = int(np.clip(round(ix), 0, W - 1))

        # Check exact grid point first (fast path)
        val = penalty[iy_int, ix_int]
        if not np.isnan(val):
            clamped = float(max(val, self._MAX_PENALTY_DB))
            logger.debug(f"  Doppler@{freq_ghz:.1f}GHz: grid({ix_int},{iy_int}) "
                          f"exact hit → raw={val:.1f}, clamped={clamped:.1f}dB")
            return clamped

        # Search nearby valid points (spiral outward)
        for r in range(1, search_radius + 1):
            y_lo = max(0, iy_int - r)
            y_hi = min(H, iy_int + r + 1)
            x_lo = max(0, ix_int - r)
            x_hi = min(W, ix_int + r + 1)
            region = penalty[y_lo:y_hi, x_lo:x_hi]
            valid_vals = region[~np.isnan(region)]
            # Clamp coverage-edge artifacts
            valid_vals = np.clip(valid_vals, self._MAX_PENALTY_DB, 0.0)
            if len(valid_vals) >= min_valid_points:
                med = float(np.median(valid_vals))
                logger.debug(f"  Doppler@{freq_ghz:.1f}GHz: grid({ix_int},{iy_int}) "
                              f"r={r}, {len(valid_vals)} pts, "
                              f"median={med:.1f}dB "
                              f"[{valid_vals.min():.1f}~{valid_vals.max():.1f}]")
                return med

        logger.debug(f"  Doppler@{freq_ghz:.1f}GHz: grid({ix_int},{iy_int}) "
                      f"<{min_valid_points} valid within r={search_radius}")
        return None  # Not enough valid coverage nearby

    def _find_map(self, freq_ghz, tol=0.05):
        """Return the map dict for freq_ghz within tolerance, or None if absent."""
        for k, v in self._maps.items():
            if abs(k - freq_ghz) < tol:
                return v
        return None

    def _find_map_key(self, freq_ghz, tol=0.05):
        """Return the float key in self._maps nearest to freq_ghz within tolerance, or None."""
        for k in self._maps:
            if abs(k - freq_ghz) < tol:
                return k
        return None

    def _world_to_grid(self, x, y, m):
        """Convert world coordinates to fractional grid indices for map m.

        Args:
            x: World x coordinate (metres).
            y: World y coordinate (metres).
            m: Map dict with keys center_x, center_y, size_x, size_y, cell_size.

        Returns:
            Tuple (ix, iy) as floating-point grid indices (column, row).
        """
        ox = m["center_x"] - m["size_x"] / 2.0
        oy = m["center_y"] - m["size_y"] / 2.0
        cs = m["cell_size"]
        return (x - ox) / cs, (y - oy) / cs


# =========================================================================
# PHYChainBLER — CIR 기반 BLER 추정 (v2: per-gnb SCS/BW)
# =========================================================================
class PHYChainBLER:
    """
    RT CIR → BLER.
    v2: subcarrier frequencies를 per-gnb SCS, bandwidth로 생성.
    """

    def __init__(self, default_batch_size: int = 1):
        """Initialize PHY chain BLER estimator with an LDPC+QAM chain cache.

        Args:
            default_batch_size: Number of codewords per batch for full-chain
                estimation. Higher values reduce Monte Carlo noise but increase
                compute time.

        Side effects:
            - Instantiates EESM and SionnaPHYAbstraction if sionna.sys is available.
        """
        self.default_batch_size = default_batch_size
        self._chain_cache: Dict[Tuple, Any] = {}
        self._eesm = EESM() if _SYS_AVAILABLE else None
        self._phy_abs = SionnaPHYAbstraction() if _SYS_AVAILABLE else None

    @staticmethod
    def get_mcs_params(mcs_index: int, freq_ghz: float) -> Tuple[int, float]:
        """Look up modulation order and code rate for a given MCS index.

        Args:
            mcs_index: MCS index (0-28). Out-of-range values are clamped to nearest.
            freq_ghz: Carrier frequency in GHz. Values < 3.0 use LTE_MCS_TABLE;
                >= 3.0 use NR_MCS_TABLE.

        Returns:
            Tuple (modulation_order, code_rate) where modulation_order is bits
            per symbol (2=QPSK, 4=16QAM, 6=64QAM).
        """
        table = LTE_MCS_TABLE if freq_ghz < 3.0 else NR_MCS_TABLE
        if mcs_index not in table:
            mcs_index = min(table.keys(), key=lambda k: abs(k - mcs_index))
        return table[mcs_index]

    @staticmethod
    def make_subcarrier_frequencies(
        bandwidth_mhz: float,
        scs_hz: float,
    ) -> np.ndarray:
        """
        Per-gnb bandwidth, SCS로 subcarrier frequency offset 생성.
        ★ v2: 하드코딩 PHY_BAND_CONFIGS 제거, 인수로 받음.
        """
        n_sc = _fft_size_from_bw_scs(bandwidth_mhz, scs_hz)
        return (np.arange(n_sc) - n_sc // 2) * scs_hz

    @staticmethod
    def compute_cfr(a_link, tau_link, frequencies):
        """CIR → CFR. a_link: [ra, ta, paths, time], tau_link: [paths]."""
        a_exp = a_link[:, :, np.newaxis, :, :]
        tau_exp = tau_link[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        f_exp = frequencies[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        phase = np.exp(-1j * 2 * np.pi * f_exp * tau_exp)
        return np.sum(a_exp * phase, axis=3)

    @staticmethod
    def compute_effective_siso_channel(H):
        """MIMO CFR → SISO via SVD beamforming."""
        num_ra, num_ta, num_sc, num_time = H.shape
        if num_ra == 1 and num_ta == 1:
            return H[0, 0, :, :]
        h_eff = np.zeros((num_sc, num_time), dtype=complex)
        for sc in range(num_sc):
            H_sc_0 = H[:, :, sc, 0]
            try:
                U, S, Vh = np.linalg.svd(H_sc_0, full_matrices=False)
                w_rx, w_tx = U[:, 0].conj(), Vh[0, :].conj()
            except np.linalg.LinAlgError:
                w_rx = np.ones(num_ra, dtype=complex) / np.sqrt(num_ra)
                w_tx = np.ones(num_ta, dtype=complex) / np.sqrt(num_ta)
            for t in range(num_time):
                h_eff[sc, t] = w_rx @ H[:, :, sc, t] @ w_tx
        return h_eff

    def estimate_bler_eesm(self, h_eff, noise_var_linear, mcs_index, freq_ghz):
        """Estimate BLER using the EESM (Effective Exponential SINR Mapping) method.

        Args:
            h_eff: Effective SISO channel coefficients, shape (num_sc, num_time),
                complex ndarray from compute_effective_siso_channel.
            noise_var_linear: Linear noise variance per subcarrier (mW or normalized).
            mcs_index: MCS index (0-28) selecting the BLER curve.
            freq_ghz: Carrier frequency in GHz (selects LTE vs NR abstraction).

        Returns:
            BLER as a float in [0, 1]. Returns 0.5 if sionna.sys is unavailable
            or if the EESM call fails.
        """
        if not _SYS_AVAILABLE:
            return 0.5
        h_power = np.mean(np.abs(h_eff) ** 2, axis=-1)
        sinr_per_sc = h_power / max(noise_var_linear, 1e-30)
        sc_per_prb = 12
        num_prb = max(1, len(sinr_per_sc) // sc_per_prb)
        sinr_per_rb = np.zeros(num_prb)
        for rb in range(num_prb):
            chunk = sinr_per_sc[rb * sc_per_prb: (rb + 1) * sc_per_prb]
            sinr_per_rb[rb] = np.mean(chunk) if len(chunk) > 0 else 1e-6
        sinr_tensor = tf.constant(sinr_per_rb[np.newaxis, :], dtype=tf.float32)
        try:
            sinr_eff = self._eesm(sinr_tensor, mcs_index)
            bler = self._phy_abs(sinr_eff, mcs_index)
            return float(tf.squeeze(bler).numpy())
        except Exception as e:
            logger.warning(f"EESM BLER failed: {e}")
            return 0.5

    def estimate_bler_full_chain(self, h_eff, noise_var_linear, mcs_index,
                                  freq_ghz, batch_size=None):
        """Estimate BLER via a full Sionna LDPC+QAM encode/transmit/decode chain.

        Runs Monte Carlo simulation: encodes random bits, applies the channel
        h_eff with AWGN, ZF-equalises, demaps, and decodes. Block error rate
        is the fraction of codewords with at least one bit error.

        Args:
            h_eff: Effective SISO channel, shape (num_sc, num_time), complex ndarray.
            noise_var_linear: Linear noise variance (normalized to TX power per SC).
            mcs_index: MCS index (0-28).
            freq_ghz: Carrier frequency in GHz (selects LTE vs NR MCS table).
            batch_size: Number of codewords per Monte Carlo trial. Defaults to
                self.default_batch_size.

        Returns:
            BLER as a float in [0, 1]. Falls back to estimate_bler_eesm if
            sionna.phy is unavailable.

        Side effects:
            - Caches the LDPC+QAM chain in self._chain_cache keyed by
              (k_info, n_coded, mod_order) for reuse across calls.
        """
        if not _PHY_AVAILABLE:
            return self.estimate_bler_eesm(h_eff, noise_var_linear, mcs_index, freq_ghz)

        bs = batch_size or self.default_batch_size
        mod_order, code_rate = self.get_mcs_params(mcs_index, freq_ghz)
        num_sc, num_time = h_eff.shape
        num_data_re = num_sc * num_time
        n_coded = num_data_re * mod_order
        k_info = int(n_coded * code_rate)
        k_info = max(22, min(k_info, 8448))
        n_coded = min(int(k_info / max(code_rate, 0.05)), 68 * 384)
        n_coded = max(n_coded, k_info + 2)

        cache_key = (k_info, n_coded, mod_order)
        if cache_key not in self._chain_cache:
            self._chain_cache[cache_key] = self._build_chain(k_info, n_coded, mod_order)
        binary_src, encoder, decoder, mapper, demapper = self._chain_cache[cache_key]

        bits = binary_src([bs, k_info])
        coded = encoder(bits)
        symbols = mapper(coded)
        num_symbols = symbols.shape[-1]

        h_flat = h_eff.flatten()[:num_symbols]
        if len(h_flat) < num_symbols:
            h_flat = np.tile(h_flat, (num_symbols // len(h_flat)) + 1)[:num_symbols]
        h_tf = tf.constant(h_flat[np.newaxis, :], dtype=tf.complex64)

        noise_std = np.sqrt(noise_var_linear / 2.0)
        noise = tf.complex(
            tf.random.normal([bs, num_symbols], stddev=noise_std, dtype=tf.float32),
            tf.random.normal([bs, num_symbols], stddev=noise_std, dtype=tf.float32),
        )
        y = tf.cast(symbols, tf.complex64) * h_tf + noise
        h_safe = tf.where(tf.abs(h_tf) > 1e-10, h_tf,
                          tf.constant(1e-10 + 0j, dtype=tf.complex64))
        x_hat = y / h_safe

        # ★★★ FIX v3: ZF 등화 후 demapper에 post-EQ noise variance 전달 ★★★
        #
        # x_hat = y/h = x + n/h  이므로,
        # post-EQ noise variance = noise_var / |h|²
        # demapper는 x_hat의 noise를 알아야 정확한 LLR 계산 가능.
        # pre-EQ noise_var를 그대로 전달하면 LLR이 |h|²배 과대 → 디코딩 실패.
        #
        h_power_mean = tf.cast(tf.reduce_mean(tf.abs(h_tf) ** 2), tf.float32)
        h_power_mean = tf.maximum(h_power_mean, tf.constant(1e-20, dtype=tf.float32))
        post_eq_noise_var = tf.constant(noise_var_linear, dtype=tf.float32) / h_power_mean

        try:
            llr = demapper(x_hat, post_eq_noise_var)
        except TypeError:
            llr = demapper([x_hat, post_eq_noise_var])
        bits_hat = decoder(llr)

        errors = tf.reduce_any(tf.not_equal(bits, bits_hat), axis=-1)
        return float(tf.reduce_mean(tf.cast(errors, tf.float32)).numpy())

    def estimate(
        self,
        a_link: np.ndarray,
        tau_link: np.ndarray,
        freq_ghz: float,
        bandwidth_mhz: float,
        scs_hz: float,
        noise_var_linear: float,
        mcs_index: int,
        method: str = "full_chain",
        batch_size: Optional[int] = None,
    ) -> float:
        """
        CIR → BLER (통합).
        ★ v2: bandwidth_mhz, scs_hz 인수 추가.
        """
        frequencies = self.make_subcarrier_frequencies(bandwidth_mhz, scs_hz)
        H = self.compute_cfr(a_link, tau_link, frequencies)
        h_eff = self.compute_effective_siso_channel(H)
        if method == "eesm":
            return self.estimate_bler_eesm(h_eff, noise_var_linear, mcs_index, freq_ghz)
        else:
            return self.estimate_bler_full_chain(h_eff, noise_var_linear, mcs_index,
                                                  freq_ghz, batch_size)

    def _build_chain(self, k_info, n_coded, mod_order):
        """Instantiate a Sionna LDPC5G + QAM encode/decode chain.

        Args:
            k_info: Number of information bits per codeword.
            n_coded: Number of coded bits per codeword (after rate-matching).
            mod_order: Modulation order in bits per symbol (2/4/6).

        Returns:
            Tuple of (BinarySource, LDPC5GEncoder, LDPC5GDecoder, Mapper, Demapper)
            ready for use in estimate_bler_full_chain.
        """
        return (
            BinarySource(),
            LDPC5GEncoder(k_info, n_coded),
            LDPC5GDecoder(LDPC5GEncoder(k_info, n_coded), hard_out=True),
            Mapper("qam", mod_order),
            Demapper("app", "qam", mod_order),
        )


# =========================================================================
# SlidingBLER — 3GPP TS 36.133 기반 슬라이딩 윈도우 BLER 측정
# =========================================================================
class SlidingBLER:
    """
    LTE/NR에서 BLER은 단일 TTI가 아닌 과거 N개 TB의 이동 평균으로 측정됨.
    3GPP TS 36.133 §7.6: Qout(OOS)은 BLER 10% @ 200ms 윈도우 기준.

    사용법:
        sliding = SlidingBLER(window_tbs=200, tbs_per_step=20)
        raw_bler = phy_chain.estimate(...)     # batch=20 → 0%, 5%, 10% 등
        smoothed = sliding.update(ue_id, raw_bler)  # 윈도우 평균
    """

    def __init__(self, window_tbs: int = 200, tbs_per_step: int = 20):
        """
        Args:
            window_tbs: 슬라이딩 윈도우 크기 (TB 수). 200 = 200ms @ 1ms TTI.
            tbs_per_step: 시뮬레이션 1 step당 전송되는 TB 수.
                          = timestep_duration_s / TTI_duration_s
                          = 0.02s / 0.001s = 20 (LTE 기준)
        """
        self.window_tbs = window_tbs
        self.tbs_per_step = tbs_per_step
        self._history: Dict[int, List[Tuple[int, int]]] = {}
        self._total_errors: Dict[int, int] = {}
        self._total_tbs: Dict[int, int] = {}

    def update(self, ue_id: int, raw_bler: float) -> float:
        """
        한 step의 raw BLER을 기록하고 윈도우 평균 BLER 반환.

        Args:
            ue_id: UE 식별자
            raw_bler: PHY chain에서 나온 즉시 BLER (batch=tbs_per_step)
        Returns:
            smoothed BLER (윈도우 내 평균)
        """
        if ue_id not in self._history:
            self._history[ue_id] = []
            self._total_errors[ue_id] = 0
            self._total_tbs[ue_id] = 0

        num_tbs = self.tbs_per_step
        num_errors = int(round(raw_bler * num_tbs))
        num_errors = min(num_errors, num_tbs)

        self._history[ue_id].append((num_errors, num_tbs))
        self._total_errors[ue_id] += num_errors
        self._total_tbs[ue_id] += num_tbs

        while self._total_tbs[ue_id] > self.window_tbs and len(self._history[ue_id]) > 1:
            old_errors, old_tbs = self._history[ue_id].pop(0)
            self._total_errors[ue_id] -= old_errors
            self._total_tbs[ue_id] -= old_tbs

        total_tbs = self._total_tbs[ue_id]
        if total_tbs <= 0:
            return raw_bler

        return self._total_errors[ue_id] / total_tbs

    def reset_ue(self, ue_id: int) -> None:
        """핸드오버 시 해당 UE의 윈도우를 리셋."""
        self._history.pop(ue_id, None)
        self._total_errors.pop(ue_id, None)
        self._total_tbs.pop(ue_id, None)

    def reset_all(self) -> None:
        """전체 리셋."""
        self._history.clear()
        self._total_errors.clear()
        self._total_tbs.clear()

    def get_window_info(self, ue_id: int) -> dict:
        """디버깅용: 현재 윈도우 상태 조회."""
        total_tbs = self._total_tbs.get(ue_id, 0)
        total_errors = self._total_errors.get(ue_id, 0)
        return {
            "window_tbs": self.window_tbs,
            "accumulated_tbs": total_tbs,
            "accumulated_errors": total_errors,
            "smoothed_bler": total_errors / max(total_tbs, 1),
            "num_steps": len(self._history.get(ue_id, [])),
            "window_fill_pct": total_tbs / self.window_tbs * 100,
        }


class MockupBLERProvider:
    """CSV 기반 고속 BLER 조회기. SINR→BLER을 사전 계산 테이블에서 즉시 조회."""

    def __init__(self, csv_dir: str, global_offset_db: float = 0.0):
        """Load pre-computed BLER lookup table from a CSV file.

        Args:
            csv_dir: Directory containing bler_all_mcs.csv (generated by
                script/plot_bler_curves.py). Raises FileNotFoundError if absent.
            global_offset_db: Global SINR shift applied before every lookup,
                e.g. +3.0 to model an implementation margin.

        Side effects:
            - Reads and caches the entire BLER table as numpy arrays.
            - Logs the number of SINR points and the applied offset.
        """
        csv_path = os.path.join(csv_dir, "bler_all_mcs.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"BLER CSV not found: {csv_path}\n"
                                    f"Run: python script/plot_bler_curves.py")
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self._sinr_arr = data[:, 0]
        self._bler_table = data[:, 1:30]  # shape: (N, 29)
        self._sinr_min = self._sinr_arr[0]
        self._sinr_max = self._sinr_arr[-1]
        self._sinr_step = self._sinr_arr[1] - self._sinr_arr[0]
        self._global_offset = global_offset_db
        logger.info(f"MockupBLER loaded: {len(self._sinr_arr)} SINR points, "
                    f"offset={global_offset_db:+.1f}dB")

    def lookup(self, sinr_db: float, mcs_index: int) -> float:
        """SINR → BLER lookup with linear interpolation."""
        mcs_index = max(0, min(28, mcs_index))
        effective_sinr = sinr_db + self._global_offset
        if effective_sinr <= self._sinr_min:
            return float(self._bler_table[0, mcs_index])
        if effective_sinr >= self._sinr_max:
            return float(self._bler_table[-1, mcs_index])
        idx_f = (effective_sinr - self._sinr_min) / self._sinr_step
        idx_lo = int(np.floor(idx_f))
        idx_hi = min(idx_lo + 1, len(self._sinr_arr) - 1)
        frac = idx_f - idx_lo
        return float(self._bler_table[idx_lo, mcs_index] +
                     frac * (self._bler_table[idx_hi, mcs_index] -
                             self._bler_table[idx_lo, mcs_index]))

    def lookup_adaptive(self, sinr_db: float) -> Tuple[float, int]:
        """Adaptive MCS: pick highest MCS with BLER <= 10%."""
        best_mcs = 0
        for mcs in range(29):
            if self.lookup(sinr_db, mcs) <= 0.1:
                best_mcs = mcs
            else:
                break
        return self.lookup(sinr_db, best_mcs), best_mcs


# =========================================================================
# SionnaRTChannelModel (v2)
# =========================================================================
class SionnaRTChannelModel(ChannelModel):
    """
    v2 Changes:
      1. _gnb_rf_params: per-gnb {bandwidth_mhz, scs_hz} 저장
      2. add_gnb(): frequency_ghz, bandwidth_mhz, scs_hz kwargs로 수신
      3. compute_all(): serving_cells 인수, per-frequency-group RT, per-link noise
      4. compute_link_bler(): interference_dbm 인수
    """

    DEFAULT_FREQUENCY_HZ = 3.5e9
    DEFAULT_TX_POWER_DBM = 46.0
    DEFAULT_NOISE_FIGURE_DB = 7.0
    DEFAULT_BANDWIDTH_HZ = 20e6

    # Per-frequency TX antenna array (matches precompute_sinr_map.py)
    # _compute_paths_for_frequency() sets scene.tx_array per frequency group
    FREQ_TX_ARRAY_CONFIGS = {
        0.9: dict(num_rows=1, num_cols=1, vertical_spacing=0.5,
                  horizontal_spacing=0.5, pattern="tr38901", polarization="VH"),  # LTE 2T2R
        1.8: dict(num_rows=1, num_cols=1, vertical_spacing=0.5,
                  horizontal_spacing=0.5, pattern="tr38901", polarization="VH"),  # LTE 2T2R
        2.1: dict(num_rows=1, num_cols=1, vertical_spacing=0.5,
                  horizontal_spacing=0.5, pattern="tr38901", polarization="VH"),  # LTE 2T2R
        3.5: dict(num_rows=4, num_cols=4, vertical_spacing=0.5,
                  horizontal_spacing=0.5, pattern="tr38901", polarization="VH"),  # NR 32T
    }

    def __init__(self):
        """Initialize the Sionna RT channel model with default RF parameters.

        Raises:
            ImportError: If the sionna package is not installed.

        Side effects:
            - Creates a PathSolver instance.
            - Initialises per-gNB storage dicts (gnb_configs, ue_configs,
              _gnb_pool, _gnb_rf_params).
            - Sets up SINRMapLookup, PHYChainBLER, and SlidingBLER instances.
            - Does NOT load a scene; call configure() to set scene_path.
        """
        if not SIONNA_AVAILABLE:
            raise ImportError("Sionna is required.")

        self._config: Optional[ChannelConfig] = None
        self._configured = False

        # Global defaults (fallback only)
        self.scene = None
        self.frequency_hz = self.DEFAULT_FREQUENCY_HZ
        self.tx_power_dbm = self.DEFAULT_TX_POWER_DBM
        self.bandwidth_hz = self.DEFAULT_BANDWIDTH_HZ
        self.noise_figure_db = self.DEFAULT_NOISE_FIGURE_DB
        self.noise_floor_dbm = compute_noise_floor(self.DEFAULT_BANDWIDTH_HZ,
                                                    self.DEFAULT_NOISE_FIGURE_DB)

        self.max_depth = 6
        self.num_samples = 1e6
        self.diffraction = True
        self.scattering = True
        self.num_time_steps = 14
        self.sampling_frequency = 1 / 35.7e-6

        self._path_solver = PathSolver()
        self._scene_path: Optional[str] = None

        # Storage
        self.gnb_configs: Dict[Tuple[int, int], GnbConfig] = {}
        self.ue_configs: Dict[int, UeConfig] = {}
        self._gnb_pool: Dict[Tuple[int, int], GnbConfig] = {}

        # ★ NEW: per-gnb RF params (fields not in GnbConfig)
        self._gnb_rf_params: Dict[Tuple[int, int], dict] = {}

        # SINR map + PHY BLER
        self._sinr_lookup = SINRMapLookup()
        self._phy_bler = PHYChainBLER()
        self._bler_method = "full_chain"
        self._bler_mcs_index = 9
        self._last_cir_cache: Dict[float, Tuple] = {}  # freq_ghz → (a, tau)

        self._bler_mode = "full_chain"  # "full_chain", "mockup", "sigmoid"
        self._mockup_bler: Optional[MockupBLERProvider] = None

        # ★ v3: Sliding window BLER (3GPP TS 36.133 방식)
        # window_tbs=200: 200ms 윈도우 (OOS/IS 판정 기준)
        # tbs_per_step=20: 20ms step / 1ms TTI = 20 TBs
        self._sliding_bler = SlidingBLER(window_tbs=200, tbs_per_step=20)

        logger.info("SionnaRTChannelModel v2 initialized")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @property
    def model_type(self) -> ChannelModelType:
        """Return the channel model type identifier."""
        return ChannelModelType.SIONNA_RT

    def configure(self, config: ChannelConfig) -> None:
        """Apply a ChannelConfig and initialise the Sionna RT scene.

        Args:
            config: ChannelConfig dataclass holding RF parameters (frequency_hz,
                tx_power_dbm, bandwidth_hz, noise_figure_db, max_depth,
                num_samples, diffraction, scattering, scene_path) and optional
                fields gnb_noise_figure_db, ue_tx_power_dbm,
                surface_distortion_mean_db, surface_distortion_std_db.

        Side effects:
            - Updates all RF parameter attributes on self.
            - Calls _init_scene(config.scene_path) to load the 3D scene.
            - Recomputes global noise floor (dBm).
            - Sets self._configured = True.
        """
        self._config = config
        self.frequency_hz = config.frequency_hz
        self.tx_power_dbm = config.tx_power_dbm
        self.bandwidth_hz = config.bandwidth_hz
        self.noise_figure_db = config.noise_figure_db
        self.max_depth = config.max_depth
        self.num_samples = config.num_samples
        self.diffraction = config.diffraction
        self.scattering = config.scattering

        self._surface_distortion_mean_db = config.surface_distortion_mean_db
        self._surface_distortion_std_db = config.surface_distortion_std_db

        # Global noise floor (fallback)
        self.noise_floor_dbm = compute_noise_floor(
            self.bandwidth_hz, self.noise_figure_db
        )

        self._gnb_noise_figure_db = getattr(config, 'gnb_noise_figure_db', 2.5)
        self._ue_tx_power_dbm = getattr(config, 'ue_tx_power_dbm', 23.0)
        self._ul_noise_floor_dbm = compute_noise_floor(
            self.bandwidth_hz, self._gnb_noise_figure_db
        )

        self._scene_path = config.scene_path
        self._init_scene(config.scene_path)
        self._configured = True

        logger.info(
            f"Configured: freq={self.frequency_hz/1e9:.2f}GHz, "
            f"bw={self.bandwidth_hz/1e6:.0f}MHz (global defaults)"
        )

    def load_sinr_maps(self, map_dir: str) -> int:
        """Load precomputed SINR maps from a directory for Doppler penalty lookup.

        Args:
            map_dir: Directory containing sinr_map_*.npz files.

        Returns:
            Number of maps successfully loaded.
        """
        return self._sinr_lookup.load_all(map_dir)

    def set_bler_config(self, method="full_chain", mcs_index=9, batch_size=1,
                        window_tbs=200, tbs_per_step=20):
        """Configure BLER estimation parameters.

        Args:
            method: BLER method for the full PHY chain — "full_chain" (Sionna
                LDPC+QAM Monte Carlo) or "eesm" (EESM abstraction).
            mcs_index: Default MCS index (0-28) when not supplied per-call.
            batch_size: Codewords per Monte Carlo batch for full_chain mode.
            window_tbs: Sliding BLER window size in transport blocks
                (200 = 200 ms at 1 ms TTI per 3GPP TS 36.133 §7.6).
            tbs_per_step: Transport blocks transmitted per simulation step.

        Side effects:
            - Updates self._bler_method, self._bler_mcs_index,
              self._phy_bler.default_batch_size.
            - Replaces self._sliding_bler with a fresh SlidingBLER instance.
        """
        self._bler_method = method
        self._bler_mcs_index = mcs_index
        self._phy_bler.default_batch_size = batch_size
        self._sliding_bler = SlidingBLER(
            window_tbs=window_tbs, tbs_per_step=tbs_per_step
        )

    def set_bler_mode(self, mode: str, csv_dir: str = "output/bler_curves",
                      sinr_offset_db: float = 0.0) -> None:
        """Set BLER computation mode.

        Args:
            mode: "full_chain" (Sionna CIR→PHY), "mockup" (CSV lookup), "sigmoid" (analytical)
            csv_dir: Directory with bler_all_mcs.csv (for mockup mode)
            sinr_offset_db: Global SINR offset for mockup lookup
        """
        if mode not in ("full_chain", "mockup", "sigmoid"):
            raise ValueError(f"Unknown BLER mode: {mode}. Use: full_chain, mockup, sigmoid")
        self._bler_mode = mode
        if mode == "mockup":
            self._mockup_bler = MockupBLERProvider(csv_dir, sinr_offset_db)
        logger.info(f"BLER mode set to: {mode}")

    def reset_sliding_bler(self, ue_id: int) -> None:
        """
        핸드오버 시 호출: 새 서빙셀 기준으로 BLER 윈도우 초기화.
        run_simulation.py에서 핸드오버 완료 후 호출:
            channel_model.reset_sliding_bler(ue_id)
        """
        self._sliding_bler.reset_ue(ue_id)

    def get_sliding_bler_info(self, ue_id: int) -> dict:
        """디버깅/로깅용: 슬라이딩 윈도우 상태 조회."""
        return self._sliding_bler.get_window_info(ue_id)

    # ------------------------------------------------------------------
    # Per-gNB noise floor
    # ------------------------------------------------------------------
    def _get_gnb_noise_floor(self, gnb_id: int, sector_id: int) -> float:
        """
        Per-gNB per-RE noise floor (matches per-RE RSRP).

        3GPP Reference:
            TS 38.215 Section 5.1.1 (RSRP is per-RE power)
        """
        rf = self._get_gnb_rf(gnb_id, sector_id)
        return compute_noise_floor_per_re(
            rf["bandwidth_mhz"],
            rf["scs_hz"],
            self.noise_figure_db
        )

    def _get_gnb_rf(self, gnb_id: int, sector_id: int) -> dict:
        """Per-gnb RF 파라미터 조회."""
        rf = self._gnb_rf_params.get((gnb_id, sector_id), {})
        gnb = self.gnb_configs.get((gnb_id, sector_id))
        freq = gnb.frequency_ghz if gnb else self.frequency_hz / 1e9
        return {
            "frequency_ghz": freq,
            "bandwidth_mhz": rf.get("bandwidth_mhz", default_bandwidth_mhz(freq)),
            "scs_hz": rf.get("scs_hz", default_scs_hz(freq)),
        }

    # ------------------------------------------------------------------
    # Scene init
    # ------------------------------------------------------------------
    def _get_tx_array_for_freq(self, freq_ghz: float) -> dict:
        """Get TX array config for a given frequency (nearest match)."""
        nearest = min(self.FREQ_TX_ARRAY_CONFIGS.keys(),
                      key=lambda k: abs(k - freq_ghz))
        return self.FREQ_TX_ARRAY_CONFIGS[nearest]

    def _init_scene(self, scene_path: Optional[str] = None) -> None:
        """Load (or reload) the Sionna RT 3D scene and configure antenna arrays.

        Args:
            scene_path: Path to a Mitsuba .xml scene file. If None or the path
                does not exist, falls back to the built-in Sionna Munich scene.

        Side effects:
            - Assigns self.scene with the loaded Sionna Scene object.
            - Sets scene.frequency to self.frequency_hz.
            - Sets scene.tx_array (per-frequency PlanarArray) and scene.rx_array
              (1×2 VH UE receive array).
        """
        if scene_path and os.path.exists(scene_path):
            self.scene = load_scene(scene_path)
        else:
            self.scene = load_scene(sionna.rt.scene.munich)

        self.scene.frequency = self.frequency_hz

        # Initial TX array from default frequency (overridden per-freq in _compute_paths_for_frequency)
        freq_ghz = self.frequency_hz / 1e9
        tx_params = self._get_tx_array_for_freq(freq_ghz)
        self.scene.tx_array = PlanarArray(**tx_params)

        self.scene.rx_array = PlanarArray(
            num_rows=1, num_cols=2, vertical_spacing=0.5,
            horizontal_spacing=0.5, pattern="tr38901", polarization="VH"
        )

    # ------------------------------------------------------------------
    # add_gnb (v2: per-gnb frequency/bandwidth/scs)
    # ------------------------------------------------------------------
    def add_gnb(self, gnb_id: int, position: Tuple[float, float, float],
                **kwargs) -> None:
        """Register a gNB/eNB and add its transmitter to the Sionna scene.

        Args:
            gnb_id: Unique integer identifier for the base station.
            position: 3-D world position (x, y, z) in metres.
            **kwargs:
                sector_id (int): Sector index, default 1.
                azimuth_deg (float): Horizontal beam direction, default 0.
                downtilt_deg (float): Vertical tilt (negative = down), default -1.
                tx_power_dbm (float): Conducted TX power, defaults to model global.
                antenna_gain_dbi (float): Antenna gain, default 23.
                antenna_num_ports (int): Number of TX ports, default 2.
                frequency_ghz (float): Carrier frequency, defaults to model global.
                bandwidth_mhz (float): Channel bandwidth; derived from frequency if absent.
                scs_hz (float): Subcarrier spacing; derived from frequency if absent.
                name (str): Human-readable label, default "gNB_<gnb_id>".
                rat_type (str): "nr" or "lte", default "nr".

        Side effects:
            - Stores GnbConfig in self.gnb_configs[(gnb_id, sector_id)].
            - Stores RF params in self._gnb_rf_params[(gnb_id, sector_id)].
            - Adds (or replaces) a Sionna Transmitter named gNB_<gnb_id>_S<sector_id>
              in self.scene.

        Raises:
            RuntimeError: If configure() has not been called.
        """
        if not self._configured:
            raise RuntimeError("Channel model not configured.")

        sector_id = kwargs.get('sector_id', 1)
        azimuth_deg = kwargs.get('azimuth_deg', 0.0)
        downtilt_deg = kwargs.get('downtilt_deg', -1.0)
        tx_power_dbm = kwargs.get('tx_power_dbm', self.tx_power_dbm)
        antenna_gain_dbi = kwargs.get('antenna_gain_dbi', 23.0)
        antenna_num_ports = kwargs.get('antenna_num_ports', 2)
        name = kwargs.get('name', f"gNB_{gnb_id}")

        # ★ per-gnb frequency (not global!)
        freq_ghz = kwargs.get('frequency_ghz', self.frequency_hz / 1e9)

        config = GnbConfig(
            gnb_id=gnb_id, name=name, sector_id=sector_id,
            position=position, azimuth_deg=azimuth_deg,
            downtilt_deg=downtilt_deg, tx_power_dbm=tx_power_dbm,
            frequency_ghz=freq_ghz,
            antenna_gain_dbi=antenna_gain_dbi,
            antenna_num_ports=antenna_num_ports,
            rat_type=kwargs.get('rat_type', 'nr'),
        )

        key = (gnb_id, sector_id)
        self.gnb_configs[key] = config

        # ★ per-gnb RF params
        self._gnb_rf_params[key] = {
            "bandwidth_mhz": kwargs.get('bandwidth_mhz',
                                        default_bandwidth_mhz(freq_ghz)),
            "scs_hz": kwargs.get('scs_hz', default_scs_hz(freq_ghz)),
        }

        # Add to scene
        tx_name = f"gNB_{gnb_id}_S{sector_id}"
        if tx_name in self.scene.transmitters:
            self.scene.remove(tx_name)# 이름이 있을때 지우고 다시 붙이는 이유 혹시 겹치는 pci 때문에 그럼
            

        self.scene.add(Transmitter(
            name=tx_name,
            position=[float(p) for p in position],
            orientation=[float(np.deg2rad(azimuth_deg)),
                         float(np.deg2rad(downtilt_deg)), 0.0],
        ))

    def add_ue(self, ue_id, position, **kwargs):
        """Register a UE and add its receiver to the Sionna scene.

        Args:
            ue_id: UE identifier (int or str).
            position: 3-D world position (x, y, z) in metres.
            **kwargs:
                car_id: Optional vehicle identifier.
                timestamp (float): Current simulation time in seconds, default 0.
                velocity (tuple): 3-D velocity vector (vx, vy, vz) m/s.

        Side effects:
            - Stores UeConfig in self.ue_configs[ue_id].
            - Adds (or replaces) a Sionna Receiver named UE_<ue_id> in self.scene.
            - If velocity is provided, sets receiver.velocity.

        Raises:
            RuntimeError: If configure() has not been called.
        """
        if not self._configured:
            raise RuntimeError("Channel model not configured.")
        config = UeConfig(
            ue_id=ue_id, position=position,
            car_id=kwargs.get('car_id'),
            timestamp=kwargs.get('timestamp', 0.0),
            velocity=kwargs.get('velocity'),
        )
        self.ue_configs[ue_id] = config
        rx_name = f"UE_{ue_id}"
        if rx_name in self.scene.receivers:
            self.scene.remove(rx_name)
        receiver = Receiver(name=rx_name,
                            position=[float(p) for p in position],
                            orientation=[0.0, 0.0, 0.0])
        if config.velocity:
            receiver.velocity = [float(v) for v in config.velocity]
        self.scene.add(receiver)

    def update_ue_position(self, ue_id, position, velocity=None):
        """Update position (and optionally velocity) of an already-registered UE.

        Args:
            ue_id: UE identifier matching a prior add_ue call.
            position: New 3-D world position (x, y, z) in metres.
            velocity: Optional 3-D velocity vector (vx, vy, vz) in m/s.

        Side effects:
            - Updates self.ue_configs[ue_id].position (and .velocity if given).
            - Propagates the update to the Sionna Receiver in self.scene if present.
        """
        config = self.ue_configs[ue_id]
        config.position = position
        if velocity is not None:
            config.velocity = velocity
        rx_name = f"UE_{ue_id}"
        if rx_name in self.scene.receivers:
            self.scene.receivers[rx_name].position = [float(p) for p in position]
            if velocity is not None:
                self.scene.receivers[rx_name].velocity = [float(v) for v in velocity]

    # ------------------------------------------------------------------
    # CSV loading (v2: per-gnb RF params)
    # ------------------------------------------------------------------
    def load_gnb_from_csv(self, csv_path: str) -> int:
        """Load all gNB/eNB sectors from a CSV file, replacing any existing configs.

        Args:
            csv_path: Path to enb_coordinates_converted_<region>.csv. Required
                columns are those consumed by GnbConfig.from_csv_row(). Optional
                columns bandwidth_mhz and scs_hz set per-gNB RF params; values
                are derived from frequency_ghz when absent.

        Returns:
            Number of gNB sectors loaded.

        Side effects:
            - Clears self.gnb_configs and self._gnb_rf_params.
            - Removes all existing Transmitters from self.scene.
            - Adds a new Transmitter to self.scene for each loaded sector.

        Raises:
            RuntimeError: If configure() has not been called.
        """
        if not self._configured:
            raise RuntimeError("Channel model not configured.")
        df = pd.read_csv(csv_path)
        self.gnb_configs.clear()
        self._gnb_rf_params.clear()
        for tx_name in list(self.scene.transmitters.keys()):
            self.scene.remove(tx_name)
        count = 0
        for _, row in df.iterrows():
            config = GnbConfig.from_csv_row(row)
            key = (config.gnb_id, config.sector_id)
            self.gnb_configs[key] = config

            # ★ CSV에서 bandwidth_mhz, scs_hz 읽기 (없으면 기본값)
            row_lower = {k.strip().lower(): v for k, v in row.items()}
            bw_mhz = float(row_lower.get("bandwidth_mhz",
                           default_bandwidth_mhz(config.frequency_ghz)))
            scs = float(row_lower.get("scs_hz",
                        default_scs_hz(config.frequency_ghz)))
            self._gnb_rf_params[key] = {
                "bandwidth_mhz": bw_mhz,
                "scs_hz": scs,
            }

            tx_name = f"gNB_{config.gnb_id}_S{config.sector_id}"
            self.scene.add(Transmitter(
                name=tx_name, position=list(config.position),
                orientation=[np.deg2rad(config.azimuth_deg),
                             np.deg2rad(config.downtilt_deg), 0.0],
            ))
            count += 1
        logger.info(f"Loaded {count} gNB sectors from {csv_path}")
        return count

    # ------------------------------------------------------------------
    # Multi-frequency path computation
    # ------------------------------------------------------------------
    def _group_gnbs_by_frequency(self, tol=0.05) -> Dict[float, List[Tuple[int, int]]]:
        """Active gNBs를 주파수별로 그룹핑."""
        groups: Dict[float, List] = {}
        for (gnb_id, sec_id), gnb_cfg in self.gnb_configs.items():
            freq = gnb_cfg.frequency_ghz
            matched = None
            for k in groups:
                if abs(k - freq) < tol:
                    matched = k
                    break
            if matched is not None:
                groups[matched].append((gnb_id, sec_id))
            else:
                groups[freq] = [(gnb_id, sec_id)]
        return groups

    def _compute_paths_for_frequency(self, freq_ghz: float) -> Any:
        """
        ★ 특정 주파수로 scene.frequency + tx_array 설정 후 path 계산.
        주파수별 안테나 배열을 매칭 (LTE 2T2R vs NR 32T).
        (Sionna RT는 scene.frequency 기준으로 재질 반사/투과 계수를 결정)
        """
        import time as _time
        _tp = _time.perf_counter()
        self.scene.frequency = freq_ghz * 1e9

        # ★ Per-frequency TX antenna array (matches precompute_sinr_map.py)
        nearest_freq = min(self.FREQ_TX_ARRAY_CONFIGS.keys(),
                          key=lambda k: abs(k - freq_ghz))
        tx_params = self.FREQ_TX_ARRAY_CONFIGS[nearest_freq]
        self.scene.tx_array = PlanarArray(**tx_params)
        logger.debug(f"  TX array for {freq_ghz}GHz: {tx_params['num_rows']}x"
                     f"{tx_params['num_cols']}x{tx_params['polarization']}")
        try:
            num_tx = len(self.scene.transmitters)
            max_samples = int(min(self.num_samples, 4e9 / max(num_tx * 10, 1)))
            max_samples = max(max_samples, 10000)
            paths = self._path_solver(
                scene=self.scene,
                max_depth=self.max_depth,
                samples_per_src=max_samples,
                synthetic_array=True,
                los=True,
                specular_reflection=True,
                diffuse_reflection=self.scattering,
                refraction=True,
                diffraction=self.diffraction,
            )
            logger.debug(f"[TIMING] _compute_paths ({freq_ghz}GHz): {_time.perf_counter() - _tp:.3f}s")
            return paths
        except Exception as e:
            logger.error(f"Path computation failed at {freq_ghz}GHz: {e}")
            return None

    # ------------------------------------------------------------------
    # RSRP from CIR (unchanged)
    # ------------------------------------------------------------------
    def _compute_rsrp_from_cir(self, a, tau, tx_idx, rx_idx,
                                tx_power_dbm,
                                bandwidth_mhz=100.0,
                                scs_hz=30000.0):
        """Compute RSRP (dBm), RMS delay spread (ns), and path count from a CIR tensor.

        Implements 3GPP TS 38.215 Section 5.1.1 per-RE RSRP definition:
        total path power is divided by the number of subcarriers (n_sc) so that
        RSRP reflects the power of a single resource element.

        Args:
            a: Complex path amplitude tensor, shape (rx, [polarisation,] tx,
                [streams,] paths [, time]) as returned by Sionna paths.cir().
                May also be a tuple (real, imag) for numpy-serialised tensors.
            tau: Path delay tensor (seconds), compatible shape with a.
            tx_idx: Transmitter index within the CIR batch dimension.
            rx_idx: Receiver index within the CIR batch dimension.
            tx_power_dbm: Conducted TX power of this gNB sector in dBm.
            bandwidth_mhz: Channel bandwidth in MHz (used to compute n_sc).
            scs_hz: Subcarrier spacing in Hz (used to compute n_sc).

        Returns:
            Tuple (rsrp_dbm, delay_spread_ns, num_paths):
                rsrp_dbm: Per-RE RSRP in dBm; -200.0 if no valid paths.
                delay_spread_ns: RMS delay spread in nanoseconds.
                num_paths: Number of valid (non-zero, within max delay) paths.
        """
        try:
            if isinstance(a, tuple) and len(a) == 2:
                a_np = np.sqrt(np.array(a[0]) ** 2 + np.array(a[1]) ** 2)
            else:
                a_np = np.abs(np.array(a))
            tau_np = np.array(tau)

            if a_np.ndim == 6:
                a_pair = a_np[rx_idx, 0, tx_idx, :, :, :]
                path_powers = np.sum(np.abs(a_pair) ** 2, axis=0)
                path_powers = np.mean(path_powers, axis=-1)
                tau_pair = (tau_np[rx_idx, tx_idx, :] if tau_np.ndim == 3
                            else tau_np[rx_idx, 0, tx_idx, 0, :] if tau_np.ndim == 5
                            else np.zeros(path_powers.shape))
            elif a_np.ndim == 4:
                a_pair = a_np[rx_idx, tx_idx, :, :]
                path_powers = np.mean(np.abs(a_pair) ** 2, axis=-1)
                tau_pair = (tau_np[rx_idx, tx_idx, :] if tau_np.ndim == 3
                            else np.zeros(path_powers.shape))
            else:
                return -200.0, 0.0, 0

            max_delay_s = getattr(self._config, 'max_path_length_m', 1500.0) / 3e8
            valid_mask = (path_powers > 1e-20) & (tau_pair <= max_delay_s)
            num_paths = int(np.sum(valid_mask))
            if num_paths == 0:
                return -200.0, 0.0, 0

            total_power = np.sum(path_powers[valid_mask])
            rsrp_dbm = tx_power_dbm + 10 * np.log10(total_power + 1e-30)

            # Per-RE normalization (3GPP TS 38.215 Section 5.1.1)
            n_sc = _fft_size_from_bw_scs(bandwidth_mhz, scs_hz)
            rsrp_dbm -= 10.0 * np.log10(n_sc)

            if num_paths > 1:
                p_norm = path_powers[valid_mask] / np.sum(path_powers[valid_mask])
                tau_valid = tau_pair[valid_mask]
                tau_mean = np.sum(p_norm * tau_valid)
                delay_spread_ns = np.sqrt(np.sum(p_norm * (tau_valid - tau_mean) ** 2)) * 1e9
            else:
                delay_spread_ns = 0.0
            return rsrp_dbm, delay_spread_ns, num_paths
        except Exception as e:
            logger.error(f"RSRP from CIR error: {e}")
            return -200.0, 0.0, 0



    def _compute_distance(self, pos1, pos2):
        """Return Euclidean distance in metres between two 3-D position tuples."""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    def _extract_link_cir(self, a, tau, tx_idx, rx_idx):
        """Extract per-link complex CIR tensors for a specific TX/RX pair.

        Args:
            a: Full batch amplitude tensor from paths.cir(out_type="numpy").
                Accepted shapes: 6-D (rx, pol, tx, streams, paths, time) or
                4-D (rx, tx, paths, time). May be a (real, imag) tuple.
            tau: Delay tensor matching a's rx/tx dimensions.
            tx_idx: Index of the transmitter in the batch.
            rx_idx: Index of the receiver in the batch.

        Returns:
            Tuple (a_link, tau_link):
                a_link: Complex ndarray of shape (ra, ta, paths, time) for the
                    requested link, or None on error/unsupported shape.
                tau_link: 1-D delay array (paths,) in seconds, or None on error.
        """
        try:
            if isinstance(a, tuple) and len(a) == 2:
                a_complex = np.array(a[0]) + 1j * np.array(a[1])
            else:
                a_complex = np.array(a)
            tau_np = np.array(tau)

            if a_complex.ndim == 6:
                a_link = a_complex[rx_idx, :, tx_idx, :, :, :]
                tau_link = (tau_np[rx_idx, tx_idx, :] if tau_np.ndim == 3
                            else tau_np[rx_idx, 0, tx_idx, 0, :] if tau_np.ndim == 5
                            else np.zeros(a_link.shape[2]))
            elif a_complex.ndim == 4:
                a_link = a_complex[rx_idx, tx_idx, :, :]
                a_link = a_link[np.newaxis, np.newaxis, :, :]
                tau_link = (tau_np[rx_idx, tx_idx, :] if tau_np.ndim == 3
                            else np.zeros(a_link.shape[2]))
            else:
                return None, None
            return a_link, tau_link
        except Exception as e:
            logger.error(f"CIR extraction error: {e}")
            return None, None

    # ==================================================================
    # compute_all() v2 — MAIN
    # ==================================================================
    def compute_all(
        self,
        timestamp: Optional[float] = None,
        mcs_index: Optional[int] = None,
        serving_cells: Optional[Dict[int, int]] = None,
    ) -> Dict[Tuple[int, int, int], ChannelState]:
        """
        ★ v2 Changes:
          - serving_cells: {ue_id: gnb_id} — BLER을 이 셀 기준으로 계산
          - Per-frequency-group RT path computation
          - Per-link noise floor from gnb's bandwidth
        """
        if not self._configured:
            raise RuntimeError("Channel model not configured.")

        import time as _time
        _t_all_start = _time.perf_counter()

        mcs = mcs_index if mcs_index is not None else self._bler_mcs_index
        serving_cells = serving_cells or {}
        results: Dict[Tuple[int, int, int], ChannelState] = {}
        BEAM_ISOLATION_DB = 15.0  # RT model: antenna patterns already in RSRP via ray tracing (precomputed maps use 0.0)

        has_moving_ue = any(
            ue.velocity is not None and any(v != 0 for v in ue.velocity)
            for ue in self.ue_configs.values()
        )

        # === Step 1: Per-frequency-group path computation ===
        _t1 = _time.perf_counter()
        freq_groups = self._group_gnbs_by_frequency()
        tx_name_to_idx = {n: i for i, n in enumerate(self.scene.transmitters.keys())}
        rx_name_to_idx = {n: i for i, n in enumerate(self.scene.receivers.keys())}

        # ★ 주파수별 CIR 캐시
        freq_cir: Dict[float, Tuple] = {}  # freq → (a, tau)
        for freq_ghz in freq_groups:
            paths = self._compute_paths_for_frequency(freq_ghz)
            if paths is not None:
                try:
                    if has_moving_ue:
                        a, tau = paths.cir(
                            num_time_steps=self.num_time_steps,
                            sampling_frequency=self.sampling_frequency,
                            out_type="numpy",
                        )
                    else:
                        a, tau = paths.cir(out_type="numpy")
                    freq_cir[freq_ghz] = (a, tau)
                except Exception as e:
                    logger.error(f"CIR error at {freq_ghz}GHz: {e}")
        self._last_cir_cache = freq_cir
        logger.debug(f"[TIMING] Step 1 (paths): {_time.perf_counter() - _t1:.3f}s "
                     f"({len(freq_groups)} freq groups)")

        # === Step 2: RSRP per link ===
        _t2 = _time.perf_counter()
        rsrp_per_ue: Dict[int, Dict[Tuple[int, int], float]] = {}
        for ue_id, ue_config in self.ue_configs.items():
            rsrp_per_ue[ue_id] = {}
            for (gnb_id, sector_id), gnb_config in self.gnb_configs.items():
                tx_name = f"gNB_{gnb_id}_S{sector_id}"
                rx_name = f"UE_{ue_id}"
                distance_m = self._compute_distance(ue_config.position, gnb_config.position)
                conducted_power = gnb_config.tx_power_dbm

                # 해당 gNB 주파수에 해당하는 CIR 사용
                gnb_freq = gnb_config.frequency_ghz
                matched_freq = self._match_freq_key(gnb_freq, freq_cir)
                a, tau = freq_cir.get(matched_freq, (None, None)) if matched_freq else (None, None)

                if a is not None and tau is not None:
                    tx_idx = tx_name_to_idx.get(tx_name)
                    rx_idx = rx_name_to_idx.get(rx_name)
                    if tx_idx is not None and rx_idx is not None:
                        rf = self._get_gnb_rf(gnb_id, sector_id)
                        rsrp_dbm, delay_ns, n_paths = self._compute_rsrp_from_cir(
                            a, tau, tx_idx, rx_idx, conducted_power,
                            bandwidth_mhz=rf["bandwidth_mhz"],
                            scs_hz=rf["scs_hz"])
                    else:
                        rsrp_dbm, delay_ns, n_paths = -200.0, 0.0, 0
                else:
                    rsrp_dbm, delay_ns, n_paths = -200.0, 0.0, 0

                # Propagation losses
                pen_loss = getattr(self._config, 'penetration_loss_db', 0.0) if self._config else 0.0
                surf_mean = getattr(self, '_surface_distortion_mean_db', 0.0)
                surf_std = getattr(self, '_surface_distortion_std_db', 0.0)
                if rsrp_dbm > -200.0:
                    rsrp_dbm, _ = apply_propagation_losses(rsrp_dbm, pen_loss, surf_mean, surf_std)

                rsrp_per_ue[ue_id][(gnb_id, sector_id)] = rsrp_dbm

                state = ChannelState(
                    ue_id=ue_id, gnb_id=gnb_id, sector_id=sector_id,
                    rsrp_dbm=rsrp_dbm, delay_spread_ns=delay_ns,
                    num_paths=n_paths, distance_m=distance_m,
                    timestamp=timestamp or ue_config.timestamp,
                    path_loss_db=conducted_power - rsrp_dbm,
                    rat_type=gnb_config.rat_type,
                )
                results[(ue_id, gnb_id, sector_id)] = state
        logger.debug(f"[TIMING] Step 2 (RSRP): {_time.perf_counter() - _t2:.3f}s "
                     f"({len(results)} links)")

        # === Step 3: SINR (per-link noise floor) + Doppler penalty from map ===
        _t3 = _time.perf_counter()
        # ★ Build frequency lookup for co-channel interference filtering
        _freq_map = {(gid, sid): cfg.frequency_ghz
                     for (gid, sid), cfg in self.gnb_configs.items()}
        _doppler_applied = 0
        _doppler_missed = 0
        _serving_debug = {}  # ue_id → debug info for serving cell
        for (ue_id, gnb_id, sector_id), state in results.items():
            ue_config = self.ue_configs.get(ue_id)
            gnb_config = self.gnb_configs.get((gnb_id, sector_id))
            freq_ghz = gnb_config.frequency_ghz if gnb_config else self.frequency_hz / 1e9
            link_noise_floor = self._get_gnb_noise_floor(gnb_id, sector_id)

            # RSRP-ratio SINR — co-channel interference only (same frequency)
            # Different frequencies don't cause co-channel interference
            interferer_rsrps = [
                r for (og, os), r in rsrp_per_ue[ue_id].items()
                if (og != gnb_id or os != sector_id)
                and abs(_freq_map.get((og, os), -1) - freq_ghz) < 0.05
            ]
            sinr_base = compute_sinr_linear_components(
                state.rsrp_dbm, interferer_rsrps,
                link_noise_floor, BEAM_ISOLATION_DB
            )[0]
            state.sinr_db = sinr_base
            state.sinr_base_db = sinr_base

            # ★ Doppler penalty from precomputed SINR map
            #   Map은 시뮬레이션 시작시 한번 로드, UE (x,y) 위치로 조회
            #   penalty = sinr_doppler - sinr_static (항상 <= 0)
            #   Serving cell 무관 → 위치 기반 Doppler 감쇄만 적용
            penalty = None
            ##

            if self._sinr_lookup.loaded and ue_config:
                penalty = self._sinr_lookup.get_doppler_penalty(
                    ue_config.position[0], ue_config.position[1],
                    freq_ghz=freq_ghz,
                )
                if penalty is not None:
                    state.sinr_db += penalty
                    state.doppler_penalty_db = penalty
                    _doppler_applied += 1
                else:
                    _doppler_missed += 1
                

            # Collect debug info for serving cell (logged below)
            serving_gnb = serving_cells.get(ue_id)
            if serving_gnb == gnb_id:
                _serving_debug[ue_id] = dict(
                    gnb_id=gnb_id, freq_ghz=freq_ghz,
                    rsrp=state.rsrp_dbm, sinr_base=sinr_base,
                    penalty=penalty, sinr_final=state.sinr_db,
                    noise_floor=link_noise_floor,
                    n_interferers=len(interferer_rsrps),
                    pos=(ue_config.position[0], ue_config.position[1]) if ue_config else None,
                )

        # Debug: serving cell SINR breakdown
        for uid, d in _serving_debug.items():
            pen_str = f"{d['penalty']:.1f}dB" if d['penalty'] is not None else "N/A"
            logger.debug(
                f"  [SINR debug] UE{uid} serving eNB{d['gnb_id']} "
                f"({d['freq_ghz']:.1f}GHz): "
                f"RSRP={d['rsrp']:.1f}, noise={d['noise_floor']:.1f}, "
                f"interf={d['n_interferers']}, "
                f"SINR_base={d['sinr_base']:.1f} + Doppler={pen_str} "
                f"→ SINR={d['sinr_final']:.1f} dB "
                f"pos=({d['pos'][0]:.0f},{d['pos'][1]:.0f})" if d['pos'] else ""
            )

        if self._sinr_lookup.loaded and (_doppler_applied + _doppler_missed) > 0:
            total = _doppler_applied + _doppler_missed
            logger.info(f"  Doppler map: {_doppler_applied}/{total} applied "
                         f"({100*_doppler_applied/total:.0f}%), "
                         f"{_doppler_missed} missed")

        # === Step 3b: UL SINR ===
        for (ue_id, gnb_id, sector_id), state in results.items():
            gnb_config = self.gnb_configs.get((gnb_id, sector_id))
            if gnb_config is None:
                continue
            conducted_power = gnb_config.tx_power_dbm

            # ★ per-gnb UL noise floor
            rf = self._gnb_rf_params.get((gnb_id, sector_id), {})
            bw_mhz = rf.get("bandwidth_mhz", self.bandwidth_hz / 1e6)
            ul_noise_floor = compute_noise_floor(bw_mhz * 1e6, self._gnb_noise_figure_db)

            state.ul_sinr_db = compute_ul_sinr(
                dl_rsrp_dbm=state.rsrp_dbm,
                gnb_ref_tx_power_dbm=conducted_power,
                ue_tx_power_dbm=self._ue_tx_power_dbm,
                ul_noise_floor_dbm=ul_noise_floor,
            )
        logger.debug(f"[TIMING] Step 3 (SINR): {_time.perf_counter() - _t3:.3f}s")



        # === Step 4: BLER for serving cell ===
        _t4 = _time.perf_counter()
        from phy.bler_calculator import BLERCalculator
        _sigmoid = BLERCalculator()

        for ue_id in self.ue_configs:
            # ★ serving cell 결정: serving_cells 인수 우선, 없으면 best RSRP
            serving_gnb = serving_cells.get(ue_id)
            if serving_gnb is not None:
                # serving_cells에서 제공된 gnb_id → best RSRP sector 선택
                serving_key = None
                best_rsrp = -np.inf
                for (uid, gid, sid), st in results.items():
                    if uid == ue_id and gid == serving_gnb:
                        if st.rsrp_dbm > best_rsrp:
                            best_rsrp = st.rsrp_dbm
                            serving_key = (uid, gid, sid)
            else:
                # Fallback: best RSRP
                serving_key = None
                best_rsrp = -np.inf
                for (uid, gid, sid), st in results.items():
                    if uid == ue_id and st.rsrp_dbm > best_rsrp:
                        best_rsrp = st.rsrp_dbm
                        serving_key = (uid, gid, sid)

            if serving_key is None:
                logger.debug(f"BLER skip UE{ue_id}: serving_key not found")
                continue

            _, best_gnb, best_sec = serving_key
            state = results[serving_key]

            # Mode dispatch for BLER computation
            if self._bler_mode == "mockup" and self._mockup_bler:
                state.bler = self._mockup_bler.lookup(state.sinr_db, mcs)
                state.bler_instant = state.bler
            elif self._bler_mode == "sigmoid":
                state.bler = _sigmoid.sinr_to_bler(state.sinr_db, mcs)
                state.bler_instant = state.bler
            else:
                # Full chain: CIR → PHY encoding/decoding → BLER
                gnb_config = self.gnb_configs.get((best_gnb, best_sec))
                if gnb_config is None:
                    logger.debug(f"BLER skip UE{ue_id}: gnb_config not found for ({best_gnb},{best_sec})")
                    continue
                freq_ghz = gnb_config.frequency_ghz
                matched_freq = self._match_freq_key(freq_ghz, freq_cir)
                if matched_freq is None:
                    logger.debug(f"BLER skip UE{ue_id}: no CIR for {freq_ghz}GHz "
                                 f"(available: {list(freq_cir.keys())})")
                    continue
                a, tau = freq_cir[matched_freq]

                tx_name = f"gNB_{best_gnb}_S{best_sec}"
                rx_name = f"UE_{ue_id}"
                tx_idx = tx_name_to_idx.get(tx_name)
                rx_idx = rx_name_to_idx.get(rx_name)
                if tx_idx is None or rx_idx is None:
                    logger.debug(f"BLER skip UE{ue_id}: tx/rx name not in CIR "
                                 f"(tx={tx_name}→{tx_idx}, rx={rx_name}→{rx_idx})")
                    continue

                a_link, tau_link = self._extract_link_cir(a, tau, tx_idx, rx_idx)
                if a_link is None:
                    logger.debug(f"BLER skip UE{ue_id}: CIR extraction failed")
                    continue

                # ★ per-link noise + interference
                link_noise_floor = self._get_gnb_noise_floor(best_gnb, best_sec)
                noise_lin = 10 ** (link_noise_floor / 10.0)
                iso = 10 ** (-BEAM_ISOLATION_DB / 10.0)
                serving_freq = gnb_config.frequency_ghz
                interference = sum(
                    10 ** (r / 10.0) * iso
                    for (og, os), r in rsrp_per_ue[ue_id].items()
                    if (og != best_gnb or os != best_sec) and r > -200.0
                    and abs(_freq_map.get((og, os), -1) - serving_freq) < 0.05
                )
                total_noise = noise_lin + interference

                # ★ per-gnb SCS, bandwidth for PHY chain
                rf = self._get_gnb_rf(best_gnb, best_sec)

                # noise normalization for CIR domain
                n_sc = _fft_size_from_bw_scs(rf["bandwidth_mhz"], rf["scs_hz"])
                tx_power_per_sc = 10 ** (gnb_config.tx_power_dbm / 10.0) / max(n_sc, 1)
                noise_var_normalized = total_noise / tx_power_per_sc

                try:
                    raw_bler = self._phy_bler.estimate(
                        a_link=a_link, tau_link=tau_link,
                        freq_ghz=freq_ghz,
                        bandwidth_mhz=rf["bandwidth_mhz"],
                        scs_hz=rf["scs_hz"],
                        noise_var_linear=noise_var_normalized,
                        mcs_index=mcs,
                        method=self._bler_method,
                    )
                    smoothed_bler = self._sliding_bler.update(ue_id, raw_bler)
                    state.bler = smoothed_bler
                    state.bler_instant = raw_bler
                    logger.debug(f"[BLER] UE{ue_id}: raw={raw_bler:.4f}, smoothed={smoothed_bler:.4f}")
                except Exception as e:
                    logger.warning(f"BLER failed UE{ue_id}: {e}")
                    state.bler = 1.0
                    state.bler_instant = 1.0

        # Sigmoid fallback: serving cells where full chain failed
        for (uid, gid, sid), st in results.items():
            if st.bler is None:
                st.bler = _sigmoid.sinr_to_bler(st.sinr_db, mcs)
        logger.debug(f"[TIMING] Step 4 (BLER): {_time.perf_counter() - _t4:.3f}s")

        # === Step 5: Doppler ICI post-correction ===
        _t5 = _time.perf_counter()
        self._apply_doppler_correction(results, mcs)
        logger.debug(f"[TIMING] Step 5 (Doppler): {_time.perf_counter() - _t5:.3f}s")

        _t_all_end = _time.perf_counter()
        logger.debug(f"[TIMING] compute_all total: {_t_all_end - _t_all_start:.3f}s "
                     f"({len(results)} states, {len(self.ue_configs)} UEs)")
        logger.debug(f"Computed {len(results)} channel states")
        return results

    # ------------------------------------------------------------------
    # Doppler ICI post-processing (Russell & Stuber, IEEE VTC 1995)
    # ------------------------------------------------------------------
    def _apply_doppler_correction(self, results, mcs_index):
        """
        Doppler ICI 후처리: SINR 및 BLER을 속도 기반으로 보정.

        Inter-Carrier Interference (ICI) from Doppler acts as
        signal-proportional noise, creating a SINR ceiling:

            f_d = v · f_c / c
            ICI_ratio = (π · f_d / SCS)² / 3
            SINR_eff = SINR_static / (1 + SINR_linear · ICI_ratio)

        3GPP Reference: TR 38.901 §7.6.6 (Doppler modelling)
        ICI Model: Russell & Stuber, IEEE VTC 1995
        """
        from phy.bler_calculator import BLERCalculator
        _bc = BLERCalculator()
        c = 3e8

        for (ue_id, gnb_id, sector_id), state in results.items():
            ue = self.ue_configs.get(ue_id)
            if not ue or not ue.velocity:
                continue
            v_mps = np.sqrt(sum(v ** 2 for v in ue.velocity))
            if v_mps < 0.1:
                continue

            gnb_cfg = self.gnb_configs.get((gnb_id, sector_id))
            if not gnb_cfg:
                continue
            freq_hz = gnb_cfg.frequency_ghz * 1e9
            rf = self._gnb_rf_params.get((gnb_id, sector_id), {})
            scs_hz = rf.get("scs_hz", 30000.0)

            # Doppler frequency & ICI ratio
            f_d = v_mps * freq_hz / c
            ici_ratio = (np.pi * f_d / scs_hz) ** 2 / 3.0
            if ici_ratio < 1e-12:
                continue

            # DL SINR correction
            sinr_lin = 10 ** (state.sinr_db / 10.0)
            sinr_eff = sinr_lin / (1.0 + sinr_lin * ici_ratio)
            sinr_before = state.sinr_db
            state.sinr_db = float(10.0 * np.log10(max(sinr_eff, 1e-20)))

            # UL SINR correction
            ul_lin = 10 ** (state.ul_sinr_db / 10.0)
            ul_eff = ul_lin / (1.0 + ul_lin * ici_ratio)
            state.ul_sinr_db = float(10.0 * np.log10(max(ul_eff, 1e-20)))

            # BLER: Doppler can only worsen BLER
            if state.bler is not None:
                bler_static = state.bler
                bler_doppler = _bc.sinr_to_bler(state.sinr_db, mcs_index,
                                                fading=True)
                state.bler = max(bler_static, bler_doppler)

            degradation = sinr_before - state.sinr_db
            logger.debug(
                f"UE{ue_id}-gNB{gnb_id}: Doppler v={v_mps:.1f}m/s "
                f"f_d={f_d:.0f}Hz ICI={ici_ratio:.2e} "
                f"SINR_loss={degradation:.1f}dB"
            )

    def _match_freq_key(self, freq_ghz: float, freq_dict: dict,
                        tol: float = 0.05) -> Optional[float]:
        """Return the key in freq_dict whose value is within tol GHz of freq_ghz.

        Args:
            freq_ghz: Target frequency in GHz.
            freq_dict: Dictionary keyed by frequency (GHz), e.g. freq_cir.
            tol: Match tolerance in GHz (default 0.05).

        Returns:
            The matching float key, or None if no key is within tolerance.
        """
        for k in freq_dict:
            if abs(k - freq_ghz) < tol:
                return k
        return None

    # ------------------------------------------------------------------
    # compute_link_bler v2 (with interference)
    # ------------------------------------------------------------------
    def compute_link_bler(
        self,
        ue_id: int, gnb_id: int, sector_id: int,
        mcs_index: int,
        interference_dbm: Optional[float] = None,
        method: Optional[str] = None,
    ) -> Optional[float]:
        """
        ★ v2: interference_dbm 인수 추가.
        run_simulation.py에서 serving cell 결정 후 호출:
            bler = channel_model.compute_link_bler(
                ue_id, serving_gnb, sector, mcs,
                interference_dbm=computed_from_neighbors)
        """
        gnb_config = self.gnb_configs.get((gnb_id, sector_id))
        if gnb_config is None:
            return None
        freq_ghz = gnb_config.frequency_ghz

        matched = self._match_freq_key(freq_ghz, self._last_cir_cache)
        if matched is None:
            return None
        a, tau = self._last_cir_cache[matched]

        tx_to_idx = {n: i for i, n in enumerate(self.scene.transmitters.keys())}
        rx_to_idx = {n: i for i, n in enumerate(self.scene.receivers.keys())}

        tx_idx = tx_to_idx.get(f"gNB_{gnb_id}_S{sector_id}")
        rx_idx = rx_to_idx.get(f"UE_{ue_id}")
        if tx_idx is None or rx_idx is None:
            return None

        a_link, tau_link = self._extract_link_cir(a, tau, tx_idx, rx_idx)
        if a_link is None:
            return None

        # ★ per-link noise + interference
        link_nf = self._get_gnb_noise_floor(gnb_id, sector_id)
        noise_lin = 10 ** (link_nf / 10.0)
        if interference_dbm is not None:
            noise_lin += 10 ** (interference_dbm / 10.0)

        rf = self._get_gnb_rf(gnb_id, sector_id)

        # ★★★ FIX v3: tx_power_per_SC로 정규화 (compute_all Step4와 동일) ★★★
        n_sc = _fft_size_from_bw_scs(rf["bandwidth_mhz"], rf["scs_hz"])
        tx_power_per_sc = 10 ** (gnb_config.tx_power_dbm / 10.0) / max(n_sc, 1)
        noise_var_normalized = noise_lin / tx_power_per_sc

        return self._phy_bler.estimate(
            a_link=a_link, tau_link=tau_link,
            freq_ghz=freq_ghz,
            bandwidth_mhz=rf["bandwidth_mhz"],
            scs_hz=rf["scs_hz"],
            noise_var_linear=noise_var_normalized,
            mcs_index=mcs_index,
            method=method or self._bler_method,
        )

    # ------------------------------------------------------------------
    # Doppler (unchanged)
    # ------------------------------------------------------------------
    def compute_doppler(self, ue_id):
        """Compute Doppler shift and coherence time for a registered UE.

        Args:
            ue_id: UE identifier matching a prior add_ue call.

        Returns:
            DopplerInfo namedtuple with fields (ue_id, velocity_mps,
            doppler_shift_hz, coherence_time_ms), or None if the UE is
            unknown or has no velocity set.
        """
        ue = self.ue_configs.get(ue_id)
        if not ue or not ue.velocity:
            return None
        v_mag = np.sqrt(sum(v ** 2 for v in ue.velocity))
        f_d = v_mag * self.frequency_hz / 3e8
        t_c = 1 / (2 * f_d) if f_d > 0 else float('inf')
        return DopplerInfo(ue_id=ue_id, velocity_mps=ue.velocity,
                           doppler_shift_hz=f_d, coherence_time_ms=t_c * 1000)

    # ------------------------------------------------------------------
    # Serving / neighbor helpers (unchanged)
    # ------------------------------------------------------------------
    def get_serving_cell(self, ue_id, results=None):
        """Return the (gnb_id, sector_id) with the highest RSRP for a UE.

        Args:
            ue_id: UE identifier.
            results: Optional pre-computed dict from compute_all(). If None,
                compute_all() is called internally (expensive).

        Returns:
            Tuple (gnb_id, sector_id) of the best-RSRP cell, or None if no
            results exist for this UE.
        """
        if results is None:
            results = self.compute_all()
        best = None
        best_rsrp = -np.inf
        for (uid, gid, sid), st in results.items():
            if uid == ue_id and st.rsrp_dbm > best_rsrp:
                best_rsrp = st.rsrp_dbm
                best = (gid, sid)
        return best

    def get_neighbor_cells(self, ue_id, serving_gnb_id, serving_sector_id,
                           results=None, rsrp_threshold_dbm=-120.0):
        """Return neighbor ChannelState list for a UE, excluding the serving cell.

        Args:
            ue_id: UE identifier.
            serving_gnb_id: gNB ID of the current serving cell (excluded from output).
            serving_sector_id: Sector ID of the current serving cell (excluded).
            results: Optional pre-computed dict from compute_all(). If None,
                compute_all() is called internally (expensive).
            rsrp_threshold_dbm: Minimum RSRP (dBm) to include a neighbor.
                Default -120.0 dBm.

        Returns:
            List of ChannelState objects for qualifying neighbor cells, sorted
            by RSRP descending.
        """
        if results is None:
            results = self.compute_all()
        neighbors = []
        for (uid, gid, sid), st in results.items():
            if uid == ue_id and not (gid == serving_gnb_id and sid == serving_sector_id):
                if st.rsrp_dbm >= rsrp_threshold_dbm:
                    neighbors.append(st)
        neighbors.sort(key=lambda x: x.rsrp_dbm, reverse=True)
        return neighbors

    # ------------------------------------------------------------------
    # Pool / reset (unchanged)
    # ------------------------------------------------------------------
    def add_gnb_to_pool(self, gnb_id, position, **kwargs):
        """Register a gNB in the inactive pool for use with activate_nearby_gnbs.

        Unlike add_gnb(), this does NOT add a Transmitter to the scene. The pool
        allows dynamic scene management: only gNBs near the UE are activated at
        each simulation step to limit RT computation cost.

        Args:
            gnb_id: Unique integer identifier for the base station.
            position: 3-D world position (x, y, z) in metres.
            **kwargs:
                sector_id, azimuth_deg, downtilt_deg, tx_power_dbm,
                antenna_gain_dbi, antenna_num_ports, frequency_ghz,
                bandwidth_mhz, scs_hz, name, rat_type — same as add_gnb().

        Side effects:
            - Stores GnbConfig in self._gnb_pool[(gnb_id, sector_id)].
            - Stores RF params in self._gnb_rf_params[(gnb_id, sector_id)].
        """
        sector_id = kwargs.get('sector_id', 1)
        freq_ghz = kwargs.get('frequency_ghz', self.frequency_hz / 1e9)
        config = GnbConfig(
            gnb_id=gnb_id, name=kwargs.get('name', f"gNB_{gnb_id}"),
            sector_id=sector_id, position=position,
            azimuth_deg=kwargs.get('azimuth_deg', 0.0),
            downtilt_deg=kwargs.get('downtilt_deg', -1.0),
            tx_power_dbm=kwargs.get('tx_power_dbm', self.tx_power_dbm),
            frequency_ghz=freq_ghz,
            antenna_gain_dbi=kwargs.get('antenna_gain_dbi', 23.0),
            antenna_num_ports=kwargs.get('antenna_num_ports', 2),
            rat_type=kwargs.get('rat_type', 'nr'),
        )
        key = (gnb_id, sector_id)
        self._gnb_pool[key] = config
        # ★ pool에도 RF params 저장
        self._gnb_rf_params[key] = {
            "bandwidth_mhz": kwargs.get('bandwidth_mhz', default_bandwidth_mhz(freq_ghz)),
            "scs_hz": kwargs.get('scs_hz', default_scs_hz(freq_ghz)),
        }

    def reset_scene(self):
        """Clear active gNB and UE configs and remove all scene objects.

        The gNB pool (self._gnb_pool) is preserved so that activate_nearby_gnbs
        can repopulate the scene at the next timestep. Use clear() to also wipe
        the pool.

        Side effects:
            - Clears self.gnb_configs, self.ue_configs, self._gnb_rf_params.
            - Resets all sliding BLER windows.
            - Removes all Transmitters and Receivers from self.scene.
        """
        self.gnb_configs.clear()
        self.ue_configs.clear()
        self._gnb_rf_params.clear()
        self._sliding_bler.reset_all()
        if self.scene:
            for name in list(self.scene.transmitters.keys()):
                self.scene.remove(name)
            for name in list(self.scene.receivers.keys()):
                self.scene.remove(name)

    def activate_nearby_gnbs(self, center, radius_m=2000.0,
                             max_count=None, frequency_ghz=None,
                             min_count: int = 10):
        """Activate gNBs from the pool that are within radius_m of center.

        Implements dynamic scene management for the RT model: only nearby gNBs
        are added to the scene, reducing path-solver cost. Applies an adaptive
        fallback — if fewer than min_count candidates are within radius_m, the
        closest min_count pool entries are used instead.

        Optimises scene mutation: Transmitters no longer in the candidate set
        are removed; new ones are added; already-present ones are left untouched.

        Args:
            center: 3-D reference position (x, y, z) in metres (e.g. UE position).
            radius_m: Search radius in metres. Default 2000.
            max_count: Maximum number of gNBs to activate. None = no limit.
            frequency_ghz: If set, only activate gNBs matching this frequency
                (tolerance 0.01 GHz). None = all frequencies.
            min_count: Minimum gNBs to guarantee even if fewer are within radius_m.
                Default 10.

        Returns:
            Number of gNBs activated (length of candidates list).

        Side effects:
            - Clears self.gnb_configs and repopulates it with activated entries.
            - Mutates self.scene.transmitters (adds/removes Transmitter objects).
        """
        if not self._gnb_pool:
            return 0
 
        self.gnb_configs.clear()
        
        all_eligible = []
        for key, config in self._gnb_pool.items():
            if frequency_ghz is not None and abs(config.frequency_ghz - frequency_ghz) > 0.01:
                continue
            dist = self._compute_distance(center, config.position)
            all_eligible.append((dist, key, config))
        all_eligible.sort(key=lambda x: x[0])

        candidates = [c for c in all_eligible if c[0] <= radius_m]

        # Adaptive fallback: if too few within radius, take closest min_count
        if len(candidates) < min_count and len(all_eligible) >= min_count:
            candidates = all_eligible[:min_count]
        elif len(candidates) < min_count:
            candidates = all_eligible  # use all if fewer than min_count exist

        if max_count:
            candidates = candidates[:max_count]

        # 1. candidates에 포함된 기지국의 key(이름)만 모아서 빠른 검색을 위해 Set으로 만듭니다.
        candidate_keys = {c[1] for c in candidates}

        # 2. 현재 scene에 있는 송신기 중 candidate_keys에 없는 것만 골라서 삭제합니다.
        for name in list(self.scene.transmitters.keys()):
            if name not in candidate_keys:
                self.scene.remove(name)
        
        # 3. candidates 중 현재 scene에 없는 '새로운' 기지국만 추가합니다.
        for dist, (gnb_id, sector_id), config in candidates:
            # 이미 scene에 등록되어 있는 기지국(송신기)이라면 추가 과정을 건너뜁니다.
            if (gnb_id, sector_id) in self.scene.transmitters:
                continue

            rf = self._gnb_rf_params.get((gnb_id, sector_id), {})
            self.add_gnb(
                gnb_id=config.gnb_id, position=config.position,
                sector_id=config.sector_id, azimuth_deg=config.azimuth_deg,
                downtilt_deg=config.downtilt_deg,
                tx_power_dbm=config.tx_power_dbm,
                antenna_gain_dbi=config.antenna_gain_dbi,
                antenna_num_ports=config.antenna_num_ports,
                name=config.name,
                frequency_ghz=config.frequency_ghz,
                bandwidth_mhz=rf.get("bandwidth_mhz", default_bandwidth_mhz(config.frequency_ghz)),
                scs_hz=rf.get("scs_hz", default_scs_hz(config.frequency_ghz)),
                rat_type=config.rat_type,
            )
            
        return len(candidates)
    

    def clear(self):
        """Reset all state including the gNB pool, sliding BLER windows, and scene objects.

        Unlike reset_scene(), this also wipes self._gnb_pool, making subsequent
        calls to activate_nearby_gnbs() a no-op until gNBs are re-added via
        add_gnb_to_pool() or load_gnb_from_csv().

        Side effects:
            - Clears gnb_configs, ue_configs, _gnb_pool, _gnb_rf_params.
            - Resets all sliding BLER windows.
            - Removes all Transmitters and Receivers from self.scene.
        """
        self.gnb_configs.clear()
        self.ue_configs.clear()
        self._gnb_pool.clear()
        self._gnb_rf_params.clear()
        self._sliding_bler.reset_all()
        if self.scene:
            for name in list(self.scene.transmitters.keys()):
                self.scene.remove(name)
            for name in list(self.scene.receivers.keys()):
                self.scene.remove(name)
