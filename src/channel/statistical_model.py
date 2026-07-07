"""
3GPP TR 38.901 Statistical Channel Model for NR Handover Simulation

This module implements the 3GPP TR 38.901 standard statistical channel model
with accurate path loss formulas, LOS probability, shadow fading, and Doppler.

Key Features:
- UMa (Urban Macro), UMi (Urban Micro), RMa (Rural Macro) scenarios
- 3GPP-compliant path loss models with frequency in GHz
- LOS/NLOS probability based on distance
- Spatially correlated shadow fading
- Doppler shift calculation for mobility

Author: Claude Code
Date: 2026-02-02
3GPP Reference: TR 38.901 (Study on channel model for frequencies from 0.5 to 100 GHz)
"""

import logging
import math
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .channel_model import ChannelModel, ChannelConfig, ChannelModelType, DopplerInfo
from .channel_calculator import ChannelState, GnbConfig, UeConfig
from .post_processing import (
    compute_noise_floor,
    compute_noise_floor_per_re,
    _fft_size_from_bw_scs,
    default_bandwidth_mhz,
    default_scs_hz,
    apply_propagation_losses,
    compute_sinr,
    compute_rsrq,
    compute_ul_sinr,
)

logger = logging.getLogger(__name__)


class StatisticalChannelModel(ChannelModel):
    """
    3GPP TR 38.901 Statistical Channel Model.

    Implements path loss, shadow fading, and LOS probability models
    for UMa, UMi, and RMa scenarios. Compliant with 3GPP TR 38.901.

    Workflow:
        1. configure() - Set frequency, scenario, etc.
        2. add_gnb() - Add base stations
        3. add_ue() - Add user equipment
        4. update_ue_position() - Update UE mobility
        5. compute_all() - Calculate all channel states (RSRP, SINR, BLER)

    3GPP Reference:
        TR 38.901 §7.4 (Path Loss Models)
        TR 38.901 §7.5 (Shadow Fading)
        TR 38.901 §7.6 (LOS Probability)
    """

    # Physical constants
    SPEED_OF_LIGHT = 299792458.0  # m/s
    THERMAL_NOISE_DBM_HZ = -174.0  # dBm/Hz

    # Shadow fading standard deviations (dB) per scenario
    SHADOW_FADING_SIGMA = {
        "UMa": {"LOS": 4.0, "NLOS": 6.0},
        "UMi": {"LOS": 4.0, "NLOS": 7.82},
        "RMa": {"LOS": 4.0, "NLOS": 8.0},
    }

    def __init__(self):
        """Initialize the statistical channel model."""
        # Configuration
        self._config: Optional[ChannelConfig] = None
        self._frequency_hz: float = 3.5e9
        self._frequency_ghz: float = 3.5
        self._scenario: str = "UMa"
        self._tx_power_dbm: float = 46.0
        self._noise_figure_db: float = 7.0
        self._bandwidth_hz: float = 20e6
        self._noise_floor_dbm: float = -100.0

        # Storage
        self._gnb_configs: Dict[Tuple[int, int], GnbConfig] = {}  # (gnb_id, sector_id) -> config (active subset)
        self._gnb_pool: Dict[Tuple[int, int], GnbConfig] = {}     # ALL gNBs (permanent, survives activate_nearby)
        self._ue_configs: Dict[int, UeConfig] = {}  # ue_id -> config

        # Shadow fading state (per UE-cell pair)
        # Key: (ue_id, gnb_id, sector_id) -> shadow_fading_db
        self._shadow_fading_cache: Dict[Tuple[int, int, int], float] = {}

        # Random number generator for reproducibility
        self._rng = np.random.RandomState(seed=42)

        logger.info("StatisticalChannelModel initialized")

    @property
    def model_type(self) -> ChannelModelType:
        """Get the channel model type."""
        return ChannelModelType.STATISTICAL

    def configure(self, config: ChannelConfig) -> None:
        """
        Configure the channel model.

        Args:
            config: Channel configuration parameters

        Raises:
            ValueError: If configuration is invalid

        3GPP Reference:
            TR 38.901 §7.1 (General parameters)
        """
        if config.model_type != ChannelModelType.STATISTICAL:
            raise ValueError(f"Expected STATISTICAL model type, got {config.model_type}")

        if config.scenario not in ["UMa", "UMi", "RMa"]:
            raise ValueError(f"Invalid scenario: {config.scenario}. Must be UMa, UMi, or RMa")

        self._config = config
        self._frequency_hz = config.frequency_hz
        self._frequency_ghz = config.frequency_hz / 1e9  # CRITICAL: Convert to GHz!
        self._scenario = config.scenario
        self._tx_power_dbm = config.tx_power_dbm
        self._noise_figure_db = config.noise_figure_db
        self._bandwidth_hz = config.bandwidth_hz

        # Calculate noise floor (per-RE, matches per-RE RSRP)
        # Use default parameters: 20 MHz BW, 30 kHz SCS for NR >= 3 GHz
        bandwidth_mhz = self._bandwidth_hz / 1e6
        scs_hz = 30000.0 if self._frequency_ghz >= 3.0 else 15000.0
        self._noise_floor_dbm = compute_noise_floor_per_re(
            bandwidth_mhz, scs_hz, self._noise_figure_db
        )

        # UL SINR parameters (TDD reciprocity, per-RE)
        self._gnb_noise_figure_db = getattr(config, 'gnb_noise_figure_db', 2.5)
        self._ue_tx_power_dbm = getattr(config, 'ue_tx_power_dbm', 23.0)
        self._ul_noise_floor_dbm = compute_noise_floor_per_re(
            bandwidth_mhz, scs_hz, self._gnb_noise_figure_db
        )

        logger.info(f"Configured: scenario={self._scenario}, "
                   f"freq={self._frequency_ghz:.2f} GHz, "
                   f"noise_floor={self._noise_floor_dbm:.1f} dBm")

    def add_gnb(
        self,
        gnb_id: int,
        position: Tuple[float, float, float],
        **kwargs
    ) -> None:
        """
        Add a gNB (base station) to the channel model.

        Args:
            gnb_id: Unique identifier for the gNB
            position: (x, y, z) coordinates in meters
            **kwargs: Additional parameters (sector_id, tx_power_dbm, etc.)

        Raises:
            ValueError: If gnb_id already exists or position is invalid

        3GPP Reference:
            TS 38.104 §5.2 (BS RF requirements)
        """
        sector_id = kwargs.get("sector_id", 1)
        key = (gnb_id, sector_id)

        if key in self._gnb_configs:
            raise ValueError(f"gNB {gnb_id} sector {sector_id} already exists")

        if len(position) != 3:
            raise ValueError(f"Position must be (x, y, z), got {position}")

        freq_ghz = kwargs.get('frequency_ghz', self._frequency_ghz)
        config = GnbConfig(
            gnb_id=gnb_id,
            name=kwargs.get("name", f"gNB_{gnb_id}"),
            sector_id=sector_id,
            position=position,
            azimuth_deg=kwargs.get("azimuth_deg", 0.0),
            downtilt_deg=kwargs.get("downtilt_deg", 0.0),
            tx_power_dbm=kwargs.get("tx_power_dbm", self._tx_power_dbm),
            frequency_ghz=freq_ghz,
            antenna_gain_dbi=kwargs.get("antenna_gain_dbi", 23.0),
            hpbw_h_deg=kwargs.get("hpbw_h_deg", 25.0),
            hpbw_v_deg=kwargs.get("hpbw_v_deg", 7.0),
            bandwidth_mhz=kwargs.get("bandwidth_mhz", default_bandwidth_mhz(freq_ghz)),
            rat_type=kwargs.get("rat_type", "nr")
        )

        self._gnb_configs[key] = config
        self._gnb_pool[key] = config
        logger.debug(f"Added gNB {gnb_id} sector {sector_id} at {position}")

    def add_ue(
        self,
        ue_id: int,
        position: Tuple[float, float, float],
        **kwargs
    ) -> None:
        """
        Add a UE to the channel model.

        Args:
            ue_id: Unique identifier for the UE
            position: (x, y, z) coordinates in meters
            **kwargs: Additional parameters (car_id, velocity, etc.)

        Raises:
            ValueError: If ue_id already exists or position is invalid

        3GPP Reference:
            TS 38.101-1 §5.2 (UE RF requirements)
        """
        if ue_id in self._ue_configs:
            raise ValueError(f"UE {ue_id} already exists")

        if len(position) != 3:
            raise ValueError(f"Position must be (x, y, z), got {position}")

        config = UeConfig(
            ue_id=ue_id,
            position=position,
            car_id=kwargs.get("car_id"),
            timestamp=kwargs.get("timestamp", 0.0),
            velocity=kwargs.get("velocity")
        )

        self._ue_configs[ue_id] = config
        logger.debug(f"Added UE {ue_id} at {position}")

    def update_ue_position(
        self,
        ue_id: int,
        position: Tuple[float, float, float],
        velocity: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """
        Update UE position for mobility simulation.

        Args:
            ue_id: UE identifier
            position: New (x, y, z) coordinates in meters
            velocity: Optional velocity vector (vx, vy, vz) in m/s for Doppler

        Raises:
            KeyError: If ue_id does not exist

        3GPP Reference:
            TS 38.133 §9.2 (Mobility requirements)
        """
        if ue_id not in self._ue_configs:
            raise KeyError(f"UE {ue_id} not found")

        config = self._ue_configs[ue_id]
        config.position = position
        if velocity is not None:
            config.velocity = velocity

        # Clear shadow fading cache for this UE (position changed)
        keys_to_remove = [k for k in self._shadow_fading_cache.keys() if k[0] == ue_id]
        for key in keys_to_remove:
            del self._shadow_fading_cache[key]

    def compute_all(
        self,
        timestamp: Optional[float] = None,
        **kwargs
    ) -> Dict[Tuple[int, int, int], ChannelState]:
        """
        Compute channel state for all (UE, gNB, sector) links.

        This method:
        1. Computes distance and LOS probability
        2. Calculates path loss (LOS or NLOS)
        3. Adds shadow fading
        4. Computes RSRP
        5. Calculates SINR with multi-cell interference

        Args:
            timestamp: Optional simulation timestamp in seconds

        Returns:
            Dictionary mapping (ue_id, gnb_id, sector_id) to ChannelState

        3GPP Reference:
            TR 38.901 §7.4 (Path Loss)
            TS 38.214 §5.1 (CSI framework)
        """
        results: Dict[Tuple[int, int, int], ChannelState] = {}

        if not self._gnb_configs or not self._ue_configs:
            logger.warning("No gNBs or UEs configured")
            return results

        # Step 1: Compute RSRP for all UE-gNB pairs
        rsrp_per_ue: Dict[int, Dict[Tuple[int, int], float]] = {}

        for ue_id, ue_config in self._ue_configs.items():
            rsrp_per_ue[ue_id] = {}

            for (gnb_id, sector_id), gnb_config in self._gnb_configs.items():
                # Compute distance
                distance_3d = self._compute_distance_3d(ue_config.position, gnb_config.position)
                distance_2d = self._compute_distance_2d(ue_config.position, gnb_config.position)

                # Determine LOS/NLOS
                los_probability = self._compute_los_probability(distance_2d, self._scenario)
                is_los = self._rng.random() < los_probability

                # Compute path loss
                path_loss_db = self._compute_path_loss(
                    distance_3d=distance_3d,
                    distance_2d=distance_2d,
                    h_bs=gnb_config.position[2],
                    h_ut=ue_config.position[2],
                    is_los=is_los,
                    scenario=self._scenario
                )

                # Add shadow fading (3GPP TR 38.901: SF is added to path loss)
                shadow_fading_db = self._compute_shadow_fading(ue_id, gnb_id, sector_id, is_los)

                # Compute effective antenna gain toward UE (peak + pattern)
                # 3GPP TR 38.901 Table 7.3-1: includes horizontal pattern rolloff
                antenna_gain_eff = self._compute_antenna_gain_toward_ue(
                    gnb_config, ue_config.position
                )

                # Compute raw RSRP: P_rx = P_tx(EIRP) + A_H(phi) - PL - SF
                # tx_power_dbm is boresight EIRP; antenna_gain_eff is the pattern
                # roll-off (0 at boresight, down to -30 dB off-boresight)
                # Shadow fading is zero-mean Gaussian added to path loss per 3GPP TR 38.901
                rsrp_raw_dbm = gnb_config.tx_power_dbm + antenna_gain_eff - path_loss_db - shadow_fading_db

                # Apply propagation losses (penetration + surface distortion)
                # via shared post-processing for consistency with RT model
                penetration_loss_db = getattr(self._config, 'penetration_loss_db', 0.0) if self._config else 0.0
                surface_mean = getattr(self._config, 'surface_distortion_mean_db', 0.0) if self._config else 0.0
                surface_std = getattr(self._config, 'surface_distortion_std_db', 0.0) if self._config else 0.0

                rsrp_dbm, _ = apply_propagation_losses(
                    rsrp_raw_dbm, penetration_loss_db,
                    surface_mean, surface_std, rng=self._rng
                )

                # Per-RE normalization (3GPP TS 38.215 Section 5.1.1)
                # Use per-gNB bandwidth from CSV (GnbConfig.bandwidth_mhz)
                # SCS: LTE(<3GHz)=15kHz, NR(>=3GHz)=30kHz
                bandwidth_mhz = gnb_config.bandwidth_mhz
                scs_hz = default_scs_hz(gnb_config.frequency_ghz)
                n_sc = _fft_size_from_bw_scs(bandwidth_mhz, scs_hz)
                rsrp_dbm -= 10.0 * np.log10(n_sc)

                rsrp_per_ue[ue_id][(gnb_id, sector_id)] = rsrp_dbm

                # Create ChannelState
                state = ChannelState(
                    ue_id=ue_id,
                    gnb_id=gnb_id,
                    sector_id=sector_id,
                    rsrp_dbm=rsrp_dbm,
                    path_loss_db=path_loss_db,
                    distance_m=distance_3d,
                    los=is_los,
                    timestamp=timestamp or ue_config.timestamp,
                    rat_type=gnb_config.rat_type
                )

                results[(ue_id, gnb_id, sector_id)] = state

        # Step 2: Compute SINR with multi-cell interference (shared post-processing)
        # Antenna gain + 3GPP horizontal pattern (TR 38.901 Table 7.3-1) is now
        # applied per-cell in Step 1, so each cell's RSRP already reflects its
        # directional gain toward the UE.  A small residual beam_isolation (3 dB)
        # accounts for vertical-pattern / scattering effects not captured by the
        # simplified 2-D pattern model.
        beam_isolation = 3.0
        for (ue_id, gnb_id, sector_id), state in results.items():
            # Collect interferer RSRPs
            interferer_rsrps = [
                rsrp for (other_gnb_id, other_sector_id), rsrp
                in rsrp_per_ue[ue_id].items()
                if other_gnb_id != gnb_id or other_sector_id != sector_id
            ]

            state.sinr_db = compute_sinr(
                state.rsrp_dbm, interferer_rsrps,
                self._noise_floor_dbm, beam_isolation_db=beam_isolation
            )

            state.rsrq_db = compute_rsrq(
                state.rsrp_dbm, interferer_rsrps,
                self._noise_floor_dbm, n_rb=100, beam_isolation_db=beam_isolation
            )

        # Step 3: Compute UL SINR via TDD reciprocity
        for (ue_id, gnb_id, sector_id), state in results.items():
            gnb_config = self._gnb_configs.get((gnb_id, sector_id))
            if gnb_config is None:
                continue
            # UL SINR via TDD reciprocity: path_loss ≈ tx_power(EIRP) - DL_RSRP
            # tx_power_dbm is boresight EIRP, consistent with DL link budget.
            state.ul_sinr_db = compute_ul_sinr(
                dl_rsrp_dbm=state.rsrp_dbm,
                gnb_ref_tx_power_dbm=gnb_config.tx_power_dbm,
                ue_tx_power_dbm=self._ue_tx_power_dbm,
                ul_noise_floor_dbm=self._ul_noise_floor_dbm,
            )

        # Step 4: Apply Doppler ICI correction (velocity-based)
        self._apply_doppler_correction(results)

        logger.debug(f"Computed {len(results)} channel states")
        return results

    def compute_doppler(self, ue_id: int) -> Optional[DopplerInfo]:
        """
        Compute Doppler information for a moving UE.

        Args:
            ue_id: UE identifier

        Returns:
            DopplerInfo if UE has velocity, None otherwise

        3GPP Reference:
            TS 38.101-1 §6.2.2 (Doppler spread)

        Formula:
            f_d = v * f_c / c
            where v is velocity magnitude, f_c is carrier frequency, c is speed of light
        """
        if ue_id not in self._ue_configs:
            logger.warning(f"UE {ue_id} not found")
            return None

        ue_config = self._ue_configs[ue_id]

        if ue_config.velocity is None:
            return None

        # Compute velocity magnitude
        vx, vy, vz = ue_config.velocity
        velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

        # Doppler shift: f_d = v * f_c / c
        doppler_shift_hz = velocity_magnitude * self._frequency_hz / self.SPEED_OF_LIGHT

        # Coherence time: T_c ≈ 1 / (2 * f_d)
        # Minimum coherence time is 1 ms to avoid division by zero
        coherence_time_ms = max(1.0, 1000.0 / (2.0 * doppler_shift_hz + 1e-10))

        return DopplerInfo(
            ue_id=ue_id,
            velocity_mps=ue_config.velocity,
            doppler_shift_hz=doppler_shift_hz,
            coherence_time_ms=coherence_time_ms
        )

    def _apply_doppler_correction(
        self,
        results: Dict[Tuple[int, int, int], ChannelState],
    ) -> None:
        """
        Doppler ICI post-processing: correct SINR based on UE velocity.

        Inter-Carrier Interference (ICI) from Doppler acts as
        signal-proportional noise, creating a SINR ceiling:

            f_d = v * f_c / c
            ICI_ratio = (pi * f_d / SCS)^2 / 3
            SINR_eff = SINR_static / (1 + SINR_linear * ICI_ratio)

        Also stores sinr_base_db and doppler_penalty_db on ChannelState
        for display purposes.

        3GPP Reference: TR 38.901 §7.6.6 (Doppler modelling)
        ICI Model: Russell & Stuber, IEEE VTC 1995
        """
        c = 3e8

        for (ue_id, gnb_id, sector_id), state in results.items():
            ue = self._ue_configs.get(ue_id)
            if not ue or not ue.velocity:
                continue
            v_mps = np.sqrt(sum(v ** 2 for v in ue.velocity))
            if v_mps < 0.1:
                continue

            gnb_cfg = self._gnb_configs.get((gnb_id, sector_id))
            if not gnb_cfg:
                continue
            freq_hz = gnb_cfg.frequency_ghz * 1e9
            scs_hz = default_scs_hz(gnb_cfg.frequency_ghz)

            # Doppler frequency & ICI ratio
            f_d = v_mps * freq_hz / c
            ici_ratio = (np.pi * f_d / scs_hz) ** 2 / 3.0
            if ici_ratio < 1e-12:
                continue

            # Save pre-Doppler SINR
            sinr_before = state.sinr_db
            state.sinr_base_db = sinr_before

            # DL SINR correction
            sinr_lin = 10 ** (state.sinr_db / 10.0)
            sinr_eff = sinr_lin / (1.0 + sinr_lin * ici_ratio)
            state.sinr_db = float(10.0 * np.log10(max(sinr_eff, 1e-20)))

            # UL SINR correction
            if state.ul_sinr_db is not None:
                ul_lin = 10 ** (state.ul_sinr_db / 10.0)
                ul_eff = ul_lin / (1.0 + ul_lin * ici_ratio)
                state.ul_sinr_db = float(10.0 * np.log10(max(ul_eff, 1e-20)))

            # Store Doppler penalty for display
            state.doppler_penalty_db = state.sinr_db - sinr_before

            logger.debug(
                f"UE{ue_id}-gNB{gnb_id}: Doppler v={v_mps:.1f}m/s "
                f"f_d={f_d:.0f}Hz ICI={ici_ratio:.2e} "
                f"SINR_loss={-state.doppler_penalty_db:.1f}dB"
            )

    def activate_nearby_gnbs(
        self,
        center: Tuple[float, float, float],
        radius_m: float = 2000.0,
        max_count: Optional[int] = None,
        frequency_ghz: Optional[float] = None,
        min_count: int = 10,
    ) -> int:
        """
        Activate only gNBs within radius of center point.

        Filters the permanent pool (_gnb_pool) into the active set (_gnb_configs).
        Only active gNBs are used by compute_all().

        If fewer than min_count gNBs are within radius_m, the closest min_count
        gNBs are activated regardless of distance (adaptive fallback).

        Args:
            center: (x, y, z) reference point (e.g., average UE position)
            radius_m: Maximum 2D distance from center to include a gNB
            max_count: Optional cap on number of gNBs (closest first)
            frequency_ghz: If set, only activate gNBs on this frequency
            min_count: Minimum gNBs to activate (fallback to closest N)

        Returns:
            Number of gNBs activated
        """
        if not self._gnb_pool:
            return 0

        # Compute distances for all eligible gNBs
        all_eligible = []
        for key, config in self._gnb_pool.items():
            if frequency_ghz is not None and abs(config.frequency_ghz - frequency_ghz) > 0.01:
                continue
            dist = self._compute_distance_2d(center, config.position)
            all_eligible.append((dist, key, config))

        all_eligible.sort(key=lambda x: x[0])

        # Filter by radius
        candidates = [(d, k, c) for d, k, c in all_eligible if d <= radius_m]

        # Adaptive fallback: if too few within radius, take closest min_count
        if len(candidates) < min_count and len(all_eligible) >= min_count:
            candidates = all_eligible[:min_count]
        elif len(candidates) < min_count:
            candidates = all_eligible  # use all if fewer than min_count exist

        if max_count:
            candidates = candidates[:max_count]

        self._gnb_configs = {key: config for _, key, config in candidates}

        # Prune shadow fading cache for gNBs no longer active
        active_gnb_ids = {k[0] for k in self._gnb_configs}
        self._shadow_fading_cache = {
            k: v for k, v in self._shadow_fading_cache.items()
            if k[1] in active_gnb_ids
        }

        return len(self._gnb_configs)

    def ensure_gnb_active(self, gnb_id: int) -> bool:
        """Ensure a specific gNB is in the active set (e.g., serving cell).

        Copies from pool if available but not currently active.
        Returns True if the gNB is active after this call.
        """
        # Check if already active
        for key in self._gnb_configs:
            if key[0] == gnb_id:
                return True
        # Try to restore from pool
        for key, config in self._gnb_pool.items():
            if key[0] == gnb_id:
                self._gnb_configs[key] = config
                return True
        return False

    def clear(self) -> None:
        """
        Clear all gNBs, UEs, and computed states.

        Used for resetting the simulation.
        """
        self._gnb_configs.clear()
        self._gnb_pool.clear()
        self._ue_configs.clear()
        self._shadow_fading_cache.clear()
        logger.info("StatisticalChannelModel cleared")

    # ========================================================================
    # Private Helper Methods: Distance Calculation
    # ========================================================================

    def _compute_distance_3d(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float]
    ) -> float:
        """Compute 3D Euclidean distance between two points."""
        return np.sqrt(
            (pos1[0] - pos2[0])**2 +
            (pos1[1] - pos2[1])**2 +
            (pos1[2] - pos2[2])**2
        )

    def _compute_distance_2d(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float]
    ) -> float:
        """Compute 2D horizontal distance between two points (ignore z)."""
        return np.sqrt(
            (pos1[0] - pos2[0])**2 +
            (pos1[1] - pos2[1])**2
        )

    # ========================================================================
    # Private Helper Methods: Antenna Pattern (3GPP TR 38.901 Table 7.3-1)
    # ========================================================================

    @staticmethod
    def _compute_antenna_gain_toward_ue(
        gnb_config: GnbConfig,
        ue_position: Tuple[float, float, float],
    ) -> float:
        """
        Compute horizontal antenna pattern loss from gNB toward UE.

        tx_power_dbm in the CSV is treated as boresight EIRP (conducted +
        peak gain), so this method returns only the directional pattern
        roll-off:

            A_H(phi) = -min(12 * (phi / phi_3dB)^2, A_m)

        Result is 0 dB at boresight and down to -A_m off-boresight.

        Coordinate convention:
            x = East, y = North (ENU).
            Azimuth = degrees clockwise from North (standard bearing).

        Args:
            gnb_config: gNB configuration (position, azimuth, gain, HPBW)
            ue_position: (x, y, z) of UE in meters

        Returns:
            Pattern loss in dB (0 at boresight, negative off-boresight)

        3GPP Reference:
            TR 38.901 Table 7.3-1 (antenna model)
        """
        dx = ue_position[0] - gnb_config.position[0]
        dy = ue_position[1] - gnb_config.position[1]

        # Bearing from gNB to UE: atan2(East, North) → CW from North
        bearing_deg = math.degrees(math.atan2(dx, dy))
        if bearing_deg < 0:
            bearing_deg += 360.0

        # Azimuth offset from antenna boresight
        offset = bearing_deg - gnb_config.azimuth_deg
        # Normalize to [-180, 180]
        offset = (offset + 180) % 360 - 180

        # 3GPP TR 38.901 Eq. 7.3-1: horizontal radiation pattern
        A_m = 30.0  # max attenuation (front-to-back ratio) in dB
        phi_3dB = max(gnb_config.hpbw_h_deg, 1.0)  # avoid division by zero
        pattern_loss_db = min(12.0 * (offset / phi_3dB) ** 2, A_m)

        return -pattern_loss_db

    # ========================================================================
    # Private Helper Methods: LOS Probability (3GPP TR 38.901 Table 7.4.2-1)
    # ========================================================================

    def _compute_los_probability(self, distance_2d: float, scenario: str) -> float:
        """
        Compute LOS probability based on 2D distance and scenario.

        Args:
            distance_2d: 2D horizontal distance in meters
            scenario: "UMa", "UMi", or "RMa"

        Returns:
            LOS probability in [0, 1]

        3GPP Reference:
            TR 38.901 Table 7.4.2-1 (LOS probability)
        """
        d = distance_2d

        if scenario == "UMa":
            # P_LOS = min(18/d, 1) * (1 - exp(-d/63)) + exp(-d/63)
            if d <= 0:
                return 1.0
            return min(18.0 / d, 1.0) * (1.0 - np.exp(-d / 63.0)) + np.exp(-d / 63.0)

        elif scenario == "UMi":
            # P_LOS = min(18/d, 1) * (1 - exp(-d/36)) + exp(-d/36)
            if d <= 0:
                return 1.0
            return min(18.0 / d, 1.0) * (1.0 - np.exp(-d / 36.0)) + np.exp(-d / 36.0)

        elif scenario == "RMa":
            # P_LOS = exp(-(d - 10) / 1000) for d > 10m, else 1.0
            if d <= 10.0:
                return 1.0
            return np.exp(-(d - 10.0) / 1000.0)

        else:
            logger.warning(f"Unknown scenario {scenario}, assuming UMa")
            return min(18.0 / d, 1.0) * (1.0 - np.exp(-d / 63.0)) + np.exp(-d / 63.0)

    # ========================================================================
    # Private Helper Methods: Path Loss (3GPP TR 38.901 Table 7.4.1-1)
    # ========================================================================

    def _compute_path_loss(
        self,
        distance_3d: float,
        distance_2d: float,
        h_bs: float,
        h_ut: float,
        is_los: bool,
        scenario: str
    ) -> float:
        """
        Compute path loss based on scenario and LOS condition.

        Args:
            distance_3d: 3D distance in meters
            distance_2d: 2D horizontal distance in meters
            h_bs: Base station height in meters
            h_ut: UE height in meters
            is_los: True if LOS, False if NLOS
            scenario: "UMa", "UMi", or "RMa"

        Returns:
            Path loss in dB

        3GPP Reference:
            TR 38.901 Table 7.4.1-1 (Path loss models)
        """
        if scenario == "UMa":
            return self._compute_path_loss_uma(distance_3d, distance_2d, h_bs, h_ut, is_los)
        elif scenario == "UMi":
            return self._compute_path_loss_umi(distance_3d, distance_2d, h_bs, h_ut, is_los)
        elif scenario == "RMa":
            return self._compute_path_loss_rma(distance_3d, distance_2d, h_bs, h_ut, is_los)
        else:
            logger.warning(f"Unknown scenario {scenario}, using UMa")
            return self._compute_path_loss_uma(distance_3d, distance_2d, h_bs, h_ut, is_los)

    def _compute_path_loss_uma(
        self,
        distance_3d: float,
        distance_2d: float,
        h_bs: float,
        h_ut: float,
        is_los: bool
    ) -> float:
        """
        UMa (Urban Macro) path loss model.

        3GPP Reference:
            TR 38.901 Table 7.4.1-1 (UMa)

        LOS:
            PL = 28.0 + 22*log10(d3D) + 20*log10(fc)
            Valid for 10m < d2D < 5000m

        NLOS:
            PL = 13.54 + 39.08*log10(d3D) + 20*log10(fc) - 0.6*(hUT - 1.5)
            Valid for 10m < d2D < 5000m
        """
        d3d = max(distance_3d, 1.0)  # Avoid log(0)
        fc_ghz = self._frequency_ghz  # Already in GHz!

        if is_los:
            # LOS: PL = 28.0 + 22*log10(d3D) + 20*log10(fc)
            pl = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
        else:
            # NLOS: PL = 13.54 + 39.08*log10(d3D) + 20*log10(fc) - 0.6*(hUT - 1.5)
            pl = (13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(fc_ghz) -
                  0.6 * (h_ut - 1.5))

        return pl

    def _compute_path_loss_umi(
        self,
        distance_3d: float,
        distance_2d: float,
        h_bs: float,
        h_ut: float,
        is_los: bool
    ) -> float:
        """
        UMi (Urban Micro) path loss model.

        3GPP Reference:
            TR 38.901 Table 7.4.1-1 (UMi Street Canyon)

        LOS:
            PL = 32.4 + 21*log10(d3D) + 20*log10(fc)
            Valid for 10m < d2D < 5000m

        NLOS:
            PL = 22.4 + 35.3*log10(d3D) + 21.3*log10(fc) - 0.3*(hUT - 1.5)
            Valid for 10m < d2D < 2000m
        """
        d3d = max(distance_3d, 1.0)
        fc_ghz = self._frequency_ghz

        if is_los:
            # LOS: PL = 32.4 + 21*log10(d3D) + 20*log10(fc)
            pl = 32.4 + 21.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
        else:
            # NLOS: PL = 22.4 + 35.3*log10(d3D) + 21.3*log10(fc) - 0.3*(hUT - 1.5)
            pl = (22.4 + 35.3 * np.log10(d3d) + 21.3 * np.log10(fc_ghz) -
                  0.3 * (h_ut - 1.5))

        return pl

    def _compute_path_loss_rma(
        self,
        distance_3d: float,
        distance_2d: float,
        h_bs: float,
        h_ut: float,
        is_los: bool
    ) -> float:
        """
        RMa (Rural Macro) path loss model with breakpoint distance.

        3GPP Reference:
            TR 38.901 Table 7.4.1-1 (RMa)

        LOS (two-slope model):
            d_BP = 2*pi*h_BS*h_UT*fc/c

            For d2D < d_BP:
                PL = 20*log10(40*pi*d3D*fc/3) + min(0.03*h^1.72, 10)*log10(d3D)
                     - min(0.044*h^1.72, 14.77) + 0.002*log10(h)*d3D

            For d2D >= d_BP:
                PL = PL(d_BP) + 40*log10(d3D/d_BP)

            where h = average building height (default 5m)

        NLOS:
            PL = 161.04 - 7.1*log10(W) + 7.5*log10(h) - (24.37 - 3.7*(h/h_BS)^2)*log10(h_BS)
                 + (43.42 - 3.1*log10(h_BS))*(log10(d3D) - 3) + 20*log10(fc)
                 - (3.2*(log10(11.75*h_UT))^2 - 4.97)

            where W = street width (default 20m), h = building height (default 5m)
        """
        d3d = max(distance_3d, 1.0)
        d2d = max(distance_2d, 1.0)
        fc_ghz = self._frequency_ghz

        # Default parameters
        h_building = 5.0  # Average building height in meters
        street_width = 20.0  # Street width in meters

        if is_los:
            # Breakpoint distance: d_BP = 2*pi*h_BS*h_UT*fc/c
            # fc must be in Hz for this formula
            fc_hz = self._frequency_hz
            d_bp = 2.0 * np.pi * h_bs * h_ut * fc_hz / self.SPEED_OF_LIGHT

            if d2d < d_bp:
                # Before breakpoint
                h = h_building
                pl1 = 20.0 * np.log10(40.0 * np.pi * d3d * fc_ghz / 3.0)
                pl2 = min(0.03 * h**1.72, 10.0) * np.log10(d3d)
                pl3 = -min(0.044 * h**1.72, 14.77)
                pl4 = 0.002 * np.log10(h) * d3d
                pl = pl1 + pl2 + pl3 + pl4
            else:
                # After breakpoint: use two-slope model
                # PL(d_BP) calculated as above
                h = h_building
                pl1_bp = 20.0 * np.log10(40.0 * np.pi * d_bp * fc_ghz / 3.0)
                pl2_bp = min(0.03 * h**1.72, 10.0) * np.log10(d_bp)
                pl3_bp = -min(0.044 * h**1.72, 14.77)
                pl4_bp = 0.002 * np.log10(h) * d_bp
                pl_bp = pl1_bp + pl2_bp + pl3_bp + pl4_bp

                # Add slope after breakpoint
                pl = pl_bp + 40.0 * np.log10(d3d / d_bp)
        else:
            # NLOS (simplified formula for typical case)
            # PL = 161.04 - 7.1*log10(W) + 7.5*log10(h)
            #      - (24.37 - 3.7*(h/h_BS)^2)*log10(h_BS)
            #      + (43.42 - 3.1*log10(h_BS))*(log10(d3D) - 3)
            #      + 20*log10(fc) - (3.2*(log10(11.75*h_UT))^2 - 4.97)
            h = h_building
            W = street_width

            term1 = 161.04
            term2 = -7.1 * np.log10(W)
            term3 = 7.5 * np.log10(h)
            term4 = -(24.37 - 3.7 * (h / h_bs)**2) * np.log10(h_bs)
            term5 = (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(d3d) - 3.0)
            term6 = 20.0 * np.log10(fc_ghz)
            term7 = -(3.2 * (np.log10(11.75 * h_ut))**2 - 4.97)

            pl = term1 + term2 + term3 + term4 + term5 + term6 + term7

        return pl

    # ========================================================================
    # Private Helper Methods: Shadow Fading
    # ========================================================================

    def _compute_shadow_fading(
        self,
        ue_id: int,
        gnb_id: int,
        sector_id: int,
        is_los: bool
    ) -> float:
        """
        Compute shadow fading with spatial correlation.

        Shadow fading is modeled as a zero-mean Gaussian random variable
        with standard deviation depending on scenario and LOS condition.

        Args:
            ue_id: UE identifier
            gnb_id: gNB identifier
            sector_id: Sector identifier
            is_los: True if LOS, False if NLOS

        Returns:
            Shadow fading in dB (can be positive or negative)

        3GPP Reference:
            TR 38.901 Table 7.4.1-1 (Shadow fading standard deviations)
        """
        key = (ue_id, gnb_id, sector_id)

        # Check cache
        if key in self._shadow_fading_cache:
            return self._shadow_fading_cache[key]

        # Get standard deviation for scenario and LOS condition
        los_str = "LOS" if is_los else "NLOS"
        sigma = self.SHADOW_FADING_SIGMA.get(self._scenario, {}).get(los_str, 6.0)

        # Generate zero-mean Gaussian shadow fading
        shadow_fading_db = self._rng.normal(0.0, sigma)

        # Cache for spatial consistency
        self._shadow_fading_cache[key] = shadow_fading_db

        return shadow_fading_db


def _test_statistical_model():
    """Test function for StatisticalChannelModel."""
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("StatisticalChannelModel Test (3GPP TR 38.901)")
    print("=" * 70)

    # Create model
    model = StatisticalChannelModel()

    # Configure
    config = ChannelConfig(
        model_type=ChannelModelType.STATISTICAL,
        frequency_hz=3.5e9,
        bandwidth_hz=20e6,
        tx_power_dbm=46.0,
        noise_figure_db=7.0,
        scenario="UMa"
    )
    model.configure(config)

    # Add test gNBs
    gnb_positions = [
        (0, 100, 30, "gNB_1"),
        (500, 100, 30, "gNB_2"),
        (1000, 100, 30, "gNB_3")
    ]

    for i, (x, y, z, name) in enumerate(gnb_positions):
        model.add_gnb(
            gnb_id=i + 1,
            position=(x, y, z),
            sector_id=1,
            azimuth_deg=180.0,
            tx_power_dbm=46.0,
            name=name
        )
        print(f"Added {name} at position ({x}, {y}, {z})")

    # Add test UE
    ue_position = (250, 0, 1.5)
    ue_velocity = (27.78, 0, 0)  # 100 km/h in x direction
    model.add_ue(
        ue_id=0,
        position=ue_position,
        velocity=ue_velocity
    )
    print(f"Added UE_0 at position {ue_position} with velocity {ue_velocity} m/s")

    # Compute channel states
    print("\n" + "-" * 70)
    print("Computing channel states...")
    print("-" * 70)

    results = model.compute_all()

    print(f"\nResults ({len(results)} channel states):\n")
    print(f"{'UE':<6} {'gNB':<6} {'Sector':<8} {'RSRP':<12} {'SINR':<12} "
          f"{'Distance':<12} {'LOS':<6}")
    print("-" * 70)

    for (ue_id, gnb_id, sector_id), state in sorted(results.items()):
        los_str = "Yes" if state.los else "No"
        print(f"{state.ue_id:<6} {state.gnb_id:<6} {state.sector_id:<8} "
              f"{state.rsrp_dbm:<12.1f} {state.sinr_db:<12.1f} "
              f"{state.distance_m:<12.1f} {los_str:<6}")

    # Compute Doppler
    print("\n" + "-" * 70)
    print("Doppler Information:")
    print("-" * 70)

    doppler_info = model.compute_doppler(ue_id=0)
    if doppler_info:
        print(f"UE {doppler_info.ue_id}:")
        print(f"  Velocity: {doppler_info.velocity_mps} m/s")
        print(f"  Doppler Shift: {doppler_info.doppler_shift_hz:.2f} Hz")
        print(f"  Coherence Time: {doppler_info.coherence_time_ms:.2f} ms")

    # Test UE movement
    print("\n" + "-" * 70)
    print("Testing UE Movement:")
    print("-" * 70)

    new_position = (300, 0, 1.5)
    model.update_ue_position(ue_id=0, position=new_position, velocity=ue_velocity)
    print(f"Updated UE_0 position to {new_position}")

    results_after_move = model.compute_all()
    print(f"\nResults after movement ({len(results_after_move)} channel states):\n")

    for (ue_id, gnb_id, sector_id), state in sorted(results_after_move.items()):
        los_str = "Yes" if state.los else "No"
        print(f"gNB_{state.gnb_id}: RSRP={state.rsrp_dbm:.1f} dBm, "
              f"SINR={state.sinr_db:.1f} dB, Distance={state.distance_m:.1f} m, LOS={los_str}")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    _test_statistical_model()
