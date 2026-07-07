"""
Sionna RT-based Channel Calculator for NR Handover Simulation

This module provides ray-tracing based channel computation using Sionna RT.
It supports:
- Multi-cell (gNB) deployment from CSV configuration
- Multi-UE tracking with trajectory support
- RSRP and SINR calculation with interference modeling
- Custom 3D scene loading (XML format)

Based on 3GPP TS 38.133 for measurement definitions.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .post_processing import compute_noise_floor, compute_ul_sinr

# Sionna imports with GPU configuration
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # CSV-channel mode monkeypatches the channel model and never instantiates
    # SionnaChannelCalculator, so importing TensorFlow/Sionna here just burns
    # ~4s of process startup. run_csv_simulation.py sets SIM_SKIP_SIONNA=1 to
    # take the mock path (SIONNA_AVAILABLE=False). Statistical/Sionna modes
    # leave the env unset and import normally.
    if os.environ.get("SIM_SKIP_SIONNA") == "1":
        raise ImportError("SIM_SKIP_SIONNA=1: skipping Sionna/TF (CSV mode)")

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"TensorFlow GPU(s) configured: {len(gpus)} device(s)")

    import sionna
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
    SIONNA_AVAILABLE = True
    sionna_version = getattr(sionna, '__version__', 'unknown')
    logger.info(f"Sionna {sionna_version} loaded successfully")
except ImportError as e:
    SIONNA_AVAILABLE = False
    logger.warning(f"Sionna not available: {e}. Using mock implementation.")


@dataclass
class GnbConfig:
    """gNB (base station) configuration"""
    gnb_id: int
    name: str
    sector_id: int
    position: Tuple[float, float, float]  # (x, y, z) in meters
    azimuth_deg: float = 0.0  # Antenna azimuth (horizontal direction)
    downtilt_deg: float = 0.0  # Total antenna downtilt
    tx_power_dbm: float = 46.0  # Transmit power
    frequency_ghz: float = 3.5
    antenna_gain_dbi: float = 23.0
    antenna_num_ports: int = 2  # Real number of TX antenna ports (e.g., 2T2R=2, 4T4R=4, 32T2R=32)
    rx_noise_figure_db: float = 2.5  # gNB RX noise figure (TS 38.104 Table 7.4.1-1, FR1 Wide Area BS)
    hpbw_h_deg: float = 25.0  # Horizontal half-power beamwidth
    hpbw_v_deg: float = 7.0   # Vertical half-power beamwidth
    bandwidth_mhz: float = 20.0  # System bandwidth in MHz (read from CSV or default)
    rat_type: str = "nr"       # "nr" or "lte"

    @classmethod
    def from_csv_row(cls, row: pd.Series) -> 'GnbConfig':
        """Create GnbConfig from CSV row"""
        # Detect RAT type from antenna_type field
        antenna_type = str(row.get('antenna_type', ''))
        is_lte = 'LTE' in antenna_type.upper()

        return cls(
            gnb_id=int(row.get('gnb_id', 0)),
            name=str(row.get('name', f"gNB_{row.get('gnb_id', 0)}")),
            sector_id=int(row.get('sector_id', 1)),
            position=(
                float(row.get('x_m', row.get('x', 0))),
                float(row.get('y_m', row.get('y', 0))),
                float(row.get('z_m', row.get('height_m', 30)))
            ),
            azimuth_deg=float(row.get('azimuth_deg', 0)),
            downtilt_deg=float(row['total_downtilt_deg']) if 'total_downtilt_deg' in row and pd.notna(row['total_downtilt_deg']) else float(row['downtilt_deg']) if 'downtilt_deg' in row and pd.notna(row['downtilt_deg']) else -1.0,
            tx_power_dbm=float(row.get('tx_power_dBm', row.get('tx_power_dbm', 46))),
            frequency_ghz=float(row.get('frequency_GHz', row.get('frequency_ghz', 3.5))),
            antenna_gain_dbi=float(row.get('antenna_gain_dBi', row.get('antenna_gain_dbi', 23))),
            hpbw_h_deg=float(row.get('hpbw_horizontal_deg', 25)),
            hpbw_v_deg=float(row.get('hpbw_vertical_deg', 7)),
            bandwidth_mhz=float(row.get('bandwidth_MHz', row.get('bandwidth_mhz', 20.0))),
            rx_noise_figure_db=float(row.get('rx_noise_figure_db', 2.5)),
            rat_type="lte" if is_lte else "nr",
        )


@dataclass
class UeConfig:
    """UE (user equipment) configuration"""
    ue_id: int
    position: Tuple[float, float, float]  # (x, y, z) in meters
    car_id: Optional[int] = None  # For train/car grouping
    tx_power_dbm: float = 23.0  # UE max TX power (TS 38.101-1 Table 6.2.1-1, Power Class 3)
    timestamp: float = 0.0  # Simulation time
    velocity: Optional[Tuple[float, float, float]] = None  # (vx, vy, vz) m/s

    @classmethod
    def from_csv_row(cls, row: pd.Series) -> 'UeConfig':
        """Create UeConfig from CSV row"""
        return cls(
            ue_id=int(row.get('ue_id', 0)),
            position=(
                float(row.get('x', 0)),
                float(row.get('y', 0)),
                float(row.get('z', 1.5))
            ),
            car_id=int(row['car_id']) if 'car_id' in row and pd.notna(row['car_id']) else None,
            timestamp=float(row.get('timestamp', 0))
        )


@dataclass
class ChannelState:
    """Channel state for a UE-gNB pair"""
    ue_id: int
    gnb_id: int
    sector_id: int
    rsrp_dbm: float  # Reference Signal Received Power
    rsrq_db: float = 0.0  # Reference Signal Received Quality (optional)
    sinr_db: float = -30.0  # Signal to Interference plus Noise Ratio (DL)
    ul_sinr_db: float = -30.0  # UL SINR (gNB receiver perspective, TDD reciprocity)
    path_loss_db: float = 0.0  # Total path loss
    delay_spread_ns: float = 0.0  # RMS delay spread
    num_paths: int = 0  # Number of multipath components
    timestamp: float = 0.0

    # Additional info
    distance_m: float = 0.0  # 3D distance
    los: bool = False  # Line-of-sight indicator

    # RAT type of the gNB
    rat_type: str = "nr"  # "nr" or "lte"

    # SINR breakdown (for debug display)
    sinr_base_db: float = -30.0  # SINR from RSRP-ratio before Doppler
    doppler_penalty_db: float = 0.0  # Doppler penalty from SINR map (0.0 = N/A)

    # BLER (set by channel model Step 4)
    bler: Optional[float] = None
    bler_instant: Optional[float] = None


class SionnaChannelCalculator:
    """
    Sionna RT-based channel calculator for NR handover simulation.

    Features:
    - Ray-tracing based propagation modeling
    - Multi-cell deployment with sector support
    - Multi-UE tracking with trajectory support
    - RSRP/SINR calculation with inter-cell interference

    Usage:
        calculator = SionnaChannelCalculator(
            scene_path="railway_5g_handover_scene.xml",
            gnb_csv_path="gnb_coordinates_railway_final.csv"
        )
        calculator.add_ue(ue_id=0, position=(0, 0, 1.5))
        results = calculator.compute_all()
    """

    # Default parameters
    DEFAULT_FREQUENCY_HZ = 3.5e9
    DEFAULT_TX_POWER_DBM = 46.0
    DEFAULT_NOISE_FIGURE_DB = 7.0
    DEFAULT_THERMAL_NOISE_DBM_HZ = -174.0
    DEFAULT_BANDWIDTH_HZ = 20e6  # 20 MHz (typical NR n78 deployment)

    def __init__(
        self,
        scene_path: Optional[str] = None,
        gnb_csv_path: Optional[str] = None,
        frequency_hz: float = DEFAULT_FREQUENCY_HZ,
        tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
        bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
        noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB,
        max_depth: int = 5,
        num_samples: float = 1e6,
        diffraction: bool = True,
        scattering: bool = True,
        use_gpu: bool = True,
        penetration_loss_db: float = 0.0
    ):
        """
        Initialize the channel calculator.

        Args:
            scene_path: Path to Sionna RT scene XML file
            gnb_csv_path: Path to gNB coordinates CSV file
            frequency_hz: Carrier frequency in Hz
            tx_power_dbm: Default TX power in dBm
            bandwidth_hz: System bandwidth in Hz
            noise_figure_db: UE noise figure in dB
            max_depth: Maximum ray-tracing depth (reflections)
            num_samples: Number of ray samples
            diffraction: Enable diffraction modeling
            scattering: Enable scattering modeling
            use_gpu: Use GPU acceleration if available
        """
        self.frequency_hz = frequency_hz
        self.tx_power_dbm = tx_power_dbm
        self.bandwidth_hz = bandwidth_hz
        self.noise_figure_db = noise_figure_db
        self.penetration_loss_db = penetration_loss_db  # Train/vehicle penetration loss (3GPP TR 38.901 §7.4.3)
        self.max_depth = max_depth
        self.num_samples = num_samples
        self.diffraction = diffraction
        self.scattering = scattering
        self.use_gpu = use_gpu

        # Calculate noise floor (per-RE, matches per-RE RSRP)
        # Legacy calculator: assume 20 MHz BW, 30 kHz SCS
        bandwidth_mhz = bandwidth_hz / 1e6
        scs_hz = 30000.0
        bw_hz_calc = bandwidth_mhz * 1e6
        effective_bw = bw_hz_calc * 0.9
        n_sc = int(effective_bw / scs_hz)
        n_sc = max(12, (n_sc // 12) * 12)
        total_noise = (
            self.DEFAULT_THERMAL_NOISE_DBM_HZ +
            10 * np.log10(bandwidth_hz) +
            noise_figure_db
        )
        self.noise_floor_dbm = total_noise - 10 * np.log10(n_sc)
        logger.info(f"Noise floor (per-RE): {self.noise_floor_dbm:.1f} dBm")

        # Storage
        self.gnb_configs: Dict[Tuple[int, int], GnbConfig] = {}  # (gnb_id, sector_id) -> config
        self.ue_configs: Dict[int, UeConfig] = {}  # ue_id -> config

        # Sionna scene
        self.scene = None
        self._scene_path = scene_path

        # Initialize scene if Sionna is available
        if SIONNA_AVAILABLE:
            self._init_scene(scene_path)
        else:
            logger.warning("Running in mock mode without Sionna")

        # Load gNB from CSV (works in both Sionna and mock mode)
        if gnb_csv_path:
            self.load_gnb_from_csv(gnb_csv_path)

    def _init_scene(self, scene_path: Optional[str] = None):
        """Initialize Sionna RT scene"""
        if not SIONNA_AVAILABLE:
            return

        if scene_path and os.path.exists(scene_path):
            # Load custom scene
            logger.info(f"Loading scene from: {scene_path}")
            self.scene = load_scene(scene_path)
        else:
            # Use built-in Munich scene as fallback
            logger.info("Loading built-in Munich scene")
            self.scene = load_scene(sionna.rt.scene.munich)

        # Set frequency
        self.scene.frequency = self.frequency_hz

        # Configure gNB antenna array (8x4 dual-polarized, 3GPP pattern)
        self.scene.tx_array = PlanarArray(
            num_rows=8,
            num_cols=4,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="VH"
        )

        # Configure UE antenna array (1x1 isotropic)
        self.scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V"
        )

        logger.info(f"Scene initialized: frequency={self.frequency_hz/1e9:.2f} GHz")

    def load_gnb_from_csv(self, csv_path: str) -> int:
        """
        Load gNB configurations from CSV file.

        Expected columns: gnb_id, name, sector_id, x_m, y_m, z_m,
                         azimuth_deg, total_downtilt_deg, tx_power_dBm, etc.

        Args:
            csv_path: Path to CSV file

        Returns:
            Number of gNBs loaded
        """
        df = pd.read_csv(csv_path)

        # Clear existing gNBs
        self.gnb_configs.clear()
        if SIONNA_AVAILABLE and self.scene:
            # Remove existing transmitters
            for tx_name in list(self.scene.transmitters.keys()):
                self.scene.remove(tx_name)

        count = 0
        for _, row in df.iterrows():
            config = GnbConfig.from_csv_row(row)
            key = (config.gnb_id, config.sector_id)
            self.gnb_configs[key] = config

            # Add to Sionna scene
            if SIONNA_AVAILABLE and self.scene:
                tx_name = f"gNB_{config.gnb_id}_S{config.sector_id}"

                # Convert azimuth to radians (Sionna uses radians)
                # Sionna orientation: [alpha, beta, gamma] = [yaw, pitch, roll]
                azimuth_rad = np.deg2rad(config.azimuth_deg)
                downtilt_rad = np.deg2rad(config.downtilt_deg)

                self.scene.add(Transmitter(
                    name=tx_name,
                    position=list(config.position),
                    orientation=[azimuth_rad, downtilt_rad, 0.0]
                ))

            count += 1
            logger.debug(f"Added gNB: {config.name} sector {config.sector_id} at {config.position}")

        logger.info(f"Loaded {count} gNB sectors from {csv_path}")
        return count

    def add_gnb(
        self,
        gnb_id: int,
        position: Tuple[float, float, float],
        sector_id: int = 1,
        azimuth_deg: float = 0.0,
        downtilt_deg: float = 0.0,
        tx_power_dbm: Optional[float] = None
    ) -> GnbConfig:
        """
        Add a single gNB to the calculator.

        Args:
            gnb_id: Unique gNB identifier
            position: (x, y, z) position in meters
            sector_id: Sector identifier
            azimuth_deg: Antenna azimuth in degrees
            downtilt_deg: Antenna downtilt in degrees
            tx_power_dbm: Transmit power in dBm

        Returns:
            Created GnbConfig
        """
        config = GnbConfig(
            gnb_id=gnb_id,
            name=f"gNB_{gnb_id}",
            sector_id=sector_id,
            position=position,
            azimuth_deg=azimuth_deg,
            downtilt_deg=downtilt_deg,
            tx_power_dbm=tx_power_dbm or self.tx_power_dbm
        )

        key = (gnb_id, sector_id)
        self.gnb_configs[key] = config

        # Add to Sionna scene
        if SIONNA_AVAILABLE and self.scene:
            tx_name = f"gNB_{gnb_id}_S{sector_id}"

            # Remove if exists
            if tx_name in self.scene.transmitters:
                self.scene.remove(tx_name)

            azimuth_rad = float(np.deg2rad(azimuth_deg))
            downtilt_rad = float(np.deg2rad(downtilt_deg))

            self.scene.add(Transmitter(
                name=tx_name,
                position=[float(p) for p in position],
                orientation=[azimuth_rad, downtilt_rad, 0.0]
            ))

        logger.debug(f"Added gNB {gnb_id} sector {sector_id} at {position}")
        return config

    def load_ue_from_csv(
        self,
        csv_path: str,
        timestamp: Optional[float] = None
    ) -> int:
        """
        Load UE configurations from CSV file.

        Expected columns: timestamp, ue_id, car_id, x, y, z

        Args:
            csv_path: Path to CSV file
            timestamp: If provided, only load UEs at this timestamp

        Returns:
            Number of UEs loaded
        """
        df = pd.read_csv(csv_path)

        # Filter by timestamp if specified
        if timestamp is not None:
            df = df[df['timestamp'] == timestamp]

        # Clear existing UEs
        self.ue_configs.clear()
        if SIONNA_AVAILABLE and self.scene:
            for rx_name in list(self.scene.receivers.keys()):
                self.scene.remove(rx_name)

        count = 0
        for _, row in df.iterrows():
            config = UeConfig.from_csv_row(row)
            self.ue_configs[config.ue_id] = config

            # Add to Sionna scene
            if SIONNA_AVAILABLE and self.scene:
                rx_name = f"UE_{config.ue_id}"
                self.scene.add(Receiver(
                    name=rx_name,
                    position=list(config.position),
                    orientation=[0.0, 0.0, 0.0]
                ))

            count += 1

        logger.info(f"Loaded {count} UEs from {csv_path}")
        return count

    def add_ue(
        self,
        ue_id: int,
        position: Tuple[float, float, float],
        car_id: Optional[int] = None,
        timestamp: float = 0.0
    ) -> UeConfig:
        """
        Add a single UE to the calculator.

        Args:
            ue_id: Unique UE identifier
            position: (x, y, z) position in meters
            car_id: Optional car/train identifier
            timestamp: Simulation timestamp

        Returns:
            Created UeConfig
        """
        config = UeConfig(
            ue_id=ue_id,
            position=position,
            car_id=car_id,
            timestamp=timestamp
        )

        self.ue_configs[ue_id] = config

        # Add to Sionna scene
        if SIONNA_AVAILABLE and self.scene:
            rx_name = f"UE_{ue_id}"

            # Remove if exists
            if rx_name in self.scene.receivers:
                self.scene.remove(rx_name)

            self.scene.add(Receiver(
                name=rx_name,
                position=list(position),
                orientation=[0.0, 0.0, 0.0]
            ))

        logger.debug(f"Added UE {ue_id} at {position}")
        return config

    def update_ue_position(
        self,
        ue_id: int,
        position: Tuple[float, float, float],
        timestamp: Optional[float] = None
    ):
        """
        Update UE position (for mobility simulation).

        Args:
            ue_id: UE identifier
            position: New (x, y, z) position in meters
            timestamp: Optional timestamp update
        """
        if ue_id not in self.ue_configs:
            logger.warning(f"UE {ue_id} not found, adding new UE")
            self.add_ue(ue_id, position, timestamp=timestamp or 0.0)
            return

        config = self.ue_configs[ue_id]
        config.position = position
        if timestamp is not None:
            config.timestamp = timestamp

        # Update in Sionna scene
        if SIONNA_AVAILABLE and self.scene:
            rx_name = f"UE_{ue_id}"
            if rx_name in self.scene.receivers:
                self.scene.receivers[rx_name].position = list(position)

    def update_ue_positions_from_csv(
        self,
        csv_path: str,
        timestamp: float
    ) -> int:
        """
        Update all UE positions from CSV for a given timestamp.

        Args:
            csv_path: Path to trajectory CSV file
            timestamp: Target timestamp

        Returns:
            Number of UEs updated
        """
        df = pd.read_csv(csv_path)

        # Filter by timestamp (with tolerance for float comparison)
        tolerance = 0.001
        df_filtered = df[np.abs(df['timestamp'] - timestamp) < tolerance]

        if df_filtered.empty:
            logger.warning(f"No UE positions found for timestamp {timestamp}")
            return 0

        count = 0
        for _, row in df_filtered.iterrows():
            ue_id = int(row['ue_id'])
            position = (float(row['x']), float(row['y']), float(row['z']))
            self.update_ue_position(ue_id, position, timestamp)
            count += 1

        return count

    def _compute_paths(self) -> Any:
        """
        Compute ray-tracing paths using Sionna RT.

        Returns:
            Sionna Paths object
        """
        if not SIONNA_AVAILABLE or self.scene is None:
            return None

        if not self.scene.transmitters or not self.scene.receivers:
            logger.warning("No transmitters or receivers in scene")
            return None

        try:
            paths = self.scene.compute_paths(
                max_depth=self.max_depth,
                num_samples=self.num_samples,
                diffraction=self.diffraction,
                scattering=self.scattering
            )
            return paths
        except Exception as e:
            logger.error(f"Error computing paths: {e}")
            return None

    def _compute_rsrp_from_cir(
        self,
        a: Any,
        tau: Any,
        tx_idx: int,
        rx_idx: int,
        tx_power_dbm: float
    ) -> Tuple[float, float, int]:
        """
        Compute RSRP from Channel Impulse Response.

        RSRP = TX_power + sum(path_powers)

        Args:
            a: Complex path amplitudes [batch, rx, tx, paths, time]
            tau: Path delays [batch, rx, tx, paths]
            tx_idx: Transmitter index
            rx_idx: Receiver index
            tx_power_dbm: Transmit power in dBm

        Returns:
            Tuple of (rsrp_dbm, delay_spread_ns, num_paths)
        """
        # Extract path amplitudes for this TX-RX pair
        # a shape: [batch, num_rx, num_tx, num_paths, num_time_samples]
        # tau shape: [batch, num_rx, num_tx, num_paths]

        try:
            # Get amplitudes for specific TX-RX pair
            a_pair = a[0, rx_idx, tx_idx, :, :]  # [paths, time]
            tau_pair = tau[0, rx_idx, tx_idx, :]  # [paths]

            # Convert to numpy
            a_np = np.array(a_pair)
            tau_np = np.array(tau_pair)

            # Count valid paths (non-zero amplitude)
            path_powers = np.sum(np.abs(a_np) ** 2, axis=-1)  # [paths]
            valid_mask = path_powers > 1e-20
            num_paths = int(np.sum(valid_mask))

            if num_paths == 0:
                return -200.0, 0.0, 0  # No paths = very low RSRP

            # Total received power (sum of all path powers)
            total_power = np.sum(path_powers[valid_mask])

            # Convert to dBm
            rsrp_dbm = tx_power_dbm + 10 * np.log10(total_power + 1e-30)

            # Per-RE normalization (3GPP TS 38.215 Section 5.1.1)
            # Legacy calculator: assume 20 MHz BW, 30 kHz SCS
            bandwidth_mhz = 20.0
            scs_hz = 30000.0
            bw_hz = bandwidth_mhz * 1e6
            effective_bw = bw_hz * 0.9
            n_sc = int(effective_bw / scs_hz)
            n_sc = max(12, (n_sc // 12) * 12)
            rsrp_dbm -= 10.0 * np.log10(n_sc)

            # Compute RMS delay spread
            if num_paths > 1:
                # Normalize powers
                p_norm = path_powers[valid_mask] / np.sum(path_powers[valid_mask])
                tau_valid = tau_np[valid_mask]

                # Mean delay
                tau_mean = np.sum(p_norm * tau_valid)

                # RMS delay spread
                tau_rms = np.sqrt(np.sum(p_norm * (tau_valid - tau_mean) ** 2))
                delay_spread_ns = tau_rms * 1e9
            else:
                delay_spread_ns = 0.0

            return rsrp_dbm, delay_spread_ns, num_paths

        except Exception as e:
            logger.error(f"Error computing RSRP from CIR: {e}")
            return -200.0, 0.0, 0

    def _compute_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float]
    ) -> float:
        """Compute 3D Euclidean distance"""
        return np.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )

    def _compute_fspl(self, distance_m: float) -> float:
        """
        Compute Free Space Path Loss.

        FSPL = 20*log10(d) + 20*log10(f) - 147.55 (d in m, f in Hz)
        """
        if distance_m <= 0:
            return 0.0
        return 20 * np.log10(distance_m) + 20 * np.log10(self.frequency_hz) - 147.55

    def compute_rsrp_single(
        self,
        ue_id: int,
        gnb_id: int,
        sector_id: int = 1
    ) -> Optional[ChannelState]:
        """
        Compute RSRP for a single UE-gNB pair.

        This method is less efficient than compute_all() for multiple pairs.

        Args:
            ue_id: UE identifier
            gnb_id: gNB identifier
            sector_id: Sector identifier

        Returns:
            ChannelState or None if computation fails
        """
        if ue_id not in self.ue_configs:
            logger.error(f"UE {ue_id} not found")
            return None

        key = (gnb_id, sector_id)
        if key not in self.gnb_configs:
            logger.error(f"gNB {gnb_id} sector {sector_id} not found")
            return None

        # Compute all and extract single result
        results = self.compute_all()
        return results.get((ue_id, gnb_id, sector_id))

    def compute_all(
        self,
        timestamp: Optional[float] = None,
        **kwargs
    ) -> Dict[Tuple[int, int, int], ChannelState]:
        """
        Compute channel state for all UE-gNB pairs.

        This is the main computation method that:
        1. Runs ray-tracing for all TX-RX pairs
        2. Extracts CIR and computes RSRP
        3. Calculates SINR with multi-cell interference

        Args:
            timestamp: Optional timestamp for logging

        Returns:
            Dictionary mapping (ue_id, gnb_id, sector_id) to ChannelState
        """
        results: Dict[Tuple[int, int, int], ChannelState] = {}

        if not self.gnb_configs or not self.ue_configs:
            logger.warning("No gNBs or UEs configured")
            return results

        # Step 1: Compute paths (ray-tracing)
        if SIONNA_AVAILABLE and self.scene:
            paths = self._compute_paths()

            if paths is not None:
                # Get CIR
                try:
                    a, tau = paths.cir()
                except Exception as e:
                    logger.error(f"Error getting CIR: {e}")
                    a, tau = None, None
            else:
                a, tau = None, None
        else:
            a, tau = None, None

        # Build TX/RX name to index mapping
        tx_name_to_idx = {}
        rx_name_to_idx = {}

        if SIONNA_AVAILABLE and self.scene:
            for idx, name in enumerate(self.scene.transmitters.keys()):
                tx_name_to_idx[name] = idx
            for idx, name in enumerate(self.scene.receivers.keys()):
                rx_name_to_idx[name] = idx

        # Step 2: Compute RSRP for all pairs
        rsrp_per_ue: Dict[int, Dict[Tuple[int, int], float]] = {}

        for ue_id, ue_config in self.ue_configs.items():
            rsrp_per_ue[ue_id] = {}

            for (gnb_id, sector_id), gnb_config in self.gnb_configs.items():
                # Get TX/RX indices
                tx_name = f"gNB_{gnb_id}_S{sector_id}"
                rx_name = f"UE_{ue_id}"

                # Compute distance
                distance_m = self._compute_distance(ue_config.position, gnb_config.position)

                # Compute RSRP
                if a is not None and tau is not None:
                    tx_idx = tx_name_to_idx.get(tx_name)
                    rx_idx = rx_name_to_idx.get(rx_name)

                    if tx_idx is not None and rx_idx is not None:
                        rsrp_dbm, delay_spread_ns, num_paths = self._compute_rsrp_from_cir(
                            a, tau, tx_idx, rx_idx, gnb_config.tx_power_dbm
                        )
                    else:
                        # Fallback to FSPL
                        path_loss = self._compute_fspl(distance_m)
                        rsrp_dbm = gnb_config.tx_power_dbm - path_loss
                        delay_spread_ns = 0.0
                        num_paths = 1
                else:
                    # Mock mode: use FSPL (deterministic for testing)
                    path_loss = self._compute_fspl(distance_m)
                    rsrp_dbm = gnb_config.tx_power_dbm - path_loss
                    delay_spread_ns = 0.0
                    num_paths = 1

                # Apply penetration loss (e.g., train body loss per 3GPP TR 38.901 §7.4.3)
                rsrp_dbm -= self.penetration_loss_db

                rsrp_per_ue[ue_id][(gnb_id, sector_id)] = rsrp_dbm

                # Create initial ChannelState
                state = ChannelState(
                    ue_id=ue_id,
                    gnb_id=gnb_id,
                    sector_id=sector_id,
                    rsrp_dbm=rsrp_dbm,
                    delay_spread_ns=delay_spread_ns,
                    num_paths=num_paths,
                    distance_m=distance_m,
                    timestamp=timestamp or ue_config.timestamp,
                    path_loss_db=gnb_config.tx_power_dbm - rsrp_dbm
                )

                results[(ue_id, gnb_id, sector_id)] = state

        # Step 3: Compute SINR with interference
        for (ue_id, gnb_id, sector_id), state in results.items():
            # Signal power (linear)
            signal_power = 10 ** (state.rsrp_dbm / 10)

            # Interference power (sum of all other cells)
            interference_power = 0.0
            for (other_gnb_id, other_sector_id), other_rsrp in rsrp_per_ue[ue_id].items():
                if other_gnb_id != gnb_id or other_sector_id != sector_id:
                    interference_power += 10 ** (other_rsrp / 10)

            # Noise power
            noise_power = 10 ** (self.noise_floor_dbm / 10)

            # SINR
            sinr_linear = signal_power / (interference_power + noise_power)
            state.sinr_db = 10 * np.log10(sinr_linear + 1e-30)

            # RSRQ approximation (simplified)
            # RSRQ = N * RSRP / RSSI, where RSSI = signal + interference + noise
            rssi_linear = signal_power + interference_power + noise_power
            n_rb = 100  # Assume 100 RBs
            rsrq_linear = n_rb * signal_power / rssi_linear
            state.rsrq_db = 10 * np.log10(rsrq_linear + 1e-30)

        # Step 4: Compute UL SINR via TDD reciprocity (hardcoded defaults)
        _ul_noise_floor = compute_noise_floor(self.bandwidth_hz, 2.5)  # gNB NF=2.5dB
        for (ue_id, gnb_id, sector_id), state in results.items():
            gnb_config = self.gnb_configs.get((gnb_id, sector_id))
            tx_ref = gnb_config.tx_power_dbm if gnb_config else self.tx_power_dbm
            state.ul_sinr_db = compute_ul_sinr(
                dl_rsrp_dbm=state.rsrp_dbm,
                gnb_ref_tx_power_dbm=tx_ref,
                ue_tx_power_dbm=23.0,
                ul_noise_floor_dbm=_ul_noise_floor,
            )

        logger.debug(f"Computed {len(results)} channel states")
        return results

    def get_serving_cell(
        self,
        ue_id: int,
        results: Optional[Dict[Tuple[int, int, int], ChannelState]] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Get the best serving cell (highest RSRP) for a UE.

        Args:
            ue_id: UE identifier
            results: Pre-computed results (optional, will compute if None)

        Returns:
            Tuple of (gnb_id, sector_id) or None
        """
        if results is None:
            results = self.compute_all()

        best_cell = None
        best_rsrp = -np.inf

        for (uid, gnb_id, sector_id), state in results.items():
            if uid == ue_id and state.rsrp_dbm > best_rsrp:
                best_rsrp = state.rsrp_dbm
                best_cell = (gnb_id, sector_id)

        return best_cell

    def get_neighbor_cells(
        self,
        ue_id: int,
        serving_gnb_id: int,
        serving_sector_id: int,
        results: Optional[Dict[Tuple[int, int, int], ChannelState]] = None,
        rsrp_threshold_dbm: float = -120.0
    ) -> List[ChannelState]:
        """
        Get neighbor cells sorted by RSRP.

        Args:
            ue_id: UE identifier
            serving_gnb_id: Current serving gNB
            serving_sector_id: Current serving sector
            results: Pre-computed results
            rsrp_threshold_dbm: Minimum RSRP to consider

        Returns:
            List of ChannelState for neighbors, sorted by RSRP (descending)
        """
        if results is None:
            results = self.compute_all()

        neighbors = []
        for (uid, gnb_id, sector_id), state in results.items():
            if uid == ue_id:
                # Exclude serving cell
                if gnb_id == serving_gnb_id and sector_id == serving_sector_id:
                    continue
                # Apply threshold
                if state.rsrp_dbm >= rsrp_threshold_dbm:
                    neighbors.append(state)

        # Sort by RSRP (descending)
        neighbors.sort(key=lambda x: x.rsrp_dbm, reverse=True)
        return neighbors

    def clear(self):
        """Clear all gNBs and UEs"""
        self.gnb_configs.clear()
        self.ue_configs.clear()

        if SIONNA_AVAILABLE and self.scene:
            for name in list(self.scene.transmitters.keys()):
                self.scene.remove(name)
            for name in list(self.scene.receivers.keys()):
                self.scene.remove(name)

        logger.info("Channel calculator cleared")


def _test_channel_calculator():
    """Test function for SionnaChannelCalculator"""
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("SionnaChannelCalculator Test")
    print("=" * 60)

    # Create calculator (will use mock mode if Sionna not available)
    calculator = SionnaChannelCalculator(
        frequency_hz=3.5e9,
        tx_power_dbm=46.0,
        bandwidth_hz=20e6
    )

    # Add test gNBs
    gnb_positions = [
        (0, 100, 30),
        (500, 100, 30),
        (1000, 100, 30)
    ]

    for i, pos in enumerate(gnb_positions):
        calculator.add_gnb(
            gnb_id=i + 1,
            position=pos,
            sector_id=1,
            azimuth_deg=180,  # Pointing towards y=0
            tx_power_dbm=46.0
        )
        print(f"Added gNB_{i+1} at position {pos}")

    # Add test UE
    ue_position = (250, 0, 1.5)
    calculator.add_ue(ue_id=0, position=ue_position)
    print(f"Added UE_0 at position {ue_position}")

    # Compute channel states
    print("\n" + "-" * 60)
    print("Computing channel states...")
    print("-" * 60)

    results = calculator.compute_all()

    print(f"\nResults ({len(results)} channel states):\n")
    print(f"{'UE':<6} {'gNB':<6} {'Sector':<8} {'RSRP (dBm)':<12} {'SINR (dB)':<12} {'Distance (m)':<12}")
    print("-" * 60)

    for (ue_id, gnb_id, sector_id), state in sorted(results.items()):
        print(f"{state.ue_id:<6} {state.gnb_id:<6} {state.sector_id:<8} "
              f"{state.rsrp_dbm:<12.1f} {state.sinr_db:<12.1f} {state.distance_m:<12.1f}")

    # Find serving cell
    serving = calculator.get_serving_cell(ue_id=0, results=results)
    if serving:
        print(f"\nServing cell for UE_0: gNB_{serving[0]} sector {serving[1]}")

    # Get neighbors
    if serving:
        neighbors = calculator.get_neighbor_cells(
            ue_id=0,
            serving_gnb_id=serving[0],
            serving_sector_id=serving[1],
            results=results
        )
        print(f"Neighbor cells: {[(n.gnb_id, n.sector_id, f'{n.rsrp_dbm:.1f}dBm') for n in neighbors]}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    _test_channel_calculator()
