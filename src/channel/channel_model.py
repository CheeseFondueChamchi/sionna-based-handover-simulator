"""
Abstract Channel Model Interface

Defines the interface for channel models in the NR handover simulation.
Supports both statistical models and Sionna RT ray-tracing models.

Author: Claude Code
Date: 2026-02-02
3GPP Reference: TS 38.901 (Channel Model for NR)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .channel_calculator import ChannelState


class ChannelModelType(Enum):
    """Types of channel models available."""
    STATISTICAL = "statistical"  # 3GPP statistical model (CDL/TDL)
    SIONNA_RT = "sionna_rt"      # Sionna ray-tracing
    HYBRID = "hybrid"            # Combination of both


@dataclass
class ChannelConfig:
    """
    Channel model configuration.

    Attributes:
        model_type: Type of channel model to use
        frequency_hz: Carrier frequency in Hz (default: 3.5 GHz)
        bandwidth_hz: System bandwidth in Hz (default: 100 MHz)
        tx_power_dbm: gNB transmit power in dBm (default: 46 dBm)
        noise_figure_db: UE noise figure in dB (default: 7 dB)
        scenario: 3GPP scenario for statistical model (UMa, UMi, RMa)
        los_probability: LOS probability model ("3gpp", "always", "never")
        scene_path: Path to Sionna RT scene XML file
        max_depth: Maximum reflection depth for ray-tracing
        num_samples: Number of samples for ray-tracing
        diffraction: Enable diffraction in ray-tracing
        scattering: Enable scattering in ray-tracing

    3GPP Reference:
        TS 38.901 §7.4 (UMa scenario)
        TS 38.901 §7.5 (UMi scenario)
        TS 38.901 §7.6 (RMa scenario)
    """
    model_type: ChannelModelType
    frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 46.0
    noise_figure_db: float = 7.0

    # Additional loss parameters (3GPP TR 38.901 §7.4.3)
    penetration_loss_db: float = 0.0  # Vehicle/building penetration loss (e.g., 20 dB for train)

    # Train surface distortion (random variable per link per timestep)
    surface_distortion_mean_db: float = 18.0  # Mean signal reduction from train body (dB)
    surface_distortion_std_db: float = 4.0    # Std deviation of distortion (dB)

    # Statistical model parameters
    scenario: str = "UMa"
    los_probability: Optional[str] = "3gpp"

    # Sionna RT parameters
    scene_path: Optional[str] = None
    max_depth: int = 5
    num_samples: float = 1e6
    diffraction: bool = True
    scattering: bool = True

    # UL SINR parameters (TDD reciprocity)
    gnb_noise_figure_db: float = 2.5   # gNB receiver NF (TS 38.104 Table 7.4.1-1, FR1 Wide Area BS)
    ue_tx_power_dbm: float = 23.0      # UE max TX power (TS 38.101-1 Table 6.2.1-1, Power Class 3)

    # Path filtering for RT model
    max_path_length_m: float = 1500.0  # Discard ray paths longer than this (meters)

    # Antenna array configuration
    # Detected from CSV antenna_type or set manually
    # e.g., "2T2R", "32T2R", "64T64R", "8x4_VH"
    tx_antenna_config: str = "auto"


@dataclass
class DopplerInfo:
    """
    Doppler shift information for a moving UE.

    Attributes:
        ue_id: UE identifier
        velocity_mps: Velocity vector in m/s (vx, vy, vz)
        doppler_shift_hz: Maximum Doppler shift in Hz
        coherence_time_ms: Channel coherence time in milliseconds

    3GPP Reference:
        TS 38.101-1 §6.2 (Doppler spread requirements)
    """
    ue_id: int
    velocity_mps: Tuple[float, float, float]
    doppler_shift_hz: float
    coherence_time_ms: float


class ChannelModel(ABC):
    """
    Abstract base class for channel models.

    Defines the interface that all channel models must implement,
    whether statistical (3GPP CDL/TDL) or ray-tracing (Sionna RT).

    Workflow:
        1. configure() - Set channel parameters
        2. add_gnb() - Add base stations
        3. add_ue() - Add user equipment
        4. update_ue_position() - Update UE locations (mobility)
        5. compute_all() - Calculate channel state for all links
        6. compute_doppler() - Calculate Doppler for moving UEs

    3GPP Reference:
        TS 38.901 (Channel Model for Frequency Spectrum above 6 GHz)
    """

    @abstractmethod
    def configure(self, config: ChannelConfig) -> None:
        """
        Configure the channel model.

        Args:
            config: Channel configuration parameters

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
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
            **kwargs: Additional gNB parameters (antenna array, orientation, etc.)

        Raises:
            ValueError: If gnb_id already exists or position is invalid

        3GPP Reference:
            TS 38.104 §5.2 (BS RF requirements)
        """
        pass

    @abstractmethod
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
            **kwargs: Additional UE parameters (antenna array, orientation, etc.)

        Raises:
            ValueError: If ue_id already exists or position is invalid

        3GPP Reference:
            TS 38.101-1 §5.2 (UE RF requirements)
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def compute_all(
        self,
        timestamp: Optional[float] = None,
        **kwargs
    ) -> Dict[Tuple[int, int, int], 'ChannelState']:
        """
        Compute channel state for all (gNB, UE, cell) links.

        Args:
            timestamp: Optional simulation timestamp in seconds

        Returns:
            Dictionary mapping (gnb_id, ue_id, cell_id) to ChannelState

        Note:
            cell_id is typically 0 for single-cell gNBs.
            For multi-beam gNBs, cell_id identifies the beam/sector.

        3GPP Reference:
            TS 38.214 §5.1 (CSI framework)
        """
        pass

    @abstractmethod
    def compute_doppler(self, ue_id: int) -> Optional[DopplerInfo]:
        """
        Compute Doppler information for a moving UE.

        Args:
            ue_id: UE identifier

        Returns:
            DopplerInfo if UE has velocity, None otherwise

        3GPP Reference:
            TS 38.101-1 §6.2.2 (Doppler spread)
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all gNBs, UEs, and computed states.

        Used for resetting the simulation.
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> ChannelModelType:
        """
        Get the channel model type.

        Returns:
            ChannelModelType enum value
        """
        pass
