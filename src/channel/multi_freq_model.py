"""
Multi-Frequency Channel Model for Inter-RAT Handover Simulation

Groups gNBs by frequency band and creates one StatisticalChannelModel per band.
Cross-band interference is NOT modeled (different frequencies don't interfere).

This enables mixed NR (3.5 GHz) + LTE (0.9 GHz, 1.8 GHz) deployments.
"""

import logging
from copy import copy
from typing import Dict, List, Optional, Tuple

from .channel_model import ChannelModel, ChannelConfig, ChannelModelType, DopplerInfo
from .channel_calculator import ChannelState, GnbConfig, UeConfig
from .statistical_model import StatisticalChannelModel

logger = logging.getLogger(__name__)


class MultiFreqChannelModel(ChannelModel):
    """
    Wrapper holding one StatisticalChannelModel per frequency band.

    For mixed-frequency deployments (e.g., NR 3.5 GHz + LTE 0.9 GHz),
    each band gets independent path loss, shadow fading, and SINR computation.
    Cross-band interference is not modeled (physically correct for different frequencies).
    """

    def __init__(self):
        self._band_models: Dict[float, StatisticalChannelModel] = {}  # band_key -> model
        self._gnb_band_map: Dict[int, float] = {}   # gnb_id -> band_key
        self._gnb_rat_map: Dict[int, str] = {}       # gnb_id -> "nr"/"lte"
        self._base_config: Optional[ChannelConfig] = None
        self._ue_configs: Dict[int, Tuple[Tuple[float, float, float], dict]] = {}  # ue_id -> (position, kwargs)

    @property
    def model_type(self) -> ChannelModelType:
        return ChannelModelType.STATISTICAL

    def configure(self, config: ChannelConfig) -> None:
        """Store base config for creating per-band sub-models."""
        self._base_config = config

    def add_gnb(self, gnb_id: int, position: Tuple[float, float, float], **kwargs) -> None:
        """Add a gNB, routing it to the correct band sub-model."""
        frequency_ghz = kwargs.get('frequency_ghz', 3.5)
        rat_type = kwargs.get('rat_type', 'nr')
        band_key = round(frequency_ghz, 1)

        if band_key not in self._band_models:
            if self._base_config is None:
                raise RuntimeError("MultiFreqChannelModel.configure() must be called before add_gnb()")
            # Create new sub-model for this band
            model = StatisticalChannelModel()
            band_config = copy(self._base_config)
            band_config.frequency_hz = frequency_ghz * 1e9
            model.configure(band_config)
            self._band_models[band_key] = model

            # Add any existing UEs to the new band model
            for ue_id, (ue_pos, ue_kwargs) in self._ue_configs.items():
                model.add_ue(ue_id, ue_pos, **ue_kwargs)

            logger.info(f"Created band sub-model for {frequency_ghz} GHz ({rat_type})")

        # Remove frequency_ghz and rat_type from kwargs before passing to sub-model
        sub_kwargs = {k: v for k, v in kwargs.items() if k not in ('frequency_ghz', 'rat_type')}
        self._band_models[band_key].add_gnb(gnb_id, position, **sub_kwargs)
        self._gnb_band_map[gnb_id] = band_key
        self._gnb_rat_map[gnb_id] = rat_type

    def add_ue(self, ue_id: int, position: Tuple[float, float, float], **kwargs) -> None:
        """Add a UE to ALL band sub-models."""
        self._ue_configs[ue_id] = (position, kwargs)
        for model in self._band_models.values():
            model.add_ue(ue_id, position, **kwargs)

    def update_ue_position(
        self, ue_id: int, position: Tuple[float, float, float],
        velocity: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """Update UE position in ALL band sub-models."""
        # Update stored config
        if ue_id in self._ue_configs:
            old_pos, old_kwargs = self._ue_configs[ue_id]
            self._ue_configs[ue_id] = (position, old_kwargs)

        for model in self._band_models.values():
            model.update_ue_position(ue_id, position, velocity=velocity)

    def compute_all(self, timestamp: Optional[float] = None, **kwargs) -> Dict[Tuple[int, int, int], ChannelState]:
        """Compute channel states from all bands and merge results."""
        results = {}
        for band_key, model in self._band_models.items():
            band_results = model.compute_all(timestamp, **kwargs)
            for key, state in band_results.items():
                gnb_id = key[1]
                state.rat_type = self._gnb_rat_map.get(gnb_id, "nr")
                results[key] = state
        return results

    def compute_doppler(self, ue_id: int) -> Optional[DopplerInfo]:
        """Compute Doppler from the first band model that has this UE."""
        for model in self._band_models.values():
            result = model.compute_doppler(ue_id)
            if result is not None:
                return result
        return None

    def activate_nearby_gnbs(
        self,
        center: Tuple[float, float, float],
        radius_m: float = 2000.0,
        max_count: Optional[int] = None,
        frequency_ghz: Optional[float] = None,
        min_count: int = 10,
    ) -> int:
        """Activate nearby gNBs in all band sub-models (NR + LTE)."""
        total = 0
        for band_key, model in self._band_models.items():
            if frequency_ghz is not None and abs(band_key - frequency_ghz) > 0.01:
                continue
            total += model.activate_nearby_gnbs(
                center, radius_m, max_count,
                frequency_ghz=None, min_count=min_count,
            )
        return total

    def ensure_gnb_active(self, gnb_id: int) -> bool:
        """Ensure a specific gNB is active in its band sub-model."""
        band_key = self._gnb_band_map.get(gnb_id)
        if band_key is None:
            return False
        model = self._band_models.get(band_key)
        if model is None:
            return False
        return model.ensure_gnb_active(gnb_id)

    def clear(self) -> None:
        """Clear all band sub-models."""
        for model in self._band_models.values():
            model.clear()
        self._band_models.clear()
        self._gnb_band_map.clear()
        self._gnb_rat_map.clear()
        self._ue_configs.clear()

    @property
    def band_count(self) -> int:
        """Number of frequency bands configured."""
        return len(self._band_models)

    @property
    def gnb_count(self) -> int:
        """Total number of gNBs across all bands."""
        return len(self._gnb_band_map)
