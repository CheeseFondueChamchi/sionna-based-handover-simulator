"""
Channel module for NR Handover Simulation

Provides channel calculation with:
- Ray-tracing based propagation modeling (Sionna RT)
- Multi-cell RSRP/SINR computation
- Support for custom 3D scenes
- Abstract channel model interface
"""

from .channel_calculator import (
    SionnaChannelCalculator,
    ChannelState,
    GnbConfig,
    UeConfig,
    SIONNA_AVAILABLE,
)

from .channel_model import (
    ChannelModel,
    ChannelConfig,
    ChannelModelType,
    DopplerInfo,
)

from .statistical_model import StatisticalChannelModel
from .multi_freq_model import MultiFreqChannelModel

from .channel_factory import ChannelModelFactory

from .post_processing import (
    compute_noise_floor,
    apply_propagation_losses,
    compute_sinr,
    compute_sinr_linear_components,
    compute_rsrq,
    compute_ul_sinr,
)

from .trajectory_channel_map import (
    TrajectoryChannelMap,
    TrajectoryChannelMapConfig,
)

# Import SionnaRTChannelModel only if Sionna is available
if SIONNA_AVAILABLE:
    from .sionna_rt_model import SionnaRTChannelModel
    __all__ = [
        # Original exports
        'SionnaChannelCalculator',
        'ChannelState',
        'GnbConfig',
        'UeConfig',
        'SIONNA_AVAILABLE',
        # Abstract interface
        'ChannelModel',
        'ChannelConfig',
        'ChannelModelType',
        'DopplerInfo',
        # Implementations
        'StatisticalChannelModel',
        'MultiFreqChannelModel',
        'SionnaRTChannelModel',
        # Factory
        'ChannelModelFactory',
        # Pre-computed map
        'TrajectoryChannelMap',
        'TrajectoryChannelMapConfig',
        # Post-processing (shared)
        'compute_noise_floor',
        'apply_propagation_losses',
        'compute_sinr',
        'compute_sinr_linear_components',
        'compute_rsrq',
    'compute_ul_sinr',
    ]
else:
    __all__ = [
        # Original exports
        'SionnaChannelCalculator',
        'ChannelState',
        'GnbConfig',
        'UeConfig',
        'SIONNA_AVAILABLE',
        # Abstract interface
        'ChannelModel',
        'ChannelConfig',
        'ChannelModelType',
        'DopplerInfo',
        # Implementations
        'StatisticalChannelModel',
        'MultiFreqChannelModel',
        # Factory
        'ChannelModelFactory',
        # Pre-computed map
        'TrajectoryChannelMap',
        'TrajectoryChannelMapConfig',
        # Post-processing (shared)
        'compute_noise_floor',
        'apply_propagation_losses',
        'compute_sinr',
        'compute_sinr_linear_components',
        'compute_rsrq',
    'compute_ul_sinr',
    ]
