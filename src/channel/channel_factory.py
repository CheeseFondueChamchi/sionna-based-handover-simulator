"""
Channel Model Factory for NR Handover Simulation

Provides factory pattern for creating channel model instances with
automatic fallback when requested model is unavailable.

Author: Claude Code
Date: 2026-02-02
"""

import logging
from typing import Dict, Type, List

from .channel_model import ChannelModel, ChannelConfig, ChannelModelType

logger = logging.getLogger(__name__)

# Check Sionna availability
try:
    import sionna
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    logger.warning("Sionna not available - SionnaRTChannelModel will not be registered")


class ChannelModelFactory:
    """
    Factory for creating channel model instances.

    Handles automatic fallback when requested model is unavailable:
    - If SIONNA_RT requested but Sionna not installed -> falls back to STATISTICAL
    """

    _registry: Dict[ChannelModelType, Type[ChannelModel]] = {}

    @classmethod
    def register(cls, model_type: ChannelModelType, model_class: Type[ChannelModel]) -> None:
        """Register a channel model class"""
        cls._registry[model_type] = model_class
        logger.info(f"Registered channel model: {model_type.value}")

    @classmethod
    def create(cls, config: ChannelConfig) -> ChannelModel:
        """
        Create a channel model from configuration.

        Handles fallback logic:
        - If SIONNA_RT requested but unavailable, falls back to STATISTICAL
        """
        requested_type = config.model_type

        # Check if requested model is available
        if requested_type not in cls._registry:
            # Fallback logic
            if requested_type == ChannelModelType.SIONNA_RT:
                logger.warning(
                    f"SionnaRTChannelModel not available (Sionna not installed). "
                    f"Falling back to StatisticalChannelModel."
                )
                if ChannelModelType.STATISTICAL in cls._registry:
                    config.model_type = ChannelModelType.STATISTICAL
                else:
                    raise ValueError("No channel models available!")
            else:
                raise ValueError(f"Unknown model type: {requested_type}")

        model_class = cls._registry[config.model_type]

        # Try to instantiate the model
        try:
            model = model_class()
            model.configure(config)
            return model
        except ImportError as e:
            # Handle case where model is registered but dependencies are missing
            if requested_type == ChannelModelType.SIONNA_RT:
                logger.warning(
                    f"Failed to create SionnaRTChannelModel (missing dependencies). "
                    f"Falling back to StatisticalChannelModel. Error: {e}"
                )
                if ChannelModelType.STATISTICAL in cls._registry:
                    config.model_type = ChannelModelType.STATISTICAL
                    fallback_class = cls._registry[config.model_type]
                    model = fallback_class()
                    model.configure(config)
                    return model
                else:
                    raise ValueError("No fallback channel model available!")
            else:
                # Re-raise if not a Sionna-related import error
                raise

    @classmethod
    def create_by_name(cls, name: str, **kwargs) -> ChannelModel:
        """Create a channel model by name with keyword arguments"""
        model_type = ChannelModelType(name.lower())
        config = ChannelConfig(model_type=model_type, **kwargs)
        return cls.create(config)

    @classmethod
    def available_models(cls) -> List[str]:
        """List available model types"""
        return [t.value for t in cls._registry.keys()]

    @classmethod
    def is_sionna_available(cls) -> bool:
        """Check if Sionna RT is available"""
        return SIONNA_AVAILABLE


def _register_models():
    """Auto-register available models"""
    # Always register statistical model
    from .statistical_model import StatisticalChannelModel
    ChannelModelFactory.register(ChannelModelType.STATISTICAL, StatisticalChannelModel)

    # Only register Sionna RT if available
    if SIONNA_AVAILABLE:
        from .sionna_rt_model import SionnaRTChannelModel
        ChannelModelFactory.register(ChannelModelType.SIONNA_RT, SionnaRTChannelModel)


# Auto-register on module import
_register_models()


def _test_factory():
    """Test function for ChannelModelFactory."""
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("ChannelModelFactory Test")
    print("=" * 70)

    # Check available models
    print(f"\nAvailable models: {ChannelModelFactory.available_models()}")
    print(f"Sionna available: {ChannelModelFactory.is_sionna_available()}")

    # Create statistical model
    print("\n" + "-" * 70)
    print("Creating StatisticalChannelModel...")
    print("-" * 70)

    config = ChannelConfig(
        model_type=ChannelModelType.STATISTICAL,
        frequency_hz=3.5e9,
        scenario="UMa"
    )
    model = ChannelModelFactory.create(config)
    print(f"Created model: {model.__class__.__name__}")
    print(f"Model type: {model.model_type}")

    # Test create_by_name
    print("\n" + "-" * 70)
    print("Creating model by name...")
    print("-" * 70)

    model2 = ChannelModelFactory.create_by_name("statistical", frequency_hz=2.6e9)
    print(f"Created model: {model2.__class__.__name__}")

    # Test fallback behavior (request Sionna if not available)
    if not ChannelModelFactory.is_sionna_available():
        print("\n" + "-" * 70)
        print("Testing fallback from SIONNA_RT to STATISTICAL...")
        print("-" * 70)

        config_sionna = ChannelConfig(
            model_type=ChannelModelType.SIONNA_RT,
            frequency_hz=3.5e9
        )
        model3 = ChannelModelFactory.create(config_sionna)
        print(f"Requested: SIONNA_RT")
        print(f"Got: {model3.__class__.__name__}")
        print(f"Model type: {model3.model_type}")

    print("\n" + "=" * 70)
    print("Factory test completed!")
    print("=" * 70)


if __name__ == "__main__":
    _test_factory()
