"""
HDF5 Saver for Sionna RT Channel Simulation Results

Saves CIR (Channel Impulse Response), scene configuration, and rendered images
to HDF5 files for later analysis and visualization.

Usage:
    from src.channel.hdf5_saver import SimulationSaver

    saver = SimulationSaver(output_dir="output/hdf5")
    saver.save_snapshot(scene, paths, timestamp, config_meta)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logger.warning("h5py not available. HDF5 saving disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import io
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


class SimulationSaver:
    """
    Saves Sionna RT simulation snapshots to HDF5 files.

    Each snapshot includes:
    - CIR data (a, tau)
    - Scene configuration (TX/RX positions, frequency)
    - Rendered scene image (optional)
    - Metadata (timestamp, config)
    """

    def __init__(self, output_dir: str = "output/hdf5", prefix: str = "sim"):
        """
        Initialize the simulation saver.

        Args:
            output_dir: Directory to save HDF5 files
            prefix: Filename prefix
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.snapshot_count = 0

        if not HDF5_AVAILABLE:
            logger.warning("h5py not installed. HDF5 saving will be disabled.")

    def save_snapshot(
        self,
        scene: Any,
        paths: Any,
        timestamp: float,
        config_meta: Dict[str, Any],
        save_render: bool = True,
        ue_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Save a simulation snapshot to HDF5.

        Args:
            scene: Sionna scene object
            paths: Sionna paths object from PathSolver
            timestamp: Simulation timestamp in seconds
            config_meta: Configuration metadata dict
            save_render: Whether to save rendered scene image
            ue_id: Optional UE ID for filename

        Returns:
            Path to saved file, or None if saving failed
        """
        if not HDF5_AVAILABLE:
            logger.warning("h5py not available, skipping HDF5 save")
            return None

        # Generate filename
        if ue_id is not None:
            filename = f"{self.prefix}_ue{ue_id}_t{timestamp:.2f}s_{self.snapshot_count:05d}.h5"
        else:
            filename = f"{self.prefix}_t{timestamp:.2f}s_{self.snapshot_count:05d}.h5"

        filepath = self.output_dir / filename

        try:
            # Get CIR data
            a, tau = paths.cir(out_type="numpy")

            with h5py.File(filepath, "w") as f:
                # === CIR Data ===
                cir_grp = f.create_group("cir")

                # Handle complex array (may be tuple of real/imag or complex)
                if isinstance(a, tuple) and len(a) == 2:
                    # Tuple format: (real, imag)
                    cir_grp.create_dataset("a_real", data=np.array(a[0]), compression="gzip")
                    cir_grp.create_dataset("a_imag", data=np.array(a[1]), compression="gzip")
                else:
                    # Complex array
                    a_np = np.array(a)
                    cir_grp.create_dataset("a_real", data=a_np.real, compression="gzip")
                    cir_grp.create_dataset("a_imag", data=a_np.imag, compression="gzip")

                cir_grp.create_dataset("tau", data=np.array(tau), compression="gzip")

                # === Scene Configuration ===
                scene_grp = f.create_group("scene")
                scene_grp.attrs["scene_path"] = config_meta.get("scene_path", "unknown")

                # Frequency
                try:
                    freq = float(scene.frequency.numpy()) if hasattr(scene.frequency, 'numpy') else float(scene.frequency)
                    scene_grp.attrs["frequency_hz"] = freq
                except:
                    scene_grp.attrs["frequency_hz"] = config_meta.get("frequency_hz", 3.5e9)

                # Transmitters
                tx_grp = scene_grp.create_group("transmitters")
                for name, tx in scene.transmitters.items():
                    g = tx_grp.create_group(name)
                    try:
                        pos = tx.position.numpy() if hasattr(tx.position, 'numpy') else np.array(tx.position)
                        orient = tx.orientation.numpy() if hasattr(tx.orientation, 'numpy') else np.array(tx.orientation)
                        g.create_dataset("position", data=pos)
                        g.create_dataset("orientation", data=orient)
                    except Exception as e:
                        logger.debug(f"Could not save TX {name} details: {e}")

                # Receivers
                rx_grp = scene_grp.create_group("receivers")
                for name, rx in scene.receivers.items():
                    g = rx_grp.create_group(name)
                    try:
                        pos = rx.position.numpy() if hasattr(rx.position, 'numpy') else np.array(rx.position)
                        g.create_dataset("position", data=pos)

                        # Velocity (may not always be set)
                        if hasattr(rx, 'velocity') and rx.velocity is not None:
                            vel = rx.velocity.numpy() if hasattr(rx.velocity, 'numpy') else np.array(rx.velocity)
                            g.create_dataset("velocity", data=vel)
                    except Exception as e:
                        logger.debug(f"Could not save RX {name} details: {e}")

                # === Metadata ===
                f.attrs["timestamp"] = timestamp
                f.attrs["max_depth"] = config_meta.get("max_depth", 5)
                f.attrs["samples_per_src"] = config_meta.get("samples_per_src", 1e6)
                f.attrs["snapshot_index"] = self.snapshot_count

                # === Rendered Image (optional) ===
                if save_render and MPL_AVAILABLE and PIL_AVAILABLE:
                    try:
                        img_grp = f.create_group("images")

                        # Render scene with paths
                        scene.render(paths=paths)

                        # Capture to buffer
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        plt.close()
                        buf.seek(0)

                        # Convert to numpy array
                        img_array = np.array(Image.open(buf))
                        img_grp.create_dataset("render", data=img_array, compression="gzip")

                    except Exception as e:
                        logger.debug(f"Could not save rendered image: {e}")

            self.snapshot_count += 1
            logger.info(f"Saved simulation snapshot to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save simulation snapshot: {e}")
            return None

    def save_cir_only(
        self,
        a: np.ndarray,
        tau: np.ndarray,
        timestamp: float,
        rsrp_values: Dict[int, float],
        ue_position: tuple,
        serving_cell: int,
        ue_id: int = 0
    ) -> Optional[str]:
        """
        Save CIR data without requiring Sionna scene (for mock mode).

        Args:
            a: CIR amplitudes
            tau: CIR delays
            timestamp: Simulation timestamp
            rsrp_values: Dict of {gnb_id: rsrp_dbm}
            ue_position: UE position tuple (x, y, z)
            serving_cell: Current serving cell ID
            ue_id: UE identifier

        Returns:
            Path to saved file
        """
        if not HDF5_AVAILABLE:
            return None

        filename = f"{self.prefix}_ue{ue_id}_t{timestamp:.2f}s_{self.snapshot_count:05d}.h5"
        filepath = self.output_dir / filename

        try:
            with h5py.File(filepath, "w") as f:
                # CIR Data
                cir_grp = f.create_group("cir")
                if np.iscomplexobj(a):
                    cir_grp.create_dataset("a_real", data=a.real, compression="gzip")
                    cir_grp.create_dataset("a_imag", data=a.imag, compression="gzip")
                else:
                    cir_grp.create_dataset("a_real", data=a, compression="gzip")
                cir_grp.create_dataset("tau", data=tau, compression="gzip")

                # Metadata
                f.attrs["timestamp"] = timestamp
                f.attrs["ue_id"] = ue_id
                f.attrs["serving_cell"] = serving_cell
                f.attrs["ue_position"] = np.array(ue_position)
                f.attrs["snapshot_index"] = self.snapshot_count

                # RSRP values
                rsrp_grp = f.create_group("rsrp")
                for gnb_id, rsrp in rsrp_values.items():
                    rsrp_grp.attrs[f"gnb_{gnb_id}"] = rsrp

            self.snapshot_count += 1
            logger.info(f"Saved CIR snapshot to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save CIR snapshot: {e}")
            return None


def load_simulation(filepath: str) -> Dict[str, Any]:
    """
    Load simulation data from HDF5 file.

    Args:
        filepath: Path to HDF5 file

    Returns:
        Dictionary containing loaded data
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required to load HDF5 files")

    with h5py.File(filepath, "r") as f:
        data = {}

        # CIR data
        if "cir" in f:
            a_real = f["cir"]["a_real"][:]
            if "a_imag" in f["cir"]:
                a_imag = f["cir"]["a_imag"][:]
                data["a"] = a_real + 1j * a_imag
            else:
                data["a"] = a_real
            data["tau"] = f["cir"]["tau"][:]

        # Scene info
        if "scene" in f:
            data["frequency_hz"] = f["scene"].attrs.get("frequency_hz", 3.5e9)
            data["scene_path"] = f["scene"].attrs.get("scene_path", "unknown")

            # TX positions
            if "transmitters" in f["scene"]:
                data["transmitters"] = {}
                for name in f["scene"]["transmitters"]:
                    tx_grp = f["scene"]["transmitters"][name]
                    data["transmitters"][name] = {
                        "position": tx_grp["position"][:] if "position" in tx_grp else None,
                        "orientation": tx_grp["orientation"][:] if "orientation" in tx_grp else None
                    }

            # RX positions
            if "receivers" in f["scene"]:
                data["receivers"] = {}
                for name in f["scene"]["receivers"]:
                    rx_grp = f["scene"]["receivers"][name]
                    data["receivers"][name] = {
                        "position": rx_grp["position"][:] if "position" in rx_grp else None,
                        "velocity": rx_grp["velocity"][:] if "velocity" in rx_grp else None
                    }

        # Metadata
        data["timestamp"] = f.attrs.get("timestamp", 0.0)
        data["max_depth"] = f.attrs.get("max_depth", 5)
        data["snapshot_index"] = f.attrs.get("snapshot_index", 0)

        # RSRP values
        if "rsrp" in f:
            data["rsrp"] = {}
            for key in f["rsrp"].attrs:
                if key.startswith("gnb_"):
                    gnb_id = int(key.replace("gnb_", ""))
                    data["rsrp"][gnb_id] = f["rsrp"].attrs[key]

        # Rendered image
        if "images" in f and "render" in f["images"]:
            data["render"] = f["images"]["render"][:]

    return data


def show_render(filepath: str):
    """
    Display the rendered scene image from an HDF5 file.

    Args:
        filepath: Path to HDF5 file
    """
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required to display images")

    data = load_simulation(filepath)
    if "render" in data:
        plt.figure(figsize=(12, 8))
        plt.imshow(data["render"])
        plt.axis("off")
        plt.title(f"Scene Render - t={data.get('timestamp', 0):.2f}s")
        plt.tight_layout()
        plt.show()
    else:
        print(f"No render image found in {filepath}")


def list_snapshots(directory: str) -> list:
    """
    List all HDF5 snapshot files in a directory.

    Args:
        directory: Directory path

    Returns:
        List of file paths sorted by timestamp
    """
    dir_path = Path(directory)
    files = list(dir_path.glob("*.h5"))
    return sorted(files, key=lambda x: x.stat().st_mtime)


# Test function
if __name__ == "__main__":
    print("HDF5 Saver Module")
    print(f"h5py available: {HDF5_AVAILABLE}")
    print(f"PIL available: {PIL_AVAILABLE}")
    print(f"matplotlib available: {MPL_AVAILABLE}")
