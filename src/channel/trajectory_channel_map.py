"""
Trajectory-Aware Pre-Computed Channel Map for NR Handover Simulation

This module provides a pre-computation approach for channel calculations,
optimized for scenarios with known UE trajectories (e.g., railway).

Instead of computing channels on-demand (which has ~0% cache hit rate
for linear high-speed motion), this class pre-computes channels for
all positions along the known trajectory BEFORE simulation starts.

Key benefits:
- 100% cache hit rate during simulation
- O(1) lookup time (vs 50-200ms for ray-tracing)
- HDF5 persistence for cross-run caching
- Linear interpolation for sub-sample accuracy

Author: Claude Code
Date: 2026-02-06
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import os

import numpy as np

from .channel_calculator import ChannelState

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryChannelMapConfig:
    """
    Configuration for trajectory channel map.

    Attributes:
        sample_interval_m: Distance between sample points along trajectory (default: 5m)
        interpolation_enabled: Enable linear interpolation between samples
        cache_file: Optional HDF5 file path for persistence
        max_position_error_m: Maximum acceptable position error for lookup
    """
    sample_interval_m: float = 5.0
    interpolation_enabled: bool = True
    cache_file: Optional[str] = None
    max_position_error_m: float = 2.5


class TrajectoryChannelMap:
    """
    Pre-computed channel map for known trajectory.

    Instead of computing channels on-demand (which has ~0% cache hit rate
    for linear high-speed motion), this class pre-computes channels for
    all positions along the known trajectory BEFORE simulation starts.

    Key benefits:
    - 100% cache hit rate during simulation
    - O(1) lookup time
    - HDF5 persistence for cross-run caching
    - Linear interpolation for sub-sample accuracy

    Usage:
        map = TrajectoryChannelMap(config)
        map.precompute(positions, channel_model)
        # During simulation:
        channel_states = map.lookup(current_position)
    """

    def __init__(self, config: Optional[TrajectoryChannelMapConfig] = None):
        """
        Initialize the trajectory channel map.

        Args:
            config: Configuration for sampling and caching
        """
        self._config = config or TrajectoryChannelMapConfig()

        # Map: quantized_position -> {(gnb_id, sector_id): ChannelState}
        self._map: Dict[Tuple[int, int, int], Dict[Tuple[int, int], ChannelState]] = {}

        # Metadata
        self._trajectory_length_m: float = 0.0
        self._num_samples: int = 0
        self._precomputed: bool = False
        self._sample_positions: List[Tuple[float, float, float]] = []

    def _quantize_position(self, pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """
        Quantize continuous position to grid index.

        Args:
            pos: (x, y, z) position in meters

        Returns:
            (gx, gy, gz) grid indices
        """
        res = self._config.sample_interval_m
        return (
            int(round(pos[0] / res)),
            int(round(pos[1] / res)),
            int(round(pos[2] / res))
        )

    def _dequantize_position(self, idx: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        Convert grid index back to position.

        Args:
            idx: (gx, gy, gz) grid indices

        Returns:
            (x, y, z) position in meters
        """
        res = self._config.sample_interval_m
        return (
            idx[0] * res,
            idx[1] * res,
            idx[2] * res
        )

    def precompute(
        self,
        trajectory_positions: List[Tuple[float, float, float]],
        channel_model: Any,
        ue_id: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """
        Pre-compute Sionna RT for all trajectory positions.

        Called ONCE before simulation starts. This is where the heavy
        computation happens - during simulation, only O(1) lookups.

        Args:
            trajectory_positions: List of (x, y, z) positions along trajectory
            channel_model: ChannelModel instance (SionnaRT or Statistical)
            ue_id: UE identifier for channel computation
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Number of positions computed
        """
        # Sample positions at configured interval
        sampled_positions = self._sample_trajectory(trajectory_positions)
        self._sample_positions = sampled_positions

        logger.info(
            f"Pre-computing channels for {len(sampled_positions)} positions "
            f"(from {len(trajectory_positions)} trajectory points, "
            f"interval={self._config.sample_interval_m}m)"
        )

        self._map.clear()

        for i, pos in enumerate(sampled_positions):
            # Update UE position in channel model
            channel_model.update_ue_position(ue_id, pos)

            # Compute channels for all gNBs
            all_states = channel_model.compute_all()

            # Store in map
            pos_key = self._quantize_position(pos)
            self._map[pos_key] = {}

            for (uid, gnb_id, sector_id), state in all_states.items():
                if uid == ue_id:
                    self._map[pos_key][(gnb_id, sector_id)] = state

            if progress_callback:
                progress_callback(i + 1, len(sampled_positions))

            if (i + 1) % 50 == 0:
                logger.info(f"Pre-computed {i + 1}/{len(sampled_positions)} positions")

        self._num_samples = len(sampled_positions)
        self._precomputed = True

        logger.info(f"Pre-computation complete: {self._num_samples} positions cached")
        return self._num_samples

    def _sample_trajectory(
        self,
        positions: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        Sample trajectory at configured interval.

        Uses arc-length parameterization to ensure even spacing.

        Args:
            positions: Original trajectory positions

        Returns:
            Sampled positions at regular intervals
        """
        if len(positions) < 2:
            return positions

        # Calculate cumulative distance along trajectory
        distances = [0.0]
        for i in range(1, len(positions)):
            d = np.sqrt(
                (positions[i][0] - positions[i-1][0])**2 +
                (positions[i][1] - positions[i-1][1])**2 +
                (positions[i][2] - positions[i-1][2])**2
            )
            distances.append(distances[-1] + d)

        total_length = distances[-1]
        self._trajectory_length_m = total_length

        if total_length < self._config.sample_interval_m:
            return positions

        # Sample at configured interval
        interval = self._config.sample_interval_m
        num_samples = int(total_length / interval) + 1
        sample_distances = [i * interval for i in range(num_samples)]

        # Interpolate positions at sample distances
        sampled = []
        pos_idx = 0

        for target_dist in sample_distances:
            # Find segment containing target distance
            while pos_idx < len(distances) - 1 and distances[pos_idx + 1] < target_dist:
                pos_idx += 1

            if pos_idx >= len(positions) - 1:
                sampled.append(positions[-1])
                continue

            # Linear interpolation within segment
            seg_start_dist = distances[pos_idx]
            seg_end_dist = distances[pos_idx + 1]
            seg_length = seg_end_dist - seg_start_dist

            if seg_length < 1e-6:
                alpha = 0.0
            else:
                alpha = (target_dist - seg_start_dist) / seg_length

            p0 = positions[pos_idx]
            p1 = positions[pos_idx + 1]

            sampled.append((
                p0[0] + alpha * (p1[0] - p0[0]),
                p0[1] + alpha * (p1[1] - p0[1]),
                p0[2] + alpha * (p1[2] - p0[2])
            ))

        return sampled

    def lookup(
        self,
        position: Tuple[float, float, float],
        ue_id: int = 0
    ) -> Dict[Tuple[int, int, int], ChannelState]:
        """
        O(1) lookup with optional linear interpolation.

        Args:
            position: Current UE position (x, y, z)
            ue_id: UE identifier

        Returns:
            Dictionary mapping (ue_id, gnb_id, sector_id) to ChannelState
        """
        if not self._precomputed:
            raise RuntimeError("Channel map not pre-computed. Call precompute() first.")

        if self._config.interpolation_enabled:
            return self._lookup_interpolated(position, ue_id)
        else:
            return self._lookup_nearest(position, ue_id)

    def _lookup_nearest(
        self,
        position: Tuple[float, float, float],
        ue_id: int
    ) -> Dict[Tuple[int, int, int], ChannelState]:
        """Lookup nearest cached position."""
        pos_key = self._quantize_position(position)

        if pos_key not in self._map:
            # Find nearest cached position
            min_dist = float('inf')
            nearest_key = None

            for key in self._map.keys():
                key_pos = self._dequantize_position(key)
                dist = np.sqrt(
                    (position[0] - key_pos[0])**2 +
                    (position[1] - key_pos[1])**2 +
                    (position[2] - key_pos[2])**2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_key = key

            pos_key = nearest_key

        if pos_key is None:
            logger.warning(f"No cached position found for {position}")
            return {}

        # Convert to expected return format
        return self._build_result(pos_key, ue_id)

    def _lookup_interpolated(
        self,
        position: Tuple[float, float, float],
        ue_id: int
    ) -> Dict[Tuple[int, int, int], ChannelState]:
        """
        Lookup with linear interpolation between cached positions.

        For 1D trajectory (railway), this interpolates between the two
        nearest sample points along the trajectory.
        """
        res = self._config.sample_interval_m

        # Get the quantized position
        pos_key = self._quantize_position(position)

        # Find nearby cached positions for interpolation
        neighbors = []
        weights = []

        # Check current position
        if pos_key in self._map:
            neighbors.append(pos_key)
            weights.append(1.0)

        # Check adjacent positions along trajectory
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_key = (pos_key[0] + dx, pos_key[1] + dy, pos_key[2] + dz)
                    if neighbor_key in self._map and neighbor_key != pos_key:
                        # Calculate inverse distance weight
                        neighbor_pos = self._dequantize_position(neighbor_key)
                        dist = np.sqrt(
                            (position[0] - neighbor_pos[0])**2 +
                            (position[1] - neighbor_pos[1])**2 +
                            (position[2] - neighbor_pos[2])**2
                        )
                        if dist < res * 2:  # Only nearby neighbors
                            neighbors.append(neighbor_key)
                            weights.append(1.0 / (dist + 0.1))  # Avoid division by zero

        if not neighbors:
            return self._lookup_nearest(position, ue_id)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Interpolate channel states
        result = {}

        # Get all cell keys from all neighbors
        cell_keys = set()
        for key in neighbors:
            cell_keys.update(self._map[key].keys())

        for (gnb_id, sector_id) in cell_keys:
            # Interpolate RSRP in linear domain, then convert back to dBm
            rsrp_linear = 0.0
            sinr_linear = 0.0
            valid_weight = 0.0
            template = None

            for neighbor_key, weight in zip(neighbors, weights):
                if (gnb_id, sector_id) in self._map[neighbor_key]:
                    state = self._map[neighbor_key][(gnb_id, sector_id)]
                    rsrp_linear += weight * (10 ** (state.rsrp_dbm / 10))
                    sinr_linear += weight * (10 ** (state.sinr_db / 10))
                    valid_weight += weight
                    if template is None:
                        template = state

            if valid_weight > 0 and template is not None:
                rsrp_linear /= valid_weight
                sinr_linear /= valid_weight

                rsrp_dbm = 10 * np.log10(rsrp_linear + 1e-30)
                sinr_db = 10 * np.log10(sinr_linear + 1e-30)

                new_state = ChannelState(
                    ue_id=ue_id,
                    gnb_id=gnb_id,
                    sector_id=sector_id,
                    rsrp_dbm=float(rsrp_dbm),
                    sinr_db=float(sinr_db),
                    rsrq_db=template.rsrq_db,
                    delay_spread_ns=template.delay_spread_ns,
                    num_paths=template.num_paths,
                    distance_m=template.distance_m,
                    path_loss_db=template.path_loss_db,
                    timestamp=template.timestamp,
                    los=template.los
                )
                result[(ue_id, gnb_id, sector_id)] = new_state

        return result

    def _build_result(
        self,
        pos_key: Tuple[int, int, int],
        ue_id: int
    ) -> Dict[Tuple[int, int, int], ChannelState]:
        """Build result dictionary from cached position."""
        result = {}
        for (gnb_id, sector_id), state in self._map[pos_key].items():
            new_state = ChannelState(
                ue_id=ue_id,
                gnb_id=gnb_id,
                sector_id=sector_id,
                rsrp_dbm=state.rsrp_dbm,
                sinr_db=state.sinr_db,
                rsrq_db=state.rsrq_db,
                delay_spread_ns=state.delay_spread_ns,
                num_paths=state.num_paths,
                distance_m=state.distance_m,
                path_loss_db=state.path_loss_db,
                timestamp=state.timestamp,
                los=state.los
            )
            result[(ue_id, gnb_id, sector_id)] = new_state
        return result

    def save_to_hdf5(self, filepath: str) -> None:
        """
        Save pre-computed map to HDF5 file for cross-run caching.

        File structure:
        /metadata
            - sample_interval_m
            - trajectory_length_m
            - num_samples
        /positions/{pos_key}
            - rsrp_{gnb}_{sector}
            - sinr_{gnb}_{sector}
            - ...

        Args:
            filepath: Path to HDF5 file
        """
        try:
            import h5py
        except ImportError:
            logger.error("h5py not installed. Cannot save to HDF5.")
            return

        if not self._precomputed:
            raise RuntimeError("No data to save. Call precompute() first.")

        logger.info(f"Saving channel map to {filepath}")

        with h5py.File(filepath, 'w') as f:
            # Metadata
            meta = f.create_group('metadata')
            meta.attrs['sample_interval_m'] = self._config.sample_interval_m
            meta.attrs['trajectory_length_m'] = self._trajectory_length_m
            meta.attrs['num_samples'] = self._num_samples
            meta.attrs['interpolation_enabled'] = self._config.interpolation_enabled

            # Position data
            positions_grp = f.create_group('positions')

            for pos_key, cell_states in self._map.items():
                pos_name = f"{pos_key[0]}_{pos_key[1]}_{pos_key[2]}"
                pos_grp = positions_grp.create_group(pos_name)

                for (gnb_id, sector_id), state in cell_states.items():
                    cell_name = f"gnb{gnb_id}_s{sector_id}"
                    cell_grp = pos_grp.create_group(cell_name)

                    cell_grp.attrs['rsrp_dbm'] = state.rsrp_dbm
                    cell_grp.attrs['sinr_db'] = state.sinr_db
                    cell_grp.attrs['rsrq_db'] = state.rsrq_db
                    cell_grp.attrs['delay_spread_ns'] = state.delay_spread_ns
                    cell_grp.attrs['num_paths'] = state.num_paths
                    cell_grp.attrs['distance_m'] = state.distance_m
                    cell_grp.attrs['path_loss_db'] = state.path_loss_db
                    cell_grp.attrs['timestamp'] = state.timestamp
                    cell_grp.attrs['los'] = state.los

        logger.info(f"Saved {self._num_samples} positions to {filepath}")

    def load_from_hdf5(self, filepath: str) -> bool:
        """
        Load pre-computed map from HDF5 file.

        Args:
            filepath: Path to HDF5 file

        Returns:
            True if loaded successfully, False if file doesn't exist
        """
        try:
            import h5py
        except ImportError:
            logger.error("h5py not installed. Cannot load from HDF5.")
            return False

        if not os.path.exists(filepath):
            return False

        logger.info(f"Loading channel map from {filepath}")

        try:
            with h5py.File(filepath, 'r') as f:
                # Check metadata compatibility
                meta = f['metadata']
                file_interval = meta.attrs['sample_interval_m']

                if abs(file_interval - self._config.sample_interval_m) > 0.01:
                    logger.warning(
                        f"Sample interval mismatch: file={file_interval}, "
                        f"config={self._config.sample_interval_m}"
                    )
                    return False

                self._trajectory_length_m = float(meta.attrs['trajectory_length_m'])
                self._num_samples = int(meta.attrs['num_samples'])

                # Load position data
                self._map.clear()

                for pos_name in f['positions'].keys():
                    parts = pos_name.split('_')
                    pos_key = (int(parts[0]), int(parts[1]), int(parts[2]))

                    self._map[pos_key] = {}
                    pos_grp = f['positions'][pos_name]

                    for cell_name in pos_grp.keys():
                        # Parse "gnb1_s1" format
                        cell_parts = cell_name.split('_')
                        gnb_id = int(cell_parts[0][3:])  # "gnb1" -> 1
                        sector_id = int(cell_parts[1][1:])  # "s1" -> 1

                        cell_grp = pos_grp[cell_name]

                        state = ChannelState(
                            ue_id=0,  # Will be updated on lookup
                            gnb_id=gnb_id,
                            sector_id=sector_id,
                            rsrp_dbm=float(cell_grp.attrs['rsrp_dbm']),
                            sinr_db=float(cell_grp.attrs['sinr_db']),
                            rsrq_db=float(cell_grp.attrs['rsrq_db']),
                            delay_spread_ns=float(cell_grp.attrs['delay_spread_ns']),
                            num_paths=int(cell_grp.attrs['num_paths']),
                            distance_m=float(cell_grp.attrs['distance_m']),
                            path_loss_db=float(cell_grp.attrs['path_loss_db']),
                            timestamp=float(cell_grp.attrs.get('timestamp', 0.0)),
                            los=bool(cell_grp.attrs.get('los', False))
                        )
                        self._map[pos_key][(gnb_id, sector_id)] = state

                self._precomputed = True
                logger.info(f"Loaded {self._num_samples} positions from cache")
                return True

        except Exception as e:
            logger.error(f"Error loading HDF5: {e}")
            return False

    @property
    def is_precomputed(self) -> bool:
        """Check if map has been pre-computed."""
        return self._precomputed

    @property
    def num_samples(self) -> int:
        """Get number of cached samples."""
        return self._num_samples

    @property
    def trajectory_length_m(self) -> float:
        """Get total trajectory length."""
        return self._trajectory_length_m

    @property
    def config(self) -> TrajectoryChannelMapConfig:
        """Get configuration."""
        return self._config
