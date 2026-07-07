"""
UE Mobility Module for NR Handover Simulation

Loads UE trajectory data from CSV and computes velocity vectors.
Extracted from the monolithic run_railway_simulation.py.

Author: Claude Code
Date: 2026-02-27
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_ue_trajectory(
    csv_path: str,
    ue_subset: Optional[List] = None,
    max_duration: Optional[float] = None,
    start_time: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load UE trajectories from CSV file.

    Expected columns: timestamp, ue_id, car_id (optional), x, y, z

    Args:
        csv_path: Path to UE trajectory CSV file
        ue_subset: Optional list of UE IDs to load (None = all)
        max_duration: Optional maximum timestamp in seconds (None = full)
        start_time: Optional minimum timestamp in seconds (None = 0.0)

    Returns:
        Dictionary mapping ue_id -> DataFrame sorted by timestamp.
        Each DataFrame has columns: timestamp, ue_id, car_id, x, y, z
    """
    df = pd.read_csv(csv_path)

    # Filter by UE subset (string-safe: ue_id may be e.g. "00#4")
    if ue_subset:
        df = df[df['ue_id'].astype(str).isin([str(u) for u in ue_subset])]

    # Filter by start time
    if start_time is not None:
        df = df[df['timestamp'] >= start_time]

    # Filter by duration (end time)
    if max_duration is not None:
        df = df[df['timestamp'] <= max_duration]

    # Group by UE — preserve ue_id as string (handles phone-derived strings like "00#4")
    trajectories: Dict[str, pd.DataFrame] = {}
    for ue_id, ue_df in df.groupby('ue_id'):
        trajectories[str(ue_id)] = ue_df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded trajectories for {len(trajectories)} UEs from {csv_path}")
    for ue_id, ue_df in trajectories.items():
        ts_range = ue_df['timestamp'].iloc[-1] - ue_df['timestamp'].iloc[0]
        logger.debug(f"  UE {ue_id}: {len(ue_df)} points, {ts_range:.1f}s")

    return trajectories


def compute_velocity(
    positions: List[Tuple[float, float, float]],
    dt: float
) -> Tuple[float, float, float]:
    """
    Compute velocity vector from two consecutive positions.

    Args:
        positions: List of (x, y, z) positions (at least 2 needed)
        dt: Time delta between positions in seconds

    Returns:
        Velocity vector (vx, vy, vz) in m/s
    """
    if len(positions) < 2 or dt <= 0:
        return (0.0, 0.0, 0.0)

    p1 = positions[-2]
    p2 = positions[-1]

    vx = (p2[0] - p1[0]) / dt
    vy = (p2[1] - p1[1]) / dt
    vz = (p2[2] - p1[2]) / dt

    return (vx, vy, vz)


def compute_velocity_from_df(
    trajectory_df: pd.DataFrame,
    row_index: int
) -> Tuple[float, float, float]:
    """
    Compute velocity vector from trajectory DataFrame at a given row.

    Uses backward difference: v(t) = (pos(t) - pos(t-1)) / dt

    Args:
        trajectory_df: DataFrame with columns timestamp, x, y, z
        row_index: Current row index

    Returns:
        Velocity vector (vx, vy, vz) in m/s
    """
    if row_index <= 0:
        # First point: use forward difference if possible
        if len(trajectory_df) > 1:
            row_index = 1
        else:
            return (0.0, 0.0, 0.0)

    curr = trajectory_df.iloc[row_index]
    prev = trajectory_df.iloc[row_index - 1]

    dt = curr['timestamp'] - prev['timestamp']
    if dt <= 0:
        return (0.0, 0.0, 0.0)

    vx = (curr['x'] - prev['x']) / dt
    vy = (curr['y'] - prev['y']) / dt
    vz = (curr['z'] - prev['z']) / dt

    return (float(vx), float(vy), float(vz))


def get_ue_position_at_time(
    trajectory_df: pd.DataFrame,
    timestamp: float,
    tolerance: float = 0.001
) -> Optional[Tuple[float, float, float]]:
    """
    Get UE position at a specific timestamp.

    Uses nearest timestamp within tolerance.

    Args:
        trajectory_df: DataFrame with columns timestamp, x, y, z
        timestamp: Target timestamp in seconds
        tolerance: Time tolerance for matching in seconds

    Returns:
        (x, y, z) position tuple, or None if no match found
    """
    dt_diff = np.abs(trajectory_df['timestamp'] - timestamp)
    min_idx = dt_diff.idxmin()

    if dt_diff[min_idx] > tolerance:
        return None

    row = trajectory_df.iloc[min_idx] if isinstance(min_idx, int) else trajectory_df.loc[min_idx]
    return (float(row['x']), float(row['y']), float(row['z']))


def get_timestamps(
    trajectories: Dict[str, pd.DataFrame]
) -> List[float]:
    """
    Get sorted unique timestamps across all UE trajectories.

    Args:
        trajectories: Dict from load_ue_trajectory()

    Returns:
        Sorted list of unique timestamps
    """
    all_timestamps = set()
    for ue_df in trajectories.values():
        all_timestamps.update(ue_df['timestamp'].tolist())

    return sorted(all_timestamps)
