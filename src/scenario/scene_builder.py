"""
Scene Builder for NR Handover Simulation

Loads gNB configuration from CSV and sets up the channel model.
Extracted from the monolithic run_railway_simulation.py.

Author: Claude Code
Date: 2026-02-27
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GnbSectorInfo:
    """gNB sector configuration (scenario-level, richer than channel.GnbConfig)."""
    gnb_id: int
    sector_id: int
    name: str
    position: Tuple[float, float, float]
    azimuth_deg: float
    downtilt_deg: float
    tx_power_dbm: float
    frequency_ghz: float
    bandwidth_mhz: float
    antenna_gain_dbi: float
    hpbw_h_deg: float
    antenna_type: str
    antenna_num_ports: int
    is_lte: bool
    pci: Optional[int] = None

    # CIO Ocn (3GPP TS 38.331 §6.3.2 cellIndividualOffsets) — outgoing offsets
    # FROM this serving cell TO each neighbor PCI within the 2 km radius.
    # Empty dict ⇒ Ocn=0 ⇒ A3 evaluation unaffected. Populated by
    # load_neighbor_table() at sim startup when --neighbor-table is provided.
    cio_table: Dict[int, float] = None  # type: ignore[assignment]

    # Per-cell HO parameters (None = use global default from CLI args)
    a3_offset_db: Optional[float] = None
    hysteresis_db: Optional[float] = None
    ttt_ms: Optional[float] = None
    a2_threshold_dbm: Optional[float] = None
    a2_ttt_ms: Optional[float] = None
    a5_threshold1_dbm: Optional[float] = None
    a5_threshold2_dbm: Optional[float] = None
    a5_ttt_ms: Optional[float] = None
    b1_threshold_dbm: Optional[float] = None
    b1_ttt_ms: Optional[float] = None
    b1_offset_db: Optional[float] = None
    b2_threshold1_dbm: Optional[float] = None
    b2_threshold2_dbm: Optional[float] = None
    b2_ttt_ms: Optional[float] = None
    b2_offset_db: Optional[float] = None
    n310: Optional[int] = None
    n311: Optional[int] = None
    t310_ms: Optional[float] = None
    t304_ms: Optional[float] = None
    t311_ms: Optional[float] = None
    a3_report_interval_ms: Optional[float] = None
    a2_report_interval_ms: Optional[float] = None
    is_hsr_cell: Optional[bool] = None


# Mapping of CSV column name -> Python type for per-cell HO parameters
_HO_PARAM_COLUMNS: Dict[str, type] = {
    'a3_offset_db': float, 'hysteresis_db': float, 'ttt_ms': float,
    'a2_threshold_dbm': float, 'a2_ttt_ms': float,
    'a5_threshold1_dbm': float, 'a5_threshold2_dbm': float, 'a5_ttt_ms': float,
    'b1_threshold_dbm': float, 'b1_ttt_ms': float, 'b1_offset_db': float,
    'b2_threshold1_dbm': float, 'b2_threshold2_dbm': float, 'b2_ttt_ms': float, 'b2_offset_db': float,
    'n310': int, 'n311': int, 't310_ms': float, 't304_ms': float, 't311_ms': float,
    'a3_report_interval_ms': float, 'a2_report_interval_ms': float,
}


def _parse_num_ports(antenna_type: str) -> int:
    """Extract number of TX antenna ports from antenna_type string.

    E.g., '32T2R Railway Linear Cell' -> 32, '2T2R LTE Sector Antenna' -> 2
    """
    if "32T" in antenna_type: return 32
    if "64T" in antenna_type: return 64
    if "4T" in antenna_type: return 4
    if "2T" in antenna_type: return 2
    return 2  # default


def load_gnb_config(
    csv_path: str,
    exclude_lte: bool = False
) -> List[GnbSectorInfo]:
    """
    Load gNB sector configurations from CSV file.

    Returns one entry per sector (all sectors of each gNB are included).

    Args:
        csv_path: Path to gNB CSV file
        exclude_lte: If True, skip LTE eNodeB entries

    Returns:
        List of GnbSectorInfo objects
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Tag LTE entries
    lte_mask = pd.Series(False, index=df.index)
    if 'antenna_type' in df.columns:
        lte_mask = df['antenna_type'].str.contains('LTE', case=False, na=False)

    if exclude_lte and lte_mask.any():
        n_lte = lte_mask.sum()
        df = df[~lte_mask]
        lte_mask = pd.Series(False, index=df.index)
        logger.info(f"Excluded {n_lte} LTE eNodeB entries, {len(df)} NR gNB entries remain")

    # Build sector list (all sectors per gNB)
    sectors: List[GnbSectorInfo] = []

    for idx, row in df.iterrows():
        gnb_id = int(row['gnb_id'])
        sector_id = int(row['sector_id'])

        downtilt_val = row.get('total_downtilt_deg', -1.0)
        if pd.isna(downtilt_val):
            downtilt_val = row.get('downtilt_deg', -1.0)
            if pd.isna(downtilt_val):
                downtilt_val = -1.0

        antenna_type = str(row.get('antenna_type', ''))
        row_is_lte = bool(lte_mask.iloc[idx] if idx in lte_mask.index else False)

        info = GnbSectorInfo(
            gnb_id=gnb_id,
            sector_id=sector_id,
            name=str(row['name']),
            position=(
                float(row.get('x_m', row.get('x', 0))),
                float(row.get('y_m', row.get('y', 0))),
                float(row.get('z_m', row.get('height_m', 30)))
            ),
            azimuth_deg=float(row.get('azimuth_deg', 0)),
            downtilt_deg=float(downtilt_val),
            tx_power_dbm=float(row.get('tx_power_dBm', row.get('tx_power_dbm', 46))),
            frequency_ghz=float(row.get('frequency_GHz', row.get('frequency_ghz', 3.5))),
            bandwidth_mhz=float(row.get('bandwidth_MHz', 20.0)),
            antenna_gain_dbi=float(row.get('antenna_gain_dBi', row.get('antenna_gain_dbi', 23))),
            hpbw_h_deg=float(row.get('hpbw_horizontal_deg', 25)),
            antenna_type=antenna_type,
            antenna_num_ports=_parse_num_ports(antenna_type),
            is_lte=row_is_lte
        )

        # Read PCI: prefer pci_v2 (from bs_source), fallback to pci, then parse name
        import re
        if 'pci_v2' in df.columns and pd.notna(row.get('pci_v2')):
            info.pci = int(float(row['pci_v2']))
        elif 'pci' in df.columns and pd.notna(row.get('pci')):
            info.pci = int(float(row['pci']))
        else:
            m = re.search(r'\[PCI:(\d+)', str(row.get('name', '')))
            if m:
                info.pci = int(m.group(1))

        # Read per-cell HO parameters from CSV (if columns exist)
        for col, cast in _HO_PARAM_COLUMNS.items():
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val):
                    setattr(info, col, cast(val))

        # is_hsr_cell: bool flag (0/1 or True/False)
        if 'is_hsr_cell' in df.columns:
            v = row.get('is_hsr_cell')
            if pd.notna(v):
                info.is_hsr_cell = bool(int(v)) if not isinstance(v, bool) else v

        sectors.append(info)

    # Log summary
    n_nr = sum(1 for g in sectors if not g.is_lte)
    n_lte_loaded = sum(1 for g in sectors if g.is_lte)
    unique_gnbs = len(set(g.gnb_id for g in sectors))
    logger.info(f"Loaded {len(sectors)} sectors ({unique_gnbs} gNBs) from "
                f"{os.path.basename(csv_path)} ({n_nr} NR, {n_lte_loaded} LTE)")

    freq_counts = Counter(g.frequency_ghz for g in sectors)
    for freq, cnt in sorted(freq_counts.items()):
        logger.info(f"  {freq} GHz: {cnt} sectors")

    return sectors


def load_neighbor_table(
    path: str, gnbs: List[GnbSectorInfo]
) -> int:
    """Attach per-cell `cio_table` to each `GnbSectorInfo` from a sidecar CSV.

    Sidecar schema (produced by `script/generate_neighbor_lists.py`):
      serving_gnb_id, serving_pci, neighbor_gnb_id, neighbor_pci,
      distance_m, cio_outgoing_db

    The `cio_table` on each cell is `{neighbor_pci: cio_outgoing_db}` —
    looked up at A3 evaluation time per 3GPP TS 38.331 §6.3.2 Ocn.
    Cells with no row in the sidecar (or `path` is None / missing) get
    `cio_table = {}` ⇒ A3 falls back to Ocn=0 for all neighbors (byte-
    identical to pre-CIO behavior).

    Returns: number of unique `serving_gnb_id` rows attached.
    """
    if not path or not os.path.exists(path):
        for g in gnbs:
            if g.cio_table is None:
                g.cio_table = {}
        if path:
            logger.warning(f"neighbor_table not found at {path}; using Ocn=0 everywhere")
        return 0

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"serving_gnb_id", "neighbor_pci", "cio_outgoing_db"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"neighbor_table {path} missing columns: {sorted(missing)}")

    # Build {serving_gnb_id: {neighbor_pci: ocn_db, ...}}
    table: Dict[int, Dict[int, float]] = {}
    for src, grp in df.groupby("serving_gnb_id"):
        m: Dict[int, float] = {}
        for _, row in grp.iterrows():
            try:
                pci = int(row["neighbor_pci"])
                ocn = float(row["cio_outgoing_db"])
            except (TypeError, ValueError):
                continue
            m[pci] = ocn
        table[int(src)] = m

    n_attached = 0
    for g in gnbs:
        m = table.get(int(g.gnb_id))
        g.cio_table = dict(m) if m is not None else {}
        if m is not None:
            n_attached += 1

    logger.info(
        f"Loaded neighbor_table from {path}: "
        f"{len(df)} pairs, {len(table)} serving cells, {n_attached} GnbSectorInfo rows attached"
    )
    return n_attached


def setup_channel_model(
    gnb_configs: List[GnbSectorInfo],
    channel_model_type: str = "statistical",
    scene_path: Optional[str] = None,
    frequency_hz: float = 3.5e9,
    bandwidth_hz: float = 20e6,
    tx_power_dbm: float = 46.0,
    noise_figure_db: float = 7.0,
    penetration_loss_db: float = 0.0,
    surface_distortion_mean_db: float = 3.0,
    surface_distortion_std_db: float = 4.0,
    rt_num_samples: float = 1e6,
    tx_antenna_config: str = "auto",
    max_path_length_m: float = 1500.0,
):
    """
    Create and configure a ChannelModel, then populate it with gNBs.

    Args:
        gnb_configs: List of GnbSectorInfo from load_gnb_config()
        channel_model_type: "statistical" or "sionna_rt"
        scene_path: Path to Sionna RT scene XML (for RT model)
        frequency_hz: Carrier frequency in Hz
        bandwidth_hz: System bandwidth in Hz
        tx_power_dbm: Default TX power in dBm
        noise_figure_db: UE noise figure in dB
        penetration_loss_db: Vehicle penetration loss (dB)
        surface_distortion_mean_db: Mean surface distortion (dB)
        surface_distortion_std_db: Std of surface distortion (dB)
        rt_num_samples: Ray samples per source (RT model only)
        tx_antenna_config: Antenna config name (e.g., "auto", "32T2R")

    Returns:
        Configured ChannelModel instance with gNBs loaded
    """
    from channel import ChannelModelFactory, ChannelConfig, ChannelModelType, MultiFreqChannelModel

    model_type = (ChannelModelType.SIONNA_RT if channel_model_type == "sionna_rt"
                  else ChannelModelType.STATISTICAL)

    config = ChannelConfig(
        model_type=model_type,
        frequency_hz=frequency_hz,
        bandwidth_hz=bandwidth_hz,
        tx_power_dbm=tx_power_dbm,
        noise_figure_db=noise_figure_db,
        penetration_loss_db=penetration_loss_db,
        surface_distortion_mean_db=surface_distortion_mean_db,
        surface_distortion_std_db=surface_distortion_std_db,
        scene_path=scene_path,
        num_samples=rt_num_samples,
        tx_antenna_config=tx_antenna_config,
        max_path_length_m=max_path_length_m,
        scenario="RMa"  # Railway -> Rural Macro
    )

    # Detect multiple frequency bands -> use MultiFreqChannelModel
    bands = set(round(g.frequency_ghz, 1) for g in gnb_configs)
    if len(bands) > 1 and model_type == ChannelModelType.STATISTICAL:
        channel_model = MultiFreqChannelModel()
        channel_model.configure(config)
        logger.info(f"Using MultiFreqChannelModel for {len(bands)} bands: {sorted(bands)} GHz")
    else:
        channel_model = ChannelModelFactory.create(config)

    # Add gNBs to the model
    # For RT model with pool: use add_gnb_to_pool for lazy activation
    use_pool = hasattr(channel_model, 'add_gnb_to_pool')

    for gnb in gnb_configs:
        kwargs = dict(
            gnb_id=gnb.gnb_id,
            position=gnb.position,
            sector_id=gnb.sector_id,
            azimuth_deg=gnb.azimuth_deg,
            downtilt_deg=gnb.downtilt_deg,
            tx_power_dbm=gnb.tx_power_dbm,
            antenna_gain_dbi=gnb.antenna_gain_dbi,
            antenna_num_ports=gnb.antenna_num_ports,
            hpbw_h_deg=gnb.hpbw_h_deg,
            name=gnb.name,
            frequency_ghz=gnb.frequency_ghz,
            bandwidth_mhz=gnb.bandwidth_mhz,
            rat_type='lte' if gnb.is_lte else 'nr',
        )
        if use_pool:
            channel_model.add_gnb_to_pool(**kwargs)
        else:
            channel_model.add_gnb(**kwargs)

    if use_pool:
        logger.info(f"Added {len(gnb_configs)} gNBs to RT model pool")
    else:
        logger.info(f"Added {len(gnb_configs)} gNBs to channel model")
        if hasattr(channel_model, 'band_count'):
            logger.info(f"  Bands: {channel_model.band_count}, Total gNBs: {channel_model.gnb_count}")

    return channel_model
