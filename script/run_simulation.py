#!/usr/bin/env python3
"""
NR Handover Simulation Orchestrator

Clean simulation orchestrator that uses ONLY UEStateMachine for RRC state management.
Supports both statistical and Sionna ray-tracing channel models.

Architecture:
1. Parse CLI args + load config
2. Setup scene (load_gnb_config + setup_channel_model)
3. Load UE trajectories
4. Create UEStateMachine per UE
5. Simulation loop (40ms steps):
   - Update UE positions + velocities
   - channel_model.compute_all(timestamp) -> Dict[(ue_id, gnb_id, sector_id), ChannelState]
   - For each UE: extract RSRP measurements, call state_machine.update(), log events
6. Generate report

Usage:
    python script/run_simulation.py --channel-model statistical --duration 10 --ue-subset 1
    python script/run_simulation.py --channel-model sionna_rt --scene-path railway_scene.xml
"""

import sys
import os
import math
import logging
import argparse
import time as time_module
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import Counter

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from scenario import load_gnb_config, setup_channel_model
from scenario.ue_mobility import load_ue_trajectory, compute_velocity_from_df, get_timestamps
from scenario.scene_builder import GnbSectorInfo
from rrc.rrc_controller import UEStateMachine, UEStateMachineConfig
from rrc.rrc_types import RRCState, HOFType
from channel import ChannelState, ChannelConfig, ChannelModelType
from channel.post_processing import default_bandwidth_mhz, default_scs_hz
from phy.phy_abstraction import PHYAbstraction


def _band_label(freq_ghz) -> str:
    """Map a carrier frequency (GHz) to the band label used by the dynamic
    per-band RLM Qin/Qout table (and the harness band naming)."""
    try:
        f = round(float(freq_ghz), 1)
    except (TypeError, ValueError):
        return "unknown"
    return {0.9: "900M", 1.8: "1.8G", 2.1: "2.1G", 3.5: "3.5G"}.get(f, f"{f}G")


@dataclass
class SimulationConfig:
    """Configuration for NR handover simulation."""
    # Data files
    gnb_csv: str = "enb_coordinates_converted_with_z13.csv"
    ue_csv: str = "ktx_ue_coordinates.csv"

    # Simulation parameters
    measurement_period_ms: float = 40  # 40ms measurement period
    duration_s: Optional[float] = None  # None = full trajectory
    start_time_s: Optional[float] = None  # None = 0.0 (beginning)
    ue_subset: Optional[List[int]] = None
    mcs_index: int = 9  # MCS 9: QPSK highest code rate (679/1024), SINR_10%=10.4 dB

    # RF parameters
    tx_power_dbm: float = 46.0
    frequency_ghz: float = 3.5
    bandwidth_mhz: float = 20.0
    penetration_loss_db: float = 25.0  # KTX train vehicle penetration loss (3GPP TR 38.901)
    surface_distortion_mean_db: float = 3.0
    surface_distortion_std_db: float = 4.0

    # Channel model
    channel_model_type: str = "statistical"  # "statistical" or "sionna_rt"
    scene_path: Optional[str] = None
    rt_max_gnbs: int = 60
    rt_num_samples: float = 1e6
    max_path_length_m: float = 1500.0  # Discard RT paths longer than this

    # RRC parameters
    a3_offset_db: float = 3.0
    hysteresis_db: float = 2.0
    a2_hysteresis_db: float = 3.0       # A2-specific hysteresis (3GPP TS 38.331 §5.5.4.2)
    ttt_ms: float = 40.0
    filter_coef: int = 4
    # CSV channel mode L3 passthrough. ⚠️ 2026-06-09 ops-confirmed: the wide
    # CSV is RAW L1, NOT field-L3-filtered. Operational UE applies fc4 (k=4,
    # α=0.5) before A3/RLM; that output is NOT logged. Faithful config =
    # passthrough False + filter_coef 4 (apply fc4). Default False = legacy.
    l3_filter_passthrough: bool = False
    a2_threshold_dbm: float = -118.0
    a2_ttt_ms: float = 100.0
    # reportInterval (TS 38.331 §5.5.5). LAST-RESORT FALLBACK ONLY: the
    # operational per-cell value (enb CSV 240/120) is read CSV-first via
    # _g(gnb,'a3_report_interval_ms', ...) in _build_cell_configs (line ~329),
    # which overrides this base. Verified runtime gate = 240ms, not 480.
    a3_report_interval_ms: float = 480.0
    a2_report_interval_ms: float = 1024.0
    a5_threshold1_dbm: float = -125.0
    a5_threshold2_dbm: float = -115.0
    a5_ttt_ms: float = 256.0

    # Inter-RAT (B1/B2) parameters
    b1_threshold_dbm: float = -125.0
    b1_ttt_ms: float = 256.0
    b1_offset_db: float = 0.0
    b2_threshold1_dbm: float = -130.0
    b2_threshold2_dbm: float = -125.0
    b2_ttt_ms: float = 256.0
    b2_offset_db: float = 0.0

    # RLF parameters
    n310: int = 10
    # N311 default = 2: HST research-based (see rrc_controller.UEStateMachineConfig).
    n311: int = 2
    t310_ms: float = 300.0
    t304_ms: float = 200.0
    t311_ms: float = 1000.0
    t300_ms: float = 1000.0  # RRC Connection Setup guard timer (ms)
    t301_ms: float = 400.0   # RRC Re-establishment guard timer (ms) (3GPP TS 38.331 §5.3.7)

    # RACH parameters
    preamble_initial_power_dbm: float = -104.0
    preamble_tx_max: int = 10
    power_ramping_step_db: float = 2.0

    # RLF thresholds (RSRP-based, used when use_sinr_for_rlf=False)
    qout_rsrp: float = -150.0  # Out-of-sync RSRP threshold (dBm, per-RE RSRP)
    qin_rsrp: float = -140.0   # In-sync RSRP threshold (dBm, per-RE RSRP)

    # SINR mode
    use_sinr_for_rlf: bool = True
    use_ul_sinr_for_rach: bool = True
    rlf_mcs_index: int = -1  # -1=AMC (adaptive), 0-28=fixed MCS for Qout/Qin (3GPP TS 38.133 §8.1)
    # Hypothetical-PDCCH BLER gates for SINR-mode RLM (3GPP TS 38.133 §8.1.2):
    # OOS when BLER > qout_bler, IS when BLER < qin_bler.
    qout_bler: float = 0.10
    qin_bler: float = 0.02
    # Dynamic per-band / per-speed RLM Qin/Qout SPARSE override table (2026-06-12):
    #   {band("900M"/"1.8G"/"2.1G"): {"<speed_kmh>": {qout_al, qin_al,
    #    qout_boost_db, qin_boost_db}}}. The serving cell's band + the live UE
    #   speed-bucket (nearest of the configured speeds) select Qout/Qin; any
    #   (band, speed) not present falls back to the global rlm_* fit (~240 km/h).
    #   Default empty ⇒ legacy single-fit behaviour. Loaded from
    #   bo_config.json["rlm_qin_qout_by_band_speed"] in run_csv_simulation.
    rlm_qin_qout_by_band_speed: dict = field(default_factory=dict)
    # Stochastic gNB HO-decision delay parameters (vendor-side, not 3GPP).
    # std==0 → fixed delay equal to gnb_ho_decision_delay_ms (back-compat).
    gnb_ho_decision_delay_std_ms: float = 0.0
    gnb_ho_decision_delay_adapt_to_rf: bool = False

    # Nearby activation
    activation_radius_m: float = 2000.0  # Only compute cells within this radius of UE center

    # Precomputed SINR maps (from precompute_sinr_map.py)
    sinr_map_dir: Optional[str] = None  # e.g. "output/sinr_maps"

    # BLER mode
    bler_mode: str = "full_chain"  # "full_chain", "mockup", "sigmoid"
    bler_csv_dir: str = "output/bler_curves"
    bler_sinr_offset: float = 0.0

    # Output
    output_dir: str = "output/railway_sim"
    log_level: str = "INFO"
    exclude_lte: bool = False

    # Verbose CLI output
    verbose: bool = True
    verbose_interval: int = 1  # Print status every N timesteps

    # RSRQ-based A3 trigger for regular (non-HSR) cells.
    # Real operators evaluate A3 entering on RSRQ for non-HSR cells while
    # HSR cells stay on RSRP. Default ON (2026-06-09, user): every entry point
    # already enabled it; making it the default prevents a direct run from
    # silently falling back to RSRP-only A3.
    use_rsrq_for_regular_cells: bool = True

    # Vendor gNB HO Decision Algorithm delay (ms). 3GPP TS 38.331 §5.5
    # only standardises UE-side measurement reports — the actual HO
    # command (RRCReconfiguration with reconfigurationWithSync) is gNB
    # implementation. Real operator gNBs buffer A3/A5/B1/B2 reports and
    # re-validate after this delay before issuing the command. During
    # the window, if the target is no longer the strongest non-serving
    # cell (by ≥ 0.5 dB margin), the report is withdrawn (no HO).
    # 0.0 = legacy byte-identical behaviour (immediate fire on report).
    gnb_ho_decision_delay_ms: float = 0.0
    # Vendor gNB MRO post-HO blacklist (seconds). After HO source→target
    # completes, A3 entering condition for cell `source` is ignored while
    # serving=target for `post_ho_blacklist_s` seconds. 0.0 = disabled.
    post_ho_blacklist_s: float = 0.0


@dataclass
class SimulationEvent:
    """Record of a simulation event."""
    timestamp: float
    ue_id: int
    event_type: str  # HO_START, HO_COMPLETE, HO_FAIL, RLF, etc.
    source_cell: Optional[int] = None
    target_cell: Optional[int] = None
    rsrp_dbm: Optional[float] = None
    sinr_db: Optional[float] = None
    details: Optional[str] = None


@dataclass
class UEStepData:
    """Per-step UE data for verbose logging."""
    step: int
    timestamp: float
    ue_id: int
    position: Tuple[float, float, float]
    serving_cell: int
    rsrp_dbm: float
    sinr_db: float
    bler: float
    mcs: int
    status: str  # "OK", "RLF!", "HO->123", "T310 running", etc.
    n310: int
    n311: int
    timers: List[str]
    all_rsrp: Dict[int, float]
    # Detailed state for per-step log (dump-style)
    all_sinr: Dict[int, float] = field(default_factory=dict)
    # Per-cell RSRQ (dB) — populated when wide CSV provides pci_<P>_rsrq.
    # Used for RSRQ-A3 evaluation on regular (non-HSR) cells and surfaces in
    # detailed_log_ue<N>.csv top<k>_rsrq / serving_rsrq.
    all_rsrq: Dict[int, float] = field(default_factory=dict)
    rrc_state: str = ""
    ho_in_progress: bool = False
    rlf_declared: bool = False
    target_cell: Optional[int] = None
    meas_snapshot: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timer_snapshot: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    counter_snapshot: Dict[str, Dict[str, int]] = field(default_factory=dict)
    events_this_step: List[str] = field(default_factory=list)
    radio_link_status: str = "IS"  # Actual RLF sync status from state machine (IS/OOS/Gray)
    sinr_base_db: float = -30.0  # SINR from RSRP-ratio (before Doppler)
    doppler_penalty_db: float = 0.0  # Doppler penalty from SINR map (0.0 = N/A)
    # --- A3 measurement diagnostics (from MeasurementEngine.a3_diagnostics_snapshot) ---
    # Populated each step from the internal A3 evaluation state so downstream
    # analysis can see what the engine actually compared (filtered values,
    # candidate ages, per-cell TTT elapsed, entering verdict).
    a3_serv_filt_rsrq: Optional[float] = None
    a3_serv_filt_rsrp: Optional[float] = None
    a3_quantity: str = ""
    a3_filter_age_ms: Optional[float] = None
    a3_top1_cell: Optional[int] = None
    a3_top1_metric: Optional[float] = None
    a3_top1_ttt_ms: Optional[float] = None
    a3_top1_age: Optional[int] = None
    a3_top1_entering: Optional[bool] = None
    a3_top2_cell: Optional[int] = None
    a3_top2_metric: Optional[float] = None
    a3_top2_ttt_ms: Optional[float] = None
    a3_top2_age: Optional[int] = None
    a3_top2_entering: Optional[bool] = None
    a3_top3_cell: Optional[int] = None
    a3_top3_metric: Optional[float] = None
    a3_top3_ttt_ms: Optional[float] = None
    a3_top3_age: Optional[int] = None
    a3_top3_entering: Optional[bool] = None
    a3_tracker_n: int = 0
    # --- Gate-reason diagnostics (T1, 2026-06-11; observability-only) ---
    # ho_suppress_reason: why a viable HO was NOT triggered this tick
    #   ("" / UL_BLOCK_PATHA / UL_BLOCK_PATHB / SIB_BLOCK / HO_IN_PROGRESS).
    # report_block_reason / report_gap_ms: measurement-layer reportInterval
    #   suppression (REPORT_INTERVAL + gap-since-last-A3-report ms).
    # These are read-outs of decisions the FSM already makes — they never gate.
    ho_suppress_reason: str = ""
    report_block_reason: str = ""
    report_gap_ms: Optional[float] = None
    # --- Explicit staged HO signaling flow (2026-06-11) ---
    # Per-tick read-out of the HO chain stage + per-attempt gate results,
    # read from UEState. Observability for the S1→S2→S3→S4 flow; the only
    # FSM-affecting member is S4 (target-DL RRC-config delivery gate).
    ho_stage: str = ""
    ho_cmd_decoded: bool = False
    target_rach_ok: bool = False
    target_rrc_ok: bool = False
    # --- UL-block / RLM / velocity / L3-filter read-outs (2026-06-12) ---
    # ul_block_applied_rsrq: the L3-FILTERED serving RSRQ the gate compared
    #   (NOT the raw `serving_rsrq` column) — explains blocks at healthy raw RSRQ.
    # ul_block_threshold_rsrq_db: path-B threshold (default -19.3).
    # rlm_*: smoothed serving SINR + Qout/Qin hypothetical-PDCCH BLER that
    #   decide OUT_OF_SYNC/IN_SYNC (which arm/clear T310).
    # ue_velocity_kmh: UE ground speed; l3_filter_k: TS 38.331 §5.5.3.2 k
    #   (alpha = 1/2**(k/4); k=4 -> alpha 0.5) for L3-period-vs-speed checks.
    ul_block_applied_rsrq: Optional[float] = None
    ul_block_applied_rsrp: Optional[float] = None
    ul_block_threshold_rsrq_db: Optional[float] = None
    ul_block_path: str = ""
    rlm_smoothed_sinr_db: Optional[float] = None
    rlm_bler_qout: Optional[float] = None
    rlm_bler_qin: Optional[float] = None
    ue_velocity_kmh: Optional[float] = None
    l3_filter_k: Optional[int] = None


class NRSimulation:
    """Main simulation class for NR handover."""

    def __init__(self, config: SimulationConfig):
        """Construct an NR/LTE handover simulation harness.

        Args:
            config: `SimulationConfig` dataclass holding all input paths,
                channel-model selection, duration, output paths, and
                per-UE/per-cell HO parameters.

        Side effects:
            - Allocates `self.events` list (one SimulationEvent per HO_START /
              HO_COMPLETE / RLF / etc emission).
            - Instantiates `self.phy = PHYAbstraction()` (MCS table, BLER).
            - `self.step_data` accumulates per-tick rows for detailed_log
              CSV emission when verbose mode is on.
        """
        self.config = config
        self.events: List[SimulationEvent] = []
        self.logger = logging.getLogger("NRSimulation")
        self.phy = PHYAbstraction()

        self.step_data: List[UEStepData] = []  # Store step data for verbose logging
        self._verbose_header_printed = False

    # _default_scs_hz / _default_bandwidth_mhz are imported from
    # channel.post_processing (single source of truth).
    _default_scs_hz = staticmethod(default_scs_hz)
    _default_bandwidth_mhz = staticmethod(default_bandwidth_mhz)

    def _build_cell_configs(
        self, gnb_configs: List[GnbSectorInfo], sm_config: UEStateMachineConfig
    ) -> Dict[int, UEStateMachineConfig]:
        """Build per-cell UEStateMachineConfig from GnbSectorInfo HO fields.

        For each HO parameter, uses the cell-specific value if set (not None),
        otherwise falls back to the global sm_config value.
        Non-HO parameters (RACH, RLF thresholds, SINR mode) always come from sm_config.
        """
        cell_configs: Dict[int, UEStateMachineConfig] = {}

        def _g(gnb, attr, default):
            """Get per-cell HO value if set, else fall back to global default."""
            val = getattr(gnb, attr, None)
            return val if val is not None else default

        # HSR S-Measure threshold (dBm). When set, HSR cells gate A3 above
        # this RSRP. Currently DISABLED (set to None) per user request 2026-05-14.
        # To re-enable: change to -97.0 (or operator-profile value).
        _HSR_S_MEASURE_DBM = None
        for gnb in gnb_configs:
            _hsr = bool(getattr(gnb, "is_hsr_cell", False) or False)
            cell_configs[gnb.gnb_id] = UEStateMachineConfig(
                # HO measurement parameters (per-cell overrides)
                a3_offset_db=_g(gnb, 'a3_offset_db', sm_config.a3_offset_db),
                hysteresis_db=_g(gnb, 'hysteresis_db', sm_config.hysteresis_db),
                ttt_ms=_g(gnb, 'ttt_ms', sm_config.ttt_ms),
                a2_threshold_dbm=_g(gnb, 'a2_threshold_dbm', sm_config.a2_threshold_dbm),
                a2_ttt_ms=_g(gnb, 'a2_ttt_ms', sm_config.a2_ttt_ms),
                a5_threshold1_dbm=_g(gnb, 'a5_threshold1_dbm', sm_config.a5_threshold1_dbm),
                a5_threshold2_dbm=_g(gnb, 'a5_threshold2_dbm', sm_config.a5_threshold2_dbm),
                a5_ttt_ms=_g(gnb, 'a5_ttt_ms', sm_config.a5_ttt_ms),
                b1_threshold_dbm=_g(gnb, 'b1_threshold_dbm', sm_config.b1_threshold_dbm),
                b1_ttt_ms=_g(gnb, 'b1_ttt_ms', sm_config.b1_ttt_ms),
                b1_offset_db=_g(gnb, 'b1_offset_db', sm_config.b1_offset_db),
                b2_threshold1_dbm=_g(gnb, 'b2_threshold1_dbm', sm_config.b2_threshold1_dbm),
                b2_threshold2_dbm=_g(gnb, 'b2_threshold2_dbm', sm_config.b2_threshold2_dbm),
                b2_ttt_ms=_g(gnb, 'b2_ttt_ms', sm_config.b2_ttt_ms),
                b2_offset_db=_g(gnb, 'b2_offset_db', sm_config.b2_offset_db),
                n310=_g(gnb, 'n310', sm_config.n310),
                n311=_g(gnb, 'n311', sm_config.n311),
                t310_ms=_g(gnb, 't310_ms', sm_config.t310_ms),
                t304_ms=_g(gnb, 't304_ms', sm_config.t304_ms),
                t311_ms=_g(gnb, 't311_ms', sm_config.t311_ms),
                a3_report_interval_ms=_g(gnb, 'a3_report_interval_ms', sm_config.a3_report_interval_ms),
                a2_report_interval_ms=_g(gnb, 'a2_report_interval_ms', sm_config.a2_report_interval_ms),
                # CIO Ocn (3GPP TS 38.331 §6.3.2): per-(serving, neighbor) offset map.
                # None / empty => Ocn=0 in A3 evaluation (byte-identical to pre-CIO).
                cio_table=getattr(gnb, 'cio_table', None) or {},
                # S-Measure: HSR cells gate A3 above -97 dBm; regular cells off.
                s_measure_dbm=(_HSR_S_MEASURE_DBM if _hsr else None),
                # Non-HO parameters (always global)
                filter_coef=sm_config.filter_coef,
                l3_filter_passthrough=getattr(sm_config, "l3_filter_passthrough", False),
                preamble_initial_power_dbm=sm_config.preamble_initial_power_dbm,
                preamble_tx_max=sm_config.preamble_tx_max,
                power_ramping_step_db=sm_config.power_ramping_step_db,
                qout_rsrp=sm_config.qout_rsrp,
                qin_rsrp=sm_config.qin_rsrp,
                use_sinr_for_rlf=sm_config.use_sinr_for_rlf,
                use_ul_sinr_for_rach=sm_config.use_ul_sinr_for_rach,
                rlf_mcs_index=sm_config.rlf_mcs_index,
                qout_bler=getattr(sm_config, "qout_bler", 0.10),
                qin_bler=getattr(sm_config, "qin_bler", 0.02),
                gnb_ho_decision_delay_std_ms=getattr(sm_config, "gnb_ho_decision_delay_std_ms", 0.0),
                gnb_ho_decision_delay_adapt_to_rf=getattr(sm_config, "gnb_ho_decision_delay_adapt_to_rf", False),
                gnb_ho_decision_delay_ms=getattr(sm_config, "gnb_ho_decision_delay_ms", 0.0),
                post_ho_blacklist_s=getattr(sm_config, "post_ho_blacklist_s", 0.0),
            )
            # Stamp cell_id so CONFIG APPLY log can show it
            cell_configs[gnb.gnb_id]._cell_id = gnb.gnb_id

        return cell_configs

    def _setup_channel_model_v2(self, gnb_configs):
        """
        ★ v2 bridge: GnbSectorInfo의 per-sector frequency/bandwidth/scs를
        channel_model.add_gnb()에 직접 전달.

        기존 setup_channel_model()은 글로벌 frequency_hz, bandwidth_hz만 전달했음.
        v2는 GnbSectorInfo.frequency_ghz, .bandwidth_mhz를 per-gnb로 전달하고,
        SCS는 주파수 기반 기본값 사용 (CSV에 scs_hz 컬럼 있으면 그것 사용).
        """
        from channel import ChannelConfig, ChannelModelType

        # Determine model type
        if self.config.channel_model_type == "sionna_rt":
            from channel.sionna_rt_model import SionnaRTChannelModel
            channel_model = SionnaRTChannelModel()
        else:
            from channel.statistical_model import StatisticalChannelModel
            channel_model = StatisticalChannelModel()

        # Global config (fallback defaults)
        config = ChannelConfig(
            model_type=ChannelModelType.SIONNA_RT if self.config.channel_model_type == "sionna_rt"
                       else ChannelModelType.STATISTICAL,
            frequency_hz=self.config.frequency_ghz * 1e9,
            bandwidth_hz=self.config.bandwidth_mhz * 1e6,
            tx_power_dbm=self.config.tx_power_dbm,
            penetration_loss_db=self.config.penetration_loss_db,
            surface_distortion_mean_db=self.config.surface_distortion_mean_db,
            surface_distortion_std_db=self.config.surface_distortion_std_db,
            scene_path=self.config.scene_path,
            num_samples=self.config.rt_num_samples,
            max_path_length_m=self.config.max_path_length_m,
        )
        channel_model.configure(config)

        # ★ Per-gnb: add_gnb_to_pool with per-sector RF params
        for g in gnb_configs:
            # GnbSectorInfo에서 per-sector 값 추출
            freq_ghz = getattr(g, 'frequency_ghz', self.config.frequency_ghz)
            bw_mhz = getattr(g, 'bandwidth_mhz', self._default_bandwidth_mhz(freq_ghz))
            scs_hz = getattr(g, 'scs_hz', self._default_scs_hz(freq_ghz))

            channel_model.add_gnb_to_pool(
                gnb_id=g.gnb_id,
                position=g.position,
                sector_id=getattr(g, 'sector_id', 1),
                azimuth_deg=getattr(g, 'azimuth_deg', 0.0),
                downtilt_deg=getattr(g, 'total_downtilt_deg',
                             getattr(g, 'downtilt_deg', -1.0)),
                tx_power_dbm=getattr(g, 'tx_power_dbm', self.config.tx_power_dbm),
                antenna_gain_dbi=getattr(g, 'antenna_gain_dbi', 23.0),
                antenna_num_ports=getattr(g, 'antenna_num_ports', 2),
                name=getattr(g, 'name', f"gNB_{g.gnb_id}"),
                rat_type=getattr(g, 'rat_type', 'nr'),
                # ★ v2: per-gnb RF params
                frequency_ghz=freq_ghz,
                bandwidth_mhz=bw_mhz,
                scs_hz=scs_hz,
            )

        self.logger.info(f"Channel model v2: {len(gnb_configs)} gNBs pooled with per-sector RF params")
        return channel_model

    def run(self):
        """Run the complete simulation."""
        # 1. Load gNB config
        self.logger.info("Loading gNB configuration...")
        gnb_configs = load_gnb_config(self.config.gnb_csv, exclude_lte=self.config.exclude_lte)
        # Build lookup: gnb_id -> GnbSectorInfo
        gnb_lookup = {g.gnb_id: g for g in gnb_configs}
        self._gnb_lookup = gnb_lookup
        self.logger.info(f"Loaded {len(gnb_configs)} gNB sectors")

        # 2. Setup channel model
        self.logger.info(f"Setting up {self.config.channel_model_type} channel model...")
        if self.config.channel_model_type == "sionna_rt":
            # ★ v2: per-gnb frequency/bandwidth/scs를 GnbSectorInfo에서 직접 전달
            channel_model = self._setup_channel_model_v2(gnb_configs)
        else:
            channel_model = setup_channel_model(
                gnb_configs,
                channel_model_type=self.config.channel_model_type,
                scene_path=getattr(self.config, 'scene_path', None),
                frequency_hz=self.config.frequency_ghz * 1e9,
                bandwidth_hz=self.config.bandwidth_mhz * 1e6,
                tx_power_dbm=self.config.tx_power_dbm,
                penetration_loss_db=self.config.penetration_loss_db,
                surface_distortion_mean_db=self.config.surface_distortion_mean_db,
                surface_distortion_std_db=self.config.surface_distortion_std_db,
            )

        # 2b. Load precomputed SINR maps (if available)
        if self.config.sinr_map_dir and hasattr(channel_model, 'load_sinr_maps'):
            n_maps = channel_model.load_sinr_maps(self.config.sinr_map_dir)
            if n_maps > 0:
                self.logger.info(f"Loaded {n_maps} precomputed SINR maps from {self.config.sinr_map_dir}")
            else:
                self.logger.warning(f"No SINR maps found in {self.config.sinr_map_dir}")

        # 2c. Set BLER mode on channel model
        if hasattr(channel_model, 'set_bler_mode'):
            channel_model.set_bler_mode(
                self.config.bler_mode,
                csv_dir=self.config.bler_csv_dir,
                sinr_offset_db=self.config.bler_sinr_offset,
            )

        # 3. Load UE trajectories
        self.logger.info("Loading UE trajectories...")
        trajectories = load_ue_trajectory(
            self.config.ue_csv,
            ue_subset=self.config.ue_subset,
            max_duration=self.config.duration_s,
            start_time=self.config.start_time_s,
        )
        ue_ids = sorted(trajectories.keys())
        timestamps = get_timestamps(trajectories)
        if not timestamps:
            self.logger.error("No timesteps after filtering — nothing to simulate")
            import sys as _sys
            _sys.exit(1)
        self.logger.info(f"Simulation: {len(ue_ids)} UEs, {len(timestamps)} timesteps, "
                         f"{timestamps[-1] - timestamps[0]:.1f}s duration")

        # 4. Create UEStateMachine per UE
        # NB: this `sm_config` is the BASE/initial config. Per-cell HO params
        # (incl. a3/a2_report_interval_ms) are resolved CSV-first per serving
        # cell in _build_cell_configs via _g(gnb,...) and applied through
        # apply_config(); the report_interval values here are last-resort
        # fallbacks only (operational per-cell CSV = 240/120, verified runtime).
        self.logger.info("Creating UE state machines...")
        sm_config = UEStateMachineConfig(
            a3_offset_db=self.config.a3_offset_db,
            a2_threshold_dbm=self.config.a2_threshold_dbm,
            a5_threshold1_dbm=self.config.a5_threshold1_dbm,
            a5_threshold2_dbm=self.config.a5_threshold2_dbm,
            hysteresis_db=self.config.hysteresis_db,
            a2_hysteresis_db=self.config.a2_hysteresis_db,
            ttt_ms=self.config.ttt_ms,
            a2_ttt_ms=self.config.a2_ttt_ms,
            a3_report_interval_ms=self.config.a3_report_interval_ms,
            a2_report_interval_ms=self.config.a2_report_interval_ms,
            a5_ttt_ms=self.config.a5_ttt_ms,
            b1_threshold_dbm=self.config.b1_threshold_dbm,
            b1_ttt_ms=self.config.b1_ttt_ms,
            b1_offset_db=self.config.b1_offset_db,
            b2_threshold1_dbm=self.config.b2_threshold1_dbm,
            b2_threshold2_dbm=self.config.b2_threshold2_dbm,
            b2_ttt_ms=self.config.b2_ttt_ms,
            b2_offset_db=self.config.b2_offset_db,
            filter_coef=self.config.filter_coef,
            l3_filter_passthrough=getattr(self.config, "l3_filter_passthrough", False),
            n310=self.config.n310,
            n311=self.config.n311,
            t310_ms=self.config.t310_ms,
            t304_ms=self.config.t304_ms,
            t311_ms=self.config.t311_ms,
            preamble_initial_power_dbm=self.config.preamble_initial_power_dbm,
            preamble_tx_max=self.config.preamble_tx_max,
            power_ramping_step_db=self.config.power_ramping_step_db,
            # RLF thresholds
            qout_rsrp=self.config.qout_rsrp,
            qin_rsrp=self.config.qin_rsrp,
            # SINR-based RLF and RACH (3GPP-compliant, toggleable via CLI)
            use_sinr_for_rlf=self.config.use_sinr_for_rlf,
            use_ul_sinr_for_rach=self.config.use_ul_sinr_for_rach,
            rlf_mcs_index=self.config.rlf_mcs_index,  # -1=AMC, 0-28=fixed (3GPP TS 38.133 §8.1)
            qout_bler=getattr(self.config, "qout_bler", 0.10),
            qin_bler=getattr(self.config, "qin_bler", 0.02),
            gnb_ho_decision_delay_std_ms=getattr(self.config, "gnb_ho_decision_delay_std_ms", 0.0),
            gnb_ho_decision_delay_adapt_to_rf=getattr(self.config, "gnb_ho_decision_delay_adapt_to_rf", False),
            t300_ms=self.config.t300_ms,
            t301_ms=self.config.t301_ms,
            gnb_ho_decision_delay_ms=getattr(self.config, "gnb_ho_decision_delay_ms", 0.0),
            post_ho_blacklist_s=getattr(self.config, "post_ho_blacklist_s", 0.0),
        )
        state_machines = {ue_id: UEStateMachine(ue_id, sm_config) for ue_id in ue_ids}

        # Install the dynamic per-band / per-speed RLM Qin/Qout table (sparse
        # override; absent ⇒ legacy ~240 km/h single fit). The serving cell's
        # band label is derived from its carrier frequency; the live UE speed
        # (set per-tick from velocity_kmh) selects the nearest speed bucket.
        _cell_band = {gnb.gnb_id: _band_label(getattr(gnb, "frequency_ghz", 3.5))
                      for gnb in gnb_configs}
        _rlm_bs_table = getattr(self.config, "rlm_qin_qout_by_band_speed", None) or {}
        for _sm in state_machines.values():
            _sm._cell_band = _cell_band
            _sm._rlm_band_speed_table = _rlm_bs_table

        # Build per-cell HO config lookup (CSV overrides, CLI defaults as fallback)
        self._cell_configs = self._build_cell_configs(gnb_configs, sm_config)

        # Track per-UE state for event detection
        ue_serving_cells = {ue_id: -1 for ue_id in ue_ids}

        # 5. Register UEs with channel model (initial positions)
        self.logger.info("Registering UEs with channel model...")
        for ue_id in ue_ids:
            traj = trajectories[ue_id]
            first_row = traj.iloc[0]
            pos = (float(first_row['x']), float(first_row['y']), float(first_row['z']))
            channel_model.add_ue(ue_id, pos)

        # Initial activation of nearby gNBs (will be refreshed each timestep)
        if hasattr(channel_model, 'activate_nearby_gnbs'):
            self.logger.info("Activating nearby gNBs (initial)...")
            positions = []
            for ue_id in ue_ids:
                traj = trajectories[ue_id]
                first_row = traj.iloc[0]
                positions.append((float(first_row['x']), float(first_row['y']), float(first_row['z'])))
            avg_pos = tuple(sum(p[i] for p in positions) / len(positions) for i in range(3))
            n_activated = channel_model.activate_nearby_gnbs(
                center=avg_pos,
                radius_m=self.config.activation_radius_m,
                max_count=self.config.rt_max_gnbs,
            )
            self.logger.info(f"Initial activation: {n_activated} gNBs")

        # 6. Main simulation loop
        self.logger.info("Starting simulation loop...")
        start_wall = time_module.time()
        prev_timestamp = timestamps[0]

        for step_idx, timestamp in enumerate(timestamps):
            import time as _time
            _t_ts = _time.perf_counter()
            dt = timestamp - prev_timestamp if step_idx > 0 else 0.0

            # 6a. Update UE positions + velocities
            ue_positions = {}
            for ue_id in ue_ids:
                traj = trajectories[ue_id]
                # Find row closest to this timestamp
                dt_diff = (traj['timestamp'] - timestamp).abs()
                closest_idx = dt_diff.idxmin()
                if dt_diff[closest_idx] > 0.1:
                    continue  # UE not active at this timestamp
                row = traj.loc[closest_idx]
                pos = (float(row['x']), float(row['y']), float(row['z']))
                ue_positions[ue_id] = pos

                # Compute velocity from trajectory
                row_num = traj.index.get_loc(closest_idx)
                velocity = compute_velocity_from_df(traj, row_num)
#                velocity = (0.0, 0.0, 0.0)   #★ TEST: hardcode zero velocity (disable Doppler)

                channel_model.update_ue_position(ue_id, pos, velocity=velocity)

                # Per-UE ground speed (km/h) for the detailed_log read-out.
                # Prefer the recorded velocity_kmh column (CSV mode hard-codes
                # the velocity TUPLE to ~300 km/h for Doppler, so the recorded
                # field-true speed is the right value to log); else derive it
                # from the velocity vector magnitude.
                if not hasattr(self, "_ue_velocity_kmh"):
                    self._ue_velocity_kmh = {}
                _vk = None
                if "velocity_kmh" in traj.columns:
                    try:
                        _vk = float(row["velocity_kmh"])
                    except (TypeError, ValueError):
                        _vk = None
                if _vk is None or not (_vk == _vk):  # NaN guard
                    _vk = (velocity[0] ** 2 + velocity[1] ** 2
                           + velocity[2] ** 2) ** 0.5 * 3.6
                self._ue_velocity_kmh[ue_id] = _vk

            if not ue_positions:
                prev_timestamp = timestamp
                continue

            # 6b. Activate nearby gNBs based on current UE positions (NR + LTE)
            if hasattr(channel_model, 'activate_nearby_gnbs'):
                positions_list = list(ue_positions.values())
                avg_x = sum(p[0] for p in positions_list) / len(positions_list)
                avg_y = sum(p[1] for p in positions_list) / len(positions_list)
                avg_z = sum(p[2] for p in positions_list) / len(positions_list)
                channel_model.activate_nearby_gnbs(
                    center=(avg_x, avg_y, avg_z),
                    radius_m=self.config.activation_radius_m,
                )
                # Ensure serving cells remain in active set
                for uid, sm in state_machines.items():
                    serving_id = sm.state.serving_cell_id
                    if serving_id is not None and hasattr(channel_model, 'ensure_gnb_active'):
                        channel_model.ensure_gnb_active(serving_id)

            # 6c. Compute channel states
            # ★ v2: serving_cells를 전달하여 BLER이 실제 서빙셀 기준으로 계산
            serving_cells_map = {
                uid: sm.state.serving_cell_id
                for uid, sm in state_machines.items()
                if uid in ue_positions and sm.state.serving_cell_id is not None
                and sm.state.serving_cell_id >= 0
            }

            _t_compute = _time.perf_counter()
            all_states = channel_model.compute_all(
                timestamp,
                mcs_index=self.config.mcs_index if self.config.mcs_index >= 0 else 9,
                serving_cells=serving_cells_map,
            )
            self.logger.debug(f"[TIMING] compute_all: {_time.perf_counter() - _t_compute:.3f}s")
#            print("channel_model noise", )

            # 6c. Process each UE
            for ue_id, pos in ue_positions.items():
                step_data = self._process_ue_step(
                    ue_id, pos, timestamp, dt, step_idx + 1,
                    all_states, state_machines[ue_id],
                    gnb_lookup, ue_serving_cells, channel_model,
                )
                if step_data:
                    self.step_data.append(step_data)

            # 6d. Verbose per-step logging
            if self.config.verbose and (step_idx + 1) % self.config.verbose_interval == 0:
                self._print_verbose_step(step_idx + 1, len(timestamps))

            # 6e. Per-timestamp wireless summary (logger.info). Skip entirely
            # when INFO is disabled (e.g. CSV/BO runs at --log-level WARNING):
            # the summary only emits via logger.info, but its body does an
            # O(step) scan of self.step_data, so building it for discarded
            # output was ~16% of runtime. Output is unchanged.
            if self.logger.isEnabledFor(logging.INFO):
                self._log_timestamp_summary(timestamp, step_idx + 1, len(timestamps),
                                           ue_positions, state_machines,
                                           all_states, gnb_lookup)

            # Progress logging every 5 seconds of sim time
            if step_idx > 0 and timestamp % 5.0 < (timestamp - prev_timestamp + 0.001):
                elapsed = time_module.time() - start_wall
                self.logger.info(
                    f"  t={timestamp:.1f}s ({step_idx+1}/{len(timestamps)}) "
                    f"wall={elapsed:.1f}s"
                )

            self.logger.debug(f"[TIMING] timestep t={timestamp:.2f}s: {_time.perf_counter() - _t_ts:.3f}s")
            prev_timestamp = timestamp

        # 7. Generate report
        elapsed_total = time_module.time() - start_wall
        self._generate_report(ue_ids, state_machines, elapsed_total, gnb_lookup)

    def _process_ue_step(self, ue_id, position, timestamp, dt, step_num,
                         all_states, state_machine, gnb_lookup, ue_serving_cells,
                         channel_model=None):
        """Process one simulation step for a UE.

        Returns:
            UEStepData for verbose logging
        """
        import time as _time
        _t_step = _time.perf_counter()
        serving_cell = state_machine.state.serving_cell_id

        # Helper: gnb_id -> PCI label for CLI logs
        def _pci(gnb_id):
            if gnb_id is None:
                return "?"
            info = gnb_lookup.get(gnb_id)
            if info:
                pci = getattr(info, 'pci', None)
                return int(pci) if pci is not None else gnb_id
            return gnb_id

        # Extract RSRP, DL SINR, UL SINR from channel states
        # Note: channel_model.compute_all() returns Dict[(ue_id, gnb_id, sector_id), ChannelState]
        rsrp_values = {}
        sinr_values = {}
        ul_sinr_values = {}
        rsrq_values = {}
        serving_state = None
        for (uid, gnb_id, sector_id), state in all_states.items():
            if uid == ue_id:
                # 동일 gnb_id의 여러 sector 중 best RSRP sector 사용
                if gnb_id not in rsrp_values or state.rsrp_dbm > rsrp_values[gnb_id]:
                    rsrp_values[gnb_id] = state.rsrp_dbm
                    sinr_values[gnb_id] = state.sinr_db
                    ul_sinr_values[gnb_id] = state.ul_sinr_db
                    # RSRQ: prefer ChannelState.rsrq_db if set; else estimate
                    # from SINR: RSRQ_dB = 10*log10(SINR_lin / (1 + SINR_lin))
                    # (matches dm_pipeline StageF_Wide.derive_missing_rsrq).
                    _rsrq_attr = getattr(state, 'rsrq_db', None)
                    if _rsrq_attr is not None and _rsrq_attr != 0.0:
                        rsrq_values[gnb_id] = float(_rsrq_attr)
                    else:
                        try:
                            _sinr_lin = 10.0 ** (float(state.sinr_db) / 10.0)
                            rsrq_values[gnb_id] = 10.0 * math.log10(
                                _sinr_lin / (1.0 + _sinr_lin) + 1e-30)
                        except (TypeError, ValueError):
                            rsrq_values[gnb_id] = -20.0
                if gnb_id == serving_cell:
                    if serving_state is None or state.rsrp_dbm > serving_state.rsrp_dbm:
                        serving_state = state

        serving_rsrp = rsrp_values.get(serving_cell, -140.0)
        sinr_db = serving_state.sinr_db if serving_state else -30.0

        # Build raw_measurements dict for UEStateMachine
        raw_measurements = dict(rsrp_values)

        # Classify neighbor cells: intra-freq (A3), inter-freq (A5), inter-RAT (B1/B2)
        serving_info = gnb_lookup.get(serving_cell)
        serving_freq = serving_info.frequency_ghz if serving_info else self.config.frequency_ghz
        serving_is_lte = serving_info.is_lte if serving_info else False

        inter_freq_cells = []
        inter_rat_cells = []
        for gnb_id in rsrp_values:
            if gnb_id == serving_cell:
                continue
            info = gnb_lookup.get(gnb_id)
            if info:
                same_rat = (info.is_lte == serving_is_lte)
                same_freq = abs(info.frequency_ghz - serving_freq) < 0.01
                if not same_rat:
                    # Different RAT (NR↔LTE) → inter-RAT (B1/B2)
                    inter_rat_cells.append(gnb_id)
                elif not same_freq:
                    # Same RAT, different freq (e.g., LTE 0.9→1.8) → inter-freq (A5)
                    inter_freq_cells.append(gnb_id)

        # Store previous state for event detection
        prev_serving = state_machine.state.serving_cell_id
        prev_ho = state_machine.state.ho_in_progress
        prev_rlf = state_machine.state.rlf_declared
        prev_rrc_state = state_machine.state.rrc_state
        prev_target = state_machine.state.target_cell_id  # Capture before update clears it
        prev_last_hof = state_machine.state.last_hof_classification

        # RSRQ-A3 wiring: when --use-rsrq-for-regular-cells is on, build a
        # per-cell quantity map: HSR cells stay on RSRP (real HSR networks
        # use RSRP-only A3), regular cells switch to RSRQ. Cells without
        # is_hsr_cell metadata default to "rsrq" under this flag (treated as
        # regular). When the flag is off, raw_rsrq=None and quantity_for_a3
        # is empty ⇒ measurement engine stays in RSRP-A3 mode (baseline).
        if getattr(self.config, "use_rsrq_for_regular_cells", False):
            quantity_for_a3 = {}
            for _gid in rsrp_values:
                _info = gnb_lookup.get(_gid)
                _hsr_raw = getattr(_info, "is_hsr_cell", False) if _info else False
                _is_hsr = bool(_hsr_raw) if _hsr_raw is not None else False
                quantity_for_a3[int(_gid)] = "rsrp" if _is_hsr else "rsrq"
            raw_rsrq_arg = rsrq_values
        else:
            quantity_for_a3 = None
            raw_rsrq_arg = None

        # Per-tick UE speed for the dynamic per-band/per-speed RLM Qin/Qout
        # selection (UEStateMachine._rlm_band_speed_params). None ⇒ legacy fit.
        state_machine._current_velocity_kmh = (
            getattr(self, "_ue_velocity_kmh", {}) or {}).get(ue_id)

        # Update state machine (pass DL/UL SINR for SINR-based RLF + RACH)
        _t_sm = _time.perf_counter()
        ue_state = state_machine.update(timestamp, dt, raw_measurements,
                                         inter_freq_cells=inter_freq_cells,
                                         inter_rat_cells=inter_rat_cells,
                                         sinr_measurements=sinr_values,
                                         ul_sinr_measurements=ul_sinr_values,
                                         raw_rsrq_measurements=raw_rsrq_arg,
                                         quantity_for_a3=quantity_for_a3)

        self.logger.debug(f"[TIMING] state_machine.update UE{ue_id}: {_time.perf_counter() - _t_sm:.3f}s")

        # Detect and log events by comparing prev vs current state
        curr_serving = ue_state.serving_cell_id

        # HO Start
        if not prev_ho and ue_state.ho_in_progress:
            target = ue_state.target_cell_id
            ho_type_str = f" ({ue_state.ho_type})" if ue_state.ho_type else ""
            self.logger.info(f"[UE{ue_id}] HO_START{ho_type_str} at t={timestamp:.3f}s: "
                           f"PCI {_pci(prev_serving)} -> {_pci(target)}")
            self.events.append(SimulationEvent(
                timestamp=timestamp, ue_id=ue_id, event_type="HO_START",
                source_cell=prev_serving, target_cell=target,
                rsrp_dbm=serving_rsrp, sinr_db=sinr_db,
                details=ue_state.ho_type or None,
            ))

        # HO Complete (serving cell changed while HO was in progress)
        if prev_ho and not ue_state.ho_in_progress and not ue_state.rlf_declared:
            if curr_serving != prev_serving:
                self.logger.info(f"[UE{ue_id}] HO_COMPLETE at t={timestamp:.3f}s: "
                               f"PCI {_pci(prev_serving)} -> {_pci(curr_serving)}")
                self.events.append(SimulationEvent(
                    timestamp=timestamp, ue_id=ue_id, event_type="HO_COMPLETE",
                    source_cell=prev_serving, target_cell=curr_serving,
                    rsrp_dbm=rsrp_values.get(curr_serving, -140),
                ))
                # ★ v3: 새 서빙셀 기준으로 BLER 윈도우 리셋
                if channel_model and hasattr(channel_model, 'reset_sliding_bler'):
                    channel_model.reset_sliding_bler(ue_id)

        # RLF unification (default ON, 2026-05-20): SIB_READ_FAILURE and
        # RACH_PROBLEM (T304/preamble) failures all culminate in an RLF
        # declaration; treating them as separate event classes made every
        # downstream report split RLF vs RACH vs SIB and confused analysis.
        # When unified: the RLF event's cause is normalized to "RLF" and the
        # parallel SIB_READ_FAILURE / HO_FAIL events are suppressed (the single
        # RLF event already covers them — no info loss for counting).
        # Set RLF_UNIFY=0 to restore the multi-class causes for FIELD rlf_cause
        # comparison (harness/compare tooling).
        _rlf_unify = os.environ.get("RLF_UNIFY", "1") not in ("0", "false", "False", "")

        # RLF Declared
        if not prev_rlf and ue_state.rlf_declared:
            rlf_cause = ue_state.pending_context.get("rlf_cause", "")
            # Counter is reset inside _declare_rlf — read snapshot stored in
            # pending_context by _classify_hof_on_rlf before the reset.
            sib_ticks = ue_state.pending_context.get(
                "rlf_sib_ticks",
                getattr(ue_state, "sib_block_ticks_during_t310", 0),
            )
            if _rlf_unify:
                # Normalize all RLF-class causes to a single "RLF" event_type so
                # ul_block msg3 failure / SIB read failure / RACH(T304) HO failure
                # / plain T310 RLF are ALL counted as one RLF (matches FIELD's
                # rlf_cause which lumps them). The underlying sub-cause is kept in
                # `details` (cause=...) for timing analysis — it does NOT change
                # the event_type or the RLF count.
                _sub = rlf_cause or "RLF"
                self.logger.warning(f"[UE{ue_id}] RLF at t={timestamp:.3f}s in PCI {_pci(prev_serving)}")
                detail_str = f"RLF|cause={_sub}|sib_ticks={sib_ticks}"
            else:
                rlf_cause_str = f" ({rlf_cause})" if rlf_cause else ""
                self.logger.warning(f"[UE{ue_id}] RLF{rlf_cause_str} at t={timestamp:.3f}s in PCI {_pci(prev_serving)}")
                detail_str = f"{rlf_cause}|sib_ticks={sib_ticks}" if rlf_cause else f"sib_ticks={sib_ticks}"
            self.events.append(SimulationEvent(
                timestamp=timestamp, ue_id=ue_id, event_type="RLF",
                source_cell=prev_serving, rsrp_dbm=serving_rsrp, sinr_db=sinr_db,
                details=detail_str,
            ))
            # Parallel SIB_READ_FAILURE event (legacy multi-class) — suppressed
            # under RLF unification since the RLF event above already counts it.
            if not _rlf_unify and rlf_cause == "SIB_READ_FAILURE":
                self.events.append(SimulationEvent(
                    timestamp=timestamp, ue_id=ue_id,
                    event_type="SIB_READ_FAILURE",
                    source_cell=prev_serving, rsrp_dbm=serving_rsrp,
                    details="report block sustained through T310 expiry",
                ))

        # HO Failure: RLF just occurred while HO was in progress.
        # Under RLF unification this is suppressed — the RLF event at the same
        # timestamp already represents the failure (no separate HO_FAIL class).
        # NOTE: Cannot check t304.expired because _declare_rlf() calls T304.stop()
        # which sets expired=False. Use prev_ho + new RLF detection instead.
        if (not _rlf_unify) and prev_ho and ue_state.rlf_declared and not prev_rlf:
            rlf_cause = ue_state.pending_context.get("rlf_cause", "T304_EXPIRE")
            self.logger.warning(f"[UE{ue_id}] HO_FAIL ({rlf_cause}) at t={timestamp:.3f}s: "
                               f"PCI {_pci(prev_serving)} -> {_pci(prev_target)}")
            self.events.append(SimulationEvent(
                timestamp=timestamp, ue_id=ue_id, event_type="HO_FAIL",
                source_cell=prev_serving, target_cell=prev_target,
                details=rlf_cause,
            ))

        # Re-establishment complete
        if prev_rlf and not ue_state.rlf_declared and ue_state.rrc_connected:
            hof_detail = ""
            if ue_state.last_hof_classification:
                hof_detail = ue_state.last_hof_classification.hof_type.value
            self.logger.info(f"[UE{ue_id}] RE_ESTABLISH at t={timestamp:.3f}s in PCI {_pci(curr_serving)}"
                           + (f" [HOF: {hof_detail}]" if hof_detail and hof_detail != "NONE" else ""))
            self.events.append(SimulationEvent(
                timestamp=timestamp, ue_id=ue_id, event_type="RE_ESTABLISH",
                source_cell=prev_serving, target_cell=curr_serving,
                details=hof_detail or None,
            ))
            # ★ v3: 재접속 후 새 셀 기준으로 BLER 윈도우 리셋
            if channel_model and hasattr(channel_model, 'reset_sliding_bler'):
                channel_model.reset_sliding_bler(ue_id)

        # Log HOF classification when it changes
        curr_hof = ue_state.last_hof_classification
        if curr_hof and curr_hof is not prev_last_hof and curr_hof.hof_type.value != "NONE":
            self.logger.warning(
                f"[UE{ue_id}] HOF_CLASSIFIED: {curr_hof.hof_type.value} at t={timestamp:.3f}s | {curr_hof.cause}")
            self.events.append(SimulationEvent(
                timestamp=timestamp, ue_id=ue_id,
                event_type="HOF_CLASSIFIED",
                source_cell=curr_hof.last_ho_source if hasattr(curr_hof, 'last_ho_source') else prev_serving,
                target_cell=curr_hof.last_ho_target if hasattr(curr_hof, 'last_ho_target') else curr_serving,
                details=curr_hof.hof_type.value,
            ))

        # RRC Connection Setup complete (IDLE -> CONNECTED, not from re-establishment)
        from rrc.rrc_types import RRCState
        if (prev_rrc_state == RRCState.RRC_IDLE
                and ue_state.rrc_state == RRCState.RRC_CONNECTED
                and not prev_rlf):
            self.logger.info(f"[UE{ue_id}] RRC_SETUP at t={timestamp:.3f}s in PCI {_pci(curr_serving)}")
            self.events.append(SimulationEvent(
                timestamp=timestamp, ue_id=ue_id, event_type="RRC_SETUP",
                source_cell=prev_serving, target_cell=curr_serving,
                rsrp_dbm=rsrp_values.get(curr_serving, -140),
            ))
            # Reset BLER window for new connection
            if channel_model and hasattr(channel_model, 'reset_sliding_bler'):
                channel_model.reset_sliding_bler(ue_id)

        # NOTE: SIB_READ_FAILURE is no longer emitted per-tick. Field log's
        # `rlf_cause = SIB_READ_FAILURE` is a single RLF classification, not
        # a per-tick counter. The event is now emitted once at RLF time when
        # `pending_context.rlf_cause == "SIB_READ_FAILURE"` (see RLF emit
        # block above). `sib_block_blocked_target` is kept for in-state
        # introspection only (no event side-effect).

        # Measurement events (A2, A3, A5, B1, B2 trigger)
        # For A3, annotate which quantity drove the trigger (RSRP vs RSRQ)
        # so log readers can verify HSR (RSRP) vs regular (RSRQ) routing.
        # Per 3GPP TS 38.331 §5.3.5.5.2: when UE receives reconfigurationWithSync
        # (HO command), MAC is reset and measurement reporting on the source
        # PCell is suspended until target sync completes (or T304 expires →
        # RLF). Suppress MEAS_* emission while ho_in_progress so events.csv
        # reflects spec behaviour — HO_START is the last source-side report.
        _meas_engine = getattr(state_machine, "measurement_engine", None)
        _q_a3 = getattr(_meas_engine, "quantity_for_a3", {}) if _meas_engine else {}
        _serving_a3_q = _q_a3.get(int(curr_serving), "rsrp") if curr_serving is not None else "rsrp"
        if not ue_state.ho_in_progress:
            for evt_name, evt_state in ue_state.measurement_events.items():
                _details = (f"q={_serving_a3_q}" if evt_name == "A3" else "")
                if evt_state.report_sent:
                    self.events.append(SimulationEvent(
                        timestamp=timestamp, ue_id=ue_id,
                        event_type=f"MEAS_{evt_name}",
                        source_cell=curr_serving,
                        target_cell=evt_state.target_cell_id,
                        rsrp_dbm=serving_rsrp,
                        details=_details,
                    ))
                elif evt_state.triggered and evt_state.target_cell_id is not None:
                    # TTT in progress (not yet reported)
                    self.events.append(SimulationEvent(
                        timestamp=timestamp, ue_id=ue_id,
                        event_type=f"MEAS_{evt_name}_TTT",
                        source_cell=curr_serving,
                        target_cell=evt_state.target_cell_id,
                        rsrp_dbm=serving_rsrp,
                        details=_details,
                    ))

        # Update serving cell tracking
        ue_serving_cells[ue_id] = curr_serving

        # Per-cell config swap: apply new serving cell's HO parameters
        if curr_serving != prev_serving and curr_serving in self._cell_configs:
            state_machine.apply_config(self._cell_configs[curr_serving])

        # Build step data for verbose logging
        serving_rsrp_final = rsrp_values.get(curr_serving, -140.0)
        serving_sinr_final = sinr_values.get(curr_serving, 0.0)
        # SINR breakdown from ChannelState
        _sinr_base = serving_state.sinr_base_db if serving_state and hasattr(serving_state, 'sinr_base_db') else serving_sinr_final
        _doppler_pen = serving_state.doppler_penalty_db if serving_state and hasattr(serving_state, 'doppler_penalty_db') else 0.0

        # ★ BLER & MCS selection
        if self.config.mcs_index < 0:
            # Adaptive MCS: sigmoid 기반 MCS 선택 + 해당 MCS의 BLER
            bler, selected_mcs = self.phy.sinr_to_bler_adaptive(serving_sinr_final)
        elif serving_state and getattr(serving_state, 'bler', None) is not None:
            bler = serving_state.bler
            selected_mcs = self.config.mcs_index
        else:
            bler = self.phy.sinr_to_bler(serving_sinr_final, self.config.mcs_index)
            selected_mcs = self.config.mcs_index

        # Status string
        status = "OK"
        if ue_state.rlf_declared:
            reest_phase = ue_state.pending_context.get("reest_phase", "?")
            status = f"RLF! (reest:{reest_phase})"
        elif ue_state.rrc_state == RRCState.RRC_IDLE:
            setup_phase = ue_state.pending_context.get("setup_phase", "cell_selection")
            status = f"IDLE (setup:{setup_phase})"
        elif ue_state.ho_in_progress:
            status = f"HO->{ue_state.target_cell_id}"
        elif ue_state.timers.get("T310") and ue_state.timers["T310"].running:
            status = "T310 running"

        # Active timers
        active_timers = [name for name, timer in ue_state.timers.items()
                        if timer and timer.running]

        # Get counter values
        n310_counter = ue_state.counters.get("N310")
        n311_counter = ue_state.counters.get("N311")
        n310_val = n310_counter.count if n310_counter else 0
        n311_val = n311_counter.count if n311_counter else 0

        # Capture detailed state snapshot for dump-style log
        # Include ALL per-cell TTTs from measurement engine (not just best)
        meas_snap = {}
        for name, es in ue_state.measurement_events.items():
            if es.report_sent:
                meas_snap[name] = {
                    "triggered": True,
                    "ttt_rem": 0.0,
                    "report": True,
                    "target": es.target_cell_id,
                }
        # Add per-cell TTT states from measurement engine
        me = getattr(state_machine, 'measurement_engine', None)
        if me and hasattr(me, 'ttt_trackers'):
            for evt_type in ["A3", "A5"]:
                tracker = me.ttt_trackers.get(evt_type, {})
                if evt_type == "A3":
                    ttt_sec = me.config.time_to_trigger_ms / 1000.0
                elif evt_type == "A5":
                    ttt_sec = (me.config.a5_ttt_ms if me.config.a5_ttt_ms > 0
                               else me.config.time_to_trigger_ms) / 1000.0
                else:
                    ttt_sec = me.config.time_to_trigger_ms / 1000.0
                for key, elapsed in tracker.items():
                    cid = int(key) if key.isdigit() else None
                    if cid is None:
                        continue
                    remaining = max(0, (ttt_sec - elapsed) * 1000.0)
                    label = f"{evt_type}_{cid}"
                    meas_snap[label] = {
                        "triggered": True,
                        "ttt_rem": remaining,
                        "report": False,
                        "target": cid,
                    }
        timer_snap = {}
        for tname, ts in ue_state.timers.items():
            if ts.running or ts.expired:
                timer_snap[tname] = {
                    "running": ts.running,
                    "expired": ts.expired,
                    "remaining": (ts.duration - (timestamp - ts.start_time)) * 1000 if ts.running else 0,
                }
        counter_snap = {}
        for cname, cs in ue_state.counters.items():
            counter_snap[cname] = {"count": cs.count, "threshold": cs.threshold}

        # Collect events generated this step
        events_this = [e.event_type for e in self.events
                       if e.ue_id == ue_id and abs(e.timestamp - timestamp) < 0.001]

        # A3 measurement-engine diagnostics snapshot (read-only).
        # Surfaces internal A3 evaluation state into detailed_log_ue*.csv.
        a3_diag: Dict[str, Any] = {}
        if me is not None and hasattr(me, "a3_diagnostics_snapshot") and \
                curr_serving is not None:
            try:
                a3_diag = me.a3_diagnostics_snapshot(
                    int(curr_serving), current_time_s=float(timestamp)
                )
            except Exception:
                # Diagnostics are best-effort; do not affect sim correctness.
                a3_diag = {}
        a3_cands = a3_diag.get("candidates", []) if a3_diag else []

        def _a3_cand(i: int, key: str):
            if i < len(a3_cands):
                return a3_cands[i].get(key)
            return None

        self.logger.debug(f"[TIMING] _process_ue_step UE{ue_id}: {_time.perf_counter() - _t_step:.3f}s")

        return UEStepData(
            step=step_num,
            timestamp=timestamp,
            ue_id=ue_id,
            position=position,
            serving_cell=curr_serving,
            rsrp_dbm=serving_rsrp_final,
            sinr_db=serving_sinr_final,
            bler=bler,
            mcs=selected_mcs,
            status=status,
            n310=n310_val,
            n311=n311_val,
            timers=active_timers,
            all_rsrp=dict(rsrp_values),
            all_sinr=dict(sinr_values),
            all_rsrq=dict(rsrq_values),
            rrc_state=ue_state.rrc_state.name if hasattr(ue_state.rrc_state, 'name') else str(ue_state.rrc_state),
            ho_in_progress=ue_state.ho_in_progress,
            rlf_declared=ue_state.rlf_declared,
            target_cell=ue_state.target_cell_id,
            meas_snapshot=meas_snap,
            timer_snapshot=timer_snap,
            counter_snapshot=counter_snap,
            events_this_step=events_this,
            radio_link_status=ue_state.radio_link_status.name if hasattr(ue_state.radio_link_status, 'name') else str(ue_state.radio_link_status),
            sinr_base_db=_sinr_base,
            doppler_penalty_db=_doppler_pen,
            # A3 diagnostics — see UEStepData docstring above.
            a3_serv_filt_rsrq=a3_diag.get("serving_filt_rsrq") if a3_diag else None,
            a3_serv_filt_rsrp=a3_diag.get("serving_filt_rsrp") if a3_diag else None,
            a3_quantity=str(a3_diag.get("serving_a3_quantity", "")) if a3_diag else "",
            a3_filter_age_ms=a3_diag.get("filter_age_ms") if a3_diag else None,
            a3_top1_cell=_a3_cand(0, "cell_id"),
            a3_top1_metric=_a3_cand(0, "metric"),
            a3_top1_ttt_ms=_a3_cand(0, "ttt_elapsed_ms"),
            a3_top1_age=_a3_cand(0, "age"),
            a3_top1_entering=_a3_cand(0, "entering"),
            a3_top2_cell=_a3_cand(1, "cell_id"),
            a3_top2_metric=_a3_cand(1, "metric"),
            a3_top2_ttt_ms=_a3_cand(1, "ttt_elapsed_ms"),
            a3_top2_age=_a3_cand(1, "age"),
            a3_top2_entering=_a3_cand(1, "entering"),
            a3_top3_cell=_a3_cand(2, "cell_id"),
            a3_top3_metric=_a3_cand(2, "metric"),
            a3_top3_ttt_ms=_a3_cand(2, "ttt_elapsed_ms"),
            a3_top3_age=_a3_cand(2, "age"),
            a3_top3_entering=_a3_cand(2, "entering"),
            a3_tracker_n=int(a3_diag.get("tracker_n", 0)) if a3_diag else 0,
            # UL-block / RLM / velocity read-outs (2026-06-12): observability
            # for WHY the UL gate engaged (the L3-filtered RSRQ it compared,
            # not the raw), the RLM OOS/IS BLER verdict, and UE speed.
            ul_block_applied_rsrq=getattr(ue_state, "ul_block_applied_rsrq", None),
            ul_block_applied_rsrp=getattr(ue_state, "ul_block_applied_rsrp", None),
            ul_block_threshold_rsrq_db=getattr(ue_state, "ul_block_threshold_rsrq_db", None),
            ul_block_path=getattr(ue_state, "ul_block_path", "") or "",
            rlm_smoothed_sinr_db=getattr(ue_state, "rlm_smoothed_sinr_db", None),
            rlm_bler_qout=getattr(ue_state, "rlm_bler_qout", None),
            rlm_bler_qin=getattr(ue_state, "rlm_bler_qin", None),
            ue_velocity_kmh=(getattr(self, "_ue_velocity_kmh", {}) or {}).get(ue_id),
            l3_filter_k=int(getattr(self.config, "filter_coef", 4)),
            # Gate-reason diagnostics (T1): read-outs only.
            ho_suppress_reason=getattr(ue_state, "ho_suppress_reason", "") or "",
            report_block_reason=str(a3_diag.get("report_block_reason", "")) if a3_diag else "",
            report_gap_ms=a3_diag.get("report_gap_ms") if a3_diag else None,
            # Staged HO signaling flow read-outs.
            ho_stage=getattr(ue_state, "ho_stage", "") or "",
            ho_cmd_decoded=bool(getattr(ue_state, "ho_cmd_decoded", False)),
            target_rach_ok=bool(getattr(ue_state, "target_rach_ok", False)),
            target_rrc_ok=bool(getattr(ue_state, "target_rrc_ok", False)),
        )

    def _print_verbose_step(self, step_num, total_steps):
        """Print verbose per-step table to stdout."""
        # Print header on first call
        if not self._verbose_header_printed:
            print("\n" + "=" * 113)
            print(f"{'Step':>6} {'Time':>8} {'UE':>4} {'X':>10} {'Y':>10} {'Serving':>10} {'RSRP':>8} {'SINR':>8} {'MCS':>5} {'BLER':>10} {'Status':<15}")
            print("=" * 113)
            self._verbose_header_printed = True

        # Print last N steps (to avoid overwhelming output)
        recent_steps = [s for s in self.step_data if s.step == step_num]
        for data in recent_steps:
            # BLER indicator
            if data.bler > 0.10:
                bler_indicator = "▲"  # Out of sync
            elif data.bler < 0.02:
                bler_indicator = "▼"  # In sync
            else:
                bler_indicator = "─"  # Gray zone

            # Format BLER percentage
            bler_str = f"{data.bler * 100:5.2f}%{bler_indicator}"

            # Cell prefix with PCI
            _gnb_lk = getattr(self, '_gnb_lookup', None) or {}
            _info = _gnb_lk.get(data.serving_cell)
            if _info:
                _prefix = "eNB" if _info.is_lte else "gNB"
                _pci_val = getattr(_info, 'pci', data.serving_cell)
                cell_str = f"{_prefix} {int(_pci_val) if _pci_val is not None else data.serving_cell}"
            else:
                cell_str = f"C {data.serving_cell}"

            print(f"{data.step:6d} {data.timestamp:8.2f} {str(data.ue_id):>4} "
                  f"{data.position[0]:10.1f} {data.position[1]:10.1f} "
                  f"{cell_str:>10} {data.rsrp_dbm:8.1f} {data.sinr_db:8.1f} "
                  f"{data.mcs:5d} {bler_str:>10} {data.status:<15}")

    def _log_timestamp_summary(self, timestamp, step_num, total_steps,
                               ue_positions, state_machines, all_states, gnb_lookup):
        """Log per-timestamp wireless summary via logger.info."""
        # Collect visible frequencies
        visible_freqs = set()
        for (uid, gnb_id, sid) in all_states:
            if uid in ue_positions:
                info = gnb_lookup.get(gnb_id)
                if info:
                    visible_freqs.add(info.frequency_ghz)

        freq_summary = ", ".join(f"{f}GHz" for f in sorted(visible_freqs))
        visible_count = len(set(gnb_id for (_, gnb_id, _) in all_states.keys()))

        self.logger.info(f"--- [{step_num}/{total_steps}] t={timestamp:.2f}s | "
                        f"Visible cells: {visible_count} [{freq_summary}] ---")

        for ue_id in ue_positions:
            sm = state_machines[ue_id]
            state = sm.state
            serving_id = state.serving_cell_id

            # Get serving cell info
            serving_info = gnb_lookup.get(serving_id)
            if serving_info:
                cell_prefix = "eNB" if serving_info.is_lte else "gNB"
                _pci_v = getattr(serving_info, 'pci', None)
                if _pci_v is not None:
                    cell_name = f"{cell_prefix}{int(_pci_v)}"
                elif getattr(serving_info, 'name', None):
                    cell_name = serving_info.name
                else:
                    cell_name = f"{cell_prefix}{serving_id}"
                antenna_str = serving_info.antenna_type if serving_info.antenna_type else "Unknown Antenna"
                freq_str = f"{serving_info.frequency_ghz}GHz"
                bw_str = f"{serving_info.bandwidth_mhz}MHz"
                power_str = f"{serving_info.tx_power_dbm}dBm"
            else:
                cell_name = f"Cell {serving_id}"
                antenna_str = "Unknown"
                freq_str = "?.?GHz"
                bw_str = "??MHz"
                power_str = "??dBm"

            # Get RSRP/SINR from step data
            step_data = next((s for s in self.step_data if s.ue_id == ue_id and s.step == step_num), None)
            if step_data:
                rsrp = step_data.rsrp_dbm
                sinr = step_data.sinr_db
                bler = step_data.bler
                n310 = step_data.n310
                n311 = step_data.n311
                timers_str = ", ".join(step_data.timers) if step_data.timers else "none"
                all_rsrp = step_data.all_rsrp
            else:
                rsrp = -140.0
                sinr = 0.0
                bler = 1.0
                n310 = 0
                n311 = 0
                timers_str = "none"
                all_rsrp = {}

            # BLER-based sync status (actual data MCS)
            if bler > 0.10:
                bler_status = "OOS"
            elif bler < 0.02:
                bler_status = "IS"
            else:
                bler_status = "Gray"

            # RRC state
            if state.rlf_declared:
                state_str = "RRC_IDLE (RLF)"
            elif state.rrc_connected:
                state_str = "RRC_CONNECTED"
            else:
                setup_ph = state.pending_context.get("setup_phase", "cell_selection")
                state_str = f"RRC_IDLE (setup:{setup_ph})"

            # Status
            if state.rlf_declared:
                status_str = "RLF!"
            elif state.ho_in_progress:
                _tgt_info = gnb_lookup.get(state.target_cell_id)
                _tgt_pci = getattr(_tgt_info, 'pci', None) if _tgt_info else None
                _tgt_label = int(_tgt_pci) if _tgt_pci is not None else state.target_cell_id
                status_str = f"HO to PCI{_tgt_label}"
            else:
                status_str = "OK"

            self.logger.info(f"  UE{ue_id} | State: {state_str} | Status: {status_str}")
            self.logger.info(f"       | Serving: {cell_name} [{antenna_str}] {freq_str} / {bw_str} / {power_str}")
            mcs_val = step_data.mcs if step_data else self.config.mcs_index
            self.logger.info(f"       | RSRP: {rsrp:6.1f} dBm | SINR: {sinr:6.1f} dB | MCS: {mcs_val:2d} | BLER: {bler*100:5.1f}% ({bler_status})")
            # SINR breakdown: base (RSRP-ratio) + Doppler penalty from map
            sinr_base = step_data.sinr_base_db if step_data else sinr
            doppler_pen = step_data.doppler_penalty_db if step_data else 0.0
            pen_str = f"{doppler_pen:+.1f}dB" if doppler_pen != 0.0 else "N/A"
            ue_pos = step_data.position if step_data else (0, 0, 0)
            self.logger.info(f"       | SINR: base={sinr_base:+.1f} + Doppler={pen_str} → {sinr:+.1f} dB  pos=({ue_pos[0]:.0f},{ue_pos[1]:.0f})")
            self.logger.info(f"       | N310: {n310:2d}  N311: {n311:2d}  Timers: {timers_str}")

            # Top-3 cells
            top3 = sorted(all_rsrp.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_parts = []
            for i, (cid, rsrp_val) in enumerate(top3):
                info = gnb_lookup.get(cid)
                if info:
                    freq_tag = f"{info.frequency_ghz}G"
                    cell_prefix = "eNB" if info.is_lte else "gNB"
                    lte_tag = "|LTE" if info.is_lte else ""
                    _p = getattr(info, 'pci', None)
                    if _p is not None:
                        cell_label = f"{cell_prefix}{int(_p)}"
                    elif getattr(info, 'name', None):
                        cell_label = info.name
                    else:
                        cell_label = f"{cell_prefix}{cid}"
                else:
                    freq_tag = "?G"
                    cell_label = f"Cell{cid}"
                    lte_tag = ""
                top3_parts.append(f"#{i+1} {cell_label}: {rsrp_val:.1f} dBm [{freq_tag}{lte_tag}]")

            if top3_parts:
                self.logger.info(f"       | Top-3: {' '.join(top3_parts)}")

            # Measurement events (A2, A3, A5, B1, B2)
            meas_parts = []
            for evt_name in ["A2", "A3", "A5", "B1", "B2"]:
                evt = state.measurement_events.get(evt_name)
                if evt is None:
                    continue
                if evt.report_sent:
                    target_str = f"→{evt.target_cell_id}" if evt.target_cell_id else ""
                    meas_parts.append(f"{evt_name} REPORT{target_str}")
                elif evt.triggered:
                    ttt_str = f" TTT:{evt.time_to_trigger_remaining:.0f}ms" if evt.time_to_trigger_remaining > 0 else ""
                    target_str = f"→{evt.target_cell_id}" if evt.target_cell_id else ""
                    meas_parts.append(f"{evt_name} ENTER{target_str}{ttt_str}")
            if meas_parts:
                self.logger.info(f"       | Events: {' | '.join(meas_parts)}")

            # Active timers with remaining time (T304, T310, T311)
            active_timers_display = []
            for tname in ["T304", "T310", "T311"]:
                t = state.timers[tname]
                if t.running:
                    remaining_ms = t.remaining(timestamp) * 1000
                    active_timers_display.append(f"{tname}:{remaining_ms:.0f}ms")
            if active_timers_display:
                self.logger.info(f"       | Active Timers: {' | '.join(active_timers_display)}")

            # HOF classification (only show when new)
            curr_hof = state.last_hof_classification
            if curr_hof and curr_hof.hof_type.value != "NONE":
                # Show if timestamp matches (classification happened this step)
                if abs(curr_hof.timestamp - timestamp) < 0.001:
                    self.logger.info(f"       | *** HOF: {curr_hof.hof_type.value} *** {curr_hof.cause}")

    def _generate_report(self, ue_ids, state_machines, elapsed_s, gnb_lookup=None):
        """Generate simulation summary report."""
        import time as _time
        _t_report = _time.perf_counter()
        # Classify any pending HOF for UEs still in RLF at simulation end
        for ue_id in ue_ids:
            sm = state_machines[ue_id]
            sm.classify_pending_hof()

        self.logger.info("=" * 60)
        self.logger.info("SIMULATION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Wall time: {elapsed_s:.1f}s")
        self.logger.info(f"Total events: {len(self.events)}")

        # Count events by type
        event_counts = Counter(e.event_type for e in self.events)
        for etype, count in sorted(event_counts.items()):
            self.logger.info(f"  {etype}: {count}")

        # Per-UE summary
        for ue_id in ue_ids:
            ue_events = [e for e in self.events if e.ue_id == ue_id]
            ho_starts = sum(1 for e in ue_events if e.event_type == "HO_START")
            ho_completes = sum(1 for e in ue_events if e.event_type == "HO_COMPLETE")
            ho_fails = sum(1 for e in ue_events if e.event_type == "HO_FAIL")
            rlfs = sum(1 for e in ue_events if e.event_type == "RLF")

            sm = state_machines[ue_id]
            self.logger.info(
                f"  UE{ue_id}: HO={ho_starts} (success={ho_completes}, fail={ho_fails}), "
                f"RLF={rlfs}, final_cell={sm.state.serving_cell_id}"
            )

        # Save events to CSV
        os.makedirs(self.config.output_dir, exist_ok=True)
        events_path = os.path.join(self.config.output_dir, "events.csv")
        if self.events:
            import csv
            with open(events_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'ue_id', 'event_type', 'source_cell',
                    'target_cell', 'rsrp_dbm', 'sinr_db', 'details'
                ])
                writer.writeheader()
                # Helper: gnb_id -> PCI for events CSV
                def _gnb_to_pci(gnb_id):
                    if gnb_id is None:
                        return ""
                    info = gnb_lookup.get(gnb_id)
                    if info:
                        pci = getattr(info, 'pci', None)
                        return int(pci) if pci is not None else gnb_id
                    return gnb_id

                for e in self.events:
                    writer.writerow({
                        'timestamp': f"{e.timestamp:.4f}",
                        'ue_id': e.ue_id,
                        'event_type': e.event_type,
                        'source_cell': _gnb_to_pci(e.source_cell),
                        'target_cell': _gnb_to_pci(e.target_cell),
                        'rsrp_dbm': f"{e.rsrp_dbm:.1f}" if e.rsrp_dbm is not None else "",
                        'sinr_db': f"{e.sinr_db:.1f}" if e.sinr_db is not None else "",
                        'details': e.details or "",
                    })
            self.logger.info(f"Events saved to {events_path}")

        # Save per-step detailed log (dump-style CSV)
        if gnb_lookup:
            self._save_detailed_log(gnb_lookup)

        # Generate text report
        self._generate_text_report(ue_ids, state_machines)

        # HOF Classification (post-simulation)
        self._classify_hof(ue_ids, state_machines)
        self.logger.debug(f"[TIMING] _generate_report: {_time.perf_counter() - _t_report:.3f}s")

    def _save_detailed_log(self, gnb_lookup):
        """Save per-step detailed log (dump-style) as CSV, one file per UE.

        Per-step top-3 cells by RSRP are saved (cell name, RSRP, SINR).
        """
        import csv

        if not self.step_data:
            return

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Build cell name mapping
        def _cell_name(cid):
            info = gnb_lookup.get(cid)
            if info:
                prefix = "eNB" if info.is_lte else "gNB"
                pci = getattr(info, 'pci', None)
                label = int(pci) if pci is not None else cid
                return f"{prefix}{label}"
            return f"C{cid}"

        def _cell_pci(cid):
            info = gnb_lookup.get(cid)
            if info is None:
                return ""
            pci = getattr(info, 'pci', None)
            return int(pci) if pci is not None else ""

        # Group step data by UE
        ue_ids_seen = sorted(set(sd.ue_id for sd in self.step_data), key=str)

        for idx, ue_id in enumerate(ue_ids_seen):
            ue_steps = [sd for sd in self.step_data if sd.ue_id == ue_id]

            # Fixed header: top-3 cells by RSRP, with gnb_id + pci columns
            # alongside the legacy "cell name" string for downstream tools.
            header = ['step', 'timestamp', 'ue_id',
                      'top1_cell', 'top1_gnb_id', 'top1_pci',
                      'top1_rsrp', 'top1_sinr', 'top1_rsrq',
                      'top2_cell', 'top2_gnb_id', 'top2_pci',
                      'top2_rsrp', 'top2_sinr', 'top2_rsrq',
                      'top3_cell', 'top3_gnb_id', 'top3_pci',
                      'top3_rsrp', 'top3_sinr', 'top3_rsrq',
                      'serving_cell', 'serving_gnb_id', 'serving_pci',
                      'serving_rsrp', 'serving_rsrq',
                      'mcs', 'bler',
                      'rrc_state', 'status',
                      'meas_events', 'timers', 'counters', 'events',
                      # A3 measurement-engine diagnostics (read-only inspection
                      # of MeasurementEngine.evaluate() internal state).
                      'a3_serv_filt_rsrq', 'a3_serv_filt_rsrp', 'a3_quantity',
                      'a3_filter_age_ms',
                      'a3_top1_cell', 'a3_top1_metric', 'a3_top1_ttt_ms',
                      'a3_top1_age', 'a3_top1_entering',
                      'a3_top2_cell', 'a3_top2_metric', 'a3_top2_ttt_ms',
                      'a3_top2_age', 'a3_top2_entering',
                      'a3_top3_cell', 'a3_top3_metric', 'a3_top3_ttt_ms',
                      'a3_top3_age', 'a3_top3_entering',
                      'a3_tracker_n',
                      # Gate-reason diagnostics (T1) — appended last so existing
                      # positional column access is unaffected.
                      'ho_suppress_reason', 'report_block_reason', 'report_gap_ms',
                      # Staged HO signaling flow (2026-06-11) — appended last
                      # so existing positional column access is unaffected.
                      'ho_stage', 'ho_cmd_decoded', 'target_rach_ok', 'target_rrc_ok',
                      # UL-block / RLM / velocity / L3-filter read-outs
                      # (2026-06-12) — appended last (positional access safe).
                      'ul_block_applied_rsrq', 'ul_block_applied_rsrp',
                      'ul_block_threshold_rsrq_db', 'ul_block_path',
                      'rlm_smoothed_sinr_db', 'rlm_bler_qout', 'rlm_bler_qin',
                      'ue_velocity_kmh', 'l3_filter_k']

            # Filename uses sorted-position index (0,1,...) so harness/tooling
            # that expects detailed_log_ue<int>.csv keeps working when ue_id
            # is a string (e.g. "00#4"). The ue_id column inside the file
            # still carries the actual identifier.
            log_path = os.path.join(self.config.output_dir, f"detailed_log_ue{idx}.csv")
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

                for sd in ue_steps:
                    row = [sd.step, f"{sd.timestamp:.4f}", sd.ue_id]

                    # Top-3 cells by RSRP — emit (cell_name, gnb_id, pci, rsrp, sinr, rsrq)
                    top3 = sorted(sd.all_rsrp.items(), key=lambda x: x[1], reverse=True)[:3]
                    for i in range(3):
                        if i < len(top3):
                            cid, rsrp = top3[i]
                            sinr = sd.all_sinr.get(cid)
                            rsrq = sd.all_rsrq.get(cid)
                            # Preserve full xlsx float precision — never round.
                            row.extend([_cell_name(cid), int(cid), _cell_pci(cid),
                                        repr(rsrp),
                                        repr(sinr) if sinr is not None else "",
                                        repr(rsrq) if rsrq is not None else ""])
                        else:
                            row.extend(["", "", "", "", "", ""])

                    # Serving cell — name + gnb_id + pci + rsrp + rsrq
                    row.append(_cell_name(sd.serving_cell))
                    row.append(int(sd.serving_cell)
                               if sd.serving_cell is not None else "")
                    row.append(_cell_pci(sd.serving_cell)
                               if sd.serving_cell is not None else "")
                    _srv_rsrp = sd.all_rsrp.get(sd.serving_cell) if sd.serving_cell is not None else None
                    row.append(repr(_srv_rsrp) if _srv_rsrp is not None else "")
                    _srv_rsrq = sd.all_rsrq.get(sd.serving_cell) if sd.serving_cell is not None else None
                    row.append(repr(_srv_rsrq) if _srv_rsrq is not None else "")

                    # MCS and BLER
                    row.append(sd.mcs)
                    row.append(f"{sd.bler:.6f}" if sd.bler is not None else "")

                    # RRC state + flags
                    state_str = sd.rrc_state
                    flags = []
                    if sd.ho_in_progress:
                        tgt = _cell_name(sd.target_cell) if sd.target_cell else "?"
                        flags.append(f"HO->{tgt}")
                    if sd.rlf_declared:
                        flags.append("RLF")
                    if flags:
                        state_str += "(" + ",".join(flags) + ")"
                    row.append(state_str)

                    # Status
                    row.append(sd.status)

                    # Measurement events (A2/A3/A5/B1/B2 TTT status)
                    meas_parts = []
                    for mname, ms in sd.meas_snapshot.items():
                        tgt = _cell_name(ms.get("target")) if ms.get("target") else ""
                        if ms.get("report"):
                            meas_parts.append(f"{mname}(REPORT->{tgt})")
                        elif ms.get("triggered"):
                            meas_parts.append(f"{mname}(TTT={ms['ttt_rem']:.0f}ms->{tgt})")
                    row.append(" | ".join(meas_parts))

                    # Timer states (T304/T310/T311)
                    timer_parts = []
                    for tname, ti in sd.timer_snapshot.items():
                        if ti.get("running"):
                            timer_parts.append(f"{tname}(rem={ti['remaining']:.0f}ms)")
                        elif ti.get("expired"):
                            timer_parts.append(f"{tname}(EXPIRED)")
                    row.append(" | ".join(timer_parts))

                    # Counter states (N310/N311)
                    counter_parts = []
                    for cname, ci in sd.counter_snapshot.items():
                        counter_parts.append(f"{cname}={ci['count']}/{ci['threshold']}")
                    row.append(" | ".join(counter_parts))

                    # Events this step
                    row.append(" | ".join(sd.events_this_step))

                    # A3 measurement-engine diagnostics (appended last). Keep
                    # missing values as empty strings so downstream CSV
                    # tooling parses cleanly.
                    def _opt(v):
                        return "" if v is None else v

                    row.append(repr(sd.a3_serv_filt_rsrq) if sd.a3_serv_filt_rsrq is not None else "")
                    row.append(repr(sd.a3_serv_filt_rsrp) if sd.a3_serv_filt_rsrp is not None else "")
                    row.append(sd.a3_quantity or "")
                    row.append(f"{sd.a3_filter_age_ms:.3f}" if sd.a3_filter_age_ms is not None else "")
                    # top1
                    row.append(_opt(sd.a3_top1_cell))
                    row.append(repr(sd.a3_top1_metric) if sd.a3_top1_metric is not None else "")
                    row.append(f"{sd.a3_top1_ttt_ms:.3f}" if sd.a3_top1_ttt_ms is not None else "")
                    row.append(_opt(sd.a3_top1_age))
                    row.append("" if sd.a3_top1_entering is None else int(bool(sd.a3_top1_entering)))
                    # top2
                    row.append(_opt(sd.a3_top2_cell))
                    row.append(repr(sd.a3_top2_metric) if sd.a3_top2_metric is not None else "")
                    row.append(f"{sd.a3_top2_ttt_ms:.3f}" if sd.a3_top2_ttt_ms is not None else "")
                    row.append(_opt(sd.a3_top2_age))
                    row.append("" if sd.a3_top2_entering is None else int(bool(sd.a3_top2_entering)))
                    # top3
                    row.append(_opt(sd.a3_top3_cell))
                    row.append(repr(sd.a3_top3_metric) if sd.a3_top3_metric is not None else "")
                    row.append(f"{sd.a3_top3_ttt_ms:.3f}" if sd.a3_top3_ttt_ms is not None else "")
                    row.append(_opt(sd.a3_top3_age))
                    row.append("" if sd.a3_top3_entering is None else int(bool(sd.a3_top3_entering)))
                    row.append(int(sd.a3_tracker_n))
                    # Gate-reason diagnostics (T1)
                    row.append(sd.ho_suppress_reason or "")
                    row.append(sd.report_block_reason or "")
                    row.append(f"{sd.report_gap_ms:.3f}" if sd.report_gap_ms is not None else "")
                    # Staged HO signaling flow (ho_stage str; bools as "1"/"0").
                    row.append(sd.ho_stage or "")
                    row.append("1" if sd.ho_cmd_decoded else "0")
                    row.append("1" if sd.target_rach_ok else "0")
                    row.append("1" if sd.target_rrc_ok else "0")
                    # UL-block / RLM / velocity / L3-filter read-outs (order
                    # MUST match the appended fieldnames above).
                    row.append(repr(sd.ul_block_applied_rsrq) if sd.ul_block_applied_rsrq is not None else "")
                    row.append(repr(sd.ul_block_applied_rsrp) if sd.ul_block_applied_rsrp is not None else "")
                    row.append(repr(sd.ul_block_threshold_rsrq_db) if sd.ul_block_threshold_rsrq_db is not None else "")
                    row.append(sd.ul_block_path or "")
                    row.append(repr(sd.rlm_smoothed_sinr_db) if sd.rlm_smoothed_sinr_db is not None else "")
                    row.append(f"{sd.rlm_bler_qout:.6f}" if sd.rlm_bler_qout is not None else "")
                    row.append(f"{sd.rlm_bler_qin:.6f}" if sd.rlm_bler_qin is not None else "")
                    row.append(f"{sd.ue_velocity_kmh:.3f}" if sd.ue_velocity_kmh is not None else "")
                    row.append(_opt(sd.l3_filter_k))

                    writer.writerow(row)

            self.logger.info(f"Detailed log saved to {log_path}")

    def _generate_text_report(self, ue_ids, state_machines):
        """Generate human-readable text report."""
        from datetime import datetime
        from collections import Counter

        report_path = os.path.join(self.config.output_dir, "simulation_report.txt")

        # Count events
        ho_starts = sum(1 for e in self.events if e.event_type == "HO_START")
        ho_completes = sum(1 for e in self.events if e.event_type == "HO_COMPLETE")
        ho_fails = sum(1 for e in self.events if e.event_type == "HO_FAIL")
        rlfs = sum(1 for e in self.events if e.event_type == "RLF")

        # Count HOF types from UEStateMachine classifications (ground truth)
        hof_counter = Counter()
        for ue_id_sm in ue_ids:
            sm = state_machines[ue_id_sm]
            for clf in sm.state.hof_classifications:
                hof_counter[clf.hof_type.value] += 1

        too_late = hof_counter.get("TOO_LATE", 0)
        too_early = hof_counter.get("TOO_EARLY", 0)
        wrong_cell = hof_counter.get("WRONG_CELL", 0)
        ping_pong = hof_counter.get("PING_PONG", 0)
        t304_expiry = hof_counter.get("T304_EXPIRY", 0)

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("RAILWAY NR HANDOVER SIMULATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"gNB Data: {os.path.basename(self.config.gnb_csv)}\n")
            f.write(f"UE Data: {os.path.basename(self.config.ue_csv)}\n")
            f.write(f"Number of UEs: {len(ue_ids)}\n")
            f.write(f"Channel Model: {self.config.channel_model_type}\n")
            if self.config.duration_s:
                f.write(f"Duration: {self.config.duration_s:.1f}s\n")
            f.write(f"MCS Index: {self.config.mcs_index}\n\n")

            f.write("STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Handovers: {ho_starts}\n")
            f.write(f"Successful: {ho_completes}\n")
            f.write(f"Failed: {ho_fails}\n")
            f.write(f"RLF Count: {rlfs}\n")

            # T310 running time
            t310_total = 0.0
            for uid in ue_ids:
                steps = sum(1 for sd in self.step_data if sd.ue_id == uid and "T310" in sd.timers)
                t310_s = steps * 0.02
                t310_total += t310_s
            f.write(f"T310 Running Time: {t310_total:.2f}s\n")

            # RRC_IDLE time
            idle_total = 0.0
            for uid in ue_ids:
                steps = sum(1 for sd in self.step_data if sd.ue_id == uid and sd.rrc_state.startswith("RRC_IDLE"))
                idle_total += steps * 0.02
            f.write(f"RRC_IDLE Time: {idle_total:.2f}s\n")

            # Non-OK ratio: fraction of steps where status != "OK"
            total_steps = len(self.step_data)
            non_ok_steps = sum(1 for sd in self.step_data if sd.status != "OK")
            non_ok_ratio = non_ok_steps / max(total_steps, 1)
            f.write(f"Non-OK Ratio: {non_ok_ratio:.6f}\n\n")

            f.write("HOF CLASSIFICATION (3GPP TS 38.300 §15.5)\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Case 1 - Too Late HO:       {too_late}\n")
            f.write(f"  Case 2 - Too Early HO:      {too_early}\n")
            f.write(f"  Case 3 - Wrong Cell HO:     {wrong_cell}\n")
            f.write(f"  Case 4 - Ping-Pong HO:      {ping_pong}   (excluded from primary total)\n")
            f.write(f"  Case 5 - T304 Expiry:       {t304_expiry}\n")
            f.write("  " + "─" * 30 + "\n")
            # Primary HOF total aligns with field `ho_fail` semantic
            # (vendor "Intra-LTE-HO Failure" fires only on actual radio link
            # failure during HO, never for ping-pong cell oscillation).
            hof_excl_pp = too_late + too_early + wrong_cell + t304_expiry
            hof_incl_pp = hof_excl_pp + ping_pong
            f.write(f"  Total HOF Events (excl PP):  {hof_excl_pp}\n")
            f.write(f"  Total HOF Events (incl PP):  {hof_incl_pp}\n\n")

            # Per-UE HOF details
            has_hof_details = False
            for ue_id_sm in ue_ids:
                sm = state_machines[ue_id_sm]
                clfs = [c for c in sm.state.hof_classifications if c.hof_type.value != "NONE"]
                if clfs:
                    if not has_hof_details:
                        f.write("HOF DETAILS\n")
                        f.write("-" * 70 + "\n")
                        has_hof_details = True
                    f.write(f"  UE{ue_id_sm}: {len(clfs)} HOF events\n")
                    for c in clfs:
                        f.write(f"    [{c.timestamp:.3f}s] {c.hof_type.value}: {c.cause}\n")
            if has_hof_details:
                f.write("\n")

            f.write("EVENTS (first 50)\n")
            f.write("-" * 70 + "\n")
            for i, e in enumerate(self.events[:50]):
                if e.event_type == "HO_START":
                    f.write(f"[{e.timestamp:.3f}s] UE{e.ue_id}: {e.event_type} "
                           f"(Cell {e.source_cell} → {e.target_cell})\n")
                elif e.event_type == "HO_COMPLETE":
                    f.write(f"[{e.timestamp:.3f}s] UE{e.ue_id}: {e.event_type} "
                           f"(Cell {e.source_cell} → {e.target_cell})\n")
                elif e.event_type == "HO_FAIL":
                    f.write(f"[{e.timestamp:.3f}s] UE{e.ue_id}: {e.event_type} "
                           f"(Cell {e.source_cell} → {e.target_cell}) [{e.details}]\n")
                elif e.event_type == "RLF":
                    detail_str = f" [{e.details}]" if e.details else ""
                    f.write(f"[{e.timestamp:.3f}s] UE{e.ue_id}: {e.event_type} "
                           f"in Cell {e.source_cell}{detail_str}\n")
                elif e.event_type == "RE_ESTABLISH":
                    detail_str = f" [HOF: {e.details}]" if e.details and e.details != "NONE" else ""
                    f.write(f"[{e.timestamp:.3f}s] UE{e.ue_id}: {e.event_type} "
                           f"(Cell {e.source_cell} → {e.target_cell}){detail_str}\n")
                elif e.event_type.startswith("MEAS_"):
                    f.write(f"[{e.timestamp:.3f}s] UE{e.ue_id}: {e.event_type}\n")
                else:
                    f.write(f"[{e.timestamp:.3f}s] UE{e.ue_id}: {e.event_type}\n")

            if len(self.events) > 50:
                f.write(f"... ({len(self.events) - 50} more events)\n")

            f.write("\n" + "=" * 70 + "\n")

        self.logger.info(f"Text report saved to {report_path}")

    def _classify_hof(self, ue_ids, state_machines):
        """Run HOF post-classifier on events."""
        try:
            from rrc.hof_classifier import HOFPostClassifier
        except ImportError:
            self.logger.warning("HOF classifier not available")
            return

        classifier = HOFPostClassifier()

        # Convert SimulationEvent dataclasses to dicts for classifier
        event_dicts = [
            {
                'timestamp': e.timestamp,
                'ue_id': e.ue_id,
                'event_type': e.event_type,
                'source_cell': e.source_cell,
                'target_cell': e.target_cell,
                'details': e.details or "",
            }
            for e in self.events
        ]

        try:
            results = classifier.classify_from_event_list(event_dicts)
            if results:
                self.logger.info("HOF Post-Classification Results:")
                for r in results:
                    self.logger.info(f"  UE{r.ue_id}: {r.hof_type} at t={r.timestamp:.3f}s - {r.cause}")
            else:
                self.logger.info("HOF Post-Classification: No failures detected")
        except Exception as e:
            self.logger.warning(f"HOF classification failed: {e}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NR Handover Simulation")
    parser.add_argument("--gnb-csv", default="ktx_ue_coordinate_downside_pci_rsrp.csv")
    parser.add_argument("--ue-csv", default="ktx_ue_coordinate_downside.csv")
    parser.add_argument("--channel-model", default="sionna_rt", choices=["statistical", "sionna_rt"])
    parser.add_argument("--scene-path", default=None)
    parser.add_argument("--sinr-map-dir", default="output/sinr_maps",
                        help="Directory with precomputed SINR maps (e.g. output/sinr_maps)")
    parser.add_argument("--rt-num-samples", type=float, default=1e6,
                        help="Sionna RT ray samples per source (default: 1e6, lower=faster)")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--time-window", type=str, default=None,
                        help="Time window as START,END seconds (e.g., 30.0,120.5)")
    parser.add_argument("--ue-subset", type=str, default=None, help="Comma-separated UE IDs")
    parser.add_argument("--mcs-index", type=int, default=-1,
                        help="MCS index for BLER (0-28, default 9=QPSK highest, -1=adaptive AMC)")
    parser.add_argument("--output-dir", default="output/railway_sim")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--exclude-lte", action="store_true")
    # RRC params
    parser.add_argument("--a3-offset", type=float, default=3.0)
    parser.add_argument("--hysteresis", type=float, default=2.0, help="Hysteresis (dB)")
    parser.add_argument("--ttt-ms", type=float, default=40.0)
    parser.add_argument("--gnb-ho-decision-delay-ms", type=float, default=0.0,
                        help="Vendor gNB HO Decision Algorithm delay (ms). When > 0, "
                             "buffers A3/A5/B1/B2 reports for this duration before "
                             "issuing the HO command; if target loses dominance "
                             "(by ≥ 0.5 dB margin) within the window, HO is withdrawn. "
                             "Default 0 = legacy byte-identical immediate HO.")
    parser.add_argument("--a2-threshold", type=float, default=-118.0)
    parser.add_argument("--a2-ttt", type=float, default=100.0, help="A2 TTT in ms")
    parser.add_argument("--a5-threshold1", type=float, default=-125.0, help="A5 serving threshold (dBm, per-RE RSRP)")
    parser.add_argument("--a5-threshold2", type=float, default=-115.0, help="A5 neighbor threshold (dBm, per-RE RSRP)")
    parser.add_argument("--a5-ttt", type=float, default=256.0, help="A5 TTT in ms")
    # Report interval params
    parser.add_argument("--a3-report-interval", type=float, default=240.0, help="A3 report interval (ms)")
    parser.add_argument("--a2-report-interval", type=float, default=240.0, help="A2 report interval (ms)")
    # Inter-RAT (B1/B2) params
    parser.add_argument("--b1-threshold", type=float, default=-125.0, help="B1 inter-RAT threshold (dBm, per-RE RSRP)")
    parser.add_argument("--b1-ttt", type=float, default=256.0, help="B1 TTT in ms")
    parser.add_argument("--b1-offset", type=float, default=0.0, help="B1 frequency offset (dB)")
    parser.add_argument("--b2-threshold1", type=float, default=-130.0, help="B2 serving threshold (dBm, per-RE RSRP)")
    parser.add_argument("--b2-threshold2", type=float, default=-125.0, help="B2 neighbor threshold (dBm, per-RE RSRP)")
    parser.add_argument("--b2-ttt", type=float, default=256.0, help="B2 TTT in ms")
    parser.add_argument("--b2-offset", type=float, default=0.0, help="B2 frequency offset (dB)")
    # RLF / Timer params
    parser.add_argument("--n310", type=int, default=10, help="N310 counter (out-of-sync to start T310)")
    parser.add_argument("--n311", type=int, default=2,
                        help="N311 counter (in-sync to stop T310). "
                             "HST default=2 (filters Doppler-induced IS blips).")
    parser.add_argument("--t310-ms", type=float, default=1000.0, help="T310 timer duration (ms)")
    parser.add_argument("--t304-ms", type=float, default=200.0, help="T304 HO execution timer (ms)")
    parser.add_argument("--t311-ms", type=float, default=1000.0, help="T311 RLF recovery timer (ms)")
    parser.add_argument("--t300-ms", type=float, default=1000.0, help="T300 RRC Connection Setup guard timer (ms)")
    # RACH params
    parser.add_argument("--preamble-initial-power", type=float, default=-104.0, help="Initial RACH preamble power (dBm)")
    parser.add_argument("--rach-max-attempts", type=int, default=10, help="Max RACH preamble attempts")
    parser.add_argument("--power-ramping-step", type=float, default=2.0, help="RACH power ramping step (dB)")
    # RLF RSRP thresholds (used with --no-sinr-rlf)
    parser.add_argument("--qout-rsrp", type=float, default=-150.0, help="Out-of-sync RSRP threshold (dBm, per-RE RSRP)")
    parser.add_argument("--qin-rsrp", type=float, default=-140.0, help="In-sync RSRP threshold (dBm, per-RE RSRP)")
    # SINR mode toggles
    # RF params
    parser.add_argument("--penetration-loss", type=float, default=15.0, help="Vehicle penetration loss (dB), default 25 for KTX train")
    parser.add_argument("--no-penetration-loss", action="store_true", help="Disable penetration loss (0 dB). Use when train surface is modeled in RT scene geometry")
    parser.add_argument("--max-path-length", type=float, default=1500.0, help="Max RT path length in meters (default 1500)")
    parser.add_argument("--surface-distortion-mean", type=float, default=0.0, help="Surface distortion mean (dB)")
    parser.add_argument("--surface-distortion-std", type=float, default=1.0, help="Surface distortion std (dB), 0=disable")
    parser.add_argument("--filter-coef", type=int, default=4, help="L3 filter coefficient (0=no filtering)")
    parser.add_argument("--no-sinr-rlf", action="store_true", help="Disable SINR-based RLF (use RSRP only)")
    parser.add_argument("--no-ul-sinr-rach", action="store_true", help="Disable UL SINR-based RACH (use power-based)")
    parser.add_argument("--rlf-mcs-index", type=int, default=-1,
                        help="MCS index for RLF detection BLER (3GPP TS 38.133 §8.1). "
                             "-1=AMC (default), 0-28=fixed MCS")
    # BLER mode
    parser.add_argument("--bler-mode", choices=["full_chain", "mockup", "sigmoid"],
                        default="mockup",
                        help="BLER computation mode: full_chain (Sionna CIR), mockup (CSV), sigmoid (analytical)")
    parser.add_argument("--bler-csv-dir", default="output/bler_curves",
                        help="Directory with bler_all_mcs.csv (for mockup mode)")
    parser.add_argument("--bler-sinr-offset", type=float, default=0.0,
                        help="Global SINR correction for mockup BLER (dB)")
    # Verbose output
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose per-step output")
    parser.add_argument("--no-verbose", action="store_false", dest="verbose", help="Disable verbose output")
    parser.add_argument("--verbose-interval", type=int, default=1, help="Print status every N timesteps")
    return parser.parse_args()


def main():
    """Main entry point."""
    # Reproducibility: seed every RNG source to a single fixed value so
    # baseline and trials are byte-identical across runs.
    _GLOBAL_SEED = 42
    import random as _r
    import numpy as np
    _r.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    try:
        import torch as _torch
        _torch.manual_seed(_GLOBAL_SEED)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(_GLOBAL_SEED)
    except Exception:
        pass

    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse --time-window into start_time_s and duration_s
    start_time_s = None
    duration_s = args.duration
    if args.time_window:
        parts = args.time_window.split(",")
        start_time_s = float(parts[0])
        duration_s = float(parts[1])  # end_time as max_duration

    # Build config from args
    config = SimulationConfig(
        gnb_csv=args.gnb_csv,
        ue_csv=args.ue_csv,
        channel_model_type=args.channel_model,
        scene_path=args.scene_path,
        sinr_map_dir=args.sinr_map_dir,
        rt_num_samples=args.rt_num_samples,
        duration_s=duration_s,
        start_time_s=start_time_s,
        ue_subset=[int(x) for x in args.ue_subset.split(",")] if args.ue_subset else None,
        mcs_index=args.mcs_index,
        output_dir=args.output_dir,
        log_level=args.log_level,
        exclude_lte=args.exclude_lte,
        a3_offset_db=args.a3_offset,
        hysteresis_db=args.hysteresis,
        ttt_ms=args.ttt_ms,
        a2_threshold_dbm=args.a2_threshold,
        a2_ttt_ms=args.a2_ttt,
        a5_threshold1_dbm=args.a5_threshold1,
        a5_threshold2_dbm=args.a5_threshold2,
        a5_ttt_ms=args.a5_ttt,
        b1_threshold_dbm=args.b1_threshold,
        b1_ttt_ms=args.b1_ttt,
        b1_offset_db=args.b1_offset,
        b2_threshold1_dbm=args.b2_threshold1,
        b2_threshold2_dbm=args.b2_threshold2,
        b2_ttt_ms=args.b2_ttt,
        b2_offset_db=args.b2_offset,
        n310=args.n310,
        n311=args.n311,
        t310_ms=args.t310_ms,
        t304_ms=args.t304_ms,
        t311_ms=args.t311_ms,
        t300_ms=args.t300_ms,
        preamble_initial_power_dbm=args.preamble_initial_power,
        preamble_tx_max=args.rach_max_attempts,
        power_ramping_step_db=args.power_ramping_step,
        qout_rsrp=args.qout_rsrp,
        qin_rsrp=args.qin_rsrp,
        penetration_loss_db=0.0 if args.no_penetration_loss else args.penetration_loss,
        max_path_length_m=args.max_path_length,
        surface_distortion_mean_db=args.surface_distortion_mean,
        surface_distortion_std_db=args.surface_distortion_std,
        a3_report_interval_ms=args.a3_report_interval,
        a2_report_interval_ms=args.a2_report_interval,
        filter_coef=args.filter_coef,
        use_sinr_for_rlf=not args.no_sinr_rlf,
        use_ul_sinr_for_rach=not args.no_ul_sinr_rach,
        rlf_mcs_index=args.rlf_mcs_index,
        bler_mode=args.bler_mode,
        bler_csv_dir=args.bler_csv_dir,
        bler_sinr_offset=args.bler_sinr_offset,
        verbose=args.verbose,
        verbose_interval=args.verbose_interval,
        gnb_ho_decision_delay_ms=args.gnb_ho_decision_delay_ms,
    )

    sim = NRSimulation(config)
    sim.run()


if __name__ == "__main__":
    main()
