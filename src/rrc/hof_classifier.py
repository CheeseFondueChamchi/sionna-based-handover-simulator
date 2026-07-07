"""
HOF Classifier — Post-Simulation Event Log Analyzer

Implements the 3GPP TS 38.300 §15.5 MRO decision tree to classify
handover failure events from a completed simulation's event log.

This module can be used:
  1. As a standalone post-processor over any event log (CSV/JSON)
  2. As a validation tool to cross-check real-time classifications
     produced by UERRCController / UEStateMachine during simulation

Reference:
  - TS 38.300 §15.5: Mobility Robustness Optimization (MRO)
  - TR 37.816: Study on MRO for NR (Rel-16)

HOF Classification Decision Tree
=================================

    Connection Failure Detected (RLF)
                  │
        ┌─────────┴──────────┐
        │                    │
  HO was initiated     No HO initiated
        │                    │
   ┌────┴────┐         Long time in cell?
   │         │               │
 T304      HO completed    Yes → TOO LATE
expired?       │
   │      Short time
  Yes     after HO?
   │         │
  T304   ┌───┴────┐
 EXPIRY  │        │
       Yes       No
         │        │
   Where does    TOO LATE
   UE re-estab?
        │
   ┌────┼────┐
Source Target 3rd
 cell   cell  Cell
   │     │     │
 TOO   (rare) WRONG
EARLY         CELL

  Additional (no RLF required):
  A→B→A within Tpp → PING-PONG

Usage:
    from hof_classifier import HOFPostClassifier

    classifier = HOFPostClassifier(
        tstore_ue_cntxt_ms=1000.0,
        tpp_ms=1000.0
    )
    results = classifier.classify_from_event_list(event_list)
    classifier.print_summary(results)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HOFResult:
    """Single HOF classification result from post-processing."""
    hof_type: str              # TOO_LATE, TOO_EARLY, WRONG_CELL, PING_PONG, T304_EXPIRY
    timestamp: float           # When the failure was classified
    ue_id: int                 # UE that experienced the failure
    rlf_cell_id: int = -1      # Cell where RLF occurred
    reest_cell_id: int = -1    # Cell where UE re-established (-1 if none)
    last_ho_source: int = -1   # Source of most recent HO
    last_ho_target: int = -1   # Target of most recent HO
    time_since_last_ho_ms: float = -1  # Time between last HO and RLF
    cause: str = ""            # Human-readable explanation


# ═══════════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════════

class HOFPostClassifier:
    """
    Post-simulation HOF classifier.

    Replays the event log and applies the 3GPP TS 38.300 §15.5
    decision tree to classify every RLF as one of:
      Case 1: Too Late HO
      Case 2: Too Early HO
      Case 3: Wrong Cell HO
      Case 5: T304 Expiry (HO Execution Failure)

    Also detects:
      Case 4: Ping-Pong HO (no RLF, two successful HOs A→B→A within Tpp)

    Parameters:
        tstore_ue_cntxt_ms: Threshold (ms) for "short time after HO".
            Per 3GPP, this is network-configured.  Typical: 1000ms.
        tpp_ms: Threshold (ms) for ping-pong detection.
            Typical: 1000ms.
        count_ping_pong_as_hof: If False (default), Case 4 (Ping-Pong) is
            STILL detected and reported in the per-case breakdown but is
            EXCLUDED from the "Total HOF Events" line. This aligns the
            sim's headline HOF metric with the field's `ho_fail` semantic
            (vendor `Intra-LTE-HO Failure` only fires for actual radio
            link failure during HO — not for frequent A→B→A cell changes).
            Set True to restore legacy behavior (sum all 5 cases).
        tstore_max_ms: Upper bound (ms) for the "Too Late" classification
            window. RLFs occurring more than this many ms after the last
            successful HO are treated as plain radio-link failures, not
            mobility (Too Late) failures. Default 5000ms. Per 3GPP TS
            38.300 §15.5 the analyzer-side Tstore window is network
            configurable; typical values lie in 1-10s. Setting too large
            mis-attributes environmental RLFs to mobility.
        require_prior_ho_for_too_late: If True (default), an RLF with NO
            prior successful HO is classified as NONE (not Too Late). The
            UE never had a chance to hand over, so the failure is not a
            mobility failure.
    """

    def __init__(self,
                 tstore_ue_cntxt_ms: float = 1000.0,
                 tpp_ms: float = 1000.0,
                 count_ping_pong_as_hof: bool = False,
                 tstore_max_ms: float = 5000.0,
                 require_prior_ho_for_too_late: bool = True):
        """Configure the post-hoc HOF classifier (TS 38.300 §15.5 MRO).

        Args:
            tstore_ue_cntxt_ms: window (ms) for "shortly after a HO" rule;
                Tstore_UE_cntxt per spec (default 1000ms).
            tpp_ms: ping-pong detection window for A→B→A return (default
                1000ms).
            count_ping_pong_as_hof: include PING_PONG in field-aligned HOF
                counts. False matches FIELD's `ho_fail` (PP not counted).
            tstore_max_ms: upper bound on "shortly after" lookback; RLFs
                beyond this become TOO_LATE rather than TOO_EARLY/WRONG_CELL
                (default 5000ms).
            require_prior_ho_for_too_late: when True, an RLF without any
                prior HO in history is classified as RLM-only (TOO_LATE
                requires a preceding HO).

        Side effects:
            Stores all args as attributes. No state allocated.
        """
        self.tstore_ue_cntxt_ms = tstore_ue_cntxt_ms
        self.tpp_ms = tpp_ms
        self.count_ping_pong_as_hof = count_ping_pong_as_hof
        self.tstore_max_ms = tstore_max_ms
        self.require_prior_ho_for_too_late = require_prior_ho_for_too_late

    def classify_from_event_list(self, events: List[Dict]) -> List[HOFResult]:
        """
        Classify HOF events from a simulation event log.

        Args:
            events: List of event dicts with keys:
                - timestamp (float)
                - ue_id (int)
                - event_type (str): HO_START, HO_COMPLETE, HO_FAIL, RLF,
                  RRC_CONNECTED, REEST_COMPLETE, T304_EXPIRE, etc.
                - source_cell (int, optional)
                - target_cell (int, optional)
                - details (str, optional)

        Returns:
            List of HOFResult classifications.
        """
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', 0))

        # Per-UE state tracking
        ue_ho_history: Dict[int, List[Dict]] = {}    # Recent HOs per UE
        ue_rlf_context: Dict[int, Dict] = {}          # Active RLF info per UE
        ue_serving_cell: Dict[int, int] = {}           # Current serving cell per UE
        ue_ho_in_progress: Dict[int, Dict] = {}        # Active HO per UE

        results: List[HOFResult] = []

        for event in sorted_events:
            ts = event.get('timestamp', 0)
            ue = event.get('ue_id', -1)
            etype = event.get('event_type', '')
            src = event.get('source_cell', -1)
            tgt = event.get('target_cell', -1)

            # Initialize per-UE structures
            if ue not in ue_ho_history:
                ue_ho_history[ue] = []
            if ue not in ue_serving_cell:
                ue_serving_cell[ue] = src if src and src != -1 else -1

            # ── Track HO lifecycle ──
            if etype == 'HO_START':
                ue_ho_in_progress[ue] = {
                    'timestamp': ts,
                    'source_cell': src,
                    'target_cell': tgt
                }

            elif etype == 'HO_COMPLETE':
                ho_entry = {
                    'timestamp': ts,
                    'source_cell': src,
                    'target_cell': tgt,
                    'success': True
                }
                ue_ho_history[ue].append(ho_entry)
                ue_serving_cell[ue] = tgt
                ue_ho_in_progress.pop(ue, None)

                # ── Case 4: Ping-pong detection ──
                pp = self._check_ping_pong(ue, ts, src, tgt, ue_ho_history[ue])
                if pp:
                    results.append(pp)

            elif etype == 'HO_FAIL':
                ho_entry = {
                    'timestamp': ts,
                    'source_cell': src,
                    'target_cell': tgt,
                    'success': False,
                    'reason': event.get('details', '')
                }
                ue_ho_history[ue].append(ho_entry)
                ue_ho_in_progress.pop(ue, None)

            # ── RLF: Store context for deferred classification ──
            elif etype == 'RLF':
                ho_active = ue in ue_ho_in_progress
                ue_rlf_context[ue] = {
                    'rlf_time': ts,
                    'rlf_cell': src if src and src != -1 else ue_serving_cell.get(ue, -1),
                    'ho_was_in_progress': ho_active,
                    'ho_target': (ue_ho_in_progress[ue]['target_cell']
                                  if ho_active else -1)
                }
                ue_ho_in_progress.pop(ue, None)

            # ── T304 Expire: Also store context ──
            elif etype in ('T304_EXPIRE', 'T304_EXPIRY'):
                ue_rlf_context[ue] = {
                    'rlf_time': ts,
                    'rlf_cell': ue_serving_cell.get(ue, -1),
                    'ho_was_in_progress': True,
                    'ho_target': tgt if tgt and tgt != -1 else -1
                }

            # ── Re-establishment complete: Classify! ──
            elif etype in ('RE_ESTABLISH', 'REEST_COMPLETE', 'RRC_CONNECTED', 'RRC_STATE_CHANGE'):
                # Only classify if there's a pending RLF context
                if ue in ue_rlf_context:
                    reest_cell = (tgt if tgt and tgt != -1
                                  else src if src and src != -1
                                  else ue_serving_cell.get(ue, -1))

                    # Sometimes RRC_CONNECTED carries the new serving cell
                    # in source_cell rather than target_cell
                    if reest_cell == -1:
                        continue

                    hof = self._classify(
                        ue_id=ue,
                        current_time=ts,
                        reest_cell=reest_cell,
                        rlf_ctx=ue_rlf_context[ue],
                        ho_history=ue_ho_history[ue]
                    )
                    if hof and hof.hof_type != "NONE":
                        results.append(hof)

                    ue_serving_cell[ue] = reest_cell
                    del ue_rlf_context[ue]

        # Handle any remaining RLF contexts (T311 expired, no re-establishment)
        for ue, ctx in ue_rlf_context.items():
            hof_type = "T304_EXPIRY" if ctx.get('ho_was_in_progress') else "TOO_LATE"
            results.append(HOFResult(
                hof_type=hof_type,
                timestamp=ctx['rlf_time'],
                ue_id=ue,
                rlf_cell_id=ctx['rlf_cell'],
                reest_cell_id=-1,
                cause=f"RLF with no re-establishment (T311 likely expired)"
            ))

        return results

    def _classify(self, ue_id: int, current_time: float,
                  reest_cell: int, rlf_ctx: Dict,
                  ho_history: List[Dict]) -> Optional[HOFResult]:
        """
        Apply the 3GPP TS 38.300 §15.5 (MRO) decision tree on an RLF → re-estab.

        Standard 3GPP connection-failure cases & how we map them:
          • Too-Early  : RLF shortly (dt ≤ tstore_ue_cntxt) after a HO, re-estab
                         at SOURCE.                              ← matches spec.
          • Wrong-Cell : RLF shortly after a HO, re-estab at a 3rd cell.
                                                                 ← matches spec.
          • Too-Late   : RLF a while after the last HO. We gate purely on timing
                         (tstore_ue_cntxt < dt ≤ tstore_max). DEVIATION: the spec
                         also implies re-establishment in a *different* cell than
                         the one that failed; we do not check reest≠rlf_cell
                         (in practice it always holds, but the check is absent).
          • T304-Expiry: a HO-procedure failure (T304 expiry / RACH-to-target
                         fail). DEVIATION: 3GPP folds this INTO Too-Early/Wrong-
                         Cell by where the UE re-establishes; we break it out as
                         its own case (Case 5) because the field log distinguishes
                         it. Both are summed into "HOF (excl PP)" for the field
                         `ho_fail` comparison, so the headline still lines up.
        Ping-pong (Case 4) is handled separately and is NOT a spec connection-
        failure case (see _check_ping_pong).
        """
        rlf_time = rlf_ctx['rlf_time']
        rlf_cell = rlf_ctx['rlf_cell']
        ho_was_active = rlf_ctx.get('ho_was_in_progress', False)

        # Case 5: T304 Expiry
        if ho_was_active:
            ho_target = rlf_ctx.get('ho_target', -1)
            return HOFResult(
                hof_type="T304_EXPIRY",
                timestamp=current_time,
                ue_id=ue_id,
                rlf_cell_id=rlf_cell,
                reest_cell_id=reest_cell,
                last_ho_source=rlf_cell,
                last_ho_target=ho_target,
                time_since_last_ho_ms=0,
                cause=(f"T304 expired during HO {rlf_cell}->{ho_target}, "
                       f"re-estab at {reest_cell}")
            )

        # Find last successful HO
        last_ho = None
        for entry in reversed(ho_history):
            if entry.get('success', False):
                last_ho = entry
                break

        if last_ho is None:
            # No prior HO. With require_prior_ho_for_too_late=True (default),
            # treat as plain RLF (NONE) — UE never had the opportunity to
            # hand over, so this isn't a mobility failure.
            if self.require_prior_ho_for_too_late:
                return HOFResult(
                    hof_type="NONE",
                    timestamp=current_time,
                    ue_id=ue_id,
                    rlf_cell_id=rlf_cell,
                    reest_cell_id=reest_cell,
                    cause=("RLF with no prior successful HO — treated as "
                           "plain radio-link failure (not a mobility failure)")
                )
            return HOFResult(
                hof_type="TOO_LATE",
                timestamp=current_time,
                ue_id=ue_id,
                rlf_cell_id=rlf_cell,
                reest_cell_id=reest_cell,
                cause=f"RLF with no prior successful HO, re-estab at {reest_cell}"
            )

        dt_ms = (rlf_time - last_ho['timestamp']) * 1000.0
        ho_src = last_ho['source_cell']
        ho_tgt = last_ho['target_cell']

        if dt_ms <= self.tstore_ue_cntxt_ms:
            # Short time after HO
            if reest_cell == ho_src:
                return HOFResult(
                    hof_type="TOO_EARLY",
                    timestamp=current_time,
                    ue_id=ue_id,
                    rlf_cell_id=rlf_cell,
                    reest_cell_id=reest_cell,
                    last_ho_source=ho_src,
                    last_ho_target=ho_tgt,
                    time_since_last_ho_ms=dt_ms,
                    cause=(f"RLF {dt_ms:.0f}ms after HO {ho_src}->{ho_tgt}, "
                           f"re-estab back to source {reest_cell}")
                )
            elif reest_cell == ho_tgt:
                # Re-established at target — not a failure (coverage recovered)
                return HOFResult(
                    hof_type="NONE",
                    timestamp=current_time,
                    ue_id=ue_id,
                    rlf_cell_id=rlf_cell,
                    reest_cell_id=reest_cell,
                    last_ho_source=ho_src,
                    last_ho_target=ho_tgt,
                    time_since_last_ho_ms=dt_ms,
                    cause="Re-estab at target (recovered)"
                )
            else:
                return HOFResult(
                    hof_type="WRONG_CELL",
                    timestamp=current_time,
                    ue_id=ue_id,
                    rlf_cell_id=rlf_cell,
                    reest_cell_id=reest_cell,
                    last_ho_source=ho_src,
                    last_ho_target=ho_tgt,
                    time_since_last_ho_ms=dt_ms,
                    cause=(f"RLF {dt_ms:.0f}ms after HO {ho_src}->{ho_tgt}, "
                           f"re-estab at 3rd cell {reest_cell}")
                )
        else:
            # RLF more than Tstore_ue_cntxt after last HO. If it is ALSO
            # beyond Tstore_max, the failure is not plausibly a mobility
            # (Too Late) issue — by then the cell condition has had ample
            # time to stabilize and the RLF is environmental. Treat as
            # plain RLF (NONE) so it isn't double-counted as a HOF.
            if dt_ms > self.tstore_max_ms:
                return HOFResult(
                    hof_type="NONE",
                    timestamp=current_time,
                    ue_id=ue_id,
                    rlf_cell_id=rlf_cell,
                    reest_cell_id=reest_cell,
                    last_ho_source=ho_src,
                    last_ho_target=ho_tgt,
                    time_since_last_ho_ms=dt_ms,
                    cause=(f"RLF {dt_ms:.0f}ms after last HO "
                           f"(>tstore_max={self.tstore_max_ms:.0f}ms) — "
                           f"treated as plain RLF, not Too Late HOF")
                )
            return HOFResult(
                hof_type="TOO_LATE",
                timestamp=current_time,
                ue_id=ue_id,
                rlf_cell_id=rlf_cell,
                reest_cell_id=reest_cell,
                last_ho_source=ho_src,
                last_ho_target=ho_tgt,
                time_since_last_ho_ms=dt_ms,
                cause=(f"RLF {dt_ms:.0f}ms after last HO "
                       f"(>{self.tstore_ue_cntxt_ms:.0f}ms), re-estab at {reest_cell}")
            )

    def _check_ping_pong(self, ue_id: int, current_time: float,
                          source: int, target: int,
                          ho_history: List[Dict]) -> Optional[HOFResult]:
        """Check for ping-pong (A→B→A within Tpp).

        Detected on a SUCCESSFUL HO_COMPLETE (no RLF): the UE handed S→T while a
        prior successful T→S HO sits within tpp_ms (default 1000 ms) → it bounced
        back. STANDARD NOTE: ping-pong is NOT one of the 3GPP MRO connection-
        failure cases (Too-Late/Too-Early/Wrong-Cell in TS 38.300 §15.5) — there
        is no RLF, the link never failed. It is a separate mobility/"unnecessary
        HO" KPI. This is why it is reported separately and, by default
        (count_ping_pong_as_hof=False), EXCLUDED from the HOF headline so the
        sim's "HOF (excl PP)" lines up with the field log's `ho_fail` (which
        also excludes ping-pong). Keep count_ping_pong_as_hof=False.
        """
        tpp_s = self.tpp_ms / 1000.0

        # Look for a reverse HO (target→source) in recent history
        for entry in reversed(ho_history[:-1]):  # Exclude current entry
            if not entry.get('success', False):
                continue
            dt = current_time - entry['timestamp']
            if dt > tpp_s:
                break
            if (entry['source_cell'] == target and
                    entry['target_cell'] == source):
                return HOFResult(
                    hof_type="PING_PONG",
                    timestamp=current_time,
                    ue_id=ue_id,
                    last_ho_source=source,
                    last_ho_target=target,
                    time_since_last_ho_ms=dt * 1000.0,
                    cause=(f"Ping-pong: {entry['source_cell']}->{entry['target_cell']} "
                           f"then {source}->{target} within {dt*1000:.0f}ms")
                )
        return None

    # ═══════════════════════════════════════════════════════════════════
    # Reporting
    # ═══════════════════════════════════════════════════════════════════

    def print_summary(self, results: List[HOFResult]):
        """Print HOF classification summary to stdout.

        By default (count_ping_pong_as_hof=False) the headline
        "Total HOF Events" excludes Case 4 (Ping-Pong) so it lines up
        with field vendor `Intra-LTE-HO Failure` semantics. The Case 4
        count is still printed in the breakdown.
        """
        if not results:
            print("\n  HOF Classification: No failures detected.\n")
            return

        counts = Counter(r.hof_type for r in results if r.hof_type != "NONE")
        total_all = sum(counts.values())
        total_excl_pp = total_all - counts.get('PING_PONG', 0)
        headline = total_all if self.count_ping_pong_as_hof else total_excl_pp

        print("\n" + "=" * 70)
        print("  HOF CLASSIFICATION SUMMARY (3GPP TS 38.300 §15.5)")
        print("=" * 70)
        print(f"  Case 1 — Too Late HO:       {counts.get('TOO_LATE', 0):>4}")
        print(f"  Case 2 — Too Early HO:      {counts.get('TOO_EARLY', 0):>4}")
        print(f"  Case 3 — Wrong Cell HO:     {counts.get('WRONG_CELL', 0):>4}")
        print(f"  Case 4 — Ping-Pong HO:      {counts.get('PING_PONG', 0):>4}"
              + ("" if self.count_ping_pong_as_hof
                 else "  (excluded from total — field-aligned)"))
        print(f"  Case 5 — T304 Expiry:       {counts.get('T304_EXPIRY', 0):>4}")
        print(f"  {'─' * 35}")
        if self.count_ping_pong_as_hof:
            print(f"  Total HOF Events:            {headline:>4}")
        else:
            print(f"  Total HOF Events (excl PP):  {headline:>4}")
            print(f"  Total HOF Events (incl PP):  {total_all:>4}")
        print("=" * 70)

        print("\n  Detail:")
        for r in results:
            if r.hof_type != "NONE":
                print(f"    [{r.timestamp:.3f}s] UE{r.ue_id}: "
                      f"{r.hof_type:12s} | {r.cause}")
        print()

    @staticmethod
    def get_mro_recommendations(results: List[HOFResult]) -> List[str]:
        """
        Generate MRO corrective action recommendations based on HOF results.

        Returns a list of human-readable recommendations per 3GPP TS 38.300.
        """
        counts = Counter(r.hof_type for r in results if r.hof_type != "NONE")
        recs = []

        if counts.get('TOO_LATE', 0) > 0:
            recs.append(
                f"[Too Late × {counts['TOO_LATE']}] "
                "Decrease A3 Offset and/or TTT to trigger HO earlier. "
                "Consider increasing CIO for frequently-used targets.")

        if counts.get('TOO_EARLY', 0) > 0:
            recs.append(
                f"[Too Early × {counts['TOO_EARLY']}] "
                "Increase A3 Offset and/or TTT to require stronger/sustained "
                "target signal. Decrease CIO for problematic targets.")

        if counts.get('WRONG_CELL', 0) > 0:
            # Identify specific wrong cells
            wrong_cells = set()
            for r in results:
                if r.hof_type == 'WRONG_CELL':
                    wrong_cells.add((r.last_ho_target, r.reest_cell_id))
            cell_strs = [f"{t}->(should be {c})" for t, c in wrong_cells]
            recs.append(
                f"[Wrong Cell × {counts['WRONG_CELL']}] "
                f"Review NRT entries for targets: {', '.join(cell_strs)}. "
                "Consider V-ANC proactive candidate filtering.")

        if counts.get('PING_PONG', 0) > 0:
            recs.append(
                f"[Ping-Pong × {counts['PING_PONG']}] "
                "Increase A3 hysteresis and/or TTT. "
                "Consider direction-specific candidate lists (V-ANC).")

        if counts.get('T304_EXPIRY', 0) > 0:
            recs.append(
                f"[T304 Expiry × {counts['T304_EXPIRY']}] "
                "Check target cell RACH configuration and capacity. "
                "Ensure target RSRP is sufficient at HO trigger time.")

        return recs


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-simulation HOF classifier (3GPP TS 38.300 §15.5)")
    parser.add_argument("event_log", help="Path to JSON event log file")
    parser.add_argument("--tstore", type=float, default=1000.0,
                        help="Tstore_UE_cntxt threshold in ms (default: 1000)")
    parser.add_argument("--tpp", type=float, default=1000.0,
                        help="Ping-pong threshold in ms (default: 1000)")
    args = parser.parse_args()

    with open(args.event_log) as f:
        data = json.load(f)

    # Support both raw event list and full results dict
    if isinstance(data, list):
        event_list = data
    elif isinstance(data, dict) and 'events' in data:
        event_list = data['events']
    else:
        print("Error: cannot find event list in JSON file")
        exit(1)

    classifier = HOFPostClassifier(
        tstore_ue_cntxt_ms=args.tstore,
        tpp_ms=args.tpp
    )
    results = classifier.classify_from_event_list(event_list)
    classifier.print_summary(results)

    recs = classifier.get_mro_recommendations(results)
    if recs:
        print("MRO RECOMMENDATIONS:")
        print("-" * 70)
        for rec in recs:
            print(f"  • {rec}")
        print()
