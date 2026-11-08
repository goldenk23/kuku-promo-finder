"""CLI entrypoint for promo_finder.

Usage:
    python -m promo_finder parse <folder-or-zip>
    python -m promo_finder find <folder-or-zip> [--out outputs/run_name] [--episodes N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from promo_finder.finder import (
    generate_cross_episode_candidates,
    generate_single_episode_candidates,
    judge_cold_viewer,
)
from promo_finder.parse import Episode, format_timestamp, load_episodes
from promo_finder.validate import RuleValidator, ValidatedPromo


def cmd_parse(args: argparse.Namespace) -> int:
    """Parse episodes and output stats."""
    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source '{source}' does not exist.", file=sys.stderr)
        return 1

    print(f"Parsing episodes from: {source} ...\n")
    episodes = load_episodes(source)

    if not episodes:
        print("No episodes found.", file=sys.stderr)
        return 1

    total_events = 0
    kinds_total: dict[str, int] = {}
    durations: list[float] = []
    all_warnings: list[str] = []

    print(f"{'Ep':<4} {'Scenes':<7} {'Events':<7} {'Cam':<5} {'Act':<5} {'Dia':<5} {'Oth':<5} {'Duration':<10} {'Warnings'}")
    print("-" * 65)

    for ep in episodes:
        dur = ep.duration
        durations.append(dur)
        num_events = len(ep.events)
        total_events += num_events

        kinds_count: dict[str, int] = {}
        for ev in ep.events:
            kinds_count[ev.kind] = kinds_count.get(ev.kind, 0) + 1
            kinds_total[ev.kind] = kinds_total.get(ev.kind, 0) + 1

        all_warnings.extend(ep.warnings)
        warn_str = str(len(ep.warnings)) if ep.warnings else "-"

        print(
            f"{ep.number:<4} {len(ep.scenes):<7} {num_events:<7} "
            f"{kinds_count.get('camera', 0):<5} {kinds_count.get('action', 0):<5} "
            f"{kinds_count.get('dialogue', 0):<5} {kinds_count.get('other', 0):<5} "
            f"{format_timestamp(dur):<10} {warn_str}"
        )

    print("-" * 65)
    print("\n=== Summary Stats ===")
    print(f"Total episodes:      {len(episodes)}")
    print(f"Total events:        {total_events}")
    print(f"Events by kind:      camera={kinds_total.get('camera', 0)}, "
          f"action={kinds_total.get('action', 0)}, "
          f"dialogue={kinds_total.get('dialogue', 0)}, "
          f"other={kinds_total.get('other', 0)}")
    print(f"Episode durations:   min={format_timestamp(min(durations))} ({min(durations):.2f}s), "
          f"median={format_timestamp(statistics.median(durations))} ({statistics.median(durations):.2f}s), "
          f"max={format_timestamp(max(durations))} ({max(durations):.2f}s)")
    print(f"Total warnings:      {len(all_warnings)}")
    if all_warnings:
        print("Logged warnings:")
        for w in all_warnings:
            print(f"  * {w}")

    return 0


def cmd_find(args: argparse.Namespace) -> int:
    """Find and evaluate promos across episodes."""
    start_time = time.time()
    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source '{source}' does not exist.", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    print(f"Loading episodes from {source} ...")
    episodes = load_episodes(source)
    if args.episodes and args.episodes > 0:
        episodes = episodes[: args.episodes]
    print(f"Loaded {len(episodes)} episodes.")

    validator = RuleValidator(episodes)

    raw_candidates: list[dict[str, Any]] = []
    validated_promos: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    # 1. Pass A: Single Episode Candidates
    print(f"\n--- Pass A: Finding single-episode promo candidates (episodes 1 to {len(episodes)}) ---")
    dramatic_moments: list[dict[str, Any]] = []

    for ep in episodes:
        print(f"  Generating candidates for Episode {ep.number} ...", end=" ", flush=True)
        try:
            cands = generate_single_episode_candidates(
                ep, max_candidates=args.max_candidates_per_ep, use_cache=use_cache
            )
            print(f"got {len(cands)} proposals")
        except Exception as err:
            print(f"FAILED: {err}")
            cands = []

        for c_idx, c in enumerate(cands, 1):
            if "promo_id" not in c or not c["promo_id"]:
                c["promo_id"] = f"E{ep.number:02d}-P{c_idx}"
            c["pass"] = "pass_a"
            c["episode"] = ep.number
            raw_candidates.append(c)

            val_res = validator.validate_candidate(c, candidate_index=len(raw_candidates))
            if val_res.is_valid and val_res.promo:
                promo = val_res.promo
                # Save key moments for Pass B
                for seg in promo.segments:
                    for ev in seg.events:
                        if ev.kind == "dialogue":
                            dramatic_moments.append({
                                "event_id": ev.id,
                                "episode": ev.episode,
                                "kind": ev.kind,
                                "text": f"{ev.speaker}: {ev.text}",
                            })
                validated_promos.append({
                    "promo": promo,
                    "validation_repairs": promo.repairs_applied,
                })
            else:
                rejections.append({
                    "candidate": c,
                    "rule": val_res.rejection_rule,
                    "reason": val_res.rejection_reason,
                })

    # 2. Pass B: Cross-Episode Candidates (if multiple episodes available)
    if len(episodes) >= 2:
        print("\n--- Pass B: Finding cross-episode promo candidates ---")
        try:
            cross_cands = generate_cross_episode_candidates(
                episodes, dramatic_moments, max_candidates=args.max_cross_candidates, use_cache=use_cache
            )
            print(f"  Generated {len(cross_cands)} cross-episode proposals")
        except Exception as err:
            print(f"  Cross-episode generation failed: {err}")
            cross_cands = []

        for c_idx, c in enumerate(cross_cands, 1):
            if "promo_id" not in c or not c["promo_id"]:
                c["promo_id"] = f"CROSS-P{c_idx}"
            c["pass"] = "pass_b"
            raw_candidates.append(c)

            val_res = validator.validate_candidate(c, candidate_index=len(raw_candidates))
            if val_res.is_valid and val_res.promo:
                promo = val_res.promo
                validated_promos.append({
                    "promo": promo,
                    "validation_repairs": promo.repairs_applied,
                })
            else:
                rejections.append({
                    "candidate": c,
                    "rule": val_res.rejection_rule,
                    "reason": val_res.rejection_reason,
                })

    print(f"\n--- Validation Summary ---")
    print(f"Total raw candidates proposed: {len(raw_candidates)}")
    print(f"Valid promos passed:          {len(validated_promos)}")
    print(f"Rejected candidates:          {len(rejections)}")
    print("Rejections by rule:")
    for rule, count in validator.rejection_tally.items():
        if count > 0:
            print(f"  - {rule:<25}: {count}")

    # 3. Cold-Viewer Judge on Validated Promos
    print(f"\n--- Running Cold-Viewer Judge on {len(validated_promos)} validated promos ---")
    evaluated_promos: list[dict[str, Any]] = []

    for idx, item in enumerate(validated_promos, 1):
        promo: ValidatedPromo = item["promo"]
        print(f"  Judging [{idx}/{len(validated_promos)}] {promo.promo_id} ({promo.total_duration:.1f}s) ...", end=" ", flush=True)
        try:
            judge_res = judge_cold_viewer(promo, use_cache=use_cache)
            print(f"Hook={judge_res.get('hook_strength')}/5, Cliff={judge_res.get('cliffhanger_strength')}/5, Clarity={judge_res.get('standalone_clarity')}/5")
        except Exception as err:
            print(f"Judge error: {err}")
            judge_res = {
                "hook_strength": 3,
                "cliffhanger_strength": 3,
                "standalone_clarity": 3,
                "summary": "Evaluation failed",
                "open_question": "Unknown",
            }

        p_dict = promo.to_dict()
        p_dict["judge"] = judge_res
        p_dict["validation_repairs"] = item["validation_repairs"]
        evaluated_promos.append(p_dict)

    # Save outputs
    with open(out_dir / "candidates_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_candidates, f, indent=2)

    with open(out_dir / "validated.json", "w", encoding="utf-8") as f:
        json.dump(evaluated_promos, f, indent=2)

    with open(out_dir / "rejections.json", "w", encoding="utf-8") as f:
        json.dump(rejections, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nOutputs written to: {out_dir}/")
    print(f"  * candidates_raw.json ({len(raw_candidates)} items)")
    print(f"  * validated.json      ({len(evaluated_promos)} items)")
    print(f"  * rejections.json     ({len(rejections)} items)")
    print(f"Total pipeline wall time: {elapsed:.2f}s")

    # Print 3 sample promos for eyeball verification
    print("\n" + "=" * 70)
    print("3 SAMPLE PROMOS FOR EYEBALL VERIFICATION")
    print("=" * 70)
    for p in evaluated_promos[:3]:
        print(f"\nPROMO ID: {p['promo_id']} | Total Duration: {p['total_duration']}s")
        print(f"Hook ID: {p['hook_event_id']} (offset: {p['hook_offset']}s) | Cliffhanger ID: {p['cliffhanger_event_id']}")
        print(f"Segments:")
        for s in p["segments"]:
            print(f"  - Episode {s['episode']}: {s['start_ts']} to {s['end_ts']} ({s['duration']}s, events {s['start_event_id']}..{s['end_event_id']})")
        print(f"Reason: {p['reason']}")
        print(f"Cold-Viewer Evaluation:")
        j = p.get("judge", {})
        print(f"  Scores: Hook={j.get('hook_strength')}/5, Cliff={j.get('cliffhanger_strength')}/5, Clarity={j.get('standalone_clarity')}/5")
        print(f"  Summary: \"{j.get('summary')}\"")
        print(f"  Open Question: \"{j.get('open_question')}\"")
        if p["repairs_applied"]:
            print(f"  Repairs: {p['repairs_applied']}")
        print("-" * 70)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="promo_finder", description="Microdrama promo finder tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # parse subcommand
    parse_p = subparsers.add_parser("parse", help="Parse episode scripts and display summary statistics")
    parse_p.add_argument("source", help="Path to directory of episode .txt files or a .zip file")

    # find subcommand
    find_p = subparsers.add_parser("find", help="Find promo candidates and evaluate them")
    find_p.add_argument("source", help="Path to directory of episode .txt files or a .zip file")
    find_p.add_argument("--out", default="outputs/run_candidates", help="Output directory path")
    find_p.add_argument("--episodes", type=int, default=None, help="Limit number of episodes to process (for testing)")
    find_p.add_argument("--max-candidates-per-ep", type=int, default=4, help="Max candidates per episode in Pass A")
    find_p.add_argument("--max-cross-candidates", type=int, default=8, help="Max cross-episode candidates in Pass B")
    find_p.add_argument("--no-cache", action="store_true", help="Bypass LLM response cache")

    args = parser.parse_args()
    if args.command == "parse":
        return cmd_parse(args)
    if args.command == "find":
        return cmd_find(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
