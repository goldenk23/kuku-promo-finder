"""CLI entrypoint for promo_finder.

Usage:
    python -m promo_finder parse <folder-or-zip>
    python -m promo_finder find <folder-or-zip> [--out outputs/run_name] [--episodes N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from promo_finder.finder import (
    generate_cross_episode_candidates,
    generate_single_episode_candidates,
    judge_cold_viewer,
)
from promo_finder.parse import Episode, format_timestamp, load_episodes
from promo_finder.quality import (
    generate_human_review_csv,
    parse_usage_summary,
    run_judge_sanity_tests,
    run_stability_check,
    write_quality_report,
    write_run_summary,
)
from promo_finder.rank import export_ranked_promos, rank_and_deduplicate_promos
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
    events_by_id = validator.events_by_id

    raw_candidates: list[dict[str, Any]] = []
    validated_promos: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    # 1. Pass A: Single Episode Candidates (Parallelized with ThreadPoolExecutor)
    print(f"\n--- Pass A: Finding single-episode promo candidates (episodes 1 to {len(episodes)}) ---")
    ep_results: dict[int, list[dict[str, Any]]] = {}

    def _fetch_ep_candidates(ep: Episode) -> tuple[int, list[dict[str, Any]]]:
        try:
            cands = generate_single_episode_candidates(
                ep, max_candidates=args.max_candidates_per_ep, use_cache=use_cache
            )
            return ep.number, cands
        except Exception as err:
            print(f"  [Ep {ep.number} FAILED: {err}]", flush=True)
            return ep.number, []

    workers = min(args.workers, len(episodes))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_ep_candidates, ep): ep.number for ep in episodes}
        for f in as_completed(futures):
            ep_num, cands = f.result()
            ep_results[ep_num] = cands
            print(f"  Ep {ep_num:02d}: got {len(cands)} candidates", flush=True)

    # Process and validate in deterministic episode order
    dramatic_moments: list[dict[str, Any]] = []
    for ep in episodes:
        cands = ep_results.get(ep.number, [])
        for c_idx, c in enumerate(cands, 1):
            if "promo_id" not in c or not c["promo_id"]:
                c["promo_id"] = f"E{ep.number:02d}-P{c_idx}"
            c["pass"] = "pass_a"
            c["episode"] = ep.number
            raw_candidates.append(c)

            val_res = validator.validate_candidate(c, candidate_index=len(raw_candidates))
            if val_res.is_valid and val_res.promo:
                promo = val_res.promo
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

    # 2. Pass B: Cross-Episode Candidates
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

    # 3. Cold-Viewer Judge on Validated Promos (Parallelized)
    print(f"\n--- Running Cold-Viewer Judge on {len(validated_promos)} validated promos ---")
    judged_dict: dict[str, dict[str, Any]] = {}

    def _judge_single(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        promo: ValidatedPromo = item["promo"]
        try:
            j_res = judge_cold_viewer(promo, use_cache=use_cache)
            return promo.promo_id, j_res
        except Exception as err:
            logger.warning("Judge failed for %s: %s", promo.promo_id, err)
            fallback = {
                "hook_strength": 3,
                "cliffhanger_strength": 3,
                "standalone_clarity": 3,
                "summary": "Evaluation failed",
                "open_question": "Unknown",
            }
            return promo.promo_id, fallback

    judge_workers = min(args.workers, max(1, len(validated_promos)))
    with ThreadPoolExecutor(max_workers=judge_workers) as executor:
        f_map = {executor.submit(_judge_single, item): item["promo"].promo_id for item in validated_promos}
        for f in as_completed(f_map):
            p_id, j_res = f.result()
            judged_dict[p_id] = j_res
            print(f"  Judged {p_id}: Hook={j_res.get('hook_strength')}/5, Cliff={j_res.get('cliffhanger_strength')}/5, Clarity={j_res.get('standalone_clarity')}/5", flush=True)

    evaluated_promos: list[dict[str, Any]] = []
    for item in validated_promos:
        promo: ValidatedPromo = item["promo"]
        p_dict = promo.to_dict()
        p_dict["judge"] = judged_dict.get(promo.promo_id, {
            "hook_strength": 3, "cliffhanger_strength": 3, "standalone_clarity": 3,
            "summary": "Evaluation", "open_question": "What next?"
        })
        p_dict["validation_repairs"] = item["validation_repairs"]
        evaluated_promos.append(p_dict)

    # Save raw and intermediate sets
    with open(out_dir / "candidates_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_candidates, f, indent=2)

    with open(out_dir / "validated.json", "w", encoding="utf-8") as f:
        json.dump(evaluated_promos, f, indent=2)

    with open(out_dir / "rejections.json", "w", encoding="utf-8") as f:
        json.dump(rejections, f, indent=2)

    # 4. Ranking & Selection
    print("\n--- Ranking, De-duplicating, and Selecting Top Promos ---")
    ranked_promos = rank_and_deduplicate_promos(evaluated_promos, top_n=args.top_n)
    export_ranked_promos(ranked_promos, out_dir, events_by_id=events_by_id)

    # Copy usage.jsonl to run directory
    repo_usage = Path("outputs/usage.jsonl")
    if repo_usage.exists():
        shutil.copy(repo_usage, out_dir / "usage.jsonl")

    # 5. Quality & Stability
    print("\n--- Running Adversarial Quality Checks & Stability Benchmarks ---")
    print("  Running Judge Sanity Tests (5 flawed probe clips) ...")
    sanity_results = run_judge_sanity_tests(episodes, use_cache=use_cache)
    passed_probes = sum(1 for s in sanity_results if s["passed_sanity"])
    print(f"  Sanity Probe Result: {passed_probes}/5 flawed clips correctly flagged by judge.")

    print("  Running Candidate Stability Check (cache bypassed) ...")
    stability_res = run_stability_check(episodes, evaluated_promos, sample_episodes=min(3, len(episodes)))
    print(f"  Stability Overlap: {stability_res['overlap_percentage']}% temporal match across runs.")

    generate_human_review_csv(ranked_promos, out_dir / "human_review.csv", events_by_id)
    write_quality_report(
        out_dir=out_dir,
        validator_tally=validator.rejection_tally,
        total_candidates=len(raw_candidates),
        sanity_results=sanity_results,
        stability_results=stability_res,
    )

    elapsed = time.time() - start_time
    usage_info = parse_usage_summary(out_dir / "usage.jsonl")
    total_events = sum(len(ep.events) for ep in episodes)

    write_run_summary(
        out_dir=out_dir,
        total_episodes=len(episodes),
        total_events=total_events,
        total_candidates=len(raw_candidates),
        total_valid=len(evaluated_promos),
        wall_time=elapsed,
        usage=usage_info,
    )

    print(f"\nAll outputs successfully generated in: {out_dir}/")
    print(f"  * promos_ranked.json, promos_ranked.csv, promos_ranked.md")
    print(f"  * quality_report.md, run_summary.md, human_review.csv")
    print(f"  * candidates_raw.json, validated.json, rejections.json, usage.jsonl")
    print(f"Total tokens used: {usage_info['total_tokens']:,} (Cost estimate: ${usage_info['cost_usd']:.4f})")
    print(f"Wall time: {elapsed:.2f}s")

    # Print Top 15 ranked promos to console
    print("\n" + "=" * 80)
    print(f"TOP {len(ranked_promos)} RANKED PROMOS (FINAL EDITORIAL SELECTION)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Promo ID':<12} {'Dur':<6} {'Score':<7} {'Hook':<5} {'Cliff':<6} {'Clarity':<8} {'Segments'}")
    print("-" * 80)
    for p in ranked_promos:
        j = p.get("judge", {})
        seg_str = "; ".join([f"E{s['episode']}({s['start_ts']}-{s['end_ts']})" for s in p["segments"]])
        print(
            f"#{p['rank']:<4} {p['promo_id']:<12} {p['total_duration']:<5.1f}s {p['score']:<7.2f} "
            f"{j.get('hook_strength', '-'):<5} {j.get('cliffhanger_strength', '-'):<6} "
            f"{j.get('standalone_clarity', '-'):<8} {seg_str}"
        )
    print("-" * 80)

    print("\n=== TOP 5 'POST FIRST' MARKETING STRATEGY ===")
    for p in ranked_promos[:5]:
        print(f"Rank {p['rank']} ({p['promo_id']}): {p['post_first_note']}")

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
    find_p.add_argument("--out", default="outputs/full_run", help="Output directory path")
    find_p.add_argument("--episodes", type=int, default=None, help="Limit number of episodes to process")
    find_p.add_argument("--max-candidates-per-ep", type=int, default=4, help="Max candidates per episode in Pass A")
    find_p.add_argument("--max-cross-candidates", type=int, default=8, help="Max cross-episode candidates in Pass B")
    find_p.add_argument("--top-n", type=int, default=15, help="Number of ranked promos to output")
    find_p.add_argument("--workers", type=int, default=3, help="Concurrent workers for LLM calls")
    find_p.add_argument("--no-cache", action="store_true", help="Bypass LLM response cache")

    args = parser.parse_args()
    if args.command == "parse":
        return cmd_parse(args)
    if args.command == "find":
        return cmd_find(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
