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

from promo_finder.finder import run_pipeline
from promo_finder.llm import get_llm_client, set_usage_log_path
from promo_finder.parse import format_timestamp, load_episodes
from promo_finder.quality import (
    generate_human_review_csv,
    parse_usage_summary,
    run_judge_sanity_tests,
    run_stability_check,
    write_quality_report,
    write_run_summary,
)
from promo_finder.rank import export_ranked_promos, rank_and_deduplicate_promos


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
    """Find, validate, judge and rank promos, then run quality checks."""
    start_time = time.time()
    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source '{source}' does not exist.", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    # Per-run usage log, so the cost in run_summary.md covers this run only.
    usage_path = out_dir / "usage.jsonl"
    usage_path.write_text("", encoding="utf-8")
    set_usage_log_path(usage_path)

    print(f"Loading episodes from {source} ...")
    episodes = load_episodes(source)
    if args.episodes and args.episodes > 0:
        episodes = episodes[: args.episodes]
    if not episodes:
        print("No episodes found.", file=sys.stderr)
        return 1
    print(f"Loaded {len(episodes)} episodes.\n")

    result = run_pipeline(
        episodes,
        max_candidates_per_ep=args.max_candidates_per_ep,
        max_cross_candidates=args.max_cross_candidates,
        workers=args.workers,
        use_cache=use_cache,
        progress=lambda msg: print(msg, flush=True),
    )
    validator = result["validator"]
    raw_candidates = result["raw_candidates"]
    rejections = result["rejections"]
    evaluated = result["evaluated"]

    print("\n--- Validation Summary ---")
    print(f"Candidates proposed: {len(raw_candidates)}")
    print(f"Passed:              {len(evaluated)}")
    print(f"Rejected:            {len(rejections)}")
    for rule, count in validator.rejection_tally.items():
        if count > 0:
            print(f"  - {rule:<25}: {count}")

    with open(out_dir / "candidates_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_candidates, f, indent=2)
    with open(out_dir / "validated.json", "w", encoding="utf-8") as f:
        json.dump(evaluated, f, indent=2)
    with open(out_dir / "rejections.json", "w", encoding="utf-8") as f:
        json.dump(rejections, f, indent=2)

    print("\n--- Ranking, de-duplicating, selecting ---")
    ranked = rank_and_deduplicate_promos(evaluated, top_n=args.top_n)
    export_ranked_promos(ranked, out_dir)
    generate_human_review_csv(ranked, out_dir / "human_review.csv")

    print("\n--- Quality checks ---")
    print("  Judging random-window baseline and incoherent splices ...")
    sanity = run_judge_sanity_tests(episodes, ranked, use_cache=use_cache)
    print(
        f"  Top-10 mean score {sanity['top10_mean_score']} vs. random windows {sanity['random_mean_score']} "
        f"({'separates' if sanity['discriminates'] else 'does NOT clearly separate'}); "
        f"splices flagged: {sum(r['passed'] for r in sanity['splices'])}/{len(sanity['splices'])}"
    )
    print("  Re-running Pass A on 3 episodes with cache bypassed ...")
    stability = run_stability_check(episodes, evaluated, sample_episodes=min(3, len(episodes)))
    print(f"  Stability overlap: {stability['overlap_percentage']}%")

    write_quality_report(
        out_dir=out_dir,
        validator_tally=validator.rejection_tally,
        total_candidates=len(raw_candidates),
        evaluated=evaluated,
        ranked=ranked,
        sanity=sanity,
        stability=stability,
    )

    elapsed = time.time() - start_time
    usage_info = parse_usage_summary(usage_path)
    write_run_summary(
        out_dir=out_dir,
        show_name=source.stem if source.is_file() else source.name,
        model=f"{get_llm_client().provider}:{get_llm_client().model}",
        total_episodes=len(episodes),
        total_events=sum(len(ep.events) for ep in episodes),
        total_candidates=len(raw_candidates),
        total_valid=len(evaluated),
        wall_time=elapsed,
        usage=usage_info,
    )

    print(f"\nOutputs written to {out_dir}/")
    print(f"Tokens: {usage_info['total_tokens']:,} (cost ${usage_info['cost_usd']:.4f}); wall time {elapsed:.0f}s")

    print("\n" + "=" * 80)
    print(f"TOP {len(ranked)} PROMOS")
    print("=" * 80)
    print(f"{'Rank':<5} {'Promo ID':<10} {'Dur':<7} {'Score':<6} {'H/C/Cl':<7} {'Tap':<4} Segments")
    for p in ranked:
        j = p.get("judge", {})
        seg_str = "; ".join(f"E{s['episode']}({s['start_ts']}-{s['end_ts']})" for s in p["segments"])
        hcc = f"{j.get('hook_strength', '-')}/{j.get('cliffhanger_strength', '-')}/{j.get('standalone_clarity', '-')}"
        print(f"#{p['rank']:<4} {p['promo_id']:<10} {p['total_duration']:<6.1f}s {p['score']:<6.2f} {hcc:<7} {str(j.get('tap_through', '-')):<4} {seg_str}")

    print("\n=== POST FIRST ===")
    for p in ranked[:5]:
        print(p["post_first_note"])

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
