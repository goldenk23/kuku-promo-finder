"""CLI entrypoint for promo_finder.

Usage:
    python -m promo_finder parse <folder-or-zip>
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from promo_finder.parse import format_timestamp, load_episodes


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


def main() -> int:
    parser = argparse.ArgumentParser(prog="promo_finder", description="Microdrama promo finder tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # parse subcommand
    parse_p = subparsers.add_parser("parse", help="Parse episode scripts and display summary statistics")
    parse_p.add_argument("source", help="Path to directory of episode .txt files or a .zip file")

    args = parser.parse_args()
    if args.command == "parse":
        return cmd_parse(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
