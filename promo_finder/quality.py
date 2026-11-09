"""Quality checks, stability benchmark, human review sheet, run summary and cost.

1. Judge sanity: does the judge score our picks above random 35s windows, and does it give
   incoherent 3-episode splices low clarity? If not, the ranking is not telling us much.
2. Stability: re-run Pass A on a few episodes with the cache bypassed and measure overlap.
3. Human review sheet with empty verdict columns for an editor to fill in.
4. Run summary with token usage and cost at configurable per-token prices.
5. Quality report built from this run's numbers.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from promo_finder.finder import generate_single_episode_candidates, judge_cold_viewer
from promo_finder.parse import Episode
from promo_finder.rank import calculate_overlap_ratio, compute_promo_score
from promo_finder.validate import RuleValidator

logger = logging.getLogger("promo_finder.quality")

# Vertex AI standard price for gemini-2.5-flash (USD per 1M tokens; output includes thinking tokens).
# Source: https://cloud.google.com/vertex-ai/generative-ai/pricing (checked 2026-11-09).
# Override with env vars when swapping models.
PRICE_INPUT_PER_M = float(os.getenv("LLM_PRICE_INPUT_PER_M", "0.30"))
PRICE_OUTPUT_PER_M = float(os.getenv("LLM_PRICE_OUTPUT_PER_M", "2.50"))


def _window(ep: Episode, start_idx: int, target_sec: float) -> dict[str, str]:
    """Whole-event segment starting at start_idx and running about target_sec."""
    evs = ep.events
    j = start_idx
    while j + 1 < len(evs) and evs[j].end - evs[start_idx].start < target_sec:
        j += 1
    return {"start_event_id": evs[start_idx].id, "end_event_id": evs[j].id}


def _judge_candidate(validator: RuleValidator, cand: dict[str, Any], use_cache: bool) -> dict[str, Any] | None:
    res = validator.validate_candidate(cand)
    if not res.promo:
        return None
    try:
        j = judge_cold_viewer(res.promo, use_cache=use_cache)
    except Exception as err:
        logger.warning("Sanity judge failed for %s: %s", cand["promo_id"], err)
        return None
    segs = "; ".join(f"Ep {s.episode} {s.start_ts}-{s.end_ts}" for s in res.promo.segments)
    return {"promo_id": cand["promo_id"], "segments": segs, "judge": j, "score": compute_promo_score({"judge": j})}


def run_judge_sanity_tests(
    episodes: list[Episode],
    ranked_promos: list[dict[str, Any]],
    n_random: int = 10,
    n_splices: int = 3,
    seed: int = 7,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Compare judge scores on our top picks vs. random windows and incoherent splices."""
    rng = random.Random(seed)
    validator = RuleValidator(episodes)
    usable = [ep for ep in episodes if len(ep.events) > 20]

    baseline: list[dict[str, Any]] = []
    attempts = 0
    while usable and len(baseline) < n_random and attempts < n_random * 4:
        attempts += 1
        ep = rng.choice(usable)
        seg = _window(ep, rng.randrange(len(ep.events) - 1), 35.0)
        cand = {
            "promo_id": f"RANDOM-{len(baseline) + 1:02d}",
            "segments": [seg],
            "hook_event_id": seg["start_event_id"],
            "cliffhanger_event_id": seg["end_event_id"],
            "reason": "Random 35s window (baseline)",
        }
        r = _judge_candidate(validator, cand, use_cache)
        if r:
            baseline.append(r)

    splices: list[dict[str, Any]] = []
    attempts = 0
    while len(usable) >= 3 and len(splices) < n_splices and attempts < n_splices * 4:
        attempts += 1
        picked = sorted(rng.sample(usable, 3), key=lambda e: e.number)
        segs = [_window(ep, rng.randrange(len(ep.events) - 1), 10.0) for ep in picked]
        cand = {
            "promo_id": f"SPLICE-{len(splices) + 1:02d}",
            "segments": segs,
            "hook_event_id": segs[0]["start_event_id"],
            "cliffhanger_event_id": segs[-1]["end_event_id"],
            "reason": "Three unrelated 10s moments from different episodes",
        }
        r = _judge_candidate(validator, cand, use_cache)
        if r:
            r["passed"] = r["judge"].get("standalone_clarity", 5) <= 3
            splices.append(r)

    top = [p for p in ranked_promos[:10] if "error" not in p.get("judge", {})]
    top_mean = statistics.mean(p["score"] for p in top) if top else 0.0
    base_mean = statistics.mean(r["score"] for r in baseline) if baseline else 0.0
    return {
        "baseline": baseline,
        "splices": splices,
        "top10_mean_score": round(top_mean, 3),
        "random_mean_score": round(base_mean, 3),
        "random_beaten_by_top10": sum(1 for r in baseline if r["score"] < top_mean),
        # ponytail: 0.5-point margin on a ~1-5 scale is a judgment call, not a calibrated threshold.
        "discriminates": bool(baseline) and top_mean - base_mean >= 0.5,
    }


def run_stability_check(
    episodes: list[Episode], base_validated_promos: list[dict[str, Any]], sample_episodes: int = 3
) -> dict[str, Any]:
    """Re-run Pass A on the first few episodes with the cache bypassed; measure overlap with the main run."""
    sample_eps = episodes[:sample_episodes]
    sample_nums = {ep.number for ep in sample_eps}
    validator = RuleValidator(sample_eps)

    fresh_promos: list[dict[str, Any]] = []
    for ep in sample_eps:
        try:
            cands = generate_single_episode_candidates(ep, max_candidates=4, use_cache=False)
        except Exception as err:
            logger.warning("Stability check fresh run failed for ep %d: %s", ep.number, err)
            continue
        for c_idx, c in enumerate(cands, 1):
            c["promo_id"] = f"FRESH-E{ep.number:02d}-P{c_idx}"
            val_res = validator.validate_candidate(c)
            if val_res.promo:
                fresh_promos.append(val_res.promo.to_dict())

    base_sample = [
        p for p in base_validated_promos
        if not p["is_cross_episode"] and p["segments"][0]["episode"] in sample_nums
    ]
    matches = sum(
        1 for b in base_sample if any(calculate_overlap_ratio(b, f) >= 0.40 for f in fresh_promos)
    )
    pct = matches / len(base_sample) * 100.0 if base_sample else 0.0
    return {
        "sample_episodes": len(sample_eps),
        "base_candidate_count": len(base_sample),
        "fresh_candidate_count": len(fresh_promos),
        "overlap_matches": matches,
        "overlap_percentage": round(pct, 1),
    }


def generate_human_review_csv(top_promos: list[dict[str, Any]], out_path: Path) -> None:
    """Review sheet for an editor. Verdict columns are left empty on purpose."""
    if not top_promos:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in top_promos[:10]:
        rows.append({
            "rank": p["rank"],
            "promo_id": p["promo_id"],
            "duration": f"{p['total_duration']}s",
            "segments": "; ".join(f"Ep {s['episode']} ({s['start_ts']}-{s['end_ts']})" for s in p["segments"]),
            "hook_text": p.get("hook_text", ""),
            "cliffhanger_text": p.get("cliffhanger_text", ""),
            "hook_ok (Y/N)": "",
            "cliff_ok (Y/N)": "",
            "standalone_ok (Y/N)": "",
            "would_post (Y/N)": "",
            "editor_notes": "",
        })
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_usage_summary(usage_path: Path) -> dict[str, Any]:
    """Sum token usage for one run and price it."""
    totals = {"total_calls": 0, "cached_calls": 0, "live_calls": 0, "prompt_tokens": 0, "candidates_tokens": 0}
    if usage_path.exists():
        with open(usage_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                totals["total_calls"] += 1
                if entry.get("cached"):
                    totals["cached_calls"] += 1
                else:
                    totals["live_calls"] += 1
                    totals["prompt_tokens"] += entry.get("prompt_tokens", 0)
                    totals["candidates_tokens"] += entry.get("candidates_tokens", 0)
    totals["total_tokens"] = totals["prompt_tokens"] + totals["candidates_tokens"]
    cost = totals["prompt_tokens"] / 1e6 * PRICE_INPUT_PER_M + totals["candidates_tokens"] / 1e6 * PRICE_OUTPUT_PER_M
    totals["cost_usd"] = round(cost, 4)
    return totals


def _dist(values: list[Any]) -> str:
    c = Counter(values)
    return ", ".join(f"{k}: {c.get(k, 0)}" for k in (1, 2, 3, 4, 5))


def write_quality_report(
    out_dir: Path,
    validator_tally: dict[str, int],
    total_candidates: int,
    evaluated: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    sanity: dict[str, Any],
    stability: dict[str, Any],
) -> None:
    """Quality report built from this run's numbers."""
    judged = [p for p in evaluated if "error" not in p.get("judge", {})]
    n = max(1, len(judged))
    rejected = sum(validator_tally.values())
    md = [
        "# Quality Report",
        "",
        "## 1. Rule validator",
        f"{total_candidates} candidates proposed, {rejected} rejected, {len(evaluated)} passed.",
        "",
        "| Rule | Rejections |",
        "|---|---|",
    ]
    md += [f"| `{rule}` | {count} |" for rule, count in validator_tally.items() if count > 0] or ["| (none) | 0 |"]
    repairs = Counter(r.split(" ")[0] for p in evaluated for r in p.get("repairs_applied", []))
    md += [
        "",
        f"Repairs applied to passing promos: {dict(repairs) or 'none'} "
        "(Widened = cut moved so no line is cut mid-sentence; Trimmed/Snapped/Extended = whole-event boundary fixes).",
        "",
        "## 2. Judge score distribution (all passing promos)",
        f"- Judge failures: {len(evaluated) - len(judged)}",
    ]
    for key, label in [("hook_strength", "Hook"), ("cliffhanger_strength", "Cliffhanger"), ("standalone_clarity", "Clarity")]:
        vals = [p["judge"].get(key) for p in judged]
        fives = sum(1 for v in vals if v == 5) / n * 100
        md.append(f"- {label}: {_dist(vals)} ({fives:.0f}% are 5/5)")
    ceiling = [
        label for key, label in [("hook_strength", "hook"), ("cliffhanger_strength", "cliffhanger"), ("standalone_clarity", "clarity")]
        if sum(1 for p in judged if p["judge"].get(key) == 5) / n > 0.5
    ]
    if ceiling:
        md.append(f"- Warning: over half of promos get 5/5 on {', '.join(ceiling)}, so that score barely separates them.")
    cross_all = sum(1 for p in evaluated if p["is_cross_episode"])
    cross_top = sum(1 for p in ranked if p["is_cross_episode"])
    top_eps = sorted({s["episode"] for p in ranked for s in p["segments"]})
    md += [
        f"- Cross-episode promos: {cross_all} passed validation, {cross_top} in the final top {len(ranked)}.",
        f"- Episodes used by the top {len(ranked)}: {top_eps}",
        "",
        "## 3. Does the judge tell good picks from bad ones?",
        f"Random 35-second windows (seed-fixed) vs. our top 10: mean score **{sanity['random_mean_score']}** "
        f"vs. **{sanity['top10_mean_score']}**. Our top-10 mean beats {sanity['random_beaten_by_top10']} of "
        f"{len(sanity['baseline'])} random windows.",
        "",
        "**Result: the judge separates picks from random windows.**" if sanity["discriminates"] else
        "**Result: the judge does NOT clearly separate our picks from random windows (gap < 0.5).** "
        "This show is dense with conflict, so random cuts also look dramatic on paper; treat scores as a coarse filter "
        "and keep an editor on the final pick.",
        "",
        "| Probe | Segments | Hook / Cliff / Clarity | Score |",
        "|---|---|---|---|",
    ]
    for r in sanity["baseline"]:
        j = r["judge"]
        md.append(f"| {r['promo_id']} | {r['segments']} | {j.get('hook_strength')}/{j.get('cliffhanger_strength')}/{j.get('standalone_clarity')} | {r['score']} |")
    md += ["", "Incoherent splices (3 unrelated 10s moments from different episodes; expect clarity <= 3):", "",
           "| Probe | Segments | Hook / Cliff / Clarity | Flagged? |", "|---|---|---|---|"]
    for r in sanity["splices"]:
        j = r["judge"]
        md.append(f"| {r['promo_id']} | {r['segments']} | {j.get('hook_strength')}/{j.get('cliffhanger_strength')}/{j.get('standalone_clarity')} | {'yes' if r['passed'] else '**no, over-scored**'} |")

    weakest = sorted(judged, key=lambda p: (p["judge"].get("standalone_clarity", 5), compute_promo_score(p)))[:5]
    md += [
        "",
        "## 4. Where the tool picks badly (lowest-clarity promos it proposed)",
        "",
        "| Promo | Segments | Hook / Cliff / Clarity | What confused the judge |",
        "|---|---|---|---|",
    ]
    for p in weakest:
        j = p["judge"]
        segs = "; ".join(f"Ep {s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])
        confusion = str(j.get("confusion", "")).replace("|", "/").replace("\n", " ")
        md.append(f"| {p['promo_id']} | {segs} | {j.get('hook_strength')}/{j.get('cliffhanger_strength')}/{j.get('standalone_clarity')} | {confusion} |")

    stable_txt = (
        "Most picks reappear on a fresh run." if stability["overlap_percentage"] >= 70 else
        "Many picks change between runs, so a single run should not be treated as the definitive list."
    )
    md += [
        "",
        "## 5. Stability (Pass A re-run with cache bypassed)",
        f"- {stability['sample_episodes']} episodes: {stability['overlap_matches']} of {stability['base_candidate_count']} "
        f"original promos have a fresh-run match with >= 40% overlap ({stability['overlap_percentage']}%). {stable_txt}",
        "",
        "## 6. Known limits",
        "1. Script timestamps, not footage: the final edit may drift from the screenplay timing, so an editor should check cuts against video.",
        "2. Text-only judge: it reads camera/action descriptions but never sees the actual performance, music or faces.",
        "3. Same model family generates and judges, which can bias the judge toward its own picks.",
        "4. Spoilers: late-episode twists can make strong ads but give away the plot; the post-first note flags them, it does not block them.",
        "5. The 1-5 scores are coarse; ties are broken by the judge's 0-100 tap-through guess, which is not calibrated against real click data.",
    ]
    with open(out_dir / "quality_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def write_run_summary(
    out_dir: Path,
    show_name: str,
    model: str,
    total_episodes: int,
    total_events: int,
    total_candidates: int,
    total_valid: int,
    wall_time: float,
    usage: dict[str, Any],
) -> None:
    """Run summary and unit economics for this run only."""
    per_ep = usage["cost_usd"] / max(1, total_episodes)
    cache_note = (
        f"\n> {usage['cached_calls']} calls came from the local cache and cost nothing here, "
        "so this undercounts a fresh run. Use `--no-cache` for a true cost figure.\n"
        if usage["cached_calls"] else ""
    )
    summary_md = f"""# Run Summary

## Execution
- **Show**: {show_name} ({total_episodes} episodes)
- **Screenplay events parsed**: {total_events:,}
- **Candidates proposed**: {total_candidates}
- **Passed hard rules**: {total_valid}
- **Wall-clock time**: {wall_time:.0f}s
- **Model**: `{model}`

## Tokens and cost
- **API calls**: {usage['total_calls']} ({usage['live_calls']} live, {usage['cached_calls']} cached)
- **Input tokens**: {usage['prompt_tokens']:,}
- **Output tokens (incl. thinking)**: {usage['candidates_tokens']:,}
- **Prices used**: ${PRICE_INPUT_PER_M}/1M input, ${PRICE_OUTPUT_PER_M}/1M output
  ([Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing); override with `LLM_PRICE_INPUT_PER_M` / `LLM_PRICE_OUTPUT_PER_M`)
- **Run cost** (includes the quality-check calls): ${usage['cost_usd']:.4f}
- **Per episode**: ${per_ep:.4f}; **per 50-episode show**: ${per_ep * 50:.2f}
{cache_note}"""
    with open(out_dir / "run_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md + "\n")
