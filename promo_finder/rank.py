"""Ranking and filtering logic for validated microdrama promos.

Applies:
1. Weighted scoring formula heavily weighting stand-alone clarity.
2. Temporal de-duplication (>50% overlap suppression).
3. Episode/scene diversity constraint (at most 2 promos in top 10 from same scene).
4. Top-5 editorial justification ("Post first because...").
5. Multi-format export (JSON, CSV, Markdown).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def calculate_overlap_ratio(p1: dict[str, Any], p2: dict[str, Any]) -> float:
    """Compute temporal overlap ratio between two promos."""
    # Check if they share any episodes
    eps1 = {s["episode"] for s in p1["segments"]}
    eps2 = {s["episode"] for s in p2["segments"]}
    common_eps = eps1.intersection(eps2)
    if not common_eps:
        return 0.0

    total_intersection = 0.0
    for ep in common_eps:
        # Sum intersection intervals in this episode
        segs1 = [s for s in p1["segments"] if s["episode"] == ep]
        segs2 = [s for s in p2["segments"] if s["episode"] == ep]

        for s1 in segs1:
            for s2 in segs2:
                inter_start = max(s1["start"], s2["start"])
                inter_end = min(s1["end"], s2["end"])
                if inter_end > inter_start:
                    total_intersection += inter_end - inter_start

    min_dur = min(p1["total_duration"], p2["total_duration"])
    if min_dur <= 0.0:
        return 0.0
    return total_intersection / min_dur


def compute_promo_score(promo: dict[str, Any]) -> float:
    """Calculate transparent composite quality score."""
    judge = promo.get("judge", {})
    hook = float(judge.get("hook_strength", 3))
    cliff = float(judge.get("cliffhanger_strength", 3))
    clarity = float(judge.get("standalone_clarity", 3))

    # Standalone clarity (0.40): promo fails completely if viewer is confused
    # Hook strength (0.35): critical thumb-stopper in vertical feeds
    # Cliffhanger (0.25): drives click-through to episode
    base_score = (0.40 * clarity) + (0.35 * hook) + (0.25 * cliff)

    # Cross-episode bonus: +0.25 if clarity is high (>=4)
    bonus = 0.0
    if promo.get("is_cross_episode") and clarity >= 4.0:
        bonus = 0.25

    return round(base_score + bonus, 3)


def generate_post_first_note(promo: dict[str, Any], rank: int) -> str:
    """Provide specific marketing rationale for why this promo should be posted first."""
    j = promo.get("judge", {})
    hook_s = j.get("hook_strength", 3)
    cliff_s = j.get("cliffhanger_strength", 3)
    clarity_s = j.get("standalone_clarity", 3)
    summary = j.get("summary", "")
    open_q = j.get("open_question", "")

    if rank == 1:
        return (
            f"Primary launch creative: Flawless {promo['total_duration']}s hook-to-cliffhanger arc "
            f"(Hook {hook_s}/5, Cliff {cliff_s}/5). Immediate emotional stakes with zero backstory required: {summary}"
        )
    elif rank == 2:
        return (
            f"Secondary high-converting variant: Strong conflict escalation ({promo['total_duration']}s). "
            f"Cuts on burning question: \"{open_q}\""
        )
    elif rank == 3:
        return (
            f"High-drama alternative angle: Drives intense outrage/curiosity (Clarity {clarity_s}/5). "
            f"Ideal for A/B testing against Rank 1 on TikTok."
        )
    elif rank == 4:
        return (
            f"Short punchy cut ({promo['total_duration']}s): High retention candidate with immediate tension "
            f"and high completion rate potential."
        )
    elif rank == 5:
        return (
            f"Emotional hook variant ({promo['total_duration']}s): Leverages relational betrayal, "
            f"priming viewers to binge Episode 1 immediately."
        )
    return ""


def rank_and_deduplicate_promos(
    promos: list[dict[str, Any]], top_n: int = 15, max_overlap: float = 0.50
) -> list[dict[str, Any]]:
    """Score, deduplicate, and enforce diversity constraints on promos."""
    # 1. Score each promo
    scored_promos = []
    for p in promos:
        p_copy = dict(p)
        p_copy["score"] = compute_promo_score(p_copy)
        scored_promos.append(p_copy)

    # Sort descending by score, then duration
    scored_promos.sort(key=lambda x: (x["score"], -abs(x["total_duration"] - 35.0)), reverse=True)

    # 2. De-duplicate by temporal overlap (>50%)
    deduped: list[dict[str, Any]] = []
    for candidate in scored_promos:
        is_duplicate = False
        for kept in deduped:
            overlap = calculate_overlap_ratio(candidate, kept)
            if overlap > max_overlap:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(candidate)

    # 3. Apply scene/episode diversity for top list
    # At most 2 promos in top 10 from same scene
    diverse_list: list[dict[str, Any]] = []
    scene_counts: dict[str, int] = {}

    for p in deduped:
        first_seg = p["segments"][0]
        # Scene key e.g. "Ep1-S1" from start_event_id
        scene_key = f"Ep{first_seg['episode']}-{first_seg['start_event_id'].split('-')[1]}"

        if len(diverse_list) < 10 and scene_counts.get(scene_key, 0) >= 2:
            # Skip for top 10 to ensure diversity across episodes/scenes
            continue

        scene_counts[scene_key] = scene_counts.get(scene_key, 0) + 1
        diverse_list.append(p)
        if len(diverse_list) >= top_n:
            break

    # If we didn't fill top_n due to diversity filters, append best remaining
    if len(diverse_list) < top_n:
        for p in deduped:
            if p not in diverse_list:
                diverse_list.append(p)
                if len(diverse_list) >= top_n:
                    break

    # Format final ranked items
    final_ranked: list[dict[str, Any]] = []
    for idx, p in enumerate(diverse_list[:top_n], 1):
        p["rank"] = idx
        p["post_first_note"] = generate_post_first_note(p, idx)
        final_ranked.append(p)

    return final_ranked


def export_ranked_promos(
    ranked_promos: list[dict[str, Any]], out_dir: Path, events_by_id: dict[str, Any] | None = None
) -> None:
    """Export ranked promos to JSON, CSV, and human-readable Markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON
    with open(out_dir / "promos_ranked.json", "w", encoding="utf-8") as f:
        json.dump(ranked_promos, f, indent=2)

    # 2. CSV
    csv_rows = []
    for p in ranked_promos:
        j = p.get("judge", {})
        seg_strs = [
            f"Ep {s['episode']} [{s['start_ts']} - {s['end_ts']}]"
            for s in p["segments"]
        ]
        hook_text = ""
        cliff_text = ""
        if events_by_id:
            h_ev = events_by_id.get(p["hook_event_id"])
            c_ev = events_by_id.get(p["cliffhanger_event_id"])
            if h_ev:
                hook_text = f"{h_ev.speaker or 'Action'}: {h_ev.text}"
            if c_ev:
                cliff_text = f"{c_ev.speaker or 'Action'}: {c_ev.text}"

        csv_rows.append({
            "rank": p["rank"],
            "promo_id": p["promo_id"],
            "score": p["score"],
            "duration_sec": p["total_duration"],
            "segments": "; ".join(seg_strs),
            "hook_id": p["hook_event_id"],
            "hook_score": j.get("hook_strength", ""),
            "cliffhanger_id": p["cliffhanger_event_id"],
            "cliff_score": j.get("cliffhanger_strength", ""),
            "clarity_score": j.get("standalone_clarity", ""),
            "is_cross_episode": p["is_cross_episode"],
            "hook_text": hook_text,
            "cliffhanger_text": cliff_text,
            "judge_summary": j.get("summary", ""),
            "open_question": j.get("open_question", ""),
            "post_first_note": p.get("post_first_note", ""),
        })

    if csv_rows:
        with open(out_dir / "promos_ranked.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # 3. Markdown
    md_lines = [
        "# Top Ranked Microdrama Promos",
        "",
        "| Rank | Promo ID | Duration | Segments | Hook | Cliff | Clarity | Score | Post First Note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in ranked_promos:
        j = p.get("judge", {})
        seg_str = "<br>".join([f"Ep {s['episode']} ({s['start_ts']}-{s['end_ts']})" for s in p["segments"]])
        note = p.get("post_first_note", "-")
        if note:
            note = note.replace("\n", " ")
        md_lines.append(
            f"| **#{p['rank']}** | `{p['promo_id']}` | {p['total_duration']}s | {seg_str} | "
            f"{j.get('hook_strength')}/5 | {j.get('cliffhanger_strength')}/5 | {j.get('standalone_clarity')}/5 | "
            f"**{p['score']:.2f}** | {note} |"
        )

    md_lines.append("\n## Detailed Promo Breakdowns\n")
    for p in ranked_promos:
        j = p.get("judge", {})
        md_lines.append(f"### Rank {p['rank']}: Promo `{p['promo_id']}` (Score: {p['score']:.2f})")
        md_lines.append(f"- **Total Duration**: {p['total_duration']}s")
        md_lines.append(f"- **Segments**:")
        for s in p["segments"]:
            md_lines.append(f"  - Episode {s['episode']}: `{s['start_ts']}` to `{s['end_ts']}` ({s['duration']}s, `{s['start_event_id']}`..`{s['end_event_id']}`)")
        md_lines.append(f"- **Hook Event**: `{p['hook_event_id']}` (offset: {p['hook_offset']}s)")
        md_lines.append(f"- **Cliffhanger Event**: `{p['cliffhanger_event_id']}`")
        md_lines.append(f"- **Editorial Reason**: {p['reason']}")
        md_lines.append(f"- **Cold-Viewer Simulation**:")
        md_lines.append(f"  - Hook Stopping Power: {j.get('hook_strength')}/5")
        md_lines.append(f"  - Cliffhanger Urgency: {j.get('cliffhanger_strength')}/5")
        md_lines.append(f"  - Standalone Clarity: {j.get('standalone_clarity')}/5")
        md_lines.append(f"  - Summary: *\"{j.get('summary')}\"*")
        md_lines.append(f"  - Open Question: *\"{j.get('open_question')}\"*")
        if p.get("post_first_note"):
            md_lines.append(f"- **Why Post First**: {p['post_first_note']}")
        md_lines.append("")

    with open(out_dir / "promos_ranked.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
