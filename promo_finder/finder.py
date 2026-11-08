"""Candidate generation and cold-viewer judge for microdrama promos.

Implements:
1. Compact transcript rendering with token-efficient formatting.
2. Pass A: Single-episode candidate generation via Gemini.
3. Pass B: Cross-episode candidate generation via Gemini.
4. Cold-viewer LLM judge evaluating hook, cliffhanger, and stand-alone clarity.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from promo_finder.llm import generate_json
from promo_finder.parse import Episode, Event, format_timestamp
from promo_finder.validate import RuleValidator, ValidatedPromo, ValidationResult

logger = logging.getLogger("promo_finder.finder")

CANDIDATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "candidates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "promo_id": {"type": "STRING"},
                    "segments": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "start_event_id": {"type": "STRING"},
                                "end_event_id": {"type": "STRING"},
                            },
                            "required": ["start_event_id", "end_event_id"],
                        },
                    },
                    "hook_event_id": {"type": "STRING"},
                    "cliffhanger_event_id": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": ["segments", "hook_event_id", "cliffhanger_event_id", "reason"],
            },
        }
    },
    "required": ["candidates"],
}

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "hook_strength": {
            "type": "INTEGER",
            "description": "Score 1-5 for hook stopping power in first 3 seconds",
        },
        "cliffhanger_strength": {
            "type": "INTEGER",
            "description": "Score 1-5 for cliffhanger urgency and unresolved tension at cut",
        },
        "standalone_clarity": {
            "type": "INTEGER",
            "description": "Score 1-5 for standalone story stakes to a complete stranger",
        },
        "summary": {
            "type": "STRING",
            "description": "One sentence summarizing what happens in this clip",
        },
        "open_question": {
            "type": "STRING",
            "description": "The burning unresolved question left at the cut",
        },
    },
    "required": ["hook_strength", "cliffhanger_strength", "standalone_clarity", "summary", "open_question"],
}


def format_compact_transcript(episode: Episode, max_cam_chars: int = 70) -> str:
    """Render token-efficient event transcript with stable IDs and timestamps."""
    lines = [f"EPISODE {episode.number}: {episode.logline}"]
    for ev in episode.events:
        ts = f"{format_timestamp(ev.start)}-{format_timestamp(ev.end)}"
        if ev.kind == "dialogue":
            spk = ev.speaker or "SPEAKER"
            emo = f" ({ev.emotion})" if ev.emotion else ""
            lines.append(f'[{ev.id} {ts}] DIALOGUE {spk}{emo}: "{ev.text}"')
        elif ev.kind == "action":
            lines.append(f"[{ev.id} {ts}] ACTION: {ev.text}")
        elif ev.kind == "camera":
            # Truncate long camera setups to save prompt tokens
            txt = ev.text[:max_cam_chars] + "..." if len(ev.text) > max_cam_chars else ev.text
            lines.append(f"[{ev.id} {ts}] CAMERA: {txt}")
        elif ev.kind == "other":
            lines.append(f"[{ev.id} {ts}] SOUND: {ev.text}")
    return "\n".join(lines)


def generate_single_episode_candidates(
    episode: Episode, max_candidates: int = 4, use_cache: bool = True
) -> list[dict[str, Any]]:
    """Pass A: generate candidate promos within a single episode."""
    transcript = format_compact_transcript(episode)

    prompt = f"""You are an expert promo producer for vertical mobile microdramas (like ReelShort, DramaBox, ShortMax).
Your task is to identify up to {max_candidates} high-converting promotional clips from Episode {episode.number}.

WHAT MAKES A VIRAL PROMO:
- Total duration MUST be between 20 and 60 seconds (sum of segment durations).
- OPENS ON A HOOK within the first 3 seconds: A moment that stops a scrolling viewer on TikTok/Instagram Reels (a slap, a gasp, a shocking accusation, an altar abandonment, a sudden reveal, a cruel insult).
- ENDS ON A CLIFFHANGER: Cuts sharply at the moment of highest unresolved tension (right before an answer, right after an ultimatum, before a secret is exposed), leaving the viewer desperate to watch the show.
- STANDS ALONE: A viewer who has never heard of this show must instantly grasp who is against whom and what is at stake.

TRANSCRIPT FOR EPISODE {episode.number}:
{transcript}

INSTRUCTIONS:
1. Each promo can consist of 1 continuous stretch or 2 joined segments within this episode.
2. For each segment, provide EXACT start_event_id and end_event_id from the transcript.
3. Identify the hook_event_id (MUST start within the first 3 seconds of the promo).
4. Identify the cliffhanger_event_id (MUST be the final event of the promo cut).
5. Ensure total duration is between 20 and 60 seconds by checking the timestamp range [MM:SS.ss-MM:SS.ss].
6. Provide promo_id (e.g. 'E{episode.number:02d}-P1') and a concise reason why this promo converts.
"""

    resp_json, _ = generate_json(
        prompt=prompt,
        schema=CANDIDATE_SCHEMA,
        temperature=0.2,
        use_cache=use_cache,
        label=f"pass_a_ep_{episode.number:02d}",
    )
    return resp_json.get("candidates", [])


def generate_cross_episode_candidates(
    episodes: list[Episode], top_moments: list[dict[str, Any]], max_candidates: int = 8, use_cache: bool = True
) -> list[dict[str, Any]]:
    """Pass B: generate candidate promos joining moments across 2-3 episodes."""
    loglines_block = "\n".join(f"Episode {ep.number}: {ep.logline}" for ep in episodes[:25])
    moments_summary = "\n".join(
        f"[{m['event_id']} Ep {m['episode']} {m['kind']}]: {m['text'][:100]}"
        for m in top_moments[:40]
    )

    prompt = f"""You are an expert promo producer for vertical mobile microdramas (ReelShort / DramaBox style).
Find up to {max_candidates} cross-episode promos that combine setup in an earlier episode with a dramatic twist or confrontation in a later episode (2 to 3 episodes max).

TARGET FORMAT:
- Total combined duration: strictly 20 to 60 seconds.
- Segment 1: Setup / hook from earlier episode (hook in first 3s).
- Segment 2 (and optional 3): Confrontation / revenge / cliffhanger from a later episode in story order.
- Each segment must have valid start_event_id and end_event_id.
- Must end on an unresolved cliffhanger.

SERIES LOGLINES:
{loglines_block}

KEY DRAMATIC MOMENTS AVAILABLE:
{moments_summary}

Return up to {max_candidates} candidates matching the JSON schema.
"""

    resp_json, _ = generate_json(
        prompt=prompt,
        schema=CANDIDATE_SCHEMA,
        temperature=0.3,
        use_cache=use_cache,
        label="pass_b_cross_episode",
    )
    return resp_json.get("candidates", [])


def judge_cold_viewer(promo: ValidatedPromo, use_cache: bool = True) -> dict[str, Any]:
    """Cold-viewer judge: evaluates promo text with zero show context."""
    # Build text seen strictly by the viewer
    lines: list[str] = []
    current_time = 0.0

    for seg_idx, seg in enumerate(promo.segments):
        if seg_idx > 0:
            lines.append("--- [SCENE TRANSITION] ---")
        for ev in seg.events:
            ts_str = format_timestamp(current_time)
            if ev.kind == "dialogue":
                spk = ev.speaker or "CHARACTER"
                emo = f" ({ev.emotion})" if ev.emotion else ""
                lines.append(f'[{ts_str}] {spk}{emo}: "{ev.text}"')
            elif ev.kind == "action":
                lines.append(f"[{ts_str}] (Action): {ev.text}")
            elif ev.kind == "other":
                lines.append(f"[{ts_str}] (Sound): {ev.text}")
            current_time += ev.duration

    promo_text = "\n".join(lines)

    prompt = f"""You are a mobile user scrolling TikTok / Instagram Reels / YouTube Shorts.
You have NEVER seen this show before. You just stopped on this {promo.total_duration:.1f}-second video clip.

CLIP DIALOGUE & ACTIONS:
{promo_text}

Evaluate this clip strictly as a first-time scrolling viewer:
1. hook_strength (1-5): Did the opening 3 seconds hook your attention immediately?
2. cliffhanger_strength (1-5): Does the ending leave an unresolved peak tension where you NEED to see what happens next?
3. standalone_clarity (1-5): Does this make sense on its own without needing prior backstory? (5 = crystal clear stakes; 1 = total confusion / who are these people).
4. summary: One sentence summarizing what you think happened in this clip.
5. open_question: The burning unresolved question you are left with at the cut.
"""

    resp_json, _ = generate_json(
        prompt=prompt,
        schema=JUDGE_SCHEMA,
        temperature=0.1,
        use_cache=use_cache,
        label=f"judge_{promo.promo_id}",
    )
    return resp_json
