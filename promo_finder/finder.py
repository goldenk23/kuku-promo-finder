"""Candidate generation and cold-viewer judge for microdrama promos.

Implements:
1. Compact transcript rendering with token-efficient formatting.
2. Pass A: Single-episode candidate generation via Gemini.
3. Pass B: Cross-episode candidate generation via Gemini.
4. Cold-viewer LLM judge evaluating hook, cliffhanger, and stand-alone clarity.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from promo_finder.llm import generate_json
from promo_finder.parse import Episode, format_timestamp
from promo_finder.rank import compute_promo_score
from promo_finder.validate import RuleValidator, ValidatedPromo

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
        # Observations come first so the model commits to evidence before it scores.
        "first_3_seconds": {
            "type": "STRING",
            "description": "What happens in the first 3 seconds and whether it would stop your thumb",
        },
        "final_moment": {
            "type": "STRING",
            "description": "The last moment before the cut and whether it is resolved or unresolved",
        },
        "confusion": {
            "type": "STRING",
            "description": "Anything you could not follow without backstory, or 'none'",
        },
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
        "tap_through": {
            "type": "INTEGER",
            "description": "Of 100 strangers who watched to the end, how many would tap through to the show (0-100)",
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
    "required": [
        "first_3_seconds", "final_moment", "confusion", "hook_strength", "cliffhanger_strength",
        "standalone_clarity", "tap_through", "summary", "open_question",
    ],
    "propertyOrdering": [
        "first_3_seconds", "final_moment", "confusion", "hook_strength", "cliffhanger_strength",
        "standalone_clarity", "tap_through", "summary", "open_question",
    ],
}


def render_event_line(ev: Any, ts: str, max_cam_chars: int = 70, with_id: bool = True) -> str:
    """One transcript line for prompts. Camera text is truncated to save tokens."""
    prefix = f"[{ev.id} {ts}]" if with_id else f"[{ts}]"
    if ev.kind == "dialogue":
        emo = f" ({ev.emotion})" if ev.emotion else ""
        return f'{prefix} DIALOGUE {ev.speaker or "SPEAKER"}{emo}: "{ev.text}"'
    if ev.kind == "camera":
        txt = ev.text[:max_cam_chars] + "..." if len(ev.text) > max_cam_chars else ev.text
        return f"{prefix} CAMERA: {txt}"
    if ev.kind == "action":
        return f"{prefix} ACTION: {ev.text}"
    return f"{prefix} SOUND: {ev.text}"


def format_compact_transcript(episode: Episode, max_cam_chars: int = 70) -> str:
    """Render token-efficient event transcript with stable IDs and timestamps."""
    lines = [f"EPISODE {episode.number}: {episode.logline}"]
    for ev in episode.events:
        ts = f"{format_timestamp(ev.start)}-{format_timestamp(ev.end)}"
        lines.append(render_event_line(ev, ts, max_cam_chars))
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
    episodes: list[Episode], source_promos: list[ValidatedPromo], max_candidates: int = 8, use_cache: bool = True
) -> list[dict[str, Any]]:
    """Pass B: join moments from 2-3 different episodes into one promo.

    The model only sees the strongest judged single-episode clips (one per episode, many episodes),
    with their event IDs and timestamps, so it can pick real sub-ranges and add up the duration.
    """
    loglines = {ep.number: ep.logline for ep in episodes}
    blocks: list[str] = []
    for p in source_promos:
        ep_num = p.segments[0].episode
        lines = [f"CLIP {p.promo_id} (Episode {ep_num}: {loglines.get(ep_num, '')})"]
        for seg in p.segments:
            for ev in seg.events:
                lines.append(render_event_line(ev, f"{format_timestamp(ev.start)}-{format_timestamp(ev.end)}", 50))
        blocks.append("\n".join(lines))
    clips_block = "\n\n".join(blocks)

    prompt = f"""You are an expert promo producer for vertical mobile microdramas (ReelShort / DramaBox style).
Build up to {max_candidates} CROSS-EPISODE promos. Each one joins 2 or 3 short segments taken from
DIFFERENT episodes, e.g. a humiliation or setup from an early episode followed by the payback,
twist or confrontation from a later episode.

RULES:
- Every segment is a sub-range of ONE clip below: give its start_event_id and end_event_id exactly as written.
- Segments come from at least 2 different episodes and are listed in story order (earlier episode first).
- Total duration (sum of each segment's end time minus start time) is between 20 and 60 seconds.
  Keep each segment short (about 8-25 seconds) so the total fits.
- hook_event_id is in the first segment and starts within its first 3 seconds.
- cliffhanger_event_id is the last event of the last segment and leaves a question unanswered.
- The joined moments must be about the same characters and conflict, so a stranger can follow the jump.
- reason: one line on why this pairing works as an ad.

CLIPS:
{clips_block}
"""

    resp_json, _ = generate_json(
        prompt=prompt,
        schema=CANDIDATE_SCHEMA,
        temperature=0.3,
        use_cache=use_cache,
        label="pass_b_cross_episode",
    )
    return resp_json.get("candidates", [])


def build_viewer_transcript(promo: ValidatedPromo) -> str:
    """What a viewer sees, with timestamps measured from the start of the assembled clip."""
    lines: list[str] = []
    clip_offset = 0.0
    for seg_idx, seg in enumerate(promo.segments):
        if seg_idx > 0:
            lines.append("--- [CUT TO A DIFFERENT MOMENT] ---")
        for ev in seg.events:
            t = clip_offset + max(0.0, ev.start - seg.start)
            lines.append(render_event_line(ev, format_timestamp(t), max_cam_chars=90, with_id=False))
        clip_offset += seg.duration
    return "\n".join(lines)


def judge_cold_viewer(promo: ValidatedPromo, use_cache: bool = True) -> dict[str, Any]:
    """Cold-viewer judge: scores the clip with zero show context."""
    prompt = f"""You are a mobile user scrolling TikTok / Instagram Reels / YouTube Shorts.
You have NEVER seen this show. This {promo.total_duration:.1f}-second clip appears in your feed.
Timestamps are seconds from the start of the clip. CAMERA lines describe what is on screen.

CLIP:
{build_viewer_transcript(promo)}

First write down what you observe (first_3_seconds, final_moment, confusion), then score.
Score harshly and use the whole scale. Most clips in a feed are a 2 or 3; a 5 is rare.
- hook_strength: 5 = a slap, reveal or shocking line lands in the first 3 seconds;
  3 = mildly intriguing; 1 = establishing shot, small talk, or nothing happens yet.
- cliffhanger_strength: 5 = cuts mid-crisis on an urgent unanswered question;
  3 = something is open but the moment has mostly settled; 1 = the conflict resolves or the clip trails off.
- standalone_clarity: 5 = you instantly know who is against whom and what is at stake;
  3 = you get the gist but some references confuse you; 1 = you cannot tell who these people are or why it matters.
- tap_through: out of 100 strangers who watched to the end, how many tap through to watch the show.
- summary: one sentence on what happened. open_question: the question you are left with at the cut.
"""

    resp_json, _ = generate_json(
        prompt=prompt,
        schema=JUDGE_SCHEMA,
        temperature=0.1,
        use_cache=use_cache,
        label=f"judge_{promo.promo_id}",
    )
    return resp_json


def run_pipeline(
    episodes: list[Episode],
    *,
    max_candidates_per_ep: int = 4,
    max_cross_candidates: int = 8,
    cross_source_clips: int = 15,
    workers: int = 3,
    use_cache: bool = True,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Pass A -> validate -> judge -> Pass B (from the best judged clips) -> validate -> judge.

    Shared by the CLI and the Streamlit app. `progress` is only called from the calling thread.
    """
    validator = RuleValidator(episodes)
    raw_candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    judgments: dict[str, dict[str, Any]] = {}
    workers = max(1, workers)

    def validate(cand: dict[str, Any]) -> ValidatedPromo | None:
        raw_candidates.append(cand)
        res = validator.validate_candidate(cand)
        if res.is_valid and res.promo:
            return res.promo
        rejections.append({"candidate": cand, "rule": res.rejection_rule, "reason": res.rejection_reason})
        return None

    def judge_all(promos: list[ValidatedPromo]) -> None:
        def one(p: ValidatedPromo) -> tuple[str, dict[str, Any]]:
            try:
                return p.promo_id, judge_cold_viewer(p, use_cache=use_cache)
            except Exception as err:  # keep the run going; failed judgments rank last
                return p.promo_id, {"error": str(err)}

        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(promos)))) as ex:
            for pid, j in ex.map(one, promos):
                judgments[pid] = j
                if "error" in j:
                    progress(f"  Judge FAILED for {pid}: {j['error']}")
                else:
                    progress(
                        f"  Judged {pid}: hook={j.get('hook_strength')} cliff={j.get('cliffhanger_strength')} "
                        f"clarity={j.get('standalone_clarity')} tap={j.get('tap_through')}"
                    )

    # Pass A: single-episode candidates
    progress(f"--- Pass A: single-episode candidates ({len(episodes)} episodes) ---")

    def fetch(ep: Episode) -> tuple[Episode, list[dict[str, Any]], str | None]:
        try:
            return ep, generate_single_episode_candidates(ep, max_candidates_per_ep, use_cache), None
        except Exception as err:
            return ep, [], str(err)

    single: list[ValidatedPromo] = []
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(episodes)))) as ex:
        for ep, cands, err in ex.map(fetch, episodes):
            progress(f"  Ep {ep.number:02d}: " + (f"FAILED: {err}" if err else f"{len(cands)} candidates"))
            for idx, c in enumerate(cands, 1):
                # IDs are assigned here, not trusted from the model, so they are unique per run.
                c["promo_id"] = f"E{ep.number:02d}-P{idx}"
                c["pass"] = "pass_a"
                promo = validate(c)
                if promo:
                    single.append(promo)

    progress(f"--- Judging {len(single)} single-episode promos ---")
    judge_all(single)

    # Pass B: cross-episode candidates built from the best judged clip of each episode
    cross: list[ValidatedPromo] = []
    if len(episodes) >= 2 and single:
        best_per_ep: dict[int, tuple[float, ValidatedPromo]] = {}
        for p in single:
            j = judgments.get(p.promo_id, {})
            score = compute_promo_score({"judge": j})
            ep_num = p.segments[0].episode
            if ep_num not in best_per_ep or score > best_per_ep[ep_num][0]:
                best_per_ep[ep_num] = (score, p)
        top = sorted(best_per_ep.values(), key=lambda sp: sp[0], reverse=True)[:cross_source_clips]
        sources = sorted((p for _, p in top), key=lambda p: p.segments[0].episode)

        progress(f"--- Pass B: cross-episode candidates from {len(sources)} episodes ---")
        try:
            cross_cands = generate_cross_episode_candidates(episodes, sources, max_cross_candidates, use_cache)
        except Exception as err:
            progress(f"  Pass B FAILED: {err}")
            cross_cands = []
        for idx, c in enumerate(cross_cands, 1):
            c["promo_id"] = f"X-P{idx}"
            c["pass"] = "pass_b"
            promo = validate(c)
            if promo and not promo.is_cross_episode:
                raw_rule = "pass_b_single_episode"
                validator.rejection_tally[raw_rule] = validator.rejection_tally.get(raw_rule, 0) + 1
                rejections.append({"candidate": c, "rule": raw_rule, "reason": "Pass B promo uses one episode only"})
            elif promo:
                cross.append(promo)
        progress(f"  {len(cross)} of {len(cross_cands)} cross-episode proposals passed validation")
        judge_all(cross)

    evaluated: list[dict[str, Any]] = []
    for promo in single + cross:
        d = promo.to_dict()
        d["judge"] = judgments.get(promo.promo_id, {"error": "not judged"})
        evaluated.append(d)

    return {
        "validator": validator,
        "raw_candidates": raw_candidates,
        "rejections": rejections,
        "evaluated": evaluated,
    }
