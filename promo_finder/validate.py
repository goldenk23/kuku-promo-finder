"""Pure-code rule validator for microdrama promos.

Zero LLM logic. Validates and repairs promo segment boundaries against strict rules:
1. Event ID existence and ordering.
2. Promo total duration strictly between 20.0s and 60.0s (attempts event-snapped repair).
3. Hook event starts within the first 3.0s of the assembled promo.
4. Segments in chronological story order and non-overlapping.
5. Promo cuts on or immediately after the cliffhanger event.
6. Snaps all segment cuts to whole events (no mid-sentence cuts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from promo_finder.parse import Episode, Event, format_timestamp

logger = logging.getLogger("promo_finder.validate")


@dataclass
class PromoSegment:
    """A continuous stretch of events within an episode."""
    episode: int
    start_event_id: str
    end_event_id: str
    start: float  # seconds
    end: float  # seconds
    start_ts: str
    end_ts: str
    events: list[Event] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "start_event_id": self.start_event_id,
            "end_event_id": self.end_event_id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration": round(self.duration, 2),
            "event_count": len(self.events),
        }


@dataclass
class ValidatedPromo:
    """A promo that has satisfied all hard editorial and duration rules."""
    promo_id: str
    segments: list[PromoSegment]
    hook_event_id: str
    cliffhanger_event_id: str
    reason: str
    total_duration: float
    hook_offset: float  # seconds from promo start to hook start
    is_cross_episode: bool
    repairs_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "promo_id": self.promo_id,
            "total_duration": round(self.total_duration, 2),
            "hook_offset": round(self.hook_offset, 2),
            "is_cross_episode": self.is_cross_episode,
            "hook_event_id": self.hook_event_id,
            "cliffhanger_event_id": self.cliffhanger_event_id,
            "reason": self.reason,
            "repairs_applied": self.repairs_applied,
            "segments": [seg.to_dict() for seg in self.segments],
        }


@dataclass
class ValidationResult:
    """Result of validating a candidate promo."""
    promo: ValidatedPromo | None
    is_valid: bool
    rejection_rule: str | None = None
    rejection_reason: str | None = None


class RuleValidator:
    """Validates candidate promos against all screenplay events."""

    def __init__(self, episodes: list[Episode]) -> None:
        self.episodes_by_num: dict[int, Episode] = {ep.number: ep for ep in episodes}
        # Build global index of all events by ID
        self.events_by_id: dict[str, Event] = {}
        # Order map for events within each episode
        self.event_order: dict[str, int] = {}

        for ep in episodes:
            for idx, ev in enumerate(ep.events):
                self.events_by_id[ev.id] = ev
                self.event_order[ev.id] = idx

        self.rejection_tally: dict[str, int] = {
            "unknown_id": 0,
            "invalid_event_order": 0,
            "cross_episode_ordering": 0,
            "segment_overlap": 0,
            "hook_after_3s": 0,
            "cliffhanger_misplaced": 0,
            "duration_out_of_bounds": 0,
            "empty_segments": 0,
        }

    def _record_rejection(self, rule: str, reason: str) -> ValidationResult:
        self.rejection_tally[rule] = self.rejection_tally.get(rule, 0) + 1
        return ValidationResult(promo=None, is_valid=False, rejection_rule=rule, rejection_reason=reason)

    def validate_candidate(self, candidate: dict[str, Any], candidate_index: int = 1) -> ValidationResult:
        promo_id = candidate.get("promo_id") or f"P{candidate_index:03d}"
        raw_segments = candidate.get("segments", [])
        hook_id = candidate.get("hook_event_id", "").strip()
        cliffhanger_id = candidate.get("cliffhanger_event_id", "").strip()
        reason = candidate.get("reason", "").strip()

        if not raw_segments:
            return self._record_rejection("empty_segments", "Candidate has no segments")

        # 1. Verify all event IDs exist
        all_candidate_ids = [hook_id, cliffhanger_id]
        for seg in raw_segments:
            all_candidate_ids.append(seg.get("start_event_id", ""))
            all_candidate_ids.append(seg.get("end_event_id", ""))

        for ev_id in all_candidate_ids:
            if not ev_id or ev_id not in self.events_by_id:
                return self._record_rejection("unknown_id", f"Unknown or empty event ID: '{ev_id}'")

        hook_ev = self.events_by_id[hook_id]
        cliff_ev = self.events_by_id[cliffhanger_id]

        repairs: list[str] = []

        # 2. Build and snap PromoSegment objects
        assembled_segments: list[PromoSegment] = []
        for s_idx, seg_dict in enumerate(raw_segments):
            s_start_id = seg_dict["start_event_id"]
            s_end_id = seg_dict["end_event_id"]

            start_ev = self.events_by_id[s_start_id]
            end_ev = self.events_by_id[s_end_id]

            if start_ev.episode != end_ev.episode:
                return self._record_rejection(
                    "cross_episode_ordering",
                    f"Segment {s_idx} spans across episodes ({start_ev.episode} to {end_ev.episode})"
                )

            ep = self.episodes_by_num[start_ev.episode]
            start_order = self.event_order[s_start_id]
            end_order = self.event_order[s_end_id]

            if start_order > end_order:
                return self._record_rejection(
                    "invalid_event_order",
                    f"Segment start ({s_start_id}, order {start_order}) after end ({s_end_id}, order {end_order})"
                )

            # Snap slice to all whole events
            seg_events = ep.events[start_order : end_order + 1]
            seg_start_time = seg_events[0].start
            seg_end_time = seg_events[-1].end

            assembled_segments.append(
                PromoSegment(
                    episode=ep.number,
                    start_event_id=s_start_id,
                    end_event_id=s_end_id,
                    start=seg_start_time,
                    end=seg_end_time,
                    start_ts=format_timestamp(seg_start_time),
                    end_ts=format_timestamp(seg_end_time),
                    events=seg_events,
                )
            )

        # 3. Check segment ordering and overlap
        for i in range(len(assembled_segments) - 1):
            cur = assembled_segments[i]
            nxt = assembled_segments[i + 1]
            if cur.episode > nxt.episode:
                return self._record_rejection(
                    "cross_episode_ordering",
                    f"Segment episodes out of order: Ep {cur.episode} after Ep {nxt.episode}"
                )
            if cur.episode == nxt.episode:
                if cur.end > nxt.start or self.event_order[cur.end_event_id] >= self.event_order[nxt.start_event_id]:
                    return self._record_rejection(
                        "segment_overlap",
                        f"Segments {i} and {i+1} in episode {cur.episode} overlap or are out of order"
                    )

        # 4. Hook rule check & repair
        # Hook event must start within the first 3.0s of the assembled promo
        first_seg = assembled_segments[0]
        # In assembled promo, offset of hook from start of first segment
        first_seg_order = self.event_order[first_seg.start_event_id]
        
        # Check if hook_ev is in the first segment
        if hook_ev.episode != first_seg.episode or self.event_order[hook_id] < first_seg_order:
            return self._record_rejection("hook_after_3s", f"Hook event {hook_id} is before promo start")

        hook_offset = hook_ev.start - first_seg.start
        if hook_offset > 3.0:
            # Attempt repair: shift first segment start to hook_ev
            hook_order = self.event_order[hook_id]
            ep = self.episodes_by_num[first_seg.episode]
            end_order = self.event_order[first_seg.end_event_id]
            if hook_order <= end_order:
                repaired_events = ep.events[hook_order : end_order + 1]
                first_seg.start_event_id = hook_id
                first_seg.start = repaired_events[0].start
                first_seg.start_ts = format_timestamp(first_seg.start)
                first_seg.events = repaired_events
                hook_offset = 0.0
                repairs.append(f"Trimmed leading events in segment 1 to snap hook {hook_id} to 0.0s")
            else:
                return self._record_rejection("hook_after_3s", f"Hook event {hook_id} occurs at +{hook_offset:.1f}s (>3s)")

        # 5. Cliffhanger rule check & repair
        # Promo must end at or immediately following cliffhanger event
        last_seg = assembled_segments[-1]
        last_seg_end_order = self.event_order[last_seg.end_event_id]
        if cliff_ev.episode != last_seg.episode:
            return self._record_rejection("cliffhanger_misplaced", f"Cliffhanger in ep {cliff_ev.episode}, promo ends in ep {last_seg.episode}")

        cliff_order = self.event_order[cliffhanger_id]
        if cliff_order > last_seg_end_order:
            return self._record_rejection("cliffhanger_misplaced", f"Cliffhanger {cliffhanger_id} is after promo end")

        # If cliffhanger is before last event, check distance
        cliff_dist = last_seg.end - cliff_ev.end
        if cliff_dist > 4.0:
            # Attempt repair: snap last segment end to cliffhanger event
            cliff_start_order = self.event_order[last_seg.start_event_id]
            if cliff_order >= cliff_start_order:
                ep = self.episodes_by_num[last_seg.episode]
                repaired_events = ep.events[cliff_start_order : cliff_order + 1]
                last_seg.end_event_id = cliffhanger_id
                last_seg.end = repaired_events[-1].end
                last_seg.end_ts = format_timestamp(last_seg.end)
                last_seg.events = repaired_events
                repairs.append(f"Snapped last segment end to cliffhanger {cliffhanger_id}")
            else:
                return self._record_rejection("cliffhanger_misplaced", f"Cliffhanger {cliffhanger_id} ends {cliff_dist:.1f}s before promo cut")

        # 6. Duration bounds check & whole-event repair (20.0s - 60.0s)
        total_duration = sum(s.duration for s in assembled_segments)

        # Attempt extension if slightly short (15s to <20s)
        if total_duration < 20.0:
            # Try appending adjacent events to last segment within same scene if available
            ep = self.episodes_by_num[last_seg.episode]
            cur_end_order = self.event_order[last_seg.end_event_id]
            extended = False
            while total_duration < 20.0 and cur_end_order + 1 < len(ep.events):
                next_ev = ep.events[cur_end_order + 1]
                # Keep within reasonable proximity and scene
                if next_ev.scene == ep.events[cur_end_order].scene:
                    cur_end_order += 1
                    last_seg.end_event_id = next_ev.id
                    last_seg.end = next_ev.end
                    last_seg.end_ts = format_timestamp(next_ev.end)
                    last_seg.events = ep.events[self.event_order[last_seg.start_event_id] : cur_end_order + 1]
                    total_duration = sum(s.duration for s in assembled_segments)
                    extended = True
                else:
                    break
            if extended and total_duration >= 20.0:
                repairs.append(f"Extended promo to {total_duration:.1f}s by appending scene events")

        # Attempt trimming if slightly over (60s to 75s)
        if total_duration > 60.0:
            # Try trimming from end if cliffhanger wasn't at the very edge, or trim non-hook leading events
            ep = self.episodes_by_num[last_seg.episode]
            cur_end_order = self.event_order[last_seg.end_event_id]
            cliff_order = self.event_order[cliffhanger_id]
            trimmed = False
            while total_duration > 60.0 and cur_end_order > cliff_order:
                cur_end_order -= 1
                last_ev = ep.events[cur_end_order]
                last_seg.end_event_id = last_ev.id
                last_seg.end = last_ev.end
                last_seg.end_ts = format_timestamp(last_ev.end)
                last_seg.events = ep.events[self.event_order[last_seg.start_event_id] : cur_end_order + 1]
                total_duration = sum(s.duration for s in assembled_segments)
                trimmed = True
            if trimmed and total_duration <= 60.0:
                repairs.append(f"Trimmed trailing events to {total_duration:.1f}s")

        if total_duration < 20.0 or total_duration > 60.0:
            return self._record_rejection(
                "duration_out_of_bounds",
                f"Total duration {total_duration:.1f}s outside strict 20.0s-60.0s window"
            )

        is_cross = len({s.episode for s in assembled_segments}) > 1

        validated_promo = ValidatedPromo(
            promo_id=promo_id,
            segments=assembled_segments,
            hook_event_id=hook_id,
            cliffhanger_event_id=cliffhanger_id,
            reason=reason,
            total_duration=round(total_duration, 2),
            hook_offset=round(hook_offset, 2),
            is_cross_episode=is_cross,
            repairs_applied=repairs,
        )
        return ValidationResult(promo=validated_promo, is_valid=True)
