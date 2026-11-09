"""Screenplay parser for microdrama episodes.

Parses episode text files into structured dataclasses:
Episode -> Scene -> Event.
Handles format variations, case sensitivity, and logs warnings for malformed blocks without crashing.
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("promo_finder.parse")

TIME_RE = re.compile(r"Start:\s*\(?(\d+):([0-9.]+)\)?\s*\|\s*End:\s*\(?(\d+):([0-9.]+)\)?")
TAG_RE = re.compile(r"^\*\*([^*]+?):\*\*\s*$")
SPEAKER_EMOTION_RE = re.compile(r"^(.+?)\s*\((.+)\)$")
SCENE_HDR_RE = re.compile(r"(?i)^SCENE\s+(\d+)\s*$")
SCENE_HEADING_RE = re.compile(r"(?i)^SCENE\s+HEADING:\s*(.*)$")
SCENE_START_RE = re.compile(r"(?i)^SCENE\s+START:\s*\(?(\d+):([0-9.]+)\)?")
SCENE_END_RE = re.compile(r"(?i)^SCENE\s+END:\s*\(?(\d+):([0-9.]+)\)?")
EPISODE_NUM_RE = re.compile(r"(?i)^EPISODE\s+NUMBER:\s*Episode\s+(\d+)")
EPISODE_LOGLINE_RE = re.compile(r"(?i)^EPISODE\s+LOGLINE:\s*(.*)$")
CONTINUITY_RE = re.compile(r"(?i)^CONTINUITY\s+NOTES:\s*(.*)$")


def parse_timestamp(minutes_str: str | int, seconds_str: str | float) -> float:
    """Convert MM:SS.ss to seconds float."""
    return int(minutes_str) * 60.0 + float(seconds_str)


def format_timestamp(seconds: float) -> str:
    """Convert seconds float to MM:SS.ss format."""
    mins = int(seconds // 60)
    secs = seconds - (mins * 60)
    return f"{mins:02d}:{secs:05.2f}"


@dataclass
class Event:
    """A single screenplay event (camera direction, action, or dialogue line)."""
    id: str  # e.g. 'E01-S01-001'
    episode: int
    scene: int
    kind: str  # 'camera' | 'action' | 'dialogue' | 'other'
    speaker: str | None
    emotion: str | None
    start: float  # seconds
    end: float  # seconds
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def line(self) -> str:
        """Human-readable one-liner, e.g. 'MATRIARCH: Not anymore.' or '[ACTION] She slaps him.'"""
        if self.kind == "dialogue":
            return f"{self.speaker or 'SPEAKER'}: {self.text}"
        return f"[{self.kind.upper()}] {self.text}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "episode": self.episode,
            "scene": self.scene,
            "kind": self.kind,
            "speaker": self.speaker,
            "emotion": self.emotion,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "start_ts": format_timestamp(self.start),
            "end_ts": format_timestamp(self.end),
            "duration": round(self.duration, 2),
            "text": self.text,
        }


@dataclass
class Scene:
    """A scene within an episode."""
    number: int
    heading: str
    start: float
    end: float
    events: list[Event] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Episode:
    """A single episode with metadata, scenes, and flattened events."""
    number: int
    logline: str
    continuity: str
    scenes: list[Scene] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.scenes:
            return max(s.end for s in self.scenes)
        if self.events:
            return max(e.end for e in self.events)
        return 0.0


def parse_episode_text(text: str, filename: str = "<memory>") -> Episode:
    """Parse raw episode script text into an Episode object.
    
    Tolerates variations in casing, spacing, and records warnings instead of failing.
    """
    lines = text.splitlines()
    warnings: list[str] = []

    episode_num: int | None = None
    logline = ""
    continuity = ""

    # Parse metadata header before --- SCRIPT START ---
    script_start_idx = 0
    for idx, line in enumerate(lines):
        line_clean = line.strip()
        if not episode_num:
            ep_m = EPISODE_NUM_RE.match(line_clean)
            if ep_m:
                episode_num = int(ep_m.group(1))
        if not logline:
            ll_m = EPISODE_LOGLINE_RE.match(line_clean)
            if ll_m:
                logline = ll_m.group(1).strip()
        if not continuity:
            cn_m = CONTINUITY_RE.match(line_clean)
            if cn_m:
                continuity = cn_m.group(1).strip()
        if "--- SCRIPT START ---" in line.upper():
            script_start_idx = idx + 1
            break

    if episode_num is None:
        # Fallback: extract episode number from filename
        fn_match = re.search(r"episode-?(\d+)", filename, re.IGNORECASE)
        if fn_match:
            episode_num = int(fn_match.group(1))
            warnings.append(f"{filename}: Missing EPISODE NUMBER in header, inferred {episode_num} from filename")
        else:
            episode_num = 1
            warnings.append(f"{filename}: Missing EPISODE NUMBER, defaulted to 1")

    # Split lines into scenes
    # A scene starts with SCENE XX
    scene_blocks: list[tuple[int, list[tuple[int, str]]]] = []  # (scene_num, [(line_no, line)])
    curr_scene_num: int | None = None
    curr_scene_lines: list[tuple[int, str]] = []

    for idx in range(script_start_idx, len(lines)):
        line = lines[idx]
        sc_m = SCENE_HDR_RE.match(line.strip())
        if sc_m:
            if curr_scene_num is not None and curr_scene_lines:
                scene_blocks.append((curr_scene_num, curr_scene_lines))
                curr_scene_lines = []
            curr_scene_num = int(sc_m.group(1))
        elif curr_scene_num is not None:
            curr_scene_lines.append((idx + 1, line))

    if curr_scene_num is not None and curr_scene_lines:
        scene_blocks.append((curr_scene_num, curr_scene_lines))

    scenes: list[Scene] = []
    all_events: list[Event] = []
    event_counter = 0

    for sc_num, sc_lines in scene_blocks:
        heading = ""
        sc_start = 0.0
        sc_end = 0.0
        sc_events: list[Event] = []

        i = 0
        while i < len(sc_lines):
            line_no, line_str = sc_lines[i]
            line_clean = line_str.strip()

            # Heading
            hd_m = SCENE_HEADING_RE.match(line_clean)
            if hd_m:
                heading = hd_m.group(1).strip()
                i += 1
                continue

            # Scene start
            st_m = SCENE_START_RE.match(line_clean)
            if st_m:
                sc_start = parse_timestamp(st_m.group(1), st_m.group(2))
                i += 1
                continue

            # Scene end
            end_m = SCENE_END_RE.match(line_clean)
            if end_m:
                sc_end = parse_timestamp(end_m.group(1), end_m.group(2))
                i += 1
                continue

            # Check for event tag
            tag_m = TAG_RE.match(line_clean)
            if tag_m:
                raw_tag = tag_m.group(1).strip()
                # Next line must be timestamp
                if i + 1 < len(sc_lines):
                    t_line_no, t_line_str = sc_lines[i + 1]
                    time_m = TIME_RE.search(t_line_str)
                    if time_m:
                        ev_start = parse_timestamp(time_m.group(1), time_m.group(2))
                        ev_end = parse_timestamp(time_m.group(3), time_m.group(4))

                        if ev_start > ev_end:
                            warnings.append(
                                f"{filename}:{t_line_no}: Event start ({ev_start}s) is greater than end ({ev_end}s)"
                            )

                        # Collect text lines until next tag or blank section
                        j = i + 2
                        text_parts: list[str] = []
                        while j < len(sc_lines):
                            next_lno, next_lstr = sc_lines[j]
                            next_clean = next_lstr.strip()
                            if TAG_RE.match(next_clean) or SCENE_HDR_RE.match(next_clean):
                                break
                            if next_clean:
                                text_parts.append(next_clean)
                            j += 1

                        ev_text = " ".join(text_parts).strip()
                        # If dialogue is quoted, preserve text cleanly
                        if ev_text.startswith('"') and ev_text.endswith('"'):
                            ev_text = ev_text[1:-1].strip()

                        # Classify kind
                        speaker: str | None = None
                        emotion: str | None = None

                        if raw_tag.upper() == "CAMERA":
                            kind = "camera"
                        elif raw_tag.upper() == "ACTION":
                            kind = "action"
                        elif any(s in raw_tag.upper() for s in ["SOUND", "SFX", "MUSIC", "AUDIO"]):
                            kind = "other"
                        else:
                            kind = "dialogue"
                            spk_m = SPEAKER_EMOTION_RE.match(raw_tag)
                            if spk_m:
                                speaker = spk_m.group(1).strip()
                                emotion = spk_m.group(2).strip()
                            else:
                                speaker = raw_tag
                                emotion = None

                        event_counter += 1
                        ev_id = f"E{episode_num:02d}-S{sc_num:02d}-{event_counter:03d}"

                        # Check scene boundary containment
                        # Note: Small boundary bleeds (e.g. sound fx trailing 0.3s) are logged as warnings
                        if sc_end > 0 and (ev_start < sc_start - 0.05 or ev_end > sc_end + 0.05):
                            warnings.append(
                                f"{filename}:{t_line_no}: Event {ev_id} bounds ({ev_start:.2f}-{ev_end:.2f}s) "
                                f"exceed scene {sc_num} bounds ({sc_start:.2f}-{sc_end:.2f}s)"
                            )

                        ev = Event(
                            id=ev_id,
                            episode=episode_num,
                            scene=sc_num,
                            kind=kind,
                            speaker=speaker,
                            emotion=emotion,
                            start=ev_start,
                            end=ev_end,
                            text=ev_text,
                        )
                        sc_events.append(ev)
                        all_events.append(ev)

                        i = j
                        continue
                    else:
                        warnings.append(
                            f"{filename}:{line_no}: Found tag '{raw_tag}' but following line lacked timestamp: {t_line_str.strip()}"
                        )
                else:
                    warnings.append(
                        f"{filename}:{line_no}: Tag '{raw_tag}' at end of file with no timestamp"
                    )

            i += 1

        scene_obj = Scene(
            number=sc_num,
            heading=heading,
            start=sc_start,
            end=sc_end if sc_end > 0 else (max([e.end for e in sc_events], default=0.0)),
            events=sc_events,
        )
        scenes.append(scene_obj)

    return Episode(
        number=episode_num,
        logline=logline,
        continuity=continuity,
        scenes=scenes,
        events=all_events,
        warnings=warnings,
    )


def parse_episode_file(filepath: str | Path) -> Episode:
    """Parse an episode file from path."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_episode_text(text, filename=path.name)


def load_episodes(source: str | Path) -> list[Episode]:
    """Load and parse all episodes from a directory or a zip archive, sorted by episode number."""
    source_path = Path(source)
    episodes: list[Episode] = []

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path, "r") as zf:
            namelist = [n for n in zf.namelist() if n.lower().endswith(".txt") and not n.startswith("__MACOSX")]
            for name in namelist:
                with zf.open(name) as f:
                    text = f.read().decode("utf-8", errors="replace")
                episodes.append(parse_episode_text(text, filename=Path(name).name))
    elif source_path.is_dir():
        # Walk to find all .txt files
        txt_files: list[Path] = []
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(".txt"):
                    txt_files.append(Path(root) / file)
        for fpath in txt_files:
            episodes.append(parse_episode_file(fpath))
    else:
        raise ValueError(f"Source path {source} is neither a directory nor a zip file.")

    episodes.sort(key=lambda ep: ep.number)
    return episodes
