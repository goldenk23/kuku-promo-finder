"""Runnable check for screenplay parser using purely synthetic invented data.

Tests:
1. Normal parsing of scenes, events, timestamps, and kinds.
2. Event ID sequence formatting.
3. Speaker and emotion extraction.
4. Warning paths for scene boundary overflows and malformed blocks without crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from promo_finder.parse import parse_episode_text, format_timestamp


SYNTHETIC_SCRIPT = """EPISODE NUMBER: Episode 99
EPISODE LOGLINE: A detective enters a clock shop and hears a ticking mechanism just before an explosion.
CONTINUITY NOTES: Test episode with synthetic text.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CLOCK SHOP - NIGHT
Scene start: 00:00.00
Scene end: 00:30.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:03.50)
Close up on antique pendulum clock ticking slowly.

**ACTION:**
Start: (00:01.00) | End: (00:04.00)
Detective Vance pushes the creaking wooden door open.

**DETECTIVE VANCE (Suspicious):**
Start: (00:03.00) | End: (00:06.50)
"Is anyone still here?"

**SOUND FX:**
Start: (00:06.50) | End: (00:08.00)
Loud metallic click echoed in the silence.

SCENE 02
Scene heading: INT. CLOCK SHOP - BACK ROOM - CONTINUOUS
Scene start: 00:30.00
Scene end: 01:00.00

Actions and dialogue:

**CAMERA:**
Start: (00:30.00) | End: (00:33.00)
Tracking shot following Vance as he unholsters his flashlight.

**SHOPKEEPER (Panicked, Whispering):**
Start: (00:32.50) | End: (00:35.00)
"Do not step on the floorboard!"
"""

SYNTHETIC_MALFORMED_SCRIPT = """EPISODE NUMBER: Episode 88
EPISODE LOGLINE: Synthetic test for warnings.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. TEST ROOM - DAY
Scene start: 00:00.00
Scene end: 00:20.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:25.00)
Camera exceeds scene end time.

**ORPHAN TAG:**
No timestamp line here at all.

**ACTION:**
Start: (00:10.00) | End: (00:05.00)
Start is after end.
"""


def test_synthetic_parsing() -> None:
    print("Running test_synthetic_parsing...")
    ep = parse_episode_text(SYNTHETIC_SCRIPT, filename="synthetic_99.txt")

    # Header assertions
    assert ep.number == 99, f"Expected ep 99, got {ep.number}"
    assert "detective" in ep.logline.lower(), f"Unexpected logline: {ep.logline}"
    assert len(ep.scenes) == 2, f"Expected 2 scenes, got {len(ep.scenes)}"
    assert len(ep.events) == 6, f"Expected 6 events, got {len(ep.events)}"
    assert len(ep.warnings) == 0, f"Expected 0 warnings, got {ep.warnings}"

    # Scene 1 checks
    sc1 = ep.scenes[0]
    assert sc1.number == 1
    assert sc1.start == 0.0
    assert sc1.end == 30.0
    assert len(sc1.events) == 4

    # Event ID checks
    assert sc1.events[0].id == "E99-S01-001"
    assert sc1.events[1].id == "E99-S01-002"
    assert sc1.events[2].id == "E99-S01-003"
    assert sc1.events[3].id == "E99-S01-004"

    # Kinds checks
    assert sc1.events[0].kind == "camera"
    assert sc1.events[1].kind == "action"
    assert sc1.events[2].kind == "dialogue"
    assert sc1.events[3].kind == "other"

    # Dialogue speaker and emotion checks
    ev_dia = sc1.events[2]
    assert ev_dia.speaker == "DETECTIVE VANCE"
    assert ev_dia.emotion == "Suspicious"
    assert ev_dia.text == "Is anyone still here?"
    assert ev_dia.start == 3.0
    assert ev_dia.end == 6.5
    assert ev_dia.duration == 3.5

    # Scene 2 checks
    sc2 = ep.scenes[1]
    assert sc2.number == 2
    assert sc2.start == 30.0
    assert sc2.end == 60.0
    assert len(sc2.events) == 2
    assert sc2.events[0].id == "E99-S02-005"
    assert sc2.events[1].id == "E99-S02-006"
    assert sc2.events[1].speaker == "SHOPKEEPER"
    assert sc2.events[1].emotion == "Panicked, Whispering"

    # Duration check
    assert ep.duration == 60.0
    print("  -> test_synthetic_parsing PASSED.")


def test_synthetic_warnings() -> None:
    print("Running test_synthetic_warnings...")
    ep = parse_episode_text(SYNTHETIC_MALFORMED_SCRIPT, filename="synthetic_malformed.txt")

    assert ep.number == 88
    # Must not crash, but must record warnings
    assert len(ep.warnings) >= 2, f"Expected at least 2 warnings, got: {ep.warnings}"

    # Verify warning contents
    has_bound_warn = any("exceed scene" in w for w in ep.warnings)
    has_order_warn = any("start" in w and "greater than end" in w for w in ep.warnings)
    has_orphan_warn = any("No timestamp" in w or "lacked timestamp" in w for w in ep.warnings)

    assert has_bound_warn, f"Expected scene boundary warning, got: {ep.warnings}"
    assert has_order_warn, f"Expected start > end warning, got: {ep.warnings}"
    assert has_orphan_warn, f"Expected orphan tag warning, got: {ep.warnings}"

    print(f"  -> test_synthetic_warnings PASSED ({len(ep.warnings)} warnings properly captured).")


if __name__ == "__main__":
    test_synthetic_parsing()
    test_synthetic_warnings()
    print("\nALL PARSE CHECKS PASSED SUCCESSFULLY.")
