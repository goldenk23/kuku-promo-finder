# AI Error Log

Documenting concrete mistakes made by AI models (both development assistants and LLM inference calls) with evidence.

## Chunk 1: Foundation

1. **Invalid Python Syntax in Check Script (Coding Assistant)**:
   - *Error*: In `tests/check_parse.py`, attempted to use a walrus operator inside a generator comprehension: `any("exceed scene" in w for w in w_list := ep.warnings)`.
   - *Evidence*: `SyntaxError: invalid syntax` raised by Python 3.13.
   - *Resolution*: Replaced with standard iteration `any("exceed scene" in w for w in ep.warnings)`.

2. **Off-by-One Scene Count from Preamble Whitespace (Coding Assistant)**:
   - *Error*: Initial parser logic accumulated lines before `SCENE 01` into a default Scene 1 block, creating an empty ghost scene when scripts contained blank lines between `--- SCRIPT START ---` and `SCENE 01`.
   - *Evidence*: `tests/check_parse.py` failed with `AssertionError: Expected 2 scenes, got 3`.
   - *Resolution*: Parser refactored so `curr_scene_num` initializes as `None` and only begins scene accumulation once the first `SCENE XX` marker is matched.
