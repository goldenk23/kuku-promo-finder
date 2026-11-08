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

## Chunk 2: Promo Finder & LLM Integration

1. **Mutually Exclusive Google GenAI Client Arguments (Coding Assistant)**:
   - *Error*: Passed `project` and `location` alongside `api_key` to `genai.Client(vertexai=True, api_key=...)`.
   - *Evidence*: `ValueError: Project/location and API key are mutually exclusive in the client initializer.`
   - *Resolution*: Updated `llm.py` so `project` and `location` are omitted when `api_key` is provided.

2. **Hallucinated Event ID in Candidate Generation (Gemini 2.5 Flash)**:
   - *Error*: In candidate generation for Episode 4 / cross-episode proposals, Gemini generated a reference to an invented event ID that did not exist in the screenplay transcript.
   - *Evidence*: `RuleValidator` rejected candidate with rule `unknown_id` (`rejections.json`).
   - *Resolution*: The validator caught and quarantined the proposal without crashing the pipeline, proving the necessity of ID snapping over trusting LLM output.

3. **Duration Out of Bounds in LLM Candidate Generation (Gemini 2.5 Flash)**:
   - *Error*: Despite explicit prompt instructions requiring clips between 20.0s and 60.0s, Gemini proposed 3 clips outside the range (e.g. <18s or >65s) where whole-event expansion/trimming could not salvage the cut.
   - *Evidence*: `RuleValidator` rejected 3 candidate promos under `duration_out_of_bounds`.
   - *Resolution*: Rule validator rejected the invalid duration clips rather than allowing bad clips through.

4. **Vertex AI Free-Tier 429 Rate Limiting**:
   - *Error*: Rapid sequential calls for cold-viewer evaluation overwhelmed the per-minute quota on the trial key.
   - *Evidence*: `429 RESOURCE_EXHAUSTED` returned by Vertex AI.
   - *Resolution*: Implemented jittered exponential backoff and pacing in `llm.py`.
