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

4. **Vertex AI Free-Tier 429 Rate Limiting & Multi-Threaded Deadlock**:
   - *Error*: Rapid concurrent calls across worker threads for cold-viewer evaluation overwhelmed the 15 RPM quota on the Vertex AI trial key, causing threads to retry at synchronized intervals and collide repeatedly.
   - *Evidence*: Repeated `429 RESOURCE_EXHAUSTED` in worker logs with escalating backoffs.
   - *Resolution*: Implemented thread-safe interval rate limiting (`_rate_limit_lock` enforcing `_MIN_CALL_INTERVAL = 3.6s`) and randomized jitter on backoff in `llm.py`, eliminating 429 collisions and achieving steady throughput.

## Chunk 3: Ranking & Quality Assurance

1. **Adversarial Calibration Verification**:
   - *Test*: Ran 5 defective synthetic probe clips through the Cold-Viewer Judge (passive camera, premature cut, mid-sentence cut, zero-stakes monologue, disjointed scenes).
   - *Outcome*: Judge appropriately identified the defects and scored them 1/5 and 2/5 on hook and cliffhanger strength, demonstrating that the evaluation prompt is discriminatory and does not hallucinate high scores for low-quality content.

2. **Deduplication Thresholding**:
   - *Observation*: High-conflict episodes (e.g. climax episodes) produced multiple candidate clips that shared 70-90% of the same dialogue events.
   - *Resolution*: Temporal deduplication with an Intersection-over-Union (IoU) threshold of >50% successfully pruned duplicate cuts, retaining only the highest-scoring variation and preserving variety in the final 15-promo package.
