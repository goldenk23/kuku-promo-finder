# Architecture & Design Decisions

## Chunk 1: Foundation (Parser & Swappable LLM Client)

- **ID-Snapped Event Representation**:
  - *Decision*: Parse all episode scripts into discrete, timestamped `Event` objects with stable string IDs (e.g. `E01-S01-001`).
  - *Alternatives*: Let the LLM output raw start/end timestamps directly from text.
  - *Why*: LLMs consistently hallucinate or misalign floating-point timestamps. By forcing the LLM to pick IDs and computing timestamps in pure code, hallucinated timestamps are eliminated by construction.

- **Parser Robustness and Warning Strategy**:
  - *Decision*: Case-insensitive parsing for scene headings, timestamps, and action/dialogue markers. Log warnings for malformed or out-of-boundary blocks without crashing.
  - *Observed variants*:
    - Episode 6 used ALL-CAPS headers (`SCENE START`, `ACTIONS AND DIALOGUE`).
    - Episode 22 line 103 had a sound effect latch event (`41.50-42.80s`) bleeding 0.3s past Scene 1's end (`42.50s`).
  - *Why*: Allows the pipeline to run on new, unseen shows with minor formatting variations during live interview tests.

- **LLM Provider & Model Choice**:
  - *Decision*: Default to `gemini-2.5-flash` on Vertex AI via `google-genai` with `vertexai=True`. Support `openai_compat` as fallback.
  - *Alternatives*: `gemini-2.5-pro` (unavailable on project / 404), `gemini-3.1-pro-preview` (too expensive and slow for batch candidate generation), `gemini-3.1-flash-lite`.
  - *Why*: `gemini-2.5-flash` provides fast inference, high adherence to JSON schemas, and low cost per 1M tokens.

- **Deterministic Disk Caching**:
  - *Decision*: SHA256 hash of `(provider, model, temperature, prompt, schema)` stored in `.cache/llm/`.
  - *Why*: Enables free, deterministic, and instant repeated runs and quality checks.

- **Usage Auditing**:
  - *Decision*: Every LLM call records prompt tokens, candidate tokens, duration, and cache hit status to `outputs/usage.jsonl`.
  - *Why*: Accurately tracks per-show cost without manual estimation.

## Chunk 2: Promo Finder (Candidates, Rule Validation, Cold-Viewer Judge)

- **Compact Transcript Design**:
  - *Decision*: Render events compactly as `[E01-S01-001 00:00.00-00:04.80] KIND SPEAKER: text`, truncating lengthy camera descriptions to 70 chars.
  - *Why*: Keeps prompt tokens low (~1,000 tokens per episode) while providing exact IDs and time bounds to the model.

- **Two-Pass Candidate Generation (Pass A + Pass B)**:
  - *Decision*: Pass A extracts up to 4 promos per individual episode. Pass B synthesizes cross-episode arcs (combining betrayal in Ep 1 with confrontation in later episodes).
  - *Why*: Covers both single-episode self-contained clips and high-converting cross-episode arcs without overwhelming context limits.

- **Strict Code-Enforced Rule Validation with Whole-Event Repairs**:
  - *Decision*: Validate candidates purely in Python (`RuleValidator`). If duration or hook is slightly off, attempt whole-event trimming/extending without cutting dialogue mid-sentence. Reject candidates that violate non-negotiables (unknown IDs, duration <20s or >60s after repair, misplaced cliffhanger).
  - *Why*: Editorial integrity; no promo is published with invalid cuts or hallucinated IDs.

- **Cold-Viewer Judge Architecture**:
  - *Decision*: A dedicated LLM call sees ONLY the promo's own dialogue and actions without series context or loglines. Evaluates hook stopping power (1-5), cliffhanger tension (1-5), and standalone clarity (1-5).
  - *Why*: Directly simulates a scrolling stranger encountering the ad on TikTok or Instagram Reels.
