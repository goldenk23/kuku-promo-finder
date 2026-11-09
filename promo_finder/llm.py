"""Swappable LLM client supporting Google Vertex AI and OpenAI-compatible endpoints.

Key features:
1. Swappable via environment variables (LLM_PROVIDER=vertex | openai_compat).
2. Local disk caching keyed by SHA256 of request parameters.
3. Exponential backoff retry on transient errors.
4. Token usage tracking written to outputs/usage.jsonl.
5. Zero secrets logged or exposed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env if present
load_dotenv()

import random
import threading

logger = logging.getLogger("promo_finder.llm")

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache" / "llm"
USAGE_LOG_PATH = Path(os.getenv("USAGE_LOG_PATH", str(REPO_ROOT / "outputs" / "usage.jsonl")))
_usage_lock = threading.Lock()
_rate_limit_lock = threading.Lock()
_last_call_time = 0.0
# Minimum seconds between outgoing API calls across all threads (protects against Vertex AI 15 RPM quota)
_MIN_CALL_INTERVAL = 3.6


def _get_cache_key(
    provider: str,
    model: str,
    prompt: str,
    schema: dict[str, Any] | None,
    system_instruction: str | None,
    temperature: float,
) -> str:
    """Compute deterministic SHA256 cache key."""
    schema_str = json.dumps(schema, sort_keys=True) if schema else ""
    raw = f"{provider}:{model}:{temperature}:{system_instruction or ''}:{prompt}:{schema_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _log_usage(
    provider: str,
    model: str,
    prompt_tokens: int,
    candidates_tokens: int,
    cached: bool,
    duration_sec: float,
    label: str = "general",
) -> None:
    """Append call token usage to usage.jsonl."""
    USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "candidates_tokens": candidates_tokens,
        "total_tokens": prompt_tokens + candidates_tokens,
        "cached": cached,
        "duration_sec": round(duration_sec, 3),
    }
    try:
        with _usage_lock:
            with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except OSError as err:
        logger.warning("Failed to write usage log: %s", err)


class LLMClient:
    """Unified client for structured JSON generation."""

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "vertex").lower()
        if self.provider == "vertex":
            self.model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
            self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
            self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
            self.api_key = os.getenv("GEMINI_API_KEY")
            self._init_vertex()
        elif self.provider == "openai_compat":
            self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            self.api_key = os.getenv("LLM_API_KEY", "")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}. Use 'vertex' or 'openai_compat'.")

    def _init_vertex(self) -> None:
        from google import genai
        # Initialize Google GenAI client for Vertex AI
        kwargs: dict[str, Any] = {"vertexai": True}
        if self.api_key:
            # When API key is provided, project/location are mutually exclusive
            kwargs["api_key"] = self.api_key
        else:
            if self.project:
                kwargs["project"] = self.project
            if self.location:
                kwargs["location"] = self.location
        self._genai_client = genai.Client(**kwargs)

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        use_cache: bool = True,
        label: str = "general",
        max_retries: int = 6,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate structured JSON from prompt.
        
        Returns:
            (parsed_json_dict, usage_dict)
        """
        cache_key = _get_cache_key(
            self.provider, self.model, prompt, schema, system_instruction, temperature
        )
        cache_file = CACHE_DIR / f"{cache_key}.json"

        # Check cache
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                usage = {
                    "prompt_tokens": 0,
                    "candidates_tokens": 0,
                    "total_tokens": 0,
                    "cached": True,
                    "duration_sec": 0.0,
                }
                _log_usage(self.provider, self.model, 0, 0, True, 0.0, label)
                return cached_data, usage
            except Exception as err:
                logger.warning("Cache read failed for key %s: %s", cache_key, err)

        # Call model with retries
        start_time = time.time()
        result_json: dict[str, Any] = {}
        prompt_tokens = 0
        candidates_tokens = 0

        delay = 3.0
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                # Thread-safe rate limiter to strictly observe RPM bounds
                with _rate_limit_lock:
                    global _last_call_time
                    now = time.time()
                    elapsed = now - _last_call_time
                    if elapsed < _MIN_CALL_INTERVAL:
                        time.sleep(_MIN_CALL_INTERVAL - elapsed)
                    _last_call_time = time.time()

                if self.provider == "vertex":
                    result_json, prompt_tokens, candidates_tokens = self._call_vertex(
                        prompt=prompt,
                        schema=schema,
                        system_instruction=system_instruction,
                        temperature=temperature,
                    )
                else:
                    result_json, prompt_tokens, candidates_tokens = self._call_openai_compat(
                        prompt=prompt,
                        schema=schema,
                        system_instruction=system_instruction,
                        temperature=temperature,
                    )
                break
            except Exception as err:
                last_error = err
                err_str = str(err)
                # If rate limited (429), back off with jitter
                wait_time = delay + random.uniform(1.0, 3.0)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = max(delay, 5.0) + random.uniform(2.0, 5.0)
                logger.warning(
                    "LLM call attempt %d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    err,
                    wait_time,
                )
                if attempt == max_retries - 1:
                    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {err}") from err
                time.sleep(wait_time)
                delay *= 2.0

        duration = time.time() - start_time

        # Save to cache
        if use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, indent=2)
            except OSError as err:
                logger.warning("Failed to write cache file %s: %s", cache_file, err)

        usage = {
            "prompt_tokens": prompt_tokens,
            "candidates_tokens": candidates_tokens,
            "total_tokens": prompt_tokens + candidates_tokens,
            "cached": False,
            "duration_sec": round(duration, 3),
        }
        _log_usage(self.provider, self.model, prompt_tokens, candidates_tokens, False, duration, label)
        return result_json, usage

    def _call_vertex(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        system_instruction: str | None,
        temperature: float,
    ) -> tuple[dict[str, Any], int, int]:
        from google.genai import types

        config_args: dict[str, Any] = {
            "temperature": temperature,
            "response_mime_type": "application/json",
        }
        if schema:
            config_args["response_schema"] = schema
        if system_instruction:
            config_args["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_args)
        response = self._genai_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        text = response.text or "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as err:
            # Attempt basic markdown codefence stripping
            clean_text = text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            parsed = json.loads(clean_text.strip())

        p_tok = 0
        c_tok = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            p_tok = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            # Thinking tokens are billed as output on Gemini 2.5+, so count them with the response tokens.
            c_tok = (getattr(response.usage_metadata, "candidates_token_count", 0) or 0) + (
                getattr(response.usage_metadata, "thoughts_token_count", 0) or 0
            )

        return parsed, p_tok, c_tok

    def _call_openai_compat(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        system_instruction: str | None,
        temperature: float,
    ) -> tuple[dict[str, Any], int, int]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if schema:
            # json_object mode does not take a schema, so spell out the expected shape in the prompt.
            prompt += (
                "\n\nRespond with ONLY a JSON object matching this schema "
                "(types are uppercase OpenAPI names):\n" + json.dumps(schema)
            )
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        usage = data.get("usage", {})
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)

        return parsed, p_tok, c_tok


# Module-level singleton
_default_client: LLMClient | None = None
_default_client_env: tuple[str, str | None, str | None] | None = None


def get_llm_client() -> LLMClient:
    """Return the shared client, rebuilding it if LLM_PROVIDER / LLM_MODEL changed (e.g. from the UI)."""
    global _default_client, _default_client_env
    env = (os.getenv("LLM_PROVIDER", "vertex").lower(), os.getenv("LLM_MODEL"), os.getenv("LLM_BASE_URL"))
    if _default_client is None or env != _default_client_env:
        _default_client = LLMClient()
        _default_client_env = env
    return _default_client


def set_usage_log_path(path: str | Path) -> None:
    """Send usage records to a per-run file so cost numbers cover this run only."""
    global USAGE_LOG_PATH
    USAGE_LOG_PATH = Path(path)


def generate_json(
    prompt: str,
    schema: dict[str, Any] | None = None,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    use_cache: bool = True,
    label: str = "general",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Primary entry point for structured generation."""
    client = get_llm_client()
    return client.generate_json(
        prompt=prompt,
        schema=schema,
        system_instruction=system_instruction,
        temperature=temperature,
        use_cache=use_cache,
        label=label,
    )
