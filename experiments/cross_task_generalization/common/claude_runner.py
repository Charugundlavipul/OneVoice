from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None

from experiments.cross_task_generalization.common.openai_runner import (
    json_from_text,
    load_dotenv,
    SYSTEM_JSON_REPAIR,
)


class ClaudeJsonModelClient:
    """Drop-in replacement for JsonModelClient that calls the Anthropic Claude API."""

    def __init__(self, model: str, temperature: float = 0.0, max_output_tokens: int = 12000) -> None:
        load_dotenv()
        if anthropic is None:
            raise RuntimeError("anthropic package is not installed – run: pip install anthropic")
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("CLAUDE_API_KEY (or ANTHROPIC_API_KEY) is not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def call_json(self, system_prompt: str, payload: dict[str, Any], repair_attempts: int = 2) -> tuple[dict[str, Any], str]:
        raw = self._call(system_prompt, payload)
        try:
            return json_from_text(raw), raw
        except Exception:
            repaired = raw
            for _ in range(repair_attempts):
                repaired = self._call(SYSTEM_JSON_REPAIR, {"malformed_json_text": repaired})
                try:
                    return json_from_text(repaired), raw + "\n\n[JSON_REPAIRED]\n" + repaired
                except Exception:
                    continue
            raise

    def _call(self, system_prompt: str, payload: dict[str, Any]) -> str:
        import time
        max_retries = 6
        base_delay = 3.0
        for attempt in range(max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                # Extract text from response content blocks
                parts: list[str] = []
                for block in resp.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                return "\n".join(parts).strip()
            except Exception as exc:
                is_rate_limit = False
                if anthropic is not None and isinstance(exc, anthropic.RateLimitError):
                    is_rate_limit = True
                elif "rate_limit" in str(exc) or "429" in str(exc):
                    is_rate_limit = True
                
                if is_rate_limit and attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    print(f"[RATE_LIMIT] Exceeded limit for model {self.model}. Retrying in {sleep_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    raise
