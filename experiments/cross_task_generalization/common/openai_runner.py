from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


SYSTEM_JSON_REPAIR = """Repair malformed JSON. Return one valid JSON object only. No markdown and no prose."""


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def model_supports_temperature(model: str) -> bool:
    return not model.lower().startswith("gpt-5")


def response_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt
    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", "") != "message":
            continue
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", "")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts).strip()


def json_from_text(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        obj = json.loads(raw[first : last + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("response did not contain a JSON object")


class JsonModelClient:
    def __init__(self, model: str, temperature: float = 0.0, max_output_tokens: int = 12000) -> None:
        load_dotenv()
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def call_json(self, system_prompt: str, payload: dict[str, Any], repair_attempts: int = 2) -> tuple[dict[str, Any], str]:
        raw = self._call(system_prompt, payload, json_mode=True)
        try:
            return json_from_text(raw), raw
        except Exception:
            repaired = raw
            for _ in range(repair_attempts):
                repaired = self._call(SYSTEM_JSON_REPAIR, {"malformed_json_text": repaired}, json_mode=True)
                try:
                    return json_from_text(repaired), raw + "\n\n[JSON_REPAIRED]\n" + repaired
                except Exception:
                    continue
            raise

    def _call(self, system_prompt: str, payload: dict[str, Any], json_mode: bool = False) -> str:
        req: dict[str, Any] = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        if model_supports_temperature(self.model):
            req["temperature"] = self.temperature
        if json_mode:
            req["text"] = {"format": {"type": "json_object"}}
        resp = self.client.responses.create(**req)
        return response_text(resp)

