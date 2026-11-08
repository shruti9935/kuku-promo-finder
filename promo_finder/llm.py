"""One small client for any OpenAI-compatible chat API.

Swapping the model is one setting: pick a provider preset (gemini, groq, openrouter,
openai, anthropic, ollama) or point LLM_BASE_URL / LLM_API_KEY / LLM_MODEL at anything
else that speaks the OpenAI chat format (most providers, including Kuku's own gateway
if it does).

Responses are cached on disk, keyed by model + prompt, so re-runs are free and an
interrupted run picks up where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

PRESETS = {
    # name: (base_url, api-key env var, default model, seconds between calls)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-3.5-flash", 7.0),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile", 3.0),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free", 4.0),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1-mini", 0.0),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-sonnet-5-5", 0.0),
    "ollama": ("http://localhost:11434/v1", "", "qwen2.5:7b", 0.0),
}


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.3
    min_interval: float = -1.0  # -1 = use preset
    fallbacks: str = ""         # comma-separated models to try if the main one is overloaded
    cache_dir: str = ".cache/llm"
    use_cache: bool = True

    @classmethod
    def from_env(cls, prefix: str = "LLM_", **overrides) -> "LLMConfig":
        """prefix "LLM_" is the finder; "JUDGE_" configures a separate judge model."""
        def env(k, default=""):
            return os.getenv(prefix + k, "") or (os.getenv("LLM_" + k, default) if prefix != "LLM_" and k in ("PROVIDER", "BASE_URL", "API_KEY") else default)
        c = cls(
            provider=env("PROVIDER", "gemini") or "gemini",
            model=env("MODEL"),
            base_url=env("BASE_URL"),
            api_key=env("API_KEY"),
            fallbacks=env("FALLBACK_MODELS"),
        )
        for k, v in overrides.items():
            if v not in (None, ""):
                setattr(c, k, v)
        return c.resolved()

    def resolved(self) -> "LLMConfig":
        base, key_env, model, gap = PRESETS.get(self.provider, ("", "", "", 0.0))
        self.base_url = self.base_url or base
        self.model = self.model or model
        self.api_key = self.api_key or (os.getenv(key_env, "") if key_env else "") or ("ollama" if self.provider == "ollama" else "")
        if self.min_interval < 0:
            self.min_interval = gap
        if self.provider == "gemini" and not self.fallbacks:
            self.fallbacks = "gemini-2.5-flash" if self.model != "gemini-2.5-flash" else "gemini-3.6-flash"
        return self

    def public(self) -> dict:
        return {"provider": self.provider, "model": self.model, "fallbacks": self.fallbacks,
                "base_url": self.base_url, "temperature": self.temperature}


class QuotaExhausted(RuntimeError):
    """The provider's daily (or monthly) limit is used up. Retrying won't help today."""


def _is_quota(e: Exception) -> bool:
    msg = str(e).lower()
    return getattr(e, "status_code", None) == 429 and ("per day" in msg or "perday" in msg or "quota" in msg and "day" in msg)


class LLM:
    def __init__(self, cfg: LLMConfig):
        if not cfg.base_url:
            raise ValueError("No base URL. Set LLM_PROVIDER to a preset or set LLM_BASE_URL.")
        if not cfg.api_key:
            raise ValueError(f"No API key for provider '{cfg.provider}'. See .env.example.")
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, max_retries=0, timeout=180)
        self._last_call = 0.0
        self._json_mode = True
        self.models = [cfg.model] + [m.strip() for m in cfg.fallbacks.split(",") if m.strip()]
        self.model_used: dict[str, int] = {}
        self.calls = 0
        self.cache_hits = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    def _cache_path(self, messages) -> Path:
        key = json.dumps([self.cfg.base_url, self.cfg.model, self.cfg.temperature, messages], sort_keys=True)
        return Path(self.cfg.cache_dir) / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")

    def chat_json(self, system: str, user: str, use_cache: bool | None = None) -> tuple[dict, str]:
        """Returns (parsed JSON, raw text). Raises after retries fail."""
        if use_cache is None:
            use_cache = self.cfg.use_cache
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        cp = self._cache_path(messages)
        if use_cache and cp.exists():
            raw = json.loads(cp.read_text(encoding="utf-8"))["raw"]
            parsed = extract_json(raw)
            if parsed is not None:
                self.cache_hits += 1
                return parsed, raw

        last_err, exhausted = None, []
        for model in list(self.models):
            try:
                return self._call(model, messages, cp)
            except QuotaExhausted as e:
                last_err = e
                exhausted.append(model)
                self.models.remove(model)      # don't try this model again in this run
            except Exception as e:
                last_err = e
                if getattr(e, "status_code", None) in (400, 401, 403):
                    raise
        if not self.models:
            raise QuotaExhausted(f"Daily quota used up on {exhausted}. Try another model/provider or wait for the reset.")
        raise RuntimeError(f"LLM call failed on {self.models}: {str(last_err)[:300]}")

    def _call(self, model: str, messages: list, cp: Path) -> tuple[dict, str]:
        last_err = None
        user = messages[1]["content"]
        for attempt in range(4):
            wait = self.cfg.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                kwargs = dict(model=model, messages=messages, temperature=self.cfg.temperature)
                if self._json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                self._last_call = time.time()
                resp = self.client.chat.completions.create(**kwargs)
                self.calls += 1
                self.model_used[model] = self.model_used.get(model, 0) + 1
                if resp.usage:
                    self.usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
                    self.usage["completion_tokens"] += resp.usage.completion_tokens or 0
                raw = resp.choices[0].message.content or ""
                parsed = extract_json(raw)
                if parsed is None:
                    last_err = ValueError("Model did not return valid JSON")
                    messages_retry = user + "\n\nReturn ONLY one valid JSON object."
                    messages = [messages[0], {"role": "user", "content": messages_retry}]
                    continue
                cp.write_text(json.dumps({"model": model, "raw": raw}, ensure_ascii=False), encoding="utf-8")
                return parsed, raw
            except Exception as e:  # rate limits, 5xx, unsupported json mode
                last_err = e
                msg = str(e).lower()
                if self._json_mode and ("response_format" in msg or "json_object" in msg):
                    self._json_mode = False
                    continue
                if _is_quota(e):
                    raise QuotaExhausted(str(e)[:300])
                status = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404):
                    raise
                time.sleep(min(40, 5 * 2 ** attempt))
        raise RuntimeError(f"{model} failed after retries: {last_err}")


def extract_json(text: str):
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(t[a:b + 1])
        except Exception:
            return None
    return None
