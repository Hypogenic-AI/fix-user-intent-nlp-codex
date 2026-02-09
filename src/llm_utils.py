import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("openai package not available") from exc


@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0


class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.client = self._build_client()

    def _build_client(self) -> OpenAI:
        openai_key = os.getenv("OPENAI_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openai_key:
            return OpenAI(api_key=openai_key)
        if openrouter_key:
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            return OpenAI(api_key=openrouter_key, base_url=base_url)
        raise RuntimeError("Missing OPENAI_API_KEY or OPENROUTER_API_KEY in environment")

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
    def chat_json(self, system: str, user: str, config: Optional[LLMConfig] = None) -> Dict[str, Any]:
        cfg = config or LLMConfig(model=self.model)
        start = time.time()
        resp = self.client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        )
        duration = time.time() - start
        content = resp.choices[0].message.content
        data = json.loads(content)
        usage = resp.usage.model_dump() if resp.usage else {}
        return {"data": data, "usage": usage, "duration_s": duration}

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
    def chat_text(self, system: str, user: str, config: Optional[LLMConfig] = None) -> Dict[str, Any]:
        cfg = config or LLMConfig(model=self.model)
        start = time.time()
        resp = self.client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        )
        duration = time.time() - start
        content = resp.choices[0].message.content
        usage = resp.usage.model_dump() if resp.usage else {}
        return {"data": content, "usage": usage, "duration_s": duration}
