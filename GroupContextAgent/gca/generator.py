"""Генераторы ответа: шаблонный (основной эксперимент) и адаптер внешней LLM (эксперимент с LLM)."""
from __future__ import annotations

import json
import urllib.request

from .context import Context
from . import text as T

PHRASE = {
    "meeting_time": "встреча в {}",
    "meeting_place": "место встречи: {}",
    "meeting_date": "дата встречи: {}",
    "deadline": "срок сдачи: {}",
    "link": "ссылка: {}",
    "door_code": "код от двери: {}",
}


class TemplateGenerator:
    """Детерминированный генератор: формирует ответ только из сведений контекста."""

    name = "template"

    def generate(self, ctx: Context) -> str:
        if "запомни" in ctx.request.lower() or "запиши" in ctx.request.lower():
            return "Запомнил."
        if not ctx.keys:
            return "Уточните, пожалуйста, что именно нужно подсказать."
        known = [PHRASE[k].format(ctx.facts[k]) for k in ctx.keys if k in ctx.facts]
        unknown = [T.KEY_RU[k] for k in ctx.keys if k not in ctx.facts]
        parts = []
        if known:
            s = T.join_ru(known)
            parts.append(s[0].upper() + s[1:] + ".")
        if unknown:
            parts.append(f"Сведений о том, что вы спрашиваете ({T.join_ru(unknown)}), у меня для вас нет.")
        return " ".join(parts)


class OpenAICompatibleGenerator:
    """Адаптер для LLM с OpenAI-совместимым API (эксперимент: experiments/run_llm.py)."""

    name = "llm"

    def __init__(self, base_url: str, model: str, api_key: str, temperature: float = 0.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key, self.temperature = model, api_key, temperature

    def generate(self, ctx: Context) -> str:
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "user", "content": ctx.prompt()}]}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()
