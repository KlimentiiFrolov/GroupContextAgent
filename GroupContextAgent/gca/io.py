"""Чтение диалогов и участников из JSON."""
from __future__ import annotations

import json
from pathlib import Path

from .models import Message, Participant

AGENT = Participant("agent", "Агент", ("агент", "бот", "ассистент"), is_agent=True)


def participants_from(d: dict) -> list[Participant]:
    ps = [Participant(p["id"], p["name"], tuple(p.get("aliases", ()))) for p in d["participants"]]
    return ps + [AGENT]


def messages_from(d: dict) -> list[Message]:
    return [Message(m["id"], m["speaker"], m["text"],
                    visible_to=frozenset(m["visible_to"]) if m.get("visible_to") else None,
                    reply_to=m.get("reply_to")) for m in d["messages"]]


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
