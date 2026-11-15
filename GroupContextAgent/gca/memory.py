"""Групповая память M_t и знания участников B_t: версии фактов и их доступность."""
from __future__ import annotations

from .models import Fact, GroupState, Message


def update(state: GroupState, msg: Message, facts: dict[str, str]) -> list[Fact]:
    """Добавляет новые факты, при изменении значения создаёт новую версию и деактивирует старую."""
    audience = msg.visible_to if msg.visible_to is not None else state.present()
    added = []
    for key, value in facts.items():
        current = [f for f in state.memory if f.key == key and f.active]
        if current and current[-1].value == value:
            cur = current[-1]
            cur.visible_to = frozenset(cur.visible_to | audience)  # факт стал известен новым участникам
            for u in audience:
                state.knowledge.setdefault(u, set()).add(cur.fact_id)
            continue
        version = 1 + max((f.version for f in state.memory if f.key == key), default=0)
        if current and msg.visible_to is None:
            for f in current:
                f.active = False
        fact = Fact(key, value, msg.speaker, msg.turn, version, frozenset(audience))
        state.memory.append(fact)
        added.append(fact)
        for u in audience:
            state.knowledge.setdefault(u, set()).add(fact.fact_id)
    return added


def accessible(state: GroupState, user: str, keys: list[str], check_access: bool = True) -> dict[str, Fact]:
    """Актуальные версии фактов по ключам, доступные участнику user (B_user)."""
    result = {}
    for key in keys:
        versions = [f for f in state.memory if f.key == key and (not check_access or user in f.visible_to)]
        if not versions:
            continue
        active = [f for f in versions if f.active] or versions
        result[key] = max(active, key=lambda f: f.turn)
    return result
