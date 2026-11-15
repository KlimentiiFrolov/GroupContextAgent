"""Структуры данных состояния группы G_t = {U, H_t, E_t, B_t, M_t, A_t}."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

ALL = "ALL"  # обозначение адресации всей группе


@dataclass
class Participant:
    """Участник диалога (элемент множества U)."""
    id: str
    name: str
    aliases: tuple[str, ...] = ()
    is_agent: bool = False

    def address_forms(self) -> set[str]:
        return {self.name.lower(), *[a.lower() for a in self.aliases]}


@dataclass
class Message:
    """Реплика диалога m_t."""
    id: str
    speaker: str
    text: str
    turn: int = 0
    visible_to: Optional[frozenset[str]] = None  # None — реплика видна всем участникам
    reply_to: Optional[str] = None

    def visible(self, pid: str) -> bool:
        return self.visible_to is None or pid in self.visible_to


@dataclass
class Fact:
    """Запись групповой памяти M_t: факт с источником, версией и доступностью."""
    key: str
    value: str
    source: str
    turn: int
    version: int
    visible_to: frozenset[str]
    active: bool = True

    @property
    def fact_id(self) -> str:
        return f"{self.key}_v{self.version}"


@dataclass
class GroupState:
    """Состояние группового взаимодействия G_t."""
    participants: dict[str, Participant]                                  # U
    history: list[Message] = field(default_factory=list)                   # H_t
    graph: dict[str, dict[str, float]] = field(default_factory=dict)       # E_t: веса W_ij
    partner: dict[str, tuple[str, int]] = field(default_factory=dict)      # E_t: текущий собеседник участника
    pending: dict[str, tuple[str, int]] = field(default_factory=dict)      # E_t: обращения, ожидающие ответа
    knowledge: dict[str, set[str]] = field(default_factory=dict)           # B_t: идентификаторы известных фактов
    memory: list[Fact] = field(default_factory=list)                       # M_t
    addressivity: dict[str, float] = field(default_factory=dict)           # A_t
    turn: int = 0

    def __post_init__(self) -> None:
        ids = list(self.participants)
        for i in ids:
            self.graph.setdefault(i, {j: 0.0 for j in ids if j != i})
            self.knowledge.setdefault(i, set())

    def present(self) -> frozenset[str]:
        return frozenset(self.participants)

    def humans(self) -> list[str]:
        return [p for p, v in self.participants.items() if not v.is_agent]

    def agent_id(self) -> str:
        return next(p for p, v in self.participants.items() if v.is_agent)

    def active_facts(self) -> dict[str, Fact]:
        return {f.key: f for f in self.memory if f.active}

    def to_dict(self) -> dict:
        """Представление состояния в формате JSON (приложение А отчёта)."""
        return {
            "turn": self.turn,
            "participants": [{"id": p.id, "role": "llm" if p.is_agent else "user"} for p in self.participants.values()],
            "history_length": len(self.history),
            "interaction_graph": {i: {j: round(w, 3) for j, w in row.items() if w > 0} for i, row in self.graph.items()},
            "memory": [{"fact_id": f.fact_id, "key": f.key, "value": f.value, "source": f.source, "turn": f.turn,
                        "version": f.version, "visible_to": sorted(f.visible_to), "active": f.active}
                       for f in self.memory],
            "knowledge": {p: sorted(k) for p, k in self.knowledge.items()},
            "addressivity": {p: round(v, 3) for p, v in self.addressivity.items()},
        }
