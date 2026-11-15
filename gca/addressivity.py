"""Оценка адресованности реплики (вектор A_t, формула (5) отчёта)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .models import ALL, GroupState, Message
from . import text as T


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@dataclass(frozen=True)
class AddressivityParams:
    theta0: float = -4.0   # смещение
    theta1: float = 7.0    # явное обращение e_t(u)
    theta2: float = 5.0    # продолжение обмена с участником r_t(u)
    theta3: float = 3.0    # обращение ко всей группе g_t
    theta4: float = 2.0    # контекстная связь по графу c_t(u)
    threshold: float = 0.5  # порог включения участника в множество адресатов
    window: int = 4         # окно (в репликах), в котором учитывается текущий собеседник


@dataclass
class AddressFeatures:
    explicit: set[str]
    group: bool
    reply: dict[str, float]
    context: dict[str, float]


def features(msg: Message, state: GroupState, p: AddressivityParams, use_graph: bool = True) -> AddressFeatures:
    forms = {pid: part.address_forms() for pid, part in state.participants.items() if pid != msg.speaker}
    explicit = T.explicit_addressees(msg.text, forms)
    group = T.is_group_address(msg.text)
    reply = {u: 0.0 for u in forms}
    context = {u: 0.0 for u in forms}
    if msg.reply_to:
        src = next((m for m in state.history if m.id == msg.reply_to), None)
        if src and src.speaker in reply:
            reply[src.speaker] = 1.0
    if use_graph and not explicit and not group:
        pend = state.pending.get(msg.speaker)          # ответ на обращённый к автору вопрос
        partner = state.partner.get(msg.speaker)       # продолжение обмена: уточняющий вопрос тому же собеседнику
        if pend and pend[0] in reply and msg.turn - pend[1] <= 2 * p.window:
            reply[pend[0]] = 1.0
        elif partner and partner[0] in reply and msg.turn - partner[1] <= p.window and T.is_request(msg.text):
            reply[partner[0]] = 1.0
        row = state.graph.get(msg.speaker, {})
        total = sum(row.values())
        if total > 0:
            for u in context:
                context[u] = row.get(u, 0.0) / total
    return AddressFeatures(explicit, group, reply, context)


def estimate(msg: Message, state: GroupState, p: AddressivityParams, use_graph: bool = True) -> tuple[dict[str, float], AddressFeatures]:
    """a_t(u) = sigma(theta0 + theta1*e + theta2*r + theta3*g + theta4*c) для каждого u != автор."""
    f = features(msg, state, p, use_graph)
    scores = {}
    for u in f.reply:
        e = 1.0 if u in f.explicit else 0.0
        g = 1.0 if f.group and not f.explicit else 0.0
        z = p.theta0 + p.theta1 * e + p.theta2 * f.reply[u] + p.theta3 * g + p.theta4 * f.context[u]
        scores[u] = sigmoid(z)
    return scores, f


def addressee_set(scores: dict[str, float], f: AddressFeatures, p: AddressivityParams) -> str | frozenset[str]:
    """Множество адресатов: группа при групповом обращении, иначе участники с a_t(u) >= порога."""
    if f.group and not f.explicit:
        return ALL
    chosen = frozenset(u for u, s in scores.items() if s >= p.threshold)
    return chosen if chosen else ALL
