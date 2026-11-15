"""Граф взаимодействий E_t = (U, R_t): веса связей W_ij с затуханием (формула обновления графа)."""
from __future__ import annotations

from .models import ALL, GroupState, Message
from .text import is_request


def update(state: GroupState, msg: Message, scores: dict[str, float], addressees, lam: float = 0.9) -> None:
    """W_ij(t+1) = lambda * W_ij(t) + I(s_t = i) * a_t(j)."""
    for i, row in state.graph.items():
        for j in row:
            row[j] *= lam
            if i == msg.speaker:
                row[j] += scores.get(j, 0.0)
    # текущий собеседник: при адресации одному участнику связь фиксируется в обе стороны
    state.pending.pop(msg.speaker, None)  # автор ответил на ожидавшее его обращение
    if addressees != ALL and len(addressees) == 1:
        (u,) = tuple(addressees)
        state.partner[msg.speaker] = (u, msg.turn)
        state.partner[u] = (msg.speaker, msg.turn)
        if is_request(msg.text):
            state.pending[u] = (msg.speaker, msg.turn)
    elif addressees == ALL:
        state.partner.pop(msg.speaker, None)
