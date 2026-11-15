"""Базовый подход: решения принимаются только по текстовой истории последних реплик."""
from __future__ import annotations

from .agent import StepResult, facts_from_window
from .context import build
from .generator import TemplateGenerator
from .models import ALL, GroupState, Message, Participant
from . import text as T


class HistoryBaselineAgent:
    """Нет структурированного состояния: адресат определяется по явным обращениям и предыдущей реплике,
    факты для ответа ищутся в окне из k последних видимых агенту реплик."""

    def __init__(self, participants: list[Participant], history_window: int = 8, generator=None):
        self.state = GroupState({p.id: p for p in participants})
        self.k = history_window
        self.generator = generator or TemplateGenerator()
        self.agent_id = self.state.agent_id()
        self.log: list[StepResult] = []

    def process(self, msg: Message) -> StepResult:
        st = self.state
        st.turn += 1
        msg.turn = st.turn
        forms = {pid: p.address_forms() for pid, p in st.participants.items() if pid != msg.speaker}
        explicit = T.explicit_addressees(msg.text, forms)
        prev = st.history[-1] if st.history else None
        if explicit:
            addressees = frozenset(explicit)
        elif T.is_group_address(msg.text):
            addressees = ALL
        elif prev and prev.speaker != msg.speaker and "?" in prev.text:
            addressees = frozenset({prev.speaker})
        else:
            addressees = ALL
        st.history.append(msg)
        if msg.speaker == self.agent_id:
            return StepResult(msg.id, addressees, {}, 0.0, False)
        reply = self.agent_id in explicit or (T.is_request(msg.text) and not explicit)
        res = StepResult(msg.id, addressees, {}, 1.0 if reply else 0.0, reply)
        if reply:
            keys = T.requested_keys(msg.text)
            facts = facts_from_window(st.history, self.agent_id, keys, self.k) if keys else {}
            res.context = build(st, msg, addressees, facts)
            res.response = self.generator.generate(res.context)
            r = Message(f"{msg.id}_r", self.agent_id, res.response, visible_to=msg.visible_to, reply_to=msg.id)
            st.turn += 1
            r.turn = st.turn
            st.history.append(r)
        self.log.append(res)
        return res
