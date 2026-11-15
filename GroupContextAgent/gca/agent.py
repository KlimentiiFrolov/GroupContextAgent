"""Основной алгоритм обработки реплики (подраздел 3.6 отчёта)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import addressivity as AD, graph as GR, memory as MEM, text as T
from .context import Context, build
from .generator import TemplateGenerator
from .models import ALL, GroupState, Message, Participant
from .policy import PolicyParams, speak_probability


@dataclass
class AgentConfig:
    use_memory: bool = True      # групповая память M_t (иначе факты ищутся в окне истории)
    use_access: bool = True      # учёт доступности фактов участникам B_t
    use_graph: bool = True       # структура взаимодействий E_t
    history_window: int = 8      # окно истории для варианта без памяти
    graph_decay: float = 0.9     # lambda
    address: AD.AddressivityParams = field(default_factory=AD.AddressivityParams)
    policy: PolicyParams = field(default_factory=PolicyParams)


@dataclass
class StepResult:
    message_id: str
    addressees: object
    scores: dict[str, float]
    p_speak: float
    reply: bool
    context: Optional[Context] = None
    response: Optional[str] = None


def facts_from_window(history: list[Message], viewer: str, keys: list[str], k: int) -> dict[str, str]:
    """Поиск фактов только в последних k репликах, доступных наблюдателю (без структурированной памяти)."""
    found: dict[str, str] = {}
    for m in history[-k:]:
        if m.visible(viewer):
            for key, value in T.extract_facts(m.text).items():
                if key in keys:
                    found[key] = value
    return found


class GroupContextAgent:
    """Агент, поддерживающий состояние группы и отвечающий только при необходимости."""

    def __init__(self, participants: list[Participant], config: AgentConfig | None = None, generator=None):
        self.config = config or AgentConfig()
        self.state = GroupState({p.id: p for p in participants})
        self.generator = generator or TemplateGenerator()
        self.agent_id = self.state.agent_id()
        self.log: list[StepResult] = []

    def process(self, msg: Message) -> StepResult:
        cfg, st = self.config, self.state
        st.turn += 1
        msg.turn = st.turn
        # 1–3. анализ реплики и оценка адресованности по предыдущему состоянию графа
        scores, feats = AD.estimate(msg, st, cfg.address, cfg.use_graph)
        addressees = AD.addressee_set(scores, feats, cfg.address)
        facts = T.extract_facts(msg.text)
        # 4. обновление графа взаимодействий
        if cfg.use_graph:
            GR.update(st, msg, scores, addressees, cfg.graph_decay)
        # 5. обновление групповой памяти и знаний участников
        if cfg.use_memory:
            MEM.update(st, msg, facts)
        # 6. новое состояние G_{t+1}
        st.history.append(msg)
        st.addressivity = scores
        # 7. решение об участии
        if msg.speaker == self.agent_id:
            return StepResult(msg.id, addressees, scores, 0.0, False)
        keys = T.requested_keys(msg.text)
        available = self._facts_for(msg, keys)
        others = feats.explicit - {self.agent_id}
        o = bool(others) and self.agent_id not in feats.explicit
        a_agent = scores.get(self.agent_id, 0.0)
        p = speak_probability(a_agent, T.is_request(msg.text), bool(available), o, cfg.policy)
        res = StepResult(msg.id, addressees, scores, p, p >= cfg.policy.tau)
        # 8. формирование контекста и ответа
        if res.reply:
            res.context = build(st, msg, addressees, available)
            res.response = self.generator.generate(res.context)
            self._register_reply(msg, res.response)
        self.log.append(res)
        return res

    def _facts_for(self, msg: Message, keys: list[str]) -> dict[str, str]:
        cfg = self.config
        if not keys:
            return {}
        if not cfg.use_memory:
            return facts_from_window(self.state.history[:-1] + [msg], self.agent_id, keys, cfg.history_window)
        return {k: f.value for k, f in MEM.accessible(self.state, msg.speaker, keys, cfg.use_access).items()}

    def _register_reply(self, msg: Message, text: str) -> None:
        reply = Message(f"{msg.id}_r", self.agent_id, text, visible_to=msg.visible_to, reply_to=msg.id)
        self.state.turn += 1
        reply.turn = self.state.turn
        scores = {u: (1.0 if u == msg.speaker else 0.0) for u in self.state.participants if u != self.agent_id}
        if self.config.use_graph:
            GR.update(self.state, reply, scores, frozenset({msg.speaker}), self.config.graph_decay)
        self.state.history.append(reply)
