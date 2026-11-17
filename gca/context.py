"""Формирование контекста C_t = Phi(G_t, m_t) для языковой модели."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import ALL, GroupState, Message
from . import text as T


@dataclass
class Context:
    speaker: str
    addressees: list[str]
    request: str
    keys: list[str]
    facts: dict[str, str] = field(default_factory=dict)
    recent: list[str] = field(default_factory=list)

    def prompt(self) -> str:
        facts = "\n".join(f"- {T.KEY_RU.get(k, k)}: {v}" for k, v in self.facts.items()) or "- нет доступных сведений"
        recent = "\n".join(self.recent)
        return (
            "Ты участник группового чата и отвечаешь только на обращённые к тебе запросы.\n"
            f"Текущий говорящий: {self.speaker}.\nАдресаты реплики: {', '.join(self.addressees)}.\n"
            f"Сведения, доступные говорящему:\n{facts}\nПоследние реплики:\n{recent}\n"
            f"Текущая реплика: \"{self.request}\"\n"
            "Используй только перечисленные сведения. Ответь кратко."
        )


def build(state: GroupState, msg: Message, addressees, facts: dict[str, str], n_recent: int = 4,
          audience=None) -> Context:
    """audience — все, кто увидит ответ: в последние реплики попадают только сообщения, которые видел каждый из них.
    Без audience (базовый подход) берутся реплики, видимые автору."""
    names = state.participants
    seen = (lambda m: m.visible_all(audience)) if audience is not None else (lambda m: m.visible(msg.speaker))
    recent = [f"{names[m.speaker].name}: {m.text}" for m in state.history[-n_recent:] if seen(m)]
    adr = ["вся группа"] if addressees == ALL else [names[a].name for a in sorted(addressees)]
    return Context(names[msg.speaker].name, adr, msg.text, T.requested_keys(msg.text), facts, recent)
