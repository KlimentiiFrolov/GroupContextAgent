import json
from pathlib import Path

import pytest

from gca import GroupContextAgent, HistoryBaselineAgent, Message, AgentConfig
from gca.io import AGENT, messages_from, participants_from
from gca.models import ALL, Participant
from gca import text as T
from gca.addressivity import sigmoid
from gca.policy import PolicyParams, speak_probability

PS = [Participant("anna", "Анна", ("аня",)), Participant("boris", "Борис", ("боря",)),
      Participant("sofia", "София", ("соня",)), AGENT]


def feed(agent, *msgs):
    out = []
    for i, m in enumerate(msgs, 1):
        sp, text, *vis = m
        out.append(agent.process(Message(f"m{i}", sp, text, visible_to=frozenset(vis[0]) if vis else None)))
    return out


def test_sigmoid_bounds():
    assert sigmoid(0) == 0.5 and 0 < sigmoid(-50) < 1e-10 and sigmoid(50) > 0.999


@pytest.mark.parametrize("text,facts", [
    ("Встреча завтра в 10:00 в аудитории 305.", {"meeting_time": "10:00", "meeting_place": "аудитория 305"}),
    ("Собираемся в 9.30 в кабинете 214.", {"meeting_time": "9:30", "meeting_place": "кабинет 214"}),
    ("Агент, запомни: код от двери 4512.", {"door_code": "4512"}),
    ("Где встреча?", {}),
])
def test_extract_facts(text, facts):
    assert T.extract_facts(text) == facts


def test_explicit_address_to_agent_and_answer():
    a = GroupContextAgent(PS)
    r = feed(a, ("anna", "Встреча завтра в 10:00 в аудитории 305."), ("sofia", "Агент, напомни место встречи."))
    assert not r[0].reply and r[1].reply
    assert r[1].addressees == frozenset({"agent"})
    assert "аудитория 305" in r[1].response


def test_question_to_other_participant_is_ignored():
    a = GroupContextAgent(PS)
    r = feed(a, ("anna", "Встреча в 10:00."), ("boris", "Соня, ты помнишь, во сколько встреча?"), ("sofia", "Да, помню."))
    assert r[1].addressees == frozenset({"sofia"}) and not r[1].reply
    assert r[2].addressees == frozenset({"boris"}) and not r[2].reply


def test_implicit_reply_in_interleaved_threads():
    a = GroupContextAgent(PS + [Participant("igor", "Игорь")])
    r = feed(a, ("anna", "Борис, ты взял ноутбук?"), ("sofia", "Игорь, отчёт отправил?"),
             ("boris", "Да, взял."), ("igor", "Ещё нет."), ("boris", "А зарядку тоже брать?"))
    assert r[2].addressees == frozenset({"anna"})
    assert r[3].addressees == frozenset({"sofia"})
    assert r[4].addressees == frozenset({"anna"}) and not r[4].reply


def test_private_fact_not_disclosed():
    a = GroupContextAgent(PS)
    r = feed(a, ("anna", "Агент, запомни: код от двери 4512.", ["anna", "agent"]),
             ("boris", "Агент, какой код от двери?"), ("anna", "Агент, напомни код от двери."),
             ("anna", "Агент, напомни код от двери.", ["anna", "agent"]))
    assert "4512" not in r[1].response
    # Анна знает код, но спрашивает в общем чате: ответ увидят все, код называть нельзя
    assert r[2].reply and "4512" not in r[2].response
    assert all("4512" not in line for line in r[2].context.recent)
    # тот же вопрос в закрытом сообщении — код называется
    assert "4512" in r[3].response
    assert "door_code_v1" in a.state.knowledge["anna"] and "door_code_v1" not in a.state.knowledge["boris"]


def test_access_checked_without_memory():
    # вариант без памяти проверяет права так же, как полный метод
    a = GroupContextAgent(PS, AgentConfig(use_memory=False))
    r = feed(a, ("anna", "Агент, запомни: код от двери 4512.", ["anna", "agent"]),
             ("boris", "Агент, какой код от двери?"), ("anna", "Агент, напомни код от двери."),
             ("anna", "Агент, напомни код от двери.", ["anna", "agent"]))
    assert "4512" not in r[1].response and "4512" not in r[2].response
    assert "4512" in r[3].response


def test_baseline_leaks_private_fact():
    b = HistoryBaselineAgent(PS)
    r = feed(b, ("anna", "Агент, запомни: код от двери 4512.", ["anna", "agent"]), ("boris", "Агент, какой код от двери?"))
    assert "4512" in r[1].response


def test_fact_versioning():
    a = GroupContextAgent(PS)
    r = feed(a, ("anna", "Встреча в 10:00."), ("boris", "Встречу перенесли на 11:30."), ("sofia", "Агент, во сколько встреча?"))
    versions = [(f.value, f.version, f.active) for f in a.state.memory if f.key == "meeting_time"]
    assert versions == [("10:00", 1, False), ("11:30", 2, True)]
    assert "11:30" in r[2].response and "10:00" not in r[2].response


def test_state_updated_without_reply():
    a = GroupContextAgent(PS)
    feed(a, ("anna", "Встреча в 10:00."), ("boris", "Хорошо, я буду."))
    assert len(a.state.history) == 2 and a.state.active_facts()["meeting_time"].value == "10:00"
    assert a.log[-1].reply is False


def test_graph_decay():
    a = GroupContextAgent(PS)
    feed(a, ("anna", "Борис, привет!"))
    w1 = a.state.graph["anna"]["boris"]
    feed(a, ("sofia", "Всем привет."))
    assert a.state.graph["anna"]["boris"] == pytest.approx(0.9 * w1)


def test_group_question_without_fact_is_ignored():
    a = GroupContextAgent(PS)
    r = feed(a, ("anna", "Ребята, кто-нибудь знает, будет ли проектор?"))
    assert r[0].addressees == ALL and not r[0].reply


def test_long_range_memory_vs_baseline_window():
    msgs = [("anna", "Встреча в 12:15 в кабинете 201.")] + [("boris", "Сегодня много дел.")] * 10 + \
           [("sofia", "Агент, где встреча?")]
    assert "кабинет 201" in feed(GroupContextAgent(PS), *msgs)[-1].response
    assert "кабинет 201" not in feed(HistoryBaselineAgent(PS, history_window=8), *msgs)[-1].response


def test_policy_monotonic_in_addressivity():
    p = PolicyParams()
    assert speak_probability(0.9, True, True, False, p) > speak_probability(0.1, True, True, False, p)
    assert speak_probability(0.9, True, True, True, p) < speak_probability(0.9, True, True, False, p)


def test_demo_file_runs():
    d = json.loads(Path("data/demo_meeting.json").read_text(encoding="utf-8"))
    a = GroupContextAgent(participants_from(d))
    res = [a.process(m) for m in messages_from(d)]
    assert sum(r.reply for r in res) == 3
    assert json.dumps(a.state.to_dict(), ensure_ascii=False)


def test_ablation_flags():
    a = GroupContextAgent(PS, AgentConfig(use_access=False))
    r = feed(a, ("anna", "Агент, запомни: код от двери 4512.", ["anna", "agent"]), ("boris", "Агент, какой код от двери?"))
    assert "4512" in r[1].response
