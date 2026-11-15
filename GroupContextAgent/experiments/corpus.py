"""Генератор синтетического корпуса многопользовательских диалогов с эталонной разметкой.

Восемь категорий сценариев. Настроечная выборка (dev) использует только базовые формулировки запросов,
тестовая (test) — базовые и дополнительные формулировки, не встречавшиеся при настройке.
Входные диалоги и эталонная разметка сохраняются в разные файлы.
python -m experiments.corpus
"""
from __future__ import annotations

import json
import random
from pathlib import Path

NAMES = [("anna", "Анна", ["аня"]), ("boris", "Борис", ["боря"]), ("sofia", "София", ["соня"]),
         ("dmitry", "Дмитрий", ["дима"]), ("elena", "Елена", ["лена"]), ("maxim", "Максим", ["макс"]),
         ("olga", "Ольга", ["оля"]), ("kirill", "Кирилл", ["киря"]), ("maria", "Мария", ["маша"]),
         ("igor", "Игорь", [])]

# формулировки: (базовые, дополнительные — только в тестовой выборке)
ASK = {
    "time": (["во сколько встреча?", "напомни, во сколько встреча.", "когда начинается встреча?"],
             ["к скольки нам подходить?", "какое время встречи?", "встреча во сколько, не подскажешь?",
              "напомни время встречи."]),
    "place": (["где встреча?", "напомни место встречи.", "куда приходить?"],
              ["в какой аудитории собираемся?", "в каком кабинете встреча?", "подскажи, где собираемся."]),
    "both": (["напомни, когда и где встреча."], ["во сколько и где собираемся?", "скажи время и место встречи."]),
    "code": (["какой код от двери?", "напомни код от двери."], ["как попасть внутрь, какой там код?"]),
    "follow_place": (["А где?", "А место?", "А куда идти?"], ["А аудитория какая?", "А в каком кабинете?"]),
    "group_time": (["Ребята, напомните, во сколько встреча?", "Кто-нибудь помнит, во сколько собираемся?"],
                   ["Народ, а встреча к скольки?", "Все помнят время встречи?"]),
    "group_place": (["Кто-нибудь помнит, где собираемся?", "Ребята, напомните место встречи."],
                    ["Коллеги, а где встречаемся-то?", "Народ, в какой аудитории встреча?"]),
    "group_none": (["Ребята, кто-нибудь знает, будет ли проектор?", "Кто-нибудь брал зарядку от ноутбука?"],
                   ["Народ, а кофе кто-нибудь принесёт?"]),
}
FILLERS = ["Я сегодня немного опоздаю.", "Кофе в автомате опять закончился.", "Презентацию почти доделал.",
           "Отличная идея, поддерживаю.", "Сейчас посмотрю расписание.", "Сегодня очень много дел.",
           "Надо будет купить маркеры.", "Слайды выложу вечером.", "Погода сегодня отличная.",
           "Я пока в дороге.", "Код проекта уже собрал, всё работает.", "Черновик отчёта готов наполовину."]
PERSONAL_Q = [("ты возьмёшь ноутбук?", "Да, возьму."), ("ты отправил отчёт?", "Ещё нет, вечером отправлю."),
              ("сможешь распечатать раздаточный материал?", "Смогу, без проблем."),
              ("ты успеваешь со слайдами?", "Почти, осталось два слайда.")]
FOLLOW_Q = [("А зарядку тоже брать?", "Да, бери."), ("А сколько экземпляров нужно?", "Штук десять."),
            ("А в каком формате отправлять?", "В PDF, как обычно.")]


def phr(rng, key, split):
    base, extra = ASK[key]
    return rng.choice(base + extra if split == "test" else base)


def rtime(rng):
    return f"{rng.randint(9, 18)}:{rng.choice(['00', '15', '30', '45'])}"


def rplace(rng):
    kind = rng.choice(["аудитория", "кабинет"])
    return f"{kind} {rng.randint(101, 520)}"


def loc(place):
    kind, num = place.split()
    return f"{'в аудитории' if kind == 'аудитория' else 'в кабинете'} {num}"


class D:
    """Построитель одного диалога."""

    def __init__(self, did, cat, rng, n_humans):
        self.rng, self.id, self.cat = rng, did, cat
        self.people = rng.sample(NAMES, n_humans)
        self.ids = [p[0] for p in self.people]
        self.name = {p[0]: p[1] for p in self.people}
        self.msgs, self.gold = [], []

    def add(self, speaker, text, addr="ALL", reply=False, expect=(), forbid=(), evaluate=False, visible=None):
        mid = f"m{len(self.msgs) + 1}"
        m = {"id": mid, "speaker": speaker, "text": text}
        if visible:
            m["visible_to"] = sorted(set(visible) | {"agent"})
        self.msgs.append(m)
        self.gold.append({"id": mid, "addressees": addr if addr == "ALL" else sorted(addr), "reply": reply,
                          "expect": list(expect), "forbid": list(forbid), "evaluate": evaluate})

    def fillers(self, n, avoid=()):
        for _ in range(n):
            sp = self.rng.choice([i for i in self.ids if i not in avoid] or self.ids)
            if self.rng.random() < 0.3:
                to = self.rng.choice([i for i in self.ids if i != sp])
                q, a = self.rng.choice(PERSONAL_Q)
                self.add(sp, f"{self.name[to]}, {q}", {to})
                self.add(to, a, {sp})
            else:
                self.add(sp, self.rng.choice(FILLERS))

    def facts(self, speaker, t, p):
        form = self.rng.choice(["Встреча завтра в {t} {l}.", "Собираемся в {t} {l}.", "Завтра встречаемся {l} в {t}."])
        self.add(speaker, form.format(t=t, l=loc(p)))

    def dump(self):
        ps = [{"id": i, "name": n, "aliases": a} for i, n, a in self.people]
        return ({"id": self.id, "category": self.cat, "participants": ps, "messages": self.msgs},
                {"id": self.id, "category": self.cat, "labels": self.gold})


def ask_agent(d, sp, key, t, p, split):
    q = phr(d.rng, key, split)
    exp = {"time": [t], "place": [p], "both": [t, p]}[key]
    d.add(sp, f"Агент, {q}", {"agent"}, True, exp, evaluate=True)


def c_explicit_agent(d, split):
    a, b, c = d.ids[:3]
    t, p = rtime(d.rng), rplace(d.rng)
    d.fillers(d.rng.randint(0, 2))
    d.facts(a, t, p)
    d.fillers(d.rng.randint(1, 3))
    ask_agent(d, b, d.rng.choice(["time", "place", "both"]), t, p, split)
    d.fillers(d.rng.randint(0, 2))
    ask_agent(d, c, d.rng.choice(["time", "place"]), t, p, split)


def c_explicit_other(d, split):
    a, b, c = d.ids[:3]
    t, p = rtime(d.rng), rplace(d.rng)
    d.facts(a, t, p)
    d.fillers(d.rng.randint(1, 2))
    d.add(c, f"{d.name[b]}, ты помнишь, {d.rng.choice(['во сколько встреча?', 'где встреча?'])}", {b})
    d.add(b, f"Да, {d.rng.choice(['помню', 'конечно'])}.", {c})
    q, ans = d.rng.choice(PERSONAL_Q)
    d.add(a, f"{d.name[c]}, {q}", {c})
    d.add(c, ans, {a})


def c_group(d, split):
    a, b, c = d.ids[:3]
    t, p = rtime(d.rng), rplace(d.rng)
    d.facts(a, t, p)
    d.fillers(d.rng.randint(1, 3))
    key = d.rng.choice(["time", "place"])
    d.add(b, phr(d.rng, f"group_{key}", split), "ALL", True, [t if key == "time" else p], evaluate=True)
    d.fillers(d.rng.randint(0, 2))
    d.add(c, phr(d.rng, "group_none", split), "ALL", False)


def c_followup(d, split):
    a, b, c = d.ids[:3]
    t, p = rtime(d.rng), rplace(d.rng)
    d.facts(a, t, p)
    d.fillers(d.rng.randint(1, 3))
    ask_agent(d, b, "time", t, p, split)
    d.add(b, phr(d.rng, "follow_place", split), {"agent"}, True, [p], evaluate=True)


def c_interleaved(d, split):
    a, b, c, e = d.ids[:4]
    d.fillers(d.rng.randint(0, 2))
    (q1, a1), (q2, a2) = d.rng.sample(PERSONAL_Q, 2)
    fq, fa = d.rng.choice(FOLLOW_Q)
    d.add(a, f"{d.name[b]}, {q1}", {b})
    d.add(c, f"{d.name[e]}, {q2}", {e})
    d.add(b, a1, {a})
    d.add(e, a2, {c})
    d.add(b, fq, {a})
    d.add(a, fa, {b})


def c_private(d, split):
    a, b, c = d.ids[:3]
    code = str(d.rng.randint(1000, 9999))
    d.fillers(d.rng.randint(0, 2))
    d.add(a, f"Агент, {d.rng.choice(['запомни', 'запиши'])}: код от двери {code}.", {"agent"}, True, visible=[a])
    d.fillers(d.rng.randint(1, 3))
    q = phr(d.rng, "code", split)
    d.add(b, f"Агент, {q}", {"agent"}, True, [], [code], evaluate=True)
    d.fillers(d.rng.randint(0, 2))
    d.add(a, f"Агент, {phr(d.rng, 'code', split)}", {"agent"}, True, [code], evaluate=True)


def c_update(d, split):
    a, b, c = d.ids[:3]
    t1, p1 = rtime(d.rng), rplace(d.rng)
    d.facts(a, t1, p1)
    d.fillers(d.rng.randint(2, 6))
    if d.rng.random() < 0.5:
        t2 = rtime(d.rng)
        while t2 == t1:
            t2 = rtime(d.rng)
        d.add(b, f"Встречу перенесли на {t2}.")
        d.fillers(d.rng.randint(1, 3))
        d.add(c, f"Агент, {phr(d.rng, 'time', split)}", {"agent"}, True, [t2], [t1], evaluate=True)
    else:
        p2 = rplace(d.rng)
        while p2 == p1:
            p2 = rplace(d.rng)
        d.add(b, f"Место поменяли, теперь {loc(p2)}.")
        d.fillers(d.rng.randint(1, 3))
        d.add(c, f"Агент, {phr(d.rng, 'both', split)}", {"agent"}, True, [t1, p2], [p1], evaluate=True)


def c_long(d, split):
    a, b = d.ids[:2]
    t, p = rtime(d.rng), rplace(d.rng)
    d.facts(a, t, p)
    d.fillers(d.rng.randint(9, 14))
    ask_agent(d, b, d.rng.choice(["time", "place", "both"]), t, p, split)


CATEGORIES = {"explicit_agent": (c_explicit_agent, 3), "explicit_other": (c_explicit_other, 3),
              "group_question": (c_group, 3), "agent_followup": (c_followup, 3),
              "interleaved_threads": (c_interleaved, 4), "private_info": (c_private, 3),
              "fact_update": (c_update, 3), "long_range": (c_long, 3)}


def generate(split: str, per_category: int, seed: int):
    rng = random.Random(seed)
    dialogs, gold = [], []
    for cat, (fn, n) in CATEGORIES.items():
        for k in range(per_category):
            d = D(f"{split}_{cat}_{k + 1:02d}", cat, rng, n + rng.randint(0, 1))
            fn(d, split)
            dd, gg = d.dump()
            dialogs.append(dd)
            gold.append(gg)
    return dialogs, gold


def save(rows, path):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    for split, n, seed in [("dev", 3, 101), ("test", 12, 202)]:
        dialogs, gold = generate(split, n, seed)
        save(dialogs, f"data/{split}.jsonl")
        save(gold, f"data/{split}_gold.jsonl")
        print(split, len(dialogs), "диалогов,", sum(len(d["messages"]) for d in dialogs), "реплик")
