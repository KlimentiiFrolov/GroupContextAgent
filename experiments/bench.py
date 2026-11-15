"""Производительность агента: среднее время обработки одной реплики (без генерации LLM).
1) тестовая выборка (96 диалогов, 861 реплика) для обоих подходов;
2) один длинный диалог из N реплик (тестовые диалоги, склеенные подряд) – зависимость от длины.
python -m experiments.bench   ->  results/bench.json"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from gca import GroupContextAgent, HistoryBaselineAgent
from gca.io import load_jsonl, messages_from, participants_from

REPEATS = 5


def run_dialogs(make, dialogs):
    n, t = 0, 0.0
    for d in dialogs:
        agent = make(participants_from(d))
        msgs = messages_from(d)
        t0 = time.perf_counter()
        for m in msgs:
            agent.process(m)
        t += time.perf_counter() - t0
        n += len(msgs)
    return t, n


def long_dialog(dialogs, n_target):
    """Склейка диалогов с одинаковым составом участников в один длинный диалог."""
    base = dialogs[0]
    key = [p["id"] for p in base["participants"]]
    same = [d for d in dialogs if [p["id"] for p in d["participants"]] == key]
    out, turn = [], 0
    while len(out) < n_target:
        for d in same:
            for m in d["messages"]:
                turn += 1
                mm = dict(m, id=f"x{turn}", turn=turn)
                mm.pop("reply_to", None)
                out.append(mm)
                if len(out) == n_target:
                    break
            if len(out) == n_target:
                break
    return dict(base, messages=out)


def main():
    dialogs = load_jsonl("data/test.jsonl")
    res = {"python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
           "platform": platform.platform(), "repeats": REPEATS, "test": {}, "long": []}
    try:
        res["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except OSError:
        pass
    for name, make in [("proposed", GroupContextAgent), ("baseline", HistoryBaselineAgent)]:
        best = min(run_dialogs(make, dialogs)[0] for _ in range(REPEATS))
        n = run_dialogs(make, dialogs)[1]
        res["test"][name] = {"messages": n, "seconds": best, "ms_per_message": 1000 * best / n}
    for n in (100, 1000, 5000):
        d = long_dialog(dialogs, n)
        row = {"messages": n}
        for name, make in [("proposed", GroupContextAgent), ("baseline", HistoryBaselineAgent)]:
            best = min(run_dialogs(make, [d])[0] for _ in range(REPEATS))
            row[name + "_ms"] = 1000 * best / n
        res["long"].append(row)
    Path("results").mkdir(exist_ok=True)
    Path("results/bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
