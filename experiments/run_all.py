"""Полный эксперимент: сравнение, абляция, чувствительность, статистика, графики.
python -m experiments.run_all"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from gca.io import load_jsonl
from .evaluate import run, summary, mcnemar, wilson, bootstrap_dialogs

OUT = Path("results")


def load(split):
    return load_jsonl(f"data/{split}.jsonl"), load_jsonl(f"data/{split}_gold.jsonl")


def save_rows(rows, path):
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def by_category(rows):
    cats = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(r)
    return {c: {"addr": sum(x["addr_ok"] for x in rs) / len(rs),
                "reply": sum(x["pred_reply"] == x["gold_reply"] for x in rs) / len(rs),
                "answer": (sum(x["answer_ok"] for x in rs if x["evaluate"]) / max(1, sum(x["evaluate"] for x in rs)))
                if any(x["evaluate"] for x in rs) else None,
                "n": len(rs), "n_eval": sum(x["evaluate"] for x in rs)} for c, rs in cats.items()}


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    dev, dev_g = load("dev")
    test, test_g = load("test")
    # 1. выбор порога tau на настроечной выборке
    taus = [round(0.1 * i, 1) for i in range(1, 10)]
    res["tau_dev"] = {t: summary(run("proposed", dev, dev_g, tau=t))["reply_f1"] for t in taus}
    res["tau_test"] = {t: summary(run("proposed", test, test_g, tau=t)) for t in taus}
    # 2. основное сравнение и абляция на тестовой выборке
    rows = {s: run(s, test, test_g) for s in ["baseline", "proposed", "no_memory", "no_access", "no_graph"]}
    for s, r in rows.items():
        save_rows(r, OUT / f"test_{s}.csv")
    res["summary"] = {s: summary(r) for s, r in rows.items()}
    res["dev_summary"] = {s: summary(run(s, dev, dev_g)) for s in ["baseline", "proposed"]}
    res["categories"] = {s: by_category(rows[s]) for s in ["baseline", "proposed"]}
    n = res["summary"]["proposed"]
    res["wilson"] = {s: {m: wilson(res["summary"][s][f"{m}_k"], res["summary"][s]["n_messages" if m in ("addr", "reply") else "n_eval"])
                         for m in ["addr", "reply", "context", "answer"]} for s in ["baseline", "proposed"]}
    # 3. парные сравнения (критерий Мак-Немара)
    b, p = rows["baseline"], rows["proposed"]
    ev = [i for i, r in enumerate(b) if r["evaluate"]]
    res["mcnemar"] = {
        "addr": mcnemar([r["addr_ok"] for r in b], [r["addr_ok"] for r in p]),
        "reply": mcnemar([r["pred_reply"] == r["gold_reply"] for r in b], [r["pred_reply"] == r["gold_reply"] for r in p]),
        "answer": mcnemar([b[i]["answer_ok"] for i in ev], [p[i]["answer_ok"] for i in ev]),
    }
    # 3а. бутстреп по целым диалогам (проверка вывода без допущения о независимости реплик)
    res["bootstrap"] = bootstrap_dialogs(b, p, B=10000, seed=0)
    # 4. чувствительность базового подхода к длине окна истории
    res["window"] = {k: summary(run("baseline", test, test_g, k=k))["answer_acc"] for k in [2, 4, 6, 8, 12, 16, 24, 32]}
    # 5. корректность ответа на знакомых и новых (не встречавшихся при настройке) формулировках
    from .corpus import ASK
    extra = [t for _, e in ASK.values() for t in e]
    res["phrasing"] = {}
    for s_ in ["baseline", "proposed"]:
        ev_ = [r for r in rows[s_] if r["evaluate"]]
        new = [r for r in ev_ if any(t in r["text"] for t in extra)]
        old = [r for r in ev_ if r not in new]
        res["phrasing"][s_] = {"seen": [sum(r["answer_ok"] for r in old), len(old)],
                               "unseen": [sum(r["answer_ok"] for r in new), len(new)]}
    # 6. ошибки предлагаемого подхода
    res["errors"] = [{k: r[k] for k in ["dialog", "category", "message", "text", "gold_addr", "pred_addr", "gold_reply",
                                         "pred_reply", "p_speak", "expect", "response", "addr_ok", "answer_ok"]}
                     for r in p if (not r["addr_ok"]) or r["pred_reply"] != r["gold_reply"] or (r["evaluate"] and not r["answer_ok"])]
    Path(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for s, v in res["summary"].items():
        print(f"{s:10} addr {v['addr_acc']:.3f}  reply acc {v['reply_acc']:.3f} P {v['reply_precision']:.3f} "
              f"R {v['reply_recall']:.3f} F1 {v['reply_f1']:.3f}  ctx {v['context_acc']:.3f}  ans {v['answer_acc']:.3f}  "
              f"leak {v['leak_n']}/{v['n_forbid']}")
    print("dev:", {s: (round(v['reply_f1'], 3), round(v['answer_acc'], 3)) for s, v in res["dev_summary"].items()})
    print("tau dev F1:", res["tau_dev"])
    print("mcnemar:", res["mcnemar"])
    print("bootstrap:", res["bootstrap"])
    print("window:", res["window"])
    print("errors:", len(res["errors"]))


if __name__ == "__main__":
    main()
