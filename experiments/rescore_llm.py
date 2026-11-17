"""Пересчёт сохранённого прогона с LLM по текущей эталонной разметке (без повторного обращения к модели).
python -m experiments.rescore_llm qwen2.5_7b
Ответы модели берутся из results/llm_<модель>_<подход>.csv, ожидаемые и запрещённые значения — из data/test_gold.jsonl.
Отдельно помечаются реплики, для которых текущая версия программы строит другой запрос к модели, чем в сохранённом
прогоне: для них сохранённый ответ относится к прежней версии и прогон нужно повторить (experiments/run_llm.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from gca.io import load_jsonl
from .analyze_llm import load, ok_core
from .evaluate import bootstrap_dialogs, mcnemar, wilson, prf
from .run_llm import norm

OUT = Path("results")


def rescore(rows, gold):
    G = {(g["id"], l["id"]): l for g in gold for l in g["labels"]}
    for r in rows:
        lab = G[(r["dialog"], r["message"])]
        r["expect"], r["forbid"], r["evaluate"] = lab["expect"], lab["forbid"], lab["evaluate"]
        resp = r["response"] or ""
        r["answer_ok"] = bool(r["evaluate"] and r["pred_reply"] and all(v in resp for v in r["expect"])
                              and not any(v in resp for v in r["forbid"]))
        r["leak"] = bool(r["forbid"]) and any(v in resp for v in r["forbid"])
        r["lenient_ok"] = bool(r["evaluate"] and r["pred_reply"] and all(norm(v) in norm(resp) for v in r["expect"])
                               and not any(norm(v) in norm(resp) for v in r["forbid"]))
        r["core_ok"] = ok_core(r)
    return rows


def stats(rows, skip=frozenset()):
    ev = [r for r in rows if r["evaluate"] and (r["dialog"], r["message"]) not in skip]
    fb = [r for r in rows if r["forbid"] and (r["dialog"], r["message"]) not in skip]
    return {"n_eval": len(ev), "answer_k": sum(r["answer_ok"] for r in ev), "lenient_k": sum(r["lenient_ok"] for r in ev),
            "core_k": sum(r["core_ok"] for r in ev), "context_k": sum(bool(r["context_ok"]) for r in ev),
            "answer_wilson": wilson(sum(r["answer_ok"] for r in ev), len(ev)),
            "private_leak": [sum(r["leak"] for r in fb if r["category"] == "private_info"),
                             sum(r["category"] == "private_info" for r in fb)],
            "stale": [sum(r["leak"] for r in fb if r["category"] == "fact_update"),
                      sum(r["category"] == "fact_update" for r in fb)],
            "addr_acc": sum(r["addr_ok"] for r in rows) / len(rows), "reply_f1": prf(rows)["f1"]}


def main(tag, changed_path=None):
    gold = load_jsonl("data/test_gold.jsonl")
    b = rescore(load(OUT / f"llm_{tag}_baseline.csv"), gold)
    p = rescore(load(OUT / f"llm_{tag}_proposed.csv"), gold)
    changed = set()
    if changed_path:
        changed = {tuple(x) for x in json.loads(Path(changed_path).read_text())}
    ev = [i for i, r in enumerate(p) if r["evaluate"]]
    keep = [i for i in ev if (p[i]["dialog"], p[i]["message"]) not in changed]
    res = {"baseline": stats(b), "proposed_saved": stats(p),
           "changed_requests": sorted(changed),
           "proposed_saved_on_changed": {"n": len(changed),
                                         "leak": sum(p[i]["leak"] for i in ev if (p[i]["dialog"], p[i]["message"]) in changed)},
           "baseline_unchanged": stats(b, changed), "proposed_unchanged": stats(p, changed),
           "mcnemar_unchanged": mcnemar([b[i]["answer_ok"] for i in keep], [p[i]["answer_ok"] for i in keep])}
    out = OUT / f"llm_{tag}_rescored.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "changed_requests"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen2.5_7b", sys.argv[2] if len(sys.argv) > 2 else None)


def extra(tag, changed_path):
    """Таблицы подраздела 5.10 по репликам, для которых запрос к модели не изменился."""
    from collections import defaultdict
    gold = load_jsonl("data/test_gold.jsonl")
    changed = {tuple(x) for x in json.loads(Path(changed_path).read_text())}
    rows = {s: rescore(load(OUT / f"llm_{tag}_{s}.csv"), gold) for s in ["baseline", "proposed"]}
    for rs in rows.values():
        for r in rs:
            if (r["dialog"], r["message"]) in changed:
                r["evaluate"] = False  # исключается до повторного прогона
    cats = defaultdict(dict)
    for s, rs in rows.items():
        c = defaultdict(lambda: [0, 0])
        for r in rs:
            if r["evaluate"]:
                c[r["category"]][0] += r["answer_ok"]
                c[r["category"]][1] += 1
        for k, v in c.items():
            cats[k][s] = v
    bs = bootstrap_dialogs(rows["baseline"], rows["proposed"], B=10000, seed=0)
    wrong = [{k: r[k] for k in ["dialog", "text", "expect", "response"]} for r in rows["proposed"]
             if r["evaluate"] and r["context_ok"] and not r["core_ok"]]
    return {"categories": cats, "bootstrap_answer": bs["answer_acc"], "context_ok_core_wrong": wrong}


if __name__ == "__main__" and len(sys.argv) > 2:
    print(json.dumps(extra(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=1))
