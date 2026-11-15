"""Дополнительный анализ прогона с LLM по построчным файлам results/llm_<модель>_<подход>.csv:
бутстреп по диалогам, корректность ответа по типам сценариев, проверка по ключевым значениям.
python -m experiments.analyze_llm qwen2.5_7b"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from .evaluate import bootstrap_dialogs, mcnemar

OUT = Path("results")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if v in ("True", "False"):
                    d[k] = v == "True"
                elif v[:1] in "[{":
                    try:
                        d[k] = json.loads(v)
                    except ValueError:
                        d[k] = v
                else:
                    d[k] = v
            rows.append(d)
    return rows


def core(v):
    """Ключевая часть значения: числа, время, ссылки (без учёта падежа слов «кабинет», «аудитория»)."""
    return re.findall(r"\d+(?::\d+)?|https?://\S+", v.lower())


def ok_core(r):
    if not (r["evaluate"] and r["pred_reply"]):
        return False
    resp = r["response"].lower()
    alt = re.sub(r"(\d{1,2})[.\-](\d{2})\b", r"\1:\2", resp)

    def has(v):
        c = core(v)
        return bool(c) and all(t in resp or t in alt for t in c)
    return all(has(v) for v in r["expect"]) and not any(has(v) for v in r["forbid"])


def main(tag):
    p = load(OUT / f"llm_{tag}_proposed.csv")
    b = load(OUT / f"llm_{tag}_baseline.csv")
    ev = [i for i, r in enumerate(p) if r["evaluate"]]
    res = {"bootstrap": bootstrap_dialogs(b, p, B=10000, seed=0),
           "mcnemar_answer": mcnemar([b[i]["answer_ok"] for i in ev], [p[i]["answer_ok"] for i in ev]),
           "core": {s: [sum(ok_core(rows[i]) for i in ev), len(ev)] for s, rows in [("baseline", b), ("proposed", p)]},
           "categories": defaultdict(dict)}
    for s, rows in [("baseline", b), ("proposed", p)]:
        cat = defaultdict(lambda: [0, 0])
        for i in ev:
            c = cat[rows[i]["category"]]
            c[0] += rows[i]["answer_ok"]
            c[1] += 1
        for k, v in cat.items():
            res["categories"][k][s] = v
    order = []
    for r in p:
        if r["dialog"] not in order:
            order.append(r["dialog"])
    first = set(order[:24])
    res["first24"] = {s: [sum(rows[i]["answer_ok"] for i in ev if rows[i]["dialog"] in first),
                          sum(1 for i in ev if rows[i]["dialog"] in first)] for s, rows in [("baseline", b), ("proposed", p)]}
    res["context_ok_answer_wrong"] = [{k: p[i][k] for k in ["dialog", "category", "text", "expect", "response"]}
                                      for i in ev if p[i]["context_ok"] and not ok_core(p[i])]
    path = OUT / f"llm_{tag}_analysis.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "context_ok_answer_wrong"}, ensure_ascii=False))
    print("сохранено:", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen2.5_7b")
