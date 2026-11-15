"""Запуск систем на корпусе и расчёт метрик."""
from __future__ import annotations

import math
from dataclasses import replace

from gca import GroupContextAgent, HistoryBaselineAgent, AgentConfig
from gca.io import messages_from, participants_from
from gca.models import ALL


def make(system: str, ps, **kw):
    if system == "baseline":
        return HistoryBaselineAgent(ps, history_window=kw.get("k", 8), generator=kw.get("generator"))
    cfg = AgentConfig()
    flags = {"proposed": {}, "no_memory": {"use_memory": False}, "no_access": {"use_access": False},
             "no_graph": {"use_graph": False}}[system]
    cfg = replace(cfg, **flags)
    if "tau" in kw:
        cfg = replace(cfg, policy=replace(cfg.policy, tau=kw["tau"]))
    return GroupContextAgent(ps, cfg, generator=kw.get("generator"))


def run(system: str, dialogs, gold, **kw) -> list[dict]:
    """Возвращает построчные записи: одна запись на реплику участника."""
    rows = []
    G = {g["id"]: {l["id"]: l for l in g["labels"]} for g in gold}
    for d in dialogs:
        agent = make(system, participants_from(d), **kw)
        for m in messages_from(d):
            r = agent.process(m)
            lab = G[d["id"]][m.id]
            pred = "ALL" if r.addressees == ALL else sorted(r.addressees)
            resp = r.response or ""
            ctx = r.context.facts if r.context else {}
            rows.append({
                "dialog": d["id"], "category": d["category"], "message": m.id, "speaker": m.speaker, "text": m.text,
                "gold_addr": lab["addressees"], "pred_addr": pred, "addr_ok": pred == lab["addressees"],
                "gold_reply": lab["reply"], "pred_reply": r.reply, "p_speak": round(r.p_speak, 4),
                "evaluate": lab["evaluate"], "expect": lab["expect"], "forbid": lab["forbid"],
                "context_ok": lab["evaluate"] and r.reply and all(v in ctx.values() for v in lab["expect"])
                              and not any(v in ctx.values() for v in lab["forbid"]),
                "answer_ok": lab["evaluate"] and r.reply and all(v in resp for v in lab["expect"])
                             and not any(v in resp for v in lab["forbid"]),
                "leak": bool(lab["forbid"]) and any(v in resp for v in lab["forbid"]),
                "response": resp,
            })
    return rows


def prf(rows):
    tp = sum(r["pred_reply"] and r["gold_reply"] for r in rows)
    fp = sum(r["pred_reply"] and not r["gold_reply"] for r in rows)
    fn = sum(not r["pred_reply"] and r["gold_reply"] for r in rows)
    tn = sum(not r["pred_reply"] and not r["gold_reply"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": p, "recall": rc, "f1": f1,
            "accuracy": (tp + tn) / len(rows)}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def summary(rows) -> dict:
    ev = [r for r in rows if r["evaluate"]]
    priv = [r for r in rows if r["forbid"]]
    d = prf(rows)
    out = {"n_messages": len(rows), "n_eval": len(ev),
           "addr_acc": sum(r["addr_ok"] for r in rows) / len(rows), "addr_k": sum(r["addr_ok"] for r in rows),
           "reply_acc": d["accuracy"], "reply_k": d["TP"] + d["TN"], **{f"reply_{k}": v for k, v in d.items() if k != "accuracy"},
           "context_acc": sum(r["context_ok"] for r in ev) / len(ev), "context_k": sum(r["context_ok"] for r in ev),
           "answer_acc": sum(r["answer_ok"] for r in ev) / len(ev), "answer_k": sum(r["answer_ok"] for r in ev),
           "leak_n": sum(r["leak"] for r in priv), "n_forbid": len(priv)}
    return out


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Точный критерий Мак-Немара по парам несогласованных исходов."""
    n01 = sum((not x) and y for x, y in zip(a, b))   # A ошибся, B прав
    n10 = sum(x and (not y) for x, y in zip(a, b))
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return {"n01": n01, "n10": n10, "p": min(1.0, p)}


def bootstrap_dialogs(a_rows, b_rows, B: int = 10000, seed: int = 0) -> dict:
    """Кластерный бутстреп: повторная выборка целых диалогов (реплики одного диалога не считаются независимыми).
    Для каждой метрики возвращает разность b − a на исходной выборке, 95-процентный перцентильный интервал разности
    и долю повторных выборок, в которых разность не больше нуля."""
    import random
    from collections import defaultdict
    da, db = defaultdict(list), defaultdict(list)
    for r in a_rows:
        da[r["dialog"]].append(r)
    for r in b_rows:
        db[r["dialog"]].append(r)
    ids = sorted(da)

    def metrics(rows):
        ev = [r for r in rows if r["evaluate"]]
        return {"addr_acc": sum(r["addr_ok"] for r in rows) / len(rows),
                "reply_f1": prf(rows)["f1"],
                "answer_acc": sum(r["answer_ok"] for r in ev) / len(ev) if ev else 0.0}

    base = {k: metrics(b_rows)[k] - v for k, v in metrics(a_rows).items()}
    rng = random.Random(seed)
    diffs = {k: [] for k in base}
    for _ in range(B):
        pick = [rng.choice(ids) for _ in ids]
        ma = metrics([r for i in pick for r in da[i]])
        mb = metrics([r for i in pick for r in db[i]])
        for k in diffs:
            diffs[k].append(mb[k] - ma[k])
    out = {}
    for k, v in diffs.items():
        v.sort()
        out[k] = {"diff": base[k], "lo": v[int(0.025 * B)], "hi": v[int(0.975 * B) - 1],
                  "p_le0": sum(x <= 0 for x in v) / B}
    out["B"], out["n_dialogs"] = B, len(ids)
    return out
