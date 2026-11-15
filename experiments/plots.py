"""Графики для отчёта по results/results.json. python -m experiments.plots"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.loads(Path("results/results.json").read_text(encoding="utf-8"))
OUT = Path("docs")
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def pct(v):
    return f"{100 * v:.1f}".replace(".", ",")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, labels, a, b, ymin=0):
    x = range(len(labels))
    w = 0.38
    for off, vals, col, hatch, name in [(-w / 2 - 0.01, a, ORANGE, "///", "Базовый подход"),
                                        (w / 2 + 0.01, b, BLUE, "", "Предлагаемый метод")]:
        bs = ax.bar([i + off for i in x], [100 * v for v in vals], w, color=col, hatch=hatch, edgecolor="white",
                    linewidth=0.6, label=name)
        for r, v in zip(bs, vals):
            ax.text(r.get_x() + r.get_width() / 2, 100 * v + 1, pct(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(ymin, 108)
    ax.set_ylabel("Значение, %")
    grid(ax)


def fig_metrics():
    s = R["summary"]
    keys = [("addr_acc", "Определение\nадресата"), ("reply_acc", "Решение об\nучастии (доля\nверных)"),
            ("reply_f1", "Решение об\nучастии (F1)"), ("context_acc", "Корректность\nконтекста"),
            ("answer_acc", "Корректность\nответа")]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bars(ax, [k[1] for k in keys], [s["baseline"][k] for k, _ in keys], [s["proposed"][k] for k, _ in keys])
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig_metrics.png", dpi=220)


def fig_categories():
    names = {"explicit_agent": "Явное\nобращение", "group_question": "Вопрос\nгруппе", "agent_followup": "Уточняющий\nвопрос",
             "private_info": "Закрытые\nсведения", "fact_update": "Изменение\nфакта", "long_range": "Давний\nфакт"}
    c = R["categories"]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    bars(ax, list(names.values()), [c["baseline"][k]["answer"] for k in names], [c["proposed"][k]["answer"] for k in names])
    ax.set_ylabel("Корректность ответа, %")
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig_categories.png", dpi=220)


def fig_ablation():
    s = R["summary"]
    var = [("proposed", "Полный метод"), ("no_graph", "Без графа $E_t$"), ("no_memory", "Без памяти $M_t$"),
           ("no_access", "Без учёта\nдоступа $B_t$"), ("baseline", "Базовый подход")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.0), sharey=True)
    for ax, (key, title) in zip(axes, [("addr_acc", "Определение адресата"), ("reply_f1", "Решение об участии, F1"),
                                       ("answer_acc", "Корректность ответа")]):
        vals = [s[v][key] for v, _ in var]
        ys = list(range(len(var)))[::-1]
        ax.barh(ys, [100 * v for v in vals], 0.6, color=[BLUE] * 4 + [ORANGE])
        for y, v in zip(ys, vals):
            ax.text(100 * v - 2, y, pct(v), ha="right", va="center", fontsize=9, color="white")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_yticks(ys, [n for _, n in var])
        grid(ax, "x")
    axes[1].set_xlabel("Значение, %")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.png", dpi=220)


def fig_window():
    w = R["window"]
    ks = [int(k) for k in w]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(ks, [100 * v for v in w.values()], color=ORANGE, linewidth=2, marker="o", markersize=6,
            label="Базовый подход (окно k реплик)")
    p = R["summary"]["proposed"]["answer_acc"]
    ax.axhline(100 * p, color=BLUE, linewidth=2, linestyle="--", label=f"Предлагаемый метод ({pct(p)} %)")
    ax.set_xlabel("Длина окна истории k, реплик")
    ax.set_ylabel("Корректность ответа, %")
    ax.set_xticks(ks)
    ax.set_ylim(0, 100)
    grid(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_window.png", dpi=220)


def fig_tau():
    taus = [float(t) for t in R["tau_dev"]]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.axvspan(0.4, 0.7, color=GRID, alpha=0.7, linewidth=0)
    ax.plot(taus, [100 * v for v in R["tau_dev"].values()], color=MUTED, linewidth=2, marker="s", markersize=6,
            label="Настроечная выборка")
    ax.plot(taus, [100 * v["reply_f1"] for v in R["tau_test"].values()], color=BLUE, linewidth=2, marker="o",
            markersize=6, label="Тестовая выборка")
    ax.axvline(0.5, color=INK, linewidth=1, linestyle=":")
    ax.text(0.51, 81, "τ = 0,5", fontsize=9)
    ax.set_xlabel("Порог τ")
    ax.set_ylabel("F1 решения об участии, %")
    ax.set_ylim(80, 101)
    ax.set_xticks(taus, [str(t).replace(".", ",") for t in taus])
    grid(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_tau.png", dpi=220)


if __name__ == "__main__":
    for f in [fig_metrics, fig_categories, fig_ablation, fig_window, fig_tau]:
        f()
    print("ok")
