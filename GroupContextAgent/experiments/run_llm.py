"""Эксперимент с настоящей языковой моделью: тот же тестовый набор, ответы формирует LLM.

Модель подключается через OpenAI-совместимый API, например локально через Ollama:
    ollama pull qwen2.5:7b
    python -m experiments.run_llm --base-url http://localhost:11434/v1 --model qwen2.5:7b

или через облачный сервис (OpenRouter, Groq и т. п.):
    python -m experiments.run_llm --base-url https://openrouter.ai/api/v1 --model <имя> --api-key <ключ>

Результаты: results/llm_<модель>.json (сводка) и results/llm_<модель>_<подход>.csv (построчно).
Проверка без модели (шаблонный генератор вместо LLM): python -m experiments.run_llm --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from gca.generator import OpenAICompatibleGenerator, TemplateGenerator
from gca.io import load_jsonl
from .evaluate import run, summary, mcnemar, wilson
from .run_all import save_rows

OUT = Path("results")


def norm(s: str) -> str:
    """Мягкая нормализация для проверки ответа LLM: регистр, «ё», 10.00 -> 10:00, лишние пробелы."""
    s = s.lower().replace("ё", "е")
    s = re.sub(r"(\d{1,2})[.\-](\d{2})\b", r"\1:\2", s)
    return re.sub(r"\s+", " ", s)


class Logged:
    """Обёртка генератора: повтор при сбое, учёт времени и ошибок."""

    def __init__(self, gen, retries=2):
        self.gen, self.retries = gen, retries
        self.calls, self.errors, self.seconds = 0, 0, 0.0

    def generate(self, ctx):
        self.calls += 1
        for i in range(self.retries + 1):
            t = time.time()
            try:
                out = self.gen.generate(ctx)
                self.seconds += time.time() - t
                return out
            except Exception as e:  # noqa: BLE001 - сеть, таймаут, неверный ответ сервиса
                self.seconds += time.time() - t
                if i == self.retries:
                    self.errors += 1
                    print("  ошибка генерации:", e)
                    return ""
                time.sleep(2)


def lenient(rows):
    ev = [r for r in rows if r["evaluate"]]
    ok = sum(bool(r["pred_reply"]) and all(norm(v) in norm(r["response"]) for v in r["expect"])
             and not any(norm(v) in norm(r["response"]) for v in r["forbid"]) for r in ev)
    return ok, len(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--systems", nargs="+", default=["proposed", "baseline"])
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N диалогов (для пробного запуска)")
    ap.add_argument("--dry-run", action="store_true", help="без модели: шаблонный генератор через тот же конвейер")
    a = ap.parse_args()

    dialogs, gold = load_jsonl("data/test.jsonl"), load_jsonl("data/test_gold.jsonl")
    if a.limit:
        dialogs = dialogs[:a.limit]
    tag = "dry_run" if a.dry_run else re.sub(r"[^\w.-]+", "_", a.model)
    OUT.mkdir(exist_ok=True)
    res = {"model": "TemplateGenerator (dry run)" if a.dry_run else a.model, "base_url": None if a.dry_run else a.base_url,
           "temperature": a.temperature, "n_dialogs": len(dialogs), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "systems": {}}
    rows = {}
    for s in a.systems:
        gen = Logged(TemplateGenerator() if a.dry_run else
                     OpenAICompatibleGenerator(a.base_url, a.model, a.api_key, a.temperature))
        print(f"{s}: прогон {len(dialogs)} диалогов...")
        rows[s] = run(s, dialogs, gold, generator=gen)
        save_rows(rows[s], OUT / f"llm_{tag}_{s}.csv")
        sm = summary(rows[s])
        k, n = lenient(rows[s])
        sm.update({"answer_lenient_k": k, "answer_lenient_acc": k / n if n else 0.0,
                   "answer_wilson": wilson(sm["answer_k"], sm["n_eval"]),
                   "llm_calls": gen.calls, "llm_errors": gen.errors, "llm_seconds": round(gen.seconds, 1)})
        res["systems"][s] = sm
        print(f"  ответ верен: {sm['answer_k']}/{sm['n_eval']} ({100 * sm['answer_acc']:.1f} %), мягкая проверка: "
              f"{k}/{n}; раскрытий: {sm['leak_n']}/{sm['n_forbid']}; вызовов LLM: {gen.calls}, ошибок: {gen.errors}")
    if "proposed" in rows and "baseline" in rows:
        b, p = rows["baseline"], rows["proposed"]
        ev = [i for i, r in enumerate(b) if r["evaluate"]]
        res["mcnemar_answer"] = mcnemar([b[i]["answer_ok"] for i in ev], [p[i]["answer_ok"] for i in ev])
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = OUT / f"llm_{tag}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("сохранено:", path)


if __name__ == "__main__":
    main()
