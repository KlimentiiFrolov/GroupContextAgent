"""Запуск агента на диалоге из JSON: python main.py data/demo_meeting.json [--mode proposed|baseline] [--state]"""
import argparse
import json
from pathlib import Path

from gca import GroupContextAgent, HistoryBaselineAgent
from gca.io import messages_from, participants_from
from gca.models import ALL


def main() -> None:
    ap = argparse.ArgumentParser(description="GroupContextAgent: обработка многопользовательского диалога")
    ap.add_argument("dialog")
    ap.add_argument("--mode", choices=["proposed", "baseline"], default="proposed")
    ap.add_argument("--state", action="store_true", help="вывести итоговое состояние группы в JSON")
    a = ap.parse_args()
    d = json.loads(Path(a.dialog).read_text(encoding="utf-8"))
    ps = participants_from(d)
    agent = GroupContextAgent(ps) if a.mode == "proposed" else HistoryBaselineAgent(ps)
    names = {p.id: p.name for p in ps}
    for m in messages_from(d):
        r = agent.process(m)
        adr = "группа" if r.addressees == ALL else ", ".join(names[x] for x in sorted(r.addressees))
        print(f"[{m.turn:>2}] {names[m.speaker]}: {m.text}")
        print(f"     адресаты: {adr}; P_speak = {r.p_speak:.3f}; решение: {'ответить' if r.reply else 'наблюдать'}")
        if r.reply:
            print(f"     Агент: {r.response}")
    if a.state and a.mode == "proposed":
        print(json.dumps(agent.state.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
