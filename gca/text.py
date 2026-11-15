"""Анализ текста реплики: обращения, признаки запроса, извлечение фактов и запрашиваемых сведений."""
from __future__ import annotations

import re
from typing import Iterable

WORD = re.compile(r"[а-яёa-z0-9@]+", re.IGNORECASE)
GROUP = re.compile(r"\b(ребят[а-я]*|народ|коллеги|всем|все,|кто-нибудь|кто нибудь|кто-то|друзья)\b", re.IGNORECASE)
REQUEST_VERBS = {"напомни", "напомните", "подскажи", "подскажите", "скажи", "скажите", "уточни", "уточните",
                 "повтори", "повторите", "расскажи", "проверь", "запомни", "запиши"}

MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b")
DATE_RE = re.compile(rf"\b(\d{{1,2}})\s+({MONTHS})\b", re.IGNORECASE)
PLACE_RE = re.compile(r"\b(аудитори[яиюей]|кабинет[еа]?|переговорн[а-я]*)\s*(?:№\s*)?(\d{2,4}[а-я]?)\b", re.IGNORECASE)
LINK_RE = re.compile(r"https?://\S+")
CODE_RE = re.compile(r"\bкод[а-я]*\b[^0-9]{0,20}?(\d{3,6})\b", re.IGNORECASE)

PLACE_NORM = {"аудитори": "аудитория", "кабинет": "кабинет", "переговорн": "переговорная"}

# запрашиваемые сведения: ключ факта -> шаблоны вопроса
INTENTS = {
    "meeting_time": re.compile(r"во сколько|время|\bкогда\b|к какому часу|в котором часу", re.IGNORECASE),
    "meeting_place": re.compile(r"\bгде\b|место|куда|в какой аудитори|какой кабинет|какая аудитория", re.IGNORECASE),
    "meeting_date": re.compile(r"какого числа|какое число|дат[аеу]|в какой день", re.IGNORECASE),
    "deadline": re.compile(r"дедлайн|срок|до какого", re.IGNORECASE),
    "link": re.compile(r"ссылк", re.IGNORECASE),
    "door_code": re.compile(r"\bкод", re.IGNORECASE),
}
KEY_RU = {"meeting_time": "время встречи", "meeting_place": "место встречи", "meeting_date": "дата встречи",
          "deadline": "срок сдачи", "link": "ссылка", "door_code": "код от двери"}


def words(text: str) -> list[str]:
    return [w.lower().strip(",") for w in WORD.findall(text)]


def explicit_addressees(text: str, forms: dict[str, set[str]]) -> set[str]:
    """Явные обращения: имя в начале реплики (первые два слова), имя перед запятой или @имя."""
    low = text.lower()
    ws = words(text)
    head = set(ws[:2])
    found = set()
    for pid, names in forms.items():
        for n in names:
            if n in head or re.search(rf"(^|[\s,]){re.escape(n)},", low) or f"@{n}" in low or \
                    re.search(rf"\bи {re.escape(n)}\b,", low):
                found.add(pid)
                break
    return found


def is_group_address(text: str) -> bool:
    return bool(GROUP.search(text))


def is_request(text: str) -> bool:
    return "?" in text or bool(REQUEST_VERBS & set(words(text)))


def requested_keys(text: str) -> list[str]:
    return [k for k, rx in INTENTS.items() if rx.search(text)]


def extract_facts(text: str) -> dict[str, str]:
    """Извлекает сообщаемые факты. Из вопросов факты не извлекаются."""
    if "?" in text:
        return {}
    facts: dict[str, str] = {}
    low = text.lower()
    for m in TIME_RE.finditer(text):
        facts["meeting_time"] = f"{int(m.group(1))}:{m.group(2)}"
    for m in DATE_RE.finditer(text):
        key = "deadline" if re.search(r"дедлайн|сдать|срок|сдача", low) else "meeting_date"
        facts[key] = f"{int(m.group(1))} {m.group(2).lower()}"
    for m in PLACE_RE.finditer(text):
        kind = next(v for k, v in PLACE_NORM.items() if m.group(1).lower().startswith(k))
        facts["meeting_place"] = f"{kind} {m.group(2)}"
    for m in LINK_RE.finditer(text):
        facts["link"] = m.group(0).rstrip(".,")
    for m in CODE_RE.finditer(text):
        facts["door_code"] = m.group(1)
        facts.pop("meeting_time", None)
    return facts


def join_ru(items: Iterable[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " и " + items[-1]
