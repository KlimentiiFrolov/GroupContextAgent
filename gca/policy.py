"""Решение об участии агента: P_speak = sigma(b0 + b1*a(agent) + b2*q + b3*f - b4*o), d_t = [P_speak >= tau]."""
from __future__ import annotations

from dataclasses import dataclass

from .addressivity import sigmoid


@dataclass(frozen=True)
class PolicyParams:
    beta0: float = -3.0
    beta1: float = 4.0   # адресованность агенту
    beta2: float = 1.5   # признак вопроса или просьбы
    beta3: float = 1.5   # наличие доступного запрашиваемого факта
    beta4: float = 3.0   # обращение исключительно к другим участникам
    tau: float = 0.5


def speak_probability(a_agent: float, q: bool, f: bool, o: bool, p: PolicyParams) -> float:
    return sigmoid(p.beta0 + p.beta1 * a_agent + p.beta2 * q + p.beta3 * f - p.beta4 * o)
