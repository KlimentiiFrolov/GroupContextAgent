"""GroupContextAgent — прототип LLM-агента с контекстным моделированием группового взаимодействия."""
from .models import Message, Participant, Fact, GroupState
from .agent import GroupContextAgent, AgentConfig
from .baseline import HistoryBaselineAgent

__all__ = ["Message", "Participant", "Fact", "GroupState", "GroupContextAgent", "AgentConfig", "HistoryBaselineAgent"]
