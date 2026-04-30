"""Agents — Engineering reasoning and expert reporting for FMCW signal diagnostics.

This package implements the two upper layers of the diagnostic stack:
  - EngineeringReasoner: multi-step physical/mathematical reasoning chain
  - LLMReasoner: LLM-powered reasoning via OpenAI-compatible API
  - ExpertReporter: LaTeX diagnostic report generation
"""

__all__ = [
    "EngineeringReasoner",
    "ReasoningResult",
    "LLMReasoner",
    "LLMConfig",
    "ExpertReporter",
    "KnowledgeBase",
    "Orchestrator",
]

from .engineering_reasoner import EngineeringReasoner, KnowledgeBase, ReasoningResult

try:
    from .llm_reasoner import LLMConfig, LLMReasoner
except ImportError:
    pass

from .expert_reporter import ExpertReporter
from .orchestrator import Orchestrator
