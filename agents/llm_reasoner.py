"""
LLM-Powered Reasoning Agent
============================
Extends the rule-based EngineeringReasoner with real LLM API calls for
deep diagnostic reasoning on anomalous frames.

Supports any OpenAI-compatible API (MiMo, DeepSeek, OpenAI, etc.).

Architecture
------------
  Rule Engine (fast filter)
       │
       ├── frame is nominal → skip LLM, return rule result directly
       │
       └── frame has anomalies → serialise features + knowledge base
                                    │
                                    ▼
                              LLM API call (4-step reasoning prompt)
                                    │
                                    ▼
                              parse structured JSON response
                                    │
                                    ▼
                              ReasoningResult with real token usage

Usage::

    from agents.llm_reasoner import LLMReasoner

    reasoner = LLMReasoner(
        api_key="sk-...",
        base_url="https://api.xiaomimimo.com/v1",
        model="mimo-v2.5",
    )
    result = reasoner.reason(feature_record)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .engineering_reasoner import (
    EngineeringReasoner,
    Hypothesis,
    KnowledgeBase,
    ReasoningResult,
    StepResult,
)

logger = logging.getLogger(__name__)

# Auto-load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass

# Try to import openai; gracefully degrade if not installed
try:
    from openai import OpenAI

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Configuration for the LLM reasoning backend.

    Attributes
    ----------
    api_key : str
        API key.  If empty, reads from ``MIMO_API_KEY``,
        ``DEEPSEEK_API_KEY``, or ``OPENAI_API_KEY`` env vars.
    base_url : str
        OpenAI-compatible base URL.
    model : str
        Model name to use.
    temperature : float
        Sampling temperature (0 = deterministic).
    max_tokens : int
        Maximum output tokens.
    timeout_s : float
        HTTP request timeout in seconds.
    """

    api_key: str = ""
    base_url: str = "https://api.xiaomimimo.com/v1"
    model: str = "mimo-v2.5"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: float = 60.0

    @property
    def resolved_api_key(self) -> str:
        """Resolve API key from config or environment variables."""
        if self.api_key:
            return self.api_key
        for env_var in ("MIMO_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            val = os.environ.get(env_var, "")
            if val:
                return val
        return ""

    def auto_detect_provider(self) -> None:
        """Auto-configure base_url and model from the resolved API key source."""
        if os.environ.get("DEEPSEEK_API_KEY") and self.base_url == "https://api.xiaomimimo.com/v1":
            self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        elif os.environ.get("MIMO_API_KEY"):
            self.base_url = os.environ.get("MIMO_BASE_URL", self.base_url)
            self.model = os.environ.get("MIMO_MODEL", self.model)
        elif os.environ.get("OPENAI_API_KEY") and self.base_url == "https://api.xiaomimimo.com/v1":
            self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


_REASONING_SYSTEM_PROMPT = """You are an expert FMCW radar signal diagnostic engineer.
Your task is to analyse structured signal feature data and perform multi-step
diagnostic reasoning to identify the root cause of anomalies.

You have access to a domain knowledge base covering:
- FMCW radar physics (beat frequency, phase sensitivity, Doppler)
- Multipath reflection discrimination
- Mechanical vibration vs. biological micro-Doppler differentiation
- Systematic diagnostic decision tree

Follow this reasoning chain:

STEP 1 — Anomaly Classification:
Parse the signal features. Classify each anomaly by type:
  phase_jump, mechanical_vibration, periodic_displacement,
  micro_doppler, narrowband_interference, broadband_noise

STEP 2 — Physical Modelling:
For each anomaly, apply physical models:
  - Phase jump → displacement: Δx = Δφ · λ / (4π)
  - Doppler frequency → velocity: v = f_d · λ / 2
  - Micro-Doppler frequency band check (respiration 0.1–0.5 Hz,
    heartbeat 0.8–3 Hz)
  - Multipath: check phase correlation with direct target

STEP 3 — Interference Exclusion:
Rule out environmental confounders:
  - Temperature effects on noise floor
  - Nearby machinery (industrial/lab environment?)
  - Multi-target cross-interference
  - AC mains frequency interference (50/60 Hz)

STEP 4 — Root-Cause Synthesis:
Rank hypotheses by confidence. For each hypothesis provide:
  - Name (e.g., multipath_reflection, mechanical_vibration,
    biosignal_micro_doppler, external_interference,
    receiver_noise_excursion, nominal_operation)
  - Confidence (0–1)
  - Supporting evidence list
  - Recommended action

Return ONLY valid JSON in this exact format:
{
  "primary_diagnosis": "hypothesis_name",
  "requires_human_review": true_or_false,
  "steps": [
    {
      "step_name": "Anomaly Classification",
      "findings": ["finding 1", "finding 2"],
      "active_hypotheses": ["h1", "h2"],
      "eliminated_hypotheses": []
    },
    {
      "step_name": "Physical Modelling",
      "findings": ["..."],
      "active_hypotheses": ["..."],
      "eliminated_hypotheses": ["..."]
    },
    {
      "step_name": "Interference Exclusion",
      "findings": ["..."],
      "active_hypotheses": ["..."],
      "eliminated_hypotheses": ["..."]
    },
    {
      "step_name": "Root-Cause Synthesis",
      "findings": ["..."],
      "active_hypotheses": ["primary_hypothesis"],
      "eliminated_hypotheses": ["..."]
    }
  ],
  "hypotheses": [
    {
      "name": "hypothesis_name",
      "confidence": 0.85,
      "evidence": ["evidence 1", "evidence 2"],
      "recommendation": "actionable recommendation"
    }
  ]
}

Do NOT include markdown fences, explanations, or any text outside the JSON."""


def _build_user_prompt(
    feature_text: str, knowledge_context: str, metadata: Dict[str, Any]
) -> str:
    """Construct the user prompt combining features, knowledge, and metadata."""
    meta_lines = "\n".join(f"  {k}: {v}" for k, v in metadata.items())
    return f"""## Domain Knowledge Base

{knowledge_context}

---

## Signal Features (Current Frame)

{feature_text}

---

## Environmental Context

{meta_lines}

---

Analyse the above signal data using the 4-step diagnostic reasoning chain.
Return ONLY the JSON result as specified."""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Dict[str, Any]:
    """Robust JSON extraction from LLM output (handles markdown fences)."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to extract valid JSON from LLM response: {text[:200]}...")


def _parse_llm_response(raw_json: Dict[str, Any]) -> ReasoningResult:
    """Convert the LLM's JSON response into a ReasoningResult."""
    # Parse steps
    steps = []
    for s in raw_json.get("steps", []):
        steps.append(
            StepResult(
                step_name=s.get("step_name", "Unknown"),
                findings=s.get("findings", []),
                eliminated_hypotheses=s.get("eliminated_hypotheses", []),
                active_hypotheses=s.get("active_hypotheses", []),
            )
        )

    # Parse hypotheses
    hypotheses = []
    for h in raw_json.get("hypotheses", []):
        hypotheses.append(
            Hypothesis(
                name=h.get("name", "unknown"),
                confidence=float(h.get("confidence", 0.5)),
                evidence=h.get("evidence", []),
                recommendation=h.get("recommendation", ""),
            )
        )

    # Sort by confidence
    hypotheses.sort(key=lambda h: h.confidence, reverse=True)

    primary_name = raw_json.get("primary_diagnosis", "")
    primary = None
    if primary_name and hypotheses:
        for h in hypotheses:
            if h.name == primary_name:
                primary = h
                break
    if primary is None and hypotheses:
        primary = hypotheses[0]

    return ReasoningResult(
        frame_id=0,  # will be set by caller
        input_summary="LLM-powered diagnostic reasoning",
        steps=steps,
        hypotheses=hypotheses,
        primary_diagnosis=primary,
        requires_human_review=bool(raw_json.get("requires_human_review", False)),
    )


# ---------------------------------------------------------------------------
# LLM Reasoner
# ---------------------------------------------------------------------------


class LLMReasoner:
    """LLM-powered diagnostic reasoning agent.

    This agent sends structured signal feature text + domain knowledge to
    an LLM for deep multi-step reasoning.  It extends the rule-based
    EngineeringReasoner: nominal frames use the fast rule engine, anomalous
    frames get LLM reasoning.

    Parameters
    ----------
    config : LLMConfig
        API connection parameters.
    knowledge : KnowledgeBase or None
        Domain knowledge base instance.
    rule_fallback : bool
        If True, fall back to rule-based reasoner when LLM is unavailable.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        knowledge: Optional[KnowledgeBase] = None,
        rule_fallback: bool = True,
    ) -> None:
        self.config = config or LLMConfig()
        self.knowledge = knowledge or KnowledgeBase()
        self._rule_fallback = rule_fallback
        self._rule_reasoner = EngineeringReasoner(knowledge=self.knowledge)

        # Token tracking
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_llm_calls: int = 0
        self._total_rule_fallbacks: int = 0

        # Lazy client init
        self._client: Any = None

    # -- public API ----------------------------------------------------------

    def reason(self, record, feature_text: str = "") -> ReasoningResult:
        """Run diagnostic reasoning on a FeatureRecord.

        If the rule engine considers the frame nominal, returns immediately.
        Otherwise, sends the structured text + knowledge to the LLM.

        Parameters
        ----------
        record : FeatureRecord
            Extracted feature record.
        feature_text : str
            Pre-serialised text (if already available), otherwise generated.

        Returns
        -------
        ReasoningResult
        """
        # Stage 1: Fast rule-based pre-screening
        rule_result = self._rule_reasoner.reason(record)

        # If nominal with high confidence, skip LLM
        if (
            rule_result.primary_diagnosis
            and rule_result.primary_diagnosis.name == "nominal_operation"
            and rule_result.primary_diagnosis.confidence > 0.8
        ):
            logger.info("Frame %d: nominal (rule engine), skipping LLM call", record.frame_id)
            return rule_result

        # Stage 2: Build prompt and call LLM
        if not feature_text:
            from signal_pipeline.text_serializer import TextSerializer

            serializer = TextSerializer()
            feature_text = serializer.serialize(record)

        # Auto-detect provider from env vars (DeepSeek vs MiMo vs OpenAI)
        self.config.auto_detect_provider()
        logger.info(
            "LLM config: base_url=%s model=%s key=%s...",
            self.config.base_url,
            self.config.model,
            self.config.resolved_api_key[:10],
        )

        return self._call_llm(record, feature_text, rule_result)

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def total_llm_calls(self) -> int:
        return self._total_llm_calls

    @property
    def total_rule_fallbacks(self) -> int:
        return self._total_rule_fallbacks

    def stats(self) -> str:
        return (
            f"LLM Reasoner: {self._total_llm_calls} LLM calls, "
            f"{self._total_rule_fallbacks} rule fallbacks, "
            f"{self.total_tokens:,} tokens ("
            f"{self._total_prompt_tokens:,} prompt + "
            f"{self._total_completion_tokens:,} completion)"
        )

    # -- internals -----------------------------------------------------------

    def _get_client(self):
        """Lazily initialise the OpenAI client."""
        if self._client is not None:
            return self._client

        if not _HAS_OPENAI:
            raise ImportError(
                "openai package is required for LLM reasoning. "
                "Install with: pip install openai"
            )

        api_key = self.config.resolved_api_key
        if not api_key:
            raise ValueError(
                "No API key configured. Set MIMO_API_KEY, DEEPSEEK_API_KEY, "
                "or OPENAI_API_KEY environment variable, or pass api_key to LLMConfig."
            )

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
        )
        return self._client

    def _call_llm(
        self,
        record,
        feature_text: str,
        rule_result: ReasoningResult,
    ) -> ReasoningResult:
        """Call the LLM API and parse the response.

        Falls back to rule_result on any failure if rule_fallback is enabled.
        """
        t_start = time.perf_counter()

        # Check if LLM is available
        api_key = self.config.resolved_api_key
        if not api_key or not _HAS_OPENAI:
            logger.warning("LLM not available (no API key or openai not installed), "
                           "using rule-based fallback")
            self._total_rule_fallbacks += 1
            return rule_result

        try:
            client = self._get_client()
        except Exception as exc:
            logger.warning("Failed to init LLM client: %s", exc)
            if self._rule_fallback:
                self._total_rule_fallbacks += 1
                return rule_result
            raise

        # Build prompt
        knowledge_ctx = self.knowledge.full_context()
        user_prompt = _build_user_prompt(
            feature_text, knowledge_ctx, record.metadata
        )

        # Truncate if too long (conservative: ~100k char ≈ 25k tokens)
        max_prompt_chars = 100_000
        if len(user_prompt) > max_prompt_chars:
            user_prompt = user_prompt[:max_prompt_chars] + "\n\n[TRUNCATED]"
            logger.warning("Prompt truncated to %d chars", max_prompt_chars)

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            logger.error("LLM API call failed: %s", exc)
            if self._rule_fallback:
                self._total_rule_fallbacks += 1
                return rule_result
            raise

        elapsed_s = time.perf_counter() - t_start

        # Track token usage
        usage = response.usage
        if usage:
            self._total_prompt_tokens += usage.prompt_tokens or 0
            self._total_completion_tokens += usage.completion_tokens or 0
            logger.info(
                "LLM call: %d prompt + %d completion tokens in %.1fs",
                usage.prompt_tokens or 0,
                usage.completion_tokens or 0,
                elapsed_s,
            )
        self._total_llm_calls += 1

        # Parse response
        raw_text = response.choices[0].message.content or ""
        try:
            parsed = _extract_json(raw_text)
            result = _parse_llm_response(parsed)
            result.frame_id = record.frame_id
            return result
        except Exception as exc:
            logger.error("Failed to parse LLM response: %s\nRaw: %s", exc, raw_text[:500])
            if self._rule_fallback:
                self._total_rule_fallbacks += 1
                return rule_result
            raise


# ---------------------------------------------------------------------------
# Smoke test (requires API key)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from signal_pipeline.feature_extractor import ExtractionConfig, FeatureExtractor
    from signal_pipeline.fmcw_simulator import (
        FMCWSimulator,
        MultipathConfig,
        SimulationConfig,
        Target,
    )

    print("=" * 60)
    print("  LLM Reasoner Smoke Test")
    print("=" * 60)

    # Simulate a biosignal scenario
    scfg = SimulationConfig(num_chirps=64)
    targets = [
        Target(
            range_m=1.5,
            velocity_mps=0.0,
            rcs_dbsm=5.0,
            micro_doppler={"frequency_hz": 1.2, "amplitude_mm": 0.5},
        ),
    ]
    mp = MultipathConfig(enabled=True, path_delay_s=2e-9)
    sim = FMCWSimulator(scfg, targets, multipath=mp, seed=42)
    sig = sim.generate()

    extractor = FeatureExtractor(ExtractionConfig())
    record = extractor.process(
        sig,
        frame_id=1,
        metadata={
            "scenario": "biosignal_monitoring",
            "subject": "human_respiration",
            "room": "clinical_room",
            "temp_c": 36.5,
        },
    )

    api_key = os.environ.get("MIMO_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        print("\n⚠️  No API key set. Running rule-engine fallback only.")
        print("   Set MIMO_API_KEY or OPENAI_API_KEY to test LLM reasoning.\n")

    reasoner = LLMReasoner(
        config=LLMConfig(
            api_key=api_key,
            base_url="https://api.xiaomimimo.com/v1",
            model="mimo-v2.5",
        ),
        rule_fallback=True,
    )

    result = reasoner.reason(record)
    print(f"\nPrimary diagnosis: {result.primary_diagnosis.name if result.primary_diagnosis else 'NONE'}")
    print(f"Confidence: {result.primary_diagnosis.confidence if result.primary_diagnosis else 0}")
    print(f"Steps: {len(result.steps)}")
    print(f"Hypotheses: {len(result.hypotheses)}")
    print(f"Human review: {result.requires_human_review}")
    print(f"\n{reasoner.stats()}")
