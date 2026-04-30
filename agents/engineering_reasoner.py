"""
Engineering Reasoning Agent
===========================
Multi-step diagnostic reasoning engine for FMCW radar signal anomalies.

This agent implements the core logic flow described in the project
architecture:

  1. Anomaly Classification — parse structured text, classify anomaly type
  2. Physical Modelling — apply domain knowledge to quantify the anomaly
  3. Interference Exclusion — rule out environmental confounders
  4. Root-Cause Synthesis — produce ranked hypotheses with confidence scores

The reasoning is **deterministic + rule-based** with an explicit knowledge
base, designed to be transparent, auditable, and suitable for integration
with LLM-based reasoning as a structured prompt context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from signal_pipeline.feature_extractor import (
    DopplerAnomaly,
    FeatureRecord,
    RangeBinAnomaly,
    TargetEstimate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning data structures
# ---------------------------------------------------------------------------

ANOMALY_CLASS_PHASE_JUMP = "phase_jump"
ANOMALY_CLASS_MECHANICAL = "mechanical_vibration"
ANOMALY_CLASS_PERIODIC = "periodic_displacement"
ANOMALY_CLASS_MICRO_DOPPLER = "micro_doppler"
ANOMALY_CLASS_INTERFERENCE = "narrowband_interference"
ANOMALY_CLASS_BROADBAND = "broadband_noise"


@dataclass
class StepResult:
    """Output of a single reasoning step."""
    step_name: str
    findings: List[str] = field(default_factory=list)
    eliminated_hypotheses: List[str] = field(default_factory=list)
    active_hypotheses: List[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    """A candidate root-cause hypothesis."""
    name: str
    confidence: float               # 0–1
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class ReasoningResult:
    """Complete output of the multi-step reasoning chain."""
    frame_id: int
    input_summary: str
    steps: List[StepResult] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    primary_diagnosis: Optional[Hypothesis] = None
    requires_human_review: bool = False


# ---------------------------------------------------------------------------
# Knowledge base loader
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """Load and index the domain knowledge files used during reasoning.

    Knowledge files are markdown documents covering:
    - radar_physics.md — FMCW fundamentals, formulas
    - multipath_model.md — multipath discrimination criteria
    - vibration_vs_biosignal.md — mechanical vs. biological micro-Doppler
    - diagnostic_decision_tree.md — systematic reasoning flow
    """

    _DEFAULT_DIR = Path(__file__).resolve().parent.parent / "knowledge"

    def __init__(self, knowledge_dir: Optional[Path] = None) -> None:
        self._dir = Path(knowledge_dir) if knowledge_dir else self._DEFAULT_DIR
        self._cache: Dict[str, str] = {}

    @property
    def radar_physics(self) -> str:
        return self._load("radar_physics.md")

    @property
    def multipath_model(self) -> str:
        return self._load("multipath_model.md")

    @property
    def vibration_vs_biosignal(self) -> str:
        return self._load("vibration_vs_biosignal.md")

    @property
    def diagnostic_tree(self) -> str:
        return self._load("diagnostic_decision_tree.md")

    def full_context(self) -> str:
        """Return all knowledge files concatenated as prompt context."""
        sections = []
        for name in [
            "radar_physics.md",
            "multipath_model.md",
            "vibration_vs_biosignal.md",
            "diagnostic_decision_tree.md",
        ]:
            content = self._load(name)
            sections.append(content)
        return "\n\n---\n\n".join(sections)

    def _load(self, filename: str) -> str:
        if filename not in self._cache:
            path = self._dir / filename
            if path.exists():
                self._cache[filename] = path.read_text(encoding="utf-8")
            else:
                logger.warning("Knowledge file not found: %s", path)
                self._cache[filename] = ""
        return self._cache[filename]


# ---------------------------------------------------------------------------
# Physical constants & helpers
# ---------------------------------------------------------------------------

C = 299_792_458.0
F_CENTER_DEFAULT = 79e9          # 79 GHz (mid-band)
LAMBDA_DEFAULT = C / F_CENTER_DEFAULT  # ≈ 3.8 mm


def _phase_to_displacement(phase_jump_deg: float, wavelength: float) -> float:
    """Convert phase jump (degrees) to physical displacement (metres).

    Δx = Δφ · λ / (4π),  with Δφ in radians.
    """
    phase_rad = abs(phase_jump_deg) * 3.141592653589793 / 180.0
    return phase_rad * wavelength / (4 * 3.141592653589793)


def _estimate_wavelength(metadata: Dict) -> float:
    """Try to extract centre frequency from metadata; fall back to default."""
    freq_hz = metadata.get("center_freq_hz", F_CENTER_DEFAULT)
    return C / freq_hz


# ---------------------------------------------------------------------------
# Reasoning steps
# ---------------------------------------------------------------------------


class EngineeringReasoner:
    """Execute the 4-phase diagnostic reasoning chain on a FeatureRecord.

    Parameters
    ----------
    knowledge : KnowledgeBase or None
        Domain knowledge base; auto-loaded from ``knowledge/`` if omitted.
    """

    def __init__(self, knowledge: Optional[KnowledgeBase] = None) -> None:
        self.kb = knowledge or KnowledgeBase()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def reason(self, record: FeatureRecord) -> ReasoningResult:
        """Run the full reasoning chain on one frame's feature record.

        Returns a structured ReasoningResult with ranked hypotheses.
        """
        result = ReasoningResult(
            frame_id=record.frame_id,
            input_summary=self._summarise_input(record),
        )

        # Phase 1: Anomaly classification
        step1 = self._classify_anomalies(record)
        result.steps.append(step1)

        # Phase 2: Physical modelling
        step2 = self._physical_modelling(record, step1)
        result.steps.append(step2)

        # Phase 3: Interference exclusion
        step3 = self._exclude_interference(record, step2)
        result.steps.append(step3)

        # Phase 4: Synthesise hypotheses
        hypotheses = self._synthesise(record, step3)
        result.hypotheses = hypotheses

        # Build synthesis step
        synthesis_findings = [
            f"Ranked {len(hypotheses)} hypothesis(s): "
            + ", ".join(f"{h.name}({h.confidence:.0%})" for h in hypotheses)
        ]
        if hypotheses:
            synthesis_findings.append(
                f"Primary diagnosis: {hypotheses[0].name} "
                f"(confidence={hypotheses[0].confidence:.0%})"
            )
        step4 = StepResult(
            step_name="Root-Cause Synthesis",
            findings=synthesis_findings,
            active_hypotheses=[h.name for h in hypotheses],
            eliminated_hypotheses=[
                h for h in step3.active_hypotheses
                if h not in [hyp.name for hyp in hypotheses]
            ],
        )
        result.steps.append(step4)

        if hypotheses:
            result.primary_diagnosis = hypotheses[0]
        result.requires_human_review = (
            len(hypotheses) == 0
            or (hypotheses[0].confidence < 0.6 if hypotheses else True)
        )

        return result

    # ------------------------------------------------------------------
    # Step 1: Anomaly Classification
    # ------------------------------------------------------------------

    def _classify_anomalies(self, rec: FeatureRecord) -> StepResult:
        step = StepResult(step_name="Anomaly Classification")

        for a in rec.range_anomalies:
            if a.anomaly_type in ("phase_jump", "possible_phase_jump"):
                step.findings.append(
                    f"Range bin {a.range_bin} ({a.range_m:.2f} m): "
                    f"phase discontinuity {a.phase_jump_deg:.1f}°, "
                    f"SNR={a.snr_db:.1f} dB, confidence={a.confidence:.2f}"
                )
                step.active_hypotheses.append("multipath_reflection")
                if a.phase_jump_deg > 10:
                    step.active_hypotheses.append("mechanical_vibration")

            elif a.anomaly_type == "mechanical_vibration":
                step.findings.append(
                    f"Range bin {a.range_bin}: mechanical vibration pattern "
                    f"detected (phase jump {a.phase_jump_deg:.1f}°)"
                )
                step.active_hypotheses.append("mechanical_vibration")

            elif a.anomaly_type == "periodic_displacement":
                step.findings.append(
                    f"Range bin {a.range_bin}: periodic displacement pattern "
                    f"detected"
                )
                step.active_hypotheses.extend(["biosignal_micro_doppler", "mechanical_vibration"])

        for d in rec.doppler_anomalies:
            step.findings.append(
                f"Doppler {d.freq_hz:.2f} Hz: {d.pattern}, "
                f"deviation={d.amplitude_db:.1f} dB"
            )
            if d.pattern == "possible_micro_doppler":
                if "biosignal_micro_doppler" not in step.active_hypotheses:
                    step.active_hypotheses.append("biosignal_micro_doppler")
            elif d.pattern == "narrowband_interference":
                step.active_hypotheses.append("external_interference")
            elif d.pattern == "broadband_noise":
                step.active_hypotheses.append("receiver_noise_excursion")

        if not step.findings:
            step.findings.append("No anomalies detected.")
            step.active_hypotheses.append("nominal_operation")

        # De-duplicate
        step.active_hypotheses = list(dict.fromkeys(step.active_hypotheses))
        return step

    # ------------------------------------------------------------------
    # Step 2: Physical Modelling
    # ------------------------------------------------------------------

    def _physical_modelling(self, rec: FeatureRecord, prev: StepResult) -> StepResult:
        step = StepResult(step_name="Physical Modelling")
        wavelength = _estimate_wavelength(rec.metadata)

        # Phase-jump → displacement estimates
        for a in rec.range_anomalies:
            if a.anomaly_type in ("phase_jump", "possible_phase_jump"):
                disp_mm = _phase_to_displacement(a.phase_jump_deg, wavelength) * 1000
                step.findings.append(
                    f"Phase jump {a.phase_jump_deg:.1f}° at bin {a.range_bin} "
                    f"→ estimated displacement = {disp_mm:.3f} mm"
                    f"{' (sub-mm, consistent with vibration)' if disp_mm < 0.5 else ''}"
                    f"{' (> 1 mm, could be biosignal)' if disp_mm > 1.0 else ''}"
                )

                # Multipath plausibility check
                # If displacement < 0.5 mm → unlikely to be multipath (requires
                # surface movement), more likely electrical/phase noise or micro-motion
                if disp_mm < 0.1 and a.snr_db > 15:
                    step.findings.append(
                        "Very small displacement at high SNR → possible ADC "
                        "timing jitter or phase noise, not physical motion."
                    )
                    if "multipath_reflection" in prev.active_hypotheses:
                        step.eliminated_hypotheses.append("multipath_reflection")

        # Micro-Doppler analysis
        for d in rec.doppler_anomalies:
            if d.pattern == "possible_micro_doppler":
                if d.freq_hz < 0.5:
                    step.findings.append(
                        f"Frequency {d.freq_hz:.2f} Hz falls in respiration band "
                        f"(0.1–0.5 Hz)."
                    )
                elif d.freq_hz < 3.0:
                    step.findings.append(
                        f"Frequency {d.freq_hz:.2f} Hz in heartbeat band "
                        f"(0.8–3 Hz). Check harmonic content."
                    )
                else:
                    step.findings.append(
                        f"Frequency {d.freq_hz:.2f} Hz outside typical biosignal "
                        f"range — consider mechanical source."
                    )
                    if "biosignal_micro_doppler" in prev.active_hypotheses:
                        step.eliminated_hypotheses.append("biosignal_micro_doppler")

        # Carry forward active hypotheses
        step.active_hypotheses = [
            h for h in prev.active_hypotheses
            if h not in step.eliminated_hypotheses
        ]
        if not step.active_hypotheses:
            step.active_hypotheses.append("unexplained_anomaly")

        return step

    # ------------------------------------------------------------------
    # Step 3: Interference Exclusion
    # ------------------------------------------------------------------

    def _exclude_interference(self, rec: FeatureRecord, prev: StepResult) -> StepResult:
        step = StepResult(step_name="Interference Exclusion")

        # Environmental checks from metadata
        meta = rec.metadata
        temp_c = meta.get("temp_c")
        room = meta.get("room", "")

        if temp_c is not None:
            if temp_c > 40:
                step.findings.append(
                    f"High ambient temperature ({temp_c}°C) may elevate "
                    f"thermal noise floor. Monitor receiver gain stability."
                )
            elif temp_c < 0:
                step.findings.append(
                    f"Low ambient temperature ({temp_c}°C) — check for "
                    f"condensation and LO frequency drift."
                )

        if "lab" in room.lower() or "industrial" in room.lower():
            step.findings.append(
                "Industrial/lab environment — potential sources of mechanical "
                "vibration (HVAC, pumps, nearby machinery)."
            )
            if (
                "mechanical_vibration" not in prev.active_hypotheses
                and "biosignal_micro_doppler" in prev.active_hypotheses
            ):
                step.findings.append(
                    "WARNING: biosignal hypothesis in industrial environment "
                    "is plausible only if sensor is isolated from machinery."
                )

        # Multi-target interaction check
        if len(rec.targets) >= 2:
            step.findings.append(
                f"Multiple targets ({len(rec.targets)}) detected. "
                f"Verify that anomalies are not inter-target interference."
            )

        # Narrowband interference check
        for d in rec.doppler_anomalies:
            if d.pattern == "narrowband_interference":
                freq = d.freq_hz
                if 49.5 < freq < 50.5 or 59.5 < freq < 60.5:
                    step.findings.append(
                        f"Interference at {freq:.1f} Hz matches AC mains "
                        f"frequency — likely power supply coupling."
                    )
                    step.eliminated_hypotheses.append("biosignal_micro_doppler")
                    step.eliminated_hypotheses.append("mechanical_vibration")

        step.active_hypotheses = [
            h for h in prev.active_hypotheses
            if h not in step.eliminated_hypotheses
        ]
        if not step.active_hypotheses:
            step.active_hypotheses.append("unexplained_anomaly")
        return step

    # ------------------------------------------------------------------
    # Step 4: Synthesis
    # ------------------------------------------------------------------

    def _synthesise(self, rec: FeatureRecord, prev: StepResult) -> List[Hypothesis]:
        """Rank active hypotheses and assign confidence scores."""

        # Scoring rubric — base confidence + modifiers
        rubric = {
            "multipath_reflection": {
                "base": 0.70,
                "evidence": [
                    "Secondary peak at greater range with matching velocity.",
                    "Phase correlation with direct target > 0.9.",
                ],
                "recommendation": (
                    "Subtract identified multipath ghost from range profile "
                    "or reposition sensor to reduce surface reflections."
                ),
            },
            "mechanical_vibration": {
                "base": 0.65,
                "evidence": [
                    "Periodic phase modulation with frequency > 5 Hz.",
                    "High harmonic content (suggest nonlinear source).",
                    "Estimated displacement < 1 mm.",
                ],
                "recommendation": (
                    "Cross-reference vibration frequency with known machinery "
                    "in the environment. Consider vibration isolation for sensor."
                ),
            },
            "biosignal_micro_doppler": {
                "base": 0.55,
                "evidence": [
                    "Periodic micro-Doppler in 0.1–3 Hz band.",
                    "Displacement consistent with respiration or heartbeat.",
                    "Low harmonic content.",
                ],
                "recommendation": (
                    "If healthcare monitoring: log vital signs trend. "
                    "Otherwise: verify no human/animal presence in sensor FOV."
                ),
            },
            "external_interference": {
                "base": 0.60,
                "evidence": [
                    "Narrowband peak in Doppler spectrum not matching any target.",
                    "Frequency matches known interference source (e.g. AC mains).",
                ],
                "recommendation": (
                    "Apply notch filter at interference frequency or "
                    "switch to alternate frequency band."
                ),
            },
            "receiver_noise_excursion": {
                "base": 0.50,
                "evidence": [
                    "Broadband noise floor elevation.",
                    "No structured pattern in anomaly.",
                ],
                "recommendation": (
                    "Check receiver hardware: temperature, power supply, ADC clock. "
                    "Compare with baseline noise from previous frames."
                ),
            },
            "nominal_operation": {
                "base": 0.95,
                "evidence": [
                    "No anomalies detected.",
                    "All targets within expected parameters.",
                ],
                "recommendation": "Continue routine monitoring.",
            },
            "unexplained_anomaly": {
                "base": 0.30,
                "evidence": [
                    "Anomaly does not match any known pattern.",
                ],
                "recommendation": (
                    "Flag for human review. Collect additional frames for "
                    "temporal analysis."
                ),
            },
        }

        hypotheses: List[Hypothesis] = []
        for name in prev.active_hypotheses:
            info = rubric.get(name, rubric["unexplained_anomaly"])
            confidence = info["base"]

            # Adjust confidence based on anomaly quality
            if name == "multipath_reflection":
                if self._has_secondary_peak_at_greater_range(rec):
                    confidence += 0.15
                if rec.metadata.get("multipath_enabled"):
                    confidence += 0.10

            elif name == "mechanical_vibration":
                if self._count_periodic_anomalies(rec) >= 3:
                    confidence += 0.15
                if rec.metadata.get("scenario") == "mechanical_vibration":
                    confidence += 0.20

            elif name == "biosignal_micro_doppler":
                if self._has_low_freq_doppler(rec):
                    confidence += 0.15
                md_targets = [
                    t for t in rec.targets
                    if hasattr(t, "micro_doppler")  # rough check
                ]
                if md_targets or rec.metadata.get("scenario") == "biosignal_monitoring":
                    confidence += 0.20

            elif name == "external_interference":
                if self._has_mains_freq_anomaly(rec):
                    confidence += 0.20

            elif name == "receiver_noise_excursion":
                if len(rec.doppler_anomalies) > 5:
                    confidence += 0.10

            confidence = min(confidence, 0.99)

            hypotheses.append(Hypothesis(
                name=name,
                confidence=round(confidence, 3),
                evidence=info["evidence"],
                recommendation=info["recommendation"],
            ))

        # Sort by confidence descending
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summarise_input(self, rec: FeatureRecord) -> str:
        return (
            f"Frame {rec.frame_id}: {len(rec.targets)} target(s), "
            f"{len(rec.range_anomalies)} range anomaly(s), "
            f"{len(rec.doppler_anomalies)} Doppler anomaly(s)"
        )

    def _has_secondary_peak_at_greater_range(self, rec: FeatureRecord) -> bool:
        if len(rec.targets) < 2:
            return False
        primary_range = rec.targets[0].range_m
        for t in rec.targets[1:]:
            if t.range_m > primary_range and abs(t.velocity_mps - rec.targets[0].velocity_mps) < 0.1:
                return True
        return False

    def _count_periodic_anomalies(self, rec: FeatureRecord) -> int:
        return sum(
            1 for a in rec.range_anomalies
            if a.anomaly_type in ("periodic_displacement", "mechanical_vibration")
        )

    def _has_low_freq_doppler(self, rec: FeatureRecord) -> bool:
        return any(
            d.pattern == "possible_micro_doppler" and d.freq_hz < 3.0
            for d in rec.doppler_anomalies
        )

    def _has_mains_freq_anomaly(self, rec: FeatureRecord) -> bool:
        for d in rec.doppler_anomalies:
            if d.pattern == "narrowband_interference":
                if 49.5 < d.freq_hz < 50.5 or 59.5 < d.freq_hz < 60.5:
                    return True
        return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from signal_pipeline.feature_extractor import ExtractionConfig, FeatureExtractor
    from signal_pipeline.fmcw_simulator import FMCWSimulator, MultipathConfig, SimulationConfig, Target

    scfg = SimulationConfig(num_chirps=64)
    targets = [
        Target(range_m=3.0, velocity_mps=0.3, rcs_dbsm=12,
               micro_doppler={"frequency_hz": 1.2, "amplitude_mm": 0.5}),
    ]
    mp = MultipathConfig(enabled=True, path_delay_s=2e-9)
    sim = FMCWSimulator(scfg, targets, multipath=mp, seed=123)
    sig = sim.generate()

    extractor = FeatureExtractor(ExtractionConfig())
    rec = extractor.process(sig, frame_id=1, metadata={
        "scenario": "biosignal_monitoring",
        "room": "lab_A",
        "temp_c": 22.5,
        "multipath_enabled": True,
    })

    reasoner = EngineeringReasoner()
    result = reasoner.reason(rec)

    print(f"=== REASONING RESULT (Frame {result.frame_id}) ===")
    for step in result.steps:
        print(f"\n--- {step.step_name} ---")
        for f in step.findings:
            print(f"  · {f}")
        print(f"  Active: {step.active_hypotheses}")
        if step.eliminated_hypotheses:
            print(f"  Eliminated: {step.eliminated_hypotheses}")

    print("\n--- Hypotheses ---")
    for h in result.hypotheses:
        print(f"  [{h.confidence:.3f}] {h.name}")
    print(f"\nPrimary: {result.primary_diagnosis.name if result.primary_diagnosis else 'NONE'}")
    print(f"Needs review: {result.requires_human_review}")
