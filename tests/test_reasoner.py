"""Tests for the engineering reasoning agent."""

import numpy as np
import pytest

from agents.engineering_reasoner import (
    EngineeringReasoner,
    Hypothesis,
    KnowledgeBase,
    ReasoningResult,
)
from signal_pipeline.feature_extractor import (
    DopplerAnomaly,
    ExtractionConfig,
    FeatureExtractor,
    FeatureRecord,
    RangeBinAnomaly,
    TargetEstimate,
)
from signal_pipeline.fmcw_simulator import (
    FMCWSimulator,
    SimulationConfig,
    Target,
)


class TestKnowledgeBase:
    def test_loads_radar_physics(self):
        kb = KnowledgeBase()
        content = kb.radar_physics
        assert "FMCW" in content or "fmcw" in content.lower()
        assert len(content) > 100

    def test_loads_multipath_model(self):
        kb = KnowledgeBase()
        content = kb.multipath_model
        assert len(content) > 100

    def test_loads_vibration_vs_biosignal(self):
        kb = KnowledgeBase()
        content = kb.vibration_vs_biosignal
        assert len(content) > 100

    def test_loads_diagnostic_tree(self):
        kb = KnowledgeBase()
        content = kb.diagnostic_tree
        assert len(content) > 100

    def test_full_context(self):
        kb = KnowledgeBase()
        ctx = kb.full_context()
        assert len(ctx) > 400


class TestEngineeringReasoner:
    @pytest.fixture
    def reasoner(self):
        return EngineeringReasoner()

    @pytest.fixture
    def normal_record(self):
        """A nominal frame with no significant anomalies."""
        cfg = SimulationConfig(num_chirps=32)
        targets = [Target(range_m=5.0, velocity_mps=0.5, rcs_dbsm=15)]
        sim = FMCWSimulator(cfg, targets, seed=42)
        sig = sim.generate()
        extractor = FeatureExtractor(ExtractionConfig())
        return extractor.process(sig, frame_id=1, metadata={
            "scenario": "normal",
            "room": "lab",
            "temp_c": 22.0,
        })

    def test_reason_returns_result(self, reasoner, normal_record):
        result = reasoner.reason(normal_record)
        assert isinstance(result, ReasoningResult)
        assert result.frame_id == 1

    def test_reason_has_steps(self, reasoner, normal_record):
        result = reasoner.reason(normal_record)
        assert len(result.steps) == 4
        step_names = [s.step_name for s in result.steps]
        assert "Anomaly Classification" in step_names
        assert "Physical Modelling" in step_names
        assert "Interference Exclusion" in step_names

    def test_reason_has_hypotheses(self, reasoner, normal_record):
        result = reasoner.reason(normal_record)
        assert len(result.hypotheses) > 0

    def test_hypotheses_sorted_by_confidence(self, reasoner, normal_record):
        result = reasoner.reason(normal_record)
        confidences = [h.confidence for h in result.hypotheses]
        assert confidences == sorted(confidences, reverse=True)

    def test_confidence_in_range(self, reasoner, normal_record):
        result = reasoner.reason(normal_record)
        for h in result.hypotheses:
            assert 0.0 <= h.confidence <= 1.0

    def test_primary_diagnosis_set(self, reasoner, normal_record):
        result = reasoner.reason(normal_record)
        assert result.primary_diagnosis is not None
        assert isinstance(result.primary_diagnosis, Hypothesis)

    def test_each_hypothesis_has_evidence(self, reasoner, normal_record):
        result = reasoner.reason(normal_record)
        for h in result.hypotheses:
            assert len(h.evidence) > 0

    def test_each_hypothesis_has_recommendation(self, reasoner, normal_record):
        result = reasoner.reason(normal_record)
        for h in result.hypotheses:
            assert len(h.recommendation) > 0


class TestPhaseToDisplacement:
    """Verify the phase-to-displacement conversion helper."""
    def test_zero_phase(self):
        from agents.engineering_reasoner import _phase_to_displacement
        assert _phase_to_displacement(0.0, 0.0038) == 0.0

    def test_known_conversion(self):
        # At 79 GHz, λ ≈ 3.8 mm.  1 mm displacement → phase shift
        # Δφ = 4π·Δx / λ = 4π·0.001 / 0.003797 ≈ 3.31 rad ≈ 189.6°
        # So 92° → Δx = 92°·λ/(4π·180/π) = (92·π/180)·λ/(4π) = 92·λ/720
        from agents.engineering_reasoner import _phase_to_displacement, LAMBDA_DEFAULT
        disp = _phase_to_displacement(92.0, LAMBDA_DEFAULT)
        # LAMBDA_DEFAULT = C / 79e9 ≈ 0.003797 m
        # 92° * 0.003797 / (4π * 180/π) = 92 * 0.003797 / 720 ≈ 0.000485 m
        expected_mm = 92 * LAMBDA_DEFAULT * 1000 / 720
        assert abs(disp * 1000 - expected_mm) < 0.001, (
            f"Expected ~{expected_mm:.3f} mm, got {disp*1000:.3f} mm"
        )
