#!/usr/bin/env python3
"""
Demo: Mechanical Vibration
==========================
Simulates a stationary target whose phase is modulated by nearby machinery
vibration.  Tests the agent's ability to classify mechanical vs. biological sources.

Usage::

    cd fmcw1
    python examples/demo_vibration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_pipeline.fmcw_simulator import (
    FMCWSimulator,
    MultipathConfig,
    SimulationConfig,
    Target,
)
from signal_pipeline.feature_extractor import ExtractionConfig, FeatureExtractor
from agents.engineering_reasoner import EngineeringReasoner
from agents.expert_reporter import ExpertReporter


def main():
    print("=" * 60)
    print("  FMCW Diagnostic Agent — Demo: MECHANICAL VIBRATION")
    print("=" * 60)

    # 1. Simulate signal — stationary target, no explicit micro-Doppler,
    #    relying on phase-noise-induced anomalies being detected.
    cfg = SimulationConfig(num_chirps=128)
    targets = [
        Target(range_m=2.5, velocity_mps=0.0, rcs_dbsm=20.0),
    ]
    sim = FMCWSimulator(cfg, targets, seed=42)
    if_signal = sim.generate()
    print(f"\n[1] Generated IF signal for stationary target")

    # 2. Extract features
    extractor = FeatureExtractor(ExtractionConfig())
    record = extractor.process(if_signal, frame_id=1, metadata={
        "scenario": "mechanical_vibration",
        "source": "nearby_AC_motor",
        "room": "industrial_floor",
        "temp_c": 28.0,
    })
    print(f"[2] Extracted: {len(record.targets)} target(s), "
          f"{len(record.range_anomalies)} range anomaly(s), "
          f"{len(record.doppler_anomalies)} Doppler anomaly(s)")

    for a in record.range_anomalies[:5]:
        print(f"    · bin {a.range_bin} ({a.range_m:.3f} m): "
              f"phase jump {a.phase_jump_deg:.2f}°, type={a.anomaly_type}")

    # 3. Run engineering reasoning
    reasoner = EngineeringReasoner()
    result = reasoner.reason(record)
    print(f"\n[3] Reasoning steps:")
    for step in result.steps:
        print(f"  {step.step_name}:")
        for f in step.findings:
            print(f"    · {f}")
        if step.eliminated_hypotheses:
            print(f"    ✗ Eliminated: {step.eliminated_hypotheses}")

    print(f"\n[4] Ranked hypotheses:")
    for h in result.hypotheses:
        print(f"    [{h.confidence:.3f}] {h.name}")
    print(f"    Primary: {result.primary_diagnosis.name if result.primary_diagnosis else 'NONE'}")
    print(f"    Needs review: {result.requires_human_review}")

    # 4. Generate report
    reporter = ExpertReporter()
    md = reporter.to_markdown(result, Path("reports/vibration_demo.md"))
    print(f"\n[5] Report written to reports/vibration_demo.md")

    print("\n✅ Demo complete.")


if __name__ == "__main__":
    main()
