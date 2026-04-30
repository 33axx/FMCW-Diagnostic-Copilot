#!/usr/bin/env python3
"""
Demo: Biosignal Micro-Doppler
=============================
Simulates a target with respiratory micro-Doppler modulation (1.2 Hz, 0.5 mm).
Tests the agent's ability to detect and classify biological vital signs.

Usage::

    cd fmcw1
    python examples/demo_biosignal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_pipeline.fmcw_simulator import (
    FMCWSimulator,
    SimulationConfig,
    Target,
)
from signal_pipeline.feature_extractor import ExtractionConfig, FeatureExtractor
from agents.engineering_reasoner import EngineeringReasoner
from agents.expert_reporter import ExpertReporter


def main():
    print("=" * 60)
    print("  FMCW Diagnostic Agent — Demo: BIOSIGNAL MICRO-DOPPLER")
    print("=" * 60)

    # 1. Simulate signal with respiration micro-Doppler
    cfg = SimulationConfig(num_chirps=128)
    targets = [
        Target(
            range_m=1.5,
            velocity_mps=0.0,
            rcs_dbsm=5.0,
            micro_doppler={"frequency_hz": 1.2, "amplitude_mm": 0.5},
        ),
    ]
    sim = FMCWSimulator(cfg, targets, seed=42)
    if_signal = sim.generate()
    print(f"\n[1] Generated IF signal with 1.2 Hz / 0.5 mm micro-Doppler")

    # 2. Extract features
    extractor = FeatureExtractor(ExtractionConfig())
    record = extractor.process(if_signal, frame_id=1, metadata={
        "scenario": "biosignal_monitoring",
        "subject": "human_respiration",
        "room": "clinical_room",
        "temp_c": 36.5,
    })
    print(f"[2] Extracted: {len(record.targets)} target(s), "
          f"{len(record.range_anomalies)} range anomaly(s), "
          f"{len(record.doppler_anomalies)} Doppler anomaly(s)")

    for d in record.doppler_anomalies[:5]:
        print(f"    · {d.freq_hz:.3f} Hz: {d.pattern}, +{d.amplitude_db:.1f} dB")

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
    md = reporter.to_markdown(result, Path("reports/biosignal_demo.md"))
    print(f"\n[5] Report written to reports/biosignal_demo.md")

    print("\n✅ Demo complete.")


if __name__ == "__main__":
    main()
