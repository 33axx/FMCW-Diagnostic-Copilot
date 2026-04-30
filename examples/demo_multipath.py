#!/usr/bin/env python3
"""
Demo: Multipath Interference
============================
Simulates a target with a strong desktop-surface reflection (multipath).
Tests whether the agent correctly discriminates the ghost peak from a real target.

Usage::

    cd fmcw1
    python examples/demo_multipath.py
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
    print("  FMCW Diagnostic Agent — Demo: MULTIPATH INTERFERENCE")
    print("=" * 60)

    # 1. Simulate signal with multipath
    cfg = SimulationConfig(num_chirps=128)
    targets = [
        Target(range_m=4.0, velocity_mps=0.3, rcs_dbsm=12.0),
    ]
    multipath = MultipathConfig(
        enabled=True,
        path_delay_s=2e-9,         # ~0.3 m extra path (desktop reflection)
        attenuation_db=-6.0,
        phase_shift_rad=0.4,
    )
    sim = FMCWSimulator(cfg, targets, multipath=multipath, seed=99)
    if_signal = sim.generate()
    print(f"\n[1] Generated IF signal with multipath impairment")

    # 2. Extract features
    extractor = FeatureExtractor(ExtractionConfig())
    record = extractor.process(if_signal, frame_id=1, metadata={
        "scenario": "multipath",
        "room": "small_chamber",
        "temp_c": 23.0,
        "multipath_enabled": True,
    })
    print(f"[2] Extracted: {len(record.targets)} target(s), "
          f"{len(record.range_anomalies)} range anomaly(s), "
          f"{len(record.doppler_anomalies)} Doppler anomaly(s)")

    # Display targets — we expect to see two (direct + ghost)
    for t in record.targets:
        print(f"    → {t.range_m:.3f} m, {t.velocity_mps:.3f} m/s, SNR={t.snr_db:.1f} dB")

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
    md = reporter.to_markdown(result, Path("reports/multipath_demo.md"))
    print(f"\n[5] Report written to reports/multipath_demo.md")

    print("\n✅ Demo complete.")


if __name__ == "__main__":
    main()
