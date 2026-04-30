#!/usr/bin/env python3
"""
Demo: Normal Operation
======================
Simulates a clean FMCW scene with two targets, no impairments.
Verifies that the diagnostic agent correctly identifies nominal operation.

Usage::

    cd fmcw1
    python examples/demo_normal.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_pipeline.fmcw_simulator import (
    FMCWSimulator,
    SimulationConfig,
    Target,
)
from signal_pipeline.feature_extractor import ExtractionConfig, FeatureExtractor
from signal_pipeline.text_serializer import TextSerializer
from agents.engineering_reasoner import EngineeringReasoner
from agents.expert_reporter import ExpertReporter


def main():
    print("=" * 60)
    print("  FMCW Diagnostic Agent — Demo: NORMAL OPERATION")
    print("=" * 60)

    # 1. Simulate clean signal
    cfg = SimulationConfig(num_chirps=64)
    targets = [
        Target(range_m=5.0, velocity_mps=0.8, rcs_dbsm=15.0),
        Target(range_m=12.0, velocity_mps=-2.0, rcs_dbsm=10.0),
    ]
    sim = FMCWSimulator(cfg, targets, seed=42)
    if_signal = sim.generate()
    print(f"\n[1] Generated IF signal: shape={if_signal.shape}")

    # 2. Extract features
    extractor = FeatureExtractor(ExtractionConfig())
    record = extractor.process(if_signal, frame_id=1, metadata={
        "scenario": "normal",
        "room": "indoor_lab",
        "temp_c": 22.0,
    })
    print(f"[2] Extracted: {len(record.targets)} target(s), "
          f"{len(record.range_anomalies)} range anomaly(s), "
          f"{len(record.doppler_anomalies)} Doppler anomaly(s)")

    # 3. Serialize to text
    serializer = TextSerializer()
    text = serializer.serialize(record)
    print(f"\n[3] Serialized text ({len(text)} chars):")
    print(text[:500] + "..." if len(text) > 500 else text)

    # 4. Run engineering reasoning
    reasoner = EngineeringReasoner()
    result = reasoner.reason(record)
    print(f"\n[4] Reasoning complete")
    for h in result.hypotheses:
        print(f"    [{h.confidence:.3f}] {h.name}")
    print(f"    Primary: {result.primary_diagnosis.name if result.primary_diagnosis else 'NONE'}")

    # 5. Generate report
    reporter = ExpertReporter()
    md_report = reporter.to_markdown(result)
    print(f"\n[5] Markdown report ({len(md_report)} chars)")
    print(md_report[:800])

    print("\n✅ Demo complete.")


if __name__ == "__main__":
    main()
