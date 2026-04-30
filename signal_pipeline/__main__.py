"""
Signal Pipeline — CLI entry point
==================================
Run a complete signal simulation → feature extraction → text serialization
pipeline and print the structured text output.

Usage::

    python -m signal_pipeline                  # default config
    python -m signal_pipeline --scenario multipath
    python -m signal_pipeline --scenario biosignal --output result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .feature_extractor import ExtractionConfig, FeatureExtractor
from .fmcw_simulator import (
    FMCWSimulator,
    MultipathConfig,
    SimulationConfig,
    Target,
)
from .text_serializer import TextSerializer


# ---------------------------------------------------------------------------
# Scenario presets
# ---------------------------------------------------------------------------


def _build_scenario(name: str) -> Dict:
    """Return a dict with keys: sim_config, targets, multipath, metadata."""
    base_cfg = SimulationConfig(num_chirps=128)

    scenarios: Dict[str, Dict] = {
        "normal": {
            "sim_config": base_cfg,
            "targets": [
                Target(range_m=3.0, velocity_mps=0.5, rcs_dbsm=15.0),
                Target(range_m=8.0, velocity_mps=-1.2, rcs_dbsm=10.0),
            ],
            "multipath": MultipathConfig(enabled=False),
            "metadata": {"scenario": "normal", "room": "indoor_lab", "temp_c": 22.0},
        },
        "multipath": {
            "sim_config": base_cfg,
            "targets": [
                Target(range_m=5.0, velocity_mps=0.3, rcs_dbsm=12.0),
            ],
            "multipath": MultipathConfig(
                enabled=True,
                path_delay_s=2e-9,     # ~0.3 m extra path — desktop reflection
                attenuation_db=-4.0,
                phase_shift_rad=0.4,
            ),
            "metadata": {"scenario": "multipath", "room": "small_chamber", "temp_c": 23.0},
        },
        "vibration": {
            "sim_config": base_cfg,
            "targets": [
                Target(range_m=2.0, velocity_mps=0.0, rcs_dbsm=20.0),
            ],
            "multipath": MultipathConfig(enabled=False),
            "metadata": {
                "scenario": "mechanical_vibration",
                "source": "nearby_AC_motor",
                "temp_c": 25.0,
            },
        },
        "biosignal": {
            "sim_config": base_cfg,
            "targets": [
                Target(
                    range_m=1.5,
                    velocity_mps=0.0,
                    rcs_dbsm=5.0,
                    micro_doppler={"frequency_hz": 1.2, "amplitude_mm": 0.5},
                ),
            ],
            "multipath": MultipathConfig(enabled=False),
            "metadata": {
                "scenario": "biosignal_monitoring",
                "subject": "human_respiration",
                "temp_c": 36.5,
            },
        },
    }
    return scenarios.get(name, scenarios["normal"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FMCW Signal Pipeline — simulate, extract, serialize"
    )
    parser.add_argument(
        "--scenario",
        choices=["normal", "multipath", "vibration", "biosignal"],
        default="normal",
        help="Pre-configured simulation scenario.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write serialized text to file (default: print to stdout).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also export structured JSON to this path.",
    )
    parser.add_argument(
        "--raw-stats",
        action="store_true",
        help="Include raw numeric statistics in text output.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args(argv)

    scenario = _build_scenario(args.scenario)

    # 1. Simulate
    sim = FMCWSimulator(
        config=scenario["sim_config"],
        targets=scenario["targets"],
        multipath=scenario["multipath"],
        seed=args.seed,
    )
    if_signal = sim.generate()

    # 2. Extract features
    extractor = FeatureExtractor(ExtractionConfig())
    record = extractor.process(
        if_signal,
        frame_id=1,
        metadata=scenario["metadata"],
    )

    # 3. Serialize
    serializer = TextSerializer(include_raw_statistics=args.raw_stats)
    text_output = serializer.serialize(record)

    # 4. Output
    if args.output:
        args.output.write_text(text_output, encoding="utf-8")
        print(f"Text output written to {args.output}")
    else:
        print(text_output)

    if args.json:
        args.json.write_text(serializer.to_json(record), encoding="utf-8")
        print(f"JSON output written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
