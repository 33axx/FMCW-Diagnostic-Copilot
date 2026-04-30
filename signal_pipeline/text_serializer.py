"""
Text Serializer
===============
Converts structured `FeatureRecord` objects into natural-language text
sequences suitable for consumption by the engineering reasoning agent.

Design rationale
----------------
LLMs reason most effectively over well-structured natural language, not
raw numeric arrays.  This module bridges the signal-processing and
reasoning layers by producing a compact, information-dense text
representation of each frame's features.

Output format
-------------
Each frame is serialised as a multi-paragraph block with labelled
sections (targets, range anomalies, Doppler anomalies, metadata),
using consistent SI units and domain terminology.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np

from .feature_extractor import (
    DopplerAnomaly,
    FeatureRecord,
    RangeBinAnomaly,
    TargetEstimate,
)


# ---------------------------------------------------------------------------
# Input record (before feature extraction, for raw data serialisation)
# ---------------------------------------------------------------------------


@dataclass
class SignalRecord:
    """Lightweight record capturing the raw signal context.

    Used for logging / traceability before feature extraction.
    """

    frame_id: int
    timestamp: str                              # ISO-8601
    n_chirps: int
    n_samples: int
    bandwidth_hz: float
    center_freq_hz: float
    chirp_duration_s: float
    num_targets_simulated: int = 0
    multipath_enabled: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


class TextSerializer:
    """Convert FeatureRecord → natural language text block.

    Parameters
    ----------
    include_raw_statistics : bool
        Append raw numeric statistics at the end of each block.
    max_anomalies_per_section : int
        Truncate long anomaly lists to keep prompt length manageable.
    locale : str
        Influences unit formatting (future extension).
    """

    def __init__(
        self,
        include_raw_statistics: bool = False,
        max_anomalies_per_section: int = 20,
    ) -> None:
        self.include_raw_statistics = include_raw_statistics
        self.max_anomalies_per_section = max_anomalies_per_section

    # -- public API ----------------------------------------------------------

    def serialize(self, record: FeatureRecord) -> str:
        """Produce a complete text block for one frame."""
        parts: List[str] = []
        parts.append(self._header(record))
        parts.append(self._targets_section(record))
        parts.append(self._range_anomalies_section(record))
        parts.append(self._doppler_anomalies_section(record))
        parts.append(self._metadata_section(record))
        if self.include_raw_statistics:
            parts.append(self._raw_stats(record))
        return "\n\n".join(parts)

    def serialize_batch(
        self, records: Sequence[FeatureRecord], separator: str = "\n\n---\n\n"
    ) -> str:
        """Serialize multiple frames with a separator between them."""
        return separator.join(self.serialize(r) for r in records)

    def to_json(self, record: FeatureRecord) -> str:
        """Export a FeatureRecord as JSON for structured API consumption."""
        return json.dumps(
            {
                "frame_id": record.frame_id,
                "timestamp": record.timestamp,
                "targets": [
                    {
                        "range_m": round(t.range_m, 4),
                        "velocity_mps": round(t.velocity_mps, 4),
                        "snr_db": round(t.snr_db, 2),
                    }
                    for t in record.targets
                ],
                "range_anomalies": [
                    {
                        "range_bin": a.range_bin,
                        "range_m": round(a.range_m, 4),
                        "phase_jump_deg": round(a.phase_jump_deg, 2),
                        "snr_db": round(a.snr_db, 2),
                        "type": a.anomaly_type,
                        "confidence": round(a.confidence, 3),
                    }
                    for a in record.range_anomalies
                ],
                "doppler_anomalies": [
                    {
                        "freq_hz": round(d.freq_hz, 4),
                        "amplitude_db": round(d.amplitude_db, 2),
                        "pattern": d.pattern,
                    }
                    for d in record.doppler_anomalies
                ],
                "metadata": record.metadata,
            },
            indent=2,
            default=str,
        )

    # -- section builders ----------------------------------------------------

    def _header(self, rec: FeatureRecord) -> str:
        ts = rec.timestamp or datetime.now(timezone.utc).isoformat()
        return (
            f"[FRAME {rec.frame_id}] Timestamp: {ts}\n"
            f"Sensor: FMCW Radar  |  Mode: Surveillance\n"
            f"Range bins: {len(rec.range_bins)}  |  "
            f"Doppler bins: {len(rec.doppler_bins)}"
        )

    def _targets_section(self, rec: FeatureRecord) -> str:
        if not rec.targets:
            return "TARGETS: None detected in this frame."

        lines = ["TARGETS:"]
        for i, t in enumerate(rec.targets, 1):
            lines.append(
                f"  [{i}] Range = {t.range_m:.2f} m, "
                f"Velocity = {t.velocity_mps:+.2f} m/s, "
                f"SNR = {t.snr_db:.1f} dB"
            )
        return "\n".join(lines)

    def _range_anomalies_section(self, rec: FeatureRecord) -> str:
        anomalies = rec.range_anomalies[: self.max_anomalies_per_section]
        if not anomalies:
            return "RANGE-DOMAIN ANOMALIES: None."

        lines = ["RANGE-DOMAIN ANOMALIES:"]
        for a in anomalies:
            lines.append(
                f"  · Bin {a.range_bin} ({a.range_m:.3f} m): "
                f"Phase discontinuity {a.phase_jump_deg:.1f}°, "
                f"SNR = {a.snr_db:.1f} dB, "
                f"type = {a.anomaly_type}, confidence = {a.confidence:.2f}"
            )
        if len(rec.range_anomalies) > self.max_anomalies_per_section:
            lines.append(
                f"  … ({len(rec.range_anomalies) - self.max_anomalies_per_section} "
                f"more anomalies truncated)"
            )
        return "\n".join(lines)

    def _doppler_anomalies_section(self, rec: FeatureRecord) -> str:
        anomalies = rec.doppler_anomalies[: self.max_anomalies_per_section]
        if not anomalies:
            return "DOPPLER-DOMAIN ANOMALIES: None."

        lines = ["DOPPLER-DOMAIN ANOMALIES:"]
        for d in anomalies:
            lines.append(
                f"  · {d.freq_hz:.2f} Hz: {d.pattern}, "
                f"amplitude deviation = {d.amplitude_db:.1f} dB"
            )
        if len(rec.doppler_anomalies) > self.max_anomalies_per_section:
            lines.append(
                f"  … ({len(rec.doppler_anomalies) - self.max_anomalies_per_section} "
                f"more anomalies truncated)"
            )
        return "\n".join(lines)

    def _metadata_section(self, rec: FeatureRecord) -> str:
        """Render user-supplied metadata as key-value pairs."""
        if not rec.metadata:
            return "METADATA: (none)"
        lines = ["METADATA:"]
        for key, val in rec.metadata.items():
            lines.append(f"  {key}: {val}")
        return "\n".join(lines)

    def _raw_stats(self, rec: FeatureRecord) -> str:
        """Append raw numeric statistics block."""
        rd = rec.range_doppler_map
        return (
            "RAW STATISTICS:\n"
            f"  RD map shape: {rd.shape}\n"
            f"  RD power min/mean/max: "
            f"{np.min(rd):.1f} / {np.mean(rd):.1f} / {np.max(rd):.1f} dB"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .feature_extractor import ExtractionConfig, FeatureExtractor
    from .fmcw_simulator import FMCWSimulator, MultlipathConfig, SimulationConfig, Target

    scfg = SimulationConfig(num_chirps=64)
    targets = [
        Target(range_m=4.0, velocity_mps=0.2, rcs_dbsm=10,
               micro_doppler={"frequency_hz": 1.2, "amplitude_mm": 0.5}),
    ]
    sim = FMCWSimulator(scfg, targets, seed=42)
    sig = sim.generate()

    extractor = FeatureExtractor(ExtractionConfig())
    rec = extractor.process(sig, frame_id=42, metadata={"room": "lab_A", "temp_c": 22.5})

    ser = TextSerializer(include_raw_statistics=True)
    print(ser.serialize(rec))
    print("\n--- JSON ---")
    print(ser.to_json(rec)[:500])
