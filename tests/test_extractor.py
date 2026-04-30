"""Tests for the feature extractor."""

import numpy as np
import pytest

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


class TestFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor(ExtractionConfig())

    @pytest.fixture
    def if_signal(self):
        cfg = SimulationConfig(num_chirps=64)
        targets = [
            Target(range_m=4.0, velocity_mps=0.5, rcs_dbsm=12),
        ]
        sim = FMCWSimulator(cfg, targets, seed=42)
        return sim.generate()

    def test_process_returns_feature_record(self, extractor, if_signal):
        record = extractor.process(if_signal, frame_id=1)
        assert isinstance(record, FeatureRecord)
        assert record.frame_id == 1

    def test_range_axis_length(self, extractor, if_signal):
        record = extractor.process(if_signal)
        assert len(record.range_bins) == if_signal.shape[1]

    def test_doppler_axis_length(self, extractor, if_signal):
        record = extractor.process(if_signal)
        assert len(record.doppler_bins) == if_signal.shape[0]

    def test_range_doppler_map_shape(self, extractor, if_signal):
        record = extractor.process(if_signal)
        assert record.range_doppler_map.shape == if_signal.shape

    def test_range_doppler_map_db(self, extractor, if_signal):
        record = extractor.process(if_signal)
        # dB values should not be absurd (should be in a reasonable range)
        assert np.all(record.range_doppler_map < 200)
        assert np.all(np.isfinite(record.range_doppler_map))

    def test_metadata_passed_through(self, extractor, if_signal):
        meta = {"test_key": "test_value", "temp_c": 22.0}
        record = extractor.process(if_signal, metadata=meta)
        assert record.metadata["test_key"] == "test_value"
        assert record.metadata["temp_c"] == 22.0

    def test_target_detection_snr_positive(self, extractor, if_signal):
        record = extractor.process(if_signal)
        for t in record.targets:
            assert t.snr_db > 0, "Detected target should have positive SNR"

    def test_output_data_types(self, extractor, if_signal):
        record = extractor.process(if_signal)
        for t in record.targets:
            assert isinstance(t.range_m, float)
            assert isinstance(t.velocity_mps, float)
            assert isinstance(t.snr_db, float)
        for a in record.range_anomalies:
            assert isinstance(a.range_bin, int)
            assert isinstance(a.phase_jump_deg, float)
        for d in record.doppler_anomalies:
            assert isinstance(d.freq_hz, float)


class TestExtractionConfig:
    def test_defaults(self):
        cfg = ExtractionConfig()
        assert cfg.cfar_guard_cells > 0
        assert cfg.cfar_reference_cells > 0
        assert cfg.phase_jump_threshold_deg > 0
