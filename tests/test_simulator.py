"""Tests for the FMCW signal simulator."""

import numpy as np
import pytest

from signal_pipeline.fmcw_simulator import (
    FMCWSimulator,
    MultipathConfig,
    SimulationConfig,
    Target,
)


class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.bandwidth_hz == 4e9
        assert cfg.slope_hz_per_s == pytest.approx(1e14)

    def test_custom_bandwidth(self):
        cfg = SimulationConfig(f_start_hz=76e9, f_stop_hz=78e9)
        assert cfg.bandwidth_hz == 2e9


class TestFMCWSimulator:
    @pytest.fixture
    def basic_config(self):
        return SimulationConfig(num_chirps=32)

    @pytest.fixture
    def single_target(self):
        return [Target(range_m=5.0, velocity_mps=1.0, rcs_dbsm=10.0)]

    def test_generate_shape(self, basic_config, single_target):
        sim = FMCWSimulator(basic_config, single_target, seed=42)
        sig = sim.generate()
        assert sig.ndim == 2
        assert sig.shape[0] == 32  # num_chirps
        assert sig.shape[1] >= 256  # num_samples (power of 2)

    def test_generate_complex_dtype(self, basic_config, single_target):
        sim = FMCWSimulator(basic_config, single_target, seed=42)
        sig = sim.generate()
        assert np.iscomplexobj(sig)

    def test_seed_reproducibility(self, basic_config, single_target):
        sim1 = FMCWSimulator(basic_config, single_target, seed=42)
        sim2 = FMCWSimulator(basic_config, single_target, seed=42)
        sig1 = sim1.generate()
        sig2 = sim2.generate()
        assert np.allclose(sig1, sig2)

    def test_different_seeds_different(self, basic_config, single_target):
        sim1 = FMCWSimulator(basic_config, single_target, seed=42)
        sim2 = FMCWSimulator(basic_config, single_target, seed=99)
        sig1 = sim1.generate()
        sig2 = sim2.generate()
        # Different noise seeds produce different noise realizations.
        # The signals themselves are close (same targets), but the noise
        # component should cause a measurable difference.
        diff = np.max(np.abs(sig1 - sig2))
        assert diff > 0, f"Expected non-zero difference between seeds, got {diff}"

    def test_range_resolution(self, basic_config):
        sim = FMCWSimulator(basic_config, [Target(range_m=5.0, velocity_mps=0)])
        res = sim.range_resolution_m()
        # ΔR = c / (2B), B = 4 GHz → ΔR ≈ 0.0375 m
        assert 0.03 < res < 0.05

    def test_velocity_resolution(self, basic_config):
        sim = FMCWSimulator(basic_config, [Target(range_m=5.0, velocity_mps=0)])
        res = sim.velocity_resolution_mps()
        assert res > 0

    def test_multipath_adds_signal(self, basic_config, single_target):
        mp = MultipathConfig(enabled=True, path_delay_s=5e-9, attenuation_db=-3)
        sim_no_mp = FMCWSimulator(basic_config, single_target, multipath=MultipathConfig(enabled=False), seed=42)
        sim_mp = FMCWSimulator(basic_config, single_target, multipath=mp, seed=42)
        sig_no = sim_no_mp.generate()
        sig_yes = sim_mp.generate()
        # With multipath, the signal should differ
        assert not np.allclose(sig_no, sig_yes)

    def test_multiple_targets(self, basic_config):
        targets = [
            Target(range_m=3.0, velocity_mps=0.5, rcs_dbsm=10),
            Target(range_m=8.0, velocity_mps=-2.0, rcs_dbsm=5),
        ]
        sim = FMCWSimulator(basic_config, targets, seed=42)
        sig = sim.generate()
        assert sig.shape == (32, sim._samples_per_chirp)

    def test_micro_doppler_modulates_signal(self, basic_config):
        t1 = Target(range_m=1.5, velocity_mps=0, rcs_dbsm=5)
        t2 = Target(range_m=1.5, velocity_mps=0, rcs_dbsm=5,
                     micro_doppler={"frequency_hz": 1.2, "amplitude_mm": 1.0})
        sim1 = FMCWSimulator(basic_config, [t1], seed=42)
        sim2 = FMCWSimulator(basic_config, [t2], seed=42)
        sig1 = sim1.generate()
        sig2 = sim2.generate()
        # Micro-Doppler changes the signal
        assert not np.allclose(sig1, sig2)

    def test_output_not_all_nan(self, basic_config, single_target):
        sim = FMCWSimulator(basic_config, single_target, seed=42)
        sig = sim.generate()
        assert not np.any(np.isnan(sig))
        assert not np.any(np.isinf(sig))
