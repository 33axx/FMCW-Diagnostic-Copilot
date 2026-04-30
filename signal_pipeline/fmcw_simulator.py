"""
FMCW Radar Signal Simulator
============================
Generates realistic Frequency-Modulated Continuous Wave (FMCW) radar
intermediate-frequency (IF) signals with configurable targets, multipath
reflections, phase noise, and environmental conditions.

Mathematical foundation
-----------------------
The transmitted chirp is modelled as:

    s_TX(t) = A * cos(2π * f_c * t + π * (B / T_c) * t²)

After mixing with the delayed echo from a target at range R with radial
velocity v, the IF beat signal for a single chirp is:

    s_IF(t) ≈ A_rx * cos(2π * f_b * t + φ)

where:
    f_b = (2 * B * R) / (c * T_c)          beat frequency (range)
    φ   = 4π * f_c * R / c                  phase (proportional to range,
                                               sensitive to sub-mm motion)

For multiple chirps, the phase variation across chirps encodes velocity
(Doppler effect).  The range-Doppler map is obtained via a 2D FFT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Target:
    """A point target in the radar field of view.

    Attributes
    ----------
    range_m : float
        Radial distance from radar (metres).
    velocity_mps : float
        Radial velocity (positive = receding, m/s).
    rcs_dbsm : float
        Radar cross-section in dBsm; controls echo amplitude.
    micro_doppler : Optional[Dict[str, float]]
        If provided, superimpose micro-Doppler modulation.
        Keys: ``frequency_hz``, ``amplitude_mm``.
    """

    range_m: float
    velocity_mps: float
    rcs_dbsm: float = 0.0
    micro_doppler: Optional[Dict[str, float]] = None


@dataclass
class MultipathConfig:
    """Multipath reflection settings injected into the simulation.

    Attributes
    ----------
    enabled : bool
        Toggle multipath on/off.
    path_delay_s : float
        Additional delay of the reflected path (seconds).
    attenuation_db : float
        Attenuation relative to the direct path (dB).
    phase_shift_rad : float
        Phase shift applied to the reflected copy (rad).
    """

    enabled: bool = False
    path_delay_s: float = 5e-9   # ~0.75 m extra path
    attenuation_db: float = -6.0
    phase_shift_rad: float = 0.0


@dataclass
class SimulationConfig:
    """Global parameters for one FMCW frame.

    Attributes
    ----------
    f_start_hz : float
        Start frequency of the chirp (Hz).
    f_stop_hz : float
        Stop frequency of the chirp (Hz).
    chirp_duration_s : float
        Duration of a single chirp ramp (s).
    num_chirps : int
        Number of chirps per frame (defines Doppler resolution).
    sample_rate_hz : float
        ADC sampling rate (Hz).
    num_samples_per_chirp : int
        Samples per chirp; derived if omitted.
    noise_figure_db : float
        Receiver noise figure (dB).
    phase_noise_dbc_hz_at_1khz : float
        Simplified phase-noise level (dBc/Hz @ 1 kHz offset).
    temperature_celsius : float
        Ambient temperature for noise floor computation (°C).
    """

    f_start_hz: float = 77e9
    f_stop_hz: float = 81e9
    chirp_duration_s: float = 40e-6
    num_chirps: int = 128
    sample_rate_hz: float = 10e6
    num_samples_per_chirp: int = 0          # 0 → derive from sample_rate * chirp_duration
    noise_figure_db: float = 12.0
    phase_noise_dbc_hz_at_1khz: float = -90.0
    temperature_celsius: float = 25.0

    @property
    def bandwidth_hz(self) -> float:
        return self.f_stop_hz - self.f_start_hz

    @property
    def slope_hz_per_s(self) -> float:
        return self.bandwidth_hz / self.chirp_duration_s


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------

C = 299_792_458.0          # speed of light  m/s
K_B = 1.380649e-23         # Boltzmann's constant  J/K


def _sample_count(config: SimulationConfig) -> int:
    if config.num_samples_per_chirp > 0:
        return config.num_samples_per_chirp
    n = int(config.sample_rate_hz * config.chirp_duration_s)
    # round to nearest power of 2 for efficient FFT
    return 2 ** (n - 1).bit_length()


def _thermal_noise_power(config: SimulationConfig) -> float:
    """Noise power in linear units (variance of complex Gaussian)."""
    T_kelvin = config.temperature_celsius + 273.15
    B = config.sample_rate_hz
    # k * T * B * 10^(NF/10)
    return K_B * T_kelvin * B * 10 ** (config.noise_figure_db / 10.0)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class FMCWSimulator:
    """Generate synthetic FMCW IF signals with realistic impairments.

    Usage::

        config = SimulationConfig()
        targets = [Target(range_m=5.0, velocity_mps=1.5)]
        sim = FMCWSimulator(config, targets)
        if_signal = sim.generate()
    """

    def __init__(
        self,
        config: SimulationConfig,
        targets: List[Target],
        multipath: Optional[MultipathConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.targets = targets
        self.multipath = multipath or MultipathConfig()
        self._rng = np.random.default_rng(seed)

        self.n_samples = _sample_count(config)
        self.n_chirps = config.num_chirps
        self._samples_per_chirp = self.n_samples

        # Derived constants
        self._f_center = (config.f_start_hz + config.f_stop_hz) / 2.0
        self._slope = config.slope_hz_per_s
        self._chirp_time = config.chirp_duration_s

        # Time axes
        self._t_fast = np.arange(self.n_samples) / config.sample_rate_hz
        self._t_slow = np.arange(self.n_chirps) * self._chirp_time

    # -- public API ----------------------------------------------------------

    def generate(self) -> np.ndarray:
        """Return complex baseband IF signal, shape (n_chirps, n_samples)."""
        signal = np.zeros((self.n_chirps, self.n_samples), dtype=np.complex128)

        for target in self.targets:
            signal += self._render_target(target)

        if self.multipath.enabled:
            signal += self._render_multipath(signal)

        signal += self._generate_noise()
        return signal.astype(np.complex64)

    def range_resolution_m(self) -> float:
        """Range resolution (metres) determined by chirp bandwidth."""
        return C / (2 * self.config.bandwidth_hz)

    def velocity_resolution_mps(self) -> float:
        """Velocity resolution (m/s) determined by frame duration."""
        wavelength = C / self._f_center
        return wavelength / (2 * self.n_chirps * self._chirp_time)

    def max_range_m(self) -> float:
        """Unambiguous range (metres) from ADC sample rate."""
        return (C * self.config.sample_rate_hz) / (2 * self._slope)

    # -- target rendering ----------------------------------------------------

    def _render_target(self, target: Target) -> np.ndarray:
        """Build the 2D IF signal for a single point target."""
        range_m = target.range_m
        velocity = target.velocity_mps
        amplitude = 10 ** (target.rcs_dbsm / 20.0)

        # Beat frequency (range)
        f_beat = (2 * self._slope * range_m) / C

        signal = np.zeros((self.n_chirps, self.n_samples), dtype=np.complex128)
        for chirp_idx in range(self.n_chirps):
            # Phase includes range + Doppler contribution
            # ΔR = velocity * t_slow  →  phase_RX = 4π (R + v*t_slow) / λ
            r_instant = range_m + velocity * chirp_idx * self._chirp_time
            phi = 4 * math.pi * self._f_center * r_instant / C

            signal[chirp_idx, :] = amplitude * np.exp(
                1j * (2 * math.pi * f_beat * self._t_fast + phi)
            )

        # Micro-Doppler overlay
        if target.micro_doppler:
            signal *= self._apply_micro_doppler(target.micro_doppler, amplitude)

        return signal

    def _apply_micro_doppler(
        self, md: Dict[str, float], amplitude: float
    ) -> np.ndarray:
        """Phase-modulate the signal with periodic micro-Doppler.

        Simulates small-scale oscillatory motion (e.g. heartbeat ~1 Hz,
        respiration ~0.2 Hz) superimposed on the target echo.
        """
        freq = md.get("frequency_hz", 1.0)
        amp_mm = md.get("amplitude_mm", 1.0)
        amp_m = amp_mm * 1e-3

        # Phase modulation: φ_md = 4π * A_m * sin(2π f_md t_slow) / λ
        wavelength = C / self._f_center
        modulation = np.zeros((self.n_chirps, 1), dtype=np.float64)
        for chirp_idx in range(self.n_chirps):
            t_slow = chirp_idx * self._chirp_time
            modulation[chirp_idx, 0] = (
                4 * math.pi * amp_m * math.sin(2 * math.pi * freq * t_slow) / wavelength
            )
        return np.exp(1j * modulation)

    # -- multipath -----------------------------------------------------------

    def _render_multipath(self, direct_signal: np.ndarray) -> np.ndarray:
        """Add a delayed, attenuated copy of the direct-path signal."""
        mp = self.multipath
        atten_linear = 10 ** (mp.attenuation_db / 20.0)
        delay_samples = int(mp.path_delay_s * self.config.sample_rate_hz)

        if delay_samples >= self.n_samples:
            return np.zeros_like(direct_signal)

        multipath_signal = np.zeros_like(direct_signal)
        multipath_signal[:, delay_samples:] = (
            direct_signal[:, : self.n_samples - delay_samples]
        )
        return atten_linear * multipath_signal * np.exp(1j * mp.phase_shift_rad)

    # -- noise ---------------------------------------------------------------

    def _generate_noise(self) -> np.ndarray:
        """Complex AWGN + phase noise approximation."""
        noise_power = _thermal_noise_power(self.config)
        # Complex AWGN: real & imag each have variance noise_power/2
        awgn = (
            self._rng.normal(
                0, math.sqrt(noise_power / 2), (self.n_chirps, self.n_samples)
            )
            + 1j
            * self._rng.normal(
                0, math.sqrt(noise_power / 2), (self.n_chirps, self.n_samples)
            )
        )

        # Simplified phase noise: low-frequency random walk in phase
        phase_noise_level = 10 ** (self.config.phase_noise_dbc_hz_at_1khz / 20.0)
        phase_walk = np.cumsum(
            self._rng.normal(0, phase_noise_level, (self.n_chirps, self.n_samples)),
            axis=1,
        )
        return awgn * np.exp(1j * phase_walk)


# ---------------------------------------------------------------------------
# Smoke-test (run with `python -m signal_pipeline.fmcw_simulator`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = SimulationConfig()
    tgts = [
        Target(range_m=3.0, velocity_mps=0.5, rcs_dbsm=10.0),
        Target(range_m=7.5, velocity_mps=-2.0, rcs_dbsm=5.0,
               micro_doppler={"frequency_hz": 1.2, "amplitude_mm": 0.5}),
    ]
    mp = MultipathConfig(enabled=True, path_delay_s=1e-9)
    sim = FMCWSimulator(cfg, tgts, multipath=mp, seed=42)
    sig = sim.generate()
    print(f"IF signal shape: {sig.shape}")
    print(f"Range  resolution: {sim.range_resolution_m():.3f} m")
    print(f"Velocity resolution: {sim.velocity_resolution_mps():.3f} m/s")
    print(f"Max unambiguous range: {sim.max_range_m():.2f} m")
    print(f"Signal power (dB): {10*np.log10(np.mean(np.abs(sig)**2)):.1f}")
