"""
Feature Extractor
=================
Converts raw FMCW IF signals into structured feature records suitable
for downstream analysis by the engineering reasoning agent.

Pipeline
--------
  Raw IF → Range FFT → Doppler FFT → CFAR peak detection → Phase extraction
                                               ↓
                                    Anomaly detection ← phase unwrapping
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from numpy.fft import fft, fftshift
from scipy import signal as sp_signal


C = 299_792_458.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExtractionConfig:
    """Parameters controlling feature extraction sensitivity.

    Attributes
    ----------
    cfar_guard_cells : int
        Guard cells on each side of the cell under test.
    cfar_reference_cells : int
        Reference cells on each side (excludes guard cells).
    cfar_threshold_factor : float
        Multiplier applied to the noise estimate; higher = fewer false alarms.
    phase_jump_threshold_deg : float
        Minimum phase change between consecutive chirps to flag as an anomaly (degrees).
    spectral_deviation_threshold_db : float
        Power deviation above local median to flag as a spectral anomaly (dB).
    micro_doppler_min_freq_hz : float
        Lower bound for micro-Doppler search band (Hz).
    micro_doppler_max_freq_hz : float
        Upper bound for micro-Doppler search band (Hz).
    """

    cfar_guard_cells: int = 4
    cfar_reference_cells: int = 16
    cfar_threshold_factor: float = 2.5
    phase_jump_threshold_deg: float = 2.0
    spectral_deviation_threshold_db: float = 6.0
    micro_doppler_min_freq_hz: float = 0.1
    micro_doppler_max_freq_hz: float = 5.0


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class RangeBinAnomaly:
    """An anomaly detected within a specific range bin."""

    range_bin: int
    range_m: float
    phase_jump_deg: float                    # magnitude of phase discontinuity
    snr_db: float                            # signal-to-noise ratio at this bin
    anomaly_type: str = "unknown"            # "phase_jump", "spectral_peak", "micro_doppler"
    confidence: float = 0.0                  # 0–1


@dataclass
class DopplerAnomaly:
    """An anomaly in the Doppler domain."""

    freq_hz: float
    amplitude_db: float
    pattern: str                             # "micro_doppler_periodic", "broadband_noise", "narrowband_tone"
    harmonic_count: int = 1


@dataclass
class TargetEstimate:
    """Estimated target properties extracted from the range-Doppler map."""

    range_m: float
    velocity_mps: float
    snr_db: float
    range_bin: int = -1
    doppler_bin: int = -1


@dataclass
class FeatureRecord:
    """Complete feature record for one frame of IF data.

    This is the structured output consumed by TextSerializer and the
    engineering reasoning agent.
    """

    frame_id: int
    timestamp: str                                    # ISO-8601
    range_bins: List[float]                           # each bin's range in metres
    doppler_bins: List[float]                         # each bin's velocity in m/s
    targets: List[TargetEstimate]
    range_anomalies: List[RangeBinAnomaly]
    doppler_anomalies: List[DopplerAnomaly]
    range_doppler_map: np.ndarray                     # 2-D power in dB
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Extract structured features from raw FMCW IF signals.

    Parameters
    ----------
    config : ExtractionConfig
        Sensitivity parameters.
    sim_config : optional
        SimulationConfig used to derive range/velocity axis mapping.
    """

    def __init__(
        self,
        config: ExtractionConfig,
        f_start_hz: float = 77e9,
        f_stop_hz: float = 81e9,
        chirp_duration_s: float = 40e-6,
        sample_rate_hz: float = 10e6,
        frame_interval_s: float = 0.05,
    ) -> None:
        self.config = config
        self._f_center = (f_start_hz + f_stop_hz) / 2.0
        self._bandwidth = f_stop_hz - f_start_hz
        self._slope = self._bandwidth / chirp_duration_s
        self._sample_rate = sample_rate_hz
        self._frame_interval_s = frame_interval_s

    # -- main entry point ----------------------------------------------------

    def process(
        self, if_signal: np.ndarray, frame_id: int = 0, metadata: Optional[Dict] = None
    ) -> FeatureRecord:
        """Run the full extraction pipeline on one frame of IF data.

        Parameters
        ----------
        if_signal : np.ndarray
            Complex baseband IF, shape (n_chirps, n_samples).
        frame_id : int
            Sequential frame identifier.
        metadata : dict or None
            Arbitrary key-value metadata (environmental, sensor, etc.).

        Returns
        -------
        FeatureRecord
        """
        n_chirps, n_samples = if_signal.shape

        # -- axes --
        range_bins = self._compute_range_axis(n_samples)
        doppler_bins = self._compute_velocity_axis(n_chirps)

        # -- 2-D FFT (Range-Doppler map) --
        rd_map = self._range_doppler_fft(if_signal)
        rd_db = self._power_db(rd_map)

        # -- CFAR peak detection --
        targets = list(self._cfar_detect(rd_db, range_bins, doppler_bins))

        # -- Phase analysis --
        range_anomalies = list(self._detect_phase_anomalies(if_signal, range_bins))

        # -- Doppler anomaly search --
        doppler_anomalies = list(self._detect_doppler_anomalies(rd_db, doppler_bins))

        return FeatureRecord(
            frame_id=frame_id,
            timestamp=metadata.get("timestamp", "") if metadata else "",
            range_bins=range_bins.tolist(),
            doppler_bins=doppler_bins.tolist(),
            targets=targets,
            range_anomalies=range_anomalies,
            doppler_anomalies=doppler_anomalies,
            range_doppler_map=rd_db.astype(np.float32),
            metadata=metadata or {},
        )

    # -- FFT processing ------------------------------------------------------

    def _range_doppler_fft(self, if_signal: np.ndarray) -> np.ndarray:
        """Compute the 2-D range-Doppler FFT.

        Step 1: Range FFT across fast-time (axis=1)
        Step 2: Doppler FFT across slow-time (axis=0)
        Returns complex-valued range-Doppler map.
        """
        # Range FFT (per chirp)
        range_fft = fft(if_signal, axis=1)

        # Doppler FFT (per range bin)
        rd_map = fft(range_fft, axis=0)
        # Shift so that zero-Doppler is at centre
        return fftshift(rd_map, axes=0)

    def _power_db(self, spectrum: np.ndarray) -> np.ndarray:
        """Convert complex spectrum to dB power, guarding against log(0)."""
        power = np.abs(spectrum) ** 2
        power = np.maximum(power, np.finfo(power.dtype).tiny)
        return 10 * np.log10(power)

    # -- axes -----------------------------------------------------------------

    def _compute_range_axis(self, n_samples: int) -> np.ndarray:
        """Range values (m) for each range bin."""
        freq_per_bin = self._sample_rate / n_samples
        # f_beat = 2*slope*R / c  →  R = f_beat * c / (2*slope)
        freqs = np.arange(n_samples) * freq_per_bin
        return freqs * C / (2 * self._slope)

    def _compute_velocity_axis(self, n_chirps: int) -> np.ndarray:
        """Velocity values (m/s) for each Doppler bin (zero-centred)."""
        wavelength = C / self._f_center
        prf = 1.0 / self._chirp_duration  # approximate; actual depends on idle time
        freq_per_bin = prf / n_chirps
        # Doppler shift: f_d = 2*v / λ  →  v = f_d * λ / 2
        freqs = (np.arange(n_chirps) - n_chirps // 2) * freq_per_bin
        return freqs * wavelength / 2.0

    # -- CFAR peak detection -------------------------------------------------

    def _cfar_detect(
        self,
        rd_db: np.ndarray,
        range_axis: np.ndarray,
        velocity_axis: np.ndarray,
    ) -> Iterator[TargetEstimate]:
        """Cell-Averaging CFAR (CA-CFAR) in the range-Doppler domain.

        For efficiency, CFAR is applied along the range dimension on each
        Doppler slice.  A peak is declared when its power exceeds the
        local noise estimate by the configured threshold factor.
        """
        cfg = self.config
        guard = cfg.cfar_guard_cells
        ref = cfg.cfar_reference_cells
        n_doppler, n_range = rd_db.shape

        # Pre-compute range-wise noise estimate via 1-D sliding window
        for d_idx in range(n_doppler):
            row = rd_db[d_idx, :]
            for r_idx in range(ref + guard, n_range - ref - guard):
                cell_under_test = row[r_idx]
                left_window = row[r_idx - ref - guard : r_idx - guard]
                right_window = row[r_idx + guard : r_idx + guard + ref]
                noise_est = (np.mean(left_window) + np.mean(right_window)) / 2.0
                threshold = noise_est + cfg.cfar_threshold_factor

                if cell_under_test > threshold:
                    # Approximate SNR
                    snr = cell_under_test - noise_est
                    yield TargetEstimate(
                        range_m=float(range_axis[r_idx]),
                        velocity_mps=float(velocity_axis[d_idx]),
                        snr_db=float(snr),
                        range_bin=r_idx,
                        doppler_bin=d_idx,
                    )

    # -- Phase analysis ------------------------------------------------------

    def _detect_phase_anomalies(
        self, if_signal: np.ndarray, range_axis: np.ndarray
    ) -> Iterator[RangeBinAnomaly]:
        """Scan for phase discontinuities across slow-time (Doppler axis).

        For each range bin, extract the phase of the range-FFT peak, unwrap
        it across chirps, and flag bins where the phase delta exceeds the
        configured threshold.
        """
        cfg = self.config
        n_chirps, n_samples = if_signal.shape

        range_fft = fft(if_signal, axis=1)
        # Dominant range-bin per chirp
        peak_bin_per_chirp = np.argmax(np.abs(range_fft), axis=1)

        for bin_idx in range(n_samples):
            # Gather phase across chirps for this range bin
            phase_series = np.angle(range_fft[:, bin_idx])
            phase_unwrapped = np.unwrap(phase_series)
            phase_delta = np.abs(np.diff(phase_unwrapped))
            phase_delta_deg = np.rad2deg(phase_delta)

            jump_indices = np.where(phase_delta_deg > cfg.phase_jump_threshold_deg)[0]
            if len(jump_indices) == 0:
                continue

            # Compute median SNR at this bin across chirps
            snr_series = 20 * np.log10(
                np.abs(range_fft[:, bin_idx])
                / (np.mean(np.abs(range_fft)) + 1e-12)
            )
            median_snr = float(np.median(snr_series))
            max_jump = float(np.max(phase_delta_deg[jump_indices]))
            confidence = min(1.0, max_jump / (cfg.phase_jump_threshold_deg * 3))

            # Classify anomaly type
            a_type = "possible_phase_jump"
            if len(jump_indices) >= 3:
                a_type = "mechanical_vibration"
            elif 0.8 < np.mean(np.diff(jump_indices)) < 1.2 and len(jump_indices) >= 2:
                a_type = "periodic_displacement"  # could be micro-Doppler

            yield RangeBinAnomaly(
                range_bin=bin_idx,
                range_m=float(range_axis[bin_idx]),
                phase_jump_deg=max_jump,
                snr_db=median_snr,
                anomaly_type=a_type,
                confidence=confidence,
            )

    # -- Doppler anomaly search -----------------------------------------------

    def _detect_doppler_anomalies(
        self, rd_db: np.ndarray, velocity_axis: np.ndarray
    ) -> Iterator[DopplerAnomaly]:
        """Identify anomalous peaks in the collapsed Doppler spectrum."""
        cfg = self.config
        doppler_profile = np.mean(rd_db, axis=1)       # collapse across range
        median_power = float(np.median(doppler_profile))

        for d_idx, power_db in enumerate(doppler_profile):
            deviation = power_db - median_power
            if deviation < cfg.spectral_deviation_threshold_db:
                continue

            freq_hz = abs(float(velocity_axis[d_idx]))
            # Classify pattern
            if cfg.micro_doppler_min_freq_hz <= freq_hz <= cfg.micro_doppler_max_freq_hz:
                pattern = "possible_micro_doppler"
            elif deviation > 15:
                pattern = "narrowband_interference"
            else:
                pattern = "broadband_noise"

            yield DopplerAnomaly(
                freq_hz=freq_hz,
                amplitude_db=float(deviation),
                pattern=pattern,
            )

    @property
    def _chirp_duration(self) -> float:
        """Chirp duration — used by velocity axis calculation."""
        return 40e-6  # consistent with SimulationConfig default


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .fmcw_simulator import FMCWSimulator, MultipathConfig, SimulationConfig, Target

    scfg = SimulationConfig(num_chirps=64)
    targets = [
        Target(range_m=2.5, velocity_mps=0.3, rcs_dbsm=10,
               micro_doppler={"frequency_hz": 1.2, "amplitude_mm": 0.5}),
    ]
    sim = FMCWSimulator(scfg, targets, seed=123)
    if_signal = sim.generate()

    ecfg = ExtractionConfig()
    extractor = FeatureExtractor(ecfg)
    record = extractor.process(if_signal, frame_id=1, metadata={"sensor": "test"})

    print(f"Frame {record.frame_id}")
    print(f"  Targets detected: {len(record.targets)}")
    print(f"  Range anomalies:  {len(record.range_anomalies)}")
    print(f"  Doppler anomalies: {len(record.doppler_anomalies)}")
    for t in record.targets:
        print(f"    → Target @ {t.range_m:.2f} m, {t.velocity_mps:.2f} m/s, SNR={t.snr_db:.1f} dB")
    for a in record.range_anomalies:
        print(f"    ⚠  Range bin {a.range_bin} ({a.range_m:.2f} m): "
              f"phase jump {a.phase_jump_deg:.1f}°, type={a.anomaly_type}")
    for d in record.doppler_anomalies:
        print(f"    ⚠  Doppler {d.freq_hz:.2f} Hz: {d.pattern}, {d.amplitude_db:.1f} dB above median")
