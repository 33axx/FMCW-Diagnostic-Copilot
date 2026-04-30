# Mechanical Vibration vs. Biological Micro-Doppler

## Problem Statement

Both mechanical vibration (e.g. motors, pumps, HVAC) and biological
micro-Doppler (respiration, heartbeat) produce periodic phase modulation
in the radar IF signal.  Discriminating them is essential for:

- Industrial predictive maintenance (vibration → bearing fault)
- Healthcare monitoring (micro-Doppler → vital signs)
- Security (distinguishing machinery from human presence)

## Spectral Signatures

| Feature                   | Mechanical Vibration            | Biological Micro-Doppler          |
|---------------------------|---------------------------------|-----------------------------------|
| **Frequency range**       | 10–500 Hz (broad)              | 0.1–3 Hz (narrow)                 |
| **Harmonic content**      | Rich harmonics (nonlinear)      | Few harmonics (≈ sinusoidal)      |
| **Amplitude modulation**  | Often wideband / impulsive      | Smooth, quasi-periodic            |
| **Q-factor**              | High (sharp spectral lines)     | Low (broadened by natural variation) |
| **Temporal stability**    | Stationary (runs continuously)  | Non-stationary (breath-hold, movement) |
| **Phase coherence**       | High over seconds               | Drifts over tens of seconds       |

## Quantitative Discriminators

### 1. Frequency Bounds
$$
\begin{aligned}
f_{\rm vibration} &> 5\ \rm Hz\quad\text{(highly likely)} \\
f_{\rm biosignal} &< 3\ \rm Hz\quad\text{(highly likely)} \\
\end{aligned}
$$

**Overlap zone**: 3–5 Hz is ambiguous — use additional features.

### 2. Harmonic Ratio
Mechanical vibration generically produces odd-harmonic series (from nonlinear
stiffness in bearings/gears).  Compute:
$$
H\!R = \frac{\sum_{k=2}^N P(k f_0)}{\sum_{k=1}^N P(k f_0)}
$$
where $P(f)$ is Doppler power at frequency $f$ and $f_0$ is the fundamental.
- $HR > 0.3$ → **mechanical**
- $HR < 0.15$ → **biological**

### 3. Amplitude Stability (Coefficient of Variation)
Over a sliding 10-second window, measure CV of the peak Doppler amplitude:
$$
CV = \frac{\sigma_A}{\mu_A}
$$
- $CV < 0.15$ → **mechanical** (stable)
- $CV > 0.25$ → **biological** (natural breath-to-breath variability)

### 4. Micro-Doppler Displacement Amplitude
Estimate physical displacement from phase:
$$
\Delta x_{\rm pk} = \frac{\lambda \cdot \Delta\phi_{\rm pk}}{4\pi}
$$

| Source            | Typical $\Delta x$ |
|-------------------|--------------------|
| Human respiration | 1–12 mm            |
| Human heartbeat   | 0.2–0.5 mm         |
| AC motor (60 Hz)  | 0.01–0.05 mm       |
| Pump / compressor | 0.05–0.5 mm        |
| Bearing fault     | 0.001–0.01 mm      |

## Decision Logic

```
IF fundamental_freq < 0.5 Hz:
    → probable respiration (check displacement > 1 mm)
ELIF 0.5 Hz < fundamental_freq < 3 Hz AND displacement > 0.2 mm:
    → probable heartbeat (check CV > 0.25)
ELIF fundamental_freq > 5 Hz AND harmonics(HR > 0.3):
    → mechanical vibration (check Q-factor)
ELIF fundamental_freq > 5 Hz AND harmonics(HR < 0.15):
    → possible fan / low-harmonic machinery (further investigation)
ELSE:
    → ambiguous — request temporal context or additional frames
```
