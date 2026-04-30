# FMCW Radar Physics Reference

## Basic Principles

**Frequency-Modulated Continuous Wave** (FMCW) radar transmits a chirp signal
whose frequency increases linearly with time.  The received echo is mixed with
the transmitted signal to produce an intermediate-frequency (IF) beat signal.

### Chirp Model

Transmitted signal for a single chirp:

$$
s_{\rm TX}(t) = A \cos\!\big(2\pi f_c t + \pi \frac{B}{T_c} t^2\big)
$$

where
- $f_c$ — centre frequency (e.g. 77–81 GHz for automotive radar)
- $B$ — bandwidth (sweep range)
- $T_c$ — chirp duration

### Beat Frequency → Range

For a stationary target at range $R$, the round-trip delay is $\tau = 2R / c$.
After de-chirping (mixing TX with RX), the IF signal is approximately:

$$
s_{\rm IF}(t) \propto \cos\!\big(2\pi f_b t + \phi\big)
$$

where the beat frequency is

$$
f_b = \frac{2B}{c\,T_c} R = \mu \cdot \frac{2R}{c}
$$

with chirp slope $\mu = B / T_c$.  **Range resolution** is limited by chirp
bandwidth:

$$
\Delta R = \frac{c}{2B}
$$

### Phase Sensitivity & Micro-Doppler

The phase of the IF signal encodes fine range variations:

$$
\phi = \frac{4\pi f_c R}{c}
$$

A displacement $\Delta R$ causes a phase shift $\Delta\phi = 4\pi f_c \Delta R / c$.
At 77 GHz, $\lambda \approx 3.9$ mm, so a **1 mm displacement → ≈ 92° phase
shift**.  This is the physical basis for detecting sub-millimetre motion
(respiration, heartbeat, vibration).

### Doppler → Velocity

Across successive chirps, the phase changes due to target motion:

$$
\Delta\phi_{\rm chirp} = \frac{4\pi f_c v T_c}{c}
$$

Velocity resolution for $N_c$ chirps:

$$
\Delta v = \frac{c}{2 f_c N_c T_c}
$$

---

## Common Signal Impairments

| Impairment       | Physical Cause                          | Signature in IF Signal                |
|------------------|-----------------------------------------|---------------------------------------|
| Phase noise      | Oscillator instability                  | Broadening of range-Doppler peaks     |
| IQ imbalance     | Gain/phase mismatch in mixer            | Image (ghost) targets                 |
| ADC saturation   | Strong nearby reflector                 | Harmonics in range profile            |
| Multipath        | Reflections from walls / surfaces       | Secondary peaks with consistent delay |
| Interference     | Other radars / EMI                      | Burst-like power in specific bins     |
| Thermal noise    | Johnson–Nyquist noise in receiver       | Uniform white noise floor             |

---

## Key Formulas for Anomaly Diagnosis

1. **Multipath range difference**
   $$ \Delta R_{\rm mp} = \frac{c \cdot \Delta \tau_{\rm mp}}{2} $$
   where $\Delta\tau_{\rm mp}$ is the delay between direct and reflected paths.

2. **Vibration displacement from phase**
   $$ \Delta x = \frac{\Delta\phi \cdot \lambda}{4\pi} $$
   where $\Delta\phi$ is the measured phase jump (unwrapped).

3. **Micro-Doppler frequency**
   $$ f_{\rm md} = \frac{2 f_c}{c} v_{\rm osc} $$
   where $v_{\rm osc}$ is the peak velocity of the oscillating surface.

4. **SNR after Range FFT processing gain**
   $$ {\rm SNR}_{\rm range} = {\rm SNR}_{\rm raw} + 10\log_{10}(N_{\rm samp}) $$
   $$ {\rm SNR}_{\rm RD} = {\rm SNR}_{\rm range} + 10\log_{10}(N_{\rm chirps}) $$
