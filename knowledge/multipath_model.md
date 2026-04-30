# Multipath Reflection Model

## Physical Mechanism

When a radar signal reflects off surfaces (walls, floors, desks, metal
enclosures) in addition to the direct target path, the receiver sees a
superposition of delayed echoes:

$$
s_{\rm IF}(t) = s_{\rm direct}(t) + \sum_k \alpha_k s_{\rm direct}(t - \tau_k)
$$

where $\alpha_k$ is the $k$-th path's amplitude attenuation and $\tau_k$ is
its extra delay relative to the direct path.

## Multipath Signature in Range Profile

- **Ghost peak** at range $R' = R + \Delta R_{\rm mp}$ where
  $\Delta R_{\rm mp} = c \cdot \Delta\tau / 2$.
- Ghost peaks are **always at greater range than the true target** (extra path
  length can only increase the delay).
- Ghost peaks have **lower amplitude** than the direct peak ($\alpha_k < 1$).
- Ghost peaks exhibit **correlated phase behaviour** with the direct peak —
  their phase variations track each other.

## Discriminating Multipath from Real Targets

| Feature                | Multipath Ghost           | Real Secondary Target     |
|------------------------|---------------------------|---------------------------|
| Range relation         | Always $R_g > R_d$        | Arbitrary                 |
| Velocity               | Matches direct target     | Independent               |
| Phase correlation      | High ($r > 0.9$)          | Low ($r < 0.3$)           |
| Amplitude stability    | Fluctuates with geometry  | Stable                     |
| Angle-of-arrival       | Same as or near direct    | Different                  |

## Quantitative Criterion

Given direct-peak phase $\phi_d(t)$ and candidate ghost phase $\phi_g(t)$:

- **Phase difference over slow-time:**
  $$ \Delta\phi(t) = \phi_g(t) - \phi_d(t) \pmod{2\pi} $$

- If the standard deviation of $\Delta\phi$ across chirps is $< 5°$ →
  **multipath** (phase-locked).
- If the standard deviation exceeds $20°$ → **independent target**.

## Common Multipath Geometries

| Environment         | Typical $\Delta R_{\rm mp}$ | Attenuation $\alpha$    |
|---------------------|-----------------------------|-------------------------|
| Desktop surface     | 0.1–0.5 m                   | −4 to −10 dB            |
| Concrete wall (3 m) | 0.3–2.0 m                   | −8 to −20 dB            |
| Metal enclosure     | 0.05–0.2 m                  | −2 to −6 dB             |
| Floor bounce        | 0.5–3.0 m                   | −6 to −15 dB            |

## Diagnostic Flow

1. Identify any peak whose range > primary target and whose velocity matches.
2. Compute phase correlation between peaks.
3. If correlation > 0.9: **flag as probable multipath**.
4. Estimate path delay: $\Delta\tau = 2\Delta R / c$.
5. Cross-reference with known environmental geometry.
6. **Recommendation**: if multipath is confirmed, subtract ghost from the
   range profile, or adjust sensor placement.
