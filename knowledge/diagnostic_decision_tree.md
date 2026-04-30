# Diagnostic Decision Tree

This document defines the systematic reasoning path that the Engineering
Reasoning Agent follows when presented with a structured signal record.

---

## Phase 1: Anomaly Triage

```
INPUT: FeatureRecord (targets, range_anomalies, doppler_anomalies, metadata)

Q1: Are there any range-domain anomalies?
    ├── NO  → skip to Phase 1b (Doppler-only analysis)
    └── YES → for each anomaly, classify by type:
               ├── "phase_jump"            → Phase 2a
               ├── "mechanical_vibration"  → Phase 2b
               ├── "periodic_displacement" → Phase 2c
               └── "possible_phase_jump"   → Phase 2a (with lower confidence)

Q1b: Are there any Doppler-domain anomalies?
    ├── NO  → frame is nominal
    └── YES → classify by pattern:
               ├── "possible_micro_doppler"    → Phase 2c
               ├── "narrowband_interference"   → Phase 2d
               └── "broadband_noise"           → Phase 2e
```

---

## Phase 2: Root-Cause Investigation

### 2a — Phase Jump Analysis

**Possible causes** (ordered by probability):
1. Multipath reflection from nearby surface
2. Mechanical vibration (sub-mm displacement)
3. ADC glitch / timing jitter
4. Target micro-motion (e.g. hand gesture)

**Investigation steps**:
1. Check phase-jump magnitude: < 5° → likely multipath; > 30° → vibration or
   micro-motion.
2. Check SNR at the anomaly bin: low SNR (< 10 dB) → suspect noise / ADC
   glitch.
3. Examine temporal pattern: isolated single jump → transient event; periodic
   jumps → vibration or biosignal.
4. Cross-reference with target list: if a target exists at the same range →
   check phase correlation (multipath criterion).

### 2b — Mechanical Vibration Analysis

**Possible causes**:
1. Nearby rotating machinery (motor, fan, pump)
2. Building structural resonance
3. Loose mounting bracket / sensor platform

**Investigation steps**:
1. Identify fundamental frequency of the vibration.
2. Check for harmonic series (HR > 0.3 → machinery).
3. Estimate displacement amplitude.
4. Correlate with known environmental sources (AC mains = 50/60 Hz and
   harmonics).
5. If amplitude < 0.01 mm and frequency > 100 Hz → possible bearing defect.

### 2c — Micro-Doppler / Biosignal Analysis

**Possible causes**:
1. Human respiration (0.1–0.5 Hz, 1–12 mm displacement)
2. Human heartbeat (0.8–3 Hz, 0.2–0.5 mm displacement)
3. Small animal motion
4. Wind-induced vibration of lightweight objects

**Investigation steps**:
1. Measure fundamental frequency and displacement amplitude.
2. Apply vibration-vs-biosignal discriminator (see `vibration_vs_biosignal.md`).
3. Check harmonic ratio and amplitude CV.
4. If biosignal confirmed → identify subject (human vs. animal by
   displacement/frequency).
5. Flag for continuous monitoring if in healthcare context.

### 2d — Narrowband Interference

**Possible causes**:
1. Another FMCW radar in vicinity (automotive)
2. Intentional jammer
3. Switching power supply harmonic
4. Wireless communication leakage

**Investigation steps**:
1. Note the interference frequency — compare with known bands (ISM, 5G, etc.).
2. Check burst duration: continuous → power supply; intermittent → another radar.
3. Recommend frequency agility or temporal blanking if persistent.

### 2e — Broadband Noise Excursion

**Possible causes**:
1. Thermal transient (receiver heating)
2. External wideband jammer
3. ADC malfunction
4. Extreme environmental conditions

**Investigation steps**:
1. Check noise floor across all range bins.
2. Compare with baseline noise from previous frames.
3. If sudden step-change → hardware fault; if gradual drift → thermal.
4. Recommend hardware diagnostics if persistent > 10 frames.

---

## Phase 3: Synthesis

1. Aggregate all anomaly findings into a ranked list of hypotheses.
2. For each hypothesis, assign a confidence score (0–1) based on:
   - Number of matching diagnostic criteria
   - SNR quality of the evidence
   - Consistency across multiple chirps/frames
3. If the top hypothesis has confidence < 0.6 → request additional frames
   or sensor reconfiguration.
4. Output the top hypothesis with supporting evidence and recommended action.

---

## Output Format

Diagnostic conclusion should include:

```
HYPOTHESIS: [primary diagnosis]
CONFIDENCE: [0.0–1.0]
EVIDENCE:
  - [finding 1 with quantitative support]
  - [finding 2 with quantitative support]
ALTERNATIVES:
  - [alternative 1] (confidence: X)
  - [alternative 2] (confidence: Y)
RECOMMENDATION: [actionable next step]
```
