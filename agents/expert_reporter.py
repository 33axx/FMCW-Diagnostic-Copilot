"""
Expert Reporter
===============
Generates detailed diagnostic reports with LaTeX-formatted mathematical
derivations from the reasoning agent's output.

The report includes:
  - Executive summary (anomaly overview)
  - Physical derivation (key formulas with LaTeX rendering)
  - Root-cause hypothesis ranking
  - Recommended actions

Two output formats are supported:
  1. LaTeX source (Compilable to PDF with pdflatex/xelatex)
  2. Markdown (for direct GitHub rendering)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .engineering_reasoner import Hypothesis, ReasoningResult


# ---------------------------------------------------------------------------
# LaTeX report generator
# ---------------------------------------------------------------------------


class ExpertReporter:
    """Generate structured diagnostic reports from ReasoningResult.

    Parameters
    ----------
    author : str
        Name to appear in the report header.
    sensor_id : str
        Sensor/device identifier.
    """

    def __init__(
        self,
        author: str = "FMCW Diagnostic Agent",
        sensor_id: str = "FMCW-RADAR-001",
    ) -> None:
        self.author = author
        self.sensor_id = sensor_id

    # -- LaTeX ---------------------------------------------------------------

    def to_latex(self, result: ReasoningResult, output_path: Optional[Path] = None) -> str:
        """Generate a full LaTeX document as a string and optionally write to file.

        Parameters
        ----------
        result : ReasoningResult
            The reasoning chain output.
        output_path : Path or None
            If given, write the LaTeX source to this path.

        Returns
        -------
        str
            Complete LaTeX source.
        """
        doc = self._latex_template(result)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(doc, encoding="utf-8")
        return doc

    def _latex_template(self, result: ReasoningResult) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        primary = result.primary_diagnosis
        confidence = f"{primary.confidence:.0%}" if primary else "N/A"

        latex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\geometry{{margin=1in}}

\title{{FMCW Radar Diagnostic Report}}
\author{{{self.author}}}
\date{{{ts}}}

\begin{{document}}
\maketitle

\section{{Executive Summary}}

\textbf{{Frame ID:}} {result.frame_id} \\
\textbf{{Sensor:}} {self.sensor_id} \\
\textbf{{Primary Diagnosis:}} {primary.name.replace('_', ' ').title() if primary else 'None'} \\
\textbf{{Confidence:}} {confidence} \\
\textbf{{Requires Human Review:}} {'Yes' if result.requires_human_review else 'No'}

\bigskip
\noindent {result.input_summary}

"""

        # Reasoning chain
        latex += self._latex_reasoning_steps(result)

        # Hypotheses ranking
        latex += self._latex_hypotheses(result)

        # Physical derivation
        latex += self._latex_derivation_section(result)

        # Recommendation
        if primary:
            latex += self._latex_recommendation(primary)

        latex += "\n\\end{document}\n"
        return latex

    def _latex_reasoning_steps(self, result: ReasoningResult) -> str:
        out = "\\section{Diagnostic Reasoning Chain}\n\n"
        for step in result.steps:
            step_name = step.step_name.replace("_", " ").title()
            out += f"\\subsection*{{{step_name}}}\n\\begin{{itemize}}\n"
            for finding in step.findings:
                safe = finding.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
                out += f"  \\item {safe}\n"
            out += "\\end{itemize}\n"

            if step.active_hypotheses:
                hypotheses_str = ", ".join(h.replace("_", " ") for h in step.active_hypotheses)
                out += (
                    f"\\textbf{{Active hypotheses:}} {hypotheses_str}\\\\\n"
                )
            if step.eliminated_hypotheses:
                eliminated_str = ", ".join(h.replace("_", " ") for h in step.eliminated_hypotheses)
                out += (
                    f"\\textbf{{Eliminated:}} {eliminated_str}\\\\\n"
                )
            out += "\\medskip\n"
        return out

    def _latex_hypotheses(self, result: ReasoningResult) -> str:
        out = "\\section{Hypothesis Ranking}\n\n"
        out += "\\begin{tabular}{@{}lll@{}}\n"
        out += "\\toprule\n"
        out += "\\textbf{Rank} & \\textbf{Hypothesis} & \\textbf{Confidence} \\\\\n"
        out += "\\midrule\n"
        for i, h in enumerate(result.hypotheses, 1):
            name = h.name.replace("_", " ").title()
            conf = f"{h.confidence:.0\\%}"
            # Highlight top hypothesis
            if i == 1:
                name = f"\\textbf{{{name}}}"
                conf = f"\\textbf{{{conf}}}"
            out += f"  {i} & {name} & {conf} \\\\\n"
        out += "\\bottomrule\n"
        out += "\\end{tabular}\n\n"

        # Evidence for each hypothesis
        for h in result.hypotheses:
            name = h.name.replace("_", " ").title()
            out += f"\\subsection*{{{name} (Confidence: {h.confidence:.0%})}}\n"
            out += "\\begin{itemize}\n"
            for ev in h.evidence:
                safe = ev.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
                out += f"  \\item {safe}\n"
            out += "\\end{itemize}\n"
        return out

    def _latex_derivation_section(self, result: ReasoningResult) -> str:
        """Include LaTeX formulas relevant to the primary diagnosis."""
        primary = result.primary_diagnosis
        if not primary:
            return ""

        out = "\\section{Physical Derivation}\n\n"

        # Common formulas
        out += r"""
\subsection*{FMCW Range-Doppler Fundamentals}

\textbf{Beat frequency → range:}
\begin{equation}
  R = \frac{c \, f_b}{2 \mu}, \quad \mu = \frac{B}{T_c}
  \label{eq:range}
\end{equation}

\textbf{Phase sensitivity to displacement:}
\begin{equation}
  \Delta\phi = \frac{4\pi f_c \Delta R}{c}
  \label{eq:phase}
\end{equation}

For $f_c = 79\ \mathrm{GHz}$, $\lambda \approx 3.8\ \mathrm{mm}$.
A displacement of $1\ \mathrm{mm}$ yields $\Delta\phi \approx 92^\circ$.

\textbf{Doppler velocity:}
\begin{equation}
  v = \frac{c \, f_d}{2 f_c}
  \label{eq:doppler}
\end{equation}
"""

        name = primary.name
        if name == "multipath_reflection":
            out += r"""
\subsection*{Multipath Discrimination}

The phase difference between direct and reflected paths:
\begin{equation}
  \Delta\phi_{\mathrm{mp}}(t) = \phi_g(t) - \phi_d(t) \pmod{2\pi}
\end{equation}

If $\sigma(\Delta\phi_{\mathrm{mp}}) < 5^\circ$ across chirps, the secondary
peak is phase-locked to the direct path → \textbf{multipath ghost}.

Extra path length:
\begin{equation}
  \Delta R_{\mathrm{mp}} = \frac{c \cdot \Delta\tau}{2}
\end{equation}
"""
        elif name in ("mechanical_vibration",):
            out += r"""
\subsection*{Vibration Displacement from Phase}

Physical displacement amplitude:
\begin{equation}
  \Delta x = \frac{\lambda \cdot \Delta\phi_{\mathrm{pk}}}{4\pi}
\end{equation}

Harmonic ratio for mechanical source detection:
\begin{equation}
  HR = \frac{\sum_{k=2}^N P(k f_0)}{\sum_{k=1}^N P(k f_0)}
\end{equation}

$HR > 0.3$ strongly suggests a mechanical (nonlinear) vibration source.
"""
        elif name == "biosignal_micro_doppler":
            out += r"""
\subsection*{Micro-Doppler Biosignal Analysis}

Respiration band: $0.1\text{--}0.5\ \mathrm{Hz}$, displacement $1\text{--}12\ \mathrm{mm}$.

Heartbeat band: $0.8\text{--}3\ \mathrm{Hz}$, displacement $0.2\text{--}0.5\ \mathrm{mm}$.

Amplitude coefficient of variation for stationarity test:
\begin{equation}
  CV = \frac{\sigma_A}{\mu_A}
\end{equation}

$CV > 0.25$ indicates biological (non-stationary) source.
"""
        return out

    def _latex_recommendation(self, primary: Hypothesis) -> str:
        safe_rec = primary.recommendation.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
        return (
            "\\section{Recommendation}\n\n"
            f"{safe_rec}\n"
        )

    # -- Markdown ------------------------------------------------------------

    def to_markdown(self, result: ReasoningResult, output_path: Optional[Path] = None) -> str:
        """Generate a Markdown report suitable for GitHub rendering.

        Parameters
        ----------
        result : ReasoningResult
            The reasoning chain output.
        output_path : Path or None
            If given, write the markdown to this path.

        Returns
        -------
        str
            Complete Markdown source.
        """
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        primary = result.primary_diagnosis
        confidence = f"{primary.confidence:.0%}" if primary else "N/A"
        name = primary.name.replace("_", " ").title() if primary else "None"

        md = f"""# FMCW Radar Diagnostic Report

**Sensor:** {self.sensor_id}  
**Frame ID:** {result.frame_id}  
**Generated:** {ts}  
**Author:** {self.author}

---

## Executive Summary

| Field | Value |
|-------|-------|
| Primary Diagnosis | **{name}** |
| Confidence | **{confidence}** |
| Human Review Required | {'⚠️ **Yes**' if result.requires_human_review else '✅ No'} |

{result.input_summary}

---

## Diagnostic Reasoning Chain

"""
        for step in result.steps:
            step_name = step.step_name.replace("_", " ").title()
            md += f"### {step_name}\n\n"
            for finding in step.findings:
                md += f"- {finding}\n"
            if step.active_hypotheses:
                hyps = ", ".join(f"`{h}`" for h in step.active_hypotheses)
                md += f"\n**Active:** {hyps}\n"
            if step.eliminated_hypotheses:
                elims = ", ".join(f"`{h}`" for h in step.eliminated_hypotheses)
                md += f"\n**Eliminated:** {elims}\n"
            md += "\n"

        md += "## Hypothesis Ranking\n\n"
        md += "| Rank | Hypothesis | Confidence |\n"
        md += "|------|-----------|------------|\n"
        for i, h in enumerate(result.hypotheses, 1):
            name_h = h.name.replace("_", " ").title()
            conf = f"{h.confidence:.1%}"
            if i == 1:
                name_h = f"**{name_h}**"
                conf = f"**{conf}**"
            md += f"| {i} | {name_h} | {conf} |\n"
        md += "\n"

        for h in result.hypotheses:
            name_h = h.name.replace("_", " ").title()
            md += f"### {name_h} ({h.confidence:.1%})\n\n"
            md += "**Evidence:**\n"
            for ev in h.evidence:
                md += f"- {ev}\n"
            md += "\n"

        if primary:
            md += "## Recommended Action\n\n"
            md += f"{primary.recommendation}\n\n"

        md += "---\n*Report generated by FMCW Diagnostic Agent*\n"

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(md, encoding="utf-8")
        return md


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from signal_pipeline.feature_extractor import ExtractionConfig, FeatureExtractor
    from signal_pipeline.fmcw_simulator import (
        FMCWSimulator, MultipathConfig, SimulationConfig, Target,
    )
    from .engineering_reasoner import EngineeringReasoner

    scfg = SimulationConfig(num_chirps=64)
    targets = [
        Target(range_m=3.0, velocity_mps=0.3, rcs_dbsm=12,
               micro_doppler={"frequency_hz": 1.2, "amplitude_mm": 0.5}),
    ]
    mp = MultipathConfig(enabled=True, path_delay_s=2e-9)
    sim = FMCWSimulator(scfg, targets, multipath=mp, seed=123)
    sig = sim.generate()

    extractor = FeatureExtractor(ExtractionConfig())
    rec = extractor.process(sig, frame_id=1, metadata={
        "scenario": "biosignal_monitoring",
        "room": "lab_A",
        "temp_c": 22.5,
        "multipath_enabled": True,
    })

    reasoner = EngineeringReasoner()
    reasoning_result = reasoner.reason(rec)

    reporter = ExpertReporter()
    latex = reporter.to_latex(reasoning_result)
    md = reporter.to_markdown(reasoning_result)

    print("=== LATEX (first 1500 chars) ===")
    print(latex[:1500])
    print("\n=== MARKDOWN ===")
    print(md)
