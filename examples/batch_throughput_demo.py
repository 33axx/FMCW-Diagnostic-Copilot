#!/usr/bin/env python3
"""
Batch Throughput Demo — 批量信号吞吐与 Token 消耗演示
======================================================
模拟 IoT 传感器连续 24 小时高频监测场景，展示系统的：

1. 信号吞吐量（帧/秒）
2. Token 消耗估算（知识库注入 + 特征序列化 + LLM 推理）
3. 异常自动识别与根因推理
4. 终端日志输出（用于提交截图的证明材料）

运行方式（使用 LLM 推理）::

    export MIMO_API_KEY="sk-..."
    cd fmcw1
    python examples/batch_throughput_demo.py --hours 1 --llm

运行方式（仅规则引擎，无需 API key）::

    python examples/batch_throughput_demo.py --hours 1 --no-llm

预期输出:
  - 终端实时日志（可截图用于申请表单的「使用证明」）
  - reports/ 目录下每帧的诊断报告（Markdown + LaTeX）
  - 最终统计摘要
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_pipeline.fmcw_simulator import (
    FMCWSimulator,
    MultipathConfig,
    SimulationConfig,
    Target,
)
from signal_pipeline.feature_extractor import ExtractionConfig, FeatureExtractor
from signal_pipeline.text_serializer import TextSerializer
from agents.engineering_reasoner import EngineeringReasoner, KnowledgeBase
from agents.expert_reporter import ExpertReporter

# Conditionally import LLMReasoner
try:
    from agents.llm_reasoner import LLMConfig, LLMReasoner

    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║      FMCW Diagnostic Agent — Batch Throughput Demo         ║
║      复杂时序信号物理特征诊断 Agent · 批量吞吐测试           ║
╚══════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FMCW Diagnostic Agent — Batch Throughput Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/batch_throughput_demo.py --hours 1 --no-llm
  export MIMO_API_KEY="sk-..."
  python examples/batch_throughput_demo.py --hours 24 --llm
        """,
    )
    parser.add_argument(
        "--hours", type=float, default=0.1,
        help="Simulated monitoring duration in hours (default: 0.1 = 6 min).",
    )
    parser.add_argument(
        "--frame-interval-s", type=float, default=0.2,
        help="Seconds between frames (default: 0.2 = 5 fps).",
    )
    parser.add_argument(
        "--llm", action="store_true", default=False,
        help="Use LLM-powered reasoning (requires API key).",
    )
    parser.add_argument(
        "--no-llm", action="store_true", default=False,
        help="Force rule-based reasoning (no API key needed).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports"),
        help="Directory for diagnostic reports.",
    )
    parser.add_argument(
        "--scenario", choices=["normal", "multipath", "vibration", "biosignal"],
        default="biosignal",
        help="Simulation scenario.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed.",
    )
    parser.add_argument(
        "--no-reports", action="store_true",
        help="Skip writing individual frame reports (faster).",
    )
    args = parser.parse_args()

    use_llm = args.llm and not args.no_llm

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("demo")

    print(BANNER)
    print(f"  Scenario:      {args.scenario}")
    print(f"  Duration:      {args.hours:.1f} hours")
    print(f"  Frame interval: {args.frame_interval_s:.1f} s")
    print(f"  LLM reasoning: {'ON' if use_llm else 'OFF (rule engine)'}")
    print(f"  Output dir:    {args.output_dir.resolve()}")
    print()

    # Build simulation
    from signal_pipeline.__main__ import _build_scenario as build_scenario

    scenario = build_scenario(args.scenario)
    sim_cfg = scenario["sim_config"]
    if sim_cfg.num_chirps > 64:
        sim_cfg = SimulationConfig(
            f_start_hz=sim_cfg.f_start_hz,
            f_stop_hz=sim_cfg.f_stop_hz,
            chirp_duration_s=sim_cfg.chirp_duration_s,
            num_chirps=64,  # fewer chirps for speed
            sample_rate_hz=sim_cfg.sample_rate_hz,
        )

    sim = FMCWSimulator(
        config=sim_cfg,
        targets=scenario["targets"],
        multipath=scenario["multipath"],
        seed=args.seed,
    )
    extractor = FeatureExtractor(ExtractionConfig())
    serializer = TextSerializer()
    reporter = ExpertReporter()

    # Build reasoner
    if use_llm and _HAS_LLM:
        api_key = os.environ.get("MIMO_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        reasoner: object = LLMReasoner(
            config=LLMConfig(api_key=api_key),
            rule_fallback=True,
        )
    else:
        if use_llm and not _HAS_LLM:
            log.warning("openai not installed, falling back to rule engine")
        reasoner = EngineeringReasoner()

    num_frames = max(1, int(args.hours * 3600 / args.frame_interval_s))
    log.info("Starting batch: %d frames over %.1f hours\n", num_frames, args.hours)

    # ------------------------------------------------------------------
    # Batch loop
    # ------------------------------------------------------------------

    t_start = time.perf_counter()
    anomaly_count = 0
    total_tokens = 0

    for frame_id in range(1, num_frames + 1):
        t_frame = time.perf_counter()

        # 1. Generate + extract
        if_signal = sim.generate()
        record = extractor.process(
            if_signal,
            frame_id=frame_id,
            metadata={
                **scenario["metadata"],
                "frame_interval_s": args.frame_interval_s,
            },
        )

        # 2. Reason
        if use_llm and _HAS_LLM:
            feature_text = serializer.serialize(record)
            result = reasoner.reason(record, feature_text=feature_text)  # type: ignore[union-attr]
        else:
            result = reasoner.reason(record)  # type: ignore[union-attr]

        # 3. Report (every 10th frame, or if anomaly)
        if not args.no_reports and (
            frame_id % 10 == 0
            or (result.primary_diagnosis and result.primary_diagnosis.name != "nominal_operation")
        ):
            reporter.to_markdown(
                result,
                args.output_dir / f"frame_{frame_id:05d}.md",
            )

        if result.primary_diagnosis and result.primary_diagnosis.name != "nominal_operation":
            anomaly_count += 1

        # Token tracking
        if hasattr(reasoner, "total_tokens"):
            total_tokens = reasoner.total_tokens  # type: ignore[union-attr]
        elif hasattr(reasoner, "_total_tokens_estimated"):
            total_tokens = reasoner._total_tokens_estimated  # type: ignore[union-attr]

        elapsed_frame_ms = (time.perf_counter() - t_frame) * 1000

        # Log every frame or every Nth
        if frame_id % 10 == 0 or frame_id == 1:
            diag = result.primary_diagnosis.name if result.primary_diagnosis else "NONE"
            conf = result.primary_diagnosis.confidence if result.primary_diagnosis else 0
            log.info(
                "Frame %5d/%d | %6.1f ms | diag: %-25s (%.0f%%) | tokens: %s | anomalies: %d",
                frame_id,
                num_frames,
                elapsed_frame_ms,
                diag,
                conf * 100,
                f"{total_tokens:,}",
                anomaly_count,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    total_elapsed_s = time.perf_counter() - t_start
    throughput_fps = num_frames / total_elapsed_s if total_elapsed_s > 0 else 0

    print()
    print("=" * 60)
    print("  BATCH COMPLETE")
    print("=" * 60)
    print(f"  Frames processed:       {num_frames}")
    print(f"  Total elapsed:          {total_elapsed_s:.1f} s")
    print(f"  Throughput:             {throughput_fps:.1f} fps")
    print(f"  Avg per frame:          {total_elapsed_s/num_frames*1000:.1f} ms")
    print(f"  Anomalies detected:     {anomaly_count} ({anomaly_count/num_frames*100:.1f}%)")

    # Token estimation for 24h projection
    if use_llm and _HAS_LLM and hasattr(reasoner, "stats"):
        print(f"\n  {reasoner.stats()}")  # type: ignore[union-attr]
        # Project to 24 hours
        daily_est = int(total_tokens * 24 / args.hours) if args.hours > 0 else 0
        print(f"  24h projection:         ~{daily_est:,} tokens/day")
    else:
        # Rule engine token estimate
        kb_size = len(KnowledgeBase().full_context())
        est_per_frame = (kb_size // 4) + 1500 + 1500  # chars → tokens
        daily_frames = int(24 * 3600 / args.frame_interval_s)
        daily_est = est_per_frame * daily_frames
        print(f"\n  Rule engine token estimate (KB + features per frame):")
        print(f"    Per frame:  ~{est_per_frame:,} tokens")
        print(f"    Per day:    ~{daily_est:,} tokens ({daily_frames} frames)")

    # Print for screenshot
    print(f"\n  Reports: {args.output_dir.resolve()}")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
