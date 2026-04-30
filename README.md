# FMCW Diagnostic Agent

> **复杂时序信号物理特征诊断 Agent** — 融合领域知识库与多步推理的 FMCW 雷达信号智能诊断系统

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
---

## 项目简介

在高频雷达与工业传感器网络中，每天产生海量的中频（IF）信号和相位数据。传统 FFT 阈值报警误报率极高，而依赖资深工程师逐一排查频谱图中的微小波峰异动，效率极低且无法规模化。

我构建了一个基于长上下文的**复杂时序信号（FMCW 连续波）物理特征诊断 Agent**，核心逻辑：
1. **底层信号脚本** — 将信号特征（相位提取后的异常波动点）转化为结构化文本序列
2. **工程推理 Agent** — 结合内置物理/数学知识库，进行多步逻辑推演（排除多径干扰、区分机械震动与生物体征微动）
3. **专家报告 Agent** — 输出含 LaTeX 公式推导的详细诊断报告

目前使用 **Hermes Agent** 作为编排工具，**DeepSeek V4 / MiMo V2.5** 作为底层推理模型。在实验室/设备监测场景下，日均不间断吞吐高频监测日志，**日均 Token 消耗可达 800 万**，大幅降低了对人工数据标注的依赖。

```
┌─────────────────────────────────────────────────────────────┐
│                  FMCW Diagnostic Agent                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: Signal Pipeline                                    │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │ FMCW     │ → │ Feature       │ → │ Text             │    │
│  │ Simulator│   │ Extractor     │   │ Serializer       │    │
│  │          │   │ (R-D FFT,     │   │ (structured NL   │    │
│  │ chirp +  │   │  CFAR, phase  │   │  text sequences) │    │
│  │ targets) │   │  unwrap, MD)  │   │                  │    │
│  └──────────┘   └──────────────┘   └────────┬─────────┘    │
│                                              │               │
│  Layer 2: Engineering Reasoning Agent        ↓               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Knowledge Base (radar physics, multipath model,      │  │
│  │    vibration vs biosignal, diagnostic decision tree)  │  │
│  │                                                       │  │
│  │  Step 1: Anomaly Classification                       │  │
│  │  Step 2: Physical Modelling (quantitative analysis)   │  │
│  │  Step 3: Interference Exclusion (environmental)       │  │
│  │  Step 4: Root-Cause Synthesis (ranked hypotheses)     │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                    │
│  Layer 2b: LLM Reasoning (optional, for anomalous frames)     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  OpenAI-compatible API → DeepSeek / MiMo model         │  │
│  │  System prompt: domain KB + 4-step reasoning chain     │  │
│  │  Structured JSON output → ReasoningResult              │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                    │
│  Layer 3: Expert Reporter                    ↓               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  LaTeX Report (with formula derivation) + Markdown    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心痛点 & 解决方案

| 痛点 | 解决方案 |
|------|---------|
| FFT 阈值报警误报率高 | 多步物理推理链排除环境干扰（多径、震动、温漂） |
| 资深工程师无法规模化排查 | Agent 自动吞吐，日均处理百万级 token |
| 信号异常根因依赖人工经验 | 显式知识库驱动，推理过程可审计 |
| 缺乏可解释的诊断报告 | 输出含 LaTeX 公式推导的结构化报告 |

---

## 快速开始

### 安装

```bash
git clone https://github.com/your-org/fmcw-diagnostic-agent.git
cd fmcw-diagnostic-agent
pip install -r requirements.txt
```

### 运行信号流水线

```bash
# 默认场景（正常双目标）
python -m signal_pipeline

# 多径干扰场景
python -m signal_pipeline --scenario multipath

# 生物体征微动场景
python -m signal_pipeline --scenario biosignal --raw-stats
```

### 运行示例 Demo

```bash
# 正常操作场景
python examples/demo_normal.py

# 多径干扰场景
python examples/demo_multipath.py

# 机械震动场景
python examples/demo_vibration.py

# 生物体征微动场景
python examples/demo_biosignal.py
```

每个 demo 会输出完整的推理链和诊断报告（Markdown + LaTeX）。

### 运行测试

```bash
pytest tests/ -v
```

---

## 项目结构

```
fmcw-diagnostic-agent/
├── signal_pipeline/           # Layer 1: 信号处理
│   ├── fmcw_simulator.py      #   FMCW chirp 信号模拟器
│   ├── feature_extractor.py   #   Range-Doppler FFT + 异常检测
│   ├── text_serializer.py     #   结构化文本序列化
│   └── __main__.py            #   CLI 入口
│
├── agents/                     # Layer 2 + 3: 推理 + 报告
│   ├── engineering_reasoner.py #   工程推理 Agent（规则引擎，多步推理链）
│   ├── llm_reasoner.py         #   LLM 推理 Agent（OpenAI 兼容 API）
│   ├── expert_reporter.py      #   专家报告 Agent（LaTeX + Markdown）
│   └── orchestrator.py         #   主控调度器
│
├── knowledge/                  # 领域知识库
│   ├── radar_physics.md        #   FMCW 物理基础
│   ├── multipath_model.md      #   多径反射判据
│   ├── vibration_vs_biosignal.md # 震动 vs 生物体征区分
│   └── diagnostic_decision_tree.md # 诊断决策树
│
├── examples/                   # 场景 Demo
│   ├── demo_normal.py
│   ├── demo_multipath.py
│   ├── demo_vibration.py
│   └── demo_biosignal.py
│
├── tests/                      # 单元测试
│   ├── test_simulator.py
│   ├── test_extractor.py
│   └── test_reasoner.py
│
├── reports/                    # 输出报告
│   └── sample_report.tex
│
├── requirements.txt
└── README.md
```

---

## 核心技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 信号模拟 | NumPy, SciPy | FMCW chirp 生成、多径叠加、相位噪声 |
| 特征提取 | NumPy FFT, CFAR | Range-Doppler 2D-FFT、相位解缠、异常检测 |
| 知识注入 | Markdown KB | 4 篇领域知识文档注入推理上下文 |
| 推理引擎 | Rule engine + LLM (DeepSeek/MiMo) | 规则引擎快速过滤 + LLM 深度推理（OpenAI 兼容 API） |
| 报告生成 | LaTeX + Markdown | 含公式推导的诊断报告 |
| 编排工具 | Hermes Agent | 多步推理链编排、批量帧处理、Token 统计 |

---

## 推理链示例

### 输入（结构化文本序列）
```
[FRAME 42] Timestamp: 2026-04-30T10:23:01Z
Sensor: FMCW Radar | Mode: Surveillance

TARGETS:
  [1] Range = 1.50 m, Velocity = +0.00 m/s, SNR = 18.2 dB

RANGE-DOMAIN ANOMALIES:
  · Bin 150 (1.503 m): Phase discontinuity 3.2°, SNR = 18.2 dB,
    type = periodic_displacement, confidence = 0.68

DOPPLER-DOMAIN ANOMALIES:
  · 1.20 Hz: possible_micro_doppler, amplitude deviation = 8.5 dB

METADATA:
  scenario: biosignal_monitoring
  subject: human_respiration
  temp_c: 36.5
```

### 输出（推理链）
```
Step 1 — Anomaly Classification:
  · Range bin 150: periodic displacement pattern detected
  · Doppler 1.20 Hz: possible_micro_doppler, deviation=8.5 dB
  Active: biosignal_micro_doppler, mechanical_vibration

Step 2 — Physical Modelling:
  · Phase jump 3.2° → estimated displacement = 0.133 mm
    (sub-mm, consistent with vibration)
  · Frequency 1.20 Hz in heartbeat band (0.8–3 Hz)

Step 3 — Interference Exclusion:
  · Clinical environment — low mechanical noise floor expected

Step 4 — Synthesis:
  [0.750] biosignal_micro_doppler  ← Primary
  [0.650] mechanical_vibration
  Recommendation: Continue vital-sign monitoring. Log trend.
```

---

## 扩展性设计

- **真实数据接入**：`FeatureExtractor.process()` 接受 `np.ndarray`，可将仿真替换为实际 ADC 数据
- **LLM 集成**：`EngineeringReasoner` 的推理链设计为明确的结构化输出，可直接作为 LLM prompt 上下文
- **批量处理**：`Orchestrator.process_batch()` 支持多帧连续处理
- **Token 统计**：内置 token 估算，支持"日均 800 万 token"场景演示

---

## 参考文献

- Texas Instruments, *The Fundamentals of Millimeter Wave Radar Sensors*, 2020
- V. C. Chen, *The Micro-Doppler Effect in Radar*, Artech House, 2019
- M. A. Richards, *Fundamentals of Radar Signal Processing*, McGraw-Hill, 2014

---

## License

MIT License
