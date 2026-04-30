"""Signal Pipeline — FMCW radar signal simulation, feature extraction, and text serialization.

This package provides the bottom layer of the FMCW Diagnostic Agent:
converting raw radar signals into structured natural language sequences
ready for consumption by the engineering reasoning agent.
"""

__version__ = "0.1.0"
__all__ = [
    "FMCWSimulator",
    "SimulationConfig",
    "FeatureExtractor",
    "ExtractionConfig",
    "TextSerializer",
    "SignalRecord",
    "FeatureRecord",
]

from .fmcw_simulator import FMCWSimulator, SimulationConfig
from .feature_extractor import ExtractionConfig, FeatureExtractor, FeatureRecord
from .text_serializer import SignalRecord, TextSerializer
