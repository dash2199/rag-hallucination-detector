"""
RAG-Aware Hallucination Detection System

A sophisticated multi-signal analysis system for detecting hallucinations
in RAG (Retrieval-Augmented Generation) outputs by verifying factual 
grounding against source documents.
"""

from hallucination_detector.core.detector import HallucinationDetector
from hallucination_detector.core.result import DetectionResult, HallucinationSpan
from hallucination_detector.config import DetectorConfig

__version__ = "1.0.0"
__all__ = [
    "HallucinationDetector",
    "DetectionResult", 
    "HallucinationSpan",
    "DetectorConfig",
]

