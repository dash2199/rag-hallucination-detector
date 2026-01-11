"""Core detection modules."""

from hallucination_detector.core.detector import HallucinationDetector
from hallucination_detector.core.result import DetectionResult, HallucinationSpan

__all__ = ["HallucinationDetector", "DetectionResult", "HallucinationSpan"]

