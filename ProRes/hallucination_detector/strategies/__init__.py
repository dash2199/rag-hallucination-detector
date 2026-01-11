"""Detection strategy implementations."""

from hallucination_detector.strategies.base import BaseStrategy
from hallucination_detector.strategies.semantic import SemanticSimilarityStrategy
from hallucination_detector.strategies.nli import NLIEntailmentStrategy
from hallucination_detector.strategies.entity import EntityVerificationStrategy
from hallucination_detector.strategies.claim import ClaimExtractionStrategy

__all__ = [
    "BaseStrategy",
    "SemanticSimilarityStrategy",
    "NLIEntailmentStrategy",
    "EntityVerificationStrategy",
    "ClaimExtractionStrategy",
]

