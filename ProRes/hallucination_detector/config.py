"""
Configuration management for the hallucination detection system.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class DetectionStrategy(Enum):
    """Available hallucination detection strategies."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    NLI_ENTAILMENT = "nli_entailment"
    ENTITY_VERIFICATION = "entity_verification"
    CLAIM_EXTRACTION = "claim_extraction"
    ENSEMBLE = "ensemble"


class AggregationMethod(Enum):
    """Methods for aggregating multi-signal detection results."""
    WEIGHTED_AVERAGE = "weighted_average"
    MAX_SCORE = "max_score"
    VOTING = "voting"
    LEARNED = "learned"


@dataclass
class ModelConfig:
    """Configuration for ML models used in detection."""
    
    # Semantic similarity model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # NLI model for entailment checking
    nli_model: str = "facebook/bart-large-mnli"
    
    # Named Entity Recognition model
    ner_model: str = "en_core_web_trf"
    
    # Claim extraction model (T5-based)
    claim_model: str = "google/flan-t5-base"
    
    # Device configuration
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    
    # Batch processing
    batch_size: int = 32
    
    # Model caching
    cache_dir: Optional[str] = None


@dataclass
class ThresholdConfig:
    """Threshold configuration for hallucination classification."""
    
    # Semantic similarity threshold (below = potential hallucination)
    semantic_threshold: float = 0.65
    
    # NLI entailment threshold (below = contradiction/neutral)
    entailment_threshold: float = 0.7
    
    # Entity coverage threshold (percentage of entities that must be grounded)
    entity_coverage_threshold: float = 0.8
    
    # Claim verification threshold
    claim_threshold: float = 0.6
    
    # Final hallucination score threshold
    hallucination_threshold: float = 0.5


@dataclass
class DetectorConfig:
    """Main configuration for the hallucination detector."""
    
    # Detection strategies to use
    strategies: List[DetectionStrategy] = field(
        default_factory=lambda: [
            DetectionStrategy.SEMANTIC_SIMILARITY,
            DetectionStrategy.NLI_ENTAILMENT,
            DetectionStrategy.ENTITY_VERIFICATION,
        ]
    )
    
    # Aggregation method for multi-signal results
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    
    # Strategy weights for weighted aggregation
    strategy_weights: dict = field(default_factory=lambda: {
        DetectionStrategy.SEMANTIC_SIMILARITY: 0.3,
        DetectionStrategy.NLI_ENTAILMENT: 0.4,
        DetectionStrategy.ENTITY_VERIFICATION: 0.2,
        DetectionStrategy.CLAIM_EXTRACTION: 0.1,
    })
    
    # Model configuration
    models: ModelConfig = field(default_factory=ModelConfig)
    
    # Threshold configuration
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    
    # Enable span-level detection (identifies specific hallucinated spans)
    enable_span_detection: bool = True
    
    # Chunk size for processing long documents
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Logging verbosity
    verbose: bool = False
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.strategies:
            raise ValueError("At least one detection strategy must be specified")
        
        weight_sum = sum(
            self.strategy_weights.get(s, 0) for s in self.strategies
        )
        if abs(weight_sum - 1.0) > 0.01 and self.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            # Normalize weights
            for s in self.strategies:
                if s in self.strategy_weights:
                    self.strategy_weights[s] /= weight_sum

