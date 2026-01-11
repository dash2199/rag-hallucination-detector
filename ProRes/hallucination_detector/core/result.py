"""
Data structures for hallucination detection results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class HallucinationType(Enum):
    """Types of hallucinations detected."""
    FACTUAL_ERROR = "factual_error"           # Contradicts source facts
    UNSUPPORTED_CLAIM = "unsupported_claim"   # Not grounded in sources
    ENTITY_FABRICATION = "entity_fabrication" # Made-up entities
    TEMPORAL_ERROR = "temporal_error"         # Incorrect temporal info
    NUMERICAL_ERROR = "numerical_error"       # Wrong numbers/quantities
    ATTRIBUTION_ERROR = "attribution_error"   # Misattributed information
    EXTRINSIC_INFO = "extrinsic_info"        # Info not from any source


class SeverityLevel(Enum):
    """Severity levels for detected hallucinations."""
    LOW = "low"           # Minor, might be acceptable
    MEDIUM = "medium"     # Notable deviation from sources
    HIGH = "high"         # Significant hallucination
    CRITICAL = "critical" # Completely fabricated/contradictory


@dataclass
class HallucinationSpan:
    """Represents a specific span of hallucinated text."""
    
    # Text span boundaries
    start_char: int
    end_char: int
    
    # The hallucinated text
    text: str
    
    # Hallucination classification
    hallucination_type: HallucinationType
    severity: SeverityLevel
    
    # Confidence score (0-1)
    confidence: float
    
    # Evidence for the detection
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Most relevant source passage (if any partial grounding exists)
    closest_source: Optional[str] = None
    source_similarity: Optional[float] = None
    
    # Explanation of why this was flagged
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
            "hallucination_type": self.hallucination_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "closest_source": self.closest_source,
            "source_similarity": self.source_similarity,
            "explanation": self.explanation,
        }


@dataclass
class StrategyResult:
    """Result from a single detection strategy."""
    
    strategy_name: str
    hallucination_score: float  # 0-1, higher = more likely hallucination
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    spans: List[HallucinationSpan] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "hallucination_score": self.hallucination_score,
            "confidence": self.confidence,
            "details": self.details,
            "spans": [s.to_dict() for s in self.spans],
        }


@dataclass
class DetectionResult:
    """Complete result from hallucination detection."""
    
    # The generated text that was analyzed
    generated_text: str
    
    # Source documents used for verification
    source_documents: List[str]
    
    # Overall hallucination score (0-1, higher = more hallucination)
    hallucination_score: float
    
    # Whether the text is classified as containing hallucinations
    is_hallucinated: bool
    
    # Confidence in the overall assessment
    confidence: float
    
    # Results from individual strategies
    strategy_results: Dict[str, StrategyResult] = field(default_factory=dict)
    
    # Detected hallucination spans
    hallucination_spans: List[HallucinationSpan] = field(default_factory=list)
    
    # Grounded spans (text properly supported by sources)
    grounded_spans: List[Dict[str, Any]] = field(default_factory=list)
    
    # Summary statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate statistics after initialization."""
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate summary statistics."""
        if not self.hallucination_spans:
            self.statistics = {
                "total_hallucination_spans": 0,
                "hallucination_coverage": 0.0,
                "severity_distribution": {},
                "type_distribution": {},
            }
            return
        
        total_chars = len(self.generated_text)
        hallucinated_chars = sum(
            span.end_char - span.start_char 
            for span in self.hallucination_spans
        )
        
        severity_counts = {}
        type_counts = {}
        for span in self.hallucination_spans:
            sev = span.severity.value
            typ = span.hallucination_type.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            type_counts[typ] = type_counts.get(typ, 0) + 1
        
        self.statistics = {
            "total_hallucination_spans": len(self.hallucination_spans),
            "hallucination_coverage": hallucinated_chars / total_chars if total_chars > 0 else 0,
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "avg_span_confidence": sum(s.confidence for s in self.hallucination_spans) / len(self.hallucination_spans),
        }
    
    def get_high_severity_spans(self) -> List[HallucinationSpan]:
        """Get only high and critical severity spans."""
        return [
            span for span in self.hallucination_spans
            if span.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "generated_text": self.generated_text,
            "source_count": len(self.source_documents),
            "hallucination_score": self.hallucination_score,
            "is_hallucinated": self.is_hallucinated,
            "confidence": self.confidence,
            "strategy_results": {
                k: v.to_dict() for k, v in self.strategy_results.items()
            },
            "hallucination_spans": [s.to_dict() for s in self.hallucination_spans],
            "statistics": self.statistics,
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "HALLUCINATION DETECTION REPORT",
            "=" * 60,
            f"Overall Score: {self.hallucination_score:.2%} (threshold: 50%)",
            f"Classification: {'⚠️  HALLUCINATED' if self.is_hallucinated else '✅ GROUNDED'}",
            f"Confidence: {self.confidence:.2%}",
            "",
            "Strategy Results:",
        ]
        
        for name, result in self.strategy_results.items():
            lines.append(f"  • {name}: {result.hallucination_score:.2%}")
        
        if self.hallucination_spans:
            lines.extend([
                "",
                f"Detected {len(self.hallucination_spans)} hallucination span(s):",
            ])
            for i, span in enumerate(self.hallucination_spans[:5], 1):
                lines.append(f"  {i}. [{span.severity.value.upper()}] \"{span.text[:50]}...\"")
                lines.append(f"      Type: {span.hallucination_type.value}")
            
            if len(self.hallucination_spans) > 5:
                lines.append(f"  ... and {len(self.hallucination_spans) - 5} more")
        
        lines.append("=" * 60)
        return "\n".join(lines)

