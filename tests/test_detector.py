"""
Unit tests for the hallucination detector.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from hallucination_detector import HallucinationDetector, DetectorConfig
from hallucination_detector.config import DetectionStrategy, AggregationMethod
from hallucination_detector.core.result import (
    DetectionResult,
    StrategyResult,
    HallucinationSpan,
    HallucinationType,
    SeverityLevel,
)
from hallucination_detector.utils.text_processing import TextProcessor


class TestTextProcessor:
    """Tests for TextProcessor utility."""
    
    def test_preprocess_normalizes_whitespace(self):
        processor = TextProcessor()
        text = "Hello   world\n\ntest"
        result = processor.preprocess(text)
        assert result == "Hello world test"
    
    def test_preprocess_handles_unicode(self):
        processor = TextProcessor()
        text = "It\u2019s a \u201ctest\u201d"
        result = processor.preprocess(text)
        assert "'" in result
        assert '"' in result
    
    def test_split_into_sentences(self):
        processor = TextProcessor()
        text = "First sentence. Second sentence. Third one."
        sentences = processor.split_into_sentences(text)
        assert len(sentences) >= 2
    
    def test_split_handles_abbreviations(self):
        processor = TextProcessor()
        text = "Dr. Smith went to the store. He bought milk."
        sentences = processor.split_into_sentences(text)
        # Should not split on "Dr."
        assert any("Dr." in s for s in sentences)
    
    def test_chunk_text(self):
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        text = "A" * 250
        chunks = processor.chunk_text(text)
        assert len(chunks) >= 2
        # Each chunk should be roughly chunk_size
        assert all(len(c) <= 120 for c in chunks)
    
    def test_chunk_text_short_input(self):
        processor = TextProcessor(chunk_size=100)
        text = "Short text"
        chunks = processor.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"
    
    def test_calculate_overlap(self):
        processor = TextProcessor()
        text1 = "hello world foo bar"
        text2 = "hello world baz qux"
        overlap = processor.calculate_overlap(text1, text2)
        assert 0 < overlap < 1
    
    def test_extract_key_phrases(self):
        processor = TextProcessor()
        text = "Machine learning is a subset of artificial intelligence. Machine learning uses algorithms."
        phrases = processor.extract_key_phrases(text, top_n=5)
        assert "machine" in phrases or "learning" in phrases


class TestDetectorConfig:
    """Tests for DetectorConfig."""
    
    def test_default_config(self):
        config = DetectorConfig()
        assert len(config.strategies) > 0
        assert config.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE
    
    def test_validate_normalizes_weights(self):
        config = DetectorConfig(
            strategies=[
                DetectionStrategy.SEMANTIC_SIMILARITY,
                DetectionStrategy.NLI_ENTAILMENT,
            ],
            strategy_weights={
                DetectionStrategy.SEMANTIC_SIMILARITY: 1.0,
                DetectionStrategy.NLI_ENTAILMENT: 1.0,
            },
        )
        config.validate()
        # Weights should be normalized
        total = sum(config.strategy_weights.values())
        assert abs(total - 1.0) < 0.1
    
    def test_validate_raises_on_empty_strategies(self):
        config = DetectorConfig(strategies=[])
        with pytest.raises(ValueError):
            config.validate()


class TestDetectionResult:
    """Tests for DetectionResult."""
    
    def test_result_calculates_statistics(self):
        spans = [
            HallucinationSpan(
                start_char=0,
                end_char=10,
                text="test span",
                hallucination_type=HallucinationType.FACTUAL_ERROR,
                severity=SeverityLevel.HIGH,
                confidence=0.9,
            )
        ]
        
        result = DetectionResult(
            generated_text="test span and more",
            source_documents=["source"],
            hallucination_score=0.5,
            is_hallucinated=True,
            confidence=0.8,
            hallucination_spans=spans,
        )
        
        assert result.statistics["total_hallucination_spans"] == 1
        assert "severity_distribution" in result.statistics
    
    def test_result_to_dict(self):
        result = DetectionResult(
            generated_text="test",
            source_documents=["source"],
            hallucination_score=0.3,
            is_hallucinated=False,
            confidence=0.9,
        )
        
        d = result.to_dict()
        assert "hallucination_score" in d
        assert "is_hallucinated" in d
        assert d["is_hallucinated"] == False
    
    def test_result_to_json(self):
        result = DetectionResult(
            generated_text="test",
            source_documents=["source"],
            hallucination_score=0.3,
            is_hallucinated=False,
            confidence=0.9,
        )
        
        json_str = result.to_json()
        assert '"hallucination_score"' in json_str
    
    def test_get_high_severity_spans(self):
        spans = [
            HallucinationSpan(
                start_char=0, end_char=5, text="low",
                hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                severity=SeverityLevel.LOW, confidence=0.5,
            ),
            HallucinationSpan(
                start_char=10, end_char=15, text="high",
                hallucination_type=HallucinationType.FACTUAL_ERROR,
                severity=SeverityLevel.HIGH, confidence=0.9,
            ),
        ]
        
        result = DetectionResult(
            generated_text="low span high span",
            source_documents=["source"],
            hallucination_score=0.5,
            is_hallucinated=True,
            confidence=0.7,
            hallucination_spans=spans,
        )
        
        high_spans = result.get_high_severity_spans()
        assert len(high_spans) == 1
        assert high_spans[0].text == "high"


class TestHallucinationDetector:
    """Tests for HallucinationDetector."""
    
    @pytest.fixture
    def mock_detector(self):
        """Create a detector with mocked strategies."""
        with patch('hallucination_detector.core.detector.SemanticSimilarityStrategy') as mock_sem, \
             patch('hallucination_detector.core.detector.NLIEntailmentStrategy') as mock_nli, \
             patch('hallucination_detector.core.detector.EntityVerificationStrategy') as mock_ent:
            
            # Configure mocks
            for mock_cls in [mock_sem, mock_nli, mock_ent]:
                mock_instance = MagicMock()
                mock_instance.detect.return_value = StrategyResult(
                    strategy_name="mock",
                    hallucination_score=0.3,
                    confidence=0.8,
                )
                mock_cls.return_value = mock_instance
            
            config = DetectorConfig(
                strategies=[
                    DetectionStrategy.SEMANTIC_SIMILARITY,
                    DetectionStrategy.NLI_ENTAILMENT,
                    DetectionStrategy.ENTITY_VERIFICATION,
                ],
            )
            detector = HallucinationDetector(config)
            yield detector
    
    def test_detect_empty_text(self, mock_detector):
        result = mock_detector.detect(
            generated_text="",
            source_documents=["source"],
        )
        assert result.hallucination_score == 0.0
        assert result.is_hallucinated == False
    
    def test_detect_no_sources(self, mock_detector):
        result = mock_detector.detect(
            generated_text="Some generated text",
            source_documents=[],
        )
        assert result.hallucination_score == 1.0
        assert result.is_hallucinated == True
    
    def test_detect_returns_result(self, mock_detector):
        result = mock_detector.detect(
            generated_text="Generated text content",
            source_documents=["Source document content"],
        )
        
        assert isinstance(result, DetectionResult)
        assert 0 <= result.hallucination_score <= 1
        assert isinstance(result.is_hallucinated, bool)
    
    def test_detect_includes_metadata(self, mock_detector):
        result = mock_detector.detect(
            generated_text="Test text",
            source_documents=["Source"],
            query="Test query",
        )
        
        assert "processing_time_seconds" in result.metadata
        assert "strategies_used" in result.metadata
        assert result.metadata["query"] == "Test query"


class TestStrategyResult:
    """Tests for StrategyResult."""
    
    def test_strategy_result_to_dict(self):
        result = StrategyResult(
            strategy_name="test_strategy",
            hallucination_score=0.5,
            confidence=0.8,
            details={"key": "value"},
        )
        
        d = result.to_dict()
        assert d["strategy_name"] == "test_strategy"
        assert d["hallucination_score"] == 0.5
        assert "key" in d["details"]


class TestHallucinationSpan:
    """Tests for HallucinationSpan."""
    
    def test_span_to_dict(self):
        span = HallucinationSpan(
            start_char=0,
            end_char=10,
            text="test text",
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            severity=SeverityLevel.HIGH,
            confidence=0.9,
            explanation="Test explanation",
        )
        
        d = span.to_dict()
        assert d["text"] == "test text"
        assert d["hallucination_type"] == "factual_error"
        assert d["severity"] == "high"


class TestAggregation:
    """Tests for result aggregation methods."""
    
    def test_weighted_average_aggregation(self):
        # Test that weighted average works correctly
        config = DetectorConfig(
            strategies=[DetectionStrategy.SEMANTIC_SIMILARITY],
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
        )
        
        with patch('hallucination_detector.core.detector.SemanticSimilarityStrategy') as mock:
            mock_instance = MagicMock()
            mock_instance.detect.return_value = StrategyResult(
                strategy_name="semantic_similarity",
                hallucination_score=0.6,
                confidence=1.0,
            )
            mock.return_value = mock_instance
            
            detector = HallucinationDetector(config)
            result = detector.detect("test", ["source"])
            
            # With single strategy at 100% confidence, score should match
            assert abs(result.hallucination_score - 0.6) < 0.1


# Integration test (requires actual models, skipped by default)
@pytest.mark.skip(reason="Requires model downloads")
class TestIntegration:
    """Integration tests requiring actual models."""
    
    def test_full_detection_pipeline(self):
        detector = HallucinationDetector()
        
        result = detector.detect(
            generated_text="Apple was founded in 1976 by Steve Jobs.",
            source_documents=[
                "Apple Inc. was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne."
            ],
        )
        
        assert isinstance(result, DetectionResult)
        assert result.confidence > 0

