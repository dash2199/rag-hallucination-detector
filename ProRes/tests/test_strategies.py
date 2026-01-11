"""
Unit tests for detection strategies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from hallucination_detector.config import DetectorConfig
from hallucination_detector.strategies.base import BaseStrategy
from hallucination_detector.core.result import StrategyResult


class TestBaseStrategy:
    """Tests for BaseStrategy."""
    
    def test_abstract_methods(self):
        """Ensure base strategy cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseStrategy(DetectorConfig())
    
    def test_get_device_auto(self):
        """Test device detection."""
        
        class ConcreteStrategy(BaseStrategy):
            @property
            def name(self):
                return "test"
            
            def detect(self, generated_text, source_documents, query=None):
                return StrategyResult("test", 0.0, 1.0)
        
        config = DetectorConfig()
        config.models.device = "cpu"
        
        strategy = ConcreteStrategy(config)
        assert strategy._get_device() == "cpu"


class TestSemanticSimilarityStrategy:
    """Tests for SemanticSimilarityStrategy."""
    
    @pytest.fixture
    def mock_strategy(self):
        """Create strategy with mocked encoder."""
        with patch('hallucination_detector.strategies.semantic.SentenceTransformer') as mock_st:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = np.random.rand(3, 384)
            mock_st.return_value = mock_encoder
            
            from hallucination_detector.strategies.semantic import SemanticSimilarityStrategy
            config = DetectorConfig()
            strategy = SemanticSimilarityStrategy(config)
            strategy._initialized = True
            strategy._encoder = mock_encoder
            
            yield strategy
    
    def test_detect_returns_result(self, mock_strategy):
        result = mock_strategy.detect(
            generated_text="Test sentence one. Test sentence two.",
            source_documents=["Source content here."],
        )
        
        assert isinstance(result, StrategyResult)
        assert result.strategy_name == "semantic_similarity"
    
    def test_cosine_similarity_matrix(self, mock_strategy):
        a = np.array([[1, 0, 0], [0, 1, 0]])
        b = np.array([[1, 0, 0], [0, 0, 1]])
        
        sim = mock_strategy._cosine_similarity_matrix(a, b)
        
        assert sim.shape == (2, 2)
        assert abs(sim[0, 0] - 1.0) < 0.01  # Same vector
        assert abs(sim[0, 1]) < 0.01  # Orthogonal
    
    def test_determine_severity(self, mock_strategy):
        from hallucination_detector.core.result import SeverityLevel
        
        assert mock_strategy._determine_severity(0.2) == SeverityLevel.CRITICAL
        assert mock_strategy._determine_severity(0.4) == SeverityLevel.HIGH
        assert mock_strategy._determine_severity(0.5) == SeverityLevel.MEDIUM
        assert mock_strategy._determine_severity(0.6) == SeverityLevel.LOW


class TestNLIEntailmentStrategy:
    """Tests for NLIEntailmentStrategy."""
    
    @pytest.fixture
    def mock_strategy(self):
        """Create strategy with mocked classifier."""
        with patch('hallucination_detector.strategies.nli.pipeline') as mock_pipeline:
            mock_classifier = MagicMock()
            mock_classifier.return_value = {
                "labels": ["entailment", "neutral", "contradiction"],
                "scores": [0.7, 0.2, 0.1],
            }
            mock_pipeline.return_value = mock_classifier
            
            from hallucination_detector.strategies.nli import NLIEntailmentStrategy
            config = DetectorConfig()
            strategy = NLIEntailmentStrategy(config)
            strategy._initialized = True
            strategy._classifier = mock_classifier
            
            yield strategy
    
    def test_calculate_overall_score(self, mock_strategy):
        results = [
            {"status": "entailment", "confidence": 0.9},
            {"status": "neutral", "confidence": 0.7},
            {"status": "contradiction", "confidence": 0.8},
        ]
        
        score = mock_strategy._calculate_overall_score(results)
        
        # (0 + 0.5 + 1) / 3 = 0.5
        assert 0.4 < score < 0.6


class TestEntityVerificationStrategy:
    """Tests for EntityVerificationStrategy."""
    
    @pytest.fixture
    def mock_strategy(self):
        """Create strategy with mocked spaCy."""
        with patch('hallucination_detector.strategies.entity.spacy') as mock_spacy:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.ents = []
            mock_nlp.return_value = mock_doc
            mock_spacy.load.return_value = mock_nlp
            
            from hallucination_detector.strategies.entity import EntityVerificationStrategy
            config = DetectorConfig()
            strategy = EntityVerificationStrategy(config)
            strategy._initialized = True
            strategy._nlp = mock_nlp
            
            yield strategy
    
    def test_normalize_entity(self, mock_strategy):
        result = mock_strategy._normalize_entity("Apple Inc.")
        assert result == "apple inc"
    
    def test_similarity_score_identical(self, mock_strategy):
        score = mock_strategy._similarity_score("test", "test")
        assert score == 1.0
    
    def test_similarity_score_different(self, mock_strategy):
        score = mock_strategy._similarity_score("apple", "banana")
        assert 0 <= score < 1
    
    def test_determine_severity(self, mock_strategy):
        from hallucination_detector.core.result import SeverityLevel
        
        assert mock_strategy._determine_severity("CARDINAL", "100") == SeverityLevel.HIGH
        assert mock_strategy._determine_severity("PERSON", "John") == SeverityLevel.HIGH
        assert mock_strategy._determine_severity("ORG", "Company") == SeverityLevel.MEDIUM


class TestClaimExtractionStrategy:
    """Tests for ClaimExtractionStrategy."""
    
    @pytest.fixture
    def mock_strategy(self):
        """Create strategy with mocked model."""
        with patch('hallucination_detector.strategies.claim.AutoModelForSeq2SeqLM') as mock_model, \
             patch('hallucination_detector.strategies.claim.AutoTokenizer') as mock_tokenizer:
            
            mock_tok_instance = MagicMock()
            mock_tok_instance.return_value = {"input_ids": MagicMock()}
            mock_tok_instance.decode.return_value = "This is a claim."
            mock_tokenizer.from_pretrained.return_value = mock_tok_instance
            
            mock_model_instance = MagicMock()
            mock_model_instance.parameters.return_value = iter([MagicMock(is_cuda=False)])
            mock_model_instance.generate.return_value = [MagicMock()]
            mock_model.from_pretrained.return_value = mock_model_instance
            
            from hallucination_detector.strategies.claim import ClaimExtractionStrategy
            config = DetectorConfig()
            strategy = ClaimExtractionStrategy(config)
            strategy._initialized = True
            strategy._model = mock_model_instance
            strategy._tokenizer = mock_tok_instance
            
            yield strategy
    
    def test_rule_based_claim_extraction(self, mock_strategy):
        text = "Apple was founded in 1976. The company grew rapidly."
        claims = mock_strategy._rule_based_claim_extraction(text)
        
        assert len(claims) > 0
        assert all("text" in c for c in claims)
    
    def test_find_claim_position(self, mock_strategy):
        claim = "test claim"
        text = "This is a test claim in text."
        
        start, end = mock_strategy._find_claim_position(claim, text)
        
        assert start >= 0
        assert end > start
    
    def test_determine_severity(self, mock_strategy):
        from hallucination_detector.core.result import SeverityLevel
        
        # Claim with numbers - higher severity
        severity = mock_strategy._determine_severity("Revenue was $100 million", 0.9)
        assert severity in (SeverityLevel.HIGH, SeverityLevel.MEDIUM)
        
        # Simple claim - lower severity
        severity = mock_strategy._determine_severity("It is good", 0.5)
        assert severity == SeverityLevel.LOW

