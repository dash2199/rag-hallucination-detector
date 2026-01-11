"""
Semantic similarity-based hallucination detection strategy.

Detects hallucinations by measuring semantic similarity between generated
text segments and source documents using dense embeddings.
"""

import numpy as np
from typing import List, Optional, Tuple
from loguru import logger

from hallucination_detector.strategies.base import BaseStrategy
from hallucination_detector.core.result import (
    StrategyResult,
    HallucinationSpan,
    HallucinationType,
    SeverityLevel,
)
from hallucination_detector.utils.text_processing import TextProcessor


class SemanticSimilarityStrategy(BaseStrategy):
    """
    Detects hallucinations via semantic embedding similarity.
    
    Segments generated text into sentences/chunks and measures their
    semantic similarity to source documents. Low similarity indicates
    potential hallucination.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._encoder = None
        self._text_processor = TextProcessor()
    
    @property
    def name(self) -> str:
        return "semantic_similarity"
    
    def _load_models(self) -> None:
        """Load the sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        
        device = self._get_device()
        self._encoder = SentenceTransformer(
            self.config.models.embedding_model,
            device=device,
        )
        logger.debug(f"Loaded embedding model on {device}")
    
    def detect(
        self,
        generated_text: str,
        source_documents: List[str],
        query: Optional[str] = None,
    ) -> StrategyResult:
        """
        Detect hallucinations using semantic similarity.
        
        Process:
        1. Split generated text into sentences
        2. Encode all sentences and source documents
        3. For each generated sentence, find max similarity to any source
        4. Flag sentences with low similarity as potential hallucinations
        """
        self._ensure_initialized()
        
        # Split generated text into sentences with positions
        sentences_with_pos = self._text_processor.split_into_sentences_with_positions(
            generated_text
        )
        
        if not sentences_with_pos:
            return StrategyResult(
                strategy_name=self.name,
                hallucination_score=0.0,
                confidence=1.0,
                details={"note": "No sentences to analyze"},
            )
        
        # Prepare source chunks
        source_chunks = []
        for doc in source_documents:
            chunks = self._text_processor.chunk_text(
                doc,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            source_chunks.extend(chunks)
        
        if not source_chunks:
            return StrategyResult(
                strategy_name=self.name,
                hallucination_score=1.0,
                confidence=0.5,
                details={"note": "No source content for comparison"},
            )
        
        # Encode everything
        sentences = [s["text"] for s in sentences_with_pos]
        
        sentence_embeddings = self._encoder.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.config.models.batch_size,
        )
        
        source_embeddings = self._encoder.encode(
            source_chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.config.models.batch_size,
        )
        
        # Calculate similarities
        similarities = self._cosine_similarity_matrix(
            sentence_embeddings, source_embeddings
        )
        
        # Analyze each sentence
        spans = []
        sentence_scores = []
        
        for i, (sent_info, sent_sims) in enumerate(zip(sentences_with_pos, similarities)):
            max_sim = float(np.max(sent_sims))
            best_source_idx = int(np.argmax(sent_sims))
            sentence_scores.append(max_sim)
            
            # Check against threshold
            if max_sim < self.config.thresholds.semantic_threshold:
                severity = self._determine_severity(max_sim)
                
                span = HallucinationSpan(
                    start_char=sent_info["start"],
                    end_char=sent_info["end"],
                    text=sent_info["text"],
                    hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                    severity=severity,
                    confidence=1.0 - max_sim,  # Higher confidence when less similar
                    evidence={
                        "max_similarity": max_sim,
                        "threshold": self.config.thresholds.semantic_threshold,
                    },
                    closest_source=source_chunks[best_source_idx][:200],
                    source_similarity=max_sim,
                    explanation=f"Low semantic similarity ({max_sim:.2%}) to all source documents",
                )
                spans.append(span)
        
        # Calculate overall score
        avg_similarity = np.mean(sentence_scores) if sentence_scores else 1.0
        hallucination_score = 1.0 - avg_similarity
        
        # Calculate what percentage of sentences are potentially hallucinated
        hallucinated_ratio = len(spans) / len(sentences_with_pos)
        
        return StrategyResult(
            strategy_name=self.name,
            hallucination_score=hallucination_score,
            confidence=0.85,  # Semantic similarity is a solid signal
            details={
                "avg_similarity": float(avg_similarity),
                "min_similarity": float(np.min(sentence_scores)) if sentence_scores else 0,
                "max_similarity": float(np.max(sentence_scores)) if sentence_scores else 0,
                "sentences_analyzed": len(sentences_with_pos),
                "sentences_flagged": len(spans),
                "hallucinated_ratio": hallucinated_ratio,
            },
            spans=spans,
        )
    
    def _cosine_similarity_matrix(
        self, 
        embeddings_a: np.ndarray, 
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity matrix between two sets of embeddings."""
        # Normalize embeddings
        norm_a = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-9)
        norm_b = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-9)
        
        # Compute similarity matrix
        return np.dot(norm_a, norm_b.T)
    
    def _determine_severity(self, similarity: float) -> SeverityLevel:
        """Determine severity based on similarity score."""
        if similarity < 0.3:
            return SeverityLevel.CRITICAL
        elif similarity < 0.45:
            return SeverityLevel.HIGH
        elif similarity < 0.55:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def get_sentence_similarities(
        self,
        generated_text: str,
        source_documents: List[str],
    ) -> List[Tuple[str, float, str]]:
        """
        Get similarity scores for each sentence.
        
        Useful for debugging and analysis.
        
        Returns:
            List of (sentence, similarity, closest_source) tuples.
        """
        self._ensure_initialized()
        
        sentences_with_pos = self._text_processor.split_into_sentences_with_positions(
            generated_text
        )
        sentences = [s["text"] for s in sentences_with_pos]
        
        source_chunks = []
        for doc in source_documents:
            chunks = self._text_processor.chunk_text(doc)
            source_chunks.extend(chunks)
        
        if not sentences or not source_chunks:
            return []
        
        sent_emb = self._encoder.encode(sentences, convert_to_numpy=True)
        src_emb = self._encoder.encode(source_chunks, convert_to_numpy=True)
        
        similarities = self._cosine_similarity_matrix(sent_emb, src_emb)
        
        results = []
        for i, sent in enumerate(sentences):
            max_sim = float(np.max(similarities[i]))
            best_idx = int(np.argmax(similarities[i]))
            results.append((sent, max_sim, source_chunks[best_idx]))
        
        return results

