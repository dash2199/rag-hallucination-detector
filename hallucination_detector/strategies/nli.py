"""
Natural Language Inference (NLI) based hallucination detection.

Uses entailment models to determine if generated claims are logically
supported by, contradicted by, or neutral to the source documents.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from loguru import logger

from hallucination_detector.strategies.base import BaseStrategy
from hallucination_detector.core.result import (
    StrategyResult,
    HallucinationSpan,
    HallucinationType,
    SeverityLevel,
)
from hallucination_detector.utils.text_processing import TextProcessor


class NLIEntailmentStrategy(BaseStrategy):
    """
    Detects hallucinations using Natural Language Inference.
    
    For each generated sentence (hypothesis), checks if it's entailed by
    any source passage (premise). Contradictions strongly indicate
    hallucination; neutral results suggest unsupported claims.
    """
    
    # NLI label mappings
    LABEL_ENTAILMENT = "entailment"
    LABEL_CONTRADICTION = "contradiction"
    LABEL_NEUTRAL = "neutral"
    
    def __init__(self, config):
        super().__init__(config)
        self._classifier = None
        self._text_processor = TextProcessor()
    
    @property
    def name(self) -> str:
        return "nli_entailment"
    
    def _load_models(self) -> None:
        """Load the NLI model."""
        from transformers import pipeline
        
        device = self._get_device()
        device_id = 0 if device == "cuda" else -1 if device == "cpu" else device
        
        self._classifier = pipeline(
            "zero-shot-classification",
            model=self.config.models.nli_model,
            device=device_id,
        )
        logger.debug(f"Loaded NLI model on {device}")
    
    def detect(
        self,
        generated_text: str,
        source_documents: List[str],
        query: Optional[str] = None,
    ) -> StrategyResult:
        """
        Detect hallucinations using NLI-based entailment checking.
        
        Process:
        1. Extract claims/sentences from generated text
        2. For each claim, check entailment against source passages
        3. Flag contradictions and neutral statements
        """
        self._ensure_initialized()
        
        # Split into sentences
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
        
        # Prepare source passages
        source_passages = []
        for doc in source_documents:
            passages = self._text_processor.chunk_text(
                doc,
                chunk_size=300,  # Shorter for NLI
                overlap=50,
            )
            source_passages.extend(passages)
        
        if not source_passages:
            return StrategyResult(
                strategy_name=self.name,
                hallucination_score=1.0,
                confidence=0.5,
                details={"note": "No source passages for verification"},
            )
        
        # Analyze each sentence
        spans = []
        sentence_results = []
        
        for sent_info in sentences_with_pos:
            sentence = sent_info["text"]
            
            # Skip very short sentences
            if len(sentence.split()) < 4:
                continue
            
            result = self._check_entailment(sentence, source_passages)
            sentence_results.append(result)
            
            # Determine if this is a hallucination
            if result["status"] == "contradiction":
                span = HallucinationSpan(
                    start_char=sent_info["start"],
                    end_char=sent_info["end"],
                    text=sentence,
                    hallucination_type=HallucinationType.FACTUAL_ERROR,
                    severity=SeverityLevel.CRITICAL,
                    confidence=result["confidence"],
                    evidence={
                        "nli_status": result["status"],
                        "entailment_score": result["entailment_score"],
                        "contradiction_score": result["contradiction_score"],
                        "best_premise": result["best_premise"][:200] if result["best_premise"] else None,
                    },
                    closest_source=result["best_premise"][:200] if result["best_premise"] else None,
                    explanation=f"Contradicts source: {result['best_premise'][:100]}..." if result["best_premise"] else "Contradicts source information",
                )
                spans.append(span)
                
            elif result["status"] == "neutral" and result["neutral_score"] > 0.7:
                span = HallucinationSpan(
                    start_char=sent_info["start"],
                    end_char=sent_info["end"],
                    text=sentence,
                    hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                    severity=SeverityLevel.MEDIUM,
                    confidence=result["confidence"] * 0.8,
                    evidence={
                        "nli_status": result["status"],
                        "neutral_score": result["neutral_score"],
                        "entailment_score": result["entailment_score"],
                    },
                    explanation="Claim not supported by any source passage",
                )
                spans.append(span)
        
        # Calculate overall score
        hallucination_score = self._calculate_overall_score(sentence_results)
        confidence = self._calculate_confidence(sentence_results)
        
        return StrategyResult(
            strategy_name=self.name,
            hallucination_score=hallucination_score,
            confidence=confidence,
            details={
                "sentences_analyzed": len(sentence_results),
                "contradictions": sum(1 for r in sentence_results if r["status"] == "contradiction"),
                "neutral": sum(1 for r in sentence_results if r["status"] == "neutral"),
                "entailed": sum(1 for r in sentence_results if r["status"] == "entailment"),
                "avg_entailment_score": float(np.mean([r["entailment_score"] for r in sentence_results])) if sentence_results else 0,
            },
            spans=spans,
        )
    
    def _check_entailment(
        self, 
        hypothesis: str, 
        premises: List[str]
    ) -> Dict[str, Any]:
        """
        Check if hypothesis is entailed by any premise.
        
        Returns the best entailment result across all premises.
        """
        best_result = {
            "status": "neutral",
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 1.0,
            "confidence": 0.0,
            "best_premise": None,
        }
        
        # Use the NLI model with premise-hypothesis pairs
        # We frame this as checking if hypothesis can be classified given premise
        candidate_labels = [
            "This is definitely true based on the context.",
            "This contradicts the context.",
            "This cannot be determined from the context."
        ]
        
        label_mapping = {
            candidate_labels[0]: "entailment",
            candidate_labels[1]: "contradiction", 
            candidate_labels[2]: "neutral",
        }
        
        for premise in premises[:10]:  # Limit for efficiency
            # Combine premise and hypothesis for classification
            combined = f"Context: {premise}\n\nStatement: {hypothesis}"
            
            try:
                result = self._classifier(
                    combined,
                    candidate_labels,
                    multi_label=False,
                )
                
                # Extract scores
                scores = dict(zip(result["labels"], result["scores"]))
                
                entail_score = scores.get(candidate_labels[0], 0)
                contra_score = scores.get(candidate_labels[1], 0)
                neutral_score = scores.get(candidate_labels[2], 0)
                
                # Check if this premise is better
                if entail_score > best_result["entailment_score"]:
                    best_result = {
                        "status": "entailment" if entail_score > self.config.thresholds.entailment_threshold else ("contradiction" if contra_score > 0.5 else "neutral"),
                        "entailment_score": entail_score,
                        "contradiction_score": contra_score,
                        "neutral_score": neutral_score,
                        "confidence": max(entail_score, contra_score, neutral_score),
                        "best_premise": premise,
                    }
                    
                    # Early exit if we find strong entailment
                    if entail_score > 0.85:
                        break
                        
                # Check for contradiction (important to catch)
                if contra_score > 0.6 and contra_score > best_result["contradiction_score"]:
                    best_result = {
                        "status": "contradiction",
                        "entailment_score": entail_score,
                        "contradiction_score": contra_score,
                        "neutral_score": neutral_score,
                        "confidence": contra_score,
                        "best_premise": premise,
                    }
                    
            except Exception as e:
                logger.warning(f"NLI check failed for premise: {e}")
                continue
        
        return best_result
    
    def _calculate_overall_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall hallucination score from sentence results."""
        if not results:
            return 0.0
        
        # Weight: contradictions are worst, neutral is medium, entailment is good
        weights = {
            "contradiction": 1.0,
            "neutral": 0.5,
            "entailment": 0.0,
        }
        
        total_score = sum(weights.get(r["status"], 0.5) for r in results)
        return total_score / len(results)
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the overall result."""
        if not results:
            return 0.0
        
        avg_confidence = np.mean([r["confidence"] for r in results])
        return float(avg_confidence)

