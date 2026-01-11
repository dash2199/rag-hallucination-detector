"""
Claim extraction and verification strategy.

Extracts atomic claims from generated text and verifies each claim
against the source documents using a combination of techniques.
"""

from typing import List, Optional, Dict, Any
import re
from loguru import logger

from hallucination_detector.strategies.base import BaseStrategy
from hallucination_detector.core.result import (
    StrategyResult,
    HallucinationSpan,
    HallucinationType,
    SeverityLevel,
)
from hallucination_detector.utils.text_processing import TextProcessor


class ClaimExtractionStrategy(BaseStrategy):
    """
    Detects hallucinations by extracting and verifying atomic claims.
    
    Uses a claim extraction approach to decompose generated text into
    verifiable factual statements, then checks each against sources.
    """
    
    # Claim extraction prompts (for T5/FLAN models)
    CLAIM_EXTRACTION_PROMPT = "Extract factual claims from this text: {text}"
    CLAIM_VERIFICATION_PROMPT = "Is this claim supported by the context? Claim: {claim} Context: {context}"
    
    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._text_processor = TextProcessor()
    
    @property
    def name(self) -> str:
        return "claim_extraction"
    
    def _load_models(self) -> None:
        """Load the claim extraction model."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        
        device = self._get_device()
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.models.claim_model
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.models.claim_model
        )
        
        if device == "cuda":
            self._model = self._model.cuda()
        elif device == "mps":
            self._model = self._model.to("mps")
        
        self._model.eval()
        logger.debug(f"Loaded claim extraction model on {device}")
    
    def detect(
        self,
        generated_text: str,
        source_documents: List[str],
        query: Optional[str] = None,
    ) -> StrategyResult:
        """
        Detect hallucinations by extracting and verifying claims.
        
        Process:
        1. Extract atomic claims from generated text
        2. For each claim, search for supporting evidence in sources
        3. Score claims based on evidence quality
        4. Flag unsupported claims as potential hallucinations
        """
        self._ensure_initialized()
        
        # Extract claims from generated text
        claims = self._extract_claims(generated_text)
        
        if not claims:
            return StrategyResult(
                strategy_name=self.name,
                hallucination_score=0.0,
                confidence=0.6,
                details={"note": "No factual claims extracted"},
            )
        
        # Concatenate sources for verification
        combined_sources = " ".join(source_documents)
        
        # Verify each claim
        spans = []
        claim_results = []
        
        for claim_info in claims:
            claim_text = claim_info["text"]
            
            # Verify claim
            verification = self._verify_claim(claim_text, combined_sources)
            
            claim_result = {
                "claim": claim_text,
                "verified": verification["is_supported"],
                "confidence": verification["confidence"],
                "evidence": verification.get("evidence"),
            }
            claim_results.append(claim_result)
            
            # Create span for unsupported claims
            if not verification["is_supported"]:
                # Try to find the claim in the original text
                start, end = self._find_claim_position(
                    claim_text, 
                    generated_text
                )
                
                severity = self._determine_severity(
                    claim_text,
                    verification["confidence"]
                )
                
                span = HallucinationSpan(
                    start_char=start,
                    end_char=end,
                    text=claim_text,
                    hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                    severity=severity,
                    confidence=verification["confidence"],
                    evidence={
                        "verification_score": verification.get("score", 0),
                        "partial_evidence": verification.get("evidence"),
                    },
                    explanation=f"Claim not adequately supported by source documents",
                )
                spans.append(span)
        
        # Calculate overall score
        supported_count = sum(1 for r in claim_results if r["verified"])
        total_claims = len(claim_results)
        
        hallucination_score = 1.0 - (supported_count / total_claims) if total_claims > 0 else 0.0
        
        # Adjust confidence based on claim count
        base_confidence = 0.7
        confidence = min(base_confidence + (total_claims * 0.02), 0.9)
        
        return StrategyResult(
            strategy_name=self.name,
            hallucination_score=hallucination_score,
            confidence=confidence,
            details={
                "total_claims": total_claims,
                "supported_claims": supported_count,
                "unsupported_claims": total_claims - supported_count,
                "claims": claim_results[:10],  # Limit for response size
            },
            spans=spans,
        )
    
    def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract atomic factual claims from text.
        
        Uses a combination of:
        1. Rule-based extraction for simple claims
        2. Model-based extraction for complex claims
        """
        claims = []
        
        # First, use rule-based extraction
        rule_claims = self._rule_based_claim_extraction(text)
        claims.extend(rule_claims)
        
        # Then use model for more sophisticated extraction
        model_claims = self._model_based_claim_extraction(text)
        claims.extend(model_claims)
        
        # Deduplicate
        seen = set()
        unique_claims = []
        for claim in claims:
            normalized = claim["text"].lower().strip()
            if normalized not in seen and len(normalized) > 10:
                seen.add(normalized)
                unique_claims.append(claim)
        
        return unique_claims
    
    def _rule_based_claim_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Extract claims using rule-based patterns."""
        claims = []
        
        # Split into sentences
        sentences = self._text_processor.split_into_sentences(text)
        
        # Patterns that indicate factual claims
        claim_patterns = [
            r'.*\b(?:is|are|was|were)\b.*',  # Being verbs
            r'.*\b(?:has|have|had)\b.*',      # Having verbs
            r'.*\b(?:increased|decreased|grew|fell|rose)\b.*',  # Change verbs
            r'.*\b(?:founded|established|created|invented)\b.*',  # Origin verbs
            r'.*\b(?:\d+(?:\.\d+)?)\s*(?:%|percent)\b.*',  # Percentages
            r'.*\b(?:\d{4})\b.*',  # Years
            r'.*\b(?:million|billion|thousand)\b.*',  # Large numbers
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            
            # Check if sentence matches claim patterns
            is_claim = any(
                re.match(pattern, sentence, re.IGNORECASE) 
                for pattern in claim_patterns
            )
            
            if is_claim:
                claims.append({
                    "text": sentence,
                    "type": "rule_based",
                })
        
        return claims
    
    def _model_based_claim_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Extract claims using the T5/FLAN model."""
        import torch
        
        claims = []
        
        # Chunk text if too long
        chunks = self._text_processor.chunk_text(text, chunk_size=400)
        
        for chunk in chunks[:5]:  # Limit chunks for efficiency
            prompt = f"List the factual claims in this text as separate sentences:\n{chunk}"
            
            try:
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                )
                
                if next(self._model.parameters()).is_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                elif str(next(self._model.parameters()).device) == "mps":
                    inputs = {k: v.to("mps") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=4,
                        early_stopping=True,
                    )
                
                decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Parse extracted claims
                for line in decoded.split('\n'):
                    line = line.strip()
                    # Remove numbering if present
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if len(line) > 15:
                        claims.append({
                            "text": line,
                            "type": "model_based",
                        })
                        
            except Exception as e:
                logger.warning(f"Model-based claim extraction failed: {e}")
        
        return claims
    
    def _verify_claim(
        self, 
        claim: str, 
        sources: str
    ) -> Dict[str, Any]:
        """
        Verify if a claim is supported by source documents.
        
        Uses the model to check claim support.
        """
        import torch
        
        # Truncate sources if too long
        max_source_len = 1000
        if len(sources) > max_source_len:
            # Try to find relevant portion
            sources = self._find_relevant_context(claim, sources, max_source_len)
        
        prompt = f"Based on the following context, is this claim true, false, or uncertain?\n\nContext: {sources}\n\nClaim: {claim}\n\nAnswer:"
        
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
            
            if next(self._model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif str(next(self._model.parameters()).device) == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=2,
                )
            
            answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
            
            # Parse the answer
            if "true" in answer or "yes" in answer or "supported" in answer:
                return {
                    "is_supported": True,
                    "confidence": 0.8,
                    "score": 0.9,
                    "evidence": answer,
                }
            elif "false" in answer or "no" in answer or "contradicted" in answer:
                return {
                    "is_supported": False,
                    "confidence": 0.85,
                    "score": 0.1,
                    "evidence": answer,
                }
            else:
                # Uncertain - treat as potentially unsupported
                return {
                    "is_supported": False,
                    "confidence": 0.5,
                    "score": 0.4,
                    "evidence": answer,
                }
                
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return {
                "is_supported": False,
                "confidence": 0.3,
                "score": 0.5,
                "evidence": f"Verification error: {e}",
            }
    
    def _find_relevant_context(
        self, 
        claim: str, 
        sources: str, 
        max_len: int
    ) -> str:
        """Find the most relevant portion of sources for a claim."""
        # Simple keyword-based relevance
        claim_words = set(claim.lower().split())
        
        # Split sources into paragraphs
        paragraphs = sources.split('\n\n')
        if not paragraphs:
            paragraphs = [sources]
        
        # Score paragraphs by relevance
        scored = []
        for para in paragraphs:
            para_words = set(para.lower().split())
            overlap = len(claim_words & para_words)
            scored.append((overlap, para))
        
        # Sort by relevance and take top
        scored.sort(reverse=True)
        
        result = []
        current_len = 0
        for _, para in scored:
            if current_len + len(para) <= max_len:
                result.append(para)
                current_len += len(para)
            else:
                break
        
        return ' '.join(result) if result else sources[:max_len]
    
    def _find_claim_position(
        self, 
        claim: str, 
        text: str
    ) -> tuple[int, int]:
        """Find approximate position of claim in original text."""
        # Try exact match first
        idx = text.find(claim)
        if idx >= 0:
            return idx, idx + len(claim)
        
        # Try fuzzy match
        claim_words = claim.lower().split()[:5]  # First 5 words
        search_pattern = r'\b' + r'\s+\w*\s*'.join(re.escape(w) for w in claim_words)
        
        match = re.search(search_pattern, text.lower())
        if match:
            return match.start(), match.end()
        
        # Default to full text
        return 0, len(text)
    
    def _determine_severity(self, claim: str, confidence: float) -> SeverityLevel:
        """Determine severity based on claim content and confidence."""
        # Check for numerical claims (higher severity)
        has_numbers = bool(re.search(r'\d+', claim))
        
        # Check for specific entities
        has_names = bool(re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', claim))
        
        if confidence > 0.8:
            if has_numbers or has_names:
                return SeverityLevel.HIGH
            return SeverityLevel.MEDIUM
        elif confidence > 0.6:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

