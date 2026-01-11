"""
Entity-based hallucination detection strategy.

Detects hallucinations by extracting and verifying named entities
in generated text against entities present in source documents.
"""

import re
from typing import List, Optional, Dict, Set, Any, Tuple
from collections import defaultdict
from loguru import logger

from hallucination_detector.strategies.base import BaseStrategy
from hallucination_detector.core.result import (
    StrategyResult,
    HallucinationSpan,
    HallucinationType,
    SeverityLevel,
)


class EntityVerificationStrategy(BaseStrategy):
    """
    Detects hallucinations via named entity verification.
    
    Extracts entities (people, organizations, locations, dates, numbers)
    from generated text and verifies their presence in source documents.
    Fabricated entities indicate potential hallucination.
    """
    
    # Entity types that are critical for fact verification
    CRITICAL_ENTITY_TYPES = {
        "PERSON", "ORG", "GPE", "DATE", "TIME", 
        "MONEY", "PERCENT", "CARDINAL", "ORDINAL",
        "QUANTITY", "LAW", "EVENT", "PRODUCT"
    }
    
    # Entity types that are more flexible
    FLEXIBLE_ENTITY_TYPES = {
        "NORP", "FAC", "LOC", "LANGUAGE", "WORK_OF_ART"
    }
    
    def __init__(self, config):
        super().__init__(config)
        self._nlp = None
    
    @property
    def name(self) -> str:
        return "entity_verification"
    
    def _load_models(self) -> None:
        """Load the spaCy NER model."""
        import spacy
        
        try:
            self._nlp = spacy.load(self.config.models.ner_model)
        except OSError:
            # Fallback to smaller model
            logger.warning(f"Could not load {self.config.models.ner_model}, falling back to en_core_web_sm")
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Download if not available
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self._nlp = spacy.load("en_core_web_sm")
        
        logger.debug("Loaded spaCy NER model")
    
    def detect(
        self,
        generated_text: str,
        source_documents: List[str],
        query: Optional[str] = None,
    ) -> StrategyResult:
        """
        Detect hallucinations by verifying entities.
        
        Process:
        1. Extract entities from generated text
        2. Extract entities from all source documents
        3. Check which generated entities are not in sources
        4. Flag unverified critical entities as potential hallucinations
        """
        self._ensure_initialized()
        
        # Extract entities from generated text
        gen_entities = self._extract_entities(generated_text)
        
        if not gen_entities:
            return StrategyResult(
                strategy_name=self.name,
                hallucination_score=0.0,
                confidence=0.8,
                details={"note": "No entities found in generated text"},
            )
        
        # Extract entities from sources
        source_entities: Dict[str, Set[str]] = defaultdict(set)
        source_entity_variants: Dict[str, Set[str]] = defaultdict(set)
        
        for doc in source_documents:
            doc_entities = self._extract_entities(doc)
            for ent_type, entities in doc_entities.items():
                source_entities[ent_type].update(entities)
                # Also store normalized variants for fuzzy matching
                for ent in entities:
                    source_entity_variants[ent_type].add(self._normalize_entity(ent))
        
        # Verify generated entities
        spans = []
        verified_count = 0
        unverified_count = 0
        entity_details = []
        
        for ent_type, entities in gen_entities.items():
            is_critical = ent_type in self.CRITICAL_ENTITY_TYPES
            
            for entity_text, start, end in entities:
                # Check verification
                is_verified, match_type, closest_match = self._verify_entity(
                    entity_text,
                    ent_type,
                    source_entities,
                    source_entity_variants,
                )
                
                entity_details.append({
                    "text": entity_text,
                    "type": ent_type,
                    "verified": is_verified,
                    "match_type": match_type,
                    "closest_match": closest_match,
                })
                
                if is_verified:
                    verified_count += 1
                else:
                    unverified_count += 1
                    
                    # Create hallucination span for unverified critical entities
                    if is_critical:
                        severity = self._determine_severity(ent_type, entity_text)
                        
                        span = HallucinationSpan(
                            start_char=start,
                            end_char=end,
                            text=entity_text,
                            hallucination_type=self._get_hallucination_type(ent_type),
                            severity=severity,
                            confidence=0.75 if closest_match else 0.9,
                            evidence={
                                "entity_type": ent_type,
                                "found_in_sources": False,
                                "closest_match": closest_match,
                                "source_entities_of_type": list(source_entities.get(ent_type, set()))[:5],
                            },
                            closest_source=closest_match,
                            explanation=f"Entity '{entity_text}' ({ent_type}) not found in source documents" + 
                                       (f", closest match: '{closest_match}'" if closest_match else ""),
                        )
                        spans.append(span)
        
        # Calculate overall score
        total_entities = verified_count + unverified_count
        if total_entities == 0:
            hallucination_score = 0.0
            confidence = 0.5
        else:
            # Weight critical entities more heavily
            critical_unverified = len([
                s for s in spans 
                if s.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)
            ])
            
            base_score = unverified_count / total_entities
            critical_penalty = min(critical_unverified * 0.1, 0.3)
            hallucination_score = min(base_score + critical_penalty, 1.0)
            
            # Confidence based on entity count
            confidence = min(0.6 + (total_entities * 0.02), 0.95)
        
        return StrategyResult(
            strategy_name=self.name,
            hallucination_score=hallucination_score,
            confidence=confidence,
            details={
                "total_entities": total_entities,
                "verified_entities": verified_count,
                "unverified_entities": unverified_count,
                "entity_coverage": verified_count / total_entities if total_entities > 0 else 1.0,
                "entities": entity_details[:20],  # Limit for response size
            },
            spans=spans,
        )
    
    def _extract_entities(
        self, 
        text: str
    ) -> Dict[str, List[Tuple[str, int, int]]]:
        """
        Extract named entities from text.
        
        Returns:
            Dict mapping entity type to list of (text, start, end) tuples.
        """
        doc = self._nlp(text)
        entities: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
        
        for ent in doc.ents:
            entities[ent.label_].append((ent.text, ent.start_char, ent.end_char))
        
        # Also extract numbers and dates with regex for better coverage
        self._extract_regex_entities(text, entities)
        
        return entities
    
    def _extract_regex_entities(
        self, 
        text: str, 
        entities: Dict[str, List[Tuple[str, int, int]]]
    ) -> None:
        """Extract entities using regex patterns for better coverage."""
        # Numbers with units
        number_pattern = r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|dollars?|USD|million|billion|thousand|kg|km|miles?|years?|months?|days?)\b'
        for match in re.finditer(number_pattern, text, re.IGNORECASE):
            entities["QUANTITY"].append((match.group(0), match.start(), match.end()))
        
        # Years
        year_pattern = r'\b((?:19|20)\d{2})\b'
        for match in re.finditer(year_pattern, text):
            if (match.group(0), match.start(), match.end()) not in entities["DATE"]:
                entities["DATE"].append((match.group(0), match.start(), match.end()))
        
        # Percentages
        percent_pattern = r'\b(\d+(?:\.\d+)?)\s*%'
        for match in re.finditer(percent_pattern, text):
            entities["PERCENT"].append((match.group(0), match.start(), match.end()))
    
    def _verify_entity(
        self,
        entity_text: str,
        entity_type: str,
        source_entities: Dict[str, Set[str]],
        source_variants: Dict[str, Set[str]],
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Verify if entity exists in source documents.
        
        Returns:
            Tuple of (is_verified, match_type, closest_match)
        """
        # Exact match
        if entity_text in source_entities.get(entity_type, set()):
            return True, "exact", entity_text
        
        # Check across all entity types (entity might be labeled differently)
        for src_type, src_entities in source_entities.items():
            if entity_text in src_entities:
                return True, "cross_type", entity_text
        
        # Normalized match
        normalized = self._normalize_entity(entity_text)
        if normalized in source_variants.get(entity_type, set()):
            return True, "normalized", entity_text
        
        # Fuzzy match for names and organizations
        if entity_type in ("PERSON", "ORG", "GPE"):
            closest = self._find_closest_match(
                normalized,
                source_variants.get(entity_type, set())
            )
            if closest:
                return True, "fuzzy", closest
        
        # Check for partial match (entity might be part of a longer entity)
        for src_entities in source_entities.values():
            for src_ent in src_entities:
                if normalized in self._normalize_entity(src_ent) or \
                   self._normalize_entity(src_ent) in normalized:
                    return True, "partial", src_ent
        
        # Find closest non-matching entity for context
        closest_non_match = self._find_closest_match(
            normalized,
            source_variants.get(entity_type, set()),
            threshold=0.5  # Lower threshold for finding close but not matching
        )
        
        return False, "none", closest_non_match
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text for comparison."""
        # Lowercase, remove punctuation, normalize whitespace
        normalized = text.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _find_closest_match(
        self, 
        text: str, 
        candidates: Set[str],
        threshold: float = 0.8
    ) -> Optional[str]:
        """Find the closest matching entity using edit distance."""
        if not candidates:
            return None
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            score = self._similarity_score(text, candidate)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
        
        return best_match
    
    def _similarity_score(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings."""
        # Simple Jaccard similarity on character n-grams
        if not s1 or not s2:
            return 0.0
        
        def get_ngrams(s: str, n: int = 3) -> Set[str]:
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        ngrams1 = get_ngrams(s1)
        ngrams2 = get_ngrams(s2)
        
        if not ngrams1 or not ngrams2:
            return 1.0 if s1 == s2 else 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_severity(self, entity_type: str, entity_text: str) -> SeverityLevel:
        """Determine severity based on entity type."""
        # Numbers and dates are critical - can lead to serious misinformation
        if entity_type in ("CARDINAL", "MONEY", "PERCENT", "QUANTITY", "DATE", "TIME"):
            return SeverityLevel.HIGH
        
        # Person names are critical
        if entity_type == "PERSON":
            return SeverityLevel.HIGH
        
        # Organizations and locations are important
        if entity_type in ("ORG", "GPE", "FAC"):
            return SeverityLevel.MEDIUM
        
        return SeverityLevel.LOW
    
    def _get_hallucination_type(self, entity_type: str) -> HallucinationType:
        """Map entity type to hallucination type."""
        mapping = {
            "PERSON": HallucinationType.ENTITY_FABRICATION,
            "ORG": HallucinationType.ENTITY_FABRICATION,
            "GPE": HallucinationType.ENTITY_FABRICATION,
            "DATE": HallucinationType.TEMPORAL_ERROR,
            "TIME": HallucinationType.TEMPORAL_ERROR,
            "CARDINAL": HallucinationType.NUMERICAL_ERROR,
            "MONEY": HallucinationType.NUMERICAL_ERROR,
            "PERCENT": HallucinationType.NUMERICAL_ERROR,
            "QUANTITY": HallucinationType.NUMERICAL_ERROR,
        }
        return mapping.get(entity_type, HallucinationType.ENTITY_FABRICATION)

