"""
Text processing utilities for hallucination detection.
"""

import re
from typing import List, Dict, Any, Optional


class TextProcessor:
    """
    Text processing utilities for preparing text for analysis.
    
    Handles sentence splitting, chunking, normalization, and 
    other text preprocessing tasks.
    """
    
    # Sentence boundary patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    SENTENCE_PATTERN = re.compile(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+(?=[A-Z])|(?<=\n)'
    )
    
    # Abbreviations that shouldn't end sentences
    ABBREVIATIONS = {
        'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 
        'vs', 'etc', 'inc', 'ltd', 'co', 'corp',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'st', 'rd', 'th', 'ave', 'blvd',
    }
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize text processor.
        
        Args:
            chunk_size: Default chunk size for text chunking.
            chunk_overlap: Default overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        - Normalizes whitespace
        - Removes excessive newlines
        - Handles common encoding issues
        """
        if not text:
            return ""
        
        # Normalize unicode
        text = text.replace('\u2019', "'")
        text = text.replace('\u2018', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2014', '-')
        text = text.replace('\u2013', '-')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text to split.
            
        Returns:
            List of sentences.
        """
        if not text:
            return []
        
        # Use spacy-style sentence splitting
        sentences = self._smart_sentence_split(text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        return sentences
    
    def split_into_sentences_with_positions(
        self, 
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Split text into sentences with character positions.
        
        Returns:
            List of dicts with 'text', 'start', 'end' keys.
        """
        if not text:
            return []
        
        sentences = []
        current_pos = 0
        
        for sentence in self._smart_sentence_split(text):
            if not sentence.strip():
                continue
            
            # Find the actual position in original text
            start = text.find(sentence, current_pos)
            if start == -1:
                # Fallback: try normalized search
                start = current_pos
            
            end = start + len(sentence)
            
            sentences.append({
                "text": sentence,
                "start": start,
                "end": end,
            })
            
            current_pos = end
        
        return sentences
    
    def _smart_sentence_split(self, text: str) -> List[str]:
        """
        Smart sentence splitting that handles abbreviations.
        """
        # First pass: split on obvious boundaries
        parts = re.split(r'(?<=[.!?])\s+', text)
        
        sentences = []
        buffer = ""
        
        for part in parts:
            if buffer:
                part = buffer + " " + part
                buffer = ""
            
            # Check if this ends with an abbreviation
            words = part.split()
            if words:
                last_word = words[-1].rstrip('.').lower()
                if last_word in self.ABBREVIATIONS and part.endswith('.'):
                    buffer = part
                    continue
            
            sentences.append(part)
        
        if buffer:
            sentences.append(buffer)
        
        return sentences
    
    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Input text to chunk.
            chunk_size: Size of each chunk in characters.
            overlap: Overlap between chunks.
            
        Returns:
            List of text chunks.
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 100 chars
                search_start = max(end - 100, start)
                last_period = text.rfind('.', search_start, end)
                last_newline = text.rfind('\n', search_start, end)
                
                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract key phrases from text using simple TF-based approach.
        
        Args:
            text: Input text.
            top_n: Number of key phrases to return.
            
        Returns:
            List of key phrases.
        """
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'he', 'she', 'his',
            'her', 'which', 'who', 'whom', 'what', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
        }
        
        filtered = [w for w in words if w not in stopwords]
        
        # Count frequencies
        freq = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:top_n]]
    
    def calculate_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate word overlap between two texts.
        
        Returns:
            Jaccard similarity of word sets.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace to single spaces."""
        return ' '.join(text.split())
    
    def remove_citations(self, text: str) -> str:
        """Remove citation markers like [1], (Smith, 2020), etc."""
        # Numeric citations
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)
        
        # Author-year citations
        text = re.sub(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', '', text)
        
        return text.strip()

