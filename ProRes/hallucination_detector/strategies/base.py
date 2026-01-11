"""
Base class for hallucination detection strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from hallucination_detector.config import DetectorConfig
from hallucination_detector.core.result import StrategyResult


class BaseStrategy(ABC):
    """
    Abstract base class for hallucination detection strategies.
    
    Each strategy implements a specific approach to detecting hallucinations
    by comparing generated text against source documents.
    """
    
    def __init__(self, config: DetectorConfig):
        """
        Initialize the strategy.
        
        Args:
            config: Global detector configuration.
        """
        self.config = config
        self._initialized = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass
    
    @abstractmethod
    def detect(
        self,
        generated_text: str,
        source_documents: List[str],
        query: Optional[str] = None,
    ) -> StrategyResult:
        """
        Detect hallucinations using this strategy.
        
        Args:
            generated_text: The generated text to analyze.
            source_documents: Source documents for verification.
            query: Optional original query.
            
        Returns:
            StrategyResult with detection scores and spans.
        """
        pass
    
    def _ensure_initialized(self) -> None:
        """Ensure models are loaded before detection."""
        if not self._initialized:
            self._load_models()
            self._initialized = True
    
    def _load_models(self) -> None:
        """Load required models. Override in subclasses."""
        pass
    
    def _get_device(self) -> str:
        """Determine the computation device."""
        device = self.config.models.device
        
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

