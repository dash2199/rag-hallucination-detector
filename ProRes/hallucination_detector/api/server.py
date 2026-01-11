"""
FastAPI server for hallucination detection service.
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

from hallucination_detector import HallucinationDetector, DetectorConfig
from hallucination_detector.config import DetectionStrategy, AggregationMethod


# Request/Response Models
class DetectionRequest(BaseModel):
    """Request model for hallucination detection."""
    
    generated_text: str = Field(
        ...,
        description="The generated text to analyze for hallucinations",
        min_length=1,
    )
    source_documents: List[str] = Field(
        ...,
        description="List of source documents used for retrieval",
        min_items=1,
    )
    query: Optional[str] = Field(
        None,
        description="Original query that prompted the generation",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "generated_text": "Apple was founded by Steve Jobs in 1976 and is headquartered in Cupertino.",
                "source_documents": [
                    "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Apple was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne."
                ],
                "query": "When was Apple founded?",
            }
        }


class HallucinationSpanResponse(BaseModel):
    """Response model for a hallucination span."""
    
    start_char: int
    end_char: int
    text: str
    hallucination_type: str
    severity: str
    confidence: float
    explanation: str
    closest_source: Optional[str] = None


class DetectionResponse(BaseModel):
    """Response model for hallucination detection."""
    
    hallucination_score: float = Field(
        ...,
        description="Overall hallucination score (0-1, higher = more hallucination)",
    )
    is_hallucinated: bool = Field(
        ...,
        description="Whether the text is classified as containing hallucinations",
    )
    confidence: float = Field(
        ...,
        description="Confidence in the assessment",
    )
    hallucination_spans: List[HallucinationSpanResponse] = Field(
        default=[],
        description="Specific spans detected as hallucinations",
    )
    strategy_scores: dict = Field(
        default={},
        description="Scores from individual detection strategies",
    )
    statistics: dict = Field(
        default={},
        description="Summary statistics",
    )


class BatchDetectionRequest(BaseModel):
    """Request model for batch detection."""
    
    items: List[DetectionRequest] = Field(
        ...,
        description="List of items to process",
        min_items=1,
        max_items=100,
    )


class BatchDetectionResponse(BaseModel):
    """Response model for batch detection."""
    
    results: List[DetectionResponse]
    total_processed: int


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    strategies_loaded: List[str]


# Global detector instance
_detector: Optional[HallucinationDetector] = None


def get_detector() -> HallucinationDetector:
    """Get or create the detector instance."""
    global _detector
    if _detector is None:
        config = DetectorConfig(
            strategies=[
                DetectionStrategy.SEMANTIC_SIMILARITY,
                DetectionStrategy.NLI_ENTAILMENT,
                DetectionStrategy.ENTITY_VERIFICATION,
            ],
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            verbose=True,
        )
        _detector = HallucinationDetector(config)
        logger.info("Initialized hallucination detector")
    return _detector


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="RAG Hallucination Detection API",
        description="""
        Advanced hallucination detection for RAG (Retrieval-Augmented Generation) systems.
        
        This API analyzes generated text against source documents to detect:
        - Factual errors
        - Unsupported claims
        - Entity fabrications
        - Numerical/temporal errors
        
        ## Detection Strategies
        
        - **Semantic Similarity**: Measures embedding similarity between generated text and sources
        - **NLI Entailment**: Uses natural language inference to check logical support
        - **Entity Verification**: Verifies named entities exist in source documents
        - **Claim Extraction**: Extracts and verifies individual factual claims
        
        ## Usage
        
        Send a POST request to `/detect` with the generated text and source documents.
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize detector on startup."""
        logger.info("Starting hallucination detection API...")
        get_detector()  # Initialize detector
        logger.info("API ready")
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Check API health and loaded strategies."""
        detector = get_detector()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            strategies_loaded=[s.value for s in detector.config.strategies],
        )
    
    @app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
    async def detect_hallucinations(request: DetectionRequest):
        """
        Detect hallucinations in generated text.
        
        Analyzes the provided text against source documents using multiple
        detection strategies to identify potential hallucinations.
        """
        try:
            detector = get_detector()
            result = detector.detect(
                generated_text=request.generated_text,
                source_documents=request.source_documents,
                query=request.query,
            )
            
            return DetectionResponse(
                hallucination_score=result.hallucination_score,
                is_hallucinated=result.is_hallucinated,
                confidence=result.confidence,
                hallucination_spans=[
                    HallucinationSpanResponse(
                        start_char=span.start_char,
                        end_char=span.end_char,
                        text=span.text,
                        hallucination_type=span.hallucination_type.value,
                        severity=span.severity.value,
                        confidence=span.confidence,
                        explanation=span.explanation,
                        closest_source=span.closest_source,
                    )
                    for span in result.hallucination_spans
                ],
                strategy_scores={
                    name: res.hallucination_score
                    for name, res in result.strategy_results.items()
                },
                statistics=result.statistics,
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/detect/batch", response_model=BatchDetectionResponse, tags=["Detection"])
    async def detect_batch(request: BatchDetectionRequest, background_tasks: BackgroundTasks):
        """
        Process multiple detection requests in batch.
        
        More efficient for processing multiple items than individual requests.
        """
        try:
            detector = get_detector()
            results = []
            
            for item in request.items:
                result = detector.detect(
                    generated_text=item.generated_text,
                    source_documents=item.source_documents,
                    query=item.query,
                )
                
                results.append(DetectionResponse(
                    hallucination_score=result.hallucination_score,
                    is_hallucinated=result.is_hallucinated,
                    confidence=result.confidence,
                    hallucination_spans=[
                        HallucinationSpanResponse(
                            start_char=span.start_char,
                            end_char=span.end_char,
                            text=span.text,
                            hallucination_type=span.hallucination_type.value,
                            severity=span.severity.value,
                            confidence=span.confidence,
                            explanation=span.explanation,
                            closest_source=span.closest_source,
                        )
                        for span in result.hallucination_spans
                    ],
                    strategy_scores={
                        name: res.hallucination_score
                        for name, res in result.strategy_results.items()
                    },
                    statistics=result.statistics,
                ))
            
            return BatchDetectionResponse(
                results=results,
                total_processed=len(results),
            )
            
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/strategies", tags=["Info"])
    async def list_strategies():
        """List available detection strategies and their descriptions."""
        return {
            "strategies": [
                {
                    "name": "semantic_similarity",
                    "description": "Measures semantic embedding similarity between generated text and source documents",
                    "weight": 0.3,
                },
                {
                    "name": "nli_entailment",
                    "description": "Uses NLI to check if generated claims are entailed by sources",
                    "weight": 0.4,
                },
                {
                    "name": "entity_verification",
                    "description": "Verifies named entities (people, orgs, dates) exist in sources",
                    "weight": 0.2,
                },
                {
                    "name": "claim_extraction",
                    "description": "Extracts and verifies individual factual claims",
                    "weight": 0.1,
                },
            ]
        }
    
    return app


def main():
    """Run the API server."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()

