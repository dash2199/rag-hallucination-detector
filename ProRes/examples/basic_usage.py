"""
Basic usage examples for the RAG Hallucination Detection System.

This script demonstrates how to use the detector to identify
hallucinations in RAG-generated content.
"""

from hallucination_detector import HallucinationDetector, DetectorConfig
from hallucination_detector.config import DetectionStrategy, AggregationMethod


def example_basic_detection():
    """Basic example of hallucination detection."""
    print("=" * 60)
    print("Example 1: Basic Hallucination Detection")
    print("=" * 60)
    
    # Initialize detector with default settings
    detector = HallucinationDetector()
    
    # Source document (from retrieval)
    source_documents = [
        """
        Apple Inc. is an American multinational technology company headquartered 
        in Cupertino, California. Apple was founded on April 1, 1976, by Steve Jobs, 
        Steve Wozniak, and Ronald Wayne to develop and sell Wozniak's Apple I 
        personal computer. The company's second computer, the Apple II, became 
        a best seller. Apple went public in 1980 with a market valuation of $1.2 billion.
        In 2023, Apple's revenue was $383.3 billion.
        """
    ]
    
    # Generated text (from LLM) - contains some hallucinations
    generated_text = """
    Apple was founded by Steve Jobs alone in 1975 and quickly became a leader 
    in personal computing. The company is headquartered in San Francisco and 
    had revenue of $500 billion in 2023. Apple's first product, the Macintosh, 
    revolutionized the industry.
    """
    
    # Detect hallucinations
    result = detector.detect(
        generated_text=generated_text,
        source_documents=source_documents,
        query="Tell me about Apple Inc.",
    )
    
    # Print results
    print(result.summary())
    print()


def example_custom_config():
    """Example with custom configuration."""
    print("=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    # Custom configuration
    config = DetectorConfig(
        strategies=[
            DetectionStrategy.SEMANTIC_SIMILARITY,
            DetectionStrategy.ENTITY_VERIFICATION,
        ],
        aggregation_method=AggregationMethod.MAX_SCORE,
        strategy_weights={
            DetectionStrategy.SEMANTIC_SIMILARITY: 0.6,
            DetectionStrategy.ENTITY_VERIFICATION: 0.4,
        },
        enable_span_detection=True,
        verbose=True,
    )
    
    detector = HallucinationDetector(config)
    
    source_documents = [
        """
        The Great Wall of China is a series of fortifications that were built 
        across the historical northern borders of ancient Chinese states. 
        Construction began in the 7th century BC, with the most well-known 
        sections built during the Ming Dynasty (1368-1644). The wall stretches 
        approximately 21,196 kilometers (13,171 miles).
        """
    ]
    
    # Text with factual errors
    generated_text = """
    The Great Wall of China is 50,000 kilometers long and was built entirely 
    during the Han Dynasty around 500 AD. It was constructed to protect against 
    Mongolian invasions and took 1,000 years to complete. The wall is located 
    in southern China.
    """
    
    result = detector.detect(
        generated_text=generated_text,
        source_documents=source_documents,
    )
    
    print(f"Hallucination Score: {result.hallucination_score:.2%}")
    print(f"Classification: {'HALLUCINATED' if result.is_hallucinated else 'GROUNDED'}")
    print(f"\nDetected {len(result.hallucination_spans)} hallucination spans:")
    
    for i, span in enumerate(result.hallucination_spans, 1):
        print(f"\n  {i}. [{span.severity.value.upper()}] \"{span.text[:60]}...\"")
        print(f"     Type: {span.hallucination_type.value}")
        print(f"     Confidence: {span.confidence:.2%}")
        print(f"     Explanation: {span.explanation}")
    
    print()


def example_grounded_text():
    """Example with well-grounded text (no hallucinations)."""
    print("=" * 60)
    print("Example 3: Well-Grounded Text (No Hallucinations)")
    print("=" * 60)
    
    detector = HallucinationDetector()
    
    source_documents = [
        """
        Python is a high-level, general-purpose programming language. 
        Its design philosophy emphasizes code readability with the use of 
        significant indentation. Python is dynamically typed and garbage-collected. 
        It was created by Guido van Rossum and first released in 1991.
        """
    ]
    
    # Well-grounded generated text
    generated_text = """
    Python is a high-level programming language created by Guido van Rossum. 
    It was first released in 1991 and emphasizes code readability. 
    Python features dynamic typing and automatic garbage collection.
    """
    
    result = detector.detect(
        generated_text=generated_text,
        source_documents=source_documents,
    )
    
    print(f"Hallucination Score: {result.hallucination_score:.2%}")
    print(f"Classification: {'HALLUCINATED' if result.is_hallucinated else 'GROUNDED'}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Hallucination Spans: {len(result.hallucination_spans)}")
    
    print("\nStrategy Results:")
    for name, res in result.strategy_results.items():
        print(f"  • {name}: {res.hallucination_score:.2%}")
    
    print()


def example_detailed_analysis():
    """Example showing detailed analysis output."""
    print("=" * 60)
    print("Example 4: Detailed Analysis")
    print("=" * 60)
    
    detector = HallucinationDetector()
    
    source_documents = [
        """
        SpaceX (Space Exploration Technologies Corp.) is an American spacecraft 
        manufacturer and space transport company. SpaceX was founded in 2002 
        by Elon Musk with the goal of reducing space transportation costs to 
        enable the colonization of Mars. The company is headquartered in 
        Hawthorne, California. SpaceX developed the Falcon 9 rocket and the 
        Dragon spacecraft. The Falcon 9 had its first successful orbital launch 
        on June 4, 2010.
        """
    ]
    
    generated_text = """
    SpaceX was founded by Elon Musk and Jeff Bezos in 2005 to make space travel 
    more affordable. The company, based in Houston, Texas, has developed several 
    rockets including the Falcon 10 and the Phoenix spacecraft. SpaceX's first 
    successful rocket launch was in 2008.
    """
    
    result = detector.detect(
        generated_text=generated_text,
        source_documents=source_documents,
    )
    
    # Full JSON output
    print("Full Detection Result:")
    print(result.to_json())
    print()


def example_batch_processing():
    """Example of batch processing multiple texts."""
    print("=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)
    
    detector = HallucinationDetector()
    
    # Multiple items to process
    items = [
        {
            "generated_text": "The Eiffel Tower is 300 meters tall and located in Paris, France.",
            "source_documents": [
                "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall."
            ],
        },
        {
            "generated_text": "Mount Everest is 8,849 meters tall, making it the tallest mountain.",
            "source_documents": [
                "Mount Everest is Earth's highest mountain above sea level, at 8,848.86 m (29,031.7 ft)."
            ],
        },
        {
            "generated_text": "The Amazon River is the longest river in the world at 7,000 km.",
            "source_documents": [
                "The Nile is the longest river, at about 6,650 km. The Amazon is second at about 6,400 km."
            ],
        },
    ]
    
    results = detector.detect_batch(items)
    
    print(f"Processed {len(results)} items:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Item {i}:")
        print(f"  Score: {result.hallucination_score:.2%}")
        print(f"  Hallucinated: {result.is_hallucinated}")
        print(f"  Spans: {len(result.hallucination_spans)}")
        print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG HALLUCINATION DETECTION SYSTEM - EXAMPLES")
    print("=" * 60 + "\n")
    
    # Note: Running these examples requires model downloads
    # on first execution (may take a few minutes)
    
    try:
        example_basic_detection()
        example_custom_config()
        example_grounded_text()
        example_detailed_analysis()
        example_batch_processing()
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: First run may require downloading models.")

