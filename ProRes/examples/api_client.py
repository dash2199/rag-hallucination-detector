"""
Example client for the Hallucination Detection API.

Shows how to interact with the REST API for hallucination detection.
"""

import requests
import json
from typing import List, Optional


class HallucinationDetectorClient:
    """Client for the Hallucination Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API server.
        """
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def detect(
        self,
        generated_text: str,
        source_documents: List[str],
        query: Optional[str] = None,
    ) -> dict:
        """
        Detect hallucinations in generated text.
        
        Args:
            generated_text: Text to analyze.
            source_documents: Source documents for verification.
            query: Optional original query.
            
        Returns:
            Detection result dictionary.
        """
        payload = {
            "generated_text": generated_text,
            "source_documents": source_documents,
            "query": query,
        }
        
        response = requests.post(
            f"{self.base_url}/detect",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    def detect_batch(self, items: List[dict]) -> dict:
        """
        Process multiple detection requests.
        
        Args:
            items: List of detection request dictionaries.
            
        Returns:
            Batch detection results.
        """
        payload = {"items": items}
        
        response = requests.post(
            f"{self.base_url}/detect/batch",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    def list_strategies(self) -> dict:
        """List available detection strategies."""
        response = requests.get(f"{self.base_url}/strategies")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the API client."""
    
    # Initialize client
    client = HallucinationDetectorClient()
    
    print("=" * 60)
    print("Hallucination Detection API Client Example")
    print("=" * 60)
    
    # Check health
    try:
        health = client.health_check()
        print(f"\n✅ API Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Strategies: {', '.join(health['strategies_loaded'])}")
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API.")
        print("   Make sure the server is running: python -m hallucination_detector.api.server")
        return
    
    # List strategies
    print("\n" + "-" * 40)
    strategies = client.list_strategies()
    print("\nAvailable Strategies:")
    for s in strategies["strategies"]:
        print(f"  • {s['name']}: {s['description'][:50]}...")
    
    # Run detection
    print("\n" + "-" * 40)
    print("\nRunning Detection...")
    
    result = client.detect(
        generated_text="""
        Tesla was founded by Elon Musk in 2003 and is headquartered in 
        Austin, Texas. The company's first car was the Model S, released in 2008.
        Tesla has sold over 10 million electric vehicles worldwide.
        """,
        source_documents=[
            """
            Tesla, Inc. is an American multinational automotive and clean energy 
            company. Founded in July 2003 by Martin Eberhard and Marc Tarpenning, 
            the company is headquartered in Austin, Texas. Elon Musk joined as 
            chairman in 2004 and became CEO in 2008. The Tesla Roadster was the 
            company's first production car, launched in 2008. The Model S was 
            released in 2012. As of 2023, Tesla has delivered over 4 million vehicles.
            """
        ],
        query="Tell me about Tesla.",
    )
    
    print(f"\n📊 Detection Results:")
    print(f"   Hallucination Score: {result['hallucination_score']:.2%}")
    print(f"   Classification: {'⚠️ HALLUCINATED' if result['is_hallucinated'] else '✅ GROUNDED'}")
    print(f"   Confidence: {result['confidence']:.2%}")
    
    if result["hallucination_spans"]:
        print(f"\n   Detected {len(result['hallucination_spans'])} hallucination(s):")
        for span in result["hallucination_spans"]:
            print(f"\n   [{span['severity'].upper()}] \"{span['text'][:50]}...\"")
            print(f"   Type: {span['hallucination_type']}")
            print(f"   Explanation: {span['explanation']}")
    
    print("\n   Strategy Scores:")
    for name, score in result["strategy_scores"].items():
        print(f"     • {name}: {score:.2%}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

