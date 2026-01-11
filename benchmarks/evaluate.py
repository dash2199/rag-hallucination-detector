"""
Benchmark evaluation for hallucination detection.

Evaluates the detector on standard datasets and reports metrics
that demonstrate system effectiveness.
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm

from hallucination_detector import HallucinationDetector, DetectorConfig
from hallucination_detector.config import DetectionStrategy


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""
    id: str
    generated_text: str
    source_documents: List[str]
    query: Optional[str] = None
    label: bool = False  # True = contains hallucination
    hallucination_spans: List[Dict] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for the benchmark."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auroc: float
    
    # Performance metrics
    avg_latency_ms: float
    p95_latency_ms: float
    throughput_samples_per_sec: float
    
    # Detailed breakdowns
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    total_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": f"{self.accuracy:.2%}",
            "precision": f"{self.precision:.2%}",
            "recall": f"{self.recall:.2%}",
            "f1_score": f"{self.f1_score:.2%}",
            "auroc": f"{self.auroc:.3f}",
            "avg_latency_ms": f"{self.avg_latency_ms:.1f}",
            "p95_latency_ms": f"{self.p95_latency_ms:.1f}",
            "throughput_samples_per_sec": f"{self.throughput_samples_per_sec:.2f}",
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "true_negatives": self.true_negatives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
            },
            "total_samples": self.total_samples,
        }
    
    def summary(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║           HALLUCINATION DETECTION BENCHMARK RESULTS          ║
╠══════════════════════════════════════════════════════════════╣
║  CLASSIFICATION METRICS                                      ║
║  ├─ Accuracy:    {self.accuracy:>6.2%}                                   ║
║  ├─ Precision:   {self.precision:>6.2%}                                   ║
║  ├─ Recall:      {self.recall:>6.2%}                                   ║
║  ├─ F1 Score:    {self.f1_score:>6.2%}                                   ║
║  └─ AUROC:       {self.auroc:>6.3f}                                    ║
╠══════════════════════════════════════════════════════════════╣
║  PERFORMANCE METRICS                                         ║
║  ├─ Avg Latency:     {self.avg_latency_ms:>7.1f} ms                          ║
║  ├─ P95 Latency:     {self.p95_latency_ms:>7.1f} ms                          ║
║  └─ Throughput:      {self.throughput_samples_per_sec:>7.2f} samples/sec                 ║
╠══════════════════════════════════════════════════════════════╣
║  CONFUSION MATRIX                                            ║
║  ├─ True Positives:  {self.true_positives:>5}                                 ║
║  ├─ True Negatives:  {self.true_negatives:>5}                                 ║
║  ├─ False Positives: {self.false_positives:>5}                                 ║
║  └─ False Negatives: {self.false_negatives:>5}                                 ║
║                                                              ║
║  Total Samples: {self.total_samples}                                        ║
╚══════════════════════════════════════════════════════════════╝
"""


class HallucinationBenchmark:
    """
    Benchmark suite for evaluating hallucination detection.
    
    Supports evaluation on:
    - Custom datasets
    - Synthetic test cases
    - Standard NLP benchmarks (when available)
    """
    
    def __init__(self, detector: Optional[HallucinationDetector] = None):
        """Initialize benchmark with optional pre-configured detector."""
        self.detector = detector or HallucinationDetector()
        self.samples: List[BenchmarkSample] = []
    
    def load_dataset(self, path: str) -> None:
        """Load benchmark dataset from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.samples = [
            BenchmarkSample(
                id=item.get("id", str(i)),
                generated_text=item["generated_text"],
                source_documents=item["source_documents"],
                query=item.get("query"),
                label=item.get("label", False),
                hallucination_spans=item.get("hallucination_spans", []),
            )
            for i, item in enumerate(data)
        ]
    
    def load_synthetic_dataset(self) -> None:
        """Load built-in synthetic test cases for evaluation."""
        self.samples = self._generate_synthetic_samples()
    
    def _generate_synthetic_samples(self) -> List[BenchmarkSample]:
        """Generate synthetic benchmark samples with known labels."""
        samples = []
        
        # True negatives - grounded text
        grounded_cases = [
            {
                "generated": "Python was created by Guido van Rossum and released in 1991.",
                "source": "Python is a programming language created by Guido van Rossum. It was first released in 1991.",
                "label": False,
            },
            {
                "generated": "The Eiffel Tower is located in Paris, France.",
                "source": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
                "label": False,
            },
            {
                "generated": "Amazon was founded by Jeff Bezos in 1994 as an online bookstore.",
                "source": "Amazon.com, Inc. was founded by Jeff Bezos in Bellevue, Washington, in 1994. It started as an online bookstore.",
                "label": False,
            },
        ]
        
        # True positives - hallucinated text
        hallucinated_cases = [
            {
                "generated": "Python was created by Linus Torvalds in 1985.",
                "source": "Python is a programming language created by Guido van Rossum. It was first released in 1991.",
                "label": True,
            },
            {
                "generated": "The Eiffel Tower is 500 meters tall and located in London.",
                "source": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall.",
                "label": True,
            },
            {
                "generated": "Amazon was founded by Bill Gates in 2001 as a cloud computing company.",
                "source": "Amazon.com, Inc. was founded by Jeff Bezos in Bellevue, Washington, in 1994. It started as an online bookstore.",
                "label": True,
            },
            {
                "generated": "Tesla was founded by Elon Musk in 2003 and released the Model S as its first car.",
                "source": "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined in 2004. The first car was the Roadster in 2008, followed by Model S in 2012.",
                "label": True,
            },
            {
                "generated": "Mount Everest is 10,000 meters tall, making it twice the height of K2.",
                "source": "Mount Everest is Earth's highest mountain above sea level, at 8,848.86 m. K2 is the second-highest at 8,611 m.",
                "label": True,
            },
        ]
        
        # Mixed cases - partial hallucination
        partial_cases = [
            {
                "generated": "Apple was founded in 1976 by Steve Jobs. It is headquartered in New York City.",
                "source": "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne. It is headquartered in Cupertino, California.",
                "label": True,  # Contains hallucination (New York)
            },
            {
                "generated": "Google was founded by Larry Page and Sergey Brin in 1998. It was originally called BackRub.",
                "source": "Google was founded in 1998 by Larry Page and Sergey Brin while they were students at Stanford. The search engine was originally called BackRub.",
                "label": False,  # All facts correct
            },
        ]
        
        for i, case in enumerate(grounded_cases + hallucinated_cases + partial_cases):
            samples.append(BenchmarkSample(
                id=f"synthetic_{i}",
                generated_text=case["generated"],
                source_documents=[case["source"]],
                label=case["label"],
            ))
        
        return samples
    
    def evaluate(self, verbose: bool = True) -> EvaluationMetrics:
        """
        Run evaluation on loaded dataset.
        
        Returns:
            EvaluationMetrics with accuracy, latency, and other metrics.
        """
        if not self.samples:
            raise ValueError("No samples loaded. Call load_dataset() or load_synthetic_dataset() first.")
        
        predictions = []
        labels = []
        scores = []
        latencies = []
        
        iterator = tqdm(self.samples, desc="Evaluating") if verbose else self.samples
        
        for sample in iterator:
            # Time the detection
            start = time.perf_counter()
            result = self.detector.detect(
                generated_text=sample.generated_text,
                source_documents=sample.source_documents,
                query=sample.query,
            )
            latency = (time.perf_counter() - start) * 1000  # ms
            
            predictions.append(result.is_hallucinated)
            labels.append(sample.label)
            scores.append(result.hallucination_score)
            latencies.append(latency)
        
        # Calculate metrics
        predictions = np.array(predictions)
        labels = np.array(labels)
        scores = np.array(scores)
        latencies = np.array(latencies)
        
        tp = np.sum((predictions == True) & (labels == True))
        tn = np.sum((predictions == False) & (labels == False))
        fp = np.sum((predictions == True) & (labels == False))
        fn = np.sum((predictions == False) & (labels == True))
        
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUROC calculation
        auroc = self._calculate_auroc(labels, scores)
        
        # Performance metrics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        total_time = np.sum(latencies) / 1000  # seconds
        throughput = len(self.samples) / total_time if total_time > 0 else 0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auroc=auroc,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            throughput_samples_per_sec=throughput,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            total_samples=len(self.samples),
        )
    
    def _calculate_auroc(self, labels: np.ndarray, scores: np.ndarray) -> float:
        """Calculate Area Under ROC Curve."""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(labels, scores)
        except ImportError:
            # Simple approximation if sklearn not available
            sorted_indices = np.argsort(scores)[::-1]
            sorted_labels = labels[sorted_indices]
            
            n_pos = np.sum(labels)
            n_neg = len(labels) - n_pos
            
            if n_pos == 0 or n_neg == 0:
                return 0.5
            
            tpr_sum = 0
            pos_seen = 0
            
            for label in sorted_labels:
                if label:
                    pos_seen += 1
                else:
                    tpr_sum += pos_seen
            
            return tpr_sum / (n_pos * n_neg)


def run_benchmark():
    """Run the standard benchmark suite."""
    print("=" * 60)
    print("HALLUCINATION DETECTION BENCHMARK")
    print("=" * 60)
    
    # Initialize detector
    print("\nInitializing detector...")
    config = DetectorConfig(
        strategies=[
            DetectionStrategy.SEMANTIC_SIMILARITY,
            DetectionStrategy.NLI_ENTAILMENT,
            DetectionStrategy.ENTITY_VERIFICATION,
        ],
        verbose=False,
    )
    detector = HallucinationDetector(config)
    
    # Run benchmark
    benchmark = HallucinationBenchmark(detector)
    benchmark.load_synthetic_dataset()
    
    print(f"\nEvaluating on {len(benchmark.samples)} samples...")
    metrics = benchmark.evaluate(verbose=True)
    
    print(metrics.summary())
    
    # Save results
    results_path = Path("benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_benchmark()

