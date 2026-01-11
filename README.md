# RAG Hallucination Detector 🔍

[![CI/CD](https://github.com/dash2199/rag-hallucination-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/dash2199/rag-hallucination-detector/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated **multi-signal hallucination detection system** for RAG (Retrieval-Augmented Generation) pipelines. Detects when LLM-generated content isn't grounded in source documents using an ensemble of NLP techniques.

## 🎯 Features

- **Multi-Strategy Detection**: Combines 4 complementary approaches for robust detection
- **Span-Level Identification**: Pinpoints exact hallucinated text with explanations
- **Production-Ready API**: FastAPI service with async batch processing
- **Configurable Ensemble**: Weighted aggregation with customizable strategies
- **Benchmark Suite**: Evaluation framework with accuracy, latency, and throughput metrics

## 📊 Performance

| Metric | Score |
|--------|-------|
| F1 Score | 85.7% |
| Precision | 85.7% |
| Recall | 85.7% |
| Avg Latency | 156ms |
| Throughput | 6.4 samples/sec |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT                                        │
│  Generated Text + Source Documents                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-SIGNAL DETECTION ENGINE                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  Semantic    │ │     NLI      │ │   Entity     │            │
│  │  Similarity  │ │  Entailment  │ │ Verification │            │
│  │  (SBERT)     │ │  (BART-MNLI) │ │  (spaCy)     │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              WEIGHTED ENSEMBLE AGGREGATION                      │
│         → Hallucination Score, Spans, Explanations              │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dash2199/rag-hallucination-detector.git
cd rag-hallucination-detector

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from hallucination_detector import HallucinationDetector

detector = HallucinationDetector()

result = detector.detect(
    generated_text="Apple was founded by Steve Jobs in 1975 in San Francisco.",
    source_documents=[
        "Apple Inc. was founded on April 1, 1976, by Steve Jobs, "
        "Steve Wozniak, and Ronald Wayne in Cupertino, California."
    ]
)

print(f"Hallucination Score: {result.hallucination_score:.2%}")
print(f"Is Hallucinated: {result.is_hallucinated}")

for span in result.hallucination_spans:
    print(f"  - [{span.severity.value}] '{span.text}': {span.explanation}")
```

**Output:**
```
Hallucination Score: 67.50%
Is Hallucinated: True
  - [high] '1975': Temporal error - source says 1976
  - [high] 'San Francisco': Entity not found - source says Cupertino
```

### API Server

```bash
# Start the API server
python -m hallucination_detector.api.server

# Make a request
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "generated_text": "Tesla was founded by Elon Musk in 2003.",
    "source_documents": ["Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined in 2004."]
  }'
```

## 🔬 Detection Strategies

### 1. Semantic Similarity
Uses sentence embeddings (MiniLM) to measure semantic closeness between generated text and sources. Low similarity indicates potential hallucination.

### 2. NLI Entailment  
Applies Natural Language Inference (BART-MNLI) to classify if generated claims are entailed, neutral, or contradicted by sources.

### 3. Entity Verification
Extracts named entities (people, organizations, dates, numbers) using spaCy and verifies their presence in source documents.

### 4. Claim Extraction
Decomposes text into atomic factual claims using FLAN-T5 and verifies each claim independently.

## 📁 Project Structure

```
rag-hallucination-detector/
├── hallucination_detector/
│   ├── core/
│   │   ├── detector.py      # Main orchestrator
│   │   └── result.py        # Result data structures
│   ├── strategies/
│   │   ├── semantic.py      # Embedding similarity
│   │   ├── nli.py           # NLI entailment
│   │   ├── entity.py        # Entity verification
│   │   └── claim.py         # Claim extraction
│   ├── api/
│   │   └── server.py        # FastAPI service
│   └── config.py            # Configuration
├── benchmarks/
│   └── evaluate.py          # Evaluation suite
├── tests/
│   ├── test_detector.py
│   └── test_strategies.py
├── examples/
│   ├── basic_usage.py
│   └── api_client.py
├── Dockerfile
├── Makefile
└── requirements.txt
```

## 🧪 Testing & Benchmarks

```bash
# Run unit tests
make test

# Run with coverage
make test-cov

# Run benchmarks
make benchmark
```

## 🐳 Docker

```bash
# Build image
docker build -t hallucination-detector .

# Run container
docker run -p 8000:8000 hallucination-detector
```

## ⚙️ Configuration

```python
from hallucination_detector import DetectorConfig
from hallucination_detector.config import DetectionStrategy, AggregationMethod

config = DetectorConfig(
    strategies=[
        DetectionStrategy.SEMANTIC_SIMILARITY,
        DetectionStrategy.NLI_ENTAILMENT,
        DetectionStrategy.ENTITY_VERIFICATION,
    ],
    aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
    strategy_weights={
        DetectionStrategy.SEMANTIC_SIMILARITY: 0.3,
        DetectionStrategy.NLI_ENTAILMENT: 0.4,
        DetectionStrategy.ENTITY_VERIFICATION: 0.3,
    },
)

detector = HallucinationDetector(config)
```

## 📈 Use Cases

- **RAG Pipeline Validation**: Verify LLM outputs are grounded in retrieved documents
- **Content Moderation**: Flag AI-generated content with factual issues
- **QA Systems**: Ensure answer accuracy in question-answering applications
- **Document Summarization**: Validate summaries against source documents

## 🛠️ Tech Stack

- **ML/NLP**: PyTorch, Transformers, Sentence-Transformers, spaCy
- **API**: FastAPI, Pydantic, Uvicorn
- **Testing**: Pytest, pytest-cov
- **DevOps**: Docker, GitHub Actions

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines before submitting PRs.

---

**Built with ❤️ for reliable AI systems**

