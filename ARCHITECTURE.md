# System Architecture

## High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HALLUCINATION DETECTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          API LAYER (FastAPI)                        │   │
│  │  • REST endpoints (/detect, /detect/batch, /health)                 │   │
│  │  • Request validation (Pydantic)                                    │   │
│  │  • Async support for high throughput                                │   │
│  │  • CORS middleware                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DETECTION ORCHESTRATOR                         │   │
│  │  • Coordinates multiple detection strategies                        │   │
│  │  • Aggregates results using configurable methods                    │   │
│  │  • Merges overlapping hallucination spans                          │   │
│  │  • Handles errors gracefully                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│            ┌───────────────────────┼───────────────────────┐               │
│            ▼                       ▼                       ▼               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐        │
│  │    SEMANTIC      │  │       NLI        │  │     ENTITY       │        │
│  │   SIMILARITY     │  │   ENTAILMENT     │  │  VERIFICATION    │        │
│  │                  │  │                  │  │                  │        │
│  │ • Sentence-BERT  │  │ • BART-MNLI      │  │ • spaCy NER      │        │
│  │ • Cosine sim     │  │ • 3-way class    │  │ • Fuzzy matching │        │
│  │ • Chunking       │  │ • Premise-hypo   │  │ • Type checking  │        │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       RESULT AGGREGATOR                             │   │
│  │  • Weighted average / Max score / Voting                           │   │
│  │  • Span merging and deduplication                                  │   │
│  │  • Confidence calibration                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DETECTION RESULT                               │   │
│  │  • Overall hallucination score                                      │   │
│  │  • Binary classification                                            │   │
│  │  • Span-level annotations                                           │   │
│  │  • Per-strategy breakdowns                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Design Patterns Used

### 1. Strategy Pattern
Each detection approach implements a common interface, allowing easy addition of new strategies.

### 2. Ensemble Pattern  
Multiple weak signals combined for robust detection.

### 3. Factory Pattern
Configuration-driven initialization of strategies.

## Scalability Considerations

### Horizontal Scaling
- Stateless API design enables load balancing
- Docker-ready for container orchestration (K8s)

### Vertical Scaling
- GPU support for transformer models
- Batch processing for throughput optimization

### Caching Opportunities
- Model weights cached after first load
- Embedding cache for repeated documents
- Result cache for identical inputs

## Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Semantic Similarity | O(n·m·d) | O(n·d + m·d) |
| NLI Entailment | O(n·m) | O(1) |
| Entity Verification | O(n + m) | O(e) |
| Span Merging | O(s log s) | O(s) |

Where:
- n = number of generated sentences
- m = number of source chunks  
- d = embedding dimension
- e = number of entities
- s = number of spans

