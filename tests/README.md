# 🧪 Test Suite for RudyAIBot

This directory contains the comprehensive testing strategy for the RAG pipeline, covering integration tests, and RAG-specific evaluation metrics.

## 📂 Directory Structure

```text
tests/
├── __init__.py
├── conftest.py          # Shared fixtures (Mocks for Qdrant, Gemini, etc.)
├── integration/         # Integration tests
│   └── test_bot_flow.py   # End-to-end simulated user interactions
└── evaluation/          # RAG Quality Evaluation
    ├── evaluate_rag.py    # Script to calculate Hit Rate & Response Similarity
    └── test_data/
        └── golden_dataset.json  # Ground truth Q&A pairs
```

## 🚀 How to Run Tests

### 1. Install Development Dependencies
Ensure you have the test packages installed:
```bash
pip install -r requirements.txt
```

### 2. Run Integration Tests
Use `pytest` to run integration tests. 
```bash
pytest tests/integration/ # Run only integration tests
```

### 3. Run RAG Evaluation
To evaluate the RAG pipeline's accuracy against the Golden Dataset, run the evaluation script as a module from the project root. This effectively calculates **Hit Rate** and **Cosine Similarity**.

**Command:**
```bash
python -m tests.evaluation.evaluate_rag
```
