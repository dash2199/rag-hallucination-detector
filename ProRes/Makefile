.PHONY: install install-dev test lint format run-api clean

# Install production dependencies
install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov mypy black isort
	python -m spacy download en_core_web_sm

# Run tests
test:
	pytest tests/ -v --cov=hallucination_detector --cov-report=term-missing

# Run tests with coverage report
test-cov:
	pytest tests/ -v --cov=hallucination_detector --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Run linting
lint:
	mypy hallucination_detector/
	black --check hallucination_detector/ tests/
	isort --check-only hallucination_detector/ tests/

# Format code
format:
	black hallucination_detector/ tests/ examples/
	isort hallucination_detector/ tests/ examples/

# Run the API server
run-api:
	python -m hallucination_detector.api.server

# Run examples
run-examples:
	python examples/basic_usage.py

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Build package
build:
	python -m build

# Docker build
docker-build:
	docker build -t hallucination-detector .

# Docker run
docker-run:
	docker run -p 8000:8000 hallucination-detector

# Run benchmarks
benchmark:
	python benchmarks/evaluate.py

# Performance profiling
profile:
	python -m cProfile -s cumulative examples/basic_usage.py > profile.txt
	@echo "Profile saved to profile.txt"

