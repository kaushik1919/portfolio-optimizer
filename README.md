# Portfolio GNN

A research-oriented Graph Neural Network (GNN)–based portfolio optimization engine.

## Overview

This project implements a production-quality framework for GNN-based portfolio optimization, designed for quantitative research and experimentation.

## Project Structure

```
portfolio_gnn/
├── core/                  # Core infrastructure modules
│   ├── config.py          # Configuration loading and validation
│   ├── seeding.py         # Deterministic execution guarantees
│   ├── logging.py         # Standardized logging
│   └── paths.py           # Centralized path resolution
├── data/                  # Data ingestion (Phase 2+)
├── graph/                 # Graph construction (Phase 2+)
├── models/                # GNN models (Phase 3+)
└── optimization/          # Portfolio optimization (Phase 4+)

scripts/
└── run.py                 # Single experiment entry point

experiments/
└── base.yaml              # Base experiment configuration

tests/
└── test_reproducibility.py  # Reproducibility tests
```

## Phase 1: System Scaffold & Reproducibility

Phase 1 establishes the foundational infrastructure:

- ✅ Strict configuration loading from YAML with validation
- ✅ Deterministic execution via comprehensive seeding
- ✅ Standardized logging (no print statements)
- ✅ Cross-platform path resolution
- ✅ Single experiment entry point
- ✅ Reproducibility tests

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-gnn.git
cd portfolio-gnn

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### Running an Experiment

```bash
python scripts/run.py experiments/base.yaml
```

### Validating Configuration

```bash
python scripts/run.py experiments/base.yaml --validate-only
```

### Environment Variable Overrides

```bash
# Override seed
PGNN_RUNTIME_SEED=123 python scripts/run.py experiments/base.yaml

# Override log level
PGNN_RUNTIME_LOGGING_LEVEL=DEBUG python scripts/run.py experiments/base.yaml
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=portfolio_gnn

# Run specific test file
pytest tests/test_reproducibility.py -v
```

## Configuration

Experiments are configured via YAML files. Required sections:

- `data`: Data ingestion settings
- `model`: GNN architecture configuration
- `training`: Training hyperparameters
- `optimization`: Portfolio optimization settings
- `runtime`: Execution environment (seed, logging, device)

See `experiments/base.yaml` for a complete template.

## Reproducibility

All experiments are fully reproducible:

1. Seeds are set for Python random, NumPy, and PyTorch
2. Configuration is hashed for tracking
3. Timestamps are logged
4. Deterministic PyTorch algorithms are enabled

## License

MIT License - see [LICENSE](LICENSE) for details.
