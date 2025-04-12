# `furtive_growth`

This repository houses all software and notes for exploring the role of growth rate optimization in feast-famine environments

## Installation

This package uses modern Python packaging with `pyproject.toml` and can be installed using either `pip` or `uv`.

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/growth.git
cd growth

# Install in development mode
pip install -e .
```

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a faster, more reliable Python package installer. It's recommended for development and ensures consistent dependency versions via lockfiles.

```bash
# If you don't have uv installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/your-username/growth.git
cd growth

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Development Setup

This project includes development dependencies for testing:

```bash
# Install with development dependencies
uv sync --dev
```

## Project Structure

- `growth/`: Main package code
  - `model.py`: Core growth model classes
  - `callbacks.py`: Event callback functions
  - `viz.py`: Visualization utilities
- `simulations/`: Example simulations
- `tests/`: Test suite

## Example Usage

```python
from growth import Species, Ecosystem
import numpy as np

# Create species with different growth parameters
species1 = Species(lambda_max=1.0, Km=0.01, gamma=0.1)
species2 = Species(lambda_max=0.8, Km=0.005, gamma=0.05)

# Initialize an ecosystem with both species
eco = Ecosystem(species=[species1, species2])

# Run growth simulation
results = eco.grow(
    lifetime=100,        # Simulation duration
    feed_conc=1.0,       # Nutrient concentration
    delta=0.1            # Dilution rate
)

# Access the results
species_data, environment_data = results
```

## Requirements

- Python 3.10+
- Dependencies are managed in `pyproject.toml` and locked in `uv.lock`


## License
All creative work is licensed under a CC-BY 4.0 License. Code is released under a standard MIT license.