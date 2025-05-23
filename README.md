# Reinforcement Learning for Adaptive Trading (PPO Agent)

This project implements a Proximal Policy Optimization (PPO) based reinforcement learning agent for adaptive trading using historical price data. The agent learns to make trading decisions (long, short, or flat positions) based on technical indicators and price patterns.

## Project Structure

```
reinforcement_learning_adaptive_trading/
├── config/
│   └── config.yaml            # Experiment configuration
├── data/
│   ├── raw/                   # Raw historical data
│   └── processed/             # Processed data
├── environment/
│   └── trading_env.py         # Custom Gymnasium environment
├── features/
│   └── feature_engineering.py # Feature computation
├── models/                    # Saved model artifacts
├── logs/                      # Training and evaluation logs
├── utils/
│   ├── data_utils.py         # Data handling utilities
│   ├── config_utils.py       # Configuration management
│   ├── logging_utils.py      # Logging setup
│   └── metrics.py            # Performance metrics
├── training/
│   └── train.py              # Training pipeline
└── evaluation/
    └── evaluate.py           # Evaluation script
```

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
1. Configure experiment parameters in `config/config.yaml`
2. Run training:
```bash
python training/train.py
```

### Evaluation
After training, evaluate the model:
```bash
python evaluation/evaluate.py
```

## Features

- Custom Gymnasium trading environment
- PPO implementation using Stable-Baselines3
- Feature engineering for technical indicators
- Comprehensive logging and metrics
- YAML-based configuration
- Modular and maintainable code structure

## Dependencies

See `requirements.txt` for full list of dependencies. Key packages include:
- gymnasium
- stable-baselines3
- torch
- numpy
- pandas
- pyyaml
- scikit-learn 