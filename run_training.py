"""
Training pipeline entry point.

Delegates entirely to pipelines/training_pipeline.py — no logic lives here.
This file exists so the command `python run_training.py` keeps working as a
memorable shortcut without duplicating any pipeline code.

Usage:
    python run_training.py

After the run:
    mlflow ui --port 5000
    Models → supply-chain-late-delivery → promote 'staging' → 'production'
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    training_pipeline()
