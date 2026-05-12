"""
Configuration settings for the book genre classifier project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data settings
RAW_DATA_FILE = DATA_DIR / "raw_books.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_books.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
VAL_FILE = DATA_DIR / "val.csv"

# Genre categories (can be expanded)
GENRES = [
    'fiction',
    'mystery',
    'romance',
    'fantasy',
    'science_fiction',
    'thriller',
    'horror',
    'historical_fiction',
    'young_adult',
    'non_fiction',
    'biography',
    'self_help'
]

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Text processing settings
MAX_FEATURES = 15000  # Increased for more diverse vocabulary
MIN_DF = 1  # Reduced to capture more unique words
MAX_DF = 0.90  # Slightly reduced
NGRAM_RANGE = (1, 3)  # Include trigrams for better context

# Model hyperparameters
MODEL_CONFIGS = {
    'naive_bayes': {
        'alpha': 1.0
    },
    'svm': {
        'C': 1.0,
        'kernel': 'linear'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': RANDOM_STATE
    },
    'neural_network': {
        'hidden_layer_sizes': (100, 50),
        'max_iter': 500,
        'random_state': RANDOM_STATE
    }
}

# Evaluation settings
CV_FOLDS = 5
RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"
CONFUSION_MATRIX_FILE = RESULTS_DIR / "confusion_matrix.png"
ERROR_ANALYSIS_FILE = RESULTS_DIR / "error_analysis.txt"