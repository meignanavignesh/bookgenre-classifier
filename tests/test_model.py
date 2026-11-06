import pickle
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"
DATA_PATH = BASE_DIR / "data" / "final_books.csv"

EXPECTED_GENRES = {
    'biography', 'fantasy', 'fiction', 'historical_fiction',
    'horror', 'mystery', 'non_fiction', 'romance',
    'science_fiction', 'self_help', 'thriller', 'young_adult'
}


def test_dataset_file_exists():
    assert DATA_PATH.exists(), f"Dataset file missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataset must not be empty"
    assert "description" in df.columns, "Dataset must contain 'description' column"
    assert "genre" in df.columns, "Dataset must contain 'genre' column"
    assert set(df["genre"].unique()) == EXPECTED_GENRES, "Dataset must include all 12 expected genres"


def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"


def test_model_loads_and_has_classes():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    assert hasattr(model, "predict"), "Loaded model must have 'predict' method"
    assert hasattr(model, "predict_proba"), "Loaded model must have 'predict_proba' method"
    assert hasattr(model, "classes_"), "Loaded model must have 'classes_' attribute"
    assert set(model.classes_) == EXPECTED_GENRES, "Model classes must match the 12 expected genres"


def test_model_prediction_outputs():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    test_queries = [
        "A detective investigates a murder in Victorian London",
        "A young wizard discovers magical powers at an enchanted academy",
        "Space colonists travel across the galaxy on a futuristic starship"
    ]

    for query in test_queries:
        pred = model.predict([query])[0]
        probs = model.predict_proba([query])[0]
        confidence = float(np.max(probs))

        assert isinstance(pred, str)
        assert pred in EXPECTED_GENRES
        assert 0.0 <= confidence <= 1.0
        assert np.isclose(np.sum(probs), 1.0, atol=1e-3)
