import pytest
from pathlib import Path
import pandas as pd
import app

def test_app_load_model():
    classifier = app.load_model()
    assert classifier is not None, "app.load_model() returned None"
    assert hasattr(classifier, "predict_genre"), "Classifier must have 'predict_genre' method"


def test_app_load_dataset_info():
    df = app.load_dataset_info()
    assert df is not None, "app.load_dataset_info() returned None"
    assert isinstance(df, pd.DataFrame), "Loaded dataset info must be a pandas DataFrame"
    assert "genre" in df.columns
    assert len(df) > 0


def test_classifier_predict_genre_structure():
    classifier = app.load_model()
    genre, confidence, top_predictions = classifier.predict_genre(
        "A detective investigates a crime scene with forensic clues"
    )
    assert isinstance(genre, str)
    assert 0.0 <= confidence <= 1.0
    assert len(top_predictions) == 3
    for cand_genre, cand_prob in top_predictions:
        assert isinstance(cand_genre, str)
        assert 0.0 <= cand_prob <= 1.0
