# tests/test_model_loader.py

from src.inference.model_loader import load_model


def test_model_loader():
    """Ensure load_model returns a callable prediction function."""
    predict_fn = load_model()
    assert callable(predict_fn)
