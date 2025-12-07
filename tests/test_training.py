# tests/test_training.py


def test_import_train_scripts():
    """
    Light check: training scripts can be imported without crashing.
    This does NOT run training, only imports modules.
    """
    import src.training.train_student  # noqa: F401
    import src.training.train_student_kd  # noqa: F401
    import src.training.train_teacher  # noqa: F401

    assert True
