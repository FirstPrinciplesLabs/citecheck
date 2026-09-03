"""Import and top-level API smoke tests."""

import citecheck


def test_package_version():
    assert citecheck.__version__ == "0.1.0"


def test_top_level_api_exports():
    assert callable(citecheck.detect_citations)
    assert callable(citecheck.evaluate_dataset)
    assert callable(citecheck.load_dataset)
    assert callable(citecheck.classify_score)
