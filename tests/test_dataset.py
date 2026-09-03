"""Bundled dataset loader smoke tests."""

import citecheck


def test_load_bundled_dataset_counts():
    dataset = citecheck.load_dataset()

    assert len(dataset) == 42
    assert dataset.total_citations == 982
    assert len(dataset.collection_ids) == 42
