"""Threshold classifier smoke tests."""

from citecheck import config
from citecheck.core.classify import classify_score, score_for_label


def test_classify_score_boundaries():
    exact = config.EXACT_THRESHOLD
    minor = config.MINOR_THRESHOLD

    assert classify_score(exact) == "exact_match"
    assert classify_score(exact - 0.01) == "minor_hallucination"
    assert classify_score(minor) == "minor_hallucination"
    assert classify_score(minor - 0.01) == "major_hallucination"


def test_score_for_label_round_trip_regions():
    exact = config.EXACT_THRESHOLD
    minor = config.MINOR_THRESHOLD

    assert classify_score(score_for_label("exact_match")) == "exact_match"
    assert classify_score(score_for_label("minor_hallucination")) == "minor_hallucination"
    assert classify_score(score_for_label("major_hallucination")) == "major_hallucination"
    assert score_for_label("exact_match") >= exact
    assert minor <= score_for_label("minor_hallucination") < exact
    assert score_for_label("major_hallucination") < minor
