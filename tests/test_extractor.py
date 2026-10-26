"""Tests for profile feature extractor and URL sanitization."""

from fake_profile_detector.extractor import (
    extract_profile_features,
    parse_numeric_shorthand,
    sanitize_handle,
)


def test_sanitize_handle():
    assert sanitize_handle("@cristiano") == "cristiano"
    assert sanitize_handle("https://www.instagram.com/nasa/") == "nasa"
    assert sanitize_handle("https://instagram.com/test_user?igshid=123") == "test_user"
    assert sanitize_handle("plain_user") == "plain_user"


def test_parse_numeric_shorthand():
    assert parse_numeric_shorthand("680M") == 680_000_000
    assert parse_numeric_shorthand("10.5K") == 10_500
    assert parse_numeric_shorthand("1.2B") == 1_200_000_000
    assert parse_numeric_shorthand("4,141") == 4141
    assert parse_numeric_shorthand("0") == 0
    assert parse_numeric_shorthand("") == 0


def test_extract_profile_features_sample():
    features, meta, err = extract_profile_features("cristiano")
    assert err is None
    assert meta["username"] == "cristiano"
    assert features["#followers"] > 1_000_000
    assert "followers_following_ratio" in features
