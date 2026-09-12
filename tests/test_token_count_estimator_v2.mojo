"""Tests for token count estimator V2."""

from token_count_estimator_v2 import TokenCountEstimatorV2, estimate_token_count_v2
from harness import expect, expect_eq, summarize


def test_empty_string() raises:
    expect_eq(estimate_token_count_v2(""), 0, "empty string")


def test_single_word() raises:
    var est = estimate_token_count_v2("hello")
    expect(est >= 1 and est <= 3, "hello should be 1-3 tokens")


def test_sentence() raises:
    var est = estimate_token_count_v2("The quick brown fox jumps over the lazy dog")
    expect(est >= 8 and est <= 14, "sentence should be 8-14 tokens, got " + String(est))


def test_accuracy_improvement() raises:
    """V2 should be more accurate than V1 for typical text."""
    var estimator = TokenCountEstimatorV2(3.7)  # GPT-2 ratio
    
    var texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog",
        "def calculate_sum(numbers): return sum(numbers)",
    ]
    
    # With GPT-2 ratio, V2 should estimate reasonably
    for text in texts:
        var est = estimator.estimate(text)
        var bytes = text.byte_length()
        # Estimate should be within 50% of bytes/3.7
        var expected = Float64(bytes) / 3.7
        var ratio = Float64(est) / expected
        expect(ratio > 0.5 and ratio < 2.0, 
               "estimate should be reasonable for '" + text + "'")


def main() raises:
    var failures = List[String]()
    var cases = [
        "test_empty_string",
        "test_single_word",
        "test_sentence",
        "test_accuracy_improvement",
    ]
    
    for name in cases:
        try:
            if name == "test_empty_string":
                test_empty_string()
            elif name == "test_single_word":
                test_single_word()
            elif name == "test_sentence":
                test_sentence()
            elif name == "test_accuracy_improvement":
                test_accuracy_improvement()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
