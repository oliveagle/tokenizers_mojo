"""Tests for token count estimator."""

from token_count_estimator import (
    TokenCountEstimator,
    estimate_token_count,
    estimate_tokens_cheap,
)
from harness import expect, expect_eq, expect_str, summarize


def test_empty_string() raises:
    """Empty string should return 0."""
    expect_eq(estimate_tokens_cheap(""), 0, "empty string")
    expect_eq(estimate_token_count(""), 0, "empty string")


def test_single_word() raises:
    """Single word estimate should be reasonable."""
    # "hello" = 5 bytes, expected ~2 tokens (5/3.7 ≈ 1.35, but BPE merges)
    var est = estimate_token_count("hello")
    expect(est >= 1 and est <= 3, "hello should be 1-3 tokens, got " + String(est))


def test_sentence() raises:
    """Typical sentence should have reasonable estimate."""
    # "The quick brown fox jumps over the lazy dog"
    # This is a well-known pangram, typically ~9 tokens in GPT-2
    var est = estimate_token_count("The quick brown fox jumps over the lazy dog")
    expect(est >= 7 and est <= 12, "sentence should be 7-12 tokens, got " + String(est))


def test_numbers() raises:
    """Numbers should be estimated reasonably."""
    var est = estimate_token_count("1234567890")
    # Numbers often split into chunks, ~3-4 tokens for 10 digits
    expect(est >= 2 and est <= 5, "numbers should be 2-5 tokens, got " + String(est))


def test_punctuation() raises:
    """Punctuation should be estimated reasonably."""
    var est = estimate_token_count("Hello, world!")
    # Typically 3 tokens: "Hello", ",", " world", "!"
    expect(est >= 2 and est <= 5, "punctuation should be 2-5 tokens, got " + String(est))


def test_code() raises:
    """Code should have reasonable estimate."""
    var est = estimate_token_count("def hello_world():")
    # Code often has more tokens due to special characters
    expect(est >= 3 and est <= 8, "code should be 3-8 tokens, got " + String(est))


def test_cheap_vs_accurate() raises:
    """Cheap estimate should be close to accurate estimate."""
    var texts = List[String]()
    texts.append("Hello, world!")
    texts.append("The quick brown fox jumps over the lazy dog")
    texts.append("def calculate_sum(numbers): return sum(numbers)")
    texts.append("1234567890")
    texts.append("https://example.com/path/to/resource?q=value")
    
    for text in texts:
        var cheap = estimate_tokens_cheap(text)
        var accurate = estimate_token_count(text)
        var accurate_f = Float64(accurate)
        if accurate_f < 1.0:
            accurate_f = 1.0
        var ratio = Float64(cheap) / accurate_f
        expect(ratio > 0.6 and ratio < 1.5, 
               "cheap/accurate ratio should be 0.6-1.5 for '" + text + "', got " + String(ratio))


def test_batch_estimate() raises:
    """Batch estimate should sum individual estimates."""
    var estimator = TokenCountEstimator(3.7)
    var texts = List[String]()
    texts.append("hello")
    texts.append("world")
    texts.append("test")
    var batch_est = estimator.estimate_batch(texts)
    var sum_est = 0
    for text in texts:
        sum_est += estimator.estimate(text)
    expect_eq(batch_est, sum_est, "batch should equal sum of individual")


def main() raises:
    var failures = List[String]()
    var cases = [
        "test_empty_string",
        "test_single_word",
        "test_sentence",
        "test_numbers",
        "test_punctuation",
        "test_code",
        "test_cheap_vs_accurate",
        "test_batch_estimate",
    ]
    
    for name in cases:
        try:
            if name == "test_empty_string":
                test_empty_string()
            elif name == "test_single_word":
                test_single_word()
            elif name == "test_sentence":
                test_sentence()
            elif name == "test_numbers":
                test_numbers()
            elif name == "test_punctuation":
                test_punctuation()
            elif name == "test_code":
                test_code()
            elif name == "test_cheap_vs_accurate":
                test_cheap_vs_accurate()
            elif name == "test_batch_estimate":
                test_batch_estimate()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
