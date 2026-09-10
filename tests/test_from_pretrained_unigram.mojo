"""Unit tests for `from_pretrained_unigram`.

Fixture: `tests/data/test_unigram.json`

Run:  mojo run -I src -I tests tests/test_from_pretrained_unigram.mojo
"""

from from_pretrained import from_pretrained_unigram
from harness import expect, expect_eq, expect_str, summarize


def test_load_unigram() raises:
    var model = from_pretrained_unigram("tests/data/test_unigram.json")
    expect_eq(model.vocab_size(), 17, "vocab size")


def test_token_lookup() raises:
    var model = from_pretrained_unigram("tests/data/test_unigram.json")
    expect_eq(model.token_id("abc"), 13, "abc -> 13")
    expect_eq(model.token_id("def"), 14, "def -> 14")
    expect_eq(model.token_id("a"), 1, "a -> 1")
    expect_eq(model.token_id("b"), 2, "b -> 2")


def main() raises:
    var failures = List[String]()
    try:
        test_load_unigram()
    except e:
        failures.append("test_load_unigram: " + String(e))
    try:
        test_token_lookup()
    except e:
        failures.append("test_token_lookup: " + String(e))
    summarize(failures)
