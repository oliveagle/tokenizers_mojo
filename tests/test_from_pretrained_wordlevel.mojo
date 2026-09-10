"""Unit tests for `from_pretrained_wordlevel`.

Fixture: `tests/data/test_wordlevel.json`

Run:  mojo run -I src -I tests tests/test_from_pretrained_wordlevel.mojo
"""

from from_pretrained import from_pretrained_wordlevel
from harness import expect, expect_eq, expect_str, summarize


def test_load_wordlevel() raises:
    var model = from_pretrained_wordlevel("tests/data/test_wordlevel.json")
    expect_eq(model.vocab_size(), 16, "vocab size")
    expect_str(model.unk_token, "<unk>", "unk_token")


def test_encode_wordlevel() raises:
    var model = from_pretrained_wordlevel("tests/data/test_wordlevel.json")

    # "hello" should be found in vocab
    var ids = model.encode("hello")
    expect_eq(len(ids), 1, "hello should have 1 token")
    expect_eq(ids[0], 8, "hello -> 8")


def test_encode_unknown() raises:
    var model = from_pretrained_wordlevel("tests/data/test_wordlevel.json")

    # "xyz" is not in vocab, should return UNK
    var ids = model.encode("xyz")
    expect_eq(len(ids), 1, "unknown should have 1 token")
    expect_eq(ids[0], 0, "unknown should be <unk>")


def test_token_lookup() raises:
    var model = from_pretrained_wordlevel("tests/data/test_wordlevel.json")
    expect_eq(model.token_id("hello"), 8, "hello -> 8")
    expect_eq(model.token_id("world"), 9, "world -> 9")
    expect_eq(model.token_id("xyz"), -1, "xyz not in vocab")
    expect_str(model.token_for_id(8), "hello", "8 -> hello")


def main() raises:
    var failures = List[String]()
    try:
        test_load_wordlevel()
    except:
        failures.append("test_load_wordlevel")
    try:
        test_encode_wordlevel()
    except:
        failures.append("test_encode_wordlevel")
    try:
        test_encode_unknown()
    except:
        failures.append("test_encode_unknown")
    try:
        test_token_lookup()
    except:
        failures.append("test_token_lookup")
    summarize(failures)
