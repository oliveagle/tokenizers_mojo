"""Unit tests for `from_pretrained_wordpiece`.

Fixture: `tests/data/test_wordpiece.json`

Run:  mojo run -I src -I tests tests/test_from_pretrained_wordpiece.mojo
"""

from from_pretrained import from_pretrained_wordpiece
from harness import expect, expect_eq, expect_str, summarize


def test_load_wordpiece() raises:
    var model = from_pretrained_wordpiece("tests/data/test_wordpiece.json")
    expect_eq(model.vocab_size(), 37, "vocab size")
    expect_str(model.unk_token, "[UNK]", "unk_token")
    expect_str(model.continuing_subword_prefix, "##", "continuing_subword_prefix")


def test_encode_wordpiece() raises:
    var model = from_pretrained_wordpiece("tests/data/test_wordpiece.json")

    # "hello" should be tokenized using longest match
    # "h" -> 11, "e" -> 8, "l" -> 15, "##l" -> not in vocab, "o" -> 18
    # Actually, let's test with a word that has known subwords
    # "unhappy" -> ["un", "##happy"] (both in vocab)
    var ids = model.encode("unhappy")
    expect_eq(len(ids), 2, "unhappy should have 2 tokens")
    expect_eq(ids[0], 34, "un should be 34")
    expect_eq(ids[1], 35, "##happy should be 35")


def test_encode_single_char() raises:
    var model = from_pretrained_wordpiece("tests/data/test_wordpiece.json")

    # "a" is in vocab as id 4
    var ids = model.encode("a")
    expect_eq(len(ids), 1, "a should have 1 token")
    expect_eq(ids[0], 4, "a should be 4")


def test_encode_unknown() raises:
    var model = from_pretrained_wordpiece("tests/data/test_wordpiece.json")

    # "xyz" is not in vocab, should return [UNK]
    var ids = model.encode("xyz")
    expect_eq(len(ids), 1, "unknown should have 1 token")
    expect_eq(ids[0], 1, "unknown should be [UNK]")


def test_token_lookup() raises:
    var model = from_pretrained_wordpiece("tests/data/test_wordpiece.json")
    expect_eq(model.token_id("hello"), -1, "hello not in vocab")
    expect_eq(model.token_id("he"), -1, "he not in vocab")
    expect_eq(model.token_id("##llo"), -1, "##llo not in vocab")
    expect_eq(model.token_id("un"), 34, "un -> 34")
    expect_eq(model.token_id("##happy"), 35, "##happy -> 35")
    expect_str(model.token_for_id(34), "un", "34 -> un")


def main() raises:
    var failures = List[String]()
    try:
        test_load_wordpiece()
    except:
        failures.append("test_load_wordpiece")
    try:
        test_encode_wordpiece()
    except:
        failures.append("test_encode_wordpiece")
    try:
        test_encode_single_char()
    except:
        failures.append("test_encode_single_char")
    try:
        test_encode_unknown()
    except:
        failures.append("test_encode_unknown")
    try:
        test_token_lookup()
    except:
        failures.append("test_token_lookup")
    summarize(failures)
