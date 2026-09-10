"""Tests for WordLevel model.

Run:  mojo run -I src -I tests tests/test_wordlevel.mojo
"""

import wordlevel

from wordlevel import WordLevel
from harness import expect, expect_eq, expect_str, summarize


def test_basic_tokenization() raises:
    """Test basic WordLevel tokenization."""
    var vocab = Dict[String, Int]()
    vocab["<unk>"] = 0
    vocab["hello"] = 1
    vocab["world"] = 2

    var wl = WordLevel()
    wl.load_vocab_from_dict(vocab)

    # "hello" should be tokenized as "hello"
    var token = wl.tokenize_word("hello")
    expect_str(token, "hello", "token should be 'hello'")


def test_unknown_token() raises:
    """Test that unknown words return <unk>."""
    var vocab = Dict[String, Int]()
    vocab["<unk>"] = 0
    vocab["hello"] = 1

    var wl = WordLevel()
    wl.load_vocab_from_dict(vocab)

    # "xyz" is not in vocab, should return <unk>
    var token = wl.tokenize_word("xyz")
    expect_str(token, "<unk>", "token should be '<unk>'")


def test_encode() raises:
    """Test encoding words to vocab ids."""
    var vocab = Dict[String, Int]()
    vocab["<unk>"] = 0
    vocab["hello"] = 1
    vocab["world"] = 2

    var wl = WordLevel()
    wl.load_vocab_from_dict(vocab)

    # "hello" -> [1]
    var ids = wl.encode("hello")
    expect_eq(len(ids), 1, "hello should encode to 1 id")
    expect_eq(ids[0], 1, "id should be 1")

    # "xyz" -> [0] (unk)
    ids = wl.encode("xyz")
    expect_eq(len(ids), 1, "unknown should encode to 1 id")
    expect_eq(ids[0], 0, "id should be 0")


def test_decode() raises:
    """Test decoding vocab ids back to tokens."""
    var vocab = Dict[String, Int]()
    vocab["<unk>"] = 0
    vocab["hello"] = 1
    vocab["world"] = 2

    var wl = WordLevel()
    wl.load_vocab_from_dict(vocab)

    # [1, 2] -> "helloworld"
    var ids = List[Int]()
    ids.append(1)
    ids.append(2)
    var result = wl.decode(ids)
    expect_str(result, "helloworld", "decoded should be 'helloworld'")


def test_vocab_size() raises:
    """Test vocab_size returns correct count."""
    var vocab = Dict[String, Int]()
    vocab["<unk>"] = 0
    vocab["hello"] = 1
    vocab["world"] = 2

    var wl = WordLevel()
    wl.load_vocab_from_dict(vocab)

    expect_eq(wl.vocab_size(), 3, "vocab size should be 3")


def test_token_id() raises:
    """Test token_id lookup."""
    var vocab = Dict[String, Int]()
    vocab["<unk>"] = 0
    vocab["hello"] = 1

    var wl = WordLevel()
    wl.load_vocab_from_dict(vocab)

    expect_eq(wl.token_id("hello"), 1, "hello should have id 1")
    expect_eq(wl.token_id("world"), -1, "unknown should have id -1")


def test_token_for_id() raises:
    """Test token_for_id reverse lookup."""
    var vocab = Dict[String, Int]()
    vocab["<unk>"] = 0
    vocab["hello"] = 1

    var wl = WordLevel()
    wl.load_vocab_from_dict(vocab)

    expect_str(wl.token_for_id(1), "hello", "id 1 should be 'hello'")
    expect_str(wl.token_for_id(999), "", "unknown id should be empty")


def test_load_vocab_from_dict() raises:
    """Test loading vocab from dictionary."""
    var vocab = Dict[String, Int]()
    vocab["<unk>"] = 0
    vocab["foo"] = 1
    vocab["bar"] = 2

    var wl = WordLevel()
    wl.load_vocab_from_dict(vocab)

    expect_eq(wl.vocab_size(), 3, "vocab size should be 3")
    expect_eq(wl.token_id("foo"), 1, "foo should have id 1")


def test_missing_unk_token() raises:
    """Test that missing unk_token raises error."""
    var vocab = Dict[String, Int]()
    vocab["hello"] = 0

    var wl = WordLevel()
    wl.load_vocab_from_dict(vocab)
    wl.unk_token = "<unk>"

    # "xyz" is not in vocab and <unk> is not in vocab, should raise
    var raised = False
    try:
        var ids = wl.encode("xyz")
    except:
        raised = True
    expect(raised, "should raise when unk_token missing")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_basic_tokenization")
    cases.append("test_unknown_token")
    cases.append("test_encode")
    cases.append("test_decode")
    cases.append("test_vocab_size")
    cases.append("test_token_id")
    cases.append("test_token_for_id")
    cases.append("test_load_vocab_from_dict")
    cases.append("test_missing_unk_token")

    for name in cases:
        try:
            if name == "test_basic_tokenization":
                test_basic_tokenization()
            elif name == "test_unknown_token":
                test_unknown_token()
            elif name == "test_encode":
                test_encode()
            elif name == "test_decode":
                test_decode()
            elif name == "test_vocab_size":
                test_vocab_size()
            elif name == "test_token_id":
                test_token_id()
            elif name == "test_token_for_id":
                test_token_for_id()
            elif name == "test_load_vocab_from_dict":
                test_load_vocab_from_dict()
            elif name == "test_missing_unk_token":
                test_missing_unk_token()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
