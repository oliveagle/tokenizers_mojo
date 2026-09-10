"""Tests for Unigram model.

Run:  mojo run -I src -I tests tests/test_unigram.mojo
"""

import unigram

from unigram import Unigram, UnigramVocabEntry
from harness import expect, expect_eq, expect_str, summarize


def test_basic_tokenization() raises:
    """Test basic Unigram tokenization."""
    var vocab = List[UnigramVocabEntry]()
    vocab.append(UnigramVocabEntry("<unk>", 0.0))
    vocab.append(UnigramVocabEntry("a", -0.3))
    vocab.append(UnigramVocabEntry("b", -0.4))
    vocab.append(UnigramVocabEntry("c", -0.5))
    vocab.append(UnigramVocabEntry("ab", 0.0))
    vocab.append(UnigramVocabEntry("abc", 5.0))

    var model = Unigram(vocab, 0, False)

    # "abc" should be tokenized as ["abc"] (highest score)
    var tokens = model.tokenize_word("abc")
    expect_eq(len(tokens), 1, "abc should have 1 token")
    expect_str(tokens[0], "abc", "token should be 'abc'")


def test_unknown_token() raises:
    """Test that unknown characters return <unk>."""
    var vocab = List[UnigramVocabEntry]()
    vocab.append(UnigramVocabEntry("<unk>", 0.0))
    vocab.append(UnigramVocabEntry("a", -0.3))
    vocab.append(UnigramVocabEntry("b", -0.4))

    var model = Unigram(vocab, 0, False)

    # "x" is not in vocab, should return <unk>
    var tokens = model.tokenize_word("x")
    expect_eq(len(tokens), 1, "unknown should have 1 token")
    expect_str(tokens[0], "<unk>", "token should be '<unk>'")


def test_encode() raises:
    """Test encoding words to vocab ids."""
    var vocab = List[UnigramVocabEntry]()
    vocab.append(UnigramVocabEntry("<unk>", 0.0))
    vocab.append(UnigramVocabEntry("a", -0.3))
    vocab.append(UnigramVocabEntry("b", -0.4))
    vocab.append(UnigramVocabEntry("ab", 2.0))

    var model = Unigram(vocab, 0, False)

    # "ab" -> [3] (ab is at index 3 in vocab, has highest score 2.0)
    var ids = model.encode("ab")
    expect_eq(len(ids), 1, "ab should encode to 1 id")
    expect_eq(ids[0], 3, "id should be 3")


def test_decode() raises:
    """Test decoding vocab ids back to tokens."""
    var vocab = List[UnigramVocabEntry]()
    vocab.append(UnigramVocabEntry("<unk>", 0.0))
    vocab.append(UnigramVocabEntry("hello", -0.5))
    vocab.append(UnigramVocabEntry("world", -0.5))

    var model = Unigram(vocab, 0, False)

    # [1, 2] -> "helloworld"
    var ids = List[Int]()
    ids.append(1)
    ids.append(2)
    var result = model.decode(ids)
    expect_str(result, "helloworld", "decoded should be 'helloworld'")


def test_vocab_size() raises:
    """Test vocab_size returns correct count."""
    var vocab = List[UnigramVocabEntry]()
    vocab.append(UnigramVocabEntry("<unk>", 0.0))
    vocab.append(UnigramVocabEntry("a", -0.3))
    vocab.append(UnigramVocabEntry("b", -0.4))

    var model = Unigram(vocab, 0, False)
    expect_eq(model.vocab_size(), 3, "vocab size should be 3")


def test_token_id() raises:
    """Test token_id lookup."""
    var vocab = List[UnigramVocabEntry]()
    vocab.append(UnigramVocabEntry("<unk>", 0.0))
    vocab.append(UnigramVocabEntry("hello", -0.5))

    var model = Unigram(vocab, 0, False)
    expect_eq(model.token_id("hello"), 1, "hello should have id 1")
    expect_eq(model.token_id("world"), -1, "unknown should have id -1")


def test_token_for_id() raises:
    """Test token_for_id reverse lookup."""
    var vocab = List[UnigramVocabEntry]()
    vocab.append(UnigramVocabEntry("<unk>", 0.0))
    vocab.append(UnigramVocabEntry("hello", -0.5))

    var model = Unigram(vocab, 0, False)
    expect_str(model.token_for_id(1), "hello", "id 1 should be 'hello'")
    expect_str(model.token_for_id(999), "", "unknown id should be empty")


def test_empty_vocab_raises() raises:
    """Test that empty vocabulary raises error."""
    var vocab = List[UnigramVocabEntry]()
    var raised = False
    try:
        var model = Unigram(vocab, 0, False)
    except:
        raised = True
    expect(raised, "should raise on empty vocab")


def test_unk_id_out_of_range_raises() raises:
    """Test that out-of-range unk_id raises error."""
    var vocab = List[UnigramVocabEntry]()
    vocab.append(UnigramVocabEntry("<unk>", 0.0))
    var raised = False
    try:
        var model = Unigram(vocab, 5, False)  # unk_id=5 is out of range
    except:
        raised = True
    expect(raised, "should raise on out-of-range unk_id")


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
    cases.append("test_empty_vocab_raises")
    cases.append("test_unk_id_out_of_range_raises")

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
            elif name == "test_empty_vocab_raises":
                test_empty_vocab_raises()
            elif name == "test_unk_id_out_of_range_raises":
                test_unk_id_out_of_range_raises()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
