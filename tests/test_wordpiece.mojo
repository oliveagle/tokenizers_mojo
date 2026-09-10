"""Tests for WordPiece model.

Run:  mojo run -I src -I tests tests/test_wordpiece.mojo
"""

import wordpiece

from wordpiece import WordPiece
from harness import expect, expect_eq, expect_str, summarize


def test_basic_tokenization() raises:
    """Test basic WordPiece tokenization with a simple vocabulary."""
    var wp = WordPiece()
    var lines = List[String]()
    lines.append("[UNK]")
    lines.append("hello")
    lines.append("world")
    lines.append("##llo")
    lines.append("##rld")
    lines.append("he")
    wp.load_vocab_from_lines(lines)

    # "hello" should be tokenized as ["hello"]
    var tokens = wp.tokenize_word("hello")
    expect_eq(len(tokens), 1, "hello should have 1 token")
    expect_str(tokens[0], "hello", "token should be 'hello'")

    # "world" should be tokenized as ["world"]
    tokens = wp.tokenize_word("world")
    expect_eq(len(tokens), 1, "world should have 1 token")
    expect_str(tokens[0], "world", "token should be 'world'")


def test_continuing_subword() raises:
    """Test that continuing subwords get ## prefix."""
    var wp = WordPiece()
    var lines = List[String]()
    lines.append("[UNK]")
    lines.append("un")
    lines.append("##happ")
    lines.append("##y")
    lines.append("happy")
    wp.load_vocab_from_lines(lines)

    # "unhappy" should be tokenized as ["un", "##happ", "##y"]
    var tokens = wp.tokenize_word("unhappy")
    expect_eq(len(tokens), 3, "unhappy should have 3 tokens")
    expect_str(tokens[0], "un", "first token should be 'un'")
    expect_str(tokens[1], "##happ", "second token should be '##happ'")
    expect_str(tokens[2], "##y", "third token should be '##y'")


def test_unknown_token() raises:
    """Test that unknown words return [UNK]."""
    var wp = WordPiece()
    var lines = List[String]()
    lines.append("[UNK]")
    lines.append("hello")
    wp.load_vocab_from_lines(lines)

    # "xyz" is not in vocab and cannot be split, should return [UNK]
    var tokens = wp.tokenize_word("xyz")
    expect_eq(len(tokens), 1, "unknown word should have 1 token")
    expect_str(tokens[0], "[UNK]", "token should be '[UNK]'")


def test_max_input_chars() raises:
    """Test that words exceeding max_input_chars_per_word return [UNK]."""
    var wp = WordPiece()
    wp.max_input_chars_per_word = 5
    var lines = List[String]()
    lines.append("[UNK]")
    lines.append("a")
    wp.load_vocab_from_lines(lines)

    # Word longer than 5 chars should return [UNK]
    var long_word = String("abcdefghijklmnop")
    var tokens = wp.tokenize_word(long_word)
    expect_eq(len(tokens), 1, "long word should have 1 token")
    expect_str(tokens[0], "[UNK]", "token should be '[UNK]'")


def test_encode() raises:
    """Test encoding words to vocab ids."""
    var wp = WordPiece()
    var lines = List[String]()
    lines.append("[UNK]")
    lines.append("hello")
    lines.append("##llo")
    lines.append("world")
    wp.load_vocab_from_lines(lines)

    # "hello" -> [1]
    var ids = wp.encode("hello")
    expect_eq(len(ids), 1, "hello should encode to 1 id")
    expect_eq(ids[0], 1, "id should be 1")

    # "llo" is not in vocab, should return [UNK]
    ids = wp.encode("llo")
    expect_eq(len(ids), 1, "llo should encode to 1 id (UNK)")
    expect_eq(ids[0], 0, "id should be 0 (UNK)")

    # "hello" with ## prefix should work
    ids = wp.encode("hello")
    expect_eq(len(ids), 1, "hello should encode to 1 id")
    expect_eq(ids[0], 1, "id should be 1")


def test_decode() raises:
    """Test decoding vocab ids back to tokens."""
    var wp = WordPiece()
    var lines = List[String]()
    lines.append("[UNK]")
    lines.append("hello")
    lines.append("##llo")
    wp.load_vocab_from_lines(lines)

    # [1, 2] -> "hello##llo"
    var ids = List[Int]()
    ids.append(1)
    ids.append(2)
    var result = wp.decode(ids)
    expect_str(result, "hello##llo", "decoded should be 'hello##llo'")


def test_vocab_size() raises:
    """Test vocab_size returns correct count."""
    var wp = WordPiece()
    var lines = List[String]()
    lines.append("[UNK]")
    lines.append("hello")
    lines.append("world")
    wp.load_vocab_from_lines(lines)

    expect_eq(wp.vocab_size(), 3, "vocab size should be 3")


def test_token_id() raises:
    """Test token_id lookup."""
    var wp = WordPiece()
    var lines = List[String]()
    lines.append("[UNK]")
    lines.append("hello")
    wp.load_vocab_from_lines(lines)

    expect_eq(wp.token_id("hello"), 1, "hello should have id 1")
    expect_eq(wp.token_id("world"), -1, "unknown should have id -1")


def test_token_for_id() raises:
    """Test token_for_id reverse lookup."""
    var wp = WordPiece()
    var lines = List[String]()
    lines.append("[UNK]")
    lines.append("hello")
    wp.load_vocab_from_lines(lines)

    expect_str(wp.token_for_id(1), "hello", "id 1 should be 'hello'")
    expect_str(wp.token_for_id(999), "", "unknown id should be empty")


def test_load_vocab_from_lines() raises:
    """Test loading vocab from lines (BERT vocab.txt format)."""
    var lines = List[String]()
    lines.append("[UNK]")
    lines.append("[CLS]")
    lines.append("[SEP]")
    lines.append("hello")
    lines.append("world")

    var wp = WordPiece()
    wp.load_vocab_from_lines(lines)

    expect_eq(wp.vocab_size(), 5, "vocab size should be 5")
    expect_eq(wp.token_id("[CLS]"), 1, "[CLS] should have id 1")
    expect_eq(wp.token_id("hello"), 3, "hello should have id 3")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_basic_tokenization")
    cases.append("test_continuing_subword")
    cases.append("test_unknown_token")
    cases.append("test_max_input_chars")
    cases.append("test_encode")
    cases.append("test_decode")
    cases.append("test_vocab_size")
    cases.append("test_token_id")
    cases.append("test_token_for_id")
    cases.append("test_load_vocab_from_lines")

    for name in cases:
        try:
            if name == "test_basic_tokenization":
                test_basic_tokenization()
            elif name == "test_continuing_subword":
                test_continuing_subword()
            elif name == "test_unknown_token":
                test_unknown_token()
            elif name == "test_max_input_chars":
                test_max_input_chars()
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
            elif name == "test_load_vocab_from_lines":
                test_load_vocab_from_lines()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
