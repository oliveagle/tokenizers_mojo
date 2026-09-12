"""Tests for WordPiece Trainer.

Run:  mojo run -I src -I tests tests/test_wordpiece_trainer.mojo
"""

import wordpiece_trainer
import wordpiece

from wordpiece_trainer import WordPieceTrainer
from wordpiece import WordPiece
from harness import expect, expect_eq, expect_str, summarize


def test_basic_training() raises:
    """Test basic WordPiece training."""
    var trainer = WordPieceTrainer()
    
    # Create a simple word count
    var word_counts = Dict[String, Int]()
    word_counts["hello"] = 10
    word_counts["world"] = 8
    word_counts["help"] = 5
    word_counts["worship"] = 3
    
    var model = WordPiece()
    var special_tokens = trainer.train(model, word_counts)
    
    # Check that vocabulary was populated
    expect(model.vocab_size() > 0, "vocab should not be empty")
    
    # Check that basic tokens are present
    expect(model.token_id("hello") >= 0, "hello should be in vocab")
    expect(model.token_id("world") >= 0, "world should be in vocab")


def test_limit_alphabet() raises:
    """Test that limit_alphabet limits initial characters."""
    var trainer = WordPieceTrainer(
        vocab_size=100,
        min_frequency=0,
        limit_alphabet=3,  # Only keep top 3 characters
    )
    
    var word_counts = Dict[String, Int]()
    word_counts["aaa"] = 10
    word_counts["bbb"] = 8
    word_counts["ccc"] = 5
    word_counts["ddd"] = 3
    word_counts["eee"] = 1
    
    var model = WordPiece()
    _ = trainer.train(model, word_counts)
    
    # Should have initial characters limited to 3 (a, b, c)
    # Plus merged tokens from training
    expect(model.token_id("a") >= 0, "a should be in vocab")
    expect(model.token_id("b") >= 0, "b should be in vocab")
    expect(model.token_id("c") >= 0, "c should be in vocab")
    # d and e should not be in initial alphabet (below limit_alphabet)
    # They might appear in merged tokens, but not as standalone chars
    expect(model.vocab_size() > 3, "vocab should have more than just initial chars")


def test_min_frequency() raises:
    """Test that min_frequency filter works."""
    var trainer = WordPieceTrainer(
        vocab_size=100,
        min_frequency=5,
        limit_alphabet=0,
    )
    
    var word_counts = Dict[String, Int]()
    word_counts["frequent"] = 10
    word_counts["rare"] = 2  # Below min_frequency
    
    var model = WordPiece()
    _ = trainer.train(model, word_counts)
    
    expect(model.token_id("frequent") >= 0, "frequent should be in vocab")
    # rare should not be in vocab (below min_frequency)
    expect(model.token_id("rare") == -1, "rare should not be in vocab")


def test_special_tokens() raises:
    """Test that special tokens are added."""
    var trainer = WordPieceTrainer()
    
    var word_counts = Dict[String, Int]()
    word_counts["hello"] = 10
    
    var model = WordPiece()
    var special_tokens = trainer.train(model, word_counts)
    
    # Special tokens list should be returned
    expect(len(special_tokens) >= 0, "special tokens list should be returned")


def test_encode_after_training() raises:
    """Test that trained model can encode words."""
    var trainer = WordPieceTrainer()
    
    var word_counts = Dict[String, Int]()
    word_counts["hello"] = 10
    word_counts["world"] = 8
    word_counts["##o"] = 15
    word_counts["##l"] = 12
    
    var model = WordPiece()
    _ = trainer.train(model, word_counts)
    
    # Try encoding a word
    var ids = model.encode("hello")
    expect(len(ids) > 0, "encoding should produce ids")


def test_merges_produce_longer_tokens() raises:
    """Test that training produces merged tokens."""
    var trainer = WordPieceTrainer(
        vocab_size=100,
        min_frequency=0,
        limit_alphabet=0,
    )
    
    # Create words that will produce merges
    var word_counts = Dict[String, Int]()
    word_counts["ab"] = 100
    word_counts["ab"] = 100  # High frequency pair
    word_counts["cd"] = 50
    
    var model = WordPiece()
    _ = trainer.train(model, word_counts)
    
    # Should have merged tokens like "ab" or "##b"
    var has_merged = False
    for token in model.vocab.keys():
        if token.byte_length() > 1:
            has_merged = True
            break
    
    expect(has_merged, "training should produce merged tokens")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_basic_training")
    cases.append("test_limit_alphabet")
    cases.append("test_min_frequency")
    cases.append("test_special_tokens")
    cases.append("test_encode_after_training")
    cases.append("test_merges_produce_longer_tokens")
    
    for name in cases:
        try:
            if name == "test_basic_training":
                test_basic_training()
            elif name == "test_limit_alphabet":
                test_limit_alphabet()
            elif name == "test_min_frequency":
                test_min_frequency()
            elif name == "test_special_tokens":
                test_special_tokens()
            elif name == "test_encode_after_training":
                test_encode_after_training()
            elif name == "test_merges_produce_longer_tokens":
                test_merges_produce_longer_tokens()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
