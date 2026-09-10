"""Tests for WordLevel Trainer.

Run:  mojo run -I src -I tests tests/test_wordlevel_trainer.mojo
"""

import wordlevel_trainer
import wordlevel

from wordlevel_trainer import WordLevelTrainer
from wordlevel import WordLevel
from harness import expect, expect_eq, expect_str, summarize


def test_basic_training() raises:
    """Test basic WordLevel training."""
    var trainer = WordLevelTrainer()
    
    var word_counts = Dict[String, Int]()
    word_counts["the"] = 25
    word_counts["roses"] = 22
    word_counts["are"] = 24
    word_counts["red"] = 12
    word_counts["violets"] = 10
    word_counts["blue"] = 16
    
    var model = WordLevel()
    trainer.train(model, word_counts)
    
    # Check that vocabulary was populated
    expect(model.vocab_size() > 0, "vocab should not be empty")
    
    # Most frequent words should be first
    expect(model.token_id("the") == 0, "the should have id 0")
    expect(model.token_id("are") == 1, "are should have id 1")
    expect(model.token_id("roses") == 2, "roses should have id 2")


def test_vocab_size_limit() raises:
    """Test that vocab size limit is respected."""
    var trainer = WordLevelTrainer(
        vocab_size=5,
        min_frequency=0,
    )
    
    var word_counts = Dict[String, Int]()
    word_counts["a"] = 10
    word_counts["b"] = 8
    word_counts["c"] = 5
    word_counts["d"] = 3
    word_counts["e"] = 1
    word_counts["f"] = 0  # Should be excluded
    
    var model = WordLevel()
    trainer.train(model, word_counts)
    
    expect(model.vocab_size() <= 5, "vocab size should be limited to 5")


def test_min_frequency() raises:
    """Test that min_frequency filter works."""
    var trainer = WordLevelTrainer(
        vocab_size=100,
        min_frequency=15,
    )
    
    var word_counts = Dict[String, Int]()
    word_counts["frequent"] = 20
    word_counts["rare"] = 10  # Below min_frequency
    
    var model = WordLevel()
    trainer.train(model, word_counts)
    
    expect(model.token_id("frequent") >= 0, "frequent should be in vocab")
    expect(model.token_id("rare") == -1, "rare should not be in vocab")


def test_sorted_by_frequency() raises:
    """Test that words are sorted by frequency."""
    var trainer = WordLevelTrainer()
    
    var word_counts = Dict[String, Int]()
    word_counts["low"] = 5
    word_counts["high"] = 100
    word_counts["medium"] = 50
    
    var model = WordLevel()
    trainer.train(model, word_counts)
    
    # Check ordering by frequency
    expect(model.token_id("high") < model.token_id("medium"), "high should come before medium")
    expect(model.token_id("medium") < model.token_id("low"), "medium should come before low")


def test_deterministic_order() raises:
    """Test that words with same frequency are sorted alphabetically."""
    var trainer = WordLevelTrainer()
    
    var word_counts = Dict[String, Int]()
    word_counts["banana"] = 10
    word_counts["apple"] = 10
    word_counts["cherry"] = 10
    
    var model = WordLevel()
    trainer.train(model, word_counts)
    
    # With same frequency, should be sorted alphabetically
    expect(model.token_id("apple") < model.token_id("banana"), "apple should come before banana")
    expect(model.token_id("banana") < model.token_id("cherry"), "banana should come before cherry")


def test_encode_after_training() raises:
    """Test that trained model can encode words."""
    var trainer = WordLevelTrainer()
    
    var word_counts = Dict[String, Int]()
    word_counts["hello"] = 10
    word_counts["world"] = 8
    
    var model = WordLevel()
    trainer.train(model, word_counts)
    
    var ids = model.encode("hello")
    expect(len(ids) == 1, "encoding should produce 1 id")
    expect(ids[0] == 0, "hello should have id 0")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_basic_training")
    cases.append("test_vocab_size_limit")
    cases.append("test_min_frequency")
    cases.append("test_sorted_by_frequency")
    cases.append("test_deterministic_order")
    cases.append("test_encode_after_training")
    
    for name in cases:
        try:
            if name == "test_basic_training":
                test_basic_training()
            elif name == "test_vocab_size_limit":
                test_vocab_size_limit()
            elif name == "test_min_frequency":
                test_min_frequency()
            elif name == "test_sorted_by_frequency":
                test_sorted_by_frequency()
            elif name == "test_deterministic_order":
                test_deterministic_order()
            elif name == "test_encode_after_training":
                test_encode_after_training()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
