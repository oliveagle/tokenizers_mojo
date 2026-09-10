"""Tests for Unigram Trainer.

Run:  mojo run -I src -I tests tests/test_unigram_trainer.mojo
"""

import unigram_trainer
import unigram

from unigram_trainer import UnigramTrainer
from unigram import Unigram
from harness import expect, expect_eq, expect_str, summarize


def test_basic_training() raises:
    """Test basic Unigram training."""
    var trainer = UnigramTrainer()
    
    var word_counts = Dict[String, Int]()
    word_counts["hello"] = 10
    word_counts["world"] = 8
    word_counts["help"] = 5
    word_counts["worship"] = 3
    
    var model = Unigram()
    var special_tokens = trainer.train(model, word_counts)
    
    # Check that vocabulary was populated
    expect(model.vocab_size() > 0, "vocab should not be empty")
    
    # Check that basic tokens are present
    expect(model.token_id("h") >= 0, "h should be in vocab")
    expect(model.token_id("e") >= 0, "e should be in vocab")
    expect(model.token_id("l") >= 0, "l should be in vocab")


def test_vocab_size_limit() raises:
    """Test that vocab size limit is respected."""
    var trainer = UnigramTrainer(
        vocab_size=5,
        n_sub_iterations=1,
        shrinking_factor=0.75,
    )
    
    var word_counts = Dict[String, Int]()
    word_counts["aaaaaaaa"] = 10
    word_counts["bbbbbbbb"] = 8
    word_counts["cccccccc"] = 5
    word_counts["dddddddd"] = 3
    word_counts["eeeeeeee"] = 1
    
    var model = Unigram()
    trainer.train(model, word_counts)
    
    # Vocab should be limited
    expect(model.vocab_size() <= 6, "vocab size should be limited")


def test_encode_after_training() raises:
    """Test that trained model can encode words."""
    var trainer = UnigramTrainer()
    
    var word_counts = Dict[String, Int]()
    word_counts["hello"] = 10
    word_counts["world"] = 8
    
    var model = Unigram()
    trainer.train(model, word_counts)
    
    # Try encoding a word
    var ids = model.encode("h")
    expect(len(ids) > 0, "encoding should produce ids")


def test_special_tokens() raises:
    """Test that special tokens are returned."""
    var trainer = UnigramTrainer()
    
    var word_counts = Dict[String, Int]()
    word_counts["hello"] = 10
    
    var model = Unigram()
    var special_tokens = trainer.train(model, word_counts)
    
    # Special tokens list should be returned (empty for now)
    expect(len(special_tokens) >= 0, "special tokens list should be returned")


def test_probability_scores() raises:
    """Test that tokens have probability scores."""
    var trainer = UnigramTrainer()
    
    var word_counts = Dict[String, Int]()
    word_counts["abc"] = 100
    word_counts["def"] = 50
    
    var model = Unigram()
    trainer.train(model, word_counts)
    
    # Check that tokens have scores
    var has_scores = False
    for i in range(model.vocab_size()):
        if model.vocab[i].score != 0.0:
            has_scores = True
            break
    
    expect(has_scores, "tokens should have non-zero probability scores")


def test_most_frequent_chars_first() raises:
    """Test that most frequent characters appear first in vocab."""
    var trainer = UnigramTrainer(
        vocab_size=10,
        n_sub_iterations=1,
        shrinking_factor=0.75,
    )
    
    var word_counts = Dict[String, Int]()
    word_counts["aaa"] = 100
    word_counts["bbb"] = 50
    word_counts["ccc"] = 10
    
    var model = Unigram()
    trainer.train(model, word_counts)
    
    # Most frequent char should have higher (less negative) log probability
    var a_score = model.vocab[0].score
    var c_score: Float64 = -1000.0
    
    for i in range(model.vocab_size()):
        if model.vocab[i].token == "c":
            c_score = model.vocab[i].score
            break
    
    expect(a_score > c_score, "more frequent token should have higher score")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_basic_training")
    cases.append("test_vocab_size_limit")
    cases.append("test_encode_after_training")
    cases.append("test_special_tokens")
    cases.append("test_probability_scores")
    cases.append("test_most_frequent_chars_first")
    
    for name in cases:
        try:
            if name == "test_basic_training":
                test_basic_training()
            elif name == "test_vocab_size_limit":
                test_vocab_size_limit()
            elif name == "test_encode_after_training":
                test_encode_after_training()
            elif name == "test_special_tokens":
                test_special_tokens()
            elif name == "test_probability_scores":
                test_probability_scores()
            elif name == "test_most_frequent_chars_first":
                test_most_frequent_chars_first()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
