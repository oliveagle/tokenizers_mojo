"""Unit tests for the BPE Trainer (Phase 3).

Reference values are produced by HuggingFace `tokenizers` (Rust) for the
same corpus:

    low×5  lower×2  lowest×1  newest×9  new×3

Alignment rule (per docs/goals/goals.md §5): identical vocab + merge
sequence == aligned.  For configs where upstream's own id order is
non-deterministic (Rust HashMap iteration), we assert the vocab **set**
plus the exact merge sequence, and pin exact ids only where upstream is
deterministic (no prefix/suffix config).

Run:  mojo run -I src -I tests tests/test_bpe_trainer.mojo
"""

import bpe
import bpe_trainer

from bpe import BPE
from bpe_trainer import TrainerConfig, BpeTrainer
from harness import expect, expect_eq, expect_str, summarize


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def build_corpus() raises -> Dict[String, Int]:
    var wc = Dict[String, Int]()
    wc["low"] = 5
    wc["lower"] = 2
    wc["lowest"] = 1
    wc["newest"] = 9
    wc["new"] = 3
    return wc^


def expect_vocab_set(model: BPE, expected: List[String], msg: String) raises:
    """Assert `model` contains exactly the `expected` tokens (any order)."""
    expect(
        len(model.vocab) == len(expected),
        msg + ": vocab size (got " + String(len(model.vocab)) + ")",
    )
    for t in expected:
        expect(
            t in model.vocab,
            msg + ": missing token '" + t + "'",
        )


def expect_merges(model: BPE, expected: List[String], msg: String) raises:
    """Assert merge rank order equals `expected` ("a b" pairs)."""
    expect(
        len(model.merges) == len(expected),
        msg + ": merge count (got " + String(len(model.merges)) + ")",
    )
    for rank in range(len(expected)):
        var got = model.merges.get(expected[rank], -999)
        expect_eq(
            got,
            rank,
            msg + ": rank " + String(rank) + " = '" + expected[rank] + "'",
        )


# ---------------------------------------------------------------------------
# 1. minimal config (upstream-deterministic: exact ids + merges)
# ---------------------------------------------------------------------------


def test_minimal_vocab_exact() raises:
    var cfg = TrainerConfig(vocab_size=30, min_frequency=1)
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)

    var expected = List[String]()
    for t in [
        "e",
        "l",
        "n",
        "o",
        "r",
        "s",
        "t",
        "w",
        "ew",
        "new",
        "es",
        "est",
        "newest",
        "lo",
        "low",
        "er",
        "lower",
        "lowest",
    ]:
        expected.append(t)

    expect_eq(len(model.vocab), 18, "minimal vocab size")
    for i in range(18):
        expect_str(
            model.token_for_id(i),
            expected[i],
            "minimal id " + String(i),
        )


def test_minimal_merges_exact() raises:
    var cfg = TrainerConfig(vocab_size=30, min_frequency=1)
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)

    var expected = List[String]()
    for t in [
        "e w",
        "n ew",
        "e s",
        "es t",
        "new est",
        "l o",
        "lo w",
        "e r",
        "low er",
        "low est",
    ]:
        expected.append(t)
    expect_merges(model, expected, "minimal merges")


def test_minimal_encode() raises:
    var cfg = TrainerConfig(vocab_size=30, min_frequency=1)
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)

    # HF ids for the same corpus (minimal config)
    var cases = List[Tuple[String, Int]]()
    cases.append(("low", 14))
    cases.append(("lower", 16))
    cases.append(("lowest", 17))
    cases.append(("newest", 12))
    cases.append(("new", 9))
    cases.append(("lo", 13))
    for i in range(len(cases)):
        var word = cases[i][0]
        var want = cases[i][1]
        var toks = model.encode_word(word)
        expect_eq(len(toks), 1, word + ": single token")
        var got = model.token_id(toks[0])
        expect_eq(
            got, want, word + ": id " + String(got) + " want " + String(want)
        )


# ---------------------------------------------------------------------------
# 2. end_of_word_suffix (</w>, GPT-2 style) — set + merge sequence
# ---------------------------------------------------------------------------


def test_eow_vocab_set() raises:
    var cfg = TrainerConfig(
        vocab_size=30, min_frequency=1, end_of_word_suffix="</w>"
    )
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)

    var expected = List[String]()
    for t in [
        "e",
        "l",
        "n",
        "o",
        "r",
        "s",
        "t",
        "w",
        "t</w>",
        "w</w>",
        "r</w>",
        "ne",
        "we",
        "st</w>",
        "west</w>",
        "newest</w>",
        "lo",
        "low</w>",
        "new</w>",
        "wer</w>",
        "lower</w>",
        "lowest</w>",
    ]:
        expected.append(t)
    expect_vocab_set(model, expected, "eow vocab")


def test_eow_merges() raises:
    var cfg = TrainerConfig(
        vocab_size=30, min_frequency=1, end_of_word_suffix="</w>"
    )
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)

    var expected = List[String]()
    for t in [
        "n e",
        "w e",
        "s t</w>",
        "we st</w>",
        "ne west</w>",
        "l o",
        "lo w</w>",
        "ne w</w>",
        "we r</w>",
        "lo wer</w>",
        "lo west</w>",
    ]:
        expected.append(t)
    expect_merges(model, expected, "eow merges")


# ---------------------------------------------------------------------------
# 3. continuing_subword_prefix (##, BERT style) — set + merge sequence
# ---------------------------------------------------------------------------


def test_bert_vocab_set() raises:
    var cfg = TrainerConfig(
        vocab_size=30, min_frequency=1, continuing_subword_prefix="##"
    )
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)

    var expected = List[String]()
    for t in [
        "e",
        "l",
        "n",
        "o",
        "r",
        "s",
        "t",
        "w",
        "##e",
        "##o",
        "##r",
        "##s",
        "##t",
        "##w",
        "ne",
        "lo",
        "low",
        "##we",
        "##st",
        "##west",
        "newest",
        "new",
        "##wer",
        "lower",
        "lowest",
    ]:
        expected.append(t)
    expect_vocab_set(model, expected, "bert vocab")


def test_bert_merges() raises:
    var cfg = TrainerConfig(
        vocab_size=30, min_frequency=1, continuing_subword_prefix="##"
    )
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)

    var expected = List[String]()
    for t in [
        "n ##e",
        "##w ##e",
        "##s ##t",
        "##we ##st",
        "ne ##west",
        "l ##o",
        "lo ##w",
        "ne ##w",
        "##we ##r",
        "lo ##wer",
        "lo ##west",
    ]:
        expected.append(t)
    expect_merges(model, expected, "bert merges")


# ---------------------------------------------------------------------------
# 4. special tokens / initial alphabet / min_frequency / limit_alphabet
# ---------------------------------------------------------------------------


def test_special_tokens_lowest_ids() raises:
    var cfg = TrainerConfig(vocab_size=30, min_frequency=1)
    cfg.add_special_token("<s>")
    cfg.add_special_token("</s>")
    cfg.add_special_token("<unk>")
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)

    expect_eq(model.token_id("<s>"), 0, "<s> id 0")
    expect_eq(model.token_id("</s>"), 1, "</s> id 1")
    expect_eq(model.token_id("<unk>"), 2, "<unk> id 2")
    expect_str(model.token_for_id(3), "e", "first alphabet char id 3")


def test_initial_alphabet_preserved() raises:
    var cfg = TrainerConfig(vocab_size=30, min_frequency=1)
    cfg.add_initial_alphabet("z")  # not in corpus — must still appear
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)
    expect("z" in model.vocab, "initial alphabet 'z' preserved")


def test_min_frequency_stops_early() raises:
    var cfg = TrainerConfig(vocab_size=30000, min_frequency=4)
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)

    # Every merge must have had count >= 4. "e w" (count 12: newest 9 +
    # lower 2 + lowest 1) merges; low-count pairs never do.
    expect("e w" in model.merges, "e w merged (count 12 >= 4)")
    expect("lo w" in model.merges, "lo w merged (count 8 >= 4)")
    expect("e r" not in model.merges, "e r NOT merged (count 2 < 4)")
    expect("l o" in model.merges, "l o merged (count 8 >= 4)")


def test_limit_alphabet_keeps_top_chars() raises:
    var cfg = TrainerConfig(vocab_size=30, min_frequency=1, limit_alphabet=4)
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    trainer.train(build_corpus(), model)

    # Top-4 chars by count are {w(20), e(15), n(12), s|t(10 tie)}.
    # Upstream's own choice between s/t is non-deterministic; assert the
    # set is exactly {e,n,w} + one of {s,t} (total 4 alphabet chars).
    expect("e" in model.vocab, "e kept")
    expect("n" in model.vocab, "n kept")
    expect("w" in model.vocab, "w kept")
    var has_s = "s" in model.vocab
    var has_t = "t" in model.vocab
    expect(has_s or has_t, "one of s/t kept")
    expect(not (has_s and has_t), "not both s and t kept")


def test_empty_corpus_noop() raises:
    var cfg = TrainerConfig(vocab_size=30, min_frequency=1)
    var trainer = BpeTrainer(cfg)
    var model = BPE()
    var empty = Dict[String, Int]()
    trainer.train(empty, model)
    expect_eq(len(model.vocab), 0, "empty corpus -> empty vocab")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    var failures = List[String]()
    try:
        test_minimal_vocab_exact()
    except:
        failures.append("test_minimal_vocab_exact")
    try:
        test_minimal_merges_exact()
    except:
        failures.append("test_minimal_merges_exact")
    try:
        test_minimal_encode()
    except:
        failures.append("test_minimal_encode")
    try:
        test_eow_vocab_set()
    except:
        failures.append("test_eow_vocab_set")
    try:
        test_eow_merges()
    except:
        failures.append("test_eow_merges")
    try:
        test_bert_vocab_set()
    except:
        failures.append("test_bert_vocab_set")
    try:
        test_bert_merges()
    except:
        failures.append("test_bert_merges")
    try:
        test_special_tokens_lowest_ids()
    except:
        failures.append("test_special_tokens_lowest_ids")
    try:
        test_initial_alphabet_preserved()
    except:
        failures.append("test_initial_alphabet_preserved")
    try:
        test_min_frequency_stops_early()
    except:
        failures.append("test_min_frequency_stops_early")
    try:
        test_limit_alphabet_keeps_top_chars()
    except:
        failures.append("test_limit_alphabet_keeps_top_chars")
    try:
        test_empty_corpus_noop()
    except:
        failures.append("test_empty_corpus_noop")
    summarize(failures)
