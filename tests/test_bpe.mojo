"""Unit tests for the BPE model (vocab/merges loading + greedy merge).

Run:  mojo run -I src -I tests tests/test_bpe.mojo
"""

import bpe

from bpe import BPE
from fixtures import build_gpt2_minimal
from harness import expect, expect_eq, expect_str, summarize


def tokens_equal(a: List[String], b: List[String]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def expect_tokens(a: List[String], b: List[String], msg: String) raises:
    var sa = String()
    var sb = String()
    for t in a:
        sa += t + "|"
    for t in b:
        sb += t + "|"
    expect(
        tokens_equal(a, b), msg + " (expected [" + sb + "], got [" + sa + "])"
    )


# ---------------------------------------------------------------------------
# vocab / merges loading
# ---------------------------------------------------------------------------


def test_load_vocab_and_token_for_id() raises:
    var model = BPE()
    var vocab = List[String]()
    vocab.append("a")
    vocab.append("b")
    vocab.append("ab")
    model.load_vocab(vocab)
    expect_str(model.token_for_id(0), "a", "id 0 -> a")
    expect_str(model.token_for_id(2), "ab", "id 2 -> ab")
    expect_eq(model.token_id("b"), 1, "token b -> id 1")
    expect_eq(model.token_id("zz"), -1, "unknown token -> -1")


def test_load_vocab_pairs() raises:
    var model = BPE()
    var keys = List[String]()
    keys.append("x")
    keys.append("y")
    var ids = List[Int]()
    ids.append(100)
    ids.append(200)
    model.load_vocab_pairs(keys, ids)
    expect_eq(model.token_id("x"), 100, "x -> 100")
    expect_eq(model.token_id("y"), 200, "y -> 200")
    expect_str(model.token_for_id(100), "x", "100 -> x")


def test_load_merges_skips_header_and_empty() raises:
    var model = BPE()
    var lines = List[String]()
    lines.append("#version: 0.2")
    lines.append("")
    lines.append("a b")
    lines.append("b c")
    model.load_merges(lines)
    expect_eq(model.merge_rank("a", "b"), 0, "first merge rank 0")
    expect_eq(model.merge_rank("b", "c"), 1, "second merge rank 1")
    expect_eq(model.merge_rank("a", "c"), -1, "absent pair -> -1")


# ---------------------------------------------------------------------------
# greedy merge encoding (encode_word)
# ---------------------------------------------------------------------------


def test_encode_word_hello() raises:
    var model = build_gpt2_minimal()
    var parts = model.encode_word("Ġhello")
    expect_tokens(parts, ["Ġhello"], "Ġhello should merge into one token")


def test_encode_word_hello_bare() raises:
    var model = build_gpt2_minimal()
    var parts = model.encode_word("hello")
    expect_tokens(parts, ["hello"], "bare hello should merge via bare path")


def test_encode_word_world() raises:
    var model = build_gpt2_minimal()
    var parts = model.encode_word("Ġworld")
    expect_tokens(parts, ["Ġworld"], "Ġworld should merge into one token")


def test_encode_word_partial_merge() raises:
    # 'abcd' has no merges -> stays as individual chars.
    var model = build_gpt2_minimal()
    var parts = model.encode_word("abcd")
    expect_tokens(parts, ["a", "b", "c", "d"], "no merges -> chars kept")


def test_encode_word_empty() raises:
    var model = build_gpt2_minimal()
    var parts = model.encode_word("")
    expect_eq(len(parts), 0, "empty word -> no parts")


def test_encode_word_rank_ordering_hello_wins() raises:
    # In "Ġhhe" the only merge is 'Ġ h' (rank 0) -> Ġh, then 'h e'
    # (rank 10) -> he, giving [Ġh, h, e]?  Trace: Ġ h h e ->
    #   pairs: Ġh(0), hh(none), he(10) -> best Ġh -> Ġh h e
    #   pairs: Ġhh(none), he(10) -> best he -> Ġh he
    #   pairs: Ġhhe(none) -> stop
    # => [Ġh, he]
    var model = build_gpt2_minimal()
    var parts = model.encode_word("Ġhhe")
    expect_tokens(parts, ["Ġh", "he"], "greedy merges Ġh then he")


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------


def test_encode_ids_and_tokens() raises:
    var model = build_gpt2_minimal()
    var ids = model.encode("Ġhello")
    expect_eq(len(ids), 1, "one id")
    expect_eq(ids[0], model.token_id("Ġhello"), "id matches Ġhello")
    expect_str(model.token_for_id(ids[0]), "Ġhello", "token matches")


def test_decode_roundtrip() raises:
    var model = build_gpt2_minimal()
    var ids = model.encode("ĠhelloĠworld")
    var text = model.decode(ids)
    expect_str(text, "ĠhelloĠworld", "decode reverses encode")


def test_encode_unknown_token_raises() raises:
    # A model that does NOT include the full byte alphabet must raise on
    # an out-of-vocab token instead of silently dropping it (the original
    # silent-skip behaviour corrupted roundtrips -- see git history).
    var model = BPE()
    var vocab = List[String]()
    vocab.append("a")
    model.load_vocab(vocab)
    var raised = False
    try:
        var ids = model.encode("ab")  # 'b' is not in this vocab
        _ = ids
    except:
        raised = True
    expect(raised, "encode of unknown token must raise")


def test_decode_empty() raises:
    var model = build_gpt2_minimal()
    var ids = List[Int]()
    expect_str(model.decode(ids), "", "empty ids -> empty string")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_load_vocab_and_token_for_id")
    cases.append("test_load_vocab_pairs")
    cases.append("test_load_merges_skips_header_and_empty")
    cases.append("test_encode_word_hello")
    cases.append("test_encode_word_hello_bare")
    cases.append("test_encode_word_world")
    cases.append("test_encode_word_partial_merge")
    cases.append("test_encode_word_empty")
    cases.append("test_encode_word_rank_ordering_hello_wins")
    cases.append("test_encode_ids_and_tokens")
    cases.append("test_decode_roundtrip")
    cases.append("test_encode_unknown_token_raises")
    cases.append("test_decode_empty")
    for name in cases:
        try:
            if name == "test_load_vocab_and_token_for_id":
                test_load_vocab_and_token_for_id()
            elif name == "test_load_vocab_pairs":
                test_load_vocab_pairs()
            elif name == "test_load_merges_skips_header_and_empty":
                test_load_merges_skips_header_and_empty()
            elif name == "test_encode_word_hello":
                test_encode_word_hello()
            elif name == "test_encode_word_hello_bare":
                test_encode_word_hello_bare()
            elif name == "test_encode_word_world":
                test_encode_word_world()
            elif name == "test_encode_word_partial_merge":
                test_encode_word_partial_merge()
            elif name == "test_encode_word_empty":
                test_encode_word_empty()
            elif name == "test_encode_word_rank_ordering_hello_wins":
                test_encode_word_rank_ordering_hello_wins()
            elif name == "test_encode_ids_and_tokens":
                test_encode_ids_and_tokens()
            elif name == "test_decode_roundtrip":
                test_decode_roundtrip()
            elif name == "test_encode_unknown_token_raises":
                test_encode_unknown_token_raises()
            elif name == "test_decode_empty":
                test_decode_empty()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
