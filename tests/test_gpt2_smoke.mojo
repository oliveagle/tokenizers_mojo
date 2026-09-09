"""End-to-end smoke test against REAL GPT-2 data.

Builds a BPE model from the embedded closure subset of the real GPT-2
vocab.json + merges.txt (see scripts/build_gpt2_smoke.py) and verifies
that our full pipeline (pre_tokenize -> byte_map -> greedy BPE) produces
exactly the same ids as HuggingFace `tokenizers` (add_prefix_space=True).

This is the Phase 1 acceptance item: '对同一段英文输入，编码出的 ids 与
上游 Rust tokenizers 完全一致'.

Run:  mojo run -I src -I tests tests/test_gpt2_smoke.mojo
"""

import byte_level
import bpe
import tokenizer

from bpe import BPE
from tokenizer import Tokenizer
from gpt2_smoke_data import (
    n_sentences,
    sentences,
    vocab_tokens,
    vocab_ids,
    merge_lines,
    ref_ids,
)
from harness import expect, expect_eq, expect_str, summarize


def build_real_model() raises -> BPE:
    var model = BPE()
    model.load_vocab_pairs(vocab_tokens(), vocab_ids())
    model.load_merges(merge_lines())
    return model^


def test_sentence_ids_match_reference() raises:
    var tok = Tokenizer(build_real_model())
    var texts = sentences()
    var n = n_sentences()
    expect_eq(len(texts), n, "sentence count")
    for i in range(n):
        var enc = tok.encode(texts[i])
        var ours = enc.get_ids()
        var expected = ref_ids(i)
        expect_eq(len(ours), len(expected), "id count for sentence " + String(i))
        for j in range(len(expected)):
            expect_eq(
                ours[j],
                expected[j],
                "sentence " + String(i) + " id[" + String(j) + "]",
            )
        print("  sentence", i, "OK:", texts[i])


def test_every_id_has_token() raises:
    # No unknown-token drop is allowed: every id must map back to a token.
    var tok = Tokenizer(build_real_model())
    var texts = sentences()
    for i in range(len(texts)):
        var enc = tok.encode(texts[i])
        for j in range(enc.len()):
            var t = tok.model.token_for_id(enc.id_at(j))
            expect(t != "", "id has a token (sentence " + String(i) + ")")
    print("  all ids map to tokens OK")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_sentence_ids_match_reference")
    cases.append("test_every_id_has_token")
    for name in cases:
        try:
            if name == "test_sentence_ids_match_reference":
                test_sentence_ids_match_reference()
            elif name == "test_every_id_has_token":
                test_every_id_has_token()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
