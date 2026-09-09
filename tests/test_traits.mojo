"""Unit tests for the trait/interface abstraction (Phase 2).

Proves that Normalizer / PreTokenizer / Model / Decoder traits can be
used together in generic pipeline helpers with concrete implementations.

Run:  mojo run -I src -I tests tests/test_traits.mojo
"""

import bpe
import byte_level
import normalizers
import split
import whitespace

from bpe import BPE
from byte_level import ByteLevelDecoder
from normalizers import Lowercase, Strip
from whitespace import Whitespace, Metaspace
from split import SplitPreTokenizer
from fixtures import build_gpt2_minimal
from traits import Normalizer, PreTokenizer, Model, Decoder
from harness import expect, expect_eq, expect_str, summarize


def pipeline[
    N: Normalizer, P: PreTokenizer, M: Model
](n: N, p: P, m: M, text: String) raises -> List[Int]:
    """Generic: normalize -> pre-tokenize -> model.encode -> ids."""
    var norm = n.normalize(text)
    var toks = p.pre_tokenize(norm)
    var ids = List[Int]()
    for t in toks:
        var part = m.encode(t)
        for id in part:
            ids.append(id)
    return ids^


def decode_via[D: Decoder](d: D, tokens: List[String]) raises -> String:
    """Generic decoder dispatch."""
    return d.decode(tokens)


def test_generic_lowercase_whitespace_bpe() raises:
    var model = build_gpt2_minimal()
    var ids = pipeline(Lowercase(), Whitespace(), model, "Hello world")
    expect_eq(len(ids), 2, "two ids")
    expect_eq(ids[0], model.token_id("hello"), "hello id")
    expect_eq(ids[1], model.token_id("world"), "world id")


def test_generic_strip_metaspace_bpe() raises:
    # Metaspace with replacement "Ġ" produces the same tokens as the GPT-2
    # byte-level fixture (which maps spaces to Ġ), so BPE can encode them.
    var model = build_gpt2_minimal()
    var ms = Metaspace("Ġ", True, True)
    var ids = pipeline(Strip(), ms, model, "  hello world  ")
    expect_eq(len(ids), 2, "two ids after strip+metaspace")
    expect_eq(ids[0], model.token_id("Ġhello"), "metaspace hello")
    expect_eq(ids[1], model.token_id("Ġworld"), "metaspace world")


def test_generic_decoder() raises:
    var dec = ByteLevelDecoder()
    var toks = List[String]()
    toks.append("Ġhello")
    toks.append("Ġworld")
    expect_str(decode_via(dec, toks), " hello world", "generic decode")


def test_trait_objects_are_usable_directly() raises:
    var n = Lowercase()
    var p = Whitespace()
    var m = build_gpt2_minimal()
    expect_str(n.normalize("ABC"), "abc", "normalizer trait method")
    var toks = p.pre_tokenize("a b")
    expect_eq(len(toks), 2, "pretokenizer trait method")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_generic_lowercase_whitespace_bpe")
    cases.append("test_generic_strip_metaspace_bpe")
    cases.append("test_generic_decoder")
    cases.append("test_trait_objects_are_usable_directly")
    for name in cases:
        try:
            if name == "test_generic_lowercase_whitespace_bpe":
                test_generic_lowercase_whitespace_bpe()
            elif name == "test_generic_strip_metaspace_bpe":
                test_generic_strip_metaspace_bpe()
            elif name == "test_generic_decoder":
                test_generic_decoder()
            elif name == "test_trait_objects_are_usable_directly":
                test_trait_objects_are_usable_directly()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
