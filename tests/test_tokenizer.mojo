"""Unit tests for the Tokenizer orchestrator (end-to-end pipeline).

Run:  mojo run -I src -I tests tests/test_tokenizer.mojo
"""

import tokenizer

from tokenizer import Tokenizer
from fixtures import build_gpt2_minimal
from harness import expect, expect_eq, expect_str, summarize


def test_encode_hello_world() raises:
    var tok = Tokenizer(build_gpt2_minimal())
    var enc = tok.encode("hello world")
    expect_eq(enc.len(), 2, "two tokens")
    expect_str(enc.token_at(0), "Ġhello", "first token")
    expect_str(enc.token_at(1), "Ġworld", "second token")
    expect_eq(
        enc.id_at(0),
        tok.model.token_id("Ġhello"),
        "first id matches vocab",
    )
    expect_eq(
        enc.id_at(1),
        tok.model.token_id("Ġworld"),
        "second id matches vocab",
    )


def test_decode_roundtrip() raises:
    var tok = Tokenizer(build_gpt2_minimal())
    var enc = tok.encode("hello world")
    var back = tok.decode(enc)
    expect_str(back, " hello world", "decode(encode(text)) roundtrips")


def test_add_prefix_space_semantics() raises:
    # "hello" without a leading space gets "Ġhello" (add_prefix_space).
    var tok = Tokenizer(build_gpt2_minimal())
    var enc = tok.encode("hello")
    expect_eq(enc.len(), 1, "single token")
    expect_str(enc.token_at(0), "Ġhello", "prefix space added")


def test_punctuation_roundtrip() raises:
    # ',', '!' are base byte-vocab entries, so the pipeline covers them.
    var tok = Tokenizer(build_gpt2_minimal())
    var enc = tok.encode("hello, world!")
    var back = tok.decode(enc)
    expect_str(back, " hello, world!", "punctuation roundtrips")


def test_offsets_are_monotonic() raises:
    var tok = Tokenizer(build_gpt2_minimal())
    var enc = tok.encode("hello world")
    var prev = 0
    var ok = True
    for i in range(enc.len()):
        var off = enc.offset_at(i)
        if off[0] < prev:
            ok = False
        prev = off[1]
    expect(ok, "offsets must be non-decreasing")


def test_empty_input() raises:
    var tok = Tokenizer(build_gpt2_minimal())
    var enc = tok.encode("")
    # add_prefix_space adds a space to empty input -> "Ġ" pre-token,
    # which byte-maps to the single 'Ġ' vocab entry.
    expect_eq(enc.len(), 1, "empty input yields the prefix-space token")
    expect_str(enc.token_at(0), "Ġ", "token is the byte-mapped space")


def test_uppercase_roundtrip() raises:
    var tok = Tokenizer(build_gpt2_minimal())
    var enc = tok.encode("Hello World")
    var back = tok.decode(enc)
    expect_str(back, " Hello World", "uppercase text roundtrips")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_encode_hello_world")
    cases.append("test_decode_roundtrip")
    cases.append("test_add_prefix_space_semantics")
    cases.append("test_punctuation_roundtrip")
    cases.append("test_offsets_are_monotonic")
    cases.append("test_empty_input")
    cases.append("test_uppercase_roundtrip")
    for name in cases:
        try:
            if name == "test_encode_hello_world":
                test_encode_hello_world()
            elif name == "test_decode_roundtrip":
                test_decode_roundtrip()
            elif name == "test_add_prefix_space_semantics":
                test_add_prefix_space_semantics()
            elif name == "test_punctuation_roundtrip":
                test_punctuation_roundtrip()
            elif name == "test_offsets_are_monotonic":
                test_offsets_are_monotonic()
            elif name == "test_empty_input":
                test_empty_input()
            elif name == "test_uppercase_roundtrip":
                test_uppercase_roundtrip()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
