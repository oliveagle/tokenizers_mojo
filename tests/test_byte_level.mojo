"""Unit tests for ByteLevel pre-tokenizer, decoder, and byte mapping.

Run:  mojo run -I src -I tests tests/test_byte_level.mojo
"""

import byte_level

from byte_level import (
    ByteMapping,
    ByteLevelPreTokenizer,
    ByteLevelDecoder,
    simple_gpt2_split,
)
from harness import expect, expect_eq, expect_str, summarize


# ---------------------------------------------------------------------------
# ByteMapping
# ---------------------------------------------------------------------------


def test_mapping_has_256_entries() raises:
    var m = ByteMapping()
    var count = 0
    for _ in m.b2u:
        count += 1
    expect_eq(count, 256, "byte mapping must cover all 256 bytes")


def test_mapping_printable_ascii_self_maps() raises:
    var m = ByteMapping()
    var b = 0x21
    while b <= 0x7E:
        expect_str(m.b2u[b], chr(b), "printable ASCII should map to itself")
        b += 1


def test_mapping_latin1_self_maps() raises:
    var m = ByteMapping()
    var b = 0xA1
    while b <= 0xAC:
        expect_str(m.b2u[b], chr(b), "Latin-1 A1-AC should map to itself")
        b += 1
    b = 0xAE
    while b <= 0xFF:
        expect_str(m.b2u[b], chr(b), "Latin-1 AE-FF should map to itself")
        b += 1


def test_mapping_space_is_gbar() raises:
    var m = ByteMapping()
    # Byte 0x20 (space) is not printable ASCII (starts at 0x21), so it maps
    # to codepoint 288 = U+0120 (Ġ), exactly as in GPT-2's encoder.py.
    expect_str(m.b2u[0x20], "Ġ", "byte 0x20 must map to Ġ")
    expect_eq(m.u2b["Ġ"], 0x20, "Ġ must map back to byte 0x20")


def test_mapping_inverse() raises:
    var m = ByteMapping()
    var b = 0
    while b < 256:
        var c = m.b2u[b]
        expect_eq(m.u2b[c], b, "u2b must be inverse of b2u")
        b += 1


def test_mapping_newline_maps_to_cbar() raises:
    var m = ByteMapping()
    # GPT-2 convention: \n (0x0A) -> U+010A (Ċ).
    expect_str(m.b2u[0x0A], "Ċ", "byte 0x0A must map to Ċ")
    expect_eq(m.u2b["Ċ"], 0x0A, "Ċ must map back to byte 0x0A")


# ---------------------------------------------------------------------------
# simple_gpt2_split
# ---------------------------------------------------------------------------


def test_split_single_leading_space_attaches() raises:
    var splits = simple_gpt2_split(" hello world")
    expect_eq(len(splits), 2, "expected two tokens")
    expect_str(splits[0], " hello", "leading space attaches to first word")
    expect_str(splits[1], " world", "second word")


def test_split_punctuation() raises:
    var splits = simple_gpt2_split("hello, world!")
    expect_eq(len(splits), 4, "expected four tokens")
    expect_str(splits[0], "hello", "word")
    expect_str(splits[1], ",", "comma")
    expect_str(splits[2], " world", "space-word")
    expect_str(splits[3], "!", "bang")


def test_split_digits() raises:
    # "v2 api" -> ["v", "2", " api"]: digit run "2" is its own token (no
    # leading space), then " api" (space + word) as a separate token.
    var splits = simple_gpt2_split("v2 api")
    expect_eq(len(splits), 3, "expected three tokens")
    expect_str(splits[0], "v", "letters stop at digit")
    expect_str(splits[1], "2", "digit run alone")
    expect_str(splits[2], " api", "space attaches to following word")


def test_split_multiple_spaces() raises:
    # Real GPT-2 regex: \s+(?!\S)|\s+ collapses runs of whitespace.
    # "a  b" -> ["a", "  ", "b"]: the two-space run is its own token
    # (not attached to "b"), matching upstream behaviour.
    var splits = simple_gpt2_split("a  b")
    expect_eq(len(splits), 3, "expected three tokens")
    expect_str(splits[0], "a", "first word")
    expect_str(splits[1], "  ", "two-space run is its own token")
    expect_str(splits[2], "b", "word after whitespace run")


# ---------------------------------------------------------------------------
# ByteLevelPreTokenizer
# ---------------------------------------------------------------------------


def test_pretokenizer_adds_prefix_space() raises:
    var pt = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True)
    var out = pt.pre_tokenize("hello")
    expect_eq(len(out), 1, "single pre-token")
    expect_str(out[0], "Ġhello", "space is byte-mapped to Ġ prefix")


def test_pretokenizer_no_prefix_when_present() raises:
    var pt = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True)
    var out = pt.pre_tokenize(" hello")
    expect_eq(len(out), 1, "single pre-token")
    expect_str(out[0], "Ġhello", "existing space maps to Ġ")


def test_pretokenizer_two_words() raises:
    var pt = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True)
    var out = pt.pre_tokenize("hello world")
    expect_eq(len(out), 2, "two pre-tokens")
    expect_str(out[0], "Ġhello", "first word byte-mapped")
    expect_str(out[1], "Ġworld", "second word byte-mapped")


def test_pretokenizer_utf8_roundtrip() raises:
    # 'é' is two bytes (0xC3 0xA9); byte-mapping inflates it to 2 chars.
    # Disable add_prefix_space here so the test isolates the UTF-8 path
    # from the add_prefix_space behaviour covered elsewhere.
    var pt = ByteLevelPreTokenizer(add_prefix_space=False, use_regex=True)
    var out = pt.pre_tokenize("héllo")
    expect_eq(len(out), 1, "single pre-token")
    var joined = String()
    for t in out:
        joined += t
    var dec = ByteLevelDecoder()
    var back = dec.decode_string(joined)
    expect_str(back, "héllo", "multi-byte UTF-8 survives byte roundtrip")


def test_pretokenizer_no_regex_no_split() raises:
    var pt = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=False)
    var out = pt.pre_tokenize("hello world")
    expect_eq(len(out), 1, "no split: single pre-token")
    expect_str(out[0], "ĠhelloĠworld", "whole text byte-mapped")


# ---------------------------------------------------------------------------
# ByteLevelDecoder
# ---------------------------------------------------------------------------


def test_decoder_roundtrip() raises:
    var dec = ByteLevelDecoder()
    var toks = List[String]()
    toks.append("Ġhello")
    toks.append("Ġworld")
    expect_str(dec.decode(toks), " hello world", "decode reassembles text")


def test_decoder_handles_unknown_char_fallback() raises:
    # A plain 'X' is not a byte-mapped char (0x58 maps to 'X' though).
    # Use a genuinely unmapped char (e.g. U+1F600) -> falls back to UTF-8.
    var dec = ByteLevelDecoder()
    var toks = List[String]()
    toks.append("a😀b")
    var out = dec.decode(toks)
    # Fallback path should preserve the UTF-8 bytes of the emoji.
    expect_str(out, "a😀b", "unknown chars preserved via UTF-8 fallback")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_mapping_has_256_entries")
    cases.append("test_mapping_printable_ascii_self_maps")
    cases.append("test_mapping_latin1_self_maps")
    cases.append("test_mapping_space_is_gbar")
    cases.append("test_mapping_inverse")
    cases.append("test_mapping_newline_maps_to_cbar")
    cases.append("test_split_single_leading_space_attaches")
    cases.append("test_split_punctuation")
    cases.append("test_split_digits")
    cases.append("test_split_multiple_spaces")
    cases.append("test_pretokenizer_adds_prefix_space")
    cases.append("test_pretokenizer_no_prefix_when_present")
    cases.append("test_pretokenizer_two_words")
    cases.append("test_pretokenizer_utf8_roundtrip")
    cases.append("test_pretokenizer_no_regex_no_split")
    cases.append("test_decoder_roundtrip")
    cases.append("test_decoder_handles_unknown_char_fallback")
    for name in cases:
        try:
            if name == "test_mapping_has_256_entries":
                test_mapping_has_256_entries()
            elif name == "test_mapping_printable_ascii_self_maps":
                test_mapping_printable_ascii_self_maps()
            elif name == "test_mapping_latin1_self_maps":
                test_mapping_latin1_self_maps()
            elif name == "test_mapping_space_is_gbar":
                test_mapping_space_is_gbar()
            elif name == "test_mapping_inverse":
                test_mapping_inverse()
            elif name == "test_mapping_newline_maps_to_cbar":
                test_mapping_newline_maps_to_cbar()
            elif name == "test_split_single_leading_space_attaches":
                test_split_single_leading_space_attaches()
            elif name == "test_split_punctuation":
                test_split_punctuation()
            elif name == "test_split_digits":
                test_split_digits()
            elif name == "test_split_multiple_spaces":
                test_split_multiple_spaces()
            elif name == "test_pretokenizer_adds_prefix_space":
                test_pretokenizer_adds_prefix_space()
            elif name == "test_pretokenizer_no_prefix_when_present":
                test_pretokenizer_no_prefix_when_present()
            elif name == "test_pretokenizer_two_words":
                test_pretokenizer_two_words()
            elif name == "test_pretokenizer_utf8_roundtrip":
                test_pretokenizer_utf8_roundtrip()
            elif name == "test_pretokenizer_no_regex_no_split":
                test_pretokenizer_no_regex_no_split()
            elif name == "test_decoder_roundtrip":
                test_decoder_roundtrip()
            elif name == "test_decoder_handles_unknown_char_fallback":
                test_decoder_handles_unknown_char_fallback()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
