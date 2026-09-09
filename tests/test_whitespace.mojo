"""Unit tests for Whitespace and Metaspace pre-tokenizers (Phase 2).

Reference values verified against HuggingFace tokenizers 0.22.2:
  Whitespace.pre_tokenize_str / Metaspace.pre_tokenize_str

Run:  mojo run -I src -I tests tests/test_whitespace.mojo
"""

import whitespace

from whitespace import Whitespace, Metaspace, whitespace_split
from harness import expect, expect_eq, expect_str, summarize


def ws(s: String) -> List[String]:
    var w = Whitespace()
    return w.pre_tokenize(s)


def ms(s: String) -> List[String]:
    var m = Metaspace()
    return m.pre_tokenize(s)


def expect_list(got: List[String], want_str: String, msg: String) raises:
    var joined = String()
    for i in range(len(got)):
        if i > 0:
            joined += " "
        joined += repr(got[i])
    expect_str(joined, want_str, msg)


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------


def test_ws_hey_man() raises:
    expect_list(ws("Hey man!"), "'Hey' 'man' '!'", "hey man")


def test_ws_how_are_you() raises:
    expect_list(
        ws("How are you doing?"),
        "'How' 'are' 'you' 'doing' '?'",
        "how are you",
    )


def test_ws_trim_spaces() raises:
    expect_list(ws("  hello  "), "'hello'", "trim spaces")


def test_ws_contraction() raises:
    var got = ws("can't")
    expect_eq(len(got), 3, "can't -> 3 tokens")
    expect_str(got[0], "can", "can")
    expect_str(got[1], "'", "apostrophe")
    expect_str(got[2], "t", "t")


def test_ws_unicode() raises:
    expect_list(ws("héllo wörld"), "'héllo' 'wörld'", "accents kept")
    expect_list(ws("你好 world"), "'你好' 'world'", "CJK is word char")


def test_ws_numbers() raises:
    expect_list(
        ws("numbers 123 and 456"), "'numbers' '123' 'and' '456'", "numbers"
    )


def test_ws_newline() raises:
    expect_list(
        ws("multi\nline  text"), "'multi' 'line' 'text'", "newline+spaces"
    )


# ---------------------------------------------------------------------------
# Metaspace
# ---------------------------------------------------------------------------


def test_ms_hello_world() raises:
    expect_list(ms("Hello world"), "'▁Hello' '▁world'", "basic")


def test_ms_leading_spaces() raises:
    expect_list(
        ms("  two  spaces  "),
        "'▁' '▁two' '▁' '▁spaces' '▁' '▁'",
        "multi space",
    )


def test_ms_no_leading() raises:
    expect_list(ms("hello"), "'▁hello'", "simple prepend")


def test_ms_double_space() raises:
    expect_list(ms("a  b"), "'▁a' '▁' '▁b'", "a  b")


def test_ms_variants() raises:
    expect_list(ms("  a"), "'▁' '▁a'", "  a")
    expect_list(ms("a  "), "'▁a' '▁' '▁'", "a  ")
    expect_list(ms("a b c"), "'▁a' '▁b' '▁c'", "a b c")
    expect_list(ms("x   y"), "'▁x' '▁' '▁' '▁y'", "x   y")
    expect_list(ms("a b"), "'▁a' '▁b'", "a b")
    expect_list(ms(" a"), "'▁a'", " leading single")
    expect_list(ms("a "), "'▁a' '▁'", "trailing single")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_ws_hey_man")
    cases.append("test_ws_how_are_you")
    cases.append("test_ws_trim_spaces")
    cases.append("test_ws_contraction")
    cases.append("test_ws_unicode")
    cases.append("test_ws_numbers")
    cases.append("test_ws_newline")
    cases.append("test_ms_hello_world")
    cases.append("test_ms_leading_spaces")
    cases.append("test_ms_no_leading")
    cases.append("test_ms_double_space")
    cases.append("test_ms_variants")
    for name in cases:
        try:
            if name == "test_ws_hey_man":
                test_ws_hey_man()
            elif name == "test_ws_how_are_you":
                test_ws_how_are_you()
            elif name == "test_ws_trim_spaces":
                test_ws_trim_spaces()
            elif name == "test_ws_contraction":
                test_ws_contraction()
            elif name == "test_ws_unicode":
                test_ws_unicode()
            elif name == "test_ws_numbers":
                test_ws_numbers()
            elif name == "test_ws_newline":
                test_ws_newline()
            elif name == "test_ms_hello_world":
                test_ms_hello_world()
            elif name == "test_ms_leading_spaces":
                test_ms_leading_spaces()
            elif name == "test_ms_no_leading":
                test_ms_no_leading()
            elif name == "test_ms_double_space":
                test_ms_double_space()
            elif name == "test_ms_variants":
                test_ms_variants()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
