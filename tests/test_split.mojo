"""Unit tests for the Split pre-tokenizer (Phase 2).

Reference values verified against HuggingFace tokenizers 0.22.2
`Split(pattern=..., behavior=...)`.

Run:  mojo run -I src -I tests tests/test_split.mojo
"""

import split

from split import SplitPreTokenizer, split_on
from harness import expect, expect_eq, expect_str, summarize


def run(sep: String, beh: String, s: String) -> List[String]:
    var sp = SplitPreTokenizer(sep, beh)
    return sp.pre_tokenize(s)


def expect_list(got: List[String], want: String, msg: String) raises:
    var joined = String()
    for i in range(len(got)):
        if i > 0:
            joined += " "
        joined += repr(got[i])
    expect_str(joined, want, msg)


def test_removed() raises:
    expect_list(run("-", "removed", "-a"), "'a'", "leading delim removed")
    expect_list(run("-", "removed", "a-"), "'a'", "trailing delim removed")
    expect_list(run("-", "removed", "-a-"), "'a'", "both removed")
    expect_list(run("-", "removed", "a--b"), "'a' 'b'", "double delim removed")
    expect_list(
        run(" ", "removed", "Hello world foo"),
        "'Hello' 'world' 'foo'",
        "spaces",
    )
    expect_list(run(",", "removed", "a,b,c"), "'a' 'b' 'c'", "comma")


def test_isolated() raises:
    expect_list(run("-", "isolated", "-a"), "'-' 'a'", "leading isolated")
    expect_list(run("-", "isolated", "a-"), "'a' '-'", "trailing isolated")
    expect_list(run("-", "isolated", "-a-"), "'-' 'a' '-'", "both isolated")
    expect_list(
        run("-", "isolated", "a--b"), "'a' '-' '-' 'b'", "double isolated"
    )
    expect_list(
        run(" ", "isolated", "Hello world foo"),
        "'Hello' ' ' 'world' ' ' 'foo'",
        "spaces isolated",
    )


def test_merged_with_previous() raises:
    expect_list(
        run("-", "merged_with_previous", "-a"), "'-' 'a'", "leading alone"
    )
    expect_list(
        run("-", "merged_with_previous", "a-"), "'a-'", "trailing merge"
    )
    expect_list(run("-", "merged_with_previous", "-a-"), "'-' 'a-'", "mixed")
    expect_list(
        run("-", "merged_with_previous", "a--b"), "'a-' '-' 'b'", "double"
    )
    expect_list(
        run(" ", "merged_with_previous", "Hello world foo"),
        "'Hello ' 'world ' 'foo'",
        "spaces prev",
    )


def test_merged_with_next() raises:
    expect_list(run("-", "merged_with_next", "-a"), "'-a'", "leading merge")
    expect_list(run("-", "merged_with_next", "a-"), "'a' '-'", "trailing alone")
    expect_list(run("-", "merged_with_next", "-a-"), "'-a' '-'", "mixed")
    expect_list(run("-", "merged_with_next", "a--b"), "'a' '-' '-b'", "double")
    expect_list(
        run(" ", "merged_with_next", "Hello world foo"),
        "'Hello' ' world' ' foo'",
        "spaces next",
    )


def test_no_delimiter() raises:
    expect_list(run("-", "removed", "abc"), "'abc'", "no delim")
    expect_list(run("-", "isolated", "abc"), "'abc'", "no delim isolated")


def test_multichar_separator() raises:
    expect_list(
        run("::", "isolated", "a::b::c"), "'a' '::' 'b' '::' 'c'", ":: isolated"
    )
    expect_list(run("::", "merged_with_next", "a::b"), "'a' '::b'", ":: next")


def test_invert() raises:
    expect_list(
        run("-", "removed", "a-b-c"), "'a' 'b' 'c'", "non-invert baseline"
    )
    var sp = SplitPreTokenizer("-", "removed")
    # invert keeps only delimiters
    # (invert field not exposed via constructor; covered by unit directly)
    expect(not sp.invert, "default invert false")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_removed")
    cases.append("test_isolated")
    cases.append("test_merged_with_previous")
    cases.append("test_merged_with_next")
    cases.append("test_no_delimiter")
    cases.append("test_multichar_separator")
    cases.append("test_invert")
    for name in cases:
        try:
            if name == "test_removed":
                test_removed()
            elif name == "test_isolated":
                test_isolated()
            elif name == "test_merged_with_previous":
                test_merged_with_previous()
            elif name == "test_merged_with_next":
                test_merged_with_next()
            elif name == "test_no_delimiter":
                test_no_delimiter()
            elif name == "test_multichar_separator":
                test_multichar_separator()
            elif name == "test_invert":
                test_invert()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
