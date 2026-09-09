"""Unit tests for the regex engine (Phase 2 Split).

Every expected value is cross-checked against Python 3.12 `re` / the
same Thompson NFA semantics as the upstream `regex` crate (greedy
leftmost-first, backtracking SPLIT).

Run: mojo run -I src -I tests tests/test_regex.mojo
"""

from regex import regex_match, regex_find_all, regex_split
from harness import expect, expect_eq, expect_str, summarize


def expect_bool(got: Bool, want: Bool, msg: String) raises:
    if got != want:
        raise Error(
            msg + " (expected " + String(want) + ", got " + String(got) + ")"
        )


def test_literals() raises:
    expect_bool(regex_match("a", "a"), True, "a on a")
    expect_bool(regex_match("a", "b"), False, "a on b")
    expect_bool(regex_match("^a$", "a"), True, "^a$ on a")
    expect_bool(regex_match("^a$", "ab"), False, "^a$ on ab")
    expect_bool(regex_match("ab", "ab"), True, "ab on ab")


def test_dot() raises:
    expect_bool(regex_match("a.c", "abc"), True, "a.c on abc")
    expect_bool(regex_match("a.c", "axc"), True, "a.c on axc")
    expect_bool(regex_match("a.c", "ac"), False, "a.c on ac (no dot overlap)")
    expect_bool(regex_match("a.c", "aXbYc"), False, "a.c on aXbYc")


def test_star() raises:
    expect_bool(regex_match("a*", ""), True, "a* on empty")
    expect_bool(regex_match("a*", "aaa"), True, "a* on aaa")
    expect_bool(regex_match("a*", "b"), False, "a* on b (full match needs all)")
    expect_bool(regex_match("a*b", "b"), True, "a*b on b")
    expect_bool(regex_match("a*b", "aaab"), True, "a*b on aaab")


def test_plus() raises:
    expect_bool(regex_match("a+", ""), False, "a+ on empty")
    expect_bool(regex_match("a+", "aaa"), True, "a+ on aaa")
    expect_bool(regex_match("a+b", "ab"), True, "a+b on ab")
    expect_bool(regex_match("a+b", "aaab"), True, "a+b on aaab")


def test_opt() raises:
    expect_bool(regex_match("a?", ""), True, "a? on empty")
    expect_bool(regex_match("a?", "a"), True, "a? on a")
    expect_bool(regex_match("a?", "aa"), False, "a? on aa (full match >1)")
    expect_bool(regex_match("a?b", "b"), True, "a?b on b")
    expect_bool(regex_match("a?b", "ab"), True, "a?b on ab")


def test_braces_exact() raises:
    expect_bool(regex_match("a{2}", "aa"), True, "a{2} on aa")
    expect_bool(regex_match("a{2}", "a"), False, "a{2} on a")
    expect_bool(regex_match("a{2}", "aaa"), False, "a{2} on aaa (exact)")
    expect_bool(regex_match("a{3}", "aaa"), True, "a{3} on aaa")


def test_braces_min() raises:
    expect_bool(regex_match("a{2,}", "a"), False, "a{2,} on a")
    expect_bool(regex_match("a{2,}", "aa"), True, "a{2,} on aa")
    expect_bool(regex_match("a{2,}", "aaaa"), True, "a{2,} on aaaa")


def test_braces_range() raises:
    expect_bool(regex_match("a{2,3}", "a"), False, "a{2,3} on a")
    expect_bool(regex_match("a{2,3}", "aa"), True, "a{2,3} on aa")
    expect_bool(regex_match("a{2,3}", "aaa"), True, "a{2,3} on aaa")
    expect_bool(regex_match("a{2,3}", "aaaa"), False, "a{2,3} on aaaa (exact)")
    expect_bool(regex_match("ab{2,3}c", "abbc"), True, "ab{2,3}c on abbc")
    expect_bool(regex_match("ab{2,3}c", "abbbc"), True, "ab{2,3}c on abbbc")
    expect_bool(
        regex_match("ab{2,3}c", "abbbbc"), False, "ab{2,3}c on abbbbc"
    )


def test_groups() raises:
    expect_bool(regex_match("(ab)+", "ab"), True, "(ab)+ on ab")
    expect_bool(regex_match("(ab)+", "abab"), True, "(ab)+ on abab")
    expect_bool(regex_match("(ab)+", "aba"), False, "(ab)+ on aba")
    expect_bool(regex_match("(a|b)(c|d)", "ac"), True, "(a|b)(c|d) on ac")
    expect_bool(regex_match("(a|b)(c|d)", "bd"), True, "(a|b)(c|d) on bd")
    expect_bool(regex_match("(a|b)(c|d)", "ab"), False, "(a|b)(c|d) on ab")


def test_alternation() raises:
    expect_bool(regex_match("a|b", "a"), True, "a|b on a")
    expect_bool(regex_match("a|b", "b"), True, "a|b on b")
    expect_bool(regex_match("a|b", "c"), False, "a|b on c")
    expect_bool(regex_match("a|b|c", "c"), True, "a|b|c on c")
    expect_bool(regex_match("a|b|c", "d"), False, "a|b|c on d")
    expect_bool(regex_match("(a|b)+", "abba"), True, "(a|b)+ on abba")
    expect_bool(regex_match("(ab|c)+", "cabc"), True, "(ab|c)+ on cabc")


def test_char_class() raises:
    expect_bool(regex_match("[abc]+", "cab"), True, "[abc]+ on cab")
    expect_bool(regex_match("[abc]+", "cabx"), False, "[abc]+ on cabx")
    expect_bool(regex_match("[^a]+", "bcd"), True, "[^a]+ on bcd")
    expect_bool(regex_match("[^a]+", "abc"), False, "[^a]+ on abc")
    expect_bool(regex_match("[a-z]+", "hello"), True, "[a-z]+ on hello")
    expect_bool(regex_match("[a-z]+", "Hello"), False, "[a-z]+ on Hello")
    expect_bool(regex_match("[0-9]{3}", "123"), True, "[0-9]{3} on 123")
    expect_bool(regex_match("[0-9]{3}", "12"), False, "[0-9]{3} on 12")


def test_predefined_classes() raises:
    expect_bool(regex_match(r"\d+", "123"), True, "\\d+ on 123")
    expect_bool(regex_match(r"\d+", "a12"), False, "\\d+ on a12")
    expect_bool(regex_match(r"\w+", "abc_"), True, "\\w+ on abc_")
    expect_bool(regex_match(r"\w+", "ab c"), False, "\\w+ on ab c")
    expect_bool(regex_match(r"\s+", "  "), True, "\\s+ on whitespace")
    expect_bool(regex_match(r"\s+", "a "), False, "\\s+ on a space")
    expect_bool(regex_match(r"\d\w", "1a"), True, "\\d\\w on 1a")
    expect_bool(regex_match(r"\D+", "abc"), True, "\\D+ on abc")
    expect_bool(regex_match(r"\W+", "!@#"), True, "\\W+ on punct")


def test_find_all() raises:
    var m1 = regex_find_all(r"\d+", "a1 b22 c333")
    expect_eq(len(m1), 3, "findall \\d+ count")
    expect_eq(m1[0][0], 1, "findall m1[0].start")
    expect_eq(m1[0][1], 2, "findall m1[0].end")
    expect_eq(m1[1][0], 4, "findall m1[1].start")
    expect_eq(m1[1][1], 6, "findall m1[1].end")
    expect_eq(m1[2][0], 8, "findall m2.start")
    expect_eq(m1[2][1], 11, "findall m2.end")
    var m2 = regex_find_all(r"\w+", "hello world 42")
    expect_eq(len(m2), 3, "findall \\w+ count")
    expect_eq(m2[0][0], 0, "findall hello start")
    expect_eq(m2[0][1], 5, "findall hello end")
    expect_eq(m2[1][0], 6, "findall world start")
    expect_eq(m2[1][1], 11, "findall world end")
    expect_eq(m2[2][0], 12, "findall 42 start")
    expect_eq(m2[2][1], 14, "findall 42 end")


def test_split() raises:
    var s1 = regex_split("[, ]", "a,b c")
    expect_eq(len(s1), 3, "split count")
    expect_str(s1[0], "a", "split s1[0]")
    expect_str(s1[1], "b", "split s1[1]")
    expect_str(s1[2], "c", "split s1[2]")
    var s2 = regex_split(",", "a,b,c")
    expect_eq(len(s2), 3, "split comma count")
    # \s+ on " a  b ": Split pre-tokenizer 语义丢弃空片段 → ["a", "b"]
    var s3 = regex_split(r"\s+", " a  b ")
    expect_eq(len(s3), 2, "split \\s+ count (empties dropped)")
    expect_str(s3[0], "a", "split s3[0]")
    expect_str(s3[1], "b", "split s3[1]")


def main() raises:
    var failures = List[String]()
    for t in [
        ("literals", test_literals),
        ("dot", test_dot),
        ("star", test_star),
        ("plus", test_plus),
        ("opt", test_opt),
        ("braces_exact", test_braces_exact),
        ("braces_min", test_braces_min),
        ("braces_range", test_braces_range),
        ("groups", test_groups),
        ("alternation", test_alternation),
        ("char_class", test_char_class),
        ("predefined", test_predefined_classes),
        ("find_all", test_find_all),
        ("split", test_split),
    ]:
        try:
            t[1]()
            print("PASS", t[0])
        except e:
            print("FAIL", t[0], ":", e)
            failures.append(t[0])
    summarize(failures)
