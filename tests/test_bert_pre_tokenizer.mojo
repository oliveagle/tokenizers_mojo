"""Unit tests for the BERT pre-tokenizer (Phase 2).

Reference values verified against HuggingFace tokenizers 0.22.2
`BertPreTokenizer.pre_tokenize_str`.

Run:  mojo run -I src -I tests tests/test_bert_pre_tokenizer.mojo
"""

from bert_pre_tokenizer import bert_pre_tokenize
from harness import expect, expect_eq, expect_str, summarize


def expect_list(got: List[String], want: String, msg: String) raises:
    var joined = String()
    for i in range(len(got)):
        if i > 0:
            joined += " "
        joined += repr(got[i])
    expect_str(joined, want, msg)


def test_upstream_basic() raises:
    expect_list(
        bert_pre_tokenize("Hey friend!     How are you?!?"),
        "'Hey' 'friend' '!' 'How' 'are' 'you' '?' '!' '?'",
        "upstream bert.rs basic test",
    )


def test_punctuation_split() raises:
    expect_list(
        bert_pre_tokenize("Hello, world!"),
        "'Hello' ',' 'world' '!'",
        "hello, world",
    )
    expect_list(
        bert_pre_tokenize("(a) [b] {c}"),
        "'(' 'a' ')' '[' 'b' ']' '{' 'c' '}'",
        "brackets",
    )
    expect_list(
        bert_pre_tokenize("a...b"), "'a' '.' '.' '.' 'b'", "ellipsis dots"
    )
    expect_list(
        bert_pre_tokenize("test: hi; there"),
        "'test' ':' 'hi' ';' 'there'",
        "colons",
    )


def test_contractions() raises:
    var t1 = bert_pre_tokenize("can't stop")
    expect_eq(len(t1), 4, "can't -> 4")
    expect_str(t1[0], "can", "can")
    expect_str(t1[1], "'", "apostrophe")
    expect_str(t1[2], "t", "t")
    expect_str(t1[3], "stop", "stop")
    var t2 = bert_pre_tokenize("don't do it")
    expect_eq(len(t2), 5, "don't -> 5")
    expect_str(t2[0], "don", "don")
    expect_str(t2[1], "'", "apostrophe")
    expect_str(t2[2], "t", "t")
    var t3 = bert_pre_tokenize("C'est la vie!")
    expect_eq(len(t3), 6, "C'est -> 6")
    expect_str(t3[0], "C", "C")
    expect_str(t3[1], "'", "apostrophe")
    expect_str(t3[5], "!", "bang")


def test_unicode_punct() raises:
    # … (U+2026) is General Punctuation and gets isolated.
    expect_list(
        bert_pre_tokenize("Hello…世界"), "'Hello' '…' '世界'", "unicode ellipsis"
    )


def test_cjk_is_word() raises:
    expect_list(
        bert_pre_tokenize("数字123 and 中文"),
        "'数字123' 'and' '中文'",
        "CJK kept together",
    )


def test_whitespace_removed() raises:
    expect_list(
        bert_pre_tokenize("tab\there  x"),
        "'tab' 'here' 'x'",
        "tabs/spaces removed",
    )


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_upstream_basic")
    cases.append("test_punctuation_split")
    cases.append("test_contractions")
    cases.append("test_unicode_punct")
    cases.append("test_cjk_is_word")
    cases.append("test_whitespace_removed")
    for name in cases:
        try:
            if name == "test_upstream_basic":
                test_upstream_basic()
            elif name == "test_punctuation_split":
                test_punctuation_split()
            elif name == "test_contractions":
                test_contractions()
            elif name == "test_unicode_punct":
                test_unicode_punct()
            elif name == "test_cjk_is_word":
                test_cjk_is_word()
            elif name == "test_whitespace_removed":
                test_whitespace_removed()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
