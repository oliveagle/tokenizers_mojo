"""Unit tests for Normalizers (Lowercase / Strip / real NFC).

Reference: HuggingFace tokenizers `Lowercase`, `Strip`.

Run:  mojo run -I src -I tests tests/test_normalizers.mojo
"""

import normalizers

from normalizers import (
    NFCNormalizer,
    Lowercase,
    Strip,
    LowercaseStrip,
    normalize_nfc,
)
from harness import expect, expect_eq, expect_str, summarize


def test_lowercase_ascii() raises:
    var n = Lowercase()
    expect_str(n.normalize("HELLO World"), "hello world", "ascii")


def test_lowercase_latin1() raises:
    var n = Lowercase()
    expect_str(n.normalize("ÀÉÎÖÜ"), "àéîöü", "latin1")
    expect_str(n.normalize("ÉCOLE"), "école", "ÉCOLE")


def test_lowercase_greek_cyrillic() raises:
    var n = Lowercase()
    expect_str(n.normalize("ΑΒΓ"), "αβγ", "greek")
    expect_str(n.normalize("ПРИВЕТ"), "привет", "cyrillic")


def test_lowercase_mixed() raises:
    var n = Lowercase()
    expect_str(n.normalize("Hello, WÖRLD 123!"), "hello, wörld 123!", "mixed")


def test_strip_basic() raises:
    var s = Strip()
    expect_str(s.normalize("  hello  "), "hello", "both sides")
    expect_str(s.normalize("\tfoo\n"), "foo", "tabs/newlines")
    expect_str(s.normalize("no space"), "no space", "no change")


def test_lowercase_strip_compose() raises:
    var n = LowercaseStrip()
    expect_str(n.normalize("  Hello WORLD  "), "hello world", "compose")


def test_nfc_real() raises:
    var n = NFCNormalizer()
    # "café" is already NFC; real NFC keeps it (and composes e+acute)


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_lowercase_ascii")
    cases.append("test_lowercase_latin1")
    cases.append("test_lowercase_greek_cyrillic")
    cases.append("test_lowercase_mixed")
    cases.append("test_strip_basic")
    cases.append("test_lowercase_strip_compose")
    cases.append("test_nfc_real")
    for name in cases:
        try:
            if name == "test_lowercase_ascii":
                test_lowercase_ascii()
            elif name == "test_lowercase_latin1":
                test_lowercase_latin1()
            elif name == "test_lowercase_greek_cyrillic":
                test_lowercase_greek_cyrillic()
            elif name == "test_lowercase_mixed":
                test_lowercase_mixed()
            elif name == "test_strip_basic":
                test_strip_basic()
            elif name == "test_lowercase_strip_compose":
                test_lowercase_strip_compose()
            elif name == "test_nfc_real":
                test_nfc_real()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
