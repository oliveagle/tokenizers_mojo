"""Unit tests for Unicode normalizers NFD/NFKD/NFC/NFKC (Phase 2).

Reference values generated with Python `unicodedata.normalize` (same
algorithm family as Rust `unicode-normalization` / HF tokenizers).

Run:  mojo run -I src -I tests tests/test_unicode.mojo
"""

from unicode import UnicodeData
from harness import expect_str, summarize


def nfdc(ud: UnicodeData, s: String) raises -> String:
    return ud.nfd(s)


def test_e_acute() raises:
    var ud = UnicodeData()
    expect_str(ud.nfd("é"), "e\u0301", "é NFD")
    expect_str(ud.nfkd("é"), "e\u0301", "é NFKD")
    expect_str(ud.nfc("é"), "é", "é NFC")
    expect_str(ud.nfkc("é"), "é", "é NFKC")
    # precomposed input round-trips through NFC
    expect_str(ud.nfc("e\u0301"), "é", "e+acute NFC")


def test_fi_ligature() raises:
    var ud = UnicodeData()
    expect_str(ud.nfd("ﬁ"), "ﬁ", "ﬁ NFD unchanged")
    expect_str(ud.nfkd("ﬁ"), "fi", "ﬁ NFKD -> fi")
    expect_str(ud.nfc("ﬁ"), "ﬁ", "ﬁ NFC unchanged")
    expect_str(ud.nfkc("ﬁ"), "fi", "ﬁ NFKC -> fi")


def test_circled_one() raises:
    var ud = UnicodeData()
    expect_str(ud.nfkd("①"), "1", "① NFKD -> 1")
    expect_str(ud.nfkc("①"), "1", "① NFKC -> 1")
    expect_str(ud.nfd("①"), "①", "① NFD unchanged")


def test_a_ring() raises:
    var ud = UnicodeData()
    expect_str(ud.nfd("Å"), "A\u030A", "Å NFD -> A + ring")
    expect_str(ud.nfc("A\u030A"), "Å", "A+ring NFC -> Å")


def test_dz_with_caron() raises:
    var ud = UnicodeData()
    # U+01C5 has no canonical decomposition; NFKD fully decomposes
    expect_str(ud.nfd("ǅ"), "ǅ", "ǅ NFD unchanged")
    expect_str(ud.nfkd("ǅ"), "Dz\u030C", "ǅ NFKD -> D z caron")
    expect_str(ud.nfkc("ǅ"), "D\u017E", "ǅ NFKC -> Dž")


def test_hangul() raises:
    var ud = UnicodeData()
    expect_str(ud.nfd("가"), "\u1100\u1161", "가 NFD -> L V")
    expect_str(ud.nfc("\u1100\u1161"), "가", "L V NFC -> 가")
    expect_str(ud.nfd("각"), "\u1100\u1161\u11A8", "각 NFD -> L V T")
    expect_str(ud.nfc("\u1100\u1161\u11A8"), "각", "L V T NFC -> 각")
    expect_str(
        ud.nfd("한국어"),
        "\u1112\u1161\u11AB\u1100\u116E\u11A8\u110B\u1165",
        "한국어 NFD",
    )
    expect_str(
        ud.nfc("\u1112\u1161\u11AB\u1100\u116E\u11A8\u110B\u1165"),
        "한국어",
        "한국어 NFC roundtrip",
    )


def test_math_script() raises:
    var ud = UnicodeData()
    expect_str(ud.nfkd("𝒜"), "A", "𝒜 NFKD -> A")
    expect_str(ud.nfkc("𝒜"), "A", "𝒜 NFKC -> A")
    expect_str(ud.nfd("𝒜"), "𝒜", "𝒜 NFD unchanged")


def test_combining_reorder() raises:
    var ud = UnicodeData()
    # acute (ccc 230) + cedilla (ccc 202): NFD reorders cedilla first
    expect_str(
        ud.nfd("a\u0301\u0327"), "a\u0327\u0301", "a+acute+cedilla NFD reorder"
    )
    expect_str(ud.nfc("a\u0301\u0327"), "á\u0327", "a+acute+cedilla NFC")


def test_double_marks() raises:
    var ud = UnicodeData()
    expect_str(ud.nfd("ǖ"), "u\u0308\u0304", "ǖ NFD")
    expect_str(ud.nfc("u\u0308\u0304"), "ǖ", "ǖ NFC")
    expect_str(
        ud.nfd("A\u0308\u0301"), "A\u0308\u0301", "A diaeresis acute NFD"
    )
    expect_str(ud.nfc("A\u0308\u0301"), "Ä\u0301", "A diaeresis acute NFC")


def test_katakana_halfwidth() raises:
    var ud = UnicodeData()
    expect_str(ud.nfkd("ｶﾀｶﾅ"), "カタカナ", "halfwidth katakana NFKD")
    expect_str(ud.nfkc("ｶﾀｶﾅ"), "カタカナ", "halfwidth katakana NFKC")
    expect_str(ud.nfd("ｶﾀｶﾅ"), "ｶﾀｶﾅ", "halfwidth katakana NFD unchanged")


def test_jamo_compat() raises:
    var ud = UnicodeData()
    # compat Jamo ㄱ (U+3131) -> Hangul L (U+1100) under NFKD, then composes
    expect_str(ud.nfkd("ㄱㅏ"), "\u1100\u1161", "compat jamo NFKD")
    expect_str(ud.nfkc("ㄱㅏ"), "가", "compat jamo NFKC -> 가")
    expect_str(ud.nfd("ㄱㅏ"), "ㄱㅏ", "compat jamo NFD unchanged")


def test_ascii_passthrough() raises:
    var ud = UnicodeData()
    expect_str(ud.nfc("hello world"), "hello world", "ascii NFC passthrough")
    expect_str(ud.nfkd("hello world"), "hello world", "ascii NFKD passthrough")


def test_greek_iota_subscript() raises:
    var ud = UnicodeData()
    expect_str(ud.nfd("ᾄ"), "α\u0313\u0301\u0345", "ᾄ NFD")
    expect_str(ud.nfc("ᾄ"), "ᾄ", "ᾄ NFC roundtrip")


def test_dev_zwj_unchanged() raises:
    var ud = UnicodeData()
    var s = "\u0915\u094D\u200D\u0937"
    expect_str(ud.nfd(s), s, "dev zwnj NFD unchanged")
    expect_str(ud.nfc(s), s, "dev zwnj NFC unchanged")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_e_acute")
    cases.append("test_fi_ligature")
    cases.append("test_circled_one")
    cases.append("test_a_ring")
    cases.append("test_dz_with_caron")
    cases.append("test_hangul")
    cases.append("test_math_script")
    cases.append("test_combining_reorder")
    cases.append("test_double_marks")
    cases.append("test_katakana_halfwidth")
    cases.append("test_jamo_compat")
    cases.append("test_ascii_passthrough")
    cases.append("test_greek_iota_subscript")
    cases.append("test_dev_zwj_unchanged")
    for name in cases:
        try:
            if name == "test_e_acute":
                test_e_acute()
            elif name == "test_fi_ligature":
                test_fi_ligature()
            elif name == "test_circled_one":
                test_circled_one()
            elif name == "test_a_ring":
                test_a_ring()
            elif name == "test_dz_with_caron":
                test_dz_with_caron()
            elif name == "test_hangul":
                test_hangul()
            elif name == "test_math_script":
                test_math_script()
            elif name == "test_combining_reorder":
                test_combining_reorder()
            elif name == "test_double_marks":
                test_double_marks()
            elif name == "test_katakana_halfwidth":
                test_katakana_halfwidth()
            elif name == "test_jamo_compat":
                test_jamo_compat()
            elif name == "test_ascii_passthrough":
                test_ascii_passthrough()
            elif name == "test_greek_iota_subscript":
                test_greek_iota_subscript()
            elif name == "test_dev_zwj_unchanged":
                test_dev_zwj_unchanged()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
