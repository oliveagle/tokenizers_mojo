"""Unit tests for `Tokenizer.from_pretrained` (Phase 3).

Fixture: `tests/data/test_tokenizer.json` (real GPT-2 style ByteLevel BPE,
trained by HF `tokenizers`; see scripts or fixture header). Reference ids
produced by `Tokenizer.from_file(...) → tokenizer.encode(s).ids`.

Run:  mojo run -I src -I tests tests/test_from_pretrained.mojo
"""


from from_pretrained import from_pretrained
from harness import expect, expect_eq, expect_str, summarize


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ids_join(ids: List[Int]) raises -> String:
    var sb = String()
    for i in range(len(ids)):
        if i > 0:
            sb += " "
        sb += String(ids[i])
    return sb^


def expect_ids(enc_ids: List[Int], want: String, msg: String) raises:
    var got = ids_join(enc_ids)
    expect(
        got == want,
        msg + ": want [" + want + "] got [" + got + "]",
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_load_sizes() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")
    expect_eq(len(tok.model.vocab), 60, "vocab size")
    expect_eq(len(tok.model.merges), 31, "merges count")


def test_encode_simple() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")
    expect_ids(tok.encode("hello world").ids, "48 50", "hello world")
    expect_ids(tok.encode("hello").ids, "48", "hello")


def test_encode_multi_word() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")
    expect_ids(tok.encode("how are you").ids, "43 49 45", "how are you")
    expect_ids(
        tok.encode("world hello world").ids, "50 48 50", "world hello world"
    )


def test_encode_long() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")
    expect_ids(
        tok.encode("the quick brown fox").ids,
        "46 28 18 22 10 4 12 28 3 19 31 15 32 16 25",
        "the quick brown fox",
    )
    expect_ids(
        tok.encode("tokenization is fun").ids,
        "30 57 55 51 53 15 59 32 58",
        "tokenization is fun",
    )


def test_decode_roundtrip() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")
    var enc = tok.encode("hello world")
    var text = tok.decode(enc)
    # ByteLevel decode preserves the leading prefix-space
    expect_str(
        text,
        " hello world",
        "decode(encode('hello world')) (ByteLevel prefix-space)",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    var failures = List[String]()
    try:
        test_load_sizes()
    except:
        failures.append("test_load_sizes")
    try:
        test_encode_simple()
    except:
        failures.append("test_encode_simple")
    try:
        test_encode_multi_word()
    except:
        failures.append("test_encode_multi_word")
    try:
        test_encode_long()
    except:
        failures.append("test_encode_long")
    try:
        test_decode_roundtrip()
    except:
        failures.append("test_decode_roundtrip")
    summarize(failures)
