"""Unit tests for batch encode / decode (Phase 3).

Uses the HF-trained fixture tokenizer; reference ids from
`Tokenizer.from_file(...).encode(s).ids`.

Run:  mojo run -I src -I tests tests/test_batch.mojo
"""


from from_pretrained import from_pretrained
from harness import expect, expect_eq, summarize


def ids_join(ids: List[Int]) raises -> String:
    var sb = String()
    for i in range(len(ids)):
        if i > 0:
            sb += " "
        sb += String(ids[i])
    return sb^


def test_batch_matches_single() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")
    var texts = List[String]()
    texts.append("hello world")
    texts.append("how are you")
    texts.append("hello")
    texts.append("tokenization is fun")
    var encs = tok.encode_batch(texts)
    expect_eq(len(encs), 4, "batch count")

    var want = List[String]()
    want.append("48 50")
    want.append("43 49 45")
    want.append("48")
    want.append("30 57 55 51 53 15 59 32 58")

    for i in range(len(encs)):
        var single = tok.encode(texts[i])
        expect_eq(
            len(encs[i].ids),
            len(single.ids),
            "batch/single length " + String(i),
        )
        for j in range(len(encs[i].ids)):
            expect(
                encs[i].ids[j] == single.ids[j],
                "batch/single id mismatch at " + String(i) + "," + String(j),
            )
        var got = ids_join(encs[i].ids)
        expect(
            got == want[i],
            "batch ids "
            + String(i)
            + ": want ["
            + want[i]
            + "] got ["
            + got
            + "]",
        )


def test_decode_batch_roundtrip() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")
    var texts = List[String]()
    texts.append("hello world")
    texts.append("how are you")
    var encs = tok.encode_batch(texts)
    var decs = tok.decode_batch(encs)
    expect_eq(len(decs), 2, "decode_batch count")
    # ByteLevel decode preserves the leading prefix-space
    expect(decs[0] == " hello world", "dec0 == ' hello world'")
    expect(decs[1] == " how are you", "dec1 == ' how are you'")


def main() raises:
    var failures = List[String]()
    try:
        test_batch_matches_single()
    except:
        failures.append("test_batch_matches_single")
    try:
        test_decode_batch_roundtrip()
    except:
        failures.append("test_decode_batch_roundtrip")
    summarize(failures)
