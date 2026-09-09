"""Unit tests for the Encoding struct.

Run:  mojo run -I src -I tests tests/test_encoding.mojo
"""

import encoding

from encoding import Encoding
from harness import expect, expect_eq, expect_str, summarize


def test_empty_encoding() raises:
    var enc = Encoding()
    expect_eq(enc.len(), 0, "empty encoding has length 0")
    expect_eq(len(enc.get_ids()), 0, "no ids")
    expect_eq(len(enc.get_tokens()), 0, "no tokens")
    expect_eq(len(enc.get_offsets()), 0, "no offsets")


def test_push_defaults() raises:
    var enc = Encoding()
    enc.push(7, "hello", Tuple[Int, Int](0, 5))
    expect_eq(enc.len(), 1, "one token")
    expect_eq(enc.id_at(0), 7, "id")
    expect_str(enc.token_at(0), "hello", "token")
    var off = enc.offset_at(0)
    expect_eq(off[0], 0, "offset start")
    expect_eq(off[1], 5, "offset end")
    # Default mask arrays.
    expect_eq(enc.type_ids[0], 0, "type id default 0")
    expect_eq(enc.attention_mask[0], 1, "attention mask default 1")
    expect_eq(enc.special_tokens_mask[0], 0, "special token mask default 0")


def test_push_multiple() raises:
    var enc = Encoding()
    enc.push(1, "a", Tuple[Int, Int](0, 1))
    enc.push(2, "b", Tuple[Int, Int](1, 2))
    enc.push(3, "c", Tuple[Int, Int](2, 3))
    expect_eq(enc.len(), 3, "three tokens")
    var ids = enc.get_ids()
    expect_eq(ids[0], 1, "first id")
    expect_eq(ids[1], 2, "second id")
    expect_eq(ids[2], 3, "third id")


def test_merge_concatenates() raises:
    var a = Encoding()
    a.push(1, "x", Tuple[Int, Int](0, 1))
    var b = Encoding()
    b.push(2, "y", Tuple[Int, Int](1, 2))
    b.push(3, "z", Tuple[Int, Int](2, 3))
    a.merge(b)
    expect_eq(a.len(), 3, "merged length")
    expect_eq(a.id_at(1), 2, "b ids appended")
    expect_eq(a.id_at(2), 3, "b ids appended")
    expect_str(a.token_at(2), "z", "b tokens appended")


def test_to_string() raises:
    var enc = Encoding()
    enc.push(0, "Ġ", Tuple[Int, Int](0, 1))
    enc.push(1, "h", Tuple[Int, Int](1, 2))
    var s = enc.to_string()
    expect_str(s, "(0, Ġ) (1, h)", "to_string format")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_empty_encoding")
    cases.append("test_push_defaults")
    cases.append("test_push_multiple")
    cases.append("test_merge_concatenates")
    cases.append("test_to_string")
    for name in cases:
        try:
            if name == "test_empty_encoding":
                test_empty_encoding()
            elif name == "test_push_defaults":
                test_push_defaults()
            elif name == "test_push_multiple":
                test_push_multiple()
            elif name == "test_merge_concatenates":
                test_merge_concatenates()
            elif name == "test_to_string":
                test_to_string()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
