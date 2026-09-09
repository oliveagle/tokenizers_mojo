"""Unit tests for truncation / overflowing (Phase 2).

Reference values verified against HuggingFace tokenizers 0.22.2
`Encoding::truncate` and `utils::truncation::truncate_encodings`.

Run:  mojo run -I src -I tests tests/test_truncation.mojo
"""

from encoding import Encoding
from truncation import TruncationParams, truncate_encodings
from harness import expect, expect_eq, expect_str, summarize


def make_enc_v(*values: Int) -> Encoding:
    """Build an Encoding with the given ids (tokens = "tok<i>")."""
    var enc = Encoding()
    var i = 0
    for v in values:
        enc.push(v, "tok" + String(i), Tuple[Int, Int](i * 2, i * 2 + 2))
        i += 1
    return enc^


def make_enc_range(n: Int) -> Encoding:
    """Build an Encoding with ids 0..n-1."""
    var enc = Encoding()
    for i in range(n):
        enc.push(i, "tok" + String(i), Tuple[Int, Int](i * 2, i * 2 + 2))
    return enc^


def enc_ids_str(enc: Encoding) -> String:
    var s = String()
    for i in range(enc.len()):
        if i > 0:
            s += " "
        s += String(enc.ids[i])
    return s


def test_truncate_right_no_stride() raises:
    var enc = make_enc_v(1, 2, 3)
    var ov = enc.truncate(2, 0, "right")
    expect_str(enc_ids_str(enc), "1 2", "right: kept window")
    expect_eq(len(ov), 1, "right: one overflow window")
    expect_str(enc_ids_str(ov[0]), "3", "right: overflow")


def test_truncate_to_empty() raises:
    var enc = make_enc_v(1, 2, 3)
    var ov = enc.truncate(0, 0, "right")
    expect_eq(enc.len(), 0, "max_len 0 empties encoding")
    expect_eq(len(ov), 1, "max_len 0: one overflow")
    expect_str(
        enc_ids_str(ov[0]), "1 2 3", "max_len 0: whole encoding overflows"
    )


def test_truncate_with_stride() raises:
    var enc = make_enc_v(1, 2, 3, 4, 5)
    var ov = enc.truncate(4, 2, "right")
    expect_str(enc_ids_str(enc), "1 2 3 4", "stride: kept window")
    expect_eq(len(ov), 1, "stride: one overflow window")
    expect_str(enc_ids_str(ov[0]), "3 4 5", "stride: overflow window")


def test_truncate_left() raises:
    var enc = make_enc_v(1, 2, 3)
    var ov = enc.truncate(2, 0, "left")
    expect_str(enc_ids_str(enc), "2 3", "left: tail kept")
    expect_eq(len(ov), 1, "left: one overflow window")
    expect_str(enc_ids_str(ov[0]), "1", "left: overflow is the head")


def test_truncate_noop_when_within_limit() raises:
    var enc = make_enc_v(1, 2, 3)
    var ov = enc.truncate(10, 0, "right")
    expect_eq(enc.len(), 3, "noop when max_len >= len")
    expect_eq(len(ov), 0, "no overflows when within limit")


def test_truncate_bad_stride_raises() raises:
    var enc = make_enc_v(1, 2, 3)
    var failed = False
    try:
        var _ = enc.truncate(2, 2, "right")
    except:
        failed = True
    expect(failed, "stride >= max_len should raise")


def test_single_longest_first() raises:
    var enc = make_enc_range(10)
    var params = TruncationParams(3, "longest_first", "right", 0)
    var pair = Encoding()
    var res = truncate_encodings(enc, pair, False, params)
    expect_str(enc_ids_str(enc), "0 1 2", "single longest_first kept")
    expect_eq(len(res.enc_overflow), 3, "single longest_first overflow count")
    expect_str(enc_ids_str(res.enc_overflow[0]), "3 4 5", "single overflow 0")
    expect_str(enc_ids_str(res.enc_overflow[1]), "6 7 8", "single overflow 1")
    expect_str(enc_ids_str(res.enc_overflow[2]), "9", "single overflow 2")


def test_single_left_keeps_tail() raises:
    var enc = make_enc_range(10)
    var params = TruncationParams(3, "longest_first", "left", 0)
    var pair = Encoding()
    var res = truncate_encodings(enc, pair, False, params)
    expect_str(enc_ids_str(enc), "7 8 9", "single left keeps tail")
    expect_eq(len(res.enc_overflow), 3, "single left overflow count")
    expect_str(enc_ids_str(res.enc_overflow[0]), "4 5 6", "left overflow 0")
    expect_str(enc_ids_str(res.enc_overflow[1]), "1 2 3", "left overflow 1")
    expect_str(enc_ids_str(res.enc_overflow[2]), "0", "left overflow 2")


def test_pair_longest_first_splits_evenly() raises:
    # A=10, B=5, max_length=6 -> A keeps 3, B keeps 3
    var enc = make_enc_range(10)
    var pair = make_enc_range(5)
    var params = TruncationParams(6, "longest_first", "right", 0)
    var res = truncate_encodings(enc, pair, True, params)
    expect_str(enc_ids_str(enc), "0 1 2", "pair A kept 3")
    expect_str(enc_ids_str(pair), "0 1 2", "pair B kept 3")
    expect_eq(len(res.enc_overflow), 3, "pair A overflow count")
    expect_eq(len(res.pair_overflow), 1, "pair B overflow count")
    expect_str(enc_ids_str(res.pair_overflow[0]), "3 4", "pair B overflow")


def test_pair_only_second() raises:
    # A=5, B=10, max_length=8, OnlySecond -> A stays 5, B -> 3
    var enc = make_enc_range(5)
    var pair = make_enc_range(10)
    var params = TruncationParams(8, "only_second", "right", 0)
    var res = truncate_encodings(enc, pair, True, params)
    expect_str(enc_ids_str(enc), "0 1 2 3 4", "only_second: A untouched")
    expect_str(enc_ids_str(pair), "0 1 2", "only_second: B truncated")
    expect_eq(len(res.enc_overflow), 0, "only_second: no A overflow")
    expect_eq(len(res.pair_overflow), 3, "only_second: B overflow")


def test_only_first() raises:
    # A=10, B=5, max_length=8, OnlyFirst -> A -> 3, B untouched
    var enc = make_enc_range(10)
    var pair = make_enc_range(5)
    var params = TruncationParams(8, "only_first", "right", 0)
    var res = truncate_encodings(enc, pair, True, params)
    expect_str(enc_ids_str(enc), "0 1 2", "only_first: A truncated")
    expect_str(enc_ids_str(pair), "0 1 2 3 4", "only_first: B untouched")


def test_no_truncation_needed() raises:
    var enc = make_enc_v(1, 2)
    var pair = Encoding()
    var params = TruncationParams(10, "longest_first", "right", 0)
    var res = truncate_encodings(enc, pair, False, params)
    expect_eq(enc.len(), 2, "within limit: unchanged")
    expect_eq(len(res.enc_overflow), 0, "within limit: no overflow")


def test_only_second_without_pair_raises() raises:
    # truncation is needed (10 > 3) so OnlySecond without pair must raise
    var enc = make_enc_range(10)
    var pair = Encoding()
    var params = TruncationParams(3, "only_second", "right", 0)
    var failed = False
    try:
        var _ = truncate_encodings(enc, pair, False, params)
    except:
        failed = True
    expect(failed, "only_second without pair should raise")


def test_sequence_too_short_raises() raises:
    # A=1, B=1, max_length=1, OnlyFirst: to_remove=1, target=1 -> too short
    var enc = make_enc_range(1)
    var pair = make_enc_range(1)
    var params = TruncationParams(1, "only_first", "right", 0)
    var failed = False
    try:
        var _ = truncate_encodings(enc, pair, True, params)
    except:
        failed = True
    expect(failed, "target too short should raise")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_truncate_right_no_stride")
    cases.append("test_truncate_to_empty")
    cases.append("test_truncate_with_stride")
    cases.append("test_truncate_left")
    cases.append("test_truncate_noop_when_within_limit")
    cases.append("test_truncate_bad_stride_raises")
    cases.append("test_single_longest_first")
    cases.append("test_single_left_keeps_tail")
    cases.append("test_pair_longest_first_splits_evenly")
    cases.append("test_pair_only_second")
    cases.append("test_only_first")
    cases.append("test_no_truncation_needed")
    cases.append("test_only_second_without_pair_raises")
    cases.append("test_sequence_too_short_raises")
    for name in cases:
        try:
            if name == "test_truncate_right_no_stride":
                test_truncate_right_no_stride()
            elif name == "test_truncate_to_empty":
                test_truncate_to_empty()
            elif name == "test_truncate_with_stride":
                test_truncate_with_stride()
            elif name == "test_truncate_left":
                test_truncate_left()
            elif name == "test_truncate_noop_when_within_limit":
                test_truncate_noop_when_within_limit()
            elif name == "test_truncate_bad_stride_raises":
                test_truncate_bad_stride_raises()
            elif name == "test_single_longest_first":
                test_single_longest_first()
            elif name == "test_single_left_keeps_tail":
                test_single_left_keeps_tail()
            elif name == "test_pair_longest_first_splits_evenly":
                test_pair_longest_first_splits_evenly()
            elif name == "test_pair_only_second":
                test_pair_only_second()
            elif name == "test_only_first":
                test_only_first()
            elif name == "test_no_truncation_needed":
                test_no_truncation_needed()
            elif name == "test_only_second_without_pair_raises":
                test_only_second_without_pair_raises()
            elif name == "test_sequence_too_short_raises":
                test_sequence_too_short_raises()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
