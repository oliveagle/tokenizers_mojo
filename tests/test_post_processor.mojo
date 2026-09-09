"""Unit tests for post processors (RobertaProcessing / BertProcessing).

Run:  mojo run -I src -I tests tests/test_post_processor.mojo
"""

import encoding
import post_processor

from encoding import Encoding
from post_processor import RobertaProcessing, BertProcessing
from harness import expect, expect_eq, expect_str, summarize


def make_enc() -> Encoding:
    var enc = Encoding()
    enc.push(100, "hello", Tuple[Int, Int](0, 5))
    enc.push(101, "world", Tuple[Int, Int](6, 11))
    return enc^


def test_roberta_adds_cls_sep() raises:
    var rp = RobertaProcessing("<s>", 0, "</s>", 2)
    var out = rp.process(make_enc(), True)
    expect_eq(out.len(), 4, "cls + 2 + sep")
    expect_str(out.token_at(0), "<s>", "cls first")
    expect_str(out.token_at(3), "</s>", "sep last")
    expect_eq(out.special_tokens_mask[0], 1, "cls special")
    expect_eq(out.special_tokens_mask[3], 1, "sep special")
    expect_eq(out.special_tokens_mask[1], 0, "content not special")
    expect_eq(out.id_at(1), 100, "content id preserved")
    expect_eq(out.id_at(2), 101, "content id preserved")


def test_roberta_type_ids_all_zero() raises:
    var rp = RobertaProcessing("<s>", 0, "</s>", 2)
    var out = rp.process(make_enc(), True)
    for i in range(out.len()):
        expect_eq(out.type_ids[i], 0, "type id 0")
    expect_eq(out.sequence_ids[0], -1, "cls has no seq id")
    expect_eq(out.sequence_ids[1], 0, "content seq id 0")


def test_roberta_no_special() raises:
    var rp = RobertaProcessing("<s>", 0, "</s>", 2)
    var out = rp.process(make_enc(), False)
    expect_eq(out.len(), 2, "no cls/sep when add_special_tokens=False")


def test_bert_cls_sep() raises:
    var bp = BertProcessing("[CLS]", 101, "[SEP]", 102)
    var out = bp.process(make_enc(), True)
    expect_eq(out.len(), 4, "cls + 2 + sep")
    expect_str(out.token_at(0), "[CLS]", "cls")
    expect_str(out.token_at(3), "[SEP]", "sep")
    expect_eq(out.special_tokens_mask[0], 1, "cls special")
    expect_eq(out.special_tokens_mask[3], 1, "sep special")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_roberta_adds_cls_sep")
    cases.append("test_roberta_type_ids_all_zero")
    cases.append("test_roberta_no_special")
    cases.append("test_bert_cls_sep")
    for name in cases:
        try:
            if name == "test_roberta_adds_cls_sep":
                test_roberta_adds_cls_sep()
            elif name == "test_roberta_type_ids_all_zero":
                test_roberta_type_ids_all_zero()
            elif name == "test_roberta_no_special":
                test_roberta_no_special()
            elif name == "test_bert_cls_sep":
                test_bert_cls_sep()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
