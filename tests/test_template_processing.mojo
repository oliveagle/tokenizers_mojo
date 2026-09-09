"""Unit tests for TemplateProcessing post-processor (Phase 2).

Reference values verified against HuggingFace tokenizers 0.22.2
`TemplateProcessing` with:

    single = "[CLS] $A [SEP]"
    pair   = "[CLS] $A [SEP] $B:1 [SEP]:1"
    special_tokens = [("[CLS]", 4), ("[SEP]", 3)]

"hello world" -> [1, 2]; "foo" -> [6]

Run:  mojo run -I src -I tests tests/test_template_processing.mojo
"""

from encoding import Encoding
from template_processing import TemplateProcessing, template_process
from harness import expect, expect_eq, summarize


def make_cls_only() -> List[Tuple[String, Int]]:
    var out = List[Tuple[String, Int]]()
    out.append(Tuple[String, Int]("[CLS]", 4))
    return out^


def make_specials() -> List[Tuple[String, Int]]:
    var out = List[Tuple[String, Int]]()
    out.append(Tuple[String, Int]("[CLS]", 4))
    out.append(Tuple[String, Int]("[SEP]", 3))
    return out^


def push_int(mut lst: List[Int], v: Int) raises:
    lst.append(v)


def push_str(mut lst: List[String], v: String) raises:
    lst.append(v)


def make_enc(ids: List[Int], tokens: List[String]) raises -> Encoding:
    var enc = Encoding()
    for i in range(len(ids)):
        enc.push(ids[i], tokens[i], Tuple[Int, Int](0, 0))
    return enc^


def make_seq_a() raises -> Encoding:
    var ids = List[Int]()
    push_int(ids, 1)
    push_int(ids, 2)
    var toks = List[String]()
    push_str(toks, "hello")
    push_str(toks, "world")
    return make_enc(ids, toks)


def make_seq_b() raises -> Encoding:
    var ids = List[Int]()
    push_int(ids, 6)
    var toks = List[String]()
    push_str(toks, "foo")
    return make_enc(ids, toks)


def make_bert() raises -> TemplateProcessing:
    return TemplateProcessing(
        "[CLS] $A [SEP]",
        "[CLS] $A [SEP] $B:1 [SEP]:1",
        make_specials(),
    )


def test_single_bert() raises:
    var tp = make_bert()
    var out = tp.process(make_seq_a(), Encoding(), True)
    expect_eq(out.len(), 4, "cls + 2 content + sep")
    expect_eq(out.ids[0], 4, "cls id")
    expect_eq(out.ids[1], 1, "hello id")
    expect_eq(out.ids[2], 2, "world id")
    expect_eq(out.ids[3], 3, "sep id")
    for i in range(4):
        expect_eq(out.type_ids[i], 0, "single type_ids all 0")
    expect_eq(out.special_tokens_mask[0], 1, "cls special")
    expect_eq(out.special_tokens_mask[1], 0, "hello not special")
    expect_eq(out.special_tokens_mask[2], 0, "world not special")
    expect_eq(out.special_tokens_mask[3], 1, "sep special")
    expect_eq(out.sequence_ids[0], -1, "cls no seq id")
    expect_eq(out.sequence_ids[1], 0, "hello seq 0")
    expect_eq(out.sequence_ids[2], 0, "world seq 0")
    expect_eq(out.sequence_ids[3], -1, "sep no seq id")


def test_pair_bert() raises:
    var tp = make_bert()
    var out = tp.process(make_seq_a(), make_seq_b(), True)
    expect_eq(out.len(), 6, "cls + A(2) + sep + B(1) + sep")
    expect_eq(out.ids[0], 4, "cls id")
    expect_eq(out.ids[1], 1, "hello id")
    expect_eq(out.ids[2], 2, "world id")
    expect_eq(out.ids[3], 3, "sep id (after A)")
    expect_eq(out.ids[4], 6, "foo id")
    expect_eq(out.ids[5], 3, "sep id (after B)")
    for i in range(4):
        expect_eq(out.type_ids[i], 0, "pair type_ids A section 0")
    expect_eq(out.type_ids[4], 1, "pair type_ids B section 1")
    expect_eq(out.type_ids[5], 1, "pair type_ids trailing sep 1")
    expect_eq(out.special_tokens_mask[0], 1, "cls special")
    expect_eq(out.special_tokens_mask[1], 0, "hello not special")
    expect_eq(out.special_tokens_mask[2], 0, "world not special")
    expect_eq(out.special_tokens_mask[3], 1, "sep1 special")
    expect_eq(out.special_tokens_mask[4], 0, "foo not special")
    expect_eq(out.special_tokens_mask[5], 1, "sep2 special")
    expect_eq(out.sequence_ids[0], -1, "cls no seq id")
    expect_eq(out.sequence_ids[1], 0, "hello seq 0")
    expect_eq(out.sequence_ids[2], 0, "world seq 0")
    expect_eq(out.sequence_ids[3], -1, "sep1 no seq id")
    expect_eq(out.sequence_ids[4], 1, "foo seq 1")
    expect_eq(out.sequence_ids[5], -1, "sep2 no seq id")


def test_no_special_tokens() raises:
    var tp = make_bert()
    var out = tp.process(make_seq_a(), Encoding(), False)
    expect_eq(out.len(), 2, "no special when add_special_tokens=False")
    expect_eq(out.ids[0], 1, "content id preserved")
    expect_eq(out.ids[1], 2, "content id preserved")


def test_dollar_zero_form() raises:
    # single = "[CLS]:0 $0 [SEP]:0" -> all type_ids 0
    var tp = TemplateProcessing(
        "[CLS]:0 $0 [SEP]:0",
        "[CLS]:0 $0 [SEP]:0 $1:1 [SEP]:1",
        make_specials(),
    )
    var out = tp.process(make_seq_a(), Encoding(), True)
    expect_eq(out.len(), 4, "$0 form: cls + 2 + sep")
    expect_eq(out.ids[0], 4, "cls id")
    expect_eq(out.ids[1], 1, "hello id")
    expect_eq(out.ids[2], 2, "world id")
    expect_eq(out.ids[3], 3, "sep id")
    for i in range(4):
        expect_eq(out.type_ids[i], 0, "$0 form type_ids all 0")


def test_dollar_one_pair() raises:
    # $0 / $1 form in pair template
    var tp = TemplateProcessing(
        "[CLS]:0 $0 [SEP]:0",
        "[CLS]:0 $0 [SEP]:0 $1:1 [SEP]:1",
        make_specials(),
    )
    var out = tp.process(make_seq_a(), make_seq_b(), True)
    expect_eq(out.len(), 6, "$0/$1 pair: cls + A(2) + sep + B(1) + sep")
    expect_eq(out.ids[4], 6, "foo id")
    expect_eq(out.type_ids[4], 1, "B section type_id 1")
    expect_eq(out.type_ids[5], 1, "trailing sep type_id 1")


def test_explicit_type_id() raises:
    # $A:1 assigns type_id 1 to seq A
    var tp = TemplateProcessing(
        "[CLS] $A:1 [SEP]",
        "[CLS] $A [SEP]",
        make_specials(),
    )
    var out = tp.process(make_seq_a(), Encoding(), True)
    expect_eq(out.type_ids[0], 0, "cls type_id 0")
    expect_eq(out.type_ids[1], 1, "hello type_id 1 from $A:1")
    expect_eq(out.type_ids[2], 1, "world type_id 1 from $A:1")
    expect_eq(out.type_ids[3], 0, "sep type_id 0 (no :1 suffix)")


def test_explicit_a1_b1() raises:
    # Both A and B get type_id 1
    var tp = TemplateProcessing(
        "[CLS] $A:1 [SEP]",
        "[CLS] $A:1 [SEP] $B:1 [SEP]:1",
        make_specials(),
    )
    var out = tp.process(make_seq_a(), Encoding(), True)
    expect_eq(out.type_ids[1], 1, "$A:1 content type_id 1")


def test_standalone_helper() raises:
    var out = template_process(
        make_seq_a(),
        True,
        "[CLS] $A [SEP]",
        "[CLS] $A [SEP]",
        make_specials(),
    )
    expect_eq(out.len(), 4, "standalone helper output length")
    expect_eq(out.ids[0], 4, "standalone cls id")


def test_unknown_special_raises() raises:
    var failed = False
    try:
        var _ = TemplateProcessing(
            "[UNKNOWN] $A [SEP]",
            "[UNKNOWN] $A [SEP]",
            make_cls_only(),
        )
    except:
        failed = True
    expect(failed, "unknown special token should raise")


def main() raises:
    var failures = List[String]()
    var cases = List[String]()
    cases.append("test_single_bert")
    cases.append("test_pair_bert")
    cases.append("test_no_special_tokens")
    cases.append("test_dollar_zero_form")
    cases.append("test_dollar_one_pair")
    cases.append("test_explicit_type_id")
    cases.append("test_explicit_a1_b1")
    cases.append("test_standalone_helper")
    cases.append("test_unknown_special_raises")
    for name in cases:
        try:
            if name == "test_single_bert":
                test_single_bert()
            elif name == "test_pair_bert":
                test_pair_bert()
            elif name == "test_no_special_tokens":
                test_no_special_tokens()
            elif name == "test_dollar_zero_form":
                test_dollar_zero_form()
            elif name == "test_dollar_one_pair":
                test_dollar_one_pair()
            elif name == "test_explicit_type_id":
                test_explicit_type_id()
            elif name == "test_explicit_a1_b1":
                test_explicit_a1_b1()
            elif name == "test_standalone_helper":
                test_standalone_helper()
            elif name == "test_unknown_special_raises":
                test_unknown_special_raises()
            print("  PASS " + name)
        except e:
            print("  FAIL " + name + " :: " + String(e))
            failures.append(name + ": " + String(e))
    summarize(failures)
