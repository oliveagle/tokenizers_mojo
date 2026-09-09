"""Post processors -- add special tokens / type ids (Phase 2).

Behavior baseline: HuggingFace tokenizers `processors/`.

RobertaProcessing: wraps encodings with <s> ... </s>, sets all type_ids
to 0, appends 1 to special_tokens_mask for cls/sep and 0 for content
tokens.  Handles a single encoding (batch handling is Phase 3 TODO).
"""

import encoding

from encoding import Encoding


struct RobertaProcessing:
    """Roberta-style post processor: <s> + tokens + </s>.

    Attributes:
        cls_token: The CLS token string (e.g. "<s>").
        sep_token: The SEP token string (e.g. "</s>").
        cls_id: Vocab id of CLS.
        sep_id: Vocab id of SEP.
        trim_offsets: Whether to trim leading space from token offsets
            (Phase 1: pass through; TODO Phase 3).
    """

    var cls_token: String
    var sep_token: String
    var cls_id: Int
    var sep_id: Int
    var trim_offsets: Bool

    def __init__(
        out self, cls_token: String, cls_id: Int, sep_token: String, sep_id: Int
    ):
        self.cls_token = cls_token
        self.cls_id = cls_id
        self.sep_token = sep_token
        self.sep_id = sep_id
        self.trim_offsets = True

    def process(
        self, var enc: Encoding, add_special_tokens: Bool
    ) raises -> Encoding:
        """Apply Roberta processing to a single Encoding."""
        # Roberta: all type_ids = 0
        enc.set_type_ids_all(0)

        if not add_special_tokens:
            return enc^

        var out = Encoding()
        # [CLS]
        out.push_special(
            self.cls_id,
            self.cls_token,
            Tuple[Int, Int](0, 0),
        )
        out.set_sequence_id(0, -1)  # special tokens have no sequence id
        # content
        for i in range(enc.len()):
            var id = enc.id_at(i)
            var tok = enc.token_at(i)
            var off = enc.offset_at(i)
            var is_special = enc.special_tokens_mask[i]
            if is_special:
                out.push_special(id, tok, off)
            else:
                out.push(id, tok, off)
            out.set_sequence_id(i + 1, 0)
        # [SEP]
        out.push_special(
            self.sep_id,
            self.sep_token,
            Tuple[Int, Int](0, 0),
        )
        out.set_sequence_id(out.len() - 1, -1)
        return out^


struct BertProcessing:
    """BERT-style post processor: [CLS] + tokens + [SEP]."""

    var cls_token: String
    var sep_token: String
    var cls_id: Int
    var sep_id: Int

    def __init__(
        out self, cls_token: String, cls_id: Int, sep_token: String, sep_id: Int
    ):
        self.cls_token = cls_token
        self.cls_id = cls_id
        self.sep_token = sep_token
        self.sep_id = sep_id

    def process(
        self, var enc: Encoding, add_special_tokens: Bool
    ) raises -> Encoding:
        """Apply BERT processing (type_ids: CLS=0, content=0, SEP=0)."""
        if not add_special_tokens:
            return enc^

        var out = Encoding()
        out.push_special(self.cls_id, self.cls_token, Tuple[Int, Int](0, 0))
        out.set_sequence_id(0, 0)
        for i in range(enc.len()):
            var id = enc.id_at(i)
            var tok = enc.token_at(i)
            var off = enc.offset_at(i)
            var is_special = enc.special_tokens_mask[i]
            if is_special:
                out.push_special(id, tok, off)
            else:
                out.push(id, tok, off)
            out.set_sequence_id(i + 1, 0)
        out.push_special(self.sep_id, self.sep_token, Tuple[Int, Int](0, 0))
        out.set_sequence_id(out.len() - 1, 0)
        return out^


def process(
    var enc: Encoding,
    add_special_tokens: Bool,
    cls_token: String,
    cls_id: Int,
    sep_token: String,
    sep_id: Int,
) raises -> Encoding:
    """Standalone convenience (RobertaProcessing)."""
    var rp = RobertaProcessing(cls_token, cls_id, sep_token, sep_id)
    return rp.process(enc^, add_special_tokens)
