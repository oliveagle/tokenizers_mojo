"""WordPiece Tokenizer — top-level orchestrator for BERT-style models.

Pipeline: normalize -> pre_tokenize -> WordPiece.encode -> Encoding

Supports loading from HuggingFace tokenizer.json files via from_pretrained_wordpiece.
"""

import byte_level
import encoding
import normalizers
import wordpiece

from byte_level import ByteLevelPreTokenizer
from encoding import Encoding
from normalizers import NFCNormalizer
from wordpiece import WordPiece


struct WordPieceTokenizer:
    """Tokenizer for WordPiece models (BERT/DistilBERT).

    Attributes:
        normalizer: text normalizer.
        pre_tokenizer: pre-tokenizer (e.g., BertPreTokenizer).
        model: WordPiece model.
        add_special_tokens: whether to emit specials during encode.
        special_tokens: content -> vocab id for registered special tokens.
    """

    var normalizer: NFCNormalizer
    var pre_tokenizer: ByteLevelPreTokenizer
    var model: WordPiece
    var add_special_tokens: Bool
    var special_tokens: Dict[String, Int]

    def __init__(out self, var model: WordPiece):
        self.normalizer = NFCNormalizer()
        self.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=False, use_regex=False
        )
        self.model = model^
        self.add_special_tokens = True
        self.special_tokens = Dict[String, Int]()

    def __init__(out self):
        self.normalizer = NFCNormalizer()
        self.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=False, use_regex=False
        )
        self.model = WordPiece()
        self.add_special_tokens = True
        self.special_tokens = Dict[String, Int]()

    def add_special_token(mut self, token: String) -> Int:
        """Register `token` as a special token and return its vocab id."""
        var existing = self.special_tokens.get(token)
        if existing:
            return existing.value()
        var id = self.model.token_id(token)
        if id == -1:
            id = self._next_vocab_id()
            self.model.vocab[token] = id
            self.model.id_to_token[id] = token
        self.special_tokens[token] = id
        return id

    def _next_vocab_id(self) -> Int:
        """Return max_existing_vocab_id + 1."""
        var max_id = -1
        for v in self.model.vocab.values():
            if v > max_id:
                max_id = v
        return max_id + 1

    def encode(self, text: String) raises -> Encoding:
        """Encode `text` into an Encoding."""
        var enc = Encoding()
        if not self.add_special_tokens or len(self.special_tokens) == 0:
            self._encode_segment(enc, text, 0)
            return enc^

        var n = text.byte_length()
        var i = 0
        var seg_start = 0
        while i < n:
            var matched = self._match_special(text, i)
            if matched != "":
                if i > seg_start:
                    self._encode_segment(
                        enc, String(text[byte=seg_start:i]), seg_start
                    )
                var id = self.special_tokens[matched]
                enc.push_special(
                    id, matched, (i, i + matched.byte_length())
                )
                i += matched.byte_length()
                seg_start = i
            else:
                i += _codepoint_byte_len(text, i)
        if seg_start < n:
            self._encode_segment(enc, String(text[byte=seg_start:n]), seg_start)
        return enc^

    def _match_special(self, text: String, start: Int) -> String:
        """Return the longest special token matched at byte position `start`."""
        var best = String()
        var n = text.byte_length()
        for tok in self.special_tokens:
            var blen = tok.byte_length()
            if blen == 0:
                continue
            if start + blen > n:
                continue
            var window = String(text[byte=start : start + blen])
            if window == tok and blen > best.byte_length():
                best = tok
        return best

    def _encode_segment(
        self, mut enc: Encoding, text: String, base_offset: Int
    ) raises:
        """Encode a non-special segment of text."""
        var norm = self.normalizer.normalize(text)
        var pretokens = self.pre_tokenizer.pre_tokenize(norm)
        var char_idx = base_offset
        for ptok in pretokens:
            var ids = self.model.encode(ptok)
            var start = char_idx
            var end = char_idx + ptok.byte_length()
            char_idx = end
            for id in ids:
                var token_str = self.model.token_for_id(id)
                enc.push(id, token_str, (start, end))

    def decode(self, enc: Encoding) raises -> String:
        """Decode an Encoding back to a string."""
        var sb = String()
        for i in range(len(enc.ids)):
            var token = self.model.token_for_id(enc.ids[i])
            sb += token
        return sb^


def _codepoint_byte_len(text: String, byte_pos: Int) -> Int:
    """Length (in bytes) of the codepoint starting at `byte_pos`."""
    if byte_pos >= text.byte_length():
        return 0
    var ch = String(text[byte=byte_pos : byte_pos + 1])
    var first = 0
    for cp in ch.codepoints():
        first = Int(cp)
        break
    if first < 0x80:
        return 1
    if first < 0xE0:
        return 2
    if first < 0xF0:
        return 3
    return 4
