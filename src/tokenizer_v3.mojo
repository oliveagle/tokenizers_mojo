"""Tokenizer V3 — V1 components + optimized NFC normalization."""

import byte_level
import bpe
import encoding
import normalizers_optimized

from byte_level import ByteLevelPreTokenizer, ByteLevelDecoder
from bpe import BPE
from encoding import Encoding
from normalizers_optimized import NFCNormalizerOptimized

def _read_text_bytes(text: String) -> List[Int]:
    """Read all bytes from text into a List for safe byte-level access."""
    var result = List[Int]()
    result.reserve(text.byte_length())
    for b in text.bytes():
        result.append(Int(b))
    return result^


def _codepoint_byte_len_safe(text_bytes: List[Int], byte_pos: Int) -> Int:
    """Length (in bytes) of the codepoint at byte_pos, using pre-read byte list."""
    if byte_pos >= len(text_bytes):
        return 0
    var first = text_bytes[byte_pos]
    if first < 0x80:
        return 1
    if first < 0xE0:
        return 2
    if first < 0xF0:
        return 3
    return 4


def _match_special_safe(text_bytes: List[Int], special_tokens: Dict[String, Int], start: Int) -> String:
    """Match special tokens using pre-read byte list for safe access."""
    var best_len = 0
    var best_tok = String()
    var n = len(text_bytes)
    for tok in special_tokens:
        var blen = tok.byte_length()
        if blen == 0 or blen > n - start:
            continue
        if blen <= best_len:
            continue
        var tok_bytes = List[Int]()
        tok_bytes.reserve(blen)
        for b in tok.bytes():
            tok_bytes.append(Int(b))
        var is_match = True
        for j in range(blen):
            if text_bytes[start + j] != tok_bytes[j]:
                is_match = False
                break
        if is_match:
            best_len = blen
            best_tok = tok
    return best_tok


struct TokenizerV3:
    """Tokenizer with ASCII fast path for NFC."""

    var normalizer: NFCNormalizerOptimized
    var pre_tokenizer: ByteLevelPreTokenizer
    var model: BPE
    var decoder: ByteLevelDecoder
    var add_special_tokens: Bool
    var special_tokens: Dict[String, Int]

    def __init__(out self, var model: BPE):
        self.normalizer = NFCNormalizerOptimized()
        self.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=True, use_regex=True
        )
        self.model = model^
        self.decoder = ByteLevelDecoder()
        self.add_special_tokens = True
        self.special_tokens = Dict[String, Int]()

    def encode(self, text: String) raises -> Encoding:
        var enc = Encoding()
        var est = text.byte_length() // 3 + 4
        enc.ids.reserve(est)
        enc.tokens.reserve(est)
        enc.offsets.reserve(est)
        enc.type_ids.reserve(est)
        enc.attention_mask.reserve(est)
        enc.special_tokens_mask.reserve(est)
        enc.sequence_ids.reserve(est)

        if not self.add_special_tokens or len(self.special_tokens) == 0:
            self._encode_segment(enc, text, 0)
            return enc^

        # Pre-read bytes for safe byte-level access
        var text_bytes = _read_text_bytes(text)
        var n = len(text_bytes)
        var i = 0
        var seg_start = 0
        while i < n:
            var matched = _match_special_safe(text_bytes, self.special_tokens, i)
            if matched != "":
                if i > seg_start:
                    var seg = String()
                    var j = seg_start
                    while j < i:
                        seg += chr(text_bytes[j])
                        j += 1
                    self._encode_segment(enc, seg, seg_start)
                var id = self.special_tokens[matched]
                enc.push_special(
                    id, matched, (i, i + matched.byte_length())
                )
                i += matched.byte_length()
                seg_start = i
            else:
                i += _codepoint_byte_len_safe(text_bytes, i)
        if seg_start < n:
            var seg = String()
            var j = seg_start
            while j < n:
                seg += chr(text_bytes[j])
                j += 1
            self._encode_segment(enc, seg, seg_start)
        return enc^

    def _match_special(self, text: String, start: Int) -> String:
        var best_len = 0
        var best_tok = String()
        var n = text.byte_length()
        for tok in self.special_tokens:
            var blen = tok.byte_length()
            if blen == 0 or blen > n - start:
                continue
            if blen <= best_len:
                continue
            var is_match = True
            for j in range(blen):
                if text_bytes[start + j] != tok_bytes[j]:
                    is_match = False
                    break
            if is_match:
                best_len = blen
                best_tok = tok
        return best_tok

    def _encode_segment(
        self, mut enc: Encoding, text: String, base_offset: Int
    ) raises:
        var norm = self.normalizer.normalize(text)
        var pretokens = self.pre_tokenizer.pre_tokenize(norm)
        var char_idx = base_offset
        for ptok in pretokens:
            var toks = self.model.encode_word(ptok)
            var start = char_idx
            var end = char_idx + ptok.byte_length()
            char_idx = end
            for t in toks:
                var id = self.model.vocab.get(t)
                if id:
                    enc.push(id.value(), t, (start, end))

    def decode(self, enc: Encoding) raises -> String:
        return self.decoder.decode(enc.tokens)


def _codepoint_byte_len(text: String, byte_pos: Int) -> Int:
    """Length (in bytes) of the codepoint starting at `byte_pos`."""
    var text_bytes = _read_text_bytes(text)
    return _codepoint_byte_len_safe(text_bytes, byte_pos)
