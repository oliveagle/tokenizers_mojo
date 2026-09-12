"""Tokenizer V5 — optimized pre-tokenize + NFC ASCII fast path."""

import byte_level_fast
import bpe
import encoding
import normalizers_optimized

from byte_level_fast import ByteLevelPreTokenizerFast
from bpe import BPE
from encoding import Encoding
from normalizers_optimized import NFCNormalizerOptimized


struct TokenizerV5:
    """Tokenizer with optimized pre-tokenize + NFC."""

    var normalizer: NFCNormalizerOptimized
    var pre_tokenizer: ByteLevelPreTokenizerFast
    var model: BPE
    var add_special_tokens: Bool
    var special_tokens: Dict[String, Int]

    def __init__(out self, var model: BPE):
        self.normalizer = NFCNormalizerOptimized()
        self.pre_tokenizer = ByteLevelPreTokenizerFast(
            add_prefix_space=True, use_regex=True
        )
        self.model = model^
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

        # Pre-read all bytes for safe byte-level access
        # (avoids Mojo 1.0.0 text[byte=pos] codepoint boundary crash)
        var text_bytes = List[Int]()
        text_bytes.reserve(text.byte_length())
        for b in text.bytes():
            text_bytes.append(Int(b))
        var n = len(text_bytes)
        var i = 0
        var seg_start = 0
        while i < n:
            var matched = self._match_special_safe(text, text_bytes, i)
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
                    id, matched, Tuple[Int, Int](i, i + matched.byte_length())
                )
                i += matched.byte_length()
                seg_start = i
            else:
                i += _codepoint_byte_len_from_bytes(text_bytes, i)
        if seg_start < n:
            var seg = String()
            var j = seg_start
            while j < n:
                seg += chr(text_bytes[j])
                j += 1
            self._encode_segment(enc, seg, seg_start)
        return enc^

    def _match_special_safe(
        self, text: String, text_bytes: List[Int], start: Int
    ) -> String:
        """Match special tokens using pre-read byte list for safe access."""
        var best_len = 0
        var best_tok = String()
        var n = len(text_bytes)
        for tok in self.special_tokens:
            var blen = tok.byte_length()
            if blen == 0 or blen > n - start:
                continue
            if blen <= best_len:
                continue
            # Pre-read special token bytes
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
        var decoder = byte_level_fast.ByteLevelDecoderFast()
        return decoder.decode(enc.tokens)


def _codepoint_byte_len_from_bytes(text_bytes: List[Int], byte_pos: Int) -> Int:
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
