"""Tokenizer V4 — with no-shift BPE + optimized NFC."""

import byte_level
import bpe_noshift
import encoding
import normalizers_optimized

from byte_level import ByteLevelPreTokenizer
from bpe_noshift import BPENoShift
from encoding import Encoding
from normalizers_optimized import NFCNormalizerOptimized

def _read_text_bytes(text: String) -> List[Int]:
    """Read all bytes from text into a List for safe byte-level access."""
    var result = List[Int]()
    result.reserve(text.byte_length())
    for b in text.bytes():
        result.append(Int(b))
    return result^



struct TokenizerV4:
    """Tokenizer with no-shift BPE + ASCII fast NFC."""

    var normalizer: NFCNormalizerOptimized
    var pre_tokenizer: ByteLevelPreTokenizer
    var model: BPENoShift
    var decoder: ByteLevelPreTokenizer
    var add_special_tokens: Bool
    var special_tokens: Dict[String, Int]

    def __init__(out self, var model: BPENoShift):
        self.normalizer = NFCNormalizerOptimized()
        self.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=True, use_regex=True
        )
        self.model = model^
        self.decoder = ByteLevelPreTokenizer(
            add_prefix_space=True, use_regex=True
        )
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

        var n = text.byte_length()
        var i = 0
        var seg_start = 0
        while i < n:
            var matched = self._match_special(text, i)
            if matched != "":
                if i > seg_start:
                    var seg = String()
                    var sj = seg_start
                    while sj < i:
                        seg += chr(text_bytes[sj])
                        sj += 1
                    self._encode_segment(enc, seg, seg_start)
                var id = self.special_tokens[matched]
                enc.push_special(
                    id, matched, Tuple[Int, Int](i, i + matched.byte_length())
                )
                i += matched.byte_length()
                seg_start = i
            else:
                i += _codepoint_byte_len(text, i)

        if seg_start < n:
            var seg = String()
            var sj = seg_start
            while sj < n:
                seg += chr(text_bytes[sj])
                sj += 1
            self._encode_segment(enc, seg, seg_start)
        return enc^

    def _match_special(self, text: String, start: Int) -> String:
        """Return the longest special token matched at byte position `start`."""
        var text_bytes = _read_text_bytes(text)
        var best_len = 0
        var best_tok = String()
        var n = len(text_bytes)
        for tok in self.special_tokens:
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
        """Decode using byte_level decoder."""
        var byte_buf = List[Int]()
        for token in enc.tokens:
            for cp in token.codepoints():
                var cp_val = Int(cp)
                var tables = self.pre_tokenizer.byte_mapping
                var orig: Int
                if cp_val < 256:
                    orig = cp_val
                elif cp_val - 256 < 256:
                    orig = -1  # Will use Dict lookup
                else:
                    orig = cp_val

                if orig >= 0 and orig < 256:
                    if orig < 0x80:
                        byte_buf.append(orig)
                    elif orig < 0x800:
                        byte_buf.append(0xC0 | (orig >> 6))
                        byte_buf.append(0x80 | (orig & 0x3F))
                    else:
                        byte_buf.append(0xE0 | (orig >> 12))
                        byte_buf.append(0x80 | ((orig >> 6) & 0x3F))
                        byte_buf.append(0x80 | (orig & 0x3F))
                else:
                    var c = chr(cp_val)
                    var opt_b = tables.u2b.get(c)
                    if opt_b:
                        var bv = opt_b.value()
                        if bv < 0x80:
                            byte_buf.append(bv)
                        elif bv < 0x800:
                            byte_buf.append(0xC0 | (bv >> 6))
                            byte_buf.append(0x80 | (bv & 0x3F))
                        else:
                            byte_buf.append(0xE0 | (bv >> 12))
                            byte_buf.append(0x80 | ((bv >> 6) & 0x3F))
                            byte_buf.append(0x80 | (bv & 0x3F))
                    else:
                        for raw in c.bytes():
                            byte_buf.append(Int(raw))

        var result = String()
        for b in byte_buf:
            result += chr(b)
        return result


def _codepoint_byte_len(text: String, byte_pos: Int) -> Int:
    if byte_pos >= text.byte_length():
        return 0
    var first = -1
    var byte_val = text_bytes[byte_pos] if byte_pos < len(text_bytes) else 0
    var ch = chr(byte_val)
        first = Int(b)
        break
    if first < 0x80:
        return 1
    if first < 0xE0:
        return 2
    if first < 0xF0:
        return 3
    return 4
