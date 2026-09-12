"""Tokenizer V2 — orchestrator using optimized components."""

import byte_level_v2
import bpe_v2
import compact_encoding
import normalizers

from byte_level_v2 import ByteLevelPreTokenizer, ByteTables
from bpe_v2 import BPE
from compact_encoding import CompactEncoding
from normalizers import NFCNormalizer

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


struct TokenizerV2:
    """Tokenizer using V2 optimized components."""

    var normalizer: NFCNormalizer
    var pre_tokenizer: ByteLevelPreTokenizer
    var model: BPE
    var add_special_tokens: Bool
    var special_tokens: Dict[String, Int]

    def __init__(out self, var model: BPE):
        self.normalizer = NFCNormalizer()
        self.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=True, use_regex=True
        )
        self.model = model^
        self.add_special_tokens = True
        self.special_tokens = Dict[String, Int]()

    def __init__(out self):
        self.normalizer = NFCNormalizer()
        self.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=True, use_regex=True
        )
        self.model = BPE()
        self.add_special_tokens = True
        self.special_tokens = Dict[String, Int]()

    def add_special_token(mut self, token: String) -> Int:
        var existing = self.special_tokens.get(token)
        if existing:
            return existing.value()
        var id = self.model.token_id(token)
        if id == -1:
            id = self._next_vocab_id()
            self.model.add_raw_vocab(token, id)
        self.special_tokens[token] = id
        return id

    def _next_vocab_id(self) -> Int:
        var max_id = -1
        for v in self.model.vocab.values():
            if v > max_id:
                max_id = v
        return max_id + 1

    def encode(self, text: String) raises -> CompactEncoding:
        var enc = CompactEncoding()
        enc.reserve(text.byte_length() // 3 + 4)

        if not self.add_special_tokens or len(self.special_tokens) == 0:
            self._encode_segment(enc, text, 0)
            return enc^

        var text_bytes = _read_text_bytes(text)
        var n = len(text_bytes)
        var i = 0
        var seg_start = 0
        while i < n:
            var matched = _match_special_safe(text_bytes, self.special_tokens, i)
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
                    id,
                    matched,
                    Tuple[Int, Int](i, i + matched.byte_length()),
                )
                i += matched.byte_length()
                seg_start = i
            else:
                i += _codepoint_byte_len_safe(text_bytes, i)

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
        var best_len = 0
        var best_tok = String()
        var n = text.byte_length()
        var text_bytes = _read_text_bytes(text)
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
        self, mut enc: CompactEncoding, text: String, base_offset: Int
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

    def decode(self, enc: CompactEncoding) raises -> String:
        """Decode CompactEncoding back to string."""
        var tables = self.pre_tokenizer.tables
        var total_bytes = 0
        for i in range(enc.len()):
            total_bytes += enc.token_off[i * 2 + 1]

        var byte_buf = List[Int]()
        byte_buf.reserve(total_bytes)

        for i in range(enc.len()):
            var tstart = enc.token_off[i * 2]
            var tlen = enc.token_off[i * 2 + 1]
            # Decode UTF-8 codepoints from arena bytes
            var ti = tstart
            while ti < tstart + tlen:
                var b0 = enc.tokens_arena[ti]
                var cp: Int
                if b0 < 0x80:
                    cp = b0
                    ti += 1
                elif b0 < 0xE0 and ti + 1 < tstart + tlen:
                    cp = ((b0 & 0x1F) << 6) | (
                        enc.tokens_arena[ti + 1] & 0x3F
                    )
                    ti += 2
                elif b0 < 0xF0 and ti + 2 < tstart + tlen:
                    cp = ((b0 & 0x0F) << 12) | (
                        (enc.tokens_arena[ti + 1] & 0x3F) << 6
                    ) | (enc.tokens_arena[ti + 2] & 0x3F)
                    ti += 3
                elif ti + 3 < tstart + tlen:
                    cp = ((b0 & 0x07) << 18) | (
                        (enc.tokens_arena[ti + 1] & 0x3F) << 12
                    ) | (
                        (enc.tokens_arena[ti + 2] & 0x3F) << 6
                    ) | (
                        enc.tokens_arena[ti + 3] & 0x3F
                    )
                    ti += 4
                else:
                    cp = b0
                    ti += 1

                var orig: Int
                if cp < 256:
                    orig = cp
                elif cp - 256 < tables.u2b_max_idx:
                    orig = tables.u2b[cp - 256]
                else:
                    orig = cp

                if orig < 0x80:
                    byte_buf.append(orig)
                elif orig < 0x800:
                    byte_buf.append(0xC0 | (orig >> 6))
                    byte_buf.append(0x80 | (orig & 0x3F))
                else:
                    byte_buf.append(0xE0 | (orig >> 12))
                    byte_buf.append(0x80 | ((orig >> 6) & 0x3F))
                    byte_buf.append(0x80 | (orig & 0x3F))

        var result = String()
        for b in byte_buf:
            result += chr(b)
        return result


def _codepoint_byte_len(text: String, byte_pos: Int) -> Int:
    """Length (in bytes) of the codepoint starting at `byte_pos`."""
    var text_bytes = _read_text_bytes(text)
    return _codepoint_byte_len_safe(text_bytes, byte_pos)
