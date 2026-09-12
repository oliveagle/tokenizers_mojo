"""WordLevel Tokenizer — top-level orchestrator for word-level models.

Pipeline: normalize -> pre_tokenize -> WordLevel.encode -> Encoding

Supports loading from HuggingFace tokenizer.json files via from_pretrained_wordlevel.
"""

import byte_level
import encoding
import normalizers
import wordlevel

from byte_level import ByteLevelPreTokenizer
from encoding import Encoding
from normalizers import NFCNormalizer
from wordlevel import WordLevel

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


struct WordLevelTokenizer:
    """Tokenizer for WordLevel models (whole-word tokenization).

    Attributes:
        normalizer: text normalizer.
        pre_tokenizer: pre-tokenizer (e.g., Whitespace).
        model: WordLevel model.
        add_special_tokens: whether to emit specials during encode.
        special_tokens: content -> vocab id for registered special tokens.
    """

    var normalizer: NFCNormalizer
    var pre_tokenizer: ByteLevelPreTokenizer
    var model: WordLevel
    var add_special_tokens: Bool
    var special_tokens: Dict[String, Int]

    def __init__(out self, var model: WordLevel):
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
        self.model = WordLevel()
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
        """Return the longest special token matched at byte position `start`."""
        var best = String()
        var n = text.byte_length()
        for tok in self.special_tokens:
            var blen = tok.byte_length()
            if blen == 0:
                continue
            if start + blen > n:
                continue
            var window = String()
                var wb = _read_text_bytes(text)
                var wi = start
                while wi < start + blen:
                    window += chr(wb[wi])
                    wi += 1
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
            var id = self.model.token_id(ptok)
            var start = char_idx
            var end = char_idx + ptok.byte_length()
            char_idx = end
            if id != -1:
                enc.push(id, ptok, (start, end))
            else:
                # Unknown word - skip or use unk token
                pass

    def decode(self, enc: Encoding) raises -> String:
        """Decode an Encoding back to a string."""
        var sb = String()
        for i in range(len(enc.ids)):
            var token = self.model.token_for_id(enc.ids[i])
            sb += token
        return sb^


def _codepoint_byte_len(text: String, byte_pos: Int) -> Int:
    """Length (in bytes) of the codepoint starting at `byte_pos`."""
    var text_bytes = _read_text_bytes(text)
    return _codepoint_byte_len_safe(text_bytes, byte_pos)
