"""Tokenizer -- top-level orchestrator (GPT-2 + special-token support).

Pipeline (when no special tokens are registered):
  normalize -> pre_tokenize -> model.encode -> Encoding

Pipeline (with special tokens): the input is first scanned for special
tokens (longest match), then each non-special segment is encoded via the
pipeline above, and special tokens are spliced in via push_special so the
final Encoding carries the special_tokens_mask.

Phase 2 adds:
  * add_special_token(token) and special_tokens dict
  * longest-match scanning for special tokens in the raw input
  * preserved original-text byte offsets across the split

Phase 3 TODO: post_processor integration, added-token normalization,
template processing, BertPreTokenizer wiring.
"""

import byte_level
import bpe
import encoding
import normalizers

from byte_level import ByteLevelPreTokenizer, ByteLevelDecoder
from bpe import BPE
from encoding import Encoding, CompactEncoding
from normalizers import NFCNormalizer


struct Tokenizer:
    """Tokenizer orchestrator (Phase 2: + special token handling).

    Attributes:
        normalizer: text normalizer (NFC identity placeholder).
        pre_tokenizer: byte-level pre-tokenizer (splits + byte-maps).
        model: BPE model (vocab + merges; may carry special-token ids).
        decoder: byte-level decoder (inverse of byte-mapping).
        add_special_tokens: whether to emit specials during encode.
        special_tokens: content -> vocab id for registered special tokens.
    """

    var normalizer: NFCNormalizer
    var pre_tokenizer: ByteLevelPreTokenizer
    var model: BPE
    var decoder: ByteLevelDecoder
    var add_special_tokens: Bool
    var special_tokens: Dict[String, Int]

    def __init__(out self, var model: BPE):
        self.normalizer = NFCNormalizer()
        self.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=True, use_regex=True
        )
        self.model = model^
        self.decoder = ByteLevelDecoder()
        self.add_special_tokens = True
        self.special_tokens = Dict[String, Int]()

    def __init__(out self):
        self.normalizer = NFCNormalizer()
        self.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=True, use_regex=True
        )
        self.model = BPE()
        self.decoder = ByteLevelDecoder()
        self.add_special_tokens = True
        self.special_tokens = Dict[String, Int]()

    def add_special_token(mut self, token: String) -> Int:
        """Register `token` as a special token and return its vocab id.

        If the token is already present in the BPE vocab, reuses that id;
        otherwise assigns a fresh id (max_existing + 1) and adds it to
        the BPE vocab so encode() can look it up directly.
        """
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
        """Return max_existing_vocab_id + 1 (assuming non-negative)."""
        var max_id = -1
        for v in self.model.vocab.values():
            if v > max_id:
                max_id = v
        return max_id + 1

    def encode(self, text: String) raises -> Encoding:
        """Encode `text` into an Encoding, splicing in any special tokens."""
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
        # Pre-read bytes for safe byte-level access (avoids codepoint boundary crash)
        var text_bytes = _read_text_bytes(text)
        var n = len(text_bytes)
        var i = 0
        var seg_start = 0
        while i < n:
            var matched = self._match_special(text, i)
            if matched != "":
                if i > seg_start:
                    # Build substring from byte list
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

    def encode_batch(self, texts: List[String]) raises -> List[Encoding]:
        """Encode each input text in `texts` into an Encoding.

        Mirrors HuggingFace `Tokenizer::encode_batch` (sequential for now --
        Mojo 1.0.0 stdlib does not yet expose a threading primitive, so the
        upstream `par_iter` path is replaced by a sequential loop. The
        public API shape and per-encoding equivalence with `encode()` are
        preserved, so swapping in a parallel worker is a localized change.)
        """
        var out = List[Encoding]()
        for i in range(len(texts)):
            out.append(self.encode(texts[i]))
        return out^

    def decode_batch(self, encodings: List[Encoding]) raises -> List[String]:
        """Decode each Encoding in `encodings` back to text."""
        var out = List[String]()
        for i in range(len(encodings)):
            out.append(self.decode(encodings[i]))
        return out^

    def _match_special(self, text: String, start: Int) -> String:
        """Return the longest special token matched at byte position `start`,
        or "" if nothing matches.
        
        Uses raw byte access via .bytes() to avoid crashes on multi-byte
        UTF-8 characters (Mojo 1.0.0 text[byte=pos] asserts codepoint boundary).
        """
        var best_len = 0
        var best_tok = String()
        var n = text.byte_length()
        # Pre-read text bytes for safe byte-level access
        var text_bytes = List[Int]()
        text_bytes.reserve(n)
        for b in text.bytes():
            text_bytes.append(Int(b))
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
            # Compare bytes from the pre-read lists
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
        """Encode a non-special segment of text and append results to enc."""
        var norm = self.normalizer.normalize(text)
        var pretokens = self.pre_tokenizer.pre_tokenize(norm)
        var char_idx = base_offset
        for ptok in pretokens:
            var toks = self.model.encode_word(ptok)
            var start = char_idx
            var end = char_idx + self._original_char_len(ptok)
            char_idx = end
            for t in toks:
                var id = self.model.vocab.get(t)
                if id:
                    enc.push(id.value(), t, (start, end))

    def _original_char_len(self, byte_mapped: String) -> Int:
        """Approximate original char length of a byte-mapped token.

        Because byte-mapping is 1 byte -> 1 unicode char, the byte count of
        the mapped token equals the original byte count.  For ASCII that is
        the char length; for multi-byte UTF-8 original text the mapping
        inflates bytes, so this is approximate for Phase 1 (offset is best
        effort).
        """
        return byte_mapped.byte_length()

    def decode(self, enc: Encoding) raises -> String:
        """Decode an Encoding back to a string (via ByteLevelDecoder)."""
        return self.decoder.decode(enc.tokens)

    def decode_str(self, tokens: List[String]) raises -> String:
        """Decode a list of byte-mapped tokens to a string."""
        return self.decoder.decode(tokens)


def _codepoint_byte_len_from_bytes(text_bytes: List[Int], byte_pos: Int) -> Int:
    """Length (in bytes) of the codepoint starting at `byte_pos`.
    
    Uses pre-read byte list to avoid Mojo 1.0.0 codepoint boundary assertion.
    """
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


def _read_text_bytes(text: String) -> List[Int]:
    """Read all bytes from text into a List for safe byte-level access.
    
    Avoids Mojo 1.0.0's text[byte=pos] codepoint boundary assertion.
    """
    var result = List[Int]()
    result.reserve(text.byte_length())
    for b in text.bytes():
        result.append(Int(b))
    return result^
