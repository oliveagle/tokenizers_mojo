"""Tokenizer -- top-level orchestrator (GPT-2 style minimal pipeline).

Pipeline: normalize -> pre_tokenize -> model.encode -> Encoding
Decoding: Encoding.tokens -> ByteLevelDecoder.decode

Phase 1 wires together NFCNormalizer (identity) + ByteLevelPreTokenizer +
BPE model + ByteLevelDecoder.
"""

import byte_level
import bpe
import encoding
import normalizers

from byte_level import ByteLevelPreTokenizer, ByteLevelDecoder
from bpe import BPE
from encoding import Encoding
from normalizers import NFCNormalizer


struct Tokenizer:
    """GPT-2 style tokenizer (Phase 1 minimal closure).

    Attributes:
        normalizer: text normalizer (NFC identity placeholder).
        pre_tokenizer: byte-level pre-tokenizer (splits + byte-maps).
        model: BPE model (vocab + merges).
        decoder: byte-level decoder (inverse of byte-mapping).
        add_special_tokens: whether to add BOS/EOS (Phase 1: False).
    """

    var normalizer: NFCNormalizer
    var pre_tokenizer: ByteLevelPreTokenizer
    var model: BPE
    var decoder: ByteLevelDecoder
    var add_special_tokens: Bool

    def __init__(out self, var model: BPE):
        self.normalizer = NFCNormalizer()
        self.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=True, use_regex=True
        )
        self.model = model^
        self.decoder = ByteLevelDecoder()
        self.add_special_tokens = False

    def __init__(out self):
        self.normalizer = NFCNormalizer()
        self.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=True, use_regex=True
        )
        self.model = BPE()
        self.decoder = ByteLevelDecoder()
        self.add_special_tokens = False

    def encode(self, text: String) raises -> Encoding:
        """Encode a raw string into an Encoding (ids / tokens / offsets)."""
        # 1. normalize
        var norm = self.normalizer.normalize(text)
        # 2. pre-tokenize (split + byte-map)
        var pretokens = self.pre_tokenizer.pre_tokenize(norm)
        var enc = Encoding()
        var char_idx = 0
        for ptok in pretokens:
            # ptok is byte-mapped; compute byte-mapped length -> original chars
            var ids = self.model.encode(ptok)
            var start = char_idx
            var end = char_idx + self._original_char_len(ptok)
            char_idx = end
            for id in ids:
                var token_str = self.model.token_for_id(id)
                enc.push(id, token_str, Tuple[Int, Int](start, end))
        return enc^

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
