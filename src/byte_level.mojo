"""ByteLevel PreTokenizer and Decoder for GPT-2 style tokenization.

Implements the byte-level encoding described in OpenAI's gpt-2 encoder:
    https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9

Maps every byte 0-255 to a printable Unicode character so that BPE can
operate on "text" that faithfully represents arbitrary byte sequences.

Behavior baseline: HuggingFace tokenizers `pre_tokenizers/byte_level.rs`
(upstream Rust, kept read-only in submodules/tokenizers/).

Performance optimization: cache ByteMapping inside the struct.
"""

import traits

from traits import PreTokenizer, Decoder


struct ByteMapping:
    """GPT-2 byte<->unicode mapping tables (built once per instance)."""

    var b2u: Dict[Int, String]
    var u2b: Dict[String, Int]

    def __init__(out self):
        self.b2u = Dict[Int, String]()
        self.u2b = Dict[String, Int]()
        self._build()

    def _build(mut self):
        # GPT-2 convention:
        #   printable ASCII 0x21..0x7E + Latin-1 0xA1..0xAC, 0xAE..0xFF map
        #   to themselves; all remaining bytes map to codepoints 256, 257, ...
        var bs = List[Int]()
        var cs = List[Int]()

        var b = 0x21
        while b <= 0x7E:
            bs.append(b)
            cs.append(b)
            b += 1
        b = 0xA1
        while b <= 0xAC:
            bs.append(b)
            cs.append(b)
            b += 1
        b = 0xAE
        while b <= 0xFF:
            bs.append(b)
            cs.append(b)
            b += 1

        var n = 0
        b = 0
        while b <= 255:
            var found = False
            for existing in bs:
                if existing == b:
                    found = True
                    break
            if not found:
                bs.append(b)
                cs.append(256 + n)
                n += 1
            b += 1

        for i in range(len(bs)):
            self.b2u[bs[i]] = chr(cs[i])
            self.u2b[chr(cs[i])] = bs[i]

    def byte_to_char(self, byte: Int) raises -> String:
        return self.b2u[byte]

    def char_to_byte(self, c: String) raises -> Int:
        return self.u2b[c]


# ---------------------------------------------------------------------------
# GPT-2 style pre-tokenization (simplified regex substitute)
# ---------------------------------------------------------------------------

def _is_letter(cp: Int) -> Bool:
    if (cp >= 0x41 and cp <= 0x5A) or (cp >= 0x61 and cp <= 0x7A):
        return True
    if cp >= 0x00C0 and cp <= 0x024F:
        return True
    if cp >= 0x0370 and cp <= 0x04FF:
        return True
    if cp >= 0x4E00 and cp <= 0x9FFF:
        return True
    return False


def _is_digit(cp: Int) -> Bool:
    return cp >= 0x30 and cp <= 0x39


def _is_whitespace(cp: Int) -> Bool:
    return cp == 0x20 or cp == 0x09 or cp == 0x0A or cp == 0x0D


def simple_gpt2_split(text: String) -> List[String]:
    """Simplified GPT-2-style pre-tokenization split."""
    var tokens = List[String]()
    var current = String()
    var current_kind = -1  # -1 none, 0 letter, 1 digit, 2 punct, 3 space

    for cp in text.codepoints():
        var c = chr(Int(cp))
        var kind: Int = 2
        if _is_whitespace(Int(cp)):
            kind = 3
        elif _is_letter(Int(cp)):
            kind = 0
        elif _is_digit(Int(cp)):
            kind = 1

        if kind == current_kind:
            current += c
        elif current_kind == 3 and kind != 3:
            # A single leading space attaches to the following token.
            if current.byte_length() == 1:
                current += c
                current_kind = kind
            else:
                tokens.append(current^)
                current = String(c)
                current_kind = kind
        else:
            if current.byte_length() > 0:
                tokens.append(current^)
            current = String(c)
            current_kind = kind

    if current.byte_length() > 0:
        tokens.append(current^)
    return tokens^


# ---------------------------------------------------------------------------
# ByteLevelPreTokenizer
# ---------------------------------------------------------------------------


struct ByteLevelPreTokenizer(PreTokenizer):
    """GPT-2-style byte-level pre-tokenizer.

    Splits input text using a simplified regex, then maps each byte to
    a printable Unicode character (the GPT-2 byte<->unicode mapping).

    Attributes:
        add_prefix_space: if True and text does not start with a space,
            prepend a space before splitting (matching GPT-2 behaviour).
        use_regex: if True, apply the simplified GPT-2 split; otherwise
            treat the whole input as one token.
        byte_mapping: cached byte mapping to avoid rebuilding on every call.
    """

    var add_prefix_space: Bool
    var use_regex: Bool
    var byte_mapping: ByteMapping

    def __init__(out self, add_prefix_space: Bool = True, use_regex: Bool = True):
        self.add_prefix_space = add_prefix_space
        self.use_regex = use_regex
        self.byte_mapping = ByteMapping()

    def pre_tokenize(self, text: String) raises -> List[String]:
        """Split and byte-map `text` into a list of pre-tokenized strings."""
        var work = String(text)
        if self.add_prefix_space and not work.startswith(" "):
            work = " " + work

        var splits = List[String]()
        if self.use_regex:
            splits = simple_gpt2_split(work^)
        else:
            splits.append(work^)

        # Use cached byte mapping
        var result = List[String]()
        for token in splits:
            var mapped = String()
            for b in token.bytes():
                mapped += self.byte_mapping.b2u[Int(b)]
            result.append(mapped^)
        return result^

    def pre_tokenize_str(self, text: String) raises -> String:
        """Byte-map the whole text without splitting (helper for tests)."""
        var mapped = String()
        for b in text.bytes():
            mapped += self.byte_mapping.b2u[Int(b)]
        return mapped


# ---------------------------------------------------------------------------
# ByteLevelDecoder
# ---------------------------------------------------------------------------


struct ByteLevelDecoder(Decoder):
    """Decoder that reverses ByteLevel pre-tokenization.

    Converts each byte-level Unicode character back to its original byte,
    then reassembles the original UTF-8 string.
    
    Attributes:
        byte_mapping: cached byte mapping to avoid rebuilding on every call.
    """

    var byte_mapping: ByteMapping

    def __init__(out self):
        self.byte_mapping = ByteMapping()

    def decode(self, tokens: List[String]) raises -> String:
        """Decode a list of byte-mapped tokens back to the original string."""
        var byte_buf = List[Int]()
        for token in tokens:
            for cp in token.codepoints():
                var c = chr(Int(cp))
                var opt_b = self.byte_mapping.u2b.get(c)
                if opt_b:
                    byte_buf.append(opt_b.value())
                else:
                    # Unknown char: use its UTF-8 bytes directly.
                    for raw in c.bytes():
                        byte_buf.append(Int(raw))
        return _bytes_to_string(byte_buf)

    def decode_string(self, s: String) raises -> String:
        """Decode a single byte-mapped string (convenience for tests)."""
        var lst = List[String]()
        lst.append(s)
        return self.decode(lst)


def _bytes_to_string(bytes: List[Int]) -> String:
    """Assemble a UTF-8 string from a list of byte values."""
    var result = String()
    var i = 0
    var n = len(bytes)
    while i < n:
        var b0 = bytes[i]
        if b0 < 0x80:
            result += chr(b0)
            i += 1
        elif b0 < 0xE0 and i + 1 < n:
            result += chr(((b0 & 0x1F) << 6) | (bytes[i + 1] & 0x3F))
            i += 2
        elif b0 < 0xF0 and i + 2 < n:
            result += chr(
                ((b0 & 0x0F) << 12)
                | ((bytes[i + 1] & 0x3F) << 6)
                | (bytes[i + 2] & 0x3F)
            )
            i += 3
        elif i + 3 < n:
            result += chr(
                ((b0 & 0x07) << 18)
                | ((bytes[i + 1] & 0x3F) << 12)
                | ((bytes[i + 2] & 0x3F) << 6)
                | (bytes[i + 3] & 0x3F)
            )
            i += 4
        else:
            result += chr(b0)
            i += 1
    return result
