"""ByteLevel PreTokenizer and Decoder — V2 with cached table lookup.

Key optimizations over V1:
  1. ByteMapping uses two arrays:
     - b2u[strings]: pre-built String for each byte (Dict replacement)
     - b2u_cp: codepoint int for each byte (for decoder)
  2. simple_gpt2_split uses run-length character class detection.
  3. Decoder uses integer lookup + direct byte output.
"""

import traits
from traits import PreTokenizer, Decoder


struct ByteTables:
    """256-entry byte→mapped-String tables.
    
    Pre-builds all 256 mapped Strings at init time.  Lookup is
    direct array indexing — no Dict hash, no chr() allocation.
    """

    # Pre-built mapped strings: b2u_str[b] = GPT-2 mapped char for byte b
    var b2u_str: List[String]
    # Integer codepoints for decoder
    var b2u_cp: List[Int]
    var u2b: List[Int]
    var u2b_max_idx: Int

    def __init__(out self):
        self.b2u_str = List[String]()
        self.b2u_cp = List[Int]()
        self.u2b = List[Int]()
        self.u2b_max_idx = 0
        self._build()

    def _build(mut self):
        # Pre-allocate 256 entries
        for _ in range(256):
            self.b2u_str.append(String())
            self.b2u_cp.append(0)
            self.u2b.append(0)

        var n = 0
        for b in range(256):
            var cp: Int
            if (
                (b >= 0x21 and b <= 0x7E)
                or (b >= 0xA1 and b <= 0xAC)
                or (b >= 0xAE and b <= 0xFF)
            ):
                cp = b
            else:
                cp = 256 + n
                self.u2b[n] = b
                n += 1
            self.b2u_cp[b] = cp
            self.b2u_str[b] = chr(cp)
        self.u2b_max_idx = n

def _read_text_bytes(text: String) -> List[Int]:
    """Read all bytes from text into a List for safe byte-level access."""
    var result = List[Int]()
    result.reserve(text.byte_length())
    for b in text.bytes():
        result.append(Int(b))
    return result^



# ---------------------------------------------------------------------------
# GPT-2 pre-tokenization split
# ---------------------------------------------------------------------------

def _byte_at(text_bytes: List[Int], pos: Int) -> Int:
    """Read byte at position using pre-read byte list for safety."""
    if pos < len(text_bytes):
        return text_bytes[pos]
    return -1


def _byte_class(b: Int) -> Int:
    if b == 0x20 or b == 0x09 or b == 0x0A or b == 0x0D:
        return 3
    if (b >= 0x41 and b <= 0x5A) or (b >= 0x61 and b <= 0x7A):
        return 0
    if b >= 0x30 and b <= 0x39:
        return 1
    return 2


def simple_gpt2_split_v2(text: String) -> List[String]:
    """GPT-2 pre-tokenization split with run-length detection."""
    var tokens = List[String]()
    var n = text.byte_length()
    if n == 0:
        return tokens^

    # Pre-read bytes for safe access
    var text_bytes = _read_text_bytes(text)

    var run_start = 0
    var run_class = _byte_class(_byte_at(text_bytes, 0))

    var i = 1
    while i < n:
        var cls = _byte_class(_byte_at(text_bytes, i))

        if run_class == 3 and cls != 3 and (i - run_start) == 1:
            i += 1
            continue

        if cls != run_class:
            var seg = String()
            var j = run_start
            while j < i:
                seg += chr(text_bytes[j])
                j += 1
            tokens.append(seg)
            run_start = i
            run_class = cls
        i += 1

    if run_start < n:
        var seg = String()
        var j = run_start
        while j < n:
            seg += chr(text_bytes[j])
            j += 1
        tokens.append(seg)

    return tokens^


# ---------------------------------------------------------------------------
# ByteLevelPreTokenizer (V2)
# ---------------------------------------------------------------------------


struct ByteLevelPreTokenizer(PreTokenizer):
    """GPT-2-style byte-level pre-tokenizer (V2: cached table lookup)."""

    var add_prefix_space: Bool
    var use_regex: Bool
    var tables: ByteTables

    def __init__(out self, add_prefix_space: Bool = True, use_regex: Bool = True):
        self.add_prefix_space = add_prefix_space
        self.use_regex = use_regex
        self.tables = ByteTables()

    def pre_tokenize(self, text: String) raises -> List[String]:
        var work = String(text)
        if self.add_prefix_space and not work.startswith(" "):
            work = " " + work

        var splits = List[String]()
        if self.use_regex:
            splits = simple_gpt2_split_v2(work^)
        else:
            splits.append(work^)

        var result = List[String]()
        for token in splits:
            var mapped = self._map_bytes(token)
            result.append(mapped^)
        return result^

    def _map_bytes(self, token: String) -> String:
        """Map all bytes to GPT-2 unicode using cached table lookup.
        
        Uses pre-built String values from ByteTables — no chr() or Dict.
        """
        var result = String()
        for b in token.bytes():
            result += self.tables.b2u_str[Int(b)]
        return result

    def pre_tokenize_str(self, text: String) raises -> String:
        return self._map_bytes(text)


# ---------------------------------------------------------------------------
# ByteLevelDecoder (V2)
# ---------------------------------------------------------------------------


struct ByteLevelDecoder(Decoder):
    """Decoder that reverses ByteLevel pre-tokenization (V2: table lookup)."""

    var tables: ByteTables

    def __init__(out self):
        self.tables = ByteTables()

    def decode(self, tokens: List[String]) raises -> String:
        var byte_buf = List[Int]()

        for token in tokens:
            for cp in token.codepoints():
                var cp_val = Int(cp)
                var orig: Int
                if cp_val < 256:
                    orig = cp_val
                elif cp_val - 256 < self.tables.u2b_max_idx:
                    orig = self.tables.u2b[cp_val - 256]
                else:
                    orig = cp_val

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

    def decode_string(self, s: String) raises -> String:
        var lst = List[String]()
        lst.append(s)
        return self.decode(lst)
