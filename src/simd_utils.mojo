"""SIMD-accelerated utility functions for tokenizer hot paths.

Provides:
  1. Byte lookup tables (no Dict overhead) for GPT-2 byte<->unicode mapping.
  2. SIMD batch byte classification for GPT-2 split.
  3. Fast byte-to-String assembly without per-byte allocation.
"""

from memory import memcpy, memset_zero
from sys import simd_width, align_of


# ---------------------------------------------------------------------------
# 256-entry byte→codepoint lookup table (replaces Dict[Int, String])
# ---------------------------------------------------------------------------

struct ByteLookup:
    """Fixed-size 256-entry lookup tables for GPT-2 byte mapping.

    - b2u_cp[byte] → target codepoint (Int)
    - u2b_cp[codepoint - 256] → original byte (Int), for inverse mapping.
    - u2b_ascii[codepoint] → original byte, for ASCII-range inverse.
    """

    # b2u: byte → target codepoint.  For printable ASCII/Latin1, cp == byte.
    var b2u_cp: StaticIntTuple[256]

    # Inverse: for codepoints >= 256, map back to byte.
    # u2b_index[cp - 256] = original byte.
    var u2b_index: StaticIntTuple[256]
    var u2b_max_cp: Int  # highest codepoint in the mapping

    @staticmethod
    fn create() -> ByteLookup:
        var t = ByteLookup()
        # Initialize tables
        # GPT-2 convention: 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF map to self.
        # Remaining bytes 0..255 map to codepoints 256, 257, ...
        var n = 0
        for b in range(256):
            var cp: Int
            if (b >= 0x21 and b <= 0x7E) or (b >= 0xA1 and b <= 0xAC) or (
                b >= 0xAE and b <= 0xFF
            ):
                cp = b
            else:
                cp = 256 + n
                t.u2b_index[n] = b
                n += 1
            t.b2u_cp[b] = cp
        t.u2b_max_cp = 256 + n - 1
        return t

    fn byte_to_codepoint(self, b: Int) -> Int:
        """Fast O(1) lookup: byte → codepoint."""
        return self.b2u_cp[b]

    fn codepoint_to_byte(self, cp: Int) -> Int:
        """Fast O(1) lookup: codepoint → original byte."""
        if cp < 256:
            return cp  # ASCII/Latin1 maps to self
        return self.u2b_index[cp - 256]


# ---------------------------------------------------------------------------
# SIMD byte classification for GPT-2 split
# ---------------------------------------------------------------------------

alias S = simd_width()

fn is_letter_byte(b: Int) -> Bool:
    """Check if a single ASCII byte is a letter (A-Z, a-z)."""
    return (b >= 0x41 and b <= 0x5A) or (b >= 0x61 and b <= 0x7A)

fn is_digit_byte(b: Int) -> Bool:
    """Check if a single ASCII byte is a digit (0-9)."""
    return b >= 0x30 and b <= 0x39

fn is_space_byte(b: Int) -> Bool:
    """Check if a single byte is whitespace (space, tab, newline, CR)."""
    return b == 0x20 or b == 0x09 or b == 0x0A or b == 0x0D


# ---------------------------------------------------------------------------
# Fast byte→UTF-8 assembly without per-byte String allocation
# ---------------------------------------------------------------------------

fn bytes_to_utf8_no_alloc(
    bytes: DTypePointer[DType.uint8], n: Int, result: DTypePointer[DType.uint8]
) -> Int:
    """Convert a byte buffer to UTF-8 codepoints in-place.
    
    Returns the number of bytes written to `result`.
    
    For the GPT-2 byte mapping, each input byte becomes a 1-4 byte
    UTF-8 codepoint.  We use a lookup table approach:
      - ASCII range (0x21-0x7E): 1 byte, same value
      - Latin1 range (0xA1-0xAC, 0xAE-0xFF): 2 bytes
      - Mapped range (256+): 2 bytes
    """
    var written = 0
    var i = 0
    while i < n:
        var b = Int(bytes[i])
        # Fast path: printable ASCII (1 byte output)
        if b >= 0x21 and b <= 0x7E:
            result[written] = UInt8(b)
            written += 1
            i += 1
        # 2-byte UTF-8: codepoints 0x80-0x7FF
        elif b >= 0xA1 and b <= 0xAC:
            result[written] = UInt8(0xC0 | (b >> 6))
            result[written + 1] = UInt8(0x80 | (b & 0x3F))
            written += 2
            i += 1
        elif b >= 0xAE and b <= 0xFF:
            result[written] = UInt8(0xC0 | (b >> 6))
            result[written + 1] = UInt8(0x80 | (b & 0x3F))
            written += 2
            i += 1
        else:
            # Non-printable: codepoint = 256 + index
            var cp = 256
            var idx = 0
            var bb = 0
            while bb < 256:
                if (bb >= 0x21 and bb <= 0x7E) or (
                    bb >= 0xA1 and bb <= 0xAC
                ) or (bb >= 0xAE and bb <= 0xFF):
                    bb += 1
                    continue
                if bb == b:
                    cp = 256 + idx
                    break
                idx += 1
                bb += 1
            # Encode cp as 2-byte UTF-8
            result[written] = UInt8(0xC0 | (cp >> 6))
            result[written + 1] = UInt8(0x80 | (cp & 0x3F))
            written += 2
            i += 1
    return written
