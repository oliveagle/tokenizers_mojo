"""Normalizer interface + Phase-1 NFC placeholder + Phase-2 Lowercase/Strip.

Behavior baseline: HuggingFace tokenizers `normalizers/` (upstream Rust).

Phase 2: real NFD/NFKD/NFC/NFKC normalizers (UAX #15) backed by
auto-generated decomposition / composition / CCC tables in
`unicode_data.mojo` (cached once via `UnicodeData`). Lowercase
(leading/trailing whitespace removal via String.strip()).
"""

import traits
import unicode

from traits import Normalizer
from unicode import UnicodeData


def _lower_one(cp: Int) -> Int:
    """Lowercase a single codepoint (full Basic Latin + Latin-1 Supplement
    + Latin Extended + Greek + Cyrillic common range)."""
    # A-Z -> a-z
    if cp >= 0x41 and cp <= 0x5A:
        return cp + 0x20
    # À-Þ (Latin-1 uppercase, skipping × 0xD7)
    if cp >= 0xC0 and cp <= 0xDE and cp != 0xD7:
        return cp + 0x20
    # Greek uppercase Α-Ω (0x391..0x3A9)
    if cp >= 0x391 and cp <= 0x3A9:
        return cp + 0x20
    # Cyrillic uppercase А-Я (0x410..0x42F)
    if cp >= 0x410 and cp <= 0x42F:
        return cp + 0x20
    return cp


def _lower_str(s: String) -> String:
    """Lowercase an entire string codepoint by codepoint."""
    var out = String()
    for cp in s.codepoints():
        out += chr(_lower_one(Int(cp)))
    return out


# ---------------------------------------------------------------------------
# Normalizer structs
# ---------------------------------------------------------------------------


struct NFCNormalizer(Normalizer):
    """Canonical Composition (NFC) per UAX #15.

    Holds a `UnicodeData` so the decomposition / composition / CCC tables
    are built only once per normalizer instance.
    """

    var data: UnicodeData

    def __init__(out self):
        self.data = UnicodeData()

    def normalize(self, text: String) raises -> String:
        return self.data.nfc(text)


def normalize_nfc(text: String) raises -> String:
    """Free-function NFC (rebuilds tables on each call; prefer
    constructing a `UnicodeData` and calling `nfc` for hot paths)."""
    var data = UnicodeData()
    return data.nfc(text)


struct NFDNormalizer(Normalizer):
    """Canonical Decomposition (NFD) per UAX #15."""

    var data: UnicodeData

    def __init__(out self):
        self.data = UnicodeData()

    def normalize(self, text: String) raises -> String:
        return self.data.nfd(text)


struct NFKDNormalizer(Normalizer):
    """Compatibility Decomposition (NFKD) per UAX #15."""

    var data: UnicodeData

    def __init__(out self):
        self.data = UnicodeData()

    def normalize(self, text: String) raises -> String:
        return self.data.nfkd(text)


struct NFKCNormalizer(Normalizer):
    """Compatibility Composition (NFKC) per UAX #15."""

    var data: UnicodeData

    def __init__(out self):
        self.data = UnicodeData()

    def normalize(self, text: String) raises -> String:
        return self.data.nfkc(text)


struct Lowercase(Normalizer):
    """Lowercase every character (upstream: normalizers::Lowercase)."""

    def __init__(out self):
        pass

    def normalize(self, text: String) raises -> String:
        return _lower_str(text)


struct Strip(Normalizer):
    """Strip leading and trailing whitespace (upstream: Strip left+right)."""

    def __init__(out self):
        pass

    def normalize(self, text: String) raises -> String:
        return String(text.strip())


struct LowercaseStrip(Normalizer):
    """Compose Lowercase then Strip (handy for BERT-style pipelines)."""

    def __init__(out self):
        pass

    def normalize(self, text: String) raises -> String:
        return String(_lower_str(text).strip())
