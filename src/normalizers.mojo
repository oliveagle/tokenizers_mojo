"""Normalizer interface + Phase-1 NFC placeholder.

Phase 1 uses an identity normalization (no-op).  TODO Phase 2: implement
real NFC/NFD/NFKC/NFKD, Lowercase, Strip (matching upstream Rust
normalizers).
"""


struct NFCNormalizer:
    """Identity normalizer (Phase 1 placeholder)."""

    def __init__(out self):
        pass

    def normalize(self, text: String) -> String:
        # TODO Phase 2: real NFC canonical composition.
        return text


def normalize_nfc(text: String) -> String:
    """Free-function NFC (Phase 1: identity)."""
    return text
