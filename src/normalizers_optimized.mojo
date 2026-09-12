"""Normalizer optimizations — skip NFC for ASCII-only text."""

import traits
from traits import Normalizer
from unicode import UnicodeData


def _is_all_ascii(text: String) -> Bool:
    """Check if text contains only ASCII bytes (0-127)."""
    for b in text.bytes():
        if Int(b) > 127:
            return False
    return True


struct NFCNormalizerOptimized(Normalizer):
    """NFC normalizer that skips ASCII-only text (no-op case)."""

    var data: UnicodeData

    def __init__(out self):
        self.data = UnicodeData()

    def normalize(self, text: String) raises -> String:
        # Fast path: ASCII text is already NFC
        if _is_all_ascii(text):
            return text
        return self.data.nfc(text)


struct LowercaseOptimized(Normalizer):
    """Lowercase with ASCII fast path."""

    def __init__(out self):
        pass

    def normalize(self, text: String) raises -> String:
        var result = String()
        for b in text.bytes():
            var bv = Int(b)
            # A-Z -> a-z
            if bv >= 0x41 and bv <= 0x5A:
                result += chr(bv + 0x20)
            else:
                result += chr(bv)
        return result
