"""Optimized GPT-2 pre-tokenization split.

Faithfully mirrors V1's simple_gpt2_split logic exactly, including
the full Unicode letter/digit/whitespace classification from V1.
"""


def _is_letter(cp: Int) -> Bool:
    """Matches V1's _is_letter exactly."""
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
    """Matches V1's _is_digit exactly."""
    return cp >= 0x30 and cp <= 0x39


def _is_whitespace(cp: Int) -> Bool:
    """Matches V1's _is_whitespace exactly."""
    return cp == 0x20 or cp == 0x09 or cp == 0x0A or cp == 0x0D


def _codepoint_class(cp: Int) -> Int:
    """Return character class for a codepoint: 0=letter, 1=digit, 2=other, 3=space."""
    if _is_whitespace(cp):
        return 3
    if _is_letter(cp):
        return 0
    if _is_digit(cp):
        return 1
    return 2


def simple_gpt2_split_opt(text: String) -> List[String]:
    """Optimized GPT-2 pre-tokenization split.
    
    Mirrors V1's simple_gpt2_split logic exactly, using codepoint-level
    classification (not byte-level) to correctly handle multi-byte UTF-8.
    """
    var tokens = List[String]()
    var current = String()
    var current_kind = -1

    for cp in text.codepoints():
        var c = chr(Int(cp))
        var kind = _codepoint_class(Int(cp))

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
