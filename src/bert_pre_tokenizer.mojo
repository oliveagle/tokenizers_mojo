"""BERT pre-tokenizer (Phase 2).

Behavior baseline: HuggingFace tokenizers `pre_tokenizers::BertPreTokenizer`.

Two-pass split:
  1. Whitespace split (SplitDelimiterBehavior::Removed)
  2. Punctuation split (SplitDelimiterBehavior::Isolated) — each
     punctuation char becomes its own token.

Punctuation = ASCII punctuation set + common Unicode P-category ranges
(General Punctuation U+2000..U+206F, CJK Symbols U+3000..U+303F,
Fullwidth Forms U+FF00..U+FFEF, and CJK dash/ellipsis …).  CJK ideographs
(U+4E00..U+9FFF) are NOT punctuation (they form normal words).

NOTE: upstream BERT tokenizers use `BertNormalizer` with clean_text=True
to pre-insert spaces around CJK ideographs; that normalization is a
separate concern and not handled here.
"""

import whitespace
import traits

from traits import PreTokenizer
from whitespace import Whitespace


def _is_ascii_punct(cp: Int) -> Bool:
    """ASCII punctuation set (33 chars: ! through ~ excluding alnum)."""
    if cp >= 0x21 and cp <= 0x2F:
        return True
    if cp >= 0x3A and cp <= 0x40:
        return True
    if cp >= 0x5B and cp <= 0x60:
        return True
    if cp >= 0x7B and cp <= 0x7E:
        return True
    return False


def _is_unicode_punct(cp: Int) -> Bool:
    """Common Unicode punctuation categories (subset of P)."""
    # General Punctuation (includes '…' U+2026, '—' U+2014, etc.)
    if cp >= 0x2000 and cp <= 0x206F:
        return True
    # CJK Symbols and Punctuation (。、「」…)
    if cp >= 0x3000 and cp <= 0x303F:
        return True
    # Fullwidth ASCII punctuation + Halfwidth/Fullwidth Forms
    if cp >= 0xFF00 and cp <= 0xFFEF:
        return True
    # General Punctuation: — … ‘ ’ “ ” • (redundant with range above)
    # Latin-1 Supplement: ¡ ¿ « »
    if cp in (0xA1, 0xBF, 0xAB, 0xBB):
        return True
    # Hyphens and dashes (subset)
    if cp >= 0x2010 and cp <= 0x2027:
        return True
    return False


def _is_bert_punct(cp: Int) -> Bool:
    return _is_ascii_punct(cp) or _is_unicode_punct(cp)


def _split_punct_isolated(word: String) -> List[String]:
    """Split `word` on punctuation chars, keeping each isolated."""
    var out = List[String]()
    var current = String()
    for cp in word.codepoints():
        var c = chr(Int(cp))
        if _is_bert_punct(Int(cp)):
            if current.byte_length() > 0:
                out.append(current^)
                current = String()
            out.append(c)
        else:
            current += c
    if current.byte_length() > 0:
        out.append(current^)
    return out^


struct BertPreTokenizer(PreTokenizer):
    """BERT-style pre-tokenizer: whitespace-removed then punct-isolated.

    Example (matches upstream test):
      "Hey friend!     How are you?!?"
        -> ["Hey", "friend", "!", "How", "are", "you", "?", "!", "?"]
    """

    def __init__(out self):
        pass

    def pre_tokenize(self, text: String) raises -> List[String]:
        # Pass 1: whitespace split (removed)
        var ws = Whitespace()
        var words = ws.pre_tokenize(text)
        # Pass 2: punctuation split (isolated)
        var out = List[String]()
        for word in words:
            var pieces = _split_punct_isolated(word)
            for piece in pieces:
                out.append(piece)
        return out^


def bert_pre_tokenize(text: String) raises -> List[String]:
    """Standalone BertPreTokenizer helper."""
    var pt = BertPreTokenizer()
    return pt.pre_tokenize(text)
