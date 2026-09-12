"""Split pre-tokenizer (Phase 2).

Behavior baseline: HuggingFace tokenizers `pre_tokenizers::Split`.
Supports both literal separators (`SplitPreTokenizer`) and regex
patterns (`RegexSplitPreTokenizer`).

Behaviors (SplitDelimiterBehavior):
  removed             - drop delimiters, keep non-empty pieces
  isolated            - delimiters become their own tokens
  merged_with_previous- each delimiter joins the preceding piece; a
                        delimiter with an empty preceding piece stays alone
  merged_with_next    - each delimiter joins the following piece; a
                        delimiter with an empty following piece stays alone
"""

import traits

from traits import PreTokenizer
from regex import regex_find_all

def _read_text_bytes(text: String) -> List[Int]:
    """Read all bytes from text into a List for safe byte-level access."""
    var result = List[Int]()
    result.reserve(text.byte_length())
    for b in text.bytes():
        result.append(Int(b))
    return result^



struct SplitPreTokenizer(PreTokenizer):
    """Split on a literal separator with a configurable behavior.

    Attributes:
        separator: literal string to split on.
        behavior: "removed" | "isolated" | "merged_with_previous" |
            "merged_with_next".
        invert: If True, keep only the separators and drop the pieces.
    """

    var separator: String
    var behavior: String
    var invert: Bool

    def __init__(out self, separator: String, behavior: String = "removed"):
        self.separator = separator
        self.behavior = behavior
        self.invert = False

    def pre_tokenize(self, text: String) raises -> List[String]:
        """Return the split tokens per the configured behavior."""
        var tokens = List[String]()
        var n = text.byte_length()
        var sep = self.separator
        var sep_len = sep.byte_length()
        
        # Pre-read bytes for safe byte-level access
        var text_bytes = List[Int]()
        text_bytes.reserve(n)
        for b in text.bytes():
            text_bytes.append(Int(b))

        if sep_len == 0 or n == 0:
            tokens.append(text)
            return tokens^

        # Pre-read separator bytes
        var sep_bytes = List[Int]()
        for b in sep.bytes():
            sep_bytes.append(Int(b))

        # locate every separator occurrence: [start, end) byte ranges
        var seps = List[Tuple[Int, Int]]()
        var i = 0
        while i <= n - sep_len:
            # Compare bytes directly
            var is_match = True
            for j in range(sep_len):
                if text_bytes[i + j] != sep_bytes[j]:
                    is_match = False
                    break
            if is_match:
                seps.append(Tuple[Int, Int](i, i + sep_len))
                i += sep_len
            else:
                i += 1

        if len(seps) == 0:
            tokens.append(text)
            return tokens^

        if self.invert:
            for s in seps:
                var seg = String()
                var si = s[0]
                while si < s[1]:
                    seg += chr(text_bytes[si])
                    si += 1
                tokens.append(seg)
            return tokens^

        if self.behavior == "removed" or self.behavior == "isolated":
            var prev_end = 0
            for s in seps:
                if s[0] > prev_end:
                    var seg = String()
                    var sj = prev_end
                    while sj < s[0]:
                        seg += chr(text_bytes[sj])
                        sj += 1
                    tokens.append(seg)
                if self.behavior == "isolated":
                    tokens.append(sep)
                prev_end = s[1]
            if prev_end < n:
                var seg = String()
                var sj = prev_end
                while sj < n:
                    seg += chr(text_bytes[sj])
                    sj += 1
                tokens.append(seg)
            return tokens^

        if self.behavior == "merged_with_previous":
            var prev_end = 0
            for s in seps:
                var piece = String()
                var pi = prev_end
                while pi < s[0]:
                    piece += chr(text_bytes[pi])
                    pi += 1
                if piece.byte_length() > 0:
                    tokens.append(piece + sep)
                else:
                    tokens.append(sep)
                prev_end = s[1]
            if prev_end < n:
                var seg = String()
                var sj = prev_end
                while sj < n:
                    seg += chr(text_bytes[sj])
                    sj += 1
                tokens.append(seg)
            return tokens^

        # merged_with_next
        var first = String()
        var fi = 0
        while fi < seps[0][0]:
            first += chr(text_bytes[fi])
            fi += 1
        if first.byte_length() > 0:
            tokens.append(first)
        for idx in range(len(seps)):
            var s = seps[idx]
            var next_start = s[1]
            var next_end = n
            if idx + 1 < len(seps):
                next_end = seps[idx + 1][0]
            var next_piece = String()
            var ni = next_start
            while ni < next_end:
                next_piece += chr(text_bytes[ni])
                ni += 1
            if next_piece.byte_length() > 0:
                tokens.append(sep + next_piece)
            else:
                tokens.append(sep)
        return tokens^


def split_on(
    text: String, separator: String, behavior: String = "removed"
) raises -> List[String]:
    """Standalone Split pre-tokenization helper."""
    var sp = SplitPreTokenizer(separator, behavior)
    return sp.pre_tokenize(text)


struct RegexSplitPreTokenizer(PreTokenizer):
    """Split on a regex pattern with a configurable behavior.

    Same 4 SplitDelimiterBehavior semantics as SplitPreTokenizer, but
    the delimiter is a regex pattern (Pattern trait's "pattern" string).
    """

    var pattern: String
    var behavior: String
    var invert: Bool

    def __init__(out self, pattern: String, behavior: String = "removed"):
        self.pattern = pattern
        self.behavior = behavior
        self.invert = False

    def pre_tokenize(self, text: String) raises -> List[String]:
        """Split `text` on regex matches per the configured behavior."""
        var tokens = List[String]()
        var n = text.byte_length()
        if n == 0:
            tokens.append(text)
            return tokens^

        # locate every regex match: [start, end) byte ranges
        var matches = regex_find_all(self.pattern, text)

        if len(matches) == 0:
            tokens.append(text)
            return tokens^

        if self.invert:
            for m in matches:
                var seg = String()
        var mi = m[0]
        while mi < m[1]:
            seg += chr(text_bytes[mi])
            mi += 1
        tokens.append(seg)
            return tokens^

        if self.behavior == "removed" or self.behavior == "isolated":
            var prev_end = 0
            for m in matches:
                if m[0] > prev_end:
                    var seg = String()
                var pi = prev_end
                while pi < m[0]:
                    seg += chr(text_bytes[pi])
                    pi += 1
                tokens.append(seg)
                if self.behavior == "isolated":
                    var seg = String()
        var mi = m[0]
        while mi < m[1]:
            seg += chr(text_bytes[mi])
            mi += 1
        tokens.append(seg)
                prev_end = m[1]
            if prev_end < n:
                var seg = String()
            var sj = prev_end
            while sj < n:
                seg += chr(text_bytes[sj])
                sj += 1
            tokens.append(seg)
            return tokens^

        if self.behavior == "merged_with_previous":
            var prev_end = 0
            for m in matches:
                var piece = String()
                var pi = prev_end
                while pi < m[0]:
                    piece += chr(text_bytes[pi])
                    pi += 1
                var delim = String()
                var di = m[0]
                while di < m[1]:
                    delim += chr(text_bytes[di])
                    di += 1
                if piece.byte_length() > 0:
                    tokens.append(piece + delim)
                else:
                    tokens.append(delim)
                prev_end = m[1]
            if prev_end < n:
                var seg = String()
            var sj = prev_end
            while sj < n:
                seg += chr(text_bytes[sj])
                sj += 1
            tokens.append(seg)
            return tokens^

        # merged_with_next
        var first = String()
        var fi = 0
        while fi < matches[0][0]:
            first += chr(text_bytes[fi])
            fi += 1
        if first.byte_length() > 0:
            tokens.append(first)
        for idx in range(len(matches)):
            var m = matches[idx]
            var next_start = m[1]
            var next_end = n
            if idx + 1 < len(matches):
                next_end = matches[idx + 1][0]
            var next_piece = String()
            var ni = next_start
            while ni < next_end:
                next_piece += chr(text_bytes[ni])
                ni += 1
            var delim = String()
                var di = m[0]
                while di < m[1]:
                    delim += chr(text_bytes[di])
                    di += 1
            if next_piece.byte_length() > 0:
                tokens.append(delim + next_piece)
            else:
                tokens.append(delim)
        return tokens^


def regex_split_on(
    text: String, pattern: String, behavior: String = "removed"
) raises -> List[String]:
    """Standalone regex-Split pre-tokenization helper."""
    var sp = RegexSplitPreTokenizer(pattern, behavior)
    return sp.pre_tokenize(text)
