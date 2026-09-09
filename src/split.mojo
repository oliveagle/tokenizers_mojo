"""Split pre-tokenizer (Phase 2).

Behavior baseline: HuggingFace tokenizers `pre_tokenizers::Split` with a
literal String pattern (Regex patterns are Phase 3 TODO).

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

        if sep_len == 0 or n == 0:
            tokens.append(text)
            return tokens^

        # locate every separator occurrence: [start, end) byte ranges
        var seps = List[Tuple[Int, Int]]()
        var i = 0
        while i <= n - sep_len:
            if String(text[byte = i : i + sep_len]) == sep:
                seps.append(Tuple[Int, Int](i, i + sep_len))
                i += sep_len
            else:
                i += 1

        if len(seps) == 0:
            tokens.append(text)
            return tokens^

        if self.invert:
            for s in seps:
                tokens.append(String(text[byte = s[0] : s[1]]))
            return tokens^

        if self.behavior == "removed" or self.behavior == "isolated":
            var prev_end = 0
            for s in seps:
                if s[0] > prev_end:
                    tokens.append(String(text[byte = prev_end : s[0]]))
                if self.behavior == "isolated":
                    tokens.append(sep)
                prev_end = s[1]
            if prev_end < n:
                tokens.append(String(text[byte=prev_end:n]))
            return tokens^

        if self.behavior == "merged_with_previous":
            var prev_end = 0
            for s in seps:
                var piece = String(text[byte = prev_end : s[0]])
                if piece.byte_length() > 0:
                    tokens.append(piece + sep)
                else:
                    tokens.append(sep)
                prev_end = s[1]
            if prev_end < n:
                tokens.append(String(text[byte=prev_end:n]))
            return tokens^

        # merged_with_next
        var first = String(text[byte = 0 : seps[0][0]])
        if first.byte_length() > 0:
            tokens.append(first)
        for idx in range(len(seps)):
            var s = seps[idx]
            var next_start = s[1]
            var next_end = n
            if idx + 1 < len(seps):
                next_end = seps[idx + 1][0]
            var next_piece = String(text[byte=next_start:next_end])
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
