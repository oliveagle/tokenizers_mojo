"""Whitespace / Metaspace pre-tokenizers (Phase 2).

Behavior baseline:
  - Whitespace: Rust `tokenizers` pre_tokenizers::Whitespace, regex
    `\\w+|[^\\w\\s]+` (word runs and punctuation/symbol runs; whitespace is
    consumed and removed).  No byte-level mapping.
  - Metaspace: Rust `tokenizers` pre_tokenizers::Metaspace.  Replaces
    every literal space (0x20) with a replacement char (default U+2581
    '▁'), prepends one replacement when the text does not already start
    with it (PrependScheme::Always), then splits on the replacement with
    MergedWithNext semantics.  Tabs/newlines are NOT replaced (only 0x20).
"""


# ---------------------------------------------------------------------------
# Character predicates (subset of regex `\w` and `\s`)
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


def _is_word_char(cp: Int) -> Bool:
    """Regex `\\w`: letters, digits, underscore (ASCII + common ranges)."""
    if _is_letter(cp) or _is_digit(cp):
        return True
    return cp == 0x5F  # underscore


def _is_whitespace(cp: Int) -> Bool:
    return cp == 0x20 or cp == 0x09 or cp == 0x0A or cp == 0x0D


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------


struct Whitespace:
    """Split on word runs (`\\w+`) and punctuation runs (`[^\\w\\s]+`).

    Whitespace runs are removed (SplitDelimiterBehavior::Removed).  This
    matches the upstream `Whitespace` pre-tokenizer used by BERT-style
    tokenizers (minus the separate BertPreTokenizer rules).
    """

    def __init__(out self):
        pass

    def pre_tokenize(self, text: String) -> List[String]:
        """Return the list of split tokens (whitespace removed)."""
        var tokens = List[String]()
        var current = String()
        var current_kind = -1  # 0 word, 1 punct, -1 none/whitespace

        for cp in text.codepoints():
            var c = chr(Int(cp))
            var kind: Int = -1
            if _is_word_char(Int(cp)):
                kind = 0
            elif not _is_whitespace(Int(cp)):
                kind = 1

            if kind == -1:
                # whitespace: flush any pending run
                if current.byte_length() > 0:
                    tokens.append(current^)
                    current = String()
                    current_kind = -1
            elif kind == current_kind:
                current += c
            else:
                if current.byte_length() > 0:
                    tokens.append(current^)
                current = String(c)
                current_kind = kind

        if current.byte_length() > 0:
            tokens.append(current^)
        return tokens^


# ---------------------------------------------------------------------------
# Metaspace
# ---------------------------------------------------------------------------


struct Metaspace:
    """Replace spaces with a meta char and split (MergedWithNext).

    Defaults match upstream: replacement '▁', prepend Always, split true.
    """

    var replacement: String
    var prepend_always: Bool
    var split: Bool

    def __init__(out self):
        self.replacement = "▁"
        self.prepend_always = True
        self.split = True

    def __init__(
        out self, replacement: String, prepend_always: Bool, split: Bool
    ):
        self.replacement = replacement
        self.prepend_always = prepend_always
        self.split = split

    def pre_tokenize(self, text: String) -> List[String]:
        """Return Metaspace tokens (▁ for spaces, MergedWithNext split)."""
        var tokens = List[String]()
        var n = text.byte_length()
        if n == 0:
            if self.prepend_always:
                var r0 = List[String]()
                r0.append(self.replacement)
                return r0^
            return tokens^

        var i = 0
        var has_leading_space = text[byte=0] == " "

        # Step 1: handle the first word (possibly with a prepended meta).
        # When the text starts with a space, there is no first word -- the
        # leading space run is handled in the main loop below.
        if not has_leading_space:
            var m = i
            while m < n and not (text[byte=m] == " "):
                m += 1
            if self.prepend_always:
                # virtual prepended meta merges into the first word
                tokens.append(self.replacement + String(text[byte=i:m]))
            else:
                tokens.append(String(text[byte=i:m]))
            i = m

        # Step 2: process space runs + following words.
        while i < n:
            if text[byte=i] == " ":
                var j = i
                while j < n and text[byte=j] == " ":
                    j += 1
                var k = j - i
                if j < n:
                    # space run followed by a word: last space merges with
                    # the word, earlier spaces become standalone meta chars
                    var cnt = k - 1
                    while cnt > 0:
                        tokens.append(self.replacement)
                        cnt -= 1
                    var m2 = j
                    while m2 < n and not (text[byte=m2] == " "):
                        m2 += 1
                    tokens.append(self.replacement + text[byte=j:m2])
                    i = m2
                else:
                    # trailing space run: every space becomes a meta char
                    var cnt2 = k
                    while cnt2 > 0:
                        tokens.append(self.replacement)
                        cnt2 -= 1
                    i = j
            else:
                # non-space run (only possible when not prepended and text
                # starts with a non-space word handled above, or with
                # tabs/newlines mixed in)
                var m3 = i
                while m3 < n and not (text[byte=m3] == " "):
                    m3 += 1
                tokens.append(String(text[byte=i:m3]))
                i = m3

        return tokens^


# ---------------------------------------------------------------------------
# Convenience: whitespace_split (string -> List[String])
# ---------------------------------------------------------------------------


def whitespace_split(text: String) -> List[String]:
    """Standalone Whitespace pre-tokenization helper."""
    var ws = Whitespace()
    return ws.pre_tokenize(text)
