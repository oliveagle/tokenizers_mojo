"""AddedToken -- user-added / special tokens (Phase 2).

Mirrors HuggingFace tokenizers `AddedToken` (content + flags).  Added
tokens that are `special` bypass the normalizer/pre-tokenizer/BPE and are
emitted directly as their own token (matched against the raw input).
"""


struct AddedToken:
    """A token added on top of the model vocab.

    Attributes:
        content: The token string (e.g. "<s>", "<|endoftext|>").
        special: True for special tokens that bypass BPE.
        single_word: Only match when delimited by non-word chars.
        lstrip / rstrip: Strip surrounding whitespace on the left/right.
        normalized: Whether the content is itself normalized.
    """

    var content: String
    var special: Bool
    var single_word: Bool
    var lstrip: Bool
    var rstrip: Bool
    var normalized: Bool

    def __init__(out self, content: String):
        self.content = content
        self.special = False
        self.single_word = False
        self.lstrip = False
        self.rstrip = False
        self.normalized = True

    def __init__(
        out self,
        content: String,
        special: Bool,
        single_word: Bool = False,
        lstrip: Bool = False,
        rstrip: Bool = False,
        normalized: Bool = True,
    ):
        self.content = content
        self.special = special
        self.single_word = single_word
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.normalized = normalized

    def __copyinit__(self):
        pass

    def __moveinit__(mut self, mut other: Self):
        self.content = String(other.content)
        self.special = other.special
        self.single_word = other.single_word
        self.lstrip = other.lstrip
        self.rstrip = other.rstrip
        self.normalized = other.normalized
