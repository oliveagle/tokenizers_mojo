"""Trait/interface abstraction for tokenizer components (Phase 2).

Defines the four core extension-point interfaces used by `Tokenizer`:

    Normalizer   - text -> normalized text
    PreTokenizer - text -> pre-tokenized pieces
    Model        - (byte-mapped) word -> token ids
    Decoder      - tokens -> text

Concrete implementations opt in with `struct X(Trait): ...`, enabling
generic pipeline helpers and (in Phase 3) runtime-swappable components.

NOTE (Mojo 1.0.0): trait methods must be declared as
`def m(self: Self, ...) raises -> T: ...` (body `...`); `fn` is removed.
"""


trait Normalizer:
    """Normalize raw text before tokenization."""

    def normalize(self: Self, text: String) raises -> String:
        ...


trait PreTokenizer:
    """Split normalized text into pre-tokens (before the model)."""

    def pre_tokenize(self: Self, text: String) raises -> List[String]:
        ...


trait Model:
    """Map a pre-token (e.g. byte-mapped word) to token ids."""

    def encode(self: Self, word: String) raises -> List[Int]:
        ...


trait Decoder:
    """Reassemble token strings back into text."""

    def decode(self: Self, tokens: List[String]) raises -> String:
        ...
