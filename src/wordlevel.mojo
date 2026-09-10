"""WordLevel Model — whole-word vocabulary lookup.

Behavior baseline: HuggingFace tokenizers `models/wordlevel/mod.rs` (upstream Rust).

WordLevel is the simplest tokenization model: each word is looked up directly
in the vocabulary. If the word is not found, the unk_token is returned.
This is used by models that expect pre-tokenized input (e.g., ALBERT with
whitespace tokenization).
"""

import traits

from traits import Model


struct WordLevel(Model):
    """WordLevel model for whole-word tokenization.

    Attributes:
        vocab: token -> id mapping.
        id_to_token: id -> token mapping (inverse of vocab).
        unk_token: the unknown token (default: "<unk>").
    """

    var vocab: Dict[String, Int]
    var id_to_token: Dict[Int, String]
    var unk_token: String

    def __init__(out self):
        self.vocab = Dict[String, Int]()
        self.id_to_token = Dict[Int, String]()
        self.unk_token = String("<unk>")

    def token_for_id(self, id: Int) raises -> String:
        var t = self.id_to_token.get(id)
        if t:
            return t.value()
        return String("")

    def token_id(self, token: String) -> Int:
        """Return the vocab id for `token`, or -1 if not in vocab."""
        var id = self.vocab.get(token)
        if id:
            return id.value()
        return -1

    def load_vocab_from_dict(mut self, entries: Dict[String, Int]) raises:
        """Populate vocab from a token -> id dictionary."""
        self.vocab = entries.copy()
        # Rebuild id_to_token
        self.id_to_token = Dict[Int, String]()
        for token in entries.keys():
            var id = entries[token]
            if id:
                self.id_to_token[id.value()] = token

    def vocab_size(self) -> Int:
        """Return the vocabulary size."""
        return len(self.vocab)

    def tokenize_word(self, word: String) raises -> String:
        """Tokenize a word by looking it up directly in the vocabulary.

        Returns the token if found, or the unk_token if not found.
        """
        if self.vocab.get(word):
            return word
        elif self.vocab.get(self.unk_token):
            return self.unk_token
        else:
            raise Error(
                "WordLevel.tokenize: unk_token '" + self.unk_token + "' not in vocab"
            )

    def encode(self, word: String) raises -> List[Int]:
        """Encode a word into a list of vocab ids (single id or UNK)."""
        var ids = List[Int]()

        var id = self.vocab.get(word)
        if id:
            ids.append(id.value())
        elif self.vocab.get(self.unk_token):
            ids.append(self.vocab[self.unk_token])
        else:
            raise Error(
                "WordLevel.encode: unk_token '" + self.unk_token + "' not in vocab"
            )

        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Decode a list of ids back to a string."""
        var s = String()
        for i in ids:
            var t = self.id_to_token.get(i)
            if t:
                s += t.value()
        return s
