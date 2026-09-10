"""WordPiece Model — vocabulary-based subword tokenization.

Behavior baseline: HuggingFace tokenizers `models/wordpiece/mod.rs` (upstream Rust).

WordPiece is used by BERT, DistilBERT, and other transformer models.
The algorithm performs greedy longest-match-first tokenization:
  1. For each word, try to find the longest suffix that exists in the vocabulary
  2. If found, add it as a token (with "##" prefix if not the first token)
  3. If not found, continue trying shorter suffixes
  4. If no suffix is found, return the [UNK] token for the entire word
"""

import traits

from traits import Model


struct WordPiece(Model):
    """WordPiece model for BERT-style tokenization.

    Attributes:
        vocab: token -> id mapping.
        id_to_token: id -> token mapping (inverse of vocab).
        unk_token: the unknown token (default: "[UNK]").
        continuing_subword_prefix: prefix for continuing subwords (default: "##").
        max_input_chars_per_word: maximum characters per word (default: 100).
    """

    var vocab: Dict[String, Int]
    var id_to_token: Dict[Int, String]
    var unk_token: String
    var continuing_subword_prefix: String
    var max_input_chars_per_word: Int

    def __init__(out self):
        self.vocab = Dict[String, Int]()
        self.id_to_token = Dict[Int, String]()
        self.unk_token = String("[UNK]")
        self.continuing_subword_prefix = String("##")
        self.max_input_chars_per_word = 100

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

    def load_vocab_from_lines(mut self, lines: List[String]):
        """Populate vocab from a list of token strings (id = line index).

        This is the standard format for BERT vocab.txt files.
        """
        for i in range(len(lines)):
            var token = String(lines[i])
            if token.byte_length() > 0:
                self.vocab[token] = i
                self.id_to_token[i] = token

    def load_vocab_from_dict(mut self, entries: Dict[String, Int]):
        """Populate vocab from a token -> id dictionary."""
        self.vocab = entries
        # Rebuild id_to_token
        self.id_to_token = Dict[Int, String]()
        for token in entries.keys():
            var id = entries[token]
            if id:
                self.id_to_token[id.value()] = token

    def vocab_size(self) -> Int:
        """Return the vocabulary size."""
        return len(self.vocab)

    def tokenize_word(self, word: String) -> List[String]:
        """Tokenize a single word using WordPiece algorithm.

        Returns a list of subword tokens. If the word cannot be tokenized,
        returns a list containing just the unk_token.
        """
        var char_len = 0
        for _ in word.codepoints():
            char_len += 1
        if char_len > self.max_input_chars_per_word:
            var result = List[String]()
            result.append(self.unk_token)
            return result^

        var sub_tokens = List[String]()
        var start = 0
        var word_len = word.byte_length()

        while start < word_len:
            var end = word_len
            var found = False

            while start < end:
                var substr: String
                if start == 0:
                    substr = String(word[byte=start:end])
                else:
                    substr = self.continuing_subword_prefix + String(word[byte=start:end])

                if self.vocab.get(substr):
                    sub_tokens.append(substr)
                    found = True
                    break

                # Move end back by one character (handle UTF-8 properly)
                var new_end = end - 1
                # Find the previous character boundary
                while new_end > start and (ord(word[byte=new_end]) & 0xC0) == 0x80:
                    new_end -= 1
                end = new_end

            if not found:
                # Cannot tokenize this word, return [UNK]
                var result = List[String]()
                result.append(self.unk_token)
                return result^

            start = end

        return sub_tokens^

    def tokenize(self, word: String) -> List[(String, Int, Int)]:
        """Tokenize a word and return (token, start, end) tuples.

        This is the Model trait interface method.
        """
        var sub_tokens = self.tokenize_word(word)
        var result = List[(String, Int, Int)]()
        var start = 0

        for token in sub_tokens:
            var token_len = token.byte_length()
            # Adjust for ## prefix
            if token.startswith(self.continuing_subword_prefix):
                var prefix_len = self.continuing_subword_prefix.byte_length()
                result.append((token, start + prefix_len, start + token_len))
            else:
                result.append((token, start, start + token_len))
            start += token_len

        return result^

    def encode(self, word: String) raises -> List[Int]:
        """Encode a word into a list of vocab ids."""
        var tokens = self.tokenize_word(word)
        var ids = List[Int]()

        for token in tokens:
            var id = self.vocab.get(token)
            if id:
                ids.append(id.value())
            else:
                raise Error(
                    "WordPiece.encode: unknown token '" + token + "' (not in vocab)"
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
