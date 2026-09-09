"""BPE Model (GPT-2 style) — vocab/merges loading + greedy merge encoding.

Behavior baseline: HuggingFace tokenizers `models/bpe/mod.rs` (upstream Rust).

Phase 1 scope: greedy lowest-priority-pair merging (no backtracking), which
matches GPT-2 / HF ByteLevel BPE semantics for the minimal pipeline.
"""

import byte_level


struct BPE:
    """Byte-level BPE model.

    Attributes:
        vocab: token -> id mapping (e.g. from GPT-2 vocab.json).
        id_to_token: id -> token mapping (inverse of vocab).
        merges: merge ranks — key is "token1 token2" (space-joined pair),
            value is the merge priority (lower == higher priority).
        bos_token: optional special token id (unused in Phase 1 encode).
        eos_token: optional special token id (unused in Phase 1 encode).
    """

    var vocab: Dict[String, Int]
    var id_to_token: Dict[Int, String]
    var merges: Dict[String, Int]

    def __init__(out self):
        self.vocab = Dict[String, Int]()
        self.id_to_token = Dict[Int, String]()
        self.merges = Dict[String, Int]()

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

    def add_raw_vocab(mut self, token: String, id: Int):
        """Add a raw token/id entry to the vocab (used for special tokens)."""
        self.vocab[token] = id
        self.id_to_token[id] = token

    def load_vocab(mut self, entries: List[String]):
        """Populate vocab from a list of token strings (id = index)."""
        for i in range(len(entries)):
            self.vocab[entries[i]] = i
            self.id_to_token[i] = entries[i]

    def load_vocab_pairs(mut self, keys: List[String], ids: List[Int]) raises:
        """Populate vocab from parallel (token, id) lists."""
        if len(keys) != len(ids):
            raise Error("load_vocab_pairs: length mismatch")
        for i in range(len(keys)):
            self.vocab[keys[i]] = ids[i]
            self.id_to_token[ids[i]] = keys[i]

    def load_merges(mut self, lines: List[String]):
        """Populate merges from "token1 token2" lines (rank = line index).

        Skips empty lines and the standard GPT-2 "#version" header.
        """
        var rank = 0
        for line in lines:
            var l = String(line)
            if l.startswith("#version") or l.byte_length() == 0:
                continue
            self.merges[l] = rank
            rank += 1

    def merge_rank(self, token1: String, token2: String) -> Int:
        """Return merge rank for the pair, or -1 if the pair never merges."""
        return self.merges.get(token1 + " " + token2, -1)

    def encode_word(self, word: String) -> List[String]:
        """Run greedy BPE on a single word, returning BPE token strings.

        `word` is already byte-mapped (output of ByteLevelPreTokenizer).
        Each initial symbol is one unicode character (byte-level mapping).
        Iteratively find the pair with the lowest merge rank and merge it.
        """
        var parts = List[String]()
        for cp in word.codepoints():
            parts.append(chr(Int(cp)))
        if len(parts) == 0:
            return parts^

        # Greedy merge loop (matches GPT-2 `bpe()` algorithm shape).
        var n = len(parts)
        while n > 1:
            var best_rank = -1
            var best_idx = -1
            for i in range(n - 1):
                var pair = parts[i] + " " + parts[i + 1]
                var r = self.merges.get(pair, -1)
                if r != -1 and (best_rank == -1 or r < best_rank):
                    best_rank = r
                    best_idx = i
            if best_idx == -1:
                break
            var a = parts[best_idx]
            var b = parts[best_idx + 1]
            var merged = a + b
            parts[best_idx] = merged
            # Remove parts[best_idx + 1]
            var new_parts = List[String]()
            for i in range(n):
                if i != best_idx + 1:
                    new_parts.append(parts[i])
            parts = new_parts^
            n = len(parts)
        return parts^

    def encode(self, word: String) raises -> List[Int]:
        """Encode a byte-mapped word into a list of vocab ids."""
        var toks = self.encode_word(word)
        var out = List[Int]()
        for t in toks:
            var id = self.vocab.get(t)
            if id:
                out.append(id.value())
            else:
                # Unknown token: raise (matches upstream BPE with no
                # unk_token set -- GPT-2 byte-level vocab guarantees
                # every byte-mapped char and merge result is present, so
                # this only fires on a misconfigured vocab/merges pair).
                raise Error(
                    "BPE.encode: unknown token '" + t + "' (not in vocab)"
                )
        return out^

    def decode(self, ids: List[Int]) raises -> String:
        """Decode a list of ids back to a string (BPE tokens concatenated)."""
        var s = String()
        for i in ids:
            var t = self.id_to_token.get(i)
            if t:
                s += t.value()
        return s
