"""Encoding -- the unified output of a Tokenizer.

Stores parallel arrays of token IDs, string representations, offsets,
and mask arrays.  Mirrors the Encoding struct in the upstream Rust
implementation (tokenizers/src/tokenizer/encoding.rs) kept minimal for
Phase 1.
"""


struct Encoding:
    """Output of a tokenization pass (parallel arrays kept in sync)."""

    var ids: List[Int]
    var tokens: List[String]
    var offsets: List[Tuple[Int, Int]]
    var type_ids: List[Int]
    var attention_mask: List[Int]
    var special_tokens_mask: List[Int]
    var sequence_ids: List[Int]

    def __init__(out self):
        self.ids = List[Int]()
        self.tokens = List[String]()
        self.offsets = List[Tuple[Int, Int]]()
        self.type_ids = List[Int]()
        self.attention_mask = List[Int]()
        self.special_tokens_mask = List[Int]()
        self.sequence_ids = List[Int]()

    def push(mut self, id: Int, token: String, offset: Tuple[Int, Int]):
        """Append a single token to all parallel arrays."""
        self.ids.append(id)
        self.tokens.append(token)
        self.offsets.append(offset)
        self.type_ids.append(0)
        self.attention_mask.append(1)
        self.special_tokens_mask.append(0)
        self.sequence_ids.append(-1)

    def push_special(mut self, id: Int, token: String, offset: Tuple[Int, Int]):
        """Append a special token (bypasses BPE, marks special masks)."""
        self.ids.append(id)
        self.tokens.append(token)
        self.offsets.append(offset)
        self.type_ids.append(0)
        self.attention_mask.append(1)
        self.special_tokens_mask.append(1)
        self.sequence_ids.append(-1)

    def len(self) -> Int:
        return len(self.ids)

    def get_ids(self) -> List[Int]:
        var out = List[Int]()
        for v in self.ids:
            out.append(v)
        return out^

    def get_tokens(self) -> List[String]:
        var out = List[String]()
        for v in self.tokens:
            out.append(v)
        return out^

    def get_offsets(self) -> List[Tuple[Int, Int]]:
        var out = List[Tuple[Int, Int]]()
        for v in self.offsets:
            out.append(v)
        return out^

    def id_at(self, i: Int) -> Int:
        return self.ids[i]

    def token_at(self, i: Int) -> String:
        return self.tokens[i]

    def offset_at(self, i: Int) -> Tuple[Int, Int]:
        return self.offsets[i]

    def merge(mut self, other: Encoding):
        """Concatenate another Encoding's arrays onto this one."""
        for v in other.ids:
            self.ids.append(v)
        for v in other.tokens:
            self.tokens.append(v)
        for v in other.offsets:
            self.offsets.append(v)
        for v in other.type_ids:
            self.type_ids.append(v)
        for v in other.attention_mask:
            self.attention_mask.append(v)
        for v in other.special_tokens_mask:
            self.special_tokens_mask.append(v)
        for v in other.sequence_ids:
            self.sequence_ids.append(v)

    def set_sequence_id(mut self, i: Int, seq: Int):
        """Set the sequence id of token i (for post-processing)."""
        self.sequence_ids[i] = seq

    def get_sequence_ids(self) -> List[Int]:
        var out = List[Int]()
        for v in self.sequence_ids:
            out.append(v)
        return out^

    def set_type_ids_all(mut self, v: Int):
        """Set every type_id to v (Roberta sets all to 0)."""
        for i in range(len(self.type_ids)):
            self.type_ids[i] = v

    def to_string(self) -> String:
        """Human-readable summary (for debugging / tests)."""
        var s = String()
        for i in range(self.len()):
            if i > 0:
                s += " "
            s += "(" + String(self.ids[i]) + ", " + self.tokens[i] + ")"
        return s
