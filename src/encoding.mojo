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

    def _slice(self, start: Int, stop: Int) -> Encoding:
        """Copy the parallel-array window [start, stop) into a new Encoding."""
        var out = Encoding()
        for i in range(start, stop):
            out.ids.append(self.ids[i])
            out.type_ids.append(self.type_ids[i])
            out.tokens.append(self.tokens[i])
            out.offsets.append(self.offsets[i])
            out.attention_mask.append(self.attention_mask[i])
            out.special_tokens_mask.append(self.special_tokens_mask[i])
            out.sequence_ids.append(self.sequence_ids[i])
        return out^

    def truncate(
        mut self, max_len: Int, stride: Int, direction: String
    ) raises -> List[Encoding]:
        """Truncate this Encoding to at most `max_len` tokens.

        Mirrors upstream `Encoding::truncate`:
          * max_len >= len -> no-op, returns no overflows
          * max_len == 0   -> the whole encoding becomes a single overflow
            and `self` is emptied
          * else           -> `self` becomes the first window of size
            max_len (windows step by `max_len - stride`), and every later
            window is returned as an overflow Encoding.
        Requires `stride < max_len`.

        Returns a fresh List[Encoding] of overflow windows (empty when no
        truncation happened).  Unlike upstream Rust we cannot store
        `List[Encoding]` inside `Encoding` (Mojo rejects recursive
        fields), so the overflows are returned to the caller.
        """
        var encoding_len = self.len()
        var overflows = List[Encoding]()
        if max_len >= encoding_len:
            return overflows^
        if max_len == 0:
            overflows.append(self._slice(0, encoding_len))
            self.ids = List[Int]()
            self.type_ids = List[Int]()
            self.tokens = List[String]()
            self.offsets = List[Tuple[Int, Int]]()
            self.attention_mask = List[Int]()
            self.special_tokens_mask = List[Int]()
            self.sequence_ids = List[Int]()
            return overflows^
        if stride >= max_len:
            raise Error(
                "`stride` must be strictly less than `max_len="
                + String(max_len)
                + "`"
            )

        var offset = max_len - stride
        # build the [start, stop) windows, mirroring upstream parts_ranges
        var parts = List[Tuple[Int, Int]]()
        if direction == "left":
            var stop_pos = encoding_len - 1
            var ended = False
            while stop_pos >= 0 and not ended:
                var stop = stop_pos + 1
                var start = stop - max_len
                if start < 0:
                    start = 0
                if start < stop:
                    ended = start == 0
                    parts.append(Tuple[Int, Int](start, stop))
                stop_pos -= offset
        else:
            # right (default)
            var start = 0
            var ended = False
            while not ended:
                var stop = start + max_len
                if stop > encoding_len:
                    stop = encoding_len
                ended = stop == encoding_len
                parts.append(Tuple[Int, Int](start, stop))
                start += offset

        # overflows = parts[1:]
        for i in range(1, len(parts)):
            overflows.append(self._slice(parts[i][0], parts[i][1]))

        # self = parts[0]
        var first = parts[0]
        var new_ids = List[Int]()
        var new_type_ids = List[Int]()
        var new_tokens = List[String]()
        var new_offsets = List[Tuple[Int, Int]]()
        var new_attention_mask = List[Int]()
        var new_special_tokens_mask = List[Int]()
        var new_sequence_ids = List[Int]()
        for i in range(first[0], first[1]):
            new_ids.append(self.ids[i])
            new_type_ids.append(self.type_ids[i])
            new_tokens.append(self.tokens[i])
            new_offsets.append(self.offsets[i])
            new_attention_mask.append(self.attention_mask[i])
            new_special_tokens_mask.append(self.special_tokens_mask[i])
            new_sequence_ids.append(self.sequence_ids[i])
        self.ids = new_ids^
        self.type_ids = new_type_ids^
        self.tokens = new_tokens^
        self.offsets = new_offsets^
        self.attention_mask = new_attention_mask^
        self.special_tokens_mask = new_special_tokens_mask^
        self.sequence_ids = new_sequence_ids^
        return overflows^
