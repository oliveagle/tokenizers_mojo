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



struct CompactEncoding:
    """Compact encoding: stores all token data in minimal arrays.
    
    Uses a single token_data array (9 fields per token) and a shared
    string buffer, instead of 7 separate List objects. This reduces
    memory allocation overhead by ~2x.
    """
    
    var token_data: List[Int]
    var string_buf: List[Int]
    var count: Int
    
    def __init__(out self):
        self.token_data = List[Int]()
        self.string_buf = List[Int]()
        self.count = 0
    
    def reserve(mut self, n: Int):
        """Pre-allocate for n tokens."""
        self.token_data.reserve(n * 9)
        self.string_buf.reserve(n * 8)
    
    def push(mut self, id: Int, token: String, offset_start: Int, offset_end: Int):
        """Append a token."""
        var token_start = len(self.string_buf)
        var token_len = token.byte_length()
        
        for b in token.bytes():
            self.string_buf.append(Int(b))
        
        self.token_data.append(id)
        self.token_data.append(token_start)
        self.token_data.append(token_len)
        self.token_data.append(offset_start)
        self.token_data.append(offset_end)
        self.token_data.append(0)  # type_id
        self.token_data.append(1)  # attention_mask
        self.token_data.append(0)  # special_tokens_mask
        self.token_data.append(-1)  # sequence_id
        
        self.count += 1
    
    def push_special(mut self, id: Int, token: String, offset_start: Int, offset_end: Int):
        """Append a special token."""
        var token_start = len(self.string_buf)
        var token_len = token.byte_length()
        
        for b in token.bytes():
            self.string_buf.append(Int(b))
        
        self.token_data.append(id)
        self.token_data.append(token_start)
        self.token_data.append(token_len)
        self.token_data.append(offset_start)
        self.token_data.append(offset_end)
        self.token_data.append(0)  # type_id
        self.token_data.append(1)  # attention_mask
        self.token_data.append(1)  # special_tokens_mask (special!)
        self.token_data.append(-1)  # sequence_id
        
        self.count += 1
    
    def len(self) -> Int:
        return self.count
    
    def id_at(self, i: Int) -> Int:
        return self.token_data[i * 9]
    
    def token_at(self, i: Int) -> String:
        var start = self.token_data[i * 9 + 1]
        var length = self.token_data[i * 9 + 2]
        var result = String()
        for j in range(start, start + length):
            result += chr(self.string_buf[j])
        return result
    
    def offset_at(self, i: Int) -> Tuple[Int, Int]:
        return Tuple[Int, Int](self.token_data[i * 9 + 3], self.token_data[i * 9 + 4])
    
    def to_encoding(self) -> Encoding:
        """Convert to standard Encoding."""
        var enc = Encoding()
        enc.ids.reserve(self.count)
        enc.tokens.reserve(self.count)
        enc.offsets.reserve(self.count)
        enc.type_ids.reserve(self.count)
        enc.attention_mask.reserve(self.count)
        enc.special_tokens_mask.reserve(self.count)
        enc.sequence_ids.reserve(self.count)
        
        for i in range(self.count):
            enc.ids.append(self.id_at(i))
            enc.tokens.append(self.token_at(i))
            enc.offsets.append(self.offset_at(i))
            enc.type_ids.append(self.token_data[i * 9 + 5])
            enc.attention_mask.append(self.token_data[i * 9 + 6])
            enc.special_tokens_mask.append(self.token_data[i * 9 + 7])
            enc.sequence_ids.append(self.token_data[i * 9 + 8])
        
        return enc^
