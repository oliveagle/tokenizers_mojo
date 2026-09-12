"""CompactEncoding — zero-copy with arena-based token storage."""

struct CompactEncoding:
    """Encoding with arena-based string storage."""

    var _len: Int
    var _cap: Int

    var ids: List[Int]
    var raw_offsets: List[Int]
    var type_ids: List[Int]
    var att_mask: List[Int]
    var special_mask: List[Int]
    var seq_ids: List[Int]

    var tokens_arena: List[Int]
    var arena_len: Int
    var token_off: List[Int]

    def __init__(out self):
        self._len = 0
        self._cap = 16
        self.ids = List[Int]()
        self.ids.reserve(self._cap)
        self.raw_offsets = List[Int]()
        self.raw_offsets.reserve(self._cap * 2)
        self.type_ids = List[Int]()
        self.type_ids.reserve(self._cap)
        self.att_mask = List[Int]()
        self.att_mask.reserve(self._cap)
        self.special_mask = List[Int]()
        self.special_mask.reserve(self._cap)
        self.seq_ids = List[Int]()
        self.seq_ids.reserve(self._cap)
        self.tokens_arena = List[Int]()
        self.tokens_arena.reserve(256)
        self.arena_len = 0
        self.token_off = List[Int]()
        self.token_off.reserve(self._cap * 2)

    def reserve(mut self, n: Int):
        if n <= self._cap:
            return
        self._cap = n
        self.ids.reserve(n)
        self.raw_offsets.reserve(n * 2)
        self.type_ids.reserve(n)
        self.att_mask.reserve(n)
        self.special_mask.reserve(n)
        self.seq_ids.reserve(n)
        self.token_off.reserve(n * 2)

    def _ensure_cap(mut self, idx: Int):
        if idx < self._cap:
            return
        while self._cap <= idx:
            self._cap *= 2
        self.ids.reserve(self._cap)
        self.raw_offsets.reserve(self._cap * 2)
        self.type_ids.reserve(self._cap)
        self.att_mask.reserve(self._cap)
        self.special_mask.reserve(self._cap)
        self.seq_ids.reserve(self._cap)
        self.token_off.reserve(self._cap * 2)

    def _append_token_str(mut self, token: String):
        """Append token bytes to arena using .bytes() iterator."""
        var byte_count = token.byte_length()
        self.tokens_arena.reserve(self.arena_len + byte_count)
        for b in token.bytes():
            self.tokens_arena.append(Int(b))

    def push(
        mut self, id: Int, token: String, offset: Tuple[Int, Int]
    ):
        var idx = self._len
        self._ensure_cap(idx)

        while len(self.ids) <= idx:
            self.ids.append(0)
        while len(self.raw_offsets) <= idx * 2 + 1:
            self.raw_offsets.append(0)
        while len(self.type_ids) <= idx:
            self.type_ids.append(0)
        while len(self.att_mask) <= idx:
            self.att_mask.append(0)
        while len(self.special_mask) <= idx:
            self.special_mask.append(0)
        while len(self.seq_ids) <= idx:
            self.seq_ids.append(0)
        while len(self.token_off) <= idx * 2 + 1:
            self.token_off.append(0)

        self.ids[idx] = id
        self.raw_offsets[idx * 2] = offset[0]
        self.raw_offsets[idx * 2 + 1] = offset[1]
        self.type_ids[idx] = 0
        self.att_mask[idx] = 1
        self.special_mask[idx] = 0
        self.seq_ids[idx] = -1

        self.token_off[idx * 2] = self.arena_len
        self.token_off[idx * 2 + 1] = token.byte_length()
        self._append_token_str(token)
        self.arena_len = len(self.tokens_arena)

        self._len = idx + 1

    def push_special(
        mut self, id: Int, token: String, offset: Tuple[Int, Int]
    ):
        var idx = self._len
        self._ensure_cap(idx)

        while len(self.ids) <= idx:
            self.ids.append(0)
        while len(self.raw_offsets) <= idx * 2 + 1:
            self.raw_offsets.append(0)
        while len(self.type_ids) <= idx:
            self.type_ids.append(0)
        while len(self.att_mask) <= idx:
            self.att_mask.append(0)
        while len(self.special_mask) <= idx:
            self.special_mask.append(0)
        while len(self.seq_ids) <= idx:
            self.seq_ids.append(0)
        while len(self.token_off) <= idx * 2 + 1:
            self.token_off.append(0)

        self.ids[idx] = id
        self.raw_offsets[idx * 2] = offset[0]
        self.raw_offsets[idx * 2 + 1] = offset[1]
        self.type_ids[idx] = 0
        self.att_mask[idx] = 1
        self.special_mask[idx] = 1
        self.seq_ids[idx] = -1

        self.token_off[idx * 2] = self.arena_len
        self.token_off[idx * 2 + 1] = token.byte_length()
        self._append_token_str(token)
        self.arena_len = len(self.tokens_arena)

        self._len = idx + 1

    def len(self) -> Int:
        return self._len

    def id_at(self, i: Int) -> Int:
        return self.ids[i]

    def token_at(self, i: Int) -> String:
        var start = self.token_off[i * 2]
        var length = self.token_off[i * 2 + 1]
        var result = String()
        var j = start
        while j < start + length:
            result += chr(self.tokens_arena[j])
            j += 1
        return result

    def get_token_slice(self, i: Int) -> Tuple[Int, Int]:
        return Tuple[Int, Int](
            self.token_off[i * 2],
            self.token_off[i * 2 + 1],
        )

    def get_ids(self) -> List[Int]:
        var out = List[Int]()
        for i in range(self._len):
            out.append(self.ids[i])
        return out^

    def get_tokens(self) -> List[String]:
        var out = List[String]()
        for i in range(self._len):
            out.append(self.token_at(i))
        return out^

    def get_offsets(self) -> List[Tuple[Int, Int]]:
        var out = List[Tuple[Int, Int]]()
        for i in range(self._len):
            out.append(
                Tuple[Int, Int](
                    self.raw_offsets[i * 2],
                    self.raw_offsets[i * 2 + 1],
                )
            )
        return out^

    def merge(mut self, other: CompactEncoding):
        var new_len = self._len + other._len
        self._ensure_cap(new_len)

        var old_len = self._len
        while len(self.ids) < new_len:
            self.ids.append(0)
        while len(self.raw_offsets) < new_len * 2:
            self.raw_offsets.append(0)
        while len(self.type_ids) < new_len:
            self.type_ids.append(0)
        while len(self.att_mask) < new_len:
            self.att_mask.append(0)
        while len(self.special_mask) < new_len:
            self.special_mask.append(0)
        while len(self.seq_ids) < new_len:
            self.seq_ids.append(0)
        while len(self.token_off) < new_len * 2:
            self.token_off.append(0)

        if other._len > 0:
            for i in range(other._len):
                self.ids[old_len + i] = other.ids[i]
                self.type_ids[old_len + i] = other.type_ids[i]
                self.att_mask[old_len + i] = other.att_mask[i]
                self.special_mask[old_len + i] = other.special_mask[i]
                self.seq_ids[old_len + i] = other.seq_ids[i]
                self.raw_offsets[(old_len + i) * 2] = other.raw_offsets[i * 2]
                self.raw_offsets[(old_len + i) * 2 + 1] = other.raw_offsets[
                    i * 2 + 1
                ]

            var arena_base = self.arena_len
            self.tokens_arena.reserve(self.arena_len + other.arena_len)
            for i in range(other.arena_len):
                self.tokens_arena.append(other.tokens_arena[i])

            for i in range(other._len):
                var dst = old_len + i
                self.token_off[dst * 2] = (
                    other.token_off[i * 2] + arena_base
                )
                self.token_off[dst * 2 + 1] = other.token_off[i * 2 + 1]

            self.arena_len = len(self.tokens_arena)

        self._len = new_len
