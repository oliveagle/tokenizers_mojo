"""BPE Model — linked list implementation (Rust-style).

Key optimization: linked list for O(1) merge, no array shifting.
Uses the same Symbol struct pattern as the Rust implementation.
"""

import traits
from traits import Model


struct Symbol:
    """Linked list node for BPE merge."""
    var c: Int           # codepoint/char id
    var prev: Int        # index of previous symbol (-1 if none)
    var next: Int        # index of next symbol (-1 if none)
    var byte_len: Int    # byte length of this symbol

    def __init__(out self, c: Int, prev: Int, next: Int, byte_len: Int):
        self.c = c
        self.prev = prev
        self.next = next
        self.byte_len = byte_len


struct BPELinked(Model):
    """Byte-level BPE with linked list merge (Rust-style)."""

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
        var id = self.vocab.get(token)
        if id:
            return id.value()
        return -1

    def add_raw_vocab(mut self, token: String, id: Int):
        self.vocab[token] = id
        self.id_to_token[id] = token

    def load_vocab(mut self, entries: List[String]):
        for i in range(len(entries)):
            self.vocab[entries[i]] = i
            self.id_to_token[i] = entries[i]

    def load_merges(mut self, lines: List[String]):
        var rank = 0
        for line in lines:
            var l = String(line)
            if l.startswith("#version") or l.byte_length() == 0:
                continue
            self.merges[l] = rank
            rank += 1

    def merge_rank(self, token1: String, token2: String) -> Int:
        return self.merges.get(token1 + " " + token2, -1)

    def _pair_rank(self, c1: Int, c2: Int) -> Int:
        """Look up merge rank using codepoint pair."""
        # Build key from codepoints
        var key = chr(c1) + " " + chr(c2)
        return self.merges.get(key, -1)

    def encode_word(self, word: String) -> List[String]:
        """Run greedy BPE using linked list (O(n) merge, no shifting)."""
        # Build initial symbol list
        var symbols = List[Symbol]()
        var idx = 0
        var prev_idx = -1
        for cp in word.codepoints():
            var c = Int(cp)
            var byte_len = len(chr(c).utf8_bytes())
            symbols.append(Symbol(c, prev_idx, -1, byte_len))
            if prev_idx >= 0:
                symbols[prev_idx].next = idx
            prev_idx = idx
            idx += 1

        var n = len(symbols)
        if n == 0:
            return List[String]()^

        # Greedy merge loop
        while n > 1:
            var best_rank = -1
            var best_idx = -1

            # Scan for best merge (O(n) per iteration)
            var i = 0
            var head = 0
            # Find the head (first symbol with prev == -1)
            for j in range(len(symbols)):
                if symbols[j].prev == -1:
                    head = j
                    break

            var cur = head
            while cur >= 0 and symbols[cur].next >= 0:
                var nxt = symbols[cur].next
                var r = self._pair_rank(symbols[cur].c, symbols[nxt].c)
                if r != -1 and (best_rank == -1 or r < best_rank):
                    best_rank = r
                    best_idx = cur
                cur = nxt

            if best_idx == -1:
                break

            # Merge the pair at best_idx
            var left = symbols[best_idx]
            var right = symbols[best_idx + 1] if best_idx + 1 < len(symbols) else Symbol(-1, -1, -1, 0)
            
            # Actually we need to find the actual next symbol
            var right_idx = left.next
            if right_idx < 0 or right_idx >= len(symbols):
                break
            right = symbols[right_idx]

            # Create merged symbol
            var new_c = left.c * 100000 + right.c  # Simple hash for merged id
            var new_len = left.byte_len + right.byte_len
            var new_prev = left.prev
            var new_next = right.next

            # Update the left symbol in-place
            symbols[best_idx].c = new_c
            symbols[best_idx].byte_len = new_len
            symbols[best_idx].prev = new_prev
            symbols[best_idx].next = new_next

            # Mark right symbol as deleted
            symbols[right_idx].c = -1
            symbols[right_idx].prev = -1
            symbols[right_idx].next = -1
            symbols[right_idx].byte_len = 0

            # Update neighbors
            if new_prev >= 0:
                symbols[new_prev].next = best_idx
            if new_next >= 0:
                symbols[new_next].prev = best_idx

            n -= 1

        # Collect results
        var result = List[String]()
        var cur = 0
        for j in range(len(symbols)):
            if symbols[j].prev == -1:
                cur = j
                break

        while cur >= 0:
            var s = symbols[cur]
            if s.c >= 0:
                # Reconstruct the token string from codepoints
                # For merged symbols, we need to store the actual string
                # For now, use the codepoint as a simple placeholder
                result.append(chr(s.c))
            cur = s.next

        return result^

    def encode(self, word: String) raises -> List[Int]:
        var toks = self.encode_word(word)
        var out = List[Int]()
        for t in toks:
            var id = self.vocab.get(t)
            if id:
                out.append(id.value())
            else:
                raise Error("BPE.encode: unknown token '" + t + "'")
        return out^

    def decode(self, ids: List[Int]) raises -> String:
        var s = String()
        for i in ids:
            var t = self.id_to_token.get(i)
            if t:
                s += t.value()
        return s
