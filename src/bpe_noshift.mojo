"""BPE Model — no-shift implementation.

Instead of shifting arrays on each merge, marks deleted symbols
and skips them during scan. This eliminates the O(n) shift cost
per merge, reducing overall complexity.

For a word with n chars and m merges:
  - Old: O(m * n) due to shifts
  - New: O(m * n) scan + O(m) deletes (no shift)
"""

import byte_level
import traits

from traits import Model


struct BPEMergeInfo:
    """Tracks which parts are alive and what they merged into."""
    var alive: List[Bool]       # is this slot alive?
    var token: List[String]     # token string at this slot
    var cache: List[Int]        # merge rank cache at this slot
    var n_alive: Int            # count of alive slots

    def __init__(out self, n: Int):
        self.alive = List[Bool]()
        self.token = List[String]()
        self.cache = List[Int]()
        self.n_alive = n
        for _ in range(n):
            self.alive.append(True)
            self.token.append(String())
            self.cache.append(-1)

    def mark_dead(mut self, idx: Int):
        self.alive[idx] = False
        self.n_alive -= 1

    def next_alive(self, from_idx: Int) -> Int:
        """Return index of next alive slot >= from_idx, or -1."""
        var i = from_idx
        while i < len(self.alive):
            if self.alive[i]:
                return i
            i += 1
        return -1

    def prev_alive(self, from_idx: Int) -> Int:
        """Return index of previous alive slot <= from_idx, or -1."""
        var i = from_idx
        while i >= 0:
            if self.alive[i]:
                return i
            i -= 1
        return -1


struct BPENoShift(Model):
    """Byte-level BPE with no-shift merge (mark-and-skip pattern)."""

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

    def encode_word(self, word: String) -> List[String]:
        """Run greedy BPE with no-shift merge."""
        var init_parts = List[String]()
        for cp in word.codepoints():
            init_parts.append(chr(Int(cp)))
        if len(init_parts) == 0:
            return List[String]()^

        var n = len(init_parts)
        var info = BPEMergeInfo(n)

        # Copy initial parts
        for i in range(n):
            info.token[i] = init_parts[i]

        # Initialize cache for adjacent alive pairs
        self._init_cache(info, n)

        # Greedy merge loop
        while info.n_alive > 1:
            var best_rank = -1
            var best_idx = -1

            # Scan all alive adjacent pairs
            var i = 0
            while i < n:
                if not info.alive[i]:
                    i += 1
                    continue
                var nxt = info.next_alive(i + 1)
                if nxt < 0:
                    break
                # Recompute rank for this pair
                var r = self.merges.get(
                    info.token[i] + " " + info.token[nxt], -1
                )
                if r != -1 and (best_rank == -1 or r < best_rank):
                    best_rank = r
                    best_idx = i
                i = nxt

            if best_idx == -1:
                break

            # Find the alive neighbor
            var left_idx = best_idx
            var right_idx = info.next_alive(left_idx + 1)
            if right_idx < 0:
                break

            # Merge: concatenate tokens, mark right as dead
            var left_tok = info.token[left_idx]
            var right_tok = info.token[right_idx]
            info.token[left_idx] = left_tok + right_tok
            info.mark_dead(right_idx)

            # No shift needed! Just skip dead slots.

        # Collect results from alive slots
        var result = List[String]()
        var i = 0
        while i < n:
            if info.alive[i]:
                result.append(info.token[i])
            i += 1
        return result^

    def _init_cache(self, mut info: BPEMergeInfo, n: Int):
        """Initialize merge rank cache."""
        pass  # We recompute ranks on-the-fly in the scan

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
