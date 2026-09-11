"""Optimized BPE - reduced String allocations."""

import byte_level
import traits

from traits import Model


struct BPE(Model):
    """Byte-level BPE model with optimized merge loop."""

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

    def load_vocab_pairs(mut self, keys: List[String], ids: List[Int]) raises:
        if len(keys) != len(ids):
            raise Error("load_vocab_pairs: length mismatch")
        for i in range(len(keys)):
            self.vocab[keys[i]] = ids[i]
            self.id_to_token[ids[i]] = keys[i]

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
        """Run greedy BPE on a single word, returning BPE token strings.
        
        Optimizations:
        1. Pre-allocate parts list
        2. Reuse buffer for pair key construction
        3. Minimize String allocations
        """
        var parts = List[String]()
        for cp in word.codepoints():
            parts.append(chr(Int(cp)))
        if len(parts) == 0:
            return List[String]()
        
        var n = len(parts)
        
        # Cache pair ranks
        var cache = List[Int]()
        for _ in range(n):
            cache.append(-1)
        
        # Initialize cache with adjacent pair ranks
        var buf = String()
        for i in range(n - 1):
            buf = String()
            buf += parts[i]
            buf += " "
            buf += parts[i + 1]
            cache[i] = self.merges.get(buf, -1)
        
        # Greedy merge loop
        while n > 1:
            var best_rank = -1
            var best_idx = -1
            
            # Scan for best merge
            var i = 0
            var n_minus_1 = n - 1
            while i < n_minus_1:
                var r = cache[i]
                if r != -1 and (best_rank == -1 or r < best_rank):
                    best_rank = r
                    best_idx = i
                i += 1
            
            if best_idx == -1:
                break
            
            # Merge the pair - reuse buffer
            buf = String()
            buf += parts[best_idx]
            buf += parts[best_idx + 1]
            parts[best_idx] = buf
            
            # Shift parts left
            for i in range(best_idx + 1, n - 1):
                parts[i] = parts[i + 1]
            n -= 1
            
            # Shift cache left
            for i in range(best_idx, n - 1):
                cache[i] = cache[i + 1]
            
            # Update cache for affected pairs
            if best_idx > 0:
                buf = String()
                buf += parts[best_idx - 1]
                buf += " "
                buf += parts[best_idx]
                cache[best_idx - 1] = self.merges.get(buf, -1)
            if best_idx < n - 1:
                buf = String()
                buf += parts[best_idx]
                buf += " "
                buf += parts[best_idx + 1]
                cache[best_idx] = self.merges.get(buf, -1)
        
        # Pre-allocate result list
        var result = List[String]()
        result.reserve(n)
        for i in range(n):
            result.append(parts[i])
        return result^

    def encode(self, word: String) raises -> List[Int]:
        var toks = self.encode_word(word)
        var out = List[Int]()
        for t in toks:
            var id = self.vocab.get(t)
            if id:
                out.append(id.value())
            else:
                raise Error("BPE.encode: unknown token '" + t + "' (not in vocab)")
        return out^

    def decode(self, ids: List[Int]) raises -> String:
        var s = String()
        for i in ids:
            var t = self.id_to_token.get(i)
            if t:
                s += t.value()
        return s
