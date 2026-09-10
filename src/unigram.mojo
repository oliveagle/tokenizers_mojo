"""Unigram Model — probability-based subword tokenization.

Behavior baseline: HuggingFace tokenizers `models/unigram/mod.rs` (upstream Rust).

Unigram is used by SentencePiece models (T5, ALBERT, XLNet, etc.).
The algorithm uses Viterbi to find the most probable segmentation:
  1. Build a lattice of all possible tokenizations
  2. Use Viterbi algorithm to find the path with highest probability
  3. Return the tokens along that path

The vocabulary consists of (token, score) pairs where score is log-probability.
"""

import traits

from traits import Model


struct UnigramVocabEntry:
    """A vocabulary entry with token and score."""

    var token: String
    var score: Float64

    def __init__(out self, token: String, score: Float64):
        self.token = token
        self.score = score


struct UnigramNode(Copyable, Movable):
    """A node in the Unigram lattice."""

    var id: Int
    var node_id: Int
    var pos: Int
    var length: Int
    var score: Float64
    var backtrace_score: Float64
    var prev: Int  # index of previous node, -1 if none

    def __init__(
        out self,
        id: Int,
        node_id: Int,
        pos: Int,
        length: Int,
        score: Float64,
    ):
        self.id = id
        self.node_id = node_id
        self.pos = pos
        self.length = length
        self.score = score
        self.backtrace_score = 0.0
        self.prev = -1


struct UnigramLattice:
    """Lattice for Viterbi algorithm."""

    var sentence: String
    var len: Int
    var nodes: List[UnigramNode]
    var begin_nodes: List[List[Int]]  # position -> list of node indices
    var end_nodes: List[List[Int]]  # position -> list of node indices
    var bos_id: Int
    var eos_id: Int

    def __init__(out self, sentence: String, bos_id: Int, eos_id: Int):
        self.sentence = sentence
        self.len = sentence.byte_length()
        self.nodes = List[UnigramNode]()
        self.begin_nodes = List[List[Int]]()
        self.end_nodes = List[List[Int]]()

        # Initialize node lists for each position
        for _ in range(self.len + 1):
            self.begin_nodes.append(List[Int]())
            self.end_nodes.append(List[Int]())

        # Add BOS and EOS nodes
        self.bos_id = bos_id
        self.eos_id = eos_id

        var bos = UnigramNode(bos_id, 0, 0, 0, 0.0)
        var eos = UnigramNode(eos_id, 1, self.len, 0, 0.0)

        self.nodes.append(bos^)
        self.nodes.append(eos^)

        self.end_nodes[0].append(0)  # BOS at position 0
        self.begin_nodes[self.len].append(1)  # EOS at end

    def insert(mut self, pos: Int, length: Int, score: Float64, id: Int):
        """Insert a node into the lattice."""
        var node_id = len(self.nodes)
        var node = UnigramNode(id, node_id, pos, length, score)

        self.begin_nodes[pos].append(node_id)
        self.end_nodes[pos + length].append(node_id)
        self.nodes.append(node^)

    def viterbi(mut self) -> List[Int]:
        """Find the best path using Viterbi algorithm."""
        return _viterbi_impl(self)


def _viterbi_impl(mut lattice: UnigramLattice) -> List[Int]:
    """Standalone Viterbi implementation.

    Returns a list of node indices representing the best path.
    """
    var pos = 0

    while pos <= lattice.len:
        if len(lattice.begin_nodes[pos]) == 0:
            return List[Int]()

        for rnode_idx in lattice.begin_nodes[pos]:
            lattice.nodes[rnode_idx].prev = -1
            var best_score = -1e308
            var best_node_idx = -1

            for lnode_idx in lattice.end_nodes[pos]:
                var score = (
                    lattice.nodes[lnode_idx].backtrace_score
                    + lattice.nodes[rnode_idx].score
                )
                if best_node_idx == -1 or score > best_score:
                    best_node_idx = lnode_idx
                    best_score = score

            if best_node_idx != -1:
                lattice.nodes[rnode_idx].prev = best_node_idx
                lattice.nodes[rnode_idx].backtrace_score = best_score

        # Move to next character position using codepoints
        if pos < lattice.len:
            var cp = lattice.sentence.codepoints()
            var current_pos = 0
            for c in cp:
                if current_pos == pos:
                    pos += String(c).byte_length()
                    break
                current_pos += String(c).byte_length()
            if current_pos == pos:
                break
        else:
            break

    # Backtrace from EOS
    var result = List[Int]()
    var eos_idx = lattice.begin_nodes[lattice.len][0]
    var prev_idx = lattice.nodes[eos_idx].prev

    if prev_idx == -1:
        return List[Int]()

    var node_idx = prev_idx
    while node_idx != -1 and lattice.nodes[node_idx].prev != -1:
        result.append(node_idx)  # Return node index, not vocab id
        node_idx = lattice.nodes[node_idx].prev

    # Reverse to get correct order
    var reversed = List[Int]()
    for i in range(len(result) - 1, -1, -1):
        reversed.append(result[i])

    return reversed^


struct Unigram(Model):
    """Unigram model for SentencePiece-style tokenization.

    Attributes:
        vocab: list of vocabulary entries.
        token_to_id: token -> id mapping.
        id_to_token: id -> token mapping.
        unk_id: index of unknown token in vocab.
        bos_id: beginning of sentence token id.
        eos_id: end of sentence token id.
        min_score: minimum score in vocabulary.
        fuse_unk: whether to fuse unknown tokens.
        byte_fallback: whether to fallback to byte representation.
    """

    var vocab: List[UnigramVocabEntry]
    var token_to_id: Dict[String, Int]
    var id_to_token: Dict[Int, String]
    var unk_id: Int
    var bos_id: Int
    var eos_id: Int
    var min_score: Float64
    var fuse_unk: Bool
    var byte_fallback: Bool

    def __init__(out self):
        self.vocab = List[UnigramVocabEntry]()
        self.token_to_id = Dict[String, Int]()
        self.id_to_token = Dict[Int, String]()
        self.unk_id = 0
        self.bos_id = 0
        self.eos_id = 0
        self.min_score = 1e308
        self.fuse_unk = True
        self.byte_fallback = False

    def __init__(
        out self,
        vocab: List[UnigramVocabEntry],
        unk_id: Int,
        byte_fallback: Bool,
    ) raises:
        if len(vocab) == 0:
            raise Error("Unigram: vocabulary is empty but at least <unk> is needed")
        if unk_id >= len(vocab):
            raise Error("Unigram: unk_id is larger than vocabulary size")

        # Copy vocab manually
        self.vocab = List[UnigramVocabEntry]()
        for i in range(len(vocab)):
            var entry = UnigramVocabEntry(vocab[i].token, vocab[i].score)
            self.vocab.append(entry^)

        self.token_to_id = Dict[String, Int]()
        self.id_to_token = Dict[Int, String]()
        self.unk_id = unk_id
        self.bos_id = len(self.vocab) + 1
        self.eos_id = len(self.vocab) + 2
        self.fuse_unk = True
        self.byte_fallback = byte_fallback
        self.min_score = 1e308

        for i in range(len(self.vocab)):
            var token = self.vocab[i].token
            var score = self.vocab[i].score
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            if score < self.min_score:
                self.min_score = score

    def token_for_id(self, id: Int) raises -> String:
        var t = self.id_to_token.get(id)
        if t:
            return t.value()
        return String("")

    def token_id(self, token: String) -> Int:
        """Return the vocab id for `token`, or -1 if not in vocab."""
        var id = self.token_to_id.get(token)
        if id:
            return id.value()
        return -1

    def vocab_size(self) -> Int:
        """Return the vocabulary size."""
        return len(self.vocab)

    def _populate_nodes(self, mut lattice: UnigramLattice) raises:
        """Populate lattice with all possible tokenizations."""
        var unk_score = self.min_score - 10.0  # K_UNK_PENALTY = 10.0
        var length = lattice.len
        var begin_pos = 0

        while begin_pos < length:
            # Try all possible token lengths from this position
            var end_pos = begin_pos + 1
            while end_pos <= length:
                var substr = String(lattice.sentence[byte=begin_pos:end_pos])
                var id = self.token_to_id.get(substr)
                if id:
                    var vocab_id = id.value()
                    var score = self.vocab[vocab_id].score
                    lattice.insert(begin_pos, end_pos - begin_pos, score, vocab_id)
                end_pos += 1

            # If no token found at this position, insert UNK
            if len(lattice.begin_nodes[begin_pos]) == 0:
                # Find the next character boundary
                var mblen = 1
                var cp = lattice.sentence.codepoints()
                var char_pos = 0
                for c in cp:
                    if char_pos == begin_pos:
                        mblen = String(c).byte_length()
                        break
                    char_pos += String(c).byte_length()

                lattice.insert(begin_pos, mblen, unk_score, self.unk_id)

            begin_pos += 1

    def tokenize_word(self, word: String) raises -> List[String]:
        """Tokenize a single word using Viterbi algorithm."""
        var lattice = UnigramLattice(word, self.bos_id, self.eos_id)
        self._populate_nodes(lattice)

        var best_path = lattice.viterbi()
        var tokens = List[String]()

        for node_idx in best_path:
            # Bounds check - node_idx is now a node index, not vocab id
            if node_idx >= 0 and node_idx < len(lattice.nodes):
                var vocab_id = lattice.nodes[node_idx].id
                if vocab_id != self.bos_id and vocab_id != self.eos_id:
                    tokens.append(self.id_to_token[vocab_id])

        return tokens^

    def encode(self, word: String) raises -> List[Int]:
        """Encode a word into a list of vocab ids."""
        var tokens = self.tokenize_word(word)
        var ids = List[Int]()

        for token in tokens:
            var id = self.token_to_id.get(token)
            if id:
                ids.append(id.value())
            else:
                # Unknown token
                if self.unk_id < len(self.vocab):
                    ids.append(self.unk_id)
                else:
                    raise Error(
                        "Unigram.encode: unknown token '" + token + "' not in vocab"
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
