"""BPE Trainer (Phase 3).

Behavior baseline: HuggingFace tokenizers `models/bpe/trainer.rs` (upstream
Rust). Implements the standard iterative pair-counting loop:

  1. Add special tokens to vocabulary
  2. Compute initial alphabet (character inventory across corpus;
     initial_alphabet forced in; limit_alphabet truncates by count)
  3. Tokenize words (split into initial symbols, applying
     `continuing_subword_prefix` / `end_of_word_suffix` on non-first / last
     chars; chars outside the alphabet are dropped)
  4. Count all adjacent pairs across the word list
  5. Repeatedly merge the highest-count pair (ties broken by lexicographic
     (id_a, id_b) order, matching upstream Pair=(u32,u32)), add the merged
     token, and re-count pairs.

Fidelity notes (vs upstream):
- Alphabet order is by unicode char code ascending (deterministic), after
  `limit_alphabet` drops the lowest-count chars (initial_alphabet entries
  use a MAX-count sentinel so they always survive).
- Merge tie-break is numeric on symbol ids (not token strings).
- Initial symbols carry `len=1` (upstream hardcodes 1 in tokenize_words);
  merged symbols accumulate length; a pair (x, y) is only counted when
  x.len + y.len < max_token_length (upstream `Word::merge` gate).
- Phase 3 keeps it sequential (rayon-style parallel counting deferred) and
  recounts pairs per merge for correctness first.
"""

import bpe

from bpe import BPE


struct TrainerConfig(Copyable, Movable):
    """Configuration for BpeTrainer (mirrors upstream bpe/trainer.rs::Config).

    - vocab_size: stop once vocab reaches this size.
    - min_frequency: stop when the best pair count drops below this.
    - limit_alphabet: keep only the N most frequent chars (0 = no limit).
    - continuing_subword_prefix: e.g. "##" (BERT). Prepended to any non-first
      char/merge result when building tokens.
    - end_of_word_suffix: e.g. "</w>" (GPT-2). Appended to the last char of
      each word.
    - max_token_length: cap merged token length in bytes (0 = no limit).
    """

    var vocab_size: Int
    var min_frequency: Int
    var limit_alphabet: Int
    var continuing_subword_prefix: String
    var end_of_word_suffix: String
    var max_token_length: Int
    var special_tokens: List[String]
    var initial_alphabet: List[String]

    def __init__(
        out self,
        vocab_size: Int = 30000,
        min_frequency: Int = 0,
        limit_alphabet: Int = 0,
        continuing_subword_prefix: String = "",
        end_of_word_suffix: String = "",
        max_token_length: Int = 0,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.limit_alphabet = limit_alphabet
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.max_token_length = max_token_length
        self.special_tokens = List[String]()
        self.initial_alphabet = List[String]()

    def add_special_token(mut self, token: String):
        self.special_tokens.append(token)

    def add_initial_alphabet(mut self, token: String):
        self.initial_alphabet.append(token)


struct BpeTrainer(Copyable, Movable):
    """Trains a BPE model from a `word → count` corpus."""

    var config: TrainerConfig

    def __init__(out self, config: TrainerConfig = TrainerConfig()):
        self.config = config.copy()

    def __copyinit__(self):
        pass

    def train(mut self, word_counts: Dict[String, Int], mut model: BPE) raises:
        if len(word_counts) == 0:
            return

        var word_to_id = Dict[String, Int]()
        var id_to_word = List[String]()

        # -- 1. special tokens ------------------------------------------------
        for st in self.config.special_tokens:
            if st not in word_to_id:
                word_to_id[st] = len(id_to_word)
                id_to_word.append(st)

        # -- 2. initial alphabet ----------------------------------------------
        # Char frequency across corpus; initial_alphabet forced in with a
        # MAX-count sentinel so limit_alphabet truncation keeps them.
        var char_counts = Dict[String, Int]()
        for word in word_counts.keys():
            var count = word_counts[word]
            for cp in word.codepoints():
                var c = chr(Int(cp))
                char_counts[c] = char_counts.get(c, 0) + count

        var kept = List[String]()
        for c in char_counts.keys():
            if c not in kept:
                kept.append(c)
        for c in self.config.initial_alphabet:
            char_counts[c] = 1000000000  # sentinel "always keep"
            if c not in kept:
                kept.append(c)
        _sort_kept(kept, char_counts, self.config.limit_alphabet)

        for c in kept:
            if c not in word_to_id:
                word_to_id[c] = len(id_to_word)
                id_to_word.append(c)

        # -- 3. tokenize words ---------------------------------------------------
        # Each word becomes a list of symbol ids + parallel byte-lengths.
        # Symbols start with len=1 (upstream hardcodes 1). Chars outside the
        # alphabet are dropped; prefix/suffix tokens are added on first use.
        var words = List[List[Int]]()
        var word_lens = List[List[Int]]()
        var counts = List[Int]()
        var prefix = self.config.continuing_subword_prefix
        var suffix = self.config.end_of_word_suffix
        for word in word_counts.keys():
            var count = word_counts[word]
            var chars = List[String]()
            for cp in word.codepoints():
                chars.append(chr(Int(cp)))
            var syms = List[Int]()
            var lens = List[Int]()
            for i in range(len(chars)):
                var c = chars[i]
                var cid = word_to_id.get(c)
                if cid:
                    var token = c
                    if i > 0 and prefix.byte_length() > 0:
                        token = prefix + c
                    if i == len(chars) - 1 and suffix.byte_length() > 0:
                        token = token + suffix
                    var tid = word_to_id.get(token)
                    if tid:
                        syms.append(tid.value())
                    else:
                        var new_id = len(id_to_word)
                        word_to_id[token] = new_id
                        id_to_word.append(token)
                        syms.append(new_id)
                    lens.append(1)
            if len(syms) > 0:
                words.append(syms^)
                word_lens.append(lens^)
                counts.append(count)

        # -- 4+5. merge loop --------------------------------------------------------
        # Each iteration: recount all pairs, pick the best, merge everywhere.
        var merges_list = List[Tuple[String, String]]()
        var max_token_len = self.config.max_token_length
        var first_count = True
        while len(word_to_id) < self.config.vocab_size:
            var gate = 0
            if not first_count:
                gate = max_token_len
            first_count = False
            var pair_counts = _count_pairs(words, word_lens, counts, gate)
            var best_count = 0
            var best_a = 0
            var best_b = 0
            for key in pair_counts.keys():
                var cnt = pair_counts[key]
                if cnt < 1:
                    continue
                var sep = key.find("|")
                var ida = _parse_int(String(key[byte=0:sep]))
                var idb = _parse_int(String(key[byte = sep + 1 :]))
                if cnt > best_count or (
                    cnt == best_count
                    and (ida < best_a or (ida == best_a and idb < best_b))
                ):
                    best_count = cnt
                    best_a = ida
                    best_b = idb
            if best_count < 1 or best_count < self.config.min_frequency:
                break

            var part_a = id_to_word[best_a]
            var part_b = id_to_word[best_b]

            # new token = a + b (strip prefix from b like upstream)
            var token_b = part_b
            if prefix.byte_length() > 0:
                if part_b.startswith(prefix):
                    var skip = prefix.byte_length()
                    token_b = String(part_b[byte=skip:])
            var new_token = part_a + token_b

            var new_id: Int
            if new_token in word_to_id:
                new_id = word_to_id[new_token]
            else:
                new_id = len(id_to_word)
                word_to_id[new_token] = new_id
                id_to_word.append(new_token)

            merges_list.append((part_a, part_b))

            # Merge in every word (in place).
            for wi in range(len(words)):
                var w = words[wi].copy()
                var wl = word_lens[wi].copy()
                var merged_syms = List[Int]()
                var merged_lens = List[Int]()
                _merge_word(
                    merged_syms, merged_lens, w, wl, best_a, best_b, new_id
                )
                words[wi] = merged_syms^
                word_lens[wi] = merged_lens^

        # -- transfer to model ----------------------------------------------------
        model.vocab.clear()
        model.id_to_token.clear()
        for k in word_to_id.keys():
            model.vocab[k] = word_to_id[k]
        for i in range(len(id_to_word)):
            model.id_to_token[i] = id_to_word[i]
        model.merges.clear()
        for rank in range(len(merges_list)):
            model.merges[
                merges_list[rank][0] + " " + merges_list[rank][1]
            ] = rank


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_int(s: String) -> Int:
    var acc = 0
    for cp in s.codepoints():
        if chr(Int(cp)) >= "0" and chr(Int(cp)) <= "9":
            var digit = Int(cp) - 48
            acc = acc * 10 + digit
    return acc


def _sort_kept(
    mut kept: List[String],
    char_counts: Dict[String, Int],
    limit_alphabet: Int,
) raises:
    """Order `kept` exactly like upstream `compute_alphabet`.

    1. If limit_alphabet > 0, drop the (len - limit) lowest-count chars
       (count ASC; initial_alphabet sentinel = MAX survives).
    2. Sort the survivors by unicode char code ascending.
    """
    if limit_alphabet > 0 and len(kept) > limit_alphabet:
        var to_remove = len(kept) - limit_alphabet
        _sort_by_count_asc(kept, char_counts)
        var survivors = List[String]()
        for i in range(to_remove, len(kept)):
            survivors.append(kept[i])
        kept.clear()
        for c in survivors:
            kept.append(c)
    _sort_by_codepoint(kept)


def _sort_by_count_asc(
    mut lst: List[String],
    char_counts: Dict[String, Int],
) raises:
    """Stable-ish insertion sort by count ASC (ties keep relative order)."""
    for i in range(1, len(lst)):
        var cur = lst[i]
        var cur_c = char_counts.get(cur, 0)
        var j = i - 1
        while j >= 0 and char_counts.get(lst[j], 0) > cur_c:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = cur


def _sort_by_codepoint(mut lst: List[String]) raises:
    """Insertion sort by first unicode codepoint ascending."""
    for i in range(1, len(lst)):
        var cur = lst[i]
        var cur_cp = _first_cp(cur)
        var j = i - 1
        while j >= 0 and _first_cp(lst[j]) > cur_cp:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = cur


def _first_cp(s: String) -> Int:
    for cp in s.codepoints():
        return Int(cp)
    return -1


def _count_pairs(
    words: List[List[Int]],
    word_lens: List[List[Int]],
    counts: List[Int],
    max_token_len: Int,
) raises -> Dict[String, Int]:
    """Count all adjacent pairs across words (frequency-weighted).

    Pair key is "idA|idB" (numeric ids, `|` separator) so tie-breaks can
    compare numeric (id_a, id_b) like upstream Pair=(u32,u32).  When
    max_token_len > 0, a pair (x, y) is only counted if
    x.len + y.len < max_token_len (upstream Word::merge gate).
    """
    var pc = Dict[String, Int]()
    for wi in range(len(words)):
        var w = words[wi].copy()
        var wl = word_lens[wi].copy()
        for j in range(len(w) - 1):
            if max_token_len > 0 and wl[j] + wl[j + 1] >= max_token_len:
                continue
            var key = String(w[j]) + "|" + String(w[j + 1])
            pc[key] = pc.get(key, 0) + counts[wi]
    return pc^


def _merge_word(
    mut out: List[Int],
    mut out_lens: List[Int],
    word: List[Int],
    lens: List[Int],
    part_a_id: Int,
    part_b_id: Int,
    new_id: Int,
) raises:
    """Merge every adjacent (part_a_id, part_b_id) into new_id.

    Unlike a naive byte-length check, upstream always merges the pair; the
    max_token_length gate only affects which *new* pairs get counted (see
    _count_pairs). Returns (merged_symbols, merged_lens).
    """
    var i = 0
    var n = len(word)
    while i < n:
        if word[i] == part_a_id and i + 1 < n and word[i + 1] == part_b_id:
            out.append(new_id)
            out_lens.append(lens[i] + lens[i + 1])
            i += 2
        else:
            out.append(word[i])
            out_lens.append(lens[i])
            i += 1
