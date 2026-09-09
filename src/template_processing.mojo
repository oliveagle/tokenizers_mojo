"""TemplateProcessing post-processor (Phase 2).

Behavior baseline: HuggingFace tokenizers `processors::template`.

Supports the subset of the upstream template mini-language used in
practice:

    [SPECIAL_TOKEN]   -> literal special token (emits id + special mask)
    $A / $0           -> sequence A (type_id 0)
    $B / $1           -> sequence B (type_id 1)
    $A:1 / $B:0       -> sequence with explicit type_id

A full pair template is a space-separated sequence of these elements.
`single` handles one sequence; `pair` handles two (A then B).

Example (BERT):
    single = "[CLS] $A [SEP]"
    pair   = "[CLS] $A [SEP] $B:1 [SEP]:1"
    special_tokens = [("[CLS]", 4), ("[SEP]", 3)]
"""

import encoding

from encoding import Encoding


struct TemplatePiece:
    """One element of a parsed template: special literal or sequence ref.

    Attributes:
        is_special: True if this piece is a literal special token.
        token: The token string (for special tokens).
        id: The token id (for special tokens).
        seq: 0 for sequence A, 1 for sequence B (ignored for specials).
        type_id: The type_id assigned to this sequence segment.
    """

    var is_special: Bool
    var token: String
    var id: Int
    var seq: Int
    var type_id: Int

    def __init__(out self):
        self.is_special = False
        self.token = String()
        self.id = 0
        self.seq = 0
        self.type_id = 0

    def __copyinit__(self):
        pass

    def __moveinit__(mut self, mut other: Self):
        self.is_special = other.is_special
        self.token = String(other.token)
        self.id = other.id
        self.seq = other.seq
        self.type_id = other.type_id


def _parse_int(s: String) -> Int:
    var result = 0
    for cp in s.codepoints():
        var c = Int(cp)
        if c >= 48 and c <= 57:
            result = result * 10 + (c - 48)
    return result


def _parse_piece(
    piece: String, specials: Dict[String, Int]
) raises -> TemplatePiece:
    """Parse one template element into a TemplatePiece.

    Supports:
      - `$A` / `$B`         (seq 0/1, type_id 0)
      - `$0` / `$1`         (seq 0/1, type_id = digit)
      - `$A:N` / `$B:N`     (seq 0/1, explicit type_id N)
      - `$0:N` / `$1:N`     (seq 0/1, explicit type_id N)
      - `[SPECIAL]`         (special token, type_id 0)
      - `[SPECIAL]:N`       (special token with explicit type_id N)
    """
    var p = TemplatePiece()

    if piece.startswith("$"):
        # sequence reference: strip '$' then look for ':type' suffix
        var rest = String(piece[byte=1:])
        var colon_idx = -1
        var has_type = False
        var i = 0
        for cp in rest.codepoints():
            if Int(cp) == 58:  # ':'
                colon_idx = i
                has_type = True
                break
            i += 1

        var type_id = 0
        var seq: Int
        if has_type:
            var seq_part = String(rest[byte=0:colon_idx])
            var type_part = String(rest[byte = colon_idx + 1 :])
            type_id = _parse_int(type_part)
            if seq_part == "A":
                seq = 0
            elif seq_part == "B":
                seq = 1
            else:
                # $0 / $1 forms: digit IS the sequence index
                seq = _parse_int(seq_part)
        else:
            if rest == "A":
                seq = 0
            elif rest == "B":
                seq = 1
            else:
                # bare $0 / $1 -> digit is sequence index AND type_id
                seq = _parse_int(rest)
                type_id = seq
        p.is_special = False
        p.seq = seq
        p.type_id = type_id
    else:
        # special token literal, optionally with :N type_id suffix
        var colon_idx = -1
        var has_type = False
        var i = 0
        for cp in piece.codepoints():
            if Int(cp) == 58:  # ':'
                colon_idx = i
                has_type = True
                break
            i += 1

        var name: String
        var type_id = 0
        if has_type:
            name = String(piece[byte=0:colon_idx])
            var type_part = String(piece[byte = colon_idx + 1 :])
            type_id = _parse_int(type_part)
        else:
            name = piece
        var sid = specials.get(name)
        if sid:
            p.is_special = True
            p.token = name
            p.id = sid.value()
            p.type_id = type_id
        else:
            raise Error(
                "TemplateProcessing: unknown special token '" + name + "'"
            )
    return p^


def _parse_template(
    template: String, specials: Dict[String, Int]
) raises -> List[TemplatePiece]:
    """Split a template string on spaces and parse each piece."""
    var pieces = List[TemplatePiece]()
    var current = String()
    for cp in template.codepoints():
        var c = chr(Int(cp))
        if c == " ":
            if current.byte_length() > 0:
                pieces.append(_parse_piece(current^, specials))
                current = String()
        else:
            current += c
    if current.byte_length() > 0:
        pieces.append(_parse_piece(current^, specials))
    return pieces^


struct TemplateProcessing:
    """Template-based post processor (single + pair templates).

    Attributes:
        single: Parsed template for one sequence.
        pair: Parsed template for two sequences.
        special_tokens: content -> id registry.
    """

    var single: List[TemplatePiece]
    var pair: List[TemplatePiece]
    var special_tokens: Dict[String, Int]

    def __init__(
        out self,
        single_template: String,
        pair_template: String,
        specials: List[Tuple[String, Int]],
    ) raises:
        self.special_tokens = Dict[String, Int]()
        for st in specials:
            self.special_tokens[st[0]] = st[1]
        self.single = _parse_template(single_template, self.special_tokens)
        self.pair = _parse_template(pair_template, self.special_tokens)

    def process(
        self,
        var enc_a: Encoding,
        var enc_b: Encoding,
        add_special_tokens: Bool,
    ) raises -> Encoding:
        """Apply the appropriate template to A (and optionally B).

        Mojo 1.0.0 List iteration requires elements to be
        `(Implicitly)Copyable`; TemplatePiece only conforms to explicit
        `Copyable`, so we always walk the parsed template by index and
        read scalar fields off the reference `List[i]` returns.
        """
        if not add_special_tokens:
            return enc_a^
        var out = Encoding()
        var use_pair = enc_b.len() > 0
        if use_pair:
            for i in range(len(self.pair)):
                self._emit(i, True, enc_a, enc_b, out)
        else:
            for i in range(len(self.single)):
                self._emit(i, False, enc_a, enc_b, out)
        return out^

    def _emit(
        self,
        idx: Int,
        use_pair: Bool,
        enc_a: Encoding,
        enc_b: Encoding,
        mut out: Encoding,
    ) raises:
        if self._is_special(idx, use_pair):
            out.push_special(
                self._id(idx, use_pair),
                self._token(idx, use_pair),
                Tuple[Int, Int](0, 0),
            )
            out.set_sequence_id(out.len() - 1, -1)
            # honor explicit :N type_id on special tokens (e.g. [SEP]:1)
            out.type_ids[out.len() - 1] = self._type_id(idx, use_pair)
            return
        var seq = self._seq(idx, use_pair)
        var type_id = self._type_id(idx, use_pair)
        if seq == 0:
            self._emit_enc(enc_a, seq, type_id, out)
        else:
            self._emit_enc(enc_b, seq, type_id, out)

    def _is_special(self, idx: Int, use_pair: Bool) -> Bool:
        if use_pair:
            return self.pair[idx].is_special
        return self.single[idx].is_special

    def _token(self, idx: Int, use_pair: Bool) -> String:
        if use_pair:
            return self.pair[idx].token
        return self.single[idx].token

    def _id(self, idx: Int, use_pair: Bool) -> Int:
        if use_pair:
            return self.pair[idx].id
        return self.single[idx].id

    def _seq(self, idx: Int, use_pair: Bool) -> Int:
        if use_pair:
            return self.pair[idx].seq
        return self.single[idx].seq

    def _type_id(self, idx: Int, use_pair: Bool) -> Int:
        if use_pair:
            return self.pair[idx].type_id
        return self.single[idx].type_id

    def _emit_enc(
        self,
        enc: Encoding,
        seq: Int,
        type_id: Int,
        mut out: Encoding,
    ) raises:
        for i in range(enc.len()):
            var id = enc.id_at(i)
            var tok = enc.token_at(i)
            var off = enc.offset_at(i)
            if enc.special_tokens_mask[i] == 1:
                out.push_special(id, tok, off)
            else:
                out.push(id, tok, off)
            out.set_sequence_id(out.len() - 1, seq)
            out.type_ids[out.len() - 1] = type_id


def template_process(
    var enc: Encoding,
    add_special_tokens: Bool,
    single_template: String,
    pair_template: String,
    specials: List[Tuple[String, Int]],
) raises -> Encoding:
    """Standalone TemplateProcessing helper."""
    var tp = TemplateProcessing(single_template, pair_template, specials)
    var empty = Encoding()
    return tp.process(enc^, empty^, add_special_tokens)
