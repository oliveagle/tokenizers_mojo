"""Minimal backtracking regex engine (Phase 2, used by Split).

Supports the practical subset of patterns used by tokenizers:
  literals, `.`, character classes `[...]` (ranges, negation, escapes),
  predefined classes `\\d \\D \\w \\W \\s \\S`, greedy quantifiers
  `* + ? {n} {n,} {n,m}`, groups `(...)`, alternation `|`, anchors
  `^ $`.

Compiles the pattern to a small bytecode and matches with a recursive
backtracking VM (SPLIT tries the first branch first = greedy).

Behavior baseline: Python `re` / Rust `regex` for the supported subset.
"""

import whitespace

from whitespace import is_word_char, is_whitespace, is_digit

# ---------------------------------------------------------------------------
def _read_text_bytes(text: String) -> List[Int]:
    """Read all bytes from text into a List for safe byte-level access."""
    var result = List[Int]()
    result.reserve(text.byte_length())
    for b in text.bytes():
        result.append(Int(b))
    return result^


# Bytecode opcodes
# ---------------------------------------------------------------------------

comptime OP_CHAR = 0
comptime OP_DOT = 1
comptime OP_CLASS = 2
comptime OP_SPLIT = 3
comptime OP_JMP = 4
comptime OP_MATCH = 5
comptime OP_ANCHOR_START = 6
comptime OP_ANCHOR_END = 7

# char class kinds
comptime CLASS_CUSTOM = 0
comptime CLASS_DIGIT = 1
comptime CLASS_WORD = 2
comptime CLASS_SPACE = 3


struct Instr:
    """One bytecode instruction."""

    var kind: Int
    var a: Int

    def __init__(out self, kind: Int, a: Int = 0):
        self.kind = kind
        self.a = a


struct CharClass:
    """A character class: custom ranges or a predefined kind."""

    var kind: Int
    var negate: Bool
    var ranges: List[Tuple[Int, Int]]

    def __init__(out self):
        self.kind = CLASS_CUSTOM
        self.negate = False
        self.ranges = List[Tuple[Int, Int]]()

    def matches(self, cp: Int) -> Bool:
        var hit = False
        if self.kind == CLASS_DIGIT:
            hit = is_digit(cp)
        elif self.kind == CLASS_WORD:
            hit = is_word_char(cp)
        elif self.kind == CLASS_SPACE:
            hit = is_whitespace(cp)
        else:
            for r in self.ranges:
                if cp >= r[0] and cp <= r[1]:
                    hit = True
                    break
        if self.negate:
            return not hit
        return hit


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


struct Regex:
    """Compiled regex (instruction list + character classes)."""

    var instrs: List[Instr]
    var classes: List[CharClass]
    var pattern: String
    var pos: Int

    def __init__(out self, pattern: String) raises:
        self.instrs = List[Instr]()
        self.classes = List[CharClass]()
        self.pattern = pattern
        self.pos = 0
        self._parse_alternation()
        self.instrs.append(Instr(OP_MATCH, 0))

    # -- helpers -------------------------------------------------------------

    def _peek(self) raises -> Int:
        """Codepoint at self.pos (in bytes) or -1 at end."""
        if self.pos >= self.pattern.byte_length():
            return -1
        for c in self.pattern[byte=self.pos:]:
            return ord(c)
        return -1

    def _advance(mut self) raises:
        if self.pos >= self.pattern.byte_length():
            return
        # advance by this codepoint's UTF-8 byte length
        var cp = -1
        for c in self.pattern[byte=self.pos:]:
            cp = ord(c)
            break
        if cp < 0x80:
            self.pos += 1
        elif cp < 0x800:
            self.pos += 2
        elif cp < 0x10000:
            self.pos += 3
        else:
            self.pos += 4

    def _emit(mut self, kind: Int, a: Int = 0):
        self.instrs.append(Instr(kind, a))

    def _new_class(mut self, kind: Int = CLASS_CUSTOM) -> Int:
        self.classes.append(CharClass())
        var idx = len(self.classes) - 1
        self.classes[idx].kind = kind
        return idx

    # -- grammar -------------------------------------------------------------

    def _parse_alternation(mut self) raises:
        # collect alternative code segments (start indices)
        var segs = List[Int]()
        segs.append(len(self.instrs))
        self._parse_sequence()
        while self._peek() == 0x7C:  # '|'
            self._advance()
            segs.append(len(self.instrs))
            self._parse_sequence()

        if len(segs) == 1:
            return

        var k = len(segs)
        # 前缀（alternation 之前的指令，如锚点）必须保留
        var prefix_len = segs[0]
        var old = List[Instr]()
        for i in range(len(self.instrs)):
            old.append(Instr(self.instrs[i].kind, self.instrs[i].a))

        # Thompson NFA 交替链（k 个分支）：
        #   SPLIT alt1, alt2        ; alt0: right=ip+1, left=下一个 SPLIT
        #   alt0 code
        #   JMP exit
        #   SPLIT alt2, alt3
        #   alt1 code
        #   JMP exit
        #   ...
        #   alt_{k-1} code          ; 最后分支无 SPLIT
        #   exit:
        var new_instrs = List[Instr]()
        for i in range(prefix_len):
            new_instrs.append(Instr(old[i].kind, old[i].a))
        var split_positions = List[Int]()
        var alt_starts = List[Int]()
        for i in range(k):
            var s = segs[i]
            var e = len(self.instrs) if i == k - 1 else segs[i + 1]
            if i < k - 1:
                split_positions.append(len(new_instrs))
                new_instrs.append(Instr(OP_SPLIT, 0))
            alt_starts.append(len(new_instrs))
            for j in range(s, e):
                new_instrs.append(Instr(old[j].kind, old[j].a))
            if i < k - 1:
                new_instrs.append(Instr(OP_JMP, 0))
        var exit_idx = len(new_instrs)
        # patch JMPs (a==0 placeholder) to exit
        for i in range(len(new_instrs)):
            if new_instrs[i].kind == OP_JMP and new_instrs[i].a == 0:
                new_instrs[i].a = exit_idx
        # patch each SPLIT's left target: 下一个 SPLIT（若存在），否则最后分支代码
        for i in range(len(split_positions)):
            var sp = split_positions[i]
            if i + 1 < len(split_positions):
                new_instrs[sp].a = split_positions[i + 1]
            else:
                new_instrs[sp].a = alt_starts[i + 1]
        self.instrs = new_instrs^

    def _parse_sequence(mut self) raises:
        while True:
            var cp = self._peek()
            if cp == -1 or cp == 0x7C or cp == 0x29:  # end, '|', ')'
                return
            self._parse_repeat()

    def _parse_repeat(mut self) raises:
        var start = len(self.instrs)
        self._parse_atom()
        var cp = self._peek()
        if cp == 0x2A:  # '*'
            self._advance()
            self._emit_star(self._take_body(start))
        elif cp == 0x2B:  # '+'
            self._advance()
            self._emit_plus(self._take_body(start))
        elif cp == 0x3F:  # '?'
            self._advance()
            self._emit_opt(self._take_body(start))
        elif cp == 0x7B:  # '{' -> {n,m}
            self._advance()
            self._emit_braces(self._take_body(start))

    def _take_body(mut self, start: Int) -> List[Instr]:
        """Capture instrs[start:] as the quantifier body, then truncate
        instrs back to `start` so the quantifier block is self-contained."""
        var body = List[Instr]()
        for i in range(start, len(self.instrs)):
            body.append(Instr(self.instrs[i].kind, self.instrs[i].a))
        var trimmed = List[Instr]()
        for i in range(start):
            trimmed.append(Instr(self.instrs[i].kind, self.instrs[i].a))
        self.instrs = trimmed^
        return body^

    def _parse_atom(mut self) raises:
        var cp = self._peek()
        if cp == -1:
            return
        if cp == 0x28:  # '('
            self._advance()
            self._parse_alternation()
            if self._peek() == 0x29:
                self._advance()
            return
        if cp == 0x5B:  # '['
            self._advance()
            self._parse_class()
            return
        if cp == 0x5E:  # '^'
            self._advance()
            self._emit(OP_ANCHOR_START)
            return
        if cp == 0x24:  # '$'
            self._advance()
            self._emit(OP_ANCHOR_END)
            return
        if cp == 0x5C:  # '\'
            self._advance()
            self._parse_escape()
            return
        if cp == 0x2E:  # '.'
            self._advance()
            self._emit(OP_DOT)
            return
        # literal
        self._advance()
        self._emit(OP_CHAR, cp)

    def _parse_escape(mut self) raises:
        var cp = self._peek()
        if cp == -1:
            return
        self._advance()
        if cp == 0x64:  # d
            self._emit(OP_CLASS, self._new_class(CLASS_DIGIT))
        elif cp == 0x44:  # D
            var idx = self._new_class(CLASS_DIGIT)
            self.classes[idx].negate = True
            self._emit(OP_CLASS, idx)
        elif cp == 0x77:  # w
            self._emit(OP_CLASS, self._new_class(CLASS_WORD))
        elif cp == 0x57:  # W
            var idx = self._new_class(CLASS_WORD)
            self.classes[idx].negate = True
            self._emit(OP_CLASS, idx)
        elif cp == 0x73:  # s
            self._emit(OP_CLASS, self._new_class(CLASS_SPACE))
        elif cp == 0x53:  # S
            var idx = self._new_class(CLASS_SPACE)
            self.classes[idx].negate = True
            self._emit(OP_CLASS, idx)
        else:
            # escaped literal
            self._emit(OP_CHAR, cp)

    def _parse_class(mut self) raises:
        var idx = self._new_class()
        if self._peek() == 0x5E:  # leading '^' -> negate
            self._advance()
            self.classes[idx].negate = True
        var first = True
        while True:
            var cp = self._peek()
            if cp == -1:
                break
            if cp == 0x5D and not first:  # ']' ends the class
                self._advance()
                break
            first = False
            if cp == 0x5C:
                self._advance()
                var e = self._peek()
                self._advance()
                # predefined sets inside a class
                if e == 0x64:
                    self.classes[idx].ranges.append(Tuple[Int, Int](0x30, 0x39))
                    continue
                if e == 0x77:
                    self.classes[idx].ranges.append(Tuple[Int, Int](0x41, 0x5A))
                    self.classes[idx].ranges.append(Tuple[Int, Int](0x61, 0x7A))
                    self.classes[idx].ranges.append(Tuple[Int, Int](0x30, 0x39))
                    self.classes[idx].ranges.append(Tuple[Int, Int](0x5F, 0x5F))
                    continue
                if e == 0x73:
                    self.classes[idx].ranges.append(Tuple[Int, Int](0x20, 0x20))
                    self.classes[idx].ranges.append(Tuple[Int, Int](0x09, 0x09))
                    self.classes[idx].ranges.append(Tuple[Int, Int](0x0A, 0x0A))
                    self.classes[idx].ranges.append(Tuple[Int, Int](0x0D, 0x0D))
                    continue
                # escaped literal (e.g. \. \- \] \\)
                var lo = e
                var hi = e
                if self._peek() == 0x2D:  # '-' -> range
                    self._advance()
                    var r = self._peek()
                    self._advance()
                    if r == 0x5C:
                        self._advance()
                        r = self._peek()
                        self._advance()
                    hi = r
                self.classes[idx].ranges.append(Tuple[Int, Int](lo, hi))
                continue
            var lo = cp
            var hi = cp
            self._advance()
            if self._peek() == 0x2D:  # '-' -> range
                self._advance()
                var r = self._peek()
                self._advance()
                if r == 0x5C:
                    self._advance()
                    r = self._peek()
                    self._advance()
                hi = r
            self.classes[idx].ranges.append(Tuple[Int, Int](lo, hi))
        self._emit(OP_CLASS, idx)

    # -- quantifier emission -------------------------------------------------

    def _emit_star(mut self, body: List[Instr]):
        # Thompson NFA for a*:
        #   SPLIT exit, body
        #   body
        #   JMP SPLIT
        #   <exit>
        var split_idx = len(self.instrs)
        self._emit(OP_SPLIT, 0)
        for i in range(len(body)):
            self._emit(body[i].kind, body[i].a)
        self._emit(OP_JMP, split_idx)
        self.instrs[split_idx].a = len(self.instrs)  # exit = past JMP

    def _emit_plus(mut self, body: List[Instr]):
        # Thompson NFA for a+: required body match, then (a*)
        #   body              (required)
        #   SPLIT exit, body
        #   body
        #   JMP SPLIT
        #   <exit>
        for i in range(len(body)):
            self._emit(body[i].kind, body[i].a)
        var split_idx = len(self.instrs)
        self._emit(OP_SPLIT, 0)
        for i in range(len(body)):
            self._emit(body[i].kind, body[i].a)
        self._emit(OP_JMP, split_idx)
        self.instrs[split_idx].a = len(self.instrs)  # exit = past JMP

    def _emit_opt(mut self, body: List[Instr]):
        # Thompson NFA for a?:
        #   SPLIT exit, body
        #   body
        #   <exit>
        var split_idx = len(self.instrs)
        self._emit(OP_SPLIT, 0)
        for i in range(len(body)):
            self._emit(body[i].kind, body[i].a)
        self.instrs[split_idx].a = len(self.instrs)

    def _emit_braces(mut self, body: List[Instr]) raises:
        # parse {n} / {n,} / {n,m}
        var nums = List[Int]()
        var has_comma = False
        while True:
            var cp = self._peek()
            if cp == -1:
                break
            if cp >= 0x30 and cp <= 0x39:
                var v = 0
                while self._peek() >= 0x30 and self._peek() <= 0x39:
                    v = v * 10 + (self._peek() - 0x30)
                    self._advance()
                nums.append(v)
            elif cp == 0x2C:  # ','
                has_comma = True
                self._advance()
            elif cp == 0x7D:  # '}'
                self._advance()
                break
            else:
                self._advance()

        var n = 1
        var m = n
        if len(nums) >= 1:
            n = nums[0]
        if has_comma:
            if len(nums) >= 2:
                m = nums[1]  # {n,m}: bounded max
            else:
                m = -1       # {n,}: unbounded sentinel
        # else: {n} -> m == n

        # emit n copies (required matches)
        for _ in range(n):
            for i in range(len(body)):
                self._emit(body[i].kind, body[i].a)
        # emit (m-n) optional copies, or a star if unbounded
        if m == -1:
            # {n,} = n required copies + (body)* unbounded tail
            var split_idx = len(self.instrs)
            self._emit(OP_SPLIT, 0)
            for i in range(len(body)):
                self._emit(body[i].kind, body[i].a)
            self._emit(OP_JMP, split_idx)
            self.instrs[split_idx].a = len(self.instrs)
        else:
            var extra = m - n
            for _ in range(extra):
                var split_idx = len(self.instrs)
                self._emit(OP_SPLIT, 0)
                for i in range(len(body)):
                    self._emit(body[i].kind, body[i].a)
                var exit_idx = len(self.instrs)
                self.instrs[split_idx].a = exit_idx


# ---------------------------------------------------------------------------
# VM matching
# ---------------------------------------------------------------------------


def _codepoint_at(text: String, byte_pos: Int) raises -> Int:
    """Read codepoint at byte_pos using .bytes() iteration for safety."""
    if byte_pos >= text.byte_length():
        return -1
    var byte_val = 0
    var idx = 0
    for b in text.bytes():
        if idx == byte_pos:
            byte_val = Int(b)
            break
        idx += 1
    # Decode UTF-8 codepoint
    if byte_val < 0x80:
        return byte_val
    elif byte_val < 0xE0:
        # 2-byte sequence
        var cp = byte_val & 0x1F
        for b in text.bytes():
            if idx == byte_pos + 1:
                cp = (cp << 6) | (Int(b) & 0x3F)
                break
            idx += 1
        return cp
    elif byte_val < 0xF0:
        # 3-byte sequence
        var cp = byte_val & 0x0F
        var count = 0
        for b in text.bytes():
            if idx > byte_pos and count < 2:
                cp = (cp << 6) | (Int(b) & 0x3F)
                count += 1
            idx += 1
            if count >= 2:
                break
        return cp
    else:
        # 4-byte sequence
        var cp = byte_val & 0x07
        var count = 0
        for b in text.bytes():
            if idx > byte_pos and count < 3:
                cp = (cp << 6) | (Int(b) & 0x3F)
                count += 1
            idx += 1
            if count >= 3:
                break
        return cp


def _next_byte(text: String, byte_pos: Int) raises -> Int:
    """Advance byte_pos past the current codepoint (UTF-8)."""
    var cp = _codepoint_at(text, byte_pos)
    if cp < 0x80:
        return byte_pos + 1
    if cp < 0x800:
        return byte_pos + 2
    if cp < 0x10000:
        return byte_pos + 3
    return byte_pos + 4


def _vm_run(
    instrs: List[Instr],
    classes: List[CharClass],
    text: String,
    var ip: Int,
    var pos: Int,
) raises -> Int:
    """Run the VM from `ip` at byte offset `pos`; return end byte offset or -1."""
    while True:
        if instrs[ip].kind == OP_CHAR:
            if pos < text.byte_length() and _codepoint_at(text, pos) == instrs[ip].a:
                pos = _next_byte(text, pos)
                ip += 1
            else:
                return -1
        elif instrs[ip].kind == OP_DOT:
            if pos < text.byte_length():
                pos = _next_byte(text, pos)
                ip += 1
            else:
                return -1
        elif instrs[ip].kind == OP_CLASS:
            if pos < text.byte_length():
                var cp = _codepoint_at(text, pos)
                if classes[instrs[ip].a].matches(cp):
                    pos = _next_byte(text, pos)
                    ip += 1
                    continue
            return -1
        elif instrs[ip].kind == OP_SPLIT:
            var r = _vm_run(instrs, classes, text, ip + 1, pos)
            if r != -1:
                return r
            ip = instrs[ip].a
        elif instrs[ip].kind == OP_JMP:
            ip = instrs[ip].a
        elif instrs[ip].kind == OP_ANCHOR_START:
            if pos == 0:
                ip += 1
            else:
                return -1
        elif instrs[ip].kind == OP_ANCHOR_END:
            if pos == text.byte_length():
                ip += 1
            else:
                return -1
        elif instrs[ip].kind == OP_MATCH:
            return pos
        else:
            return -1


def regex_match(pattern: String, text: String) raises -> Bool:
    """Whether the whole pattern matches `text` (anchored both ends)."""
    var re = Regex("^" + pattern + "$")
    return _vm_run(re.instrs, re.classes, text, 0, 0) == text.byte_length()


def regex_find_all(pattern: String, text: String) raises -> List[Tuple[Int, Int]]:
    """Find all non-overlapping matches of `pattern` in `text`.

    Returns (start, end) byte offsets, leftmost-first.  Empty matches are
    skipped by advancing one codepoint (like Python `re`).
    """
    var re = Regex(pattern)
    var results = List[Tuple[Int, Int]]()
    var pos = 0
    var n = text.byte_length()
    while pos < n:
        var end = _vm_run(re.instrs, re.classes, text, 0, pos)
        if end != -1:
            if end > pos:
                results.append(Tuple[Int, Int](pos, end))
                pos = end
            else:
                # empty match: advance one codepoint
                pos = _next_byte(text, pos)
        else:
            pos = _next_byte(text, pos)
    return results^


def regex_split(pattern: String, text: String) raises -> List[String]:
    """Split `text` on non-overlapping regex matches (delimiters removed)."""
    var matches = regex_find_all(pattern, text)
    var out = List[String]()
    var prev = 0
    var text_bytes = _read_text_bytes(text)
    for m in matches:
        if m[0] > prev:
            var seg = String()
            var j = prev
            while j < m[0]:
                seg += chr(text_bytes[j])
                j += 1
            out.append(seg)
        prev = m[1]
    if prev < len(text_bytes):
        var seg = String()
        var j = prev
        while j < len(text_bytes):
            seg += chr(text_bytes[j])
            j += 1
        out.append(seg)
    return out^
