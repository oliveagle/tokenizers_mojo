"""Unicode normalization algorithms (Phase 2).

Implements NFD / NFKD / NFC / NFKC per UAX #15, backed by the
auto-generated tables in `unicode_data.mojo`.  Hangul composition and
decomposition are handled algorithmically (Unicode Standard, 3.12).

`UnicodeData` caches the tables once so the tokenizer hot path does not
rebuild them per token.  `NFCNormalizer` and friends hold a
`UnicodeData` (built once per normalizer instance).

Behavior baseline: Python `unicodedata` / Rust `unicode-normalization`.
"""

import unicode_data


# ---------------------------------------------------------------------------
# Binary search helpers (tables are sorted by key)
# ---------------------------------------------------------------------------


def _binary_search(keys: List[Int], target: Int) -> Int:
    """Return the index of `target` in `keys` (sorted), or -1."""
    var lo = 0
    var hi = len(keys) - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        if keys[mid] == target:
            return mid
        elif keys[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1




def _is_all_ascii(s: String) -> Bool:
    """Check if all codepoints in string are ASCII (0x00-0x7F)."""
    for cp in s.codepoints():
        if Int(cp) >= 0x80:
            return False
    return True

def _ccc(cp: Int, ccc_keys: List[Int], ccc_values: List[Int]) -> Int:
    """Canonical Combining Class of `cp` (0 when not stored)."""
    var idx = _binary_search(ccc_keys, cp)
    if idx == -1:
        return 0
    return ccc_values[idx]


# ---------------------------------------------------------------------------
# Hangul algorithmic decomposition / composition (UAX #15, 3.12)
# ---------------------------------------------------------------------------

comptime S_BASE = 0xAC00
comptime L_BASE = 0x1100
comptime V_BASE = 0x1161
comptime T_BASE = 0x11A7
comptime L_COUNT = 19
comptime V_COUNT = 21
comptime T_COUNT = 28
comptime N_COUNT = V_COUNT * T_COUNT  # 588
comptime S_COUNT = L_COUNT * N_COUNT  # 11172


def _is_hangul_s(cp: Int) -> Bool:
    return cp >= S_BASE and cp < S_BASE + S_COUNT


def _is_hangul_l(cp: Int) -> Bool:
    return cp >= L_BASE and cp < L_BASE + L_COUNT


def _is_hangul_v(cp: Int) -> Bool:
    return cp >= V_BASE and cp < V_BASE + V_COUNT


def _is_hangul_t(cp: Int) -> Bool:
    return cp > T_BASE and cp < T_BASE + T_COUNT


# ---------------------------------------------------------------------------
# UnicodeData: cached tables + normalization methods
# ---------------------------------------------------------------------------


struct UnicodeData:
    """Cached Unicode normalization tables (built once in __init__)."""

    var decomp_keys: List[Int]
    var decomp_compat: List[Int]
    var decomp_pieces: List[Int]
    var decomp_offsets: List[Int]
    var comp_keys_1: List[Int]
    var comp_keys_2: List[Int]
    var comp_targets: List[Int]
    var ccc_keys: List[Int]
    var ccc_values: List[Int]

    def __init__(out self):
        self.decomp_keys = unicode_data._decomp_keys()
        self.decomp_compat = unicode_data._decomp_compat()
        self.decomp_pieces = unicode_data._decomp_pieces_flat()
        self.decomp_offsets = unicode_data._decomp_pieces_offsets()
        self.comp_keys_1 = unicode_data._comp_keys_1()
        self.comp_keys_2 = unicode_data._comp_keys_2()
        self.comp_targets = unicode_data._comp_targets()
        self.ccc_keys = unicode_data._ccc_keys()
        self.ccc_values = unicode_data._ccc_values()

    # -- property lookups --------------------------------------------------

    def ccc_of(self, cp: Int) -> Int:
        return _ccc(cp, self.ccc_keys, self.ccc_values)

    def _decomp_lookup(self, cp: Int, compat: Bool) -> List[Int]:
        """Decomposition of `cp` (empty when none matches the mode)."""
        var result = List[Int]()
        var idx = _binary_search(self.decomp_keys, cp)
        if idx == -1:
            return result^
        if self.decomp_compat[idx] == 1 and not compat:
            return result^
        var start = self.decomp_offsets[idx]
        var stop = self.decomp_offsets[idx + 1]
        for i in range(start, stop):
            result.append(self.decomp_pieces[i])
        return result^

    def _composition_lookup(self, cp1: Int, cp2: Int) -> Int:
        """Composition target for (cp1, cp2), or -1 when none."""
        var lo = 0
        var hi = len(self.comp_keys_1) - 1
        var first = -1
        while lo <= hi:
            var mid = (lo + hi) // 2
            if self.comp_keys_1[mid] == cp1:
                first = mid
                while first > 0 and self.comp_keys_1[first - 1] == cp1:
                    first -= 1
                break
            elif self.comp_keys_1[mid] < cp1:
                lo = mid + 1
            else:
                hi = mid - 1
        if first == -1:
            return -1
        var i = first
        while i < len(self.comp_keys_1) and self.comp_keys_1[i] == cp1:
            if self.comp_keys_2[i] == cp2:
                return self.comp_targets[i]
            i += 1
        return -1

    # -- decomposition (NFD / NFKD) ----------------------------------------

    def _decompose_cp(self, cp: Int, compat: Bool, mut out: List[Int]):
        """Recursively decompose `cp` and append to `out`."""
        if _is_hangul_s(cp):
            var s_index = cp - S_BASE
            var l = L_BASE + s_index // N_COUNT
            var v = V_BASE + (s_index % N_COUNT) // T_COUNT
            var t = T_BASE + s_index % T_COUNT
            out.append(l)
            out.append(v)
            if t > T_BASE:
                out.append(t)
            return
        var pieces = self._decomp_lookup(cp, compat)
        if len(pieces) == 0:
            out.append(cp)
            return
        for piece in pieces:
            self._decompose_cp(piece, compat, out)

    def _canonical_reorder(self, mut cps: List[Int]):
        """Sort combining marks by CCC within each combining sequence.

        A combining sequence starts with a starter (CCC == 0) and includes
        all following combining marks (CCC != 0); marks are stably sorted
        by CCC.
        """
        var n = len(cps)
        var i = 1
        while i < n:
            var ccc_i = self.ccc_of(cps[i])
            if ccc_i == 0:
                i += 1
                continue
            var j = i
            while j > 0:
                var ccc_prev = self.ccc_of(cps[j - 1])
                if ccc_prev == 0 or ccc_prev <= ccc_i:
                    break
                var tmp = cps[j - 1]
                cps[j - 1] = cps[j]
                cps[j] = tmp
                j -= 1
            i += 1

    def _decompose(self, s: String, compat: Bool) raises -> String:
        var cps = List[Int]()
        for cp in s.codepoints():
            self._decompose_cp(Int(cp), compat, cps)
        self._canonical_reorder(cps)
        var out = String()
        for cp in cps:
            out += chr(cp)
        return out

    # -- composition (NFC / NFKC) ------------------------------------------

    def _compose(self, s: String, compat: Bool) raises -> String:
        """Compose `s` (input must already be decomposed)."""
        var cps = List[Int]()
        for cp in s.codepoints():
            cps.append(Int(cp))

        var result = List[Int]()
        var last_starter_idx = -1
        var last_ccc = 0

        for i in range(len(cps)):
            var cp = cps[i]
            var cp_ccc = self.ccc_of(cp)

            if last_starter_idx >= 0 and (last_ccc == 0 or cp_ccc > last_ccc):
                var starter = result[last_starter_idx]
                var composed = self._composition_lookup(starter, cp)
                # Hangul L + V
                if _is_hangul_l(starter) and _is_hangul_v(cp):
                    composed = S_BASE
                    composed += (
                        (starter - L_BASE) * V_COUNT + (cp - V_BASE)
                    ) * T_COUNT
                # Hangul LV + T
                elif (
                    _is_hangul_s(starter)
                    and (starter - S_BASE) % T_COUNT == 0
                    and _is_hangul_t(cp)
                ):
                    composed = starter + (cp - T_BASE)
                if composed != -1:
                    result[last_starter_idx] = composed
                    continue

            if cp_ccc == 0:
                last_starter_idx = len(result)
                last_ccc = 0
            else:
                if last_ccc == 0 or cp_ccc > last_ccc:
                    last_ccc = cp_ccc
            result.append(cp)

        var out = String()
        for cp in result:
            out += chr(cp)
        return out

    # -- public normalizations ----------------------------------------------

    def nfd(self, s: String) raises -> String:
        return self._decompose(s, False)

    def nfkd(self, s: String) raises -> String:
        return self._decompose(s, True)

    def nfc(self, s: String) raises -> String:
        # Fast path: ASCII-only text needs no normalization
        if _is_all_ascii(s):
            return s
        var d = self._decompose(s, False)
        return self._compose(d, False)

    def nfkc(self, s: String) raises -> String:
        var d = self._decompose(s, True)
        return self._compose(d, True)


def make_unicode_data() -> UnicodeData:
    """Build a UnicodeData (tables built once, reused across calls)."""
    return UnicodeData()


def unicode_nfd(s: String) raises -> String:
    """Standalone Canonical Decomposition (NFD)."""
    var ud = UnicodeData()
    return ud.nfd(s)


def unicode_nfkd(s: String) raises -> String:
    """Standalone Compatibility Decomposition (NFKD)."""
    var ud = UnicodeData()
    return ud.nfkd(s)


def unicode_nfc(s: String) raises -> String:
    """Standalone Canonical Composition (NFC)."""
    var ud = UnicodeData()
    return ud.nfc(s)


def unicode_nfkc(s: String) raises -> String:
    """Standalone Compatibility Composition (NFKC)."""
    var ud = UnicodeData()
    return ud.nfkc(s)
