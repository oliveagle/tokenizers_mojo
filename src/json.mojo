"""Minimal JSON parser for HF tokenizer.json (Phase 3 `from_pretrained`).

Scope (intentionally narrow):
- Object, Array, String, Int, Bool, Null
- String escapes: standard JSON (quote, backslash, slash, n, r, t, b, f, u+4hex)
- No float (vocab ids are int)
- No trailing commas; standard JSON
- Flat arena storage to sidestep Mojo struct recursion limits

API:
    var p = JsonParser(text)
    var root = p.parse()         # JsonValue index
    var arr = p.get_array(root)
    var obj = p.get_object(root)
    var s = p.get_string(idx)
    var n = p.get_int(idx)
    var b = p.get_bool(idx)
"""


struct JsonValue:
    var kind: Int
    var bool_v: Bool
    var int_v: Int
    var str_v: String
    var children: List[Int]
    var keys: List[String]

    def __init__(out self):
        self.kind = 0  # null
        self.bool_v = False
        self.int_v = 0
        self.str_v = String()
        self.children = List[Int]()
        self.keys = List[String]()

    def __copyinit__(self):
        pass


struct JsonParser:
    """One-shot parser. Construct with the JSON text, call `parse()` once."""

    var src: String
    var pos: Int
    var values: List[JsonValue]

    def __init__(out self, src: String):
        self.src = src
        self.pos = 0
        self.values = List[JsonValue]()

    def parse(mut self) raises -> Int:
        self._skip_ws()
        return self._parse_value()

    # ---- value readers --------------------------------------------------

    def kind_of(self, idx: Int) -> Int:
        return self.values[idx].kind

    def get_string(self, idx: Int) raises -> String:
        if self.values[idx].kind != 3:
            raise Error("expected string")
        return String(self.values[idx].str_v)

    def get_int(self, idx: Int) raises -> Int:
        if self.values[idx].kind != 2:
            raise Error("expected int")
        return self.values[idx].int_v

    def get_bool(self, idx: Int) -> Bool:
        return self.values[idx].kind == 1 and self.values[idx].bool_v

    def get_array(self, idx: Int) raises -> List[Int]:
        if self.values[idx].kind != 4:
            raise Error("expected array")
        return self.values[idx].children.copy()

    def get_object(self, idx: Int) raises -> List[Tuple[String, Int]]:
        if self.values[idx].kind != 5:
            raise Error("expected object")
        var out = List[Tuple[String, Int]]()
        var keys = self.values[idx].keys.copy()
        var vals = self.values[idx].children.copy()
        for i in range(len(keys)):
            out.append((keys[i], vals[i]))
        return out^

    def get_field(self, obj_idx: Int, key: String) raises -> Int:
        var keys = self.values[obj_idx].keys.copy()
        var vals = self.values[obj_idx].children.copy()
        for i in range(len(keys)):
            if keys[i] == key:
                return vals[i]
        raise Error("missing field: " + key)

    def try_get_field(self, obj_idx: Int, key: String) -> Int:
        var keys = self.values[obj_idx].keys.copy()
        var vals = self.values[obj_idx].children.copy()
        for i in range(len(keys)):
            if keys[i] == key:
                return vals[i]
        return -1

    def get_field_str(self, obj_idx: Int, key: String) raises -> String:
        return self.get_string(self.get_field(obj_idx, key))

    def get_field_bool(self, obj_idx: Int, key: String) raises -> Bool:
        var idx = self.get_field(obj_idx, key)
        if self.values[idx].kind != 1:
            raise Error("expected bool for " + key)
        return self.values[idx].bool_v

    def get_field_int(self, obj_idx: Int, key: String) raises -> Int:
        return self.get_int(self.get_field(obj_idx, key))

    def get_field_obj(self, obj_idx: Int, key: String) raises -> Int:
        return self.get_field(obj_idx, key)

    def get_field_arr(self, obj_idx: Int, key: String) raises -> List[Int]:
        return self.get_array(self.get_field(obj_idx, key))

    # ---- internal: parser ------------------------------------------------

    def _new_val(mut self, kind: Int) -> Int:
        var v = JsonValue()
        v.kind = kind
        self.values.append(v^)
        return len(self.values) - 1

    def _skip_ws(mut self):
        while self.pos < self.src.byte_length():
            var c = self._peek()
            if c == 32 or c == 9 or c == 10 or c == 13:
                self.pos += 1
            else:
                break

    def _peek(self) -> Int:
        return ord(self.src[byte=self.pos])

    def _peek_at(self, offset: Int) -> Int:
        var p = self.pos + offset
        if p >= self.src.byte_length():
            return -1
        return ord(self.src[byte=p])

    def _expect(mut self, ch: Int) raises:
        if self.pos >= self.src.byte_length():
            raise Error("unexpected end of input")
        if ord(self.src[byte=self.pos]) != ch:
            raise Error("expected char code " + String(ch))
        self.pos += 1

    def _parse_value(mut self) raises -> Int:
        self._skip_ws()
        if self.pos >= self.src.byte_length():
            raise Error("unexpected end")
        var c = self._peek()
        if c == 123:
            return self._parse_object()
        if c == 91:
            return self._parse_array()
        if c == 34:
            return self._parse_string()
        if c == 116 or c == 102:
            return self._parse_bool()
        if c == 110:
            return self._parse_null()
        if c == 45 or (c >= 48 and c <= 57):
            return self._parse_int()
        raise Error("unexpected char " + String(c) + " at " + String(self.pos))

    def _parse_object(mut self) raises -> Int:
        var obj_idx = self._new_val(5)  # object
        self._expect(123)
        self._skip_ws()
        if self._peek() == 125:
            self.pos += 1
            return obj_idx
        while True:
            self._skip_ws()
            var key_idx = self._parse_string()
            self._skip_ws()
            self._expect(58)
            var val_idx = self._parse_value()
            self.values[obj_idx].children.append(val_idx)
            self.values[obj_idx].keys.append(self.values[key_idx].str_v)
            self._skip_ws()
            if self._peek() == 44:
                self.pos += 1
                continue
            if self._peek() == 125:
                self.pos += 1
                break
            raise Error("expected , or }")
        return obj_idx

    def _parse_array(mut self) raises -> Int:
        var arr_idx = self._new_val(4)
        self._expect(91)
        self._skip_ws()
        if self._peek() == 93:
            self.pos += 1
            return arr_idx
        while True:
            self._skip_ws()
            var v = self._parse_value()
            self.values[arr_idx].children.append(v)
            self._skip_ws()
            if self._peek() == 44:
                self.pos += 1
                continue
            if self._peek() == 93:
                self.pos += 1
                break
            raise Error("expected , or ]")
        return arr_idx

    def _parse_string(mut self) raises -> Int:
        var str_idx = self._new_val(3)
        self._expect(34)
        var sb = String()
        while self.pos < self.src.byte_length():
            var c = self._peek()
            if c == 34:
                self.pos += 1
                self.values[str_idx].str_v = sb^
                return str_idx
            if c == 92:
                self.pos += 1
                var e = self._peek()
                if e == 34:
                    sb += '"'
                elif e == 92:
                    sb += "\\"
                elif e == 47:
                    sb += "/"
                elif e == 110:
                    sb += "\n"
                elif e == 114:
                    sb += "\r"
                elif e == 116:
                    sb += "\t"
                elif e == 98:
                    sb += "\b"
                elif e == 102:
                    sb += "\f"
                elif e == 117:
                    # \uXXXX — decode (BMP only; surrogate pair skipped)
                    self.pos += 1  # skip 'u', land on first hex digit
                    var code = self._parse_hex4()
                    sb += chr(code)
                else:
                    raise Error("bad escape")
                if e != 117:
                    self.pos += 1
                continue
            sb += chr(c)
            self.pos += 1
        raise Error("unterminated string")

    def _parse_hex4(mut self) raises -> Int:
        var code = 0
        for _ in range(4):
            code = code << 4
            var c = self._peek()
            self.pos += 1
            if c >= 48 and c <= 57:
                code |= c - 48
            elif c >= 97 and c <= 102:
                code |= c - 87
            elif c >= 65 and c <= 70:
                code |= c - 55
            else:
                raise Error("bad hex")
        return code

    def _parse_bool(mut self) raises -> Int:
        if self._peek() == 116:  # true
            if (
                self._peek_at(1) != 114
                or self._peek_at(2) != 117
                or self._peek_at(3) != 101
            ):
                raise Error("bad true")
            self.pos += 4
            var idx = self._new_val(1)
            self.values[idx].bool_v = True
            return idx
        if (
            self._peek_at(1) != 97
            or self._peek_at(2) != 108
            or self._peek_at(3) != 115
            or self._peek_at(4) != 101
        ):
            raise Error("bad false")
        self.pos += 5
        var idx = self._new_val(1)
        self.values[idx].bool_v = False
        return idx

    def _parse_null(mut self) raises -> Int:
        if (
            self._peek_at(1) != 117
            or self._peek_at(2) != 108
            or self._peek_at(3) != 108
        ):
            raise Error("bad null")
        self.pos += 4
        return self._new_val(0)

    def _parse_int(mut self) raises -> Int:
        var n = 0
        var neg = False
        if self._peek() == 45:
            neg = True
            self.pos += 1
        if self.pos >= self.src.byte_length():
            raise Error("bad int")
        while self.pos < self.src.byte_length():
            var c = self._peek()
            if c < 48 or c > 57:
                break
            n = n * 10 + (c - 48)
            self.pos += 1
        var idx = self._new_val(2)
        self.values[idx].int_v = -n if neg else n
        return idx
