"""ByteLevel PreTokenizer — fast path with pre-built String cache."""

import traits
from traits import PreTokenizer, Decoder
from split_optimized import simple_gpt2_split_opt


struct ByteMappingFast:
    """Pre-built 256-entry byte→mapped-String cache."""
    var b2u: List[String]
    var u2b: Dict[String, Int]

    def __init__(out self):
        self.b2u = List[String]()
        self.u2b = Dict[String, Int]()
        for _ in range(256):
            self.b2u.append(String())
        var n = 0
        for b in range(256):
            var cp: Int
            if (
                (b >= 0x21 and b <= 0x7E)
                or (b >= 0xA1 and b <= 0xAC)
                or (b >= 0xAE and b <= 0xFF)
            ):
                cp = b
            else:
                cp = 256 + n
                n += 1
            self.b2u[b] = chr(cp)
            self.u2b[chr(cp)] = b


struct ByteLevelPreTokenizerFast(PreTokenizer):
    """GPT-2-style byte-level pre-tokenizer with pre-built caches."""

    var add_prefix_space: Bool
    var use_regex: Bool
    var byte_mapping: ByteMappingFast

    def __init__(out self, add_prefix_space: Bool = True, use_regex: Bool = True):
        self.add_prefix_space = add_prefix_space
        self.use_regex = use_regex
        self.byte_mapping = ByteMappingFast()

    def pre_tokenize(self, text: String) raises -> List[String]:
        var work = String(text)
        if self.add_prefix_space and not work.startswith(" "):
            work = " " + work

        var splits = List[String]()
        if self.use_regex:
            splits = simple_gpt2_split_opt(work^)
        else:
            splits.append(work^)

        var result = List[String]()
        for token in splits:
            var mapped = String()
            for b in token.bytes():
                mapped += self.byte_mapping.b2u[Int(b)]
            result.append(mapped^)
        return result^

    def pre_tokenize_str(self, text: String) raises -> String:
        var mapped = String()
        for b in text.bytes():
            mapped += self.byte_mapping.b2u[Int(b)]
        return mapped


struct ByteLevelDecoderFast(Decoder):
    """Decoder with pre-built reverse mapping."""
    var byte_mapping: ByteMappingFast

    def __init__(out self):
        self.byte_mapping = ByteMappingFast()

    def decode(self, tokens: List[String]) raises -> String:
        var byte_buf = List[Int]()
        for token in tokens:
            for cp in token.codepoints():
                var cp_val = Int(cp)
                var orig: Int
                if cp_val < 256:
                    orig = cp_val
                else:
                    var opt_b = self.byte_mapping.u2b.get(chr(cp_val))
                    if opt_b:
                        orig = opt_b.value()
                    else:
                        orig = cp_val
                if orig < 0x80:
                    byte_buf.append(orig)
                elif orig < 0x800:
                    byte_buf.append(0xC0 | (orig >> 6))
                    byte_buf.append(0x80 | (orig & 0x3F))
                else:
                    byte_buf.append(0xE0 | (orig >> 12))
                    byte_buf.append(0x80 | ((orig >> 6) & 0x3F))
                    byte_buf.append(0x80 | (orig & 0x3F))
        var result = String()
        for b in byte_buf:
            result += chr(b)
        return result

    def decode_string(self, s: String) raises -> String:
        var lst = List[String]()
        lst.append(s)
        return self.decode(lst)
