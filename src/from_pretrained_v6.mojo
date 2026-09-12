"""Load TokenizerV6 from JSON config."""

import json
import bpe_optimized
import tokenizer_v6

from bpe_optimized import BPEOptimized
from tokenizer_v6 import TokenizerV6
from json import JsonParser


def read_file_text(path: String) raises -> String:
    var f = open(path, "r")
    var contents = f.read()
    f.close()
    return contents^


def from_pretrained_v6(path: String) raises -> TokenizerV6:
    var text = read_file_text(path)
    var p = JsonParser(text)
    var root = p.parse()

    var model_idx = p.get_field(root, "model")
    var mtype = p.get_field_str(model_idx, "type")
    if mtype != "BPE":
        raise Error("from_pretrained_v6: only BPE supported")

    var vocab_idx = p.get_field(model_idx, "vocab")
    var vocab_keys = p.values[vocab_idx].keys.copy()
    var vocab_vals = p.values[vocab_idx].children.copy()

    var model = BPEOptimized()
    for i in range(len(vocab_keys)):
        model.add_raw_vocab(vocab_keys[i], p.get_int(vocab_vals[i]))

    var merges_idx = p.get_field(model_idx, "merges")
    var merges_arr = p.get_array(merges_idx)
    for i in range(len(merges_arr)):
        var pair_idx = merges_arr[i]
        var pair = p.get_array(pair_idx)
        if len(pair) != 2:
            raise Error("expected 2-element merge pair")
        var a = p.get_string(pair[0])
        var b = p.get_string(pair[1])
        model.merges[a + " " + b] = i

    return TokenizerV6(model^)
