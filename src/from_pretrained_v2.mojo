"""Load a TokenizerV2 from a JSON config — reuses V1's from_pretrained BPE model."""

import json
import bpe_v2
import tokenizer_v2
from bpe_v2 import BPE
from tokenizer_v2 import TokenizerV2
from json import JsonParser


def read_file_text(path: String) raises -> String:
    var f = open(path, "r")
    var contents = f.read()
    f.close()
    return contents^


def from_pretrained_v2(path: String) raises -> TokenizerV2:
    """Load a TokenizerV2 from a HuggingFace tokenizer JSON config."""
    var text = read_file_text(path)
    return _from_pretrained_bpe_v2(text)


def _from_pretrained_bpe_v2(text: String) raises -> TokenizerV2:
    """Parse tokenizer.json and build a TokenizerV2."""
    var p = JsonParser(text)
    var root = p.parse()

    var model_idx = p.get_field(root, "model")
    var mtype = p.get_field_str(model_idx, "type")
    if mtype != "BPE":
        raise Error("from_pretrained_v2: only BPE supported, got '" + mtype + "'")

    # -- vocab ---------------------------------------------------------------
    var vocab_idx = p.get_field(model_idx, "vocab")
    var vocab_keys = p.values[vocab_idx].keys.copy()
    var vocab_vals = p.values[vocab_idx].children.copy()

    var model = BPE()
    for i in range(len(vocab_keys)):
        var tok = vocab_keys[i]
        var tid = p.get_int(vocab_vals[i])
        model.add_raw_vocab(tok, tid)

    # -- merges --------------------------------------------------------------
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

    return TokenizerV2(model^)
