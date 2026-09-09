"""Tokenizer.from_pretrained — load a BPE tokenizer from HF tokenizer.json.

Phase 3 scope:
- Reads a `tokenizer.json` file (serialized by HF `tokenizers`)
- Extracts: model.vocab, model.merges, model.unk_token, added_tokens,
  pre_tokenizer.type / add_prefix_space, decoder type
- Constructs a `Tokenizer` with a `BPE` model; registers special tokens

Supported (Phase 3):
- model.type == "BPE"
- pre_tokenizer.type == "ByteLevel"
- decoder.type == "ByteLevel"
- added_tokens (id/content/special)
- normalizer (ignored; Phase 3 keeps NFC default)

Unsupported (Phase 3 TODO): WordPiece/WordLevel/Unigram models, non-ByteLevel
pre_tokenizers, non-ByteLevel decoders, file search (no ~/.cache/huggingface
lookup — caller must provide an absolute path).
"""

import bpe
import byte_level
import json
import tokenizer

from bpe import BPE
from byte_level import ByteLevelPreTokenizer
from json import JsonParser
from tokenizer import Tokenizer


def read_file_text(path: String) raises -> String:
    var f = open(path, "r")
    var contents = f.read()
    f.close()
    return contents^


def from_pretrained(path: String) raises -> Tokenizer:
    """Load a tokenizer from a HF `tokenizer.json` file at `path`."""
    var text = read_file_text(path)
    return _from_pretrained_text(text)


def _from_pretrained_text(text: String) raises -> Tokenizer:
    """Parse `tokenizer.json` content and build a Tokenizer."""
    var p = JsonParser(text)
    var root = p.parse()

    var model_idx = p.get_field(root, "model")
    var mtype = p.get_field_str(model_idx, "type")
    if mtype != "BPE":
        raise Error("from_pretrained: unsupported model type '" + mtype + "'")

    # -- vocab ---------------------------------------------------------------
    var vocab_idx = p.get_field(model_idx, "vocab")
    var vocab_keys = p.values[vocab_idx].keys.copy()
    var vocab_vals = p.values[vocab_idx].children.copy()
    var max_id = 0
    for i in range(len(vocab_vals)):
        var v = p.get_int(vocab_vals[i])
        if v > max_id:
            max_id = v
    var vocab = List[String]()
    for _ in range(max_id + 1):
        vocab.append(String())
    var tokens_by_id = Dict[Int, String]()
    for i in range(len(vocab_keys)):
        var tok = vocab_keys[i]
        var tid = p.get_int(vocab_vals[i])
        tokens_by_id[tid] = tok^
    for tid in tokens_by_id.keys():
        vocab[tid] = tokens_by_id[tid]

    # -- merges --------------------------------------------------------------
    var merges_idx = p.get_field(model_idx, "merges")
    var merges_arr = p.get_array(merges_idx)
    var merge_lines = List[String]()
    for i in range(len(merges_arr)):
        var pair_idx = merges_arr[i]
        var pair = p.get_array(pair_idx)
        if len(pair) != 2:
            raise Error("expected 2-element merge pair")
        var a = p.get_string(pair[0])
        var b = p.get_string(pair[1])
        merge_lines.append(a + " " + b)

    # -- construct BPE --------------------------------------------------------
    var model = BPE()
    model.load_vocab(vocab.copy())
    model.load_merges(merge_lines.copy())

    # -- special tokens (added_tokens) ---------------------------------------
    var added_idx = p.get_field(root, "added_tokens")
    var added_arr = p.get_array(added_idx)
    var specials = List[Tuple[Int, String]]()
    for i in range(len(added_arr)):
        var at = added_arr[i]
        var id = p.get_field_int(at, "id")
        var content = p.get_field_str(at, "content")
        var is_special = False
        var spec_idx = p.try_get_field(at, "special")
        if spec_idx >= 0 and p.values[spec_idx].kind == 1:
            is_special = p.values[spec_idx].bool_v
        if is_special:
            specials.append((id, content^))

    # -- pre_tokenizer config --------------------------------------------------
    var pre_idx = p.try_get_field(root, "pre_tokenizer")
    var add_prefix_space = True
    var use_regex = True
    if pre_idx >= 0 and p.values[pre_idx].kind == 5:
        var aps_idx = p.try_get_field(pre_idx, "add_prefix_space")
        if aps_idx >= 0 and p.values[aps_idx].kind == 1:
            add_prefix_space = p.values[aps_idx].bool_v
        var ur_idx = p.try_get_field(pre_idx, "use_regex")
        if ur_idx >= 0 and p.values[ur_idx].kind == 1:
            use_regex = p.values[ur_idx].bool_v

    var tok = Tokenizer(model^)
    tok.pre_tokenizer = ByteLevelPreTokenizer(
        add_prefix_space=add_prefix_space, use_regex=use_regex
    )
    for i in range(len(specials)):
        var pair = specials[i]
        _ = tok.add_special_token(pair[1])
    return tok^
