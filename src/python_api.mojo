"""Phase 4 - Python interop bridge (file-based, subprocess).

Mojo 1.0.0's `@export` cannot reliably bridge String/Pointer across the C
ABI (see docs/adr-0002-mojo-python-interop.md).  We therefore provide a
file-based bridge: a standalone Mojo executable that

  1. reads a request from  REQ_PATH  (/tmp/tokenizers_mojo_req.json)
     {"op": "encode"|"decode", "path": "<tokenizer.json>",
      "text": "...", "ids": [1,2,...]}
  2. loads the tokenizer with `from_pretrained`
  3. runs encode / decode
  4. writes a result to RESP_PATH (/tmp/tokenizers_mojo_resp.json)

The Python side (scripts/tokenizers_mojo_py.py) wraps this in a
`TokenizerMojo` class whose `encode(text) -> list[int]` / `decode(ids)`
match the HF `tokenizers` API shape.
"""

import json

from from_pretrained import from_pretrained
from json import JsonParser


def read_text(path: String) raises -> String:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    return s^


def write_text(path: String, s: String) raises:
    var f = open(path, "w")
    f.write(s)
    f.close()


def escape_json(s: String) -> String:
    var sb = String()
    for cp in s.codepoints():
        var c = chr(Int(cp))
        if c == '"':
            sb += '\\"'
        elif c == "\\":
            sb += "\\\\"
        elif c == "\n":
            sb += "\\n"
        elif c == "\r":
            sb += "\\r"
        elif c == "\t":
            sb += "\\t"
        else:
            sb += c
    return sb^


def handle_encode(tok_path: String, text: String) raises -> String:
    var tok = from_pretrained(tok_path)
    var enc = tok.encode(text)
    var ids = String()
    var tokens = String()
    for i in range(len(enc.ids)):
        if i > 0:
            ids += ","
            tokens += ","
        ids += String(enc.ids[i])
        tokens += escape_json(enc.tokens[i])
    return (
        '{"ok":true,"ids":['
        + ids
        + '],"tokens":["'
        + tokens
        + '"],'
        + '"len":'
        + String(len(enc.ids))
        + "}"
    )


def handle_create(tok_path: String) raises -> String:
    var tok = from_pretrained(tok_path)
    return (
        '{"ok":true,"vocab":'
        + String(len(tok.model.vocab))
        + ',"merges":'
        + String(len(tok.model.merges))
        + "}"
    )


def handle_decode(tok_path: String, ids_csv: String) raises -> String:
    var tok = from_pretrained(tok_path)
    var id_list = List[Int]()
    if ids_csv.byte_length() > 0:
        var parts = ids_csv.split(",")
        for i in range(len(parts)):
            id_list.append(_to_int(String(parts[i])))
    var tokens = List[String]()
    for i in range(len(id_list)):
        tokens.append(tok.model.token_for_id(id_list[i]))
    var text = tok.decode_str(tokens)
    return '{"ok":true,"text":"' + escape_json(text) + '"}'


def _to_int(s: String) -> Int:
    var acc = 0
    var neg = False
    var i = 0
    if s.byte_length() > 0:
        var first = ord(s[byte=0])
        if first == 45:
            neg = True
            i = 1
    while i < s.byte_length():
        acc = acc * 10 + (ord(s[byte=i]) - 48)
        i += 1
    return -acc if neg else acc


def main() raises:
    var req = read_text("/tmp/tokenizers_mojo_req.json")
    var p = JsonParser(req)
    var root = p.parse()
    var op = p.get_field_str(root, "op")
    var tok_path = p.get_field_str(root, "path")

    var result: String
    if op == "create":
        result = handle_create(tok_path)
    elif op == "encode":
        var text = p.get_field_str(root, "text")
        result = handle_encode(tok_path, text)
    elif op == "decode":
        var ids_idx = p.get_field(root, "ids")
        var ids_arr = p.get_array(ids_idx)
        var ids_csv = String()
        for i in range(len(ids_arr)):
            if i > 0:
                ids_csv += ","
            ids_csv += String(p.get_int(ids_arr[i]))
        result = handle_decode(tok_path, ids_csv)
    else:
        result = '{"ok":false,"error":"unknown op ' + escape_json(op) + '"}'
    write_text("/tmp/tokenizers_mojo_resp.json", result)
