"""Phase 4 — Mojo tokenizer CLI driver (Python interop via subprocess).

Mojo 1.0.0's `@export` cannot reliably bridge `String` / `Pointer` across
the C ABI (parametric-pointer export is rejected; the String fat-pointer
ABI is unstable from ctypes).  For Python interop we therefore expose a
tiny stdin/stdout driver:

    echo "hello world" | cli encode <tokenizer.json>
    # -> 48 50

The Python side (scripts/tokenizers_mojo_cli.py) wraps this in a
`TokenizerMojo` class with `encode(text) -> list[int]`, giving
transformers-style Python access without requiring FFI.
"""

import from_pretrained
import env

from from_pretrained import from_pretrained


def cmd_encode(path: String) raises:
    var tok = from_pretrained(path)
    var text = read_stdin()
    var enc = tok.encode(text)
    var sb = String()
    for i in range(len(enc.ids)):
        if i > 0:
            sb += " "
        sb += String(enc.ids[i])
    print(sb)


def cmd_create(path: String) raises:
    var tok = from_pretrained(path)
    print("vocab=" + String(len(tok.model.vocab)))
    print("merges=" + String(len(tok.model.merges)))


def read_stdin() raises -> String:
    var buf = String()
    var line = ""
    while True:
        line = input()
        if line == "":
            break
        buf += line + "\n"
    return buf


def main() raises:
    var args = env.args()
    if len(args) < 3:
        print("usage: cli <encode|create> <tokenizer.json>")
        return
    var cmd = args[1]
    var path = args[2]
    if cmd == "encode":
        cmd_encode(path)
    elif cmd == "create":
        cmd_create(path)
    else:
        print("unknown command: " + cmd)
