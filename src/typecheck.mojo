"""Compile-only entry point for `pixi run check`.

Mojo 1.0.0 has no `mojo check` command, so type-checking the whole tree
is done by building this module, which imports every public module under
`src/`.
"""

import byte_level
import bpe
import encoding
import normalizers
import tokenizer


def main():
    pass
