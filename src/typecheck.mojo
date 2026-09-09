"""Compile-only entry point for `pixi run check`.

Mojo 1.0.0 has no `mojo check` command, so type-checking the whole tree
is done by building this module, which imports every public module under
`src/`.
"""

import bert_pre_tokenizer
import byte_level
import bpe
import encoding
import normalizers
import tokenizer
import whitespace
import added_token
import post_processor
import split
import template_processing
import truncation
import traits


def main():
    pass
