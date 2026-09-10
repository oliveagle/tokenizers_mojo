"""Compile-only entry point for `pixi run check`.

Mojo 1.0.0 has no `mojo check` command, so type-checking the whole tree
is done by building this module, which imports every public module under
`src/`.
"""

import bert_pre_tokenizer
import byte_level
import bpe
import json
import from_pretrained
import bpe_trainer
import encoding
import normalizers
import tokenizer
import whitespace
import added_token
import post_processor
import split
import template_processing
import truncation
import unicode
import unicode_data
import traits
import wordpiece
import wordlevel
import unigram
import wordpiece_trainer
import wordlevel_trainer
import unigram_trainer


def main():
    pass
