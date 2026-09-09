#!/usr/bin/env python3
"""Python interop tests: Mojo bridge output vs HF tokenizers (Phase 4).

Run:
    mojo build -I src src/python_api.mojo -o /tmp/tokenizers_mojo_bridge
    python3 tests/test_python_interop.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.tokenizers_mojo_py import TokenizerMojo
from tokenizers import Tokenizer

FIXTURE = "tests/data/test_tokenizer.json"
SENTENCES = [
    "hello world",
    "how are you",
    "hello",
    "world hello world",
    "the quick brown fox jumps over the lazy dog",
    "tokenization is fun and works on bytes",
]


def main():
    mojo = TokenizerMojo(FIXTURE)
    hf = Tokenizer.from_file(FIXTURE)

    failed = 0
    for s in SENTENCES:
        mojo_ids = mojo.encode(s)
        hf_ids = hf.encode(s).ids
        ok = mojo_ids == hf_ids
        status = "OK" if ok else "MISMATCH"
        print(f"{status}: {s!r} mojo={mojo_ids} hf={hf_ids}")
        if not ok:
            failed += 1
        # decode roundtrip
        mojo_dec = mojo.decode(mojo_ids)
        if not mojo_dec == hf.decode(hf_ids):
            print(f"  decode mismatch: mojo={mojo_dec!r} hf={hf.decode(hf_ids)!r}")
            failed += 1

    if failed:
        print(f"\n{failed} failure(s)")
        sys.exit(1)
    print("\nALL PYTHON INTEROP TESTS PASSED")


if __name__ == "__main__":
    main()
