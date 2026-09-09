#!/usr/bin/env python3
"""Rust (HF tokenizers) encode baseline — compare with examples/bench.mojo.

Run:
    python3 scripts/bench_rust.py [rounds]

Output: ns/encode + encodes/sec for HF Rust. Compare with the Mojo output
from `mojo run -I src -I tests examples/bench.mojo`.
"""
import sys
import time
from tokenizers import Tokenizer


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    tk = Tokenizer.from_file("tests/data/test_tokenizer.json")
    texts = [
        "hello world",
        "how are you",
        "the quick brown fox jumps over the lazy dog",
        "tokenization is fun and works on bytes",
        "a truly remarkable sentence for benchmarking purposes",
    ]
    t0 = time.perf_counter()
    n = 0
    for _ in range(rounds):
        for s in texts:
            n += len(tk.encode(s).ids)
    t1 = time.perf_counter()
    total = rounds * len(texts)
    secs = t1 - t0
    print(f"encodes: {total}")
    print(f"total sec: {secs}")
    print(f"ns/encode: {secs * 1e9 / total}")
    print(f"encodes/sec: {total / secs}")
    print(f"tokens touched: {n}")


if __name__ == "__main__":
    main()
