#!/usr/bin/env python3
"""Comprehensive Python (HF tokenizers) benchmark — compare with Mojo.

Run:
    python3 scripts/bench_comprehensive.py
"""
import time
from tokenizers import Tokenizer


def bench_encode(tk, texts, rounds):
    t0 = time.perf_counter()
    n = 0
    for _ in range(rounds):
        for s in texts:
            n += len(tk.encode(s).ids)
    t1 = time.perf_counter()
    return (t1 - t0) / (rounds * len(texts))


def bench_decode(tk, texts, rounds):
    # Pre-encode
    encodings = [tk.encode(s) for s in texts]
    t0 = time.perf_counter()
    for _ in range(rounds):
        for enc in encodings:
            _ = tk.decode(enc.ids)
    t1 = time.perf_counter()
    return (t1 - t0) / (rounds * len(texts))


def bench_vocab_lookup(tk, tokens, rounds):
    t0 = time.perf_counter()
    for _ in range(rounds):
        for tok in tokens:
            _ = tk.token_to_id(tok)
    t1 = time.perf_counter()
    return (t1 - t0) / (rounds * len(tokens))


def main():
    tk = Tokenizer.from_file("tests/data/test_tokenizer.json")

    short_texts = ["hi", "ok", "go"]
    medium_texts = ["hello world", "how are you", "good morning"]
    long_texts = [
        "the quick brown fox jumps over the lazy dog",
        "tokenization is fun and works on bytes",
        "a truly remarkable sentence for benchmarking purposes",
    ]
    mixed_texts = ["hi", "hello world", "the quick brown fox jumps over the lazy dog"]

    rounds = 5000

    print("=== Python (HF tokenizers) Benchmark ===")
    print("")

    print("--- 1. Pre-tokenization (ByteLevel) ---")
    print("  (HF: integrated in encode, not separately benchmarked)")
    print("")

    print("--- 2. Full Tokenizer Encode ---")
    dt_short = bench_encode(tk, short_texts, rounds) * 1e6
    dt_med = bench_encode(tk, medium_texts, rounds) * 1e6
    dt_long = bench_encode(tk, long_texts, rounds) * 1e6
    print(f"  Short (2-3 chars):  {dt_short:.2f} us/call")
    print(f"  Medium (5-12 chars): {dt_med:.2f} us/call")
    print(f"  Long (30-50 chars):  {dt_long:.2f} us/call")
    print("")

    print("--- 3. Decode ---")
    dt_decode = bench_decode(tk, long_texts, rounds) * 1e6
    print(f"  Long (30-50 chars):  {dt_decode:.2f} us/call")
    print("")

    print("--- 4. Vocab Lookup ---")
    lookup_tokens = ["Hello", "hello", "the"]
    dt_lookup = bench_vocab_lookup(tk, lookup_tokens, rounds) * 1e9
    print(f"  Single lookup:      {dt_lookup:.2f} ns/call")
    print("")

    print("--- 5. End-to-End Encode+Decode ---")
    t0 = time.perf_counter()
    total_tokens = 0
    for _ in range(rounds):
        for s in mixed_texts:
            enc = tk.encode(s)
            total_tokens += len(enc.ids)
            _ = tk.decode(enc.ids)
    t1 = time.perf_counter()
    total_calls = rounds * len(mixed_texts)
    dt_e2e = (t1 - t0) / total_calls * 1e6
    print(f"  Mixed lengths:      {dt_e2e:.2f} us/call")
    print(f"  Total tokens:       {total_tokens}")
    print("")

    print("=== Summary ===")
    print(f"Total time for {total_calls} roundtrips: {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    main()
