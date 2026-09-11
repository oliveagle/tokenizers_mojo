"""Comprehensive performance benchmark — identifies specific bottlenecks.

Tests:
1. Pre-tokenization throughput (ByteLevel)
2. Full tokenizer encode throughput (varying input lengths)
3. Decode throughput
4. Vocab lookup throughput
5. End-to-end encode/decode roundtrip

Run:
    mojo run -I src -I tests examples/bench_comprehensive.mojo
"""

from from_pretrained import from_pretrained
from tokenizer import Tokenizer
from encoding import Encoding
from std import time


def bench_pretokenize(tok: Tokenizer, texts: List[String], rounds: Int) raises -> Float64:
    """Benchmark pre-tokenization only."""
    var t0 = time.perf_counter()
    for _ in range(rounds):
        for text in texts:
            _ = tok.pre_tokenizer.pre_tokenize(text)
    var t1 = time.perf_counter()
    return (t1 - t0) / (Float64(rounds) * Float64(len(texts)))


def bench_tokenizer_encode(tok: Tokenizer, texts: List[String], rounds: Int) raises -> Float64:
    """Benchmark full tokenizer encode pipeline."""
    var t0 = time.perf_counter()
    var n = 0
    for _ in range(rounds):
        for text in texts:
            var enc = tok.encode(text)
            n += len(enc.ids)
    var t1 = time.perf_counter()
    return (t1 - t0) / (Float64(rounds) * Float64(len(texts)))


def bench_decode_single(tok: Tokenizer, text: String, rounds: Int) raises -> Float64:
    """Benchmark decoding a single text (encode first, then decode many times)."""
    var enc = tok.encode(text)
    var t0 = time.perf_counter()
    for _ in range(rounds):
        _ = tok.decode(enc)
    var t1 = time.perf_counter()
    return (t1 - t0) / Float64(rounds)


def bench_vocab_lookup(tok: Tokenizer, tokens: List[String], rounds: Int) -> Float64:
    """Benchmark vocabulary lookup."""
    var t0 = time.perf_counter()
    for _ in range(rounds):
        for tok_str in tokens:
            _ = tok.model.token_id(tok_str)
    var t1 = time.perf_counter()
    return (t1 - t0) / (Float64(rounds) * Float64(len(tokens)))


def main() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")

    # Test inputs of varying lengths
    var short_texts = List[String]()
    short_texts.append("hi")
    short_texts.append("ok")
    short_texts.append("go")

    var medium_texts = List[String]()
    medium_texts.append("hello world")
    medium_texts.append("how are you")
    medium_texts.append("good morning")

    var long_texts = List[String]()
    long_texts.append("the quick brown fox jumps over the lazy dog")
    long_texts.append("tokenization is fun and works on bytes")
    long_texts.append("a truly remarkable sentence for benchmarking purposes")

    var mixed_texts = List[String]()
    mixed_texts.append("hi")
    mixed_texts.append("hello world")
    mixed_texts.append("the quick brown fox jumps over the lazy dog")

    var rounds = 5000

    print("=== Performance Benchmark ===")
    print("")

    # 1. Pre-tokenization
    print("--- 1. Pre-tokenization (ByteLevel) ---")
    var dt_short = bench_pretokenize(tok, short_texts, rounds)
    var dt_med = bench_pretokenize(tok, medium_texts, rounds)
    var dt_long = bench_pretokenize(tok, long_texts, rounds)
    print("  Short (2-3 chars):  " + String(dt_short * 1e6) + " us/call")
    print("  Medium (5-12 chars): " + String(dt_med * 1e6) + " us/call")
    print("  Long (30-50 chars):  " + String(dt_long * 1e6) + " us/call")
    print("")

    # 2. Full tokenizer encode (includes normalize + pretokenize + BPE)
    print("--- 2. Full Tokenizer Encode ---")
    var dt_short_enc = bench_tokenizer_encode(tok, short_texts, rounds)
    var dt_med_enc = bench_tokenizer_encode(tok, medium_texts, rounds)
    var dt_long_enc = bench_tokenizer_encode(tok, long_texts, rounds)
    print("  Short (2-3 chars):  " + String(dt_short_enc * 1e6) + " us/call")
    print("  Medium (5-12 chars): " + String(dt_med_enc * 1e6) + " us/call")
    print("  Long (30-50 chars):  " + String(dt_long_enc * 1e6) + " us/call")
    print("")

    # 3. Decode
    print("--- 3. Decode ---")
    var dt_decode = bench_decode_single(tok, "the quick brown fox jumps over the lazy dog", rounds)
    print("  Long (30-50 chars):  " + String(dt_decode * 1e6) + " us/call")
    print("")

    # 4. Vocab lookup
    print("--- 4. Vocab Lookup ---")
    var lookup_tokens = List[String]()
    lookup_tokens.append("Hello")
    lookup_tokens.append("hello")
    lookup_tokens.append("the")
    var dt_lookup = bench_vocab_lookup(tok, lookup_tokens, rounds)
    print("  Single lookup:      " + String(dt_lookup * 1e9) + " ns/call")
    print("")

    # 5. End-to-end
    print("--- 5. End-to-End Encode+Decode ---")
    var t0 = time.perf_counter()
    var total_tokens = 0
    for _ in range(rounds):
        for text in mixed_texts:
            var enc = tok.encode(text)
            total_tokens += len(enc.ids)
            _ = tok.decode(enc)
    var t1 = time.perf_counter()
    var total_calls = Float64(rounds) * Float64(len(mixed_texts))
    var dt_e2e = (t1 - t0) / total_calls
    print("  Mixed lengths:      " + String(dt_e2e * 1e6) + " us/call")
    print("  Total tokens:       " + String(total_tokens))
    print("")

    # Summary
    print("=== Summary ===")
    print("Total time for " + String(Int(total_calls)) + " roundtrips: " + String(t1 - t0) + " seconds")
