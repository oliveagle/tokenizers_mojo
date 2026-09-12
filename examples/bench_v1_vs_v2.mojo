"""V1 vs V2 performance comparison benchmark.

Measures:
  1. Pre-tokenization throughput
  2. Full encode throughput  
  3. Encode + Decode roundtrip
"""

from from_pretrained import from_pretrained
from from_pretrained_v2 import from_pretrained_v2
from tokenizer import Tokenizer
from tokenizer_v2 import TokenizerV2
from compact_encoding import CompactEncoding
from std import time


def bench_encode_v1(tok: Tokenizer, texts: List[String], rounds: Int) raises -> Float64:
    var t0 = time.perf_counter()
    for _ in range(rounds):
        for text in texts:
            _ = tok.encode(text)
    var t1 = time.perf_counter()
    return (t1 - t0) / (Float64(rounds) * Float64(len(texts)))


def bench_encode_v2(tok: TokenizerV2, texts: List[String], rounds: Int) raises -> Float64:
    var t0 = time.perf_counter()
    for _ in range(rounds):
        for text in texts:
            _ = tok.encode(text)
    var t1 = time.perf_counter()
    return (t1 - t0) / (Float64(rounds) * Float64(len(texts)))


def bench_pretok_v1(
    tok: Tokenizer, texts: List[String], rounds: Int
) raises -> Float64:
    var t0 = time.perf_counter()
    for _ in range(rounds):
        for text in texts:
            _ = tok.pre_tokenizer.pre_tokenize(text)
    var t1 = time.perf_counter()
    return (t1 - t0) / (Float64(rounds) * Float64(len(texts)))


def bench_pretok_v2(
    tok: TokenizerV2, texts: List[String], rounds: Int
) raises -> Float64:
    var t0 = time.perf_counter()
    for _ in range(rounds):
        for text in texts:
            _ = tok.pre_tokenizer.pre_tokenize(text)
    var t1 = time.perf_counter()
    return (t1 - t0) / (Float64(rounds) * Float64(len(texts)))


def bench_roundtrip_v1(
    tok: Tokenizer, texts: List[String], rounds: Int
) raises -> Float64:
    var t0 = time.perf_counter()
    for _ in range(rounds):
        for text in texts:
            var enc = tok.encode(text)
            _ = tok.decode(enc)
    var t1 = time.perf_counter()
    return (t1 - t0) / (Float64(rounds) * Float64(len(texts)))


def main() raises:
    var tok1 = from_pretrained("tests/data/test_tokenizer.json")
    var tok2 = from_pretrained_v2("tests/data/test_tokenizer.json")

    var texts = List[String]()
    texts.append("hello world")
    texts.append("how are you")
    texts.append("the quick brown fox jumps over the lazy dog")
    texts.append("tokenization is fun and works on bytes")
    texts.append("a truly remarkable sentence for benchmarking purposes")

    var rounds = 5000

    print("=== V1 vs V2 Benchmark ===")
    print("")

    # 1. Pre-tokenization
    print("--- Pre-tokenization ---")
    var dt1_pt = bench_pretok_v1(tok1, texts, rounds)
    var dt2_pt = bench_pretok_v2(tok2, texts, rounds)
    print("  V1: " + String(dt1_pt * 1e6) + " us/call")
    print("  V2: " + String(dt2_pt * 1e6) + " us/call")
    print("  Speedup: " + String(dt1_pt / dt2_pt) + "x")
    print("")

    # 2. Full encode
    print("--- Full Encode ---")
    var dt1_enc = bench_encode_v1(tok1, texts, rounds)
    var dt2_enc = bench_encode_v2(tok2, texts, rounds)
    print("  V1: " + String(dt1_enc * 1e6) + " us/call")
    print("  V2: " + String(dt2_enc * 1e6) + " us/call")
    print("  Speedup: " + String(dt1_enc / dt2_enc) + "x")
    print("")

    # 3. Roundtrip
    print("--- Encode + Decode Roundtrip ---")
    var dt1_rt = bench_roundtrip_v1(tok1, texts, rounds)
    # V2 decode not fully integrated yet, skip
    print("  V1: " + String(dt1_rt * 1e6) + " us/call")
    print("")

    print("=== Done ===")
