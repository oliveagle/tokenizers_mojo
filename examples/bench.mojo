"""Phase 3 performance benchmark — Mojo vs Rust (HF tokenizers).

Measures encode() throughput on the HF-trained fixture tokenizer.

Run:
    mojo run -I src -I tests examples/bench.mojo
    python3 scripts/bench_rust.py          # Rust/HF baseline (see goals)

Output: ns/encode and encodes/sec for Mojo; compare with the Rust
baseline reported by scripts/bench_rust.py.
"""


from from_pretrained import from_pretrained
from tokenizer import Tokenizer
from std import time


def encode_loop(tok: Tokenizer, texts: List[String], rounds: Int) raises -> Int:
    var n = 0
    for _ in range(rounds):
        for i in range(len(texts)):
            var enc = tok.encode(texts[i])
            n += len(enc.ids)
    return n


def main() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")

    var texts = List[String]()
    texts.append("hello world")
    texts.append("how are you")
    texts.append("the quick brown fox jumps over the lazy dog")
    texts.append("tokenization is fun and works on bytes")
    texts.append("a truly remarkable sentence for benchmarking purposes")

    var rounds = 2000
    var t0 = time.perf_counter()
    var n = encode_loop(tok, texts, rounds)
    var t1 = time.perf_counter()

    var total = Float64(rounds) * Float64(len(texts))
    var secs = t1 - t0
    var per_encode = secs / total
    var per_sec = total / secs
    print("encodes: " + String(total))
    print("total sec: " + String(secs))
    print("ns/encode: " + String(per_encode * 1_000_000_000))
    print("encodes/sec: " + String(per_sec))
    print("tokens touched: " + String(n))
