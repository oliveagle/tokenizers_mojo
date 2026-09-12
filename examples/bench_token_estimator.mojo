"""Benchmark: token count estimator vs full tokenizer.

Shows the speed advantage of estimation for statistics/counting use cases.
"""

from token_count_estimator import TokenCountEstimator, estimate_token_count, estimate_tokens_cheap, estimate_token_count_with_ratio
from from_pretrained import from_pretrained
from tokenizer import Tokenizer
from std import time


def main() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")
    
    # Calculate actual ratio from test tokenizer
    var total_bytes = 0
    var total_tokens = 0
    var sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Hello, world!",
        "def calculate_sum(numbers): return sum(numbers)",
        "Machine learning is transforming how we process natural language",
    ]
    
    for text in sample_texts:
        var enc = tok.encode(text)
        for i in range(len(enc.ids)):
            var token = tok.model.token_for_id(enc.ids[i])
            total_bytes += token.byte_length()
            total_tokens += 1
    
    var actual_ratio = Float64(total_bytes) / Float64(total_tokens)
    print("Test tokenizer actual ratio: " + String(actual_ratio) + " bytes/token")
    print("")
    
    # Create estimator with correct ratio
    var estimator = TokenCountEstimator(actual_ratio)
    
    var texts = List[String]()
    texts.append("Hello, world!")
    texts.append("The quick brown fox jumps over the lazy dog")
    texts.append("def calculate_sum(numbers): return sum(numbers)")
    texts.append("1234567890")
    texts.append("https://example.com/path/to/resource?q=value")
    texts.append("Machine learning is transforming how we process natural language")
    texts.append("The API endpoint returns a JSON response with status code 200")
    texts.append("SELECT * FROM users WHERE age > 18 AND country = 'US'")
    
    var rounds = 10000
    
    print("=== Token Count Estimator Benchmark ===")
    print("")
    
    # Benchmark full tokenizer
    var t0 = time.perf_counter()
    var full_count = 0
    for _ in range(rounds):
        for text in texts:
            var enc = tok.encode(text)
            full_count += len(enc.ids)
    var t1 = time.perf_counter()
    var full_time = (t1 - t0) / Float64(rounds)
    
    # Benchmark cheap estimator (byte-length only, default GPT-2 ratio)
    t0 = time.perf_counter()
    var cheap_count = 0
    for _ in range(rounds):
        for text in texts:
            cheap_count += estimate_tokens_cheap(text)
    t1 = time.perf_counter()
    var cheap_time = (t1 - t0) / Float64(rounds)
    
    # Benchmark accurate estimator (with correct ratio)
    t0 = time.perf_counter()
    var acc_count = 0
    for _ in range(rounds):
        for text in texts:
            acc_count += estimator.estimate(text)
    t1 = time.perf_counter()
    var acc_time = (t1 - t0) / Float64(rounds)
    
    print("--- Speed Comparison ---")
    print("Full tokenizer:     " + String(full_time * 1e6) + " us/batch")
    print("Cheap estimator:    " + String(cheap_time * 1e6) + " us/batch")
    print("Accurate estimator: " + String(acc_time * 1e6) + " us/batch")
    print("")
    print("Speedup (cheap vs full):     " + String(full_time / cheap_time) + "x")
    print("Speedup (accurate vs full):  " + String(full_time / acc_time) + "x")
    print("")
    
    print("--- Accuracy ---")
    print("Full tokenizer count: " + String(full_count))
    print("Cheap estimator count: " + String(cheap_count))
    print("Accurate estimator count: " + String(acc_count))
    print("")
    
    # Per-text comparison
    print("--- Per-Text Accuracy ---")
    var total_full = 0
    var total_est = 0
    for text in texts:
        var enc = tok.encode(text)
        var full = len(enc.ids)
        var est = estimator.estimate(text)
        var err = abs(est - full) * 100 // max(full, 1)
        total_full += full
        total_est += est
        print("'" + text + "'")
        print("  Full: " + String(full) + " | Estimated: " + String(est) + " (" + String(err) + "% err)")
    
    var total_err = abs(total_est - total_full) * 100 // max(total_full, 1)
    print("")
    print("Overall accuracy: " + String(100 - total_err) + "% (" + String(total_err) + "% error)")
