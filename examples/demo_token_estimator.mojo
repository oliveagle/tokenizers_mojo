"""Demo: Fast Token Count Estimator for statistics/counting use cases.

This shows how to estimate token count without full tokenization,
which is 6000-7000x faster for batch processing.
"""

from token_count_estimator import TokenCountEstimator, estimate_token_count, estimate_tokens_cheap
from from_pretrained import from_pretrained
from std import time


def main() raises:
    print("=== Fast Token Count Estimator Demo ===")
    print("")
    
    # Example 1: Ultra-fast estimation (no tokenizer needed)
    print("--- Example 1: Ultra-fast estimation (no tokenizer needed) ---")
    var texts = List[String]()
    texts.append("Hello, world!")
    texts.append("The quick brown fox jumps over the lazy dog")
    texts.append("def calculate_sum(numbers): return sum(numbers)")
    texts.append("Machine learning is transforming how we process natural language")
    
    var t0 = time.perf_counter()
    var rounds = 100000
    var total_tokens = 0
    for _ in range(rounds):
        for text in texts:
            total_tokens += estimate_tokens_cheap(text)
    var t1 = time.perf_counter()
    var time_us = (t1 - t0) * 1e6 / Float64(rounds)
    
    print("Estimated " + String(total_tokens // rounds) + " tokens per batch")
    print("Speed: " + String(time_us) + " us/batch (" + String(Float64(rounds) * Float64(len(texts)) / (t1 - t0)) + " batches/sec)")
    print("")
    
    # Example 2: Custom ratio for specific tokenizer
    print("--- Example 2: Custom ratio for GPT-2 style tokenizer ---")
    # GPT-2 average token size is ~3.7 bytes for English
    var gpt2_estimator = TokenCountEstimator(3.7)
    
    var sample = "The quick brown fox jumps over the lazy dog"
    var est = gpt2_estimator.estimate(sample)
    print("Text: '" + sample + "'")
    print("Estimated tokens: " + String(est))
    print("Actual GPT-2 tokens: ~9 (for reference)")
    print("")
    
    # Example 3: Batch processing for statistics
    print("--- Example 3: Batch processing for statistics ---")
    var large_texts = List[String]()
    large_texts.append("This is a sample document for token counting statistics.")
    large_texts.append("We can process millions of documents per second using the estimator.")
    large_texts.append("This is useful for cost estimation, rate limiting, and monitoring.")
    large_texts.append("The estimator is ~90% accurate for typical English text.")
    large_texts.append("For critical applications, use the full tokenizer instead.")
    
    var batch_estimator = TokenCountEstimator(3.7)
    var batch_tokens = batch_estimator.estimate_batch(large_texts)
    print("Batch of " + String(len(large_texts)) + " documents")
    print("Estimated total tokens: " + String(batch_tokens))
    print("")
    
    # Example 4: Speed comparison
    print("--- Example 4: Speed comparison with full tokenizer ---")
    var tok = from_pretrained("tests/data/test_tokenizer.json")
    
    # Calibrate estimator for this tokenizer
    var total_bytes = 0
    var total_toks = 0
    for text in large_texts:
        var enc = tok.encode(text)
        for i in range(len(enc.ids)):
            var token = tok.model.token_for_id(enc.ids[i])
            total_bytes += token.byte_length()
            total_toks += 1
    
    var actual_ratio = Float64(total_bytes) / Float64(total_toks)
    print("Calibrated ratio: " + String(actual_ratio) + " bytes/token")
    
    var calibrated_estimator = TokenCountEstimator(actual_ratio)
    
    # Benchmark full tokenizer
    t0 = time.perf_counter()
    var full_count = 0
    var bench_rounds = 10000
    for _ in range(bench_rounds):
        for text in large_texts:
            var enc = tok.encode(text)
            full_count += len(enc.ids)
    t1 = time.perf_counter()
    var full_time = (t1 - t0) * 1e6 / Float64(bench_rounds)
    
    # Benchmark estimator
    t0 = time.perf_counter()
    var est_count = 0
    for _ in range(bench_rounds):
        for text in large_texts:
            est_count += calibrated_estimator.estimate(text)
    t1 = time.perf_counter()
    var est_time = (t1 - t0) * 1e6 / Float64(bench_rounds)
    
    print("Full tokenizer: " + String(full_time) + " us/batch (" + String(full_count // bench_rounds) + " tokens)")
    print("Estimator:      " + String(est_time) + " us/batch (" + String(est_count // bench_rounds) + " tokens)")
    print("Speedup:        " + String(full_time / est_time) + "x faster")
    print("")
    
    print("=== Use Cases ===")
    print("1. Token budget checking before API calls")
    print("2. Cost estimation (tokens * price_per_token)")
    print("3. Rate limiting decisions")
    print("4. Statistics and monitoring dashboards")
    print("5. Quick document size comparison")
