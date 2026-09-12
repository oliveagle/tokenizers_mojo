"""Benchmark: Token Count Estimator V1 vs V2 accuracy comparison."""

from token_count_estimator import TokenCountEstimator, estimate_tokens_cheap
from token_count_estimator_v2 import TokenCountEstimatorV2, estimate_token_count_v2
from from_pretrained import from_pretrained
from std import time


def main() raises:
    var tok = from_pretrained("tests/data/test_tokenizer.json")
    
    # Calibrate both estimators
    var calib_texts = List[String]()
    calib_texts.append("The quick brown fox jumps over the lazy dog")
    calib_texts.append("Hello, world!")
    calib_texts.append("def calculate_sum(numbers): return sum(numbers)")
    calib_texts.append("Machine learning is transforming how we process natural language")
    calib_texts.append("1234567890")
    
    var calib_counts = List[Int]()
    for text in calib_texts:
        var enc = tok.encode(text)
        calib_counts.append(len(enc.ids))
    
    # Calculate calibration ratio
    var total_bytes = 0
    var total_tokens = 0
    for i in range(len(calib_texts)):
        total_bytes += calib_texts[i].byte_length()
        total_tokens += calib_counts[i]
    var calibrated_ratio = Float64(total_bytes) / Float64(total_tokens)
    
    print("=== Token Count Estimator V1 vs V2 Comparison ===")
    print("")
    print("Calibration ratio: " + String(calibrated_ratio) + " bytes/token")
    print("")
    
    # Test texts
    var test_texts = List[String]()
    test_texts.append("Hello, world!")
    test_texts.append("The quick brown fox jumps over the lazy dog")
    test_texts.append("def calculate_sum(numbers): return sum(numbers)")
    test_texts.append("1234567890")
    test_texts.append("https://example.com/path/to/resource?q=value")
    test_texts.append("Machine learning is transforming how we process natural language")
    test_texts.append("The API endpoint returns a JSON response with status code 200")
    test_texts.append("SELECT * FROM users WHERE age > 18 AND country = 'US'")
    test_texts.append("import numpy as np; arr = np.array([1, 2, 3, 4, 5])")
    test_texts.append("The quick brown fox jumps over the lazy dog. This is a longer sentence.")
    
    # Create estimators
    var est_v1 = TokenCountEstimator(calibrated_ratio)
    var est_v2 = TokenCountEstimatorV2(calibrated_ratio)
    
    # Accuracy comparison
    print("--- Accuracy Comparison ---")
    var total_full = 0
    var total_v1 = 0
    var total_v2 = 0
    var total_cheap = 0
    
    for text in test_texts:
        var enc = tok.encode(text)
        var full = len(enc.ids)
        var v1 = est_v1.estimate(text)
        var v2 = est_v2.estimate(text)
        var cheap = estimate_tokens_cheap(text)
        
        total_full += full
        total_v1 += v1
        total_v2 += v2
        total_cheap += cheap
        
        var err_v1 = abs(v1 - full) * 100 // max(full, 1)
        var err_v2 = abs(v2 - full) * 100 // max(full, 1)
        var err_cheap = abs(cheap - full) * 100 // max(full, 1)
        
        print("'" + text + "'")
        print("  Full: " + String(full) + " | V1: " + String(v1) + " (" + String(err_v1) + "%) | V2: " + String(v2) + " (" + String(err_v2) + "%) | Cheap: " + String(cheap) + " (" + String(err_cheap) + "%)")
    
    print("")
    var err_total_v1 = abs(total_v1 - total_full) * 100 // max(total_full, 1)
    var err_total_v2 = abs(total_v2 - total_full) * 100 // max(total_full, 1)
    var err_total_cheap = abs(total_cheap - total_full) * 100 // max(total_full, 1)
    
    print("--- Overall Accuracy ---")
    print("Full tokenizer: " + String(total_full))
    print("V1 estimator:  " + String(total_v1) + " (" + String(err_total_v1) + "% error)")
    print("V2 estimator:  " + String(total_v2) + " (" + String(err_total_v2) + "% error)")
    print("Cheap estimator: " + String(total_cheap) + " (" + String(err_total_cheap) + "% error)")
    
    # Speed comparison
    print("")
    print("--- Speed Comparison ---")
    var rounds = 10000
    
    var t0 = time.perf_counter()
    var full_count = 0
    for _ in range(rounds):
        for text in test_texts:
            var enc = tok.encode(text)
            full_count += len(enc.ids)
    var t1 = time.perf_counter()
    var full_time = (t1 - t0) * 1e6 / Float64(rounds)
    
    t0 = time.perf_counter()
    var v1_count = 0
    for _ in range(rounds):
        for text in test_texts:
            v1_count += est_v1.estimate(text)
    t1 = time.perf_counter()
    var v1_time = (t1 - t0) * 1e6 / Float64(rounds)
    
    t0 = time.perf_counter()
    var v2_count = 0
    for _ in range(rounds):
        for text in test_texts:
            v2_count += est_v2.estimate(text)
    t1 = time.perf_counter()
    var v2_time = (t1 - t0) * 1e6 / Float64(rounds)
    
    print("Full tokenizer: " + String(full_time) + " us/batch")
    print("V1 estimator:   " + String(v1_time) + " us/batch (" + String(full_time / v1_time) + "x faster)")
    print("V2 estimator:   " + String(v2_time) + " us/batch (" + String(full_time / v2_time) + "x faster)")
    
    print("")
    print("--- Summary ---")
    print("V1 accuracy: " + String(100 - err_total_v1) + "% | Speedup: " + String(full_time / v1_time) + "x")
    print("V2 accuracy: " + String(100 - err_total_v2) + "% | Speedup: " + String(full_time / v2_time) + "x")
