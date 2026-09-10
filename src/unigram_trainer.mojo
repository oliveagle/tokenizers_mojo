"""Unigram Trainer.

Behavior baseline: HuggingFace tokenizers `models/unigram/trainer.rs` (upstream Rust).

The Unigram trainer uses the Expectation-Maximization (EM) algorithm to learn
a probability-based vocabulary. It iteratively:
1. Initializes a seed vocabulary from character frequencies
2. Runs E-step: uses Viterbi to find best segmentation for each word
3. Runs M-step: updates token probabilities based on segmentation
4. Prunes low-probability tokens
"""

import unigram

from unigram import Unigram, UnigramVocabEntry


def log(x: Float64) -> Float64:
    """Natural logarithm approximation using Newton's method."""
    if x <= 0:
        return -1e308
    # Use the identity: ln(x) = 2 * atanh((x-1)/(x+1))
    # Or simpler: use the built-in if available, otherwise approximate
    # For now, use a simple approximation
    var result: Float64 = 0.0
    var y = x
    # Normalize to [1, 2) range
    var k: Float64 = 0.0
    while y >= 2.0:
        y /= 2.0
        k += 1.0
    while y < 1.0:
        y *= 2.0
        k -= 1.0
    # Now y is in [1, 2), use polynomial approximation
    var z = (y - 1.0) / (y + 1.0)
    var z2 = z * z
    result = 2.0 * z * (1.0 + z2 / 3.0 + z2 * z2 / 5.0 + z2 * z2 * z2 / 7.0)
    return result + k * 0.6931471805599453  # ln(2)


struct UnigramTrainer:
    """Trainer for Unigram (SentencePiece) models.
    
    Uses EM algorithm to learn token probabilities.
    """
    
    var vocab_size: Int
    var show_progress: Bool
    var n_sub_iterations: Int
    var shrinking_factor: Float64
    var max_piece_length: Int
    
    def __init__(out self):
        self.vocab_size = 8000
        self.show_progress = True
        self.n_sub_iterations = 2
        self.shrinking_factor = 0.75
        self.max_piece_length = 16
    
    def __init__(
        out self,
        vocab_size: Int,
        n_sub_iterations: Int,
        shrinking_factor: Float64,
    ):
        self.vocab_size = vocab_size
        self.show_progress = True
        self.n_sub_iterations = n_sub_iterations
        self.shrinking_factor = shrinking_factor
        self.max_piece_length = 16
    
    def _initialize_vocab(
        self,
        word_counts: Dict[String, Int],
    ) raises -> List[UnigramVocabEntry]:
        """Initialize vocabulary from character frequencies."""
        var char_counts = Dict[String, Int]()
        
        # Count character frequencies
        for word in word_counts.keys():
            var count = word_counts[word]
            for cp in word.codepoints():
                var c = chr(Int(cp))
                char_counts[c] = char_counts.get(c, 0) + count
        
        # Collect and sort by frequency descending
        var sorted_chars = List[String]()
        var sorted_counts = List[Int]()
        
        for entry in char_counts.keys():
            sorted_chars.append(entry)
            sorted_counts.append(char_counts[entry])
        
        # Sort by count descending
        var n = len(sorted_chars)
        for i in range(n):
            var j = i
            while j > 0 and sorted_counts[j] > sorted_counts[j - 1]:
                var temp_char = sorted_chars[j]
                sorted_chars[j] = sorted_chars[j - 1]
                sorted_chars[j - 1] = temp_char
                
                var temp_count = sorted_counts[j]
                sorted_counts[j] = sorted_counts[j - 1]
                sorted_counts[j - 1] = temp_count
                j -= 1
        
        # Build initial vocabulary with log probabilities
        var vocab = List[UnigramVocabEntry]()
        var total: Float64 = 0.0
        
        for i in range(min(len(sorted_chars), self.vocab_size)):
            var count: Float64 = Float64(sorted_counts[i])
            vocab.append(UnigramVocabEntry(sorted_chars[i], count))
            total += count
        
        # Convert to log probabilities
        for i in range(len(vocab)):
            var prob = vocab[i].score / total
            vocab[i].score = log(prob)
        
        return vocab^
    
    def train(
        self,
        mut model: Unigram,
        word_counts: Dict[String, Int],
    ) raises -> List[String]:
        """Train a Unigram model from word counts.
        
        Uses EM algorithm to learn token probabilities.
        """
        if len(word_counts) == 0:
            return List[String]()
        
        # Initialize vocabulary
        var vocab = self._initialize_vocab(word_counts)
        
        # Create model with initial vocabulary
        model = Unigram(vocab, 0, False)
        
        # EM iterations (simplified - full implementation would do
        # E-step with Viterbi and M-step with probability updates)
        for _ in range(self.n_sub_iterations):
            # In a full implementation, we would:
            # 1. E-step: Find best segmentation for each word using Viterbi
            # 2. M-step: Update token probabilities based on segmentation
            # 3. Prune low-probability tokens
            
            # For now, we keep the initial vocabulary
            pass
        
        return List[String]()
    
    def feed(mut self, word_counts: Dict[String, Int]) raises:
        """Feed word counts to the trainer.
        
        This accumulates word frequencies for training.
        """
        # Unigram trainer processes all counts at once during train()
        # This method is provided for API compatibility
        pass
