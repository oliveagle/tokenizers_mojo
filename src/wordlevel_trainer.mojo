"""WordLevel Trainer.

Behavior baseline: HuggingFace tokenizers `models/wordlevel/trainer.rs` (upstream Rust).

The WordLevel trainer builds a vocabulary by counting word frequencies
and selecting the most frequent words up to the vocabulary size limit.
"""

import wordlevel

from wordlevel import WordLevel


struct WordLevelTrainer:
    """Trainer for WordLevel models.
    
    Builds vocabulary from word frequency counts.
    """
    
    var vocab_size: Int
    var min_frequency: Int
    var show_progress: Bool
    
    def __init__(out self):
        self.vocab_size = 30000
        self.min_frequency = 0
        self.show_progress = True
    
    def __init__(
        out self,
        vocab_size: Int,
        min_frequency: Int,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.show_progress = True
    
    def train(
        self,
        mut model: WordLevel,
        word_counts: Dict[String, Int],
    ) raises -> List[String]:
        """Train a WordLevel model from word counts.
        
        Returns an empty list (no special tokens added during training).
        """
        # Collect words that meet min_frequency
        var filtered_words = List[String]()
        var filtered_counts = List[Int]()
        
        for word in word_counts.keys():
            var count = word_counts[word]
            if count >= self.min_frequency:
                filtered_words.append(word)
                filtered_counts.append(count)
        
        # Sort by count descending, then by token ascending for determinism
        var n = len(filtered_words)
        for i in range(n):
            var j = i
            while j > 0:
                var should_swap = False
                if filtered_counts[j] > filtered_counts[j - 1]:
                    should_swap = True
                elif filtered_counts[j] == filtered_counts[j - 1]:
                    if filtered_words[j] < filtered_words[j - 1]:
                        should_swap = True
                
                if should_swap:
                    # Swap both arrays
                    var temp_word = filtered_words[j]
                    filtered_words[j] = filtered_words[j - 1]
                    filtered_words[j - 1] = temp_word
                    
                    var temp_count = filtered_counts[j]
                    filtered_counts[j] = filtered_counts[j - 1]
                    filtered_counts[j - 1] = temp_count
                    j -= 1
                else:
                    break
        
        # Build vocabulary from top words
        var vocab = Dict[String, Int]()
        var count = 0
        
        for i in range(len(filtered_words)):
            if count >= self.vocab_size:
                break
            vocab[filtered_words[i]] = count
            count += 1
        
        model.load_vocab_from_dict(vocab)
        
        return List[String]()
