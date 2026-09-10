"""WordPiece Trainer.

Behavior baseline: HuggingFace tokenizers `models/wordpiece/trainer.rs` (upstream Rust).

The WordPiece trainer is essentially a BPE trainer with:
- Default `continuing_subword_prefix` set to "##"
- Same algorithm as BPE training

This is a thin wrapper around BpeTrainer.
"""

import bpe
import bpe_trainer
import wordpiece

from bpe import BPE
from bpe_trainer import BpeTrainer, TrainerConfig
from wordpiece import WordPiece


struct WordPieceTrainer:
    """Trainer for WordPiece models.
    
    This is a thin wrapper around BpeTrainer with "##" as the default
    continuing_subword_prefix.
    """
    
    var bpe_trainer: BpeTrainer
    
    def __init__(out self):
        var config = TrainerConfig(
            vocab_size=30000,
            min_frequency=2,
            limit_alphabet=0,
            continuing_subword_prefix="##",
            end_of_word_suffix="",
            max_token_length=0,
        )
        self.bpe_trainer = BpeTrainer(config)
    
    def __init__(
        out self,
        vocab_size: Int,
        min_frequency: Int,
        limit_alphabet: Int,
    ):
        var config = TrainerConfig(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            continuing_subword_prefix="##",
            end_of_word_suffix="",
            max_token_length=0,
        )
        self.bpe_trainer = BpeTrainer(config)
    
    def train(
        mut self,
        mut model: WordPiece,
        word_counts: Dict[String, Int],
    ) raises -> List[String]:
        """Train a WordPiece model from word counts.
        
        Returns the list of special tokens added to the vocabulary.
        """
        # WordPiece training is identical to BPE training with "##" prefix
        var temp_bpe = BPE()
        self.bpe_trainer.train(word_counts, temp_bpe)
        
        # Transfer vocab from BPE to WordPiece
        model.vocab = Dict[String, Int]()
        model.id_to_token = Dict[Int, String]()
        
        for token in temp_bpe.vocab.keys():
            var id = temp_bpe.vocab[token]
            model.vocab[token] = id
            model.id_to_token[id] = token
        
        return List[String]()
