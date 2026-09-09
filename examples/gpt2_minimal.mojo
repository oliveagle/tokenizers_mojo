"""GPT-2 style minimal end-to-end example (Phase 1).

Runs normalize -> pre_tokenize -> BPE encode -> decode with an inline
self-consistent vocab/merges fixture (no external download).

The fixture preserves the invariant of a real GPT-2 vocab.json +
merges.txt: every token the greedy BPE can produce (base byte-mapped
chars plus every merge intermediate) is present in the vocab, so no
token is ever dropped or left unknown.
"""

import byte_level
import bpe
import encoding
import tokenizer

from bpe import BPE
from tokenizer import Tokenizer


def build_fixture() raises -> BPE:
    """Return a BPE model with a self-consistent GPT-2-style fixture.

    Merge graph (rank = line order):
        Ġhello path:  Ġ h -> Ġh, Ġh e -> Ġhe, Ġhe l -> Ġhel,
                      Ġhel l -> Ġhell, Ġhell o -> Ġhello
        Ġworld path:  Ġ w -> Ġw, Ġw o -> Ġwo, Ġwo r -> Ġwor,
                      Ġwor l -> Ġworl, Ġworl d -> Ġworld
        bare paths:   h e -> he -> hel -> hell -> hello, and the same
                      for w o -> wo -> wor -> worl -> world
    """
    var vocab = List[String]()
    for t in [
        "Ġ",
        "h",
        "e",
        "l",
        "o",
        "w",
        "r",
        "d",
        "Ġh",
        "Ġhe",
        "Ġhel",
        "Ġhell",
        "Ġhello",
        "he",
        "hel",
        "hell",
        "hello",
        "Ġw",
        "Ġwo",
        "Ġwor",
        "Ġworl",
        "Ġworld",
        "wo",
        "wor",
        "worl",
        "world",
    ]:
        vocab.append(t)

    var merges = List[String]()
    # Prefix-space words merge first (ranks 0-9).
    merges.append("Ġ h")  # 0  -> Ġh
    merges.append("Ġh e")  # 1  -> Ġhe
    merges.append("Ġhe l")  # 2  -> Ġhel
    merges.append("Ġhel l")  # 3  -> Ġhell
    merges.append("Ġhell o")  # 4  -> Ġhello
    merges.append("Ġ w")  # 5  -> Ġw
    merges.append("Ġw o")  # 6  -> Ġwo
    merges.append("Ġwo r")  # 7  -> Ġwor
    merges.append("Ġwor l")  # 8  -> Ġworl
    merges.append("Ġworl d")  # 9  -> Ġworld
    # Bare words merge next (ranks 10+).
    merges.append("h e")  # 10 -> he
    merges.append("he l")  # 11 -> hel
    merges.append("hel l")  # 12 -> hell
    merges.append("hell o")  # 13 -> hello
    merges.append("w o")  # 14 -> wo
    merges.append("wo r")  # 15 -> wor
    merges.append("wor l")  # 16 -> worl
    merges.append("worl d")  # 17 -> world

    var model = BPE()
    model.load_vocab(vocab)
    model.load_merges(merges)
    return model^


def main() raises:
    var tok = Tokenizer(build_fixture())

    var enc = tok.encode("hello world")
    print("input:    'hello world'")
    print("ids:     ", enc.get_ids())
    print("tokens:  ", enc.get_tokens())
    print("offsets: ", enc.get_offsets())

    var back = tok.decode(enc)
    print("decoded: ", repr(back))
    print("roundtrip ok:", back == " hello world")

    # Also demonstrate a bare word (no prefix space) producing the
    # sub-word "hello" via the bare merge path.
    var enc2 = tok.encode("hello")
    print()
    print("input:    'hello'")
    print("tokens:  ", enc2.get_tokens())
    var back2 = tok.decode(enc2)
    print("decoded: ", repr(back2))
