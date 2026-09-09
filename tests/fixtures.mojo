"""Shared inline GPT-2-style fixture used by all test files.

Mirrors the real GPT-2 vocab.json + merges.txt invariant: every byte
that byte-mapping can produce (256 entries) is present in the base
vocab, and every merge result from the merge graph is also a vocab
entry.  This means `BPE.encode()` never encounters an unknown token on
ASCII input covered by the merge graph -- exercising the "happy path"
without external downloads.
"""

import byte_level
import bpe

from byte_level import ByteMapping
from bpe import BPE


def build_gpt2_minimal() raises -> BPE:
    """Return a BPE model exercising the hello/world merge graph."""
    # Base vocab: every byte-mapped character (GPT-2 invariant).
    var m = ByteMapping()
    var vocab = List[String]()
    var b = 0
    while b < 256:
        vocab.append(m.b2u[b])
        b += 1

    # Merge intermediates along the Ġhello + Ġworld paths.
    for t in [
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
    merges.append("h e")  # 10
    merges.append("he l")  # 11
    merges.append("hel l")  # 12
    merges.append("hell o")  # 13
    merges.append("w o")  # 14
    merges.append("wo r")  # 15
    merges.append("wor l")  # 16
    merges.append("worl d")  # 17

    var model = BPE()
    model.load_vocab(vocab)
    model.load_merges(merges)
    return model^
