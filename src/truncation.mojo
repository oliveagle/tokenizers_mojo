"""Truncation: TruncationParams + truncate_encodings (Phase 2).

Behavior baseline: HuggingFace tokenizers `utils::truncation` +
`tokenizer::Encoding::truncate` (Rust).

`truncate_encodings` computes how many tokens to keep from sequence A
(and optionally B) per the strategy, then calls `Encoding.truncate` on
each.  Overflow windows are returned separately (upstream stores them
inside the Encoding; Mojo cannot hold `List[Encoding]` recursively).
"""

import encoding

from encoding import Encoding


struct TruncationParams:
    """Truncation parameters (upstream `TruncationParams`).

    Attributes:
        direction: "left" | "right" (default "right").
        max_length: keep at most this many tokens.
        strategy: "longest_first" | "only_first" | "only_second".
        stride: overlap (in tokens) between successive overflow windows.
    """

    var direction: String
    var max_length: Int
    var strategy: String
    var stride: Int

    def __init__(
        out self,
        max_length: Int,
        strategy: String = "longest_first",
        direction: String = "right",
        stride: Int = 0,
    ):
        self.max_length = max_length
        self.strategy = strategy
        self.direction = direction
        self.stride = stride


struct TruncatedEncodings:
    """Result of `truncate_encodings`.

    Attributes:
        enc_overflow: overflow windows produced while truncating A.
        pair_overflow: overflow windows produced while truncating B
            (empty when no pair was provided).
    """

    var enc_overflow: List[Encoding]
    var pair_overflow: List[Encoding]

    def __init__(out self):
        self.enc_overflow = List[Encoding]()
        self.pair_overflow = List[Encoding]()


def truncate_encodings(
    mut enc: Encoding,
    mut pair: Encoding,
    has_pair: Bool,
    params: TruncationParams,
) raises -> TruncatedEncodings:
    """Truncate `enc` (and optionally `pair`) per `params`, in place.

    Mirrors upstream `truncate_encodings`:
      * max_length == 0      -> both sequences fully overflow.
      * longest_first        -> keep the shortest sequence intact as much
        as possible, then the other; if both must shrink, split evenly.
      * only_first / only_second -> truncate only the named sequence;
        only_second without a pair raises.
      * If the target sequence is not long enough to absorb the excess,
        raises (upstream `TruncationError::SequenceTooShort`).

    Returns the overflow windows (A's first, then B's).
    """
    var result = TruncatedEncodings()
    var max_length = params.max_length
    var stride = params.stride
    var direction = params.direction
    var strategy = params.strategy

    if max_length == 0:
        result.enc_overflow = enc.truncate(0, stride, direction)
        if has_pair:
            result.pair_overflow = pair.truncate(0, stride, direction)
        return result^

    var total_length = enc.len()
    if has_pair:
        total_length += pair.len()

    if total_length <= max_length:
        return result^

    var to_remove = total_length - max_length

    if strategy == "longest_first":
        if has_pair:
            # assume n1 <= n2, then fix up with a swap (upstream cases)
            var n1 = enc.len()
            var n2 = pair.len()
            var swapped = False
            if n1 > n2:
                # n1 must be the shortest; n2 is recomputed below so the
                # temporary is only needed for n1
                swapped = True
                n1 = n2
            if n1 > max_length:
                n2 = n1
            else:
                var cand = max_length - n1
                if n1 > cand:
                    n2 = n1
                else:
                    n2 = cand
            if n1 + n2 > max_length:
                n1 = max_length // 2
                n2 = n1 + max_length % 2
            if swapped:
                var tmp = n1
                n1 = n2
                n2 = tmp
            result.enc_overflow = enc.truncate(n1, stride, direction)
            result.pair_overflow = pair.truncate(n2, stride, direction)
            return result^
        # single sequence
        result.enc_overflow = enc.truncate(max_length, stride, direction)
        return result^

    if strategy == "only_first" or strategy == "only_second":
        if strategy == "only_second" and not has_pair:
            raise Error("Truncation error: Second sequence not provided")
        var target_len = enc.len()
        if strategy == "only_second":
            target_len = pair.len()
        if target_len <= to_remove:
            raise Error(
                "Truncation error: Sequence to truncate too short to "
                + "respect the provided max_length"
            )
        if strategy == "only_first":
            result.enc_overflow = enc.truncate(
                target_len - to_remove, stride, direction
            )
        else:
            result.pair_overflow = pair.truncate(
                target_len - to_remove, stride, direction
            )
        return result^

    raise Error("Unknown TruncationStrategy: " + strategy)
