#!/usr/bin/env python3
"""Build tests/gpt2_smoke_data.mojo from real GPT-2 vocab.json + merges.txt.

The generated Mojo module embeds:
  * the smoke-test sentences,
  * HF `tokenizers` reference ids per sentence (add_prefix_space=True),
  * the *closure* subset of GPT-2 vocab/merges needed to BPE-encode those
    sentences (base byte chars + every intermediate merge result), with
    REAL GPT-2 ids and REAL merge ranks,

so the Mojo test runs fully offline and compares our BPE output against
the upstream HuggingFace tokenizers reference.

Regeneration needs network (downloads openai-community/gpt2) and the
Python `tokenizers` package; the committed generated file does not.
"""

import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "tests", "data")
OUT = os.path.join(ROOT, "tests", "gpt2_smoke_data.mojo")

GPT2_BASE = "https://huggingface.co/openai-community/gpt2/resolve/main/"

SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Hello, my world!",
    "This is a simple sentence for the smoke test.",
    "GPT-2 uses byte level tokenization",
    "A short one.",
    "numbers 123 and 456 appear here",
    "The year 2026 is coming soon",
    "OpenAI released GPT-2 in 2019",
]


def download(name):
    dst = os.path.join(DATA_DIR, name)
    if os.path.exists(dst):
        return dst
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(GPT2_BASE + name, dst)
    return dst


def bytes_to_unicode():
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = list(bs)
    seen = set(bs)
    n = 0
    for b in range(256):
        if b not in seen:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def greedy_bpe_with_intermediates(word, merges):
    """Return (final_tokens, set_of_all_intermediate_tokens_in_path)."""
    parts = list(word)
    intermediates = set(parts)
    while len(parts) > 1:
        best_rank, best_idx = -1, -1
        for i in range(len(parts) - 1):
            key = parts[i] + " " + parts[i + 1]
            r = merges.get(key, -1)
            if r != -1 and (best_rank == -1 or r < best_rank):
                best_rank, best_idx = r, i
        if best_idx == -1:
            break
        merged = parts[best_idx] + parts[best_idx + 1]
        parts[best_idx] = merged
        del parts[best_idx + 1]
        intermediates.add(merged)
    return parts, intermediates


def greedy_bpe(word, merges):
    """Return final tokens only (convenience wrapper)."""
    parts, _ = greedy_bpe_with_intermediates(word, merges)
    return parts


def mojo_str(s):
    """Escape a string as a Mojo double-quoted string literal."""
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif o < 0x20:
            out.append(f"\\u{o:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def emit_strings(name, items):
    lines = [f"def {name}() -> List[String]:", "    var out = List[String]()"]
    for it in items:
        lines.append(f"    out.append({mojo_str(it)})")
    lines.append("    return out^")
    lines.append("")
    return "\n".join(lines)


def emit_strings_const(name, value):
    return f"def {name}() -> Int:\n    return {value}\n\n"

def emit_int_lists(name, rows):
    lines = [f"def {name}(i: Int) -> List[Int]:", "    var out = List[Int]()"]
    lines.append("    if i == 0:")
    for row in rows:
        body = ", ".join(str(x) for x in row)
        lines.append(f"        for v in [{body}]: out.append(v)")
        lines.append("        return out^")
        lines.append("    elif i == " + str(rows.index(row) + 1) + ":")
    # The last generated branch is dead code for the final i; clean it up.
    # (We instead generate if/elif chains below.)
    return None


def main():
    vocab_path = download("vocab.json")
    merges_path = download("merges.txt")

    with open(vocab_path) as f:
        vocab = json.load(f)          # token -> id
    with open(merges_path) as f:
        merge_lines = f.read().splitlines()
    merges = {}
    for i, line in enumerate(merge_lines):
        if line.startswith("#version") or not line.strip():
            continue
        merges[line] = i

    b2u = bytes_to_unicode()

    def byte_map(s):
        return "".join(b2u[b] for b in s.encode("utf-8"))

    # Reference ids from HF tokenizers (add_prefix_space=True).
    from tokenizers import Tokenizer
    from tokenizers.pre_tokenizers import ByteLevel

    tok = Tokenizer.from_pretrained("gpt2")
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True, use_regex=True)
    bl = ByteLevel(add_prefix_space=True, use_regex=True)

    ref_ids_all = []
    closure_tokens = set()
    all_words = []
    for s in SENTENCES:
        words = [w for w, _ in bl.pre_tokenize_str(s)]
        all_words.append(words)
        ref = tok.encode(s).ids
        ref_ids_all.append(ref)
        # walk the merge path to collect every intermediate token
        for w in words:
            _, inter = greedy_bpe_with_intermediates(w, merges)
            closure_tokens |= inter

    # Add every base byte-mapped char too (safety net).
    for b in range(256):
        closure_tokens.add(b2u[b])

    # Sort closure tokens by real id so load_vocab_pairs order is stable.
    closure_sorted = sorted(closure_tokens, key=lambda t: vocab[t])
    closure_ids = [vocab[t] for t in closure_sorted]

    # Closure merges: pairs whose operands are both in the closure.
    closure_merges = [
        line for line in merge_lines
        if not line.startswith("#version") and line.strip()
        and line.split(" ")[0] in closure_tokens and line.split(" ")[1] in closure_tokens
    ]
    closure_merges.sort(key=lambda line: merges[line])

    # Sanity: every final token of a word must be in vocab.
    for words, ref in zip(all_words, ref_ids_all):
        ours = []
        for w in words:
            for t in greedy_bpe(w, merges):
                ours.append(vocab[t])
        assert ours == ref, f"reference mismatch for words {words}"

    out = []
    out.append('"""AUTO-GENERATED by scripts/build_gpt2_smoke.py -- DO NOT EDIT.')
    out.append("")
    out.append("Real GPT-2 (openai-community/gpt2) vocab.json + merges.txt closure")
    out.append("subset for the smoke-test sentences below. Every token the greedy")
    out.append("BPE can produce for these sentences is present, with its REAL GPT-2")
    out.append("id and REAL merge rank. Reference ids come from HuggingFace")
    out.append('`tokenizers` 0.22.2 with add_prefix_space=True.')
    out.append('"""')
    out.append("")
    out.append(emit_strings_const("n_sentences", len(SENTENCES)))
    out.append(emit_strings("sentences", SENTENCES))
    out.append(emit_strings("vocab_tokens", closure_sorted))
    out.append(emit_strings("merge_lines", closure_merges))

    ids_body = []
    ids_body.append("def ref_ids(i: Int) -> List[Int]:")
    ids_body.append("    var out = List[Int]()")
    for i, row in enumerate(ref_ids_all):
        if i == 0:
            ids_body.append("    if i == 0:")
        else:
            ids_body.append("    elif i == " + str(i) + ":")
        body = ", ".join(str(x) for x in row)
        ids_body.append(f"        for v in [{body}]: out.append(v)")
        ids_body.append("        return out^")
    ids_body.append("    return out^")
    ids_body.append("")
    out.append("\n".join(ids_body))

    # vocab_ids as a single flat list aligned with vocab_tokens
    vids = ", ".join(str(x) for x in closure_ids)
    out.append(f"def vocab_ids() -> List[Int]:")
    out.append("    var out = List[Int]()")
    out.append(f"    for v in [{vids}]: out.append(v)")
    out.append("    return out^")
    out.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(out))

    print(f"wrote {OUT}")
    print(f"  sentences: {len(SENTENCES)}")
    print(f"  closure vocab tokens: {len(closure_sorted)}")
    print(f"  closure merges: {len(closure_merges)}")
    print(f"  ref ids: {[len(r) for r in ref_ids_all]}")


if __name__ == "__main__":
    main()
