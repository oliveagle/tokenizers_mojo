#!/usr/bin/env python3
"""Python wrapper around the Mojo tokenizer bridge (Phase 4).

The Mojo engine is compiled from src/python_api.mojo into a standalone
executable that reads/writes JSON files (Mojo 1.0.0 cannot export
String/Pointer C ABI functions; see docs/adr-0002).  This module hides
that plumbing behind a `TokenizerMojo` class with the HF `tokenizers`
API shape:

    from scripts.tokenizers_mojo_py import TokenizerMojo
    tok = TokenizerMojo("tests/data/test_tokenizer.json")
    ids = tok.encode("hello world")          # [48, 50]
    text = tok.decode(ids)                   # " hello world" (ByteLevel prefix-space)
    print(len(tok), tok.get_vocab_size())

Requires the bridge binary; build it with:

    mojo build -I src src/python_api.mojo -o /tmp/tokenizers_mojo_bridge
"""
import json
import os
import subprocess
import tempfile

REQ_PATH = "/tmp/tokenizers_mojo_req.json"
RESP_PATH = "/tmp/tokenizers_mojo_resp.json"
BRIDGE = os.environ.get(
    "TOKENIZERS_MOJO_BRIDGE", "/tmp/tokenizers_mojo_bridge"
)


class TokenizerMojo:
    """HF-tokenizers-shaped wrapper over the Mojo BPE engine."""

    def __init__(self, path: str, bridge: str = BRIDGE):
        self.path = path
        self.bridge = bridge
        if not os.path.exists(bridge):
            raise FileNotFoundError(
                f"bridge not built: {bridge} (run `mojo build -I src "
                "src/python_api.mojo -o /tmp/tokenizers_mojo_bridge`)"
            )
        self._vocab_size = None
        self._merges = None
        # probe: encode empty string to force load + capture sizes via first call

    def _call(self, payload: dict) -> dict:
        with open(REQ_PATH, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        r = subprocess.run(
            [self.bridge], capture_output=True, timeout=60, text=True
        )
        if r.returncode != 0:
            raise RuntimeError(f"bridge failed: {r.stderr or r.stdout}")
        with open(RESP_PATH) as f:
            resp = json.load(f)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "unknown bridge error"))
        return resp

    def encode(self, text: str) -> list:
        resp = self._call(
            {"op": "encode", "path": self.path, "text": text}
        )
        return resp["ids"]

    def encode_tokens(self, text: str) -> list:
        resp = self._call(
            {"op": "encode", "path": self.path, "text": text}
        )
        return resp["tokens"]

    def decode(self, ids) -> str:
        resp = self._call(
            {"op": "decode", "path": self.path, "ids": list(ids)}
        )
        return resp["text"]

    def get_vocab_size(self) -> int:
        if self._vocab_size is None:
            resp = self._call({"op": "create", "path": self.path})
            self._vocab_size = resp["vocab"]
            self._merges = resp["merges"]
        return self._vocab_size

    def get_merges_count(self) -> int:
        if self._merges is None:
            resp = self._call({"op": "create", "path": self.path})
            self._vocab_size = resp["vocab"]
            self._merges = resp["merges"]
        return self._merges

    def __len__(self) -> int:
        return self.get_vocab_size()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "tests/data/test_tokenizer.json"
    tok = TokenizerMojo(path)
    for s in ["hello world", "how are you", "the quick brown fox jumps over the lazy dog"]:
        ids = tok.encode(s)
        print(f"{s!r} -> {ids}")
        print(f"  decode -> {tok.decode(ids)!r}")
