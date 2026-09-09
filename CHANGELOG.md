# Changelog

All notable changes to tokenizers_mojo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-09-10

### Phase 3 — 训练与规模化（完成）

- **BPE Trainer** (`src/bpe_trainer.mojo`): 从语料训练 vocab/merges。
  对齐 HF tokenizers：字母表字符码点排序、merge tie-break 按 (id_a,id_b)、
  `##`/`</w>` 前后缀、`max_token_length` 门限、special tokens /
  initial_alphabet / limit_alphabet / min_frequency。12 项测试。
- **`Tokenizer.from_pretrained`** (`src/from_pretrained.mojo` + 自研
  JSON 解析器 `src/json.mojo`): 读取 HF `tokenizer.json`（BPE +
  ByteLevel），编码 ids 与 HF `Tokenizer.from_file` 完全一致。5 项测试。
- **批量 encode** (`Tokenizer.encode_batch` / `decode_batch`): API 形状
  对齐上游；Mojo 1.0 无 threading 原语，顺序执行。2 项测试。
- **性能基准** (`examples/bench.mojo` + `scripts/bench_rust.py`):
  fixture 10000 encodes — Rust ~8us/enc vs Mojo ~39us/enc (~5x)。

### Phase 4 — 生态（进行中）

- ADR-0002：Mojo 1.0.0 `@export` 不支持参数化 Pointer / String
  ctypes 桥接，Python 互操作 .so 直连暂受阻，标记 `[~]` 留待工具链升级。
