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

### Phase 4 — 生态（完成）

- **Python 互操作**（file-bridge 策略）：
  - `src/python_api.mojo` → 独立可执行桥接器，JSON 文件双向通信
  - `scripts/tokenizers_mojo_py.py` → `TokenizerMojo` HF-shaped 封装
  - `tests/test_python_interop.py` → 6 句语料 encode+decode 与 HF 一致
  - ADR-0002 更新为"已接受（file-bridge）"；`.so` 直连受阻于
    Mojo 1.0.0 `@export` 限制，留待工具链升级
- 发布管线：版本 0.3.0 + CHANGELOG + README 状态 + pixi tasks
  （`build-bridge` / `test-python`）
