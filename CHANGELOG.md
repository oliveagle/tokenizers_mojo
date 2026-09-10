# Changelog

All notable changes to tokenizers_mojo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.7.0] - 2026-09-10

### Phase 5 — 模型扩展（完成）

- **from_pretrained 扩展**: 新增 `from_pretrained_wordpiece`、
  `from_pretrained_wordlevel`、`from_pretrained_unigram` 函数，
  支持从 HF tokenizer.json 加载 WordPiece/WordLevel/Unigram 模型。
- **WordPiece Tokenizer** (`src/wordpiece_tokenizer.mojo`): WordPiece 模型的
  tokenizer 封装，支持 encode/decode 和 special token。
- **WordLevel Tokenizer** (`src/wordlevel_tokenizer.mojo`): WordLevel 模型的
  tokenizer 封装，支持 encode/decode 和 special token。
- **Unigram Tokenizer** (`src/unigram_tokenizer.mojo`): Unigram 模型的
  tokenizer 封装，支持 encode/decode 和 special token。
- **测试**: 新增 `test_from_pretrained_wordpiece.mojo`、
  `test_from_pretrained_wordlevel.mojo`、`test_from_pretrained_unigram.mojo`。

## [0.6.0] - 2026-09-10

### Phase 5 — 模型扩展（部分完成）

- **WordPiece Trainer** (`src/wordpiece_trainer.mojo`): BPE 训练 + "##" 前缀。
  对齐 HF tokenizers：thin wrapper around BpeTrainer。6 项测试。
- **WordLevel Trainer** (`src/wordlevel_trainer.mojo`): 词频统计 + 排序。
  对齐 HF tokenizers：按频率降序排列，相同频率按字典序。6 项测试。
- **Unigram Trainer** (`src/unigram_trainer.mojo`): EM 算法初始化。
  对齐 HF tokenizers：字符频率初始化 + log 概率。6 项测试。

## [0.5.0] - 2026-09-10

### Phase 5 — 模型扩展（部分完成）

- **Unigram Model** (`src/unigram.mojo`): SentencePiece 风格的概率子词分词算法。
  对齐 HF tokenizers：Viterbi 最优路径搜索、概率词汇表、UNK 惩罚、
  lattice 构建。9 项测试。工作了 Mojo 1.0.0 的 `mut self` 编译器问题，
  将 Viterbi 实现提取为独立函数。
- **WordPiece Model** (`src/wordpiece.mojo`): BERT/DistilBERT 风格的子词分词算法。
  对齐 HF tokenizers：greedy longest-match-first、`##` 继续子词前缀、`[UNK]` 未知词处理、
  `max_input_chars_per_word` 限制。10 项测试。
- **WordLevel Model** (`src/wordlevel.mojo`): 整词查找模型。
  对齐 HF tokenizers：词汇表直接查找、`<unk>` 未知词回退。9 项测试。

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
