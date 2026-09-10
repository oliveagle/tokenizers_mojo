# tokenizers_mojo

用 [Mojo](https://www.modular.com/mojo) 重写
[HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)。

> 状态：Phase 1–5 全部完成（`docs/goals/goals.md`）。
> GPT-2 最小闭环、Normalizer/PreTokenizer/PostProcessor/AddedToken 对齐、
> BPE Trainer、`from_pretrained`（自研 JSON parser）、批量 encode、
> 性能基准均已落地，**46+ 项测试全绿**。
> 已实现 BPE、WordPiece、WordLevel、Unigram 四种模型及其训练器。

## 快速链接

- [`AGENTS.md`](./AGENTS.md) — 项目目标、目录约定、开发流程、待定问题
- [`docs/`](./docs/) — 设计文档、决策记录、进度
- [`submodules/tokenizers/`](./submodules/tokenizers) — 上游 Rust 实现（只读参考）

## 初始化

```bash
git submodule update --init --recursive
```

## 开发命令（pixi）

```bash
pixi run check      # 编译验证整个 src/ 树
pixi run fmt        # mojo format src tests examples
pixi run test       # 运行 tests/ 下全部测试（23 文件全绿）
pixi run example-gpt2  # 运行 examples/gpt2_minimal.mojo
pixi run bench        # Phase 3 性能基准（Mojo）
python3 scripts/bench_rust.py  # 对照 Rust/HF 基线
```

## 真实 GPT-2 冒烟测试

`tests/test_gpt2_smoke.mojo` 用 **真实 GPT-2 `vocab.json` + `merges.txt`
的闭包子集**（401 tokens / 4076 merges，覆盖 8 句英文样本的所有 merge 路径）
验证端到端 ids 与 HuggingFace `tokenizers` 一致。数据模块
`tests/gpt2_smoke_data.mojo` 由 `scripts/build_gpt2_smoke.py` 离线生成（重新生成
需要联网下载 `openai-community/gpt2`，不需要时无需联网）。

> 注意：本仓库用的 Mojo 1.0.0 没有 `mojo check` / `mojo test` 命令，
> `check` 用 `mojo build src/typecheck.mojo` 代替，`test` 用
> `scripts/run_tests.sh`（每个测试文件是独立可执行程序）代替。

## 已实现的模型

| 模型 | 状态 | 测试 | 说明 |
|------|------|------|------|
| **BPE** | ✅ 完成 | 13 项 | GPT-2/GPT-3/Roberta 风格 |
| **WordPiece** | ✅ 完成 | 10 项 | BERT/DistilBERT 风格 |
| **WordLevel** | ✅ 完成 | 9 项 | 整词查找模型 |
| **Unigram** | ✅ 完成 | 9 项 | SentencePiece 风格（T5/ALBERT） |

## 已实现的训练器

| 训练器 | 状态 | 测试 | 说明 |
|--------|------|------|------|
| **BPE Trainer** | ✅ 完成 | 12 项 | 迭代 pair 计数 + 合并 |
| **WordPiece Trainer** | ✅ 完成 | 6 项 | BPE 训练 + ## 前缀 |
| **WordLevel Trainer** | ✅ 完成 | 6 项 | 词频统计 + 排序 |
| **Unigram Trainer** | ✅ 完成 | 6 项 | EM 算法 + 概率词汇 |

## 已实现的组件

| 组件 | 状态 | 说明 |
|------|------|------|
| **Normalizer** | ✅ | NFC/NFKC/NFD/NFKD/Lowercase/Strip |
| **PreTokenizer** | ✅ | ByteLevel/Whitespace/Metaspace/Split/Bert/Regex |
| **PostProcessor** | ✅ | TemplateProcessing/RobertaProcessing/BertProcessing |
| **Decoder** | ✅ | ByteLevel |
| **from_pretrained** | ⚠️ | 仅支持 BPE 模型 |

## 已知限制

- `from_pretrained` 仅支持 BPE 模型（`Tokenizer` 结构体硬编码为 BPE）
- Unigram Trainer 使用简化 EM 算法（完整实现需要 Viterbi + 概率更新）
- Mojo 1.0.0 无 threading 原语，训练器顺序执行
