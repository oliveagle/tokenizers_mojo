# AGENTS.md

> 这份文档同时面向 AI agent 与人类协作者。它定义了项目目标、目录约定和
> 工作流程。任何对本仓库的改动都应该先读这一页。

## 项目目标

用 [Mojo](https://www.modular.com/mojo) 重写
[HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)，提供一份
高性能、可独立使用的分词器实现。

不做的事情：

- 不重新发明轮子：算法、tokenizer 类型、训练流程都尽量对齐上游
  `huggingface/tokenizers` 的语义与行为。
- 不在主仓库内修改上游实现：上游放在 `submodules/tokenizers/`，只读。

## 参考实现

上游 Rust 仓库已挂为子模块：

```bash
git submodule update --init --recursive
```

阅读时主要关注的目录：

| 上游路径 | 对应职责 |
|---|---|
| `tokenizers/src/tokenizer/` | `Tokenizer` 主结构、`Encoding`、`AddedToken` |
| `tokenizers/src/models/bpe/` | BPE 模型 |
| `tokenizers/src/models/wordpiece/` | WordPiece 模型 |
| `tokenizers/src/models/wordlevel/` | WordLevel 模型 |
| `tokenizers/src/models/unigram/` | Unigram (SentencePiece) 模型 |
| `tokenizers/src/normalizers/` | 文本规范化（NFC、NFK、Lowercase、Strip 等） |
| `tokenizers/src/pre_tokenizers/` | 预分词（ByteLevel、BertPreTokenizer、Metaspace 等） |
| `tokenizers/src/processors/` | 后处理（RobertaProcessing、TemplateProcessing 等） |
| `tokenizers/src/decoders/` | 解码器 |
| `tokenizers/src/trainers/` | 各 Model 对应的训练器 |

跨语言兼容性 / 测试向量（重要参考）：

- `tokenizers/tests/` —— Rust 端到端测试，可作为行为基准
- `bindings/python/pyo3/src/` —— Python API 形态，可作为对外 API 的参考

## 目录约定

```
.
├── AGENTS.md                 本文件
├── README.md                 项目对外介绍
├── .gitignore
├── .gitmodules               子模块声明
├── docs/                     设计、决策、进度
├── src/                      Mojo 源码
├── tests/                    单元/集成测试
├── examples/                 使用示例
└── submodules/
    └── tokenizers/           上游 Rust 实现（只读）
```

约束：

- `submodules/tokenizers/` 内不允许直接改动。如需"补丁"，应在主仓库里写
  patch 脚本或 upstream PR，不进 git 历史。
- `src/` 下每个模块都应有对应 `tests/test_<area>.mojo`；每个公共 API 都必须有测试覆盖。
- 测试运行方式（Mojo 1.0.0 无 `mojo test`）：`bash scripts/run_tests.sh`，
  每个测试文件是独立可执行程序（`mojo run -I src -I tests tests/test_*.mojo`），
  用 `tests/harness.mojo` 里的 `expect()`/`expect_str()`/`expect_eq()` 做断言。

## 工作流程（建议）

1. **先读后写**：改任何模块前，先把 `submodules/tokenizers/` 对应 Rust 源码
   完整读一遍，并确认对应行为测试在 Rust 端如何定义。
2. **小步前进**：每个原子改动单独 commit；commit message 形式
   `<scope>: <imperative>`（例：`bpe: add byte-level pretokenizer`）。
3. **行为一致优先于性能**：先与 Rust 版本做行为对齐（同一输入产出同样的
   token id 序列），再考虑 SIMD/并行优化。
4. **测试**：每个公共 API 都必须有最小可运行测试；Mojo 端测试运行命令待
   工具链确定后写入 `docs/`。
5. **设计决策**：跨模块的取舍（例如"是否需要完全镜像 HF Python API"）必须
   在 `docs/adr-XXXX-<title>.md` 里记录一次 ADR。

## 当前状态

- 仓库已初始化
- 子模块 `submodules/tokenizers` 已添加
- `AGENTS.md`、`README.md`、`docs/README.md` 已写
- `pixi.toml` 已引入（task：check / fmt / test / example-gpt2）
- Phase 1 代码已开始：`src/` 下有 byte_level / bpe / encoding / normalizers / tokenizer
- 测试：`tests/` 下 4 个测试文件全绿；示例 `examples/gpt2_minimal.mojo` 闭环通过

## 已确定的关键决策

> 详细记录见 `docs/adr-0001-stack-and-scope.md` 与 `docs/goals/goals.md`。

1. **构建/包管理**：pixi（`pixi.toml`，Mojo 工具链系统安装于 PATH）。
2. **API 形态**：Mojo-native（`struct` + `def`），不镜像 HF Python API。
3. **第一阶段范围**：BPE Model + ByteLevel PreTokenizer/Decoder +
   NFC Normalizer（Phase 1 恒等占位），跑通 GPT-2 风格最小闭环。

## 待定（后续阶段再敲定）

- [ ] Phase 2 的 Normalizer/PreTokenizer/PostProcessor 扩展优先级
- [ ] 是否引入 trait 抽象以支持多模型运行时切换（Phase 2）
- [ ] Python 互操作方式（Phase 4）
- [ ] 性能优化与基准目标（Phase 3）

## 给 AI agent 的补充说明

- 不要创建未经请求的依赖文件（`pixi.toml`、`mojoproject.toml` 等）—— 等
  上面"待定"清单里至少前 3 项确认后再动。
- 不要直接把上游 Rust 文件复制成 Mojo 文件再翻译；那是反模式。应该先读懂
  数据结构与算法，再按 Mojo 的 idiom 重写。
- 任何对 `submodules/` 的 `git add` / 修改都应该被忽略。
