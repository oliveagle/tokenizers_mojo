# 项目目标（Goals）

> 状态：本文档定义 tokenizers_mojo 的长期方向、阶段目标与验收标准。
> 所有开发工作（包括 AI agent）都应以本文档为准绳。状态字段：
> `[ ] 未开始` / `[~] 进行中` / `[x] 已完成`

---

## 1. 总目标（Mission）

用 **Mojo** 语言重写 [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)，
提供一份**高性能、可独立使用、语义与上游对齐**的分词器实现。

- 不重新发明算法：tokenizer 类型、训练流程、行为语义均对齐上游
  `huggingface/tokenizers`（Rust）。
- 上游实现只读挂在 `submodules/tokenizers/`，作为唯一行为基准。

## 2. 已确定的技术决策

| # | 决策 | 结论 | 记录 |
|---|------|------|------|
| 1 | 构建 / 包管理 | **pixi**（`pixi.toml`，Mojo 工具链系统安装于 PATH） | ADR-0001 |
| 2 | 对外 API 形态 | **Mojo-native**（`struct` + `def/static` 方法），不镜像 Python API | ADR-0001 |
| 3 | 第一阶段范围 | **BPE Model + NFC Normalizer + ByteLevel PreTokenizer/Decoder**，跑通 GPT-2 风格最小闭环 | ADR-0001 |

## 3. 分阶段目标

### Phase 0 — 项目地基 `[x]`
- [x] 空仓库初始化，挂载 tokenizers 子模块（`submodules/tokenizers`）
- [x] 目录骨架（`src/` `tests/` `examples/` `docs/`）
- [x] `pixi.toml`（task 别名：check / fmt / test / example-gpt2）
- [x] `AGENTS.md` + `README.md`
- [x] 本文档（目标文件）
- [x] ADR-0001（记录上述 3 个技术决策）

### Phase 1 — GPT-2 风格最小闭环（BPE + ByteLevel）`[x]`
目标：从「原始文本」到「token ids」再到「解码回文本」的完整链路跑通，
并用**内联小型 vocab/merges 测试夹具**验证，不依赖外网下载 GPT-2 权重。

- [x] `ByteLevel` PreTokenizer（字节↔unicode 映射 + GPT-2 分词）
- [x] `ByteLevel` Decoder（逆映射 + UTF-8 还原）
- [x] `BPE` Model（vocab/merges 加载 + 贪心 merge 编码）
- [x] `NFC` Normalizer（Phase 1 以恒等实现占位，标注 TODO）
- [x] `Encoding` 结构（ids / tokens / offsets / masks）
- [x] `Tokenizer` 主结构（normalize → pretokenize → model → encode）
- [x] 单元测试：`tests/test_byte_level.mojo`、`test_bpe.mojo`、`test_encoding.mojo`、`test_tokenizer.mojo`（42 项全绿）
- [x] 示例：`examples/gpt2_minimal.mojo`（roundtrip `' hello world'` 通过）
- [x] 用真实 GPT-2 `vocab.json` + `merges.txt` 做一次端到端冒烟验证
      （对比 HF `tokenizers` / `transformers` 输出，确认 ids 一致）
      — 见 `scripts/build_gpt2_smoke.py` + `tests/test_gpt2_smoke.mojo`

**Phase 1 验收标准**
1. [x] `pixi run test` 全绿（无外网依赖；44 项测试）。
2. [x] 对同一段英文输入，编码出的 ids 与上游 Rust `tokenizers` 完全一致
      （8 个真实句子，闭包子集 401 tokens / 4076 merges，ids 逐位匹配 HF）。
3. [x] `decode(encode(text))` 在 add_prefix_space 语义下与上游一致（含前置空格行为）。
4. [x] 所有公共类型为 `struct` + 方法，无 Python 类型渗入对外 API。

### Phase 2 — 对齐 & 扩展 `[~]`
- [x] 补全 `Normalizer`：Lowercase（ASCII/Latin-1/Greek/Cyrillic）、Strip、
      LowercaseStrip 组合；真实 NFC/NFKC/NFD/NFKD 仍为 TODO
- [x] 补全 `PreTokenizer`：`Whitespace`（`\w+|[^\w\s]+` 语义）、`Metaspace`
      （▁ 替换 + MergedWithNext 切分）、`Split`（字面量分隔符 × 4 种
      behavior，均与 HF 一致）、`BertPreTokenizer`（空白 removed + 标点
      isolated，含 Unicode P 类别）；正则 Split 待做
- [x] `PostProcessor`：RobertaProcessing（`<s>`...`</s>`）、BertProcessing
      （`[CLS]`...`[SEP]`）、TemplateProcessing（`$A/$B/$0/$1` +
      显式 `:type_id` 后缀 + `[SPECIAL]:N` 全支持，9 项 HF 参考
      测试覆盖 single/pair/无特符/`$0`/`$1`/显式 type/未知 token）
- [x] `AddedToken` 机制（content/special/single_word/lstrip/rstrip 结构，
      并已接入 Tokenizer：`add_special_token` + 最长匹配前置切分，
      `special_tokens_mask` 正确标记，ids 与 HF 一致）
- [x] 完整 `Encoding` 字段（新增 sequence_ids、push_special、type_ids 批量）；
      overflowing 待做
- [x] 引入 trait/接口抽象（`src/traits.mojo`：`Normalizer` / `PreTokenizer` /
      `Model` / `Decoder`），所有现有 struct 显式 conform，并用泛型
      `pipeline[N: Normalizer, P: PreTokenizer, M: Model]` 验证运行时多实现

### Phase 3 — 训练与规模化 `[ ]`
- [ ] BPE Trainer（从语料训练 vocab/merges）
- [ ] `Tokenizer.from_pretrained(...)` 直接读取 HF `tokenizer.json`
- [ ] 并行 / 批量 encode（多线程，参照上游 rayon 思路）
- [ ] 性能基准：与 Rust 版对比（`pixi run bench`）

### Phase 4 — 生态 `[ ]`
- [ ] Python 互操作封装（`Python` interop / `.so`），供 `transformers` 加载
- [ ] 发布管线（版本、CI、文档）

## 4. 非目标（明确不做）

- 不维护一个与上游功能不同的"分叉"算法——差异即 bug，除非记录 ADR 并给出理由。
- 不直接把 Rust 源码逐行翻译成 Mojo——先读懂算法，再按 Mojo idiom 重写。
- 不在 `submodules/` 里做任何改动或提交。
- Phase 1 不追求速度（O(n²) merge 先求正确），性能优化放到 Phase 3。

## 5. 对齐基线

上游参照版本：`submodules/tokenizers/` 当前锁定 commit
`6cfd9d385ca0ed91c10b49f0ce97d02cfde1b607`（node-v0.7.0-1032）。

行为对齐判定：对同一输入文本，**输出的 token ids 序列**与上游一致即视为通过；
`decode` 还原文本一致。

## 6. 进展跟踪

- 每完成一个阶段，更新本文档勾选状态，并在 commit message 中引用 `docs/goals/goals.md`。
- 2026-09-10：TemplateProcessing 完成（解决 Mojo 1.0.0
  `Copyable`/ImplicitlyCopyable 约束，统一以 `[X]:N` / `$X:N` 后缀语法
  处理 sequence 与 special token 的显式 type_id）。
- 跨模块决策记入 `docs/adr-*.md`；组件设计记入 `docs/design-*.md`。
