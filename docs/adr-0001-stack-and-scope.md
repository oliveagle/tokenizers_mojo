# ADR-0001：技术栈、API 形态与第一阶段范围

- **状态**：已接受（Accepted）
- **日期**：2026-09-10
- **决定人**：oliveagle
- **相关**：`docs/goals/goals.md`

## 背景

仓库 `tokenizers_mojo` 要用 Mojo 重写 HuggingFace Tokenizers。启动前必须
敲定三件事，任何一项都会影响后续所有代码的形态：

1. 构建 / 包管理工具怎么选？
2. 对外 API 长什么样？是模仿 HF Python `tokenizers` 的 API，还是
   Mojo 原生风格？
3. 第一阶段先支持哪些组件？

## 决策

### 决策 1：构建 / 包管理 —— **pixi**

- 采用 `pixi.toml` 作为项目元数据与任务入口（`check` / `fmt` / `test` /
  `example-gpt2`）。
- **Mojo 工具链系统级安装**（`~/.local/bin/mojo`，当前 1.0.0），
  pixi 只做元数据与 task 封装，不通过 conda-forge 拉 Mojo。
- 理由：.pixi 本身轻量，能覆盖 task 封装需求；直接走 pixi 下的 conda
  MoJo 环境目前不稳定，且 mojo 独立安装已经能工作。后期若 mojo 需要
  版本矩阵管理，可再补 `[feature]` / 环境定义。

### 决策 2：对外 API 形态 —— **Mojo-native**

- 公共 API 采用 `struct` + `def` 方法（构造用 `out self`，可变方法
  `mut self`，只读方法无 self 修饰）。
- **不**镜像 HF Python `tokenizers` 的 `Tokenizer` 类 API（即
  `Tokenizer(models.BPE(...)).encode("...")` 这种动态类组合风格）。
- 具体做法：`BPE` / `ByteLevel` / `NFC` 等是**独立的 Mojo struct**，由
  `Tokenizer` 组合持有，而不是通过 trait 对象或注册表动态切换。
- 理由：Mojo-native 更贴近语言习惯（值语义、无 GIL、零开销抽象），
  编译期分派更直接。保留一个 `Tokenizer` 组合体即可覆盖 Phase 1 需求。
  若未来需要运行时切换组件，再引入 trait（Mojo trait）。

### 决策 3：第一阶段范围 —— **BPE + NFC + ByteLevel**

- 目标：跑通一个 **GPT-2 风格最小闭环**：
  `text → normalize → byte-level pretokenize → BPE encode → ids`
  以及反向 `ids → byte-level decode → text`。
- 选 BPE + ByteLevel 的理由：GPT-2、RoBERTa、GPT-NeoX、Llama（词表层面）
  都是 byte-level BPE；BPE 是最通用的 tokenizer 模型；ByteLevel 的
  字节映射表（OpenAI 的 `bytes_to_unicode`）是标准且封闭的 256 映射，
  不依赖 Unicode 属性表。
- NFC Normalizer 在 Phase 1 **用恒等实现占位**（`return text`），
  标注 TODO；真正实现放到 Phase 2，理由是 Unicode 归一化需要完整的
  Unicode 数据表，先不动它不妨碍闭环跑通。
- 训练器（Trainer）明确**不在** Phase 1 范围内。Phase 1 只做
  `from_file` / 从内存加载 vocab+merges 并编码。

## 后果

- **优点**：三件事都有明确答案；项目可以立即动工。
- **代价**：Python 用户不能直接 `from tokenizers_mojo import Tokenizer`
  的 HF 兼容 API；Phase 4 再补互操作。
- **风险**：Mojo 1.0.0 的 stdlib 很小（`std.json` / `std.fs` 均不存在），
  vocab.json 解析需要借助 Python 互操作（`from std.python import Python`
  + `json` 模块），这在决策 1 "不通过 pixi 拉 Mojo"的前提下可接受。
- **回滚路径**：若后续发现 BPE + ByteLevel 不足以对齐某类模型，可追加
  新 ADR 扩展第一阶段范围。

## 验证

以 `docs/goals/goals.md` 里 Phase 1 的验收标准为准（`pixi run test`
全绿 + ids 与上游一致 + decode 还原一致）。
