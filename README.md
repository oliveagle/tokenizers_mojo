# tokenizers_mojo

用 [Mojo](https://www.modular.com/mojo) 重写
[HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)。

> 状态：Phase 1（GPT-2 风格最小闭环）进行中。`src/` 下已有
> ByteLevel PreTokenizer/Decoder、BPE Model、NFC 占位、Encoding、
> Tokenizer 编排；测试与示例可运行。

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
pixi run fmt       # mojo format src tests examples
pixi run test      # 运行 tests/ 下全部测试
pixi run example-gpt2  # 运行 examples/gpt2_minimal.mojo
```

> 注意：本仓库用的 Mojo 1.0.0 没有 `mojo check` / `mojo test` 命令，
> `check` 用 `mojo build src/typecheck.mojo` 代替，`test` 用
> `scripts/run_tests.sh`（每个测试文件是独立可执行程序）代替。
