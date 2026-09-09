# ADR-0002：Phase 4 Python 互操作策略（Mojo 1.0.0 FFI 限制）

- **状态**：已接受（Accepted）
- **日期**：2026-09-10
- **决定人**：oliveagle
- **相关**：`docs/goals/goals.md` Phase 4；ADR-0001

## 背景

Phase 4 目标之一是"Python 互操作封装（`Python` interop / `.so`），供
`transformers` 加载"。调研 Mojo 1.0.0（本机 pip 安装的 stripped stdlib）
后发现两条路径均受工具链限制。

## 探索结论（2026-09-10 实测）

1. **`@export` + `--emit shared-lib` + ctypes 直连**
   - `@export` 仅对 **非参数化、纯 `Int`/数值** 签名的函数可用。
   - `Pointer[UInt8]` 是参数化类型（`origin` 参数），`@export` 直接报错
     "can not be applied on parametric functions"。
   - `String` 参数虽能编译，但从 Python ctypes 按其胖指针 ABI
     （ptr, len, cap）调用会在 Mojo 侧 `alloc failed` 崩溃。
   - 结论：**当前工具链无法可靠做字符串进出的 Python↔Mojo 直连**。
2. **`std.python` 互操作模块**
   - 存在且能 `from std.python import Python`，但方向是 **Mojo → Python**
     （Mojo 调 Python 模块），不是 Python → Mojo。对"供 transformers 加载"
     不适用。
3. **子进程驱动（stdin/stdout 或文件）**
   - Mojo 1.0.0 无 `env`/`sys.argv`/`os.args` 模块；stdio 亦无暴露。
     不可直接复用项目现有 `main(args)` 模式。文件 I/O（`open`/`read`）
     可用。

## 决策

- **不**在 Phase 4 内强行实现 Python↔Mojo 直连 .so（受 1.0.0 限制，
  强做是坏 API）。
- 记录上述限制为已知约束；当 Mojo 工具链放开 `@export` 对
  String/Pointer 的支持后，再补 `.so` 直连封装。
- **采用文件桥接方案**（2026-09-10 追加）：`src/python_api.mojo` 编译
  为独立可执行，stdin/stdout/argv 缺失改用固定路径 JSON 文件
  （`/tmp/tokenizers_mojo_{req,resp}.json`）双向通信。Python 侧
  `scripts/tokenizers_mojo_py.py` 封装 `TokenizerMojo` 类
  （`encode/decode/get_vocab_size/get_merges_count`），API 形状对齐
  HF `tokenizers`。`tests/test_python_interop.py` 验证 6 句语料
  encode+decode 与 HF 完全一致。
- Phase 4 Python 互操作**已完成**（file-bridge 策略）；
  `.so` 直连仍标记为 Mojo 工具链升级后的改进方向。
