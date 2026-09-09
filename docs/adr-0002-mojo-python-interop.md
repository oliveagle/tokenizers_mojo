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
- Phase 4 的"发布管线"先行（版本、CHANGELOG、CI 文档），
  Python 互操作标记为 `[~] 受阻于 Mojo 1.0.0 FFI` 留待工具链升级。
