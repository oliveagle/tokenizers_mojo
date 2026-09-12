# Mojo Tokenizer 详细性能分析

## 热路径微基准测试

| 操作 | 耗时 | 说明 |
|------|------|------|
| `encode_word('hello')` | 0.358 us | 5 字符完整 BPE |
| `merge_rank(h, e)` | 0.029 us | Dict 查找 + String 拼接 |
| `String concat` | 0.007 us | `a + b` 创建新 String |
| `Dict.get` | 0.0002 us | 哈希表查找（极快）|
| `chr(65)` | 0.001 us | 从 codepoint 创建 String |
| `List.append(Int)` | 0.002 us | 整数列表追加 |
| `List.append(String)` | 0.003 us | 字符串列表追加 |
| `codepoints iter` | 0.014 us | 遍历 codepoints |
| `bytes iter` | 0.0000006 us | 遍历字节（极快）|

## 完整 encode 时间分解

| 组件 | 耗时 | 占比 |
|------|------|------|
| Normalize (NFC) | 0.15 us | 2.7% |
| Pre-tokenize | 0.95 us | 17.4% |
| BPE encode_word | 2.77 us | 50.9% |
| 其他开销 | 1.57 us | 28.9% |
| **总计** | **5.44 us** | **100%** |

## Mojo 1.0.0 的限制

无法使用:
- DTypePointer (SIMD/零拷贝)
- 堆数据结构 (O(log n) merge)
- Arena allocator
- 全局 var

## 结论

Mojo 已经比 Python 快 10x+。
V3 优化 (NFC ASCII 快速路径) 额外加速 21%。
