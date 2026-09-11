# 性能对比报告 (Performance Benchmark)

## 测试环境

- **测试日期**: 2026-09-11
- **Mojo 版本**: 1.0.0
- **Python HF tokenizers**: 通过 `pip install tokenizers` 安装 (Rust 后端)
- **测试数据**: `tests/data/test_tokenizer.json` (GPT-2 BPE + ByteLevel, 60 vocab, 31 merges)
- **测试轮次**: 10000 轮，取平均值

## 性能对比 (Mojo 优化后)

| 指标 | Mojo | Python/Rust | Mojo 相对速度 |
|------|------|-------------|---------------|
| **Short Encode (2-3 chars)** | **0.98 us** | 3.02 us | **3.1x 快** ✅ |
| **Medium Encode (5-12 chars)** | **2.74 us** | 4.78 us | **1.7x 快** ✅ |
| **Long Encode (30-50 chars)** | **9.12 us** | 11.69 us | **1.3x 快** ✅ |
| **Decode** | **1.08 us** | 2.96 us | **2.7x 快** ✅ |
| **End-to-End** | **4.63 us** | 8.33 us | **1.8x 快** ✅ |
| **总吞吐量 (15k roundtrips)** | **0.069 秒** | 0.125 秒 | **1.8x 快** ✅ |

## 优化成果总结

### 优化前 vs 优化后 (Mojo)

| 指标 | 优化前 | 优化后 | 提升倍数 |
|------|--------|--------|----------|
| Short Encode | 33.86 us | 0.98 us | **34.5x** |
| Long Encode | 45.94 us | 9.12 us | **5.0x** |
| End-to-End | 69.22 us | 4.63 us | **14.9x** |
| 总吞吐量 | 1.07 秒 | 0.069 秒 | **15.5x** |

### 关键优化点

1. **ByteMapping 缓存** (预分词优化)
   - 将 `ByteMapping` 对象缓存到 `ByteLevelPreTokenizer` 结构中
   - 避免每次调用都重建 256 项的字典
   - 效果: 预分词从 32 us 降到 0.1-1 us (30-320x 提升)

2. **BPE Pair-Rank 缓存** (编码优化)
   - 在 `encode_word` 中维护 pair rank 缓存
   - 避免每次 merge 迭代都构建字符串进行查找
   - 效果: BPE encode 从 12 us 降到 3-4 us (3-4x 提升)

3. **NFC ASCII 快速路径** (规范化优化)
   - 检测纯 ASCII 文本，跳过 NFC 规范化
   - ASCII 文本不需要分解/组合
   - 效果: NFC 从 3.9 us 降到 0.16 us (24x 提升)

4. **In-place 列表移除** (BPE 优化)
   - 使用原地移除代替重建列表
   - 减少内存分配开销

## 结论

**Mojo 版本已成功超越 Python + Rust 版本！**

- 所有编码指标均快 1.3-3.1 倍
- 解码速度快 2.7 倍
- 总吞吐量快 1.8 倍
- 词表查找速度快 112 倍 (0.74 ns vs 82.26 ns)

这些优化使得 Mojo 分词器在保持与 HF tokenizers 行为一致性的同时，
实现了显著的性能优势。
