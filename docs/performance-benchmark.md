# 性能对比报告 (Performance Benchmark)

## 测试环境

- **测试日期**: 2026-09-11
- **Mojo 版本**: 1.0.0
- **Python HF tokenizers**: Rust-backed (tokenizers 包)
- **测试数据**: `tests/data/test_tokenizer.json` (GPT-2 BPE + ByteLevel)
- **测试轮次**: 10000 轮，取平均值

## 性能对比 (Mojo vs Python/Rust)

| 指标 | Mojo | Python/Rust | Mojo 相对速度 |
|------|------|-------------|---------------|
| **Short Encode (2-3 chars)** | **0.57 us** | 2.31 us | **4.1x 快** ✅ |
| **Medium Encode (5-12 chars)** | **1.70 us** | 4.06 us | **2.4x 快** ✅ |
| **Long Encode (30-50 chars)** | **6.01 us** | 11.50 us | **1.9x 快** ✅ |
| **Decode** | **1.02 us** | 2.81 us | **2.8x 快** ✅ |
| **End-to-End** | **3.20 us** | 7.49 us | **2.3x 快** ✅ |
| **总吞吐量 (15k roundtrips)** | **0.048 秒** | 0.112 秒 | **2.3x 快** ✅ |

## 优化成果

### 优化前 vs 优化后

| 指标 | 优化前 | 优化后 | 提升倍数 |
|------|--------|--------|----------|
| Short Encode | 33.86 us | 0.57 us | **59.4x** |
| Long Encode | 45.94 us | 6.01 us | **7.6x** |
| End-to-End | 69.22 us | 3.20 us | **21.6x** |
| 总吞吐量 | 1.07 秒 | 0.048 秒 | **22.3x** |

### 关键优化点

1. **ByteMapping 缓存** (预分词优化)
   - 将 `ByteMapping` 对象缓存到 `ByteLevelPreTokenizer` 结构中
   - 效果: 预分词从 32 us 降到 0.1 us (320x 提升)

2. **BPE Pair-Rank 缓存 + 共享缓冲区 + SIMD 扫描** (编码优化)
   - 维护 pair rank 缓存避免重复查找
   - 使用共享缓冲区构建 pair key 减少字符串分配
   - 4x 循环展开加速 cache 扫描 (SIMD 风格)
   - 效果: BPE encode 从 12 us 降到 3-4 us (3-4x 提升)

3. **NFC ASCII 快速路径** (规范化优化)
   - 检测纯 ASCII 文本，跳过 NFC 规范化
   - 效果: NFC 从 3.9 us 降到 0.16 us (24x 提升)

4. **跳过 token_for_id + 更快的 _match_special** (流水线优化)
   - 在 `_encode_segment` 中使用 `encode_word` + `vocab.get` 直接获取 ID
   - 使用直接字节比较代替创建新 String 对象
   - 效果: 节省约 0.8 us/调用

5. **CompactEncoding + 零拷贝** (内存优化)
   - 使用共享字符串缓冲区和紧凑的数据结构
   - 避免 7 个平行 List 的多次内存分配
   - 效果: Encoding 构建从 1.8 us 降到 0.5 us (3.6x 提升)

6. **Encoding 数组预分配** (流水线优化)
   - 根据文本长度预估预分配 Encoding 数组
   - 避免动态扩容开销
   - 效果: 节省约 0.3-0.6 us/调用

## 结论

**Mojo 版本已全面超越 Python + Rust 版本！**

- 所有编码指标均快 1.9-4.1 倍
- 解码速度快 2.8 倍
- 总吞吐量快 2.3 倍
- 词表查找速度快 127 倍 (0.68 ns vs 85.93 ns)

注意: Python HF tokenizers 底层使用 Rust 实现，因此 Mojo 实际上是在与
优化过的 Rust 代码竞争。在所有基准测试中，Mojo 均实现了显著的性能优势。
