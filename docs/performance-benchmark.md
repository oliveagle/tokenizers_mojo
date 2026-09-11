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
| **Short Encode (2-3 chars)** | **0.85 us** | 2.31 us | **2.7x 快** ✅ |
| **Medium Encode (5-12 chars)** | **2.13 us** | 4.06 us | **1.9x 快** ✅ |
| **Long Encode (30-50 chars)** | **7.18 us** | 11.50 us | **1.6x 快** ✅ |
| **Decode** | **1.07 us** | 2.81 us | **2.6x 快** ✅ |
| **End-to-End** | **3.79 us** | 7.49 us | **2.0x 快** ✅ |
| **总吞吐量 (15k roundtrips)** | **0.057 秒** | 0.112 秒 | **2.0x 快** ✅ |

## 优化成果

### 优化前 vs 优化后

| 指标 | 优化前 | 优化后 | 提升倍数 |
|------|--------|--------|----------|
| Short Encode | 33.86 us | 0.85 us | **39.8x** |
| Long Encode | 45.94 us | 7.18 us | **6.4x** |
| End-to-End | 69.22 us | 3.79 us | **18.3x** |
| 总吞吐量 | 1.07 秒 | 0.057 秒 | **18.8x** |

### 关键优化点

1. **ByteMapping 缓存** (预分词优化)
   - 将 `ByteMapping` 对象缓存到 `ByteLevelPreTokenizer` 结构中
   - 效果: 预分词从 32 us 降到 0.1-1 us (30-320x 提升)

2. **BPE Pair-Rank 缓存 + 共享缓冲区** (编码优化)
   - 维护 pair rank 缓存避免重复查找
   - 使用共享缓冲区构建 pair key 减少字符串分配
   - 效果: BPE encode 从 12 us 降到 3-4 us (3-4x 提升)

3. **NFC ASCII 快速路径** (规范化优化)
   - 检测纯 ASCII 文本，跳过 NFC 规范化
   - 效果: NFC 从 3.9 us 降到 0.16 us (24x 提升)

4. **跳过 token_for_id 查找** (流水线优化)
   - 在 `_encode_segment` 中使用 `encode_word` + `vocab.get` 直接获取 ID
   - 避免了额外的 `id_to_token` 字典查找
   - 效果: 节省约 0.5 us/调用

5. **Encoding 数组预分配** (内存优化)
   - 在 encode 开始时预分配所有平行数组
   - 避免动态扩容开销
   - 效果: 节省约 0.3 us/调用

## 瓶颈分析

当前主要瓶颈:

1. **BPE encode_word** (~4 us for Long input)
   - O(n²) 扫描找最佳 merge pair
   - 每次 merge 需要字符串拼接
   - 优化方向: 优先队列 (O(n log n))，但 Mojo 1.0 的内存管理限制了效果

2. **Encoding push** (~1.5 us)
   - 每个 token 需要 7 次 append 操作
   - 优化方向: 使用结构化数组或批量写入

3. **预分词** (~1 us)
   - 字符分类和字符串分割
   - 已经优化到接近理论极限

## 结论

**Mojo 版本已全面超越 Python + Rust 版本！**

- 所有编码指标均快 1.6-2.7 倍
- 解码速度快 2.6 倍
- 总吞吐量快 2.0 倍
- 词表查找速度快 121 倍 (0.71 ns vs 85.93 ns)

注意: Python HF tokenizers 底层使用 Rust 实现，因此 Mojo 实际上是在与
优化过的 Rust 代码竞争。在所有基准测试中，Mojo 均实现了显著的性能优势。
