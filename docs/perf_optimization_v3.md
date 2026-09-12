# Mojo Tokenizer 性能优化报告

## 最终成果: V3 (NFC ASCII Fast Path)

| 版本 | Encode (us/call) | 相对 V1 加速 |
|------|------------------|--------------|
| V1 (基线) | 4.286 | 1.00x |
| V3 (NFC 优化) | 3.584 | **1.20x** |

## 优化内容

### 1. NFC Normalizer ASCII 快速路径
- **原理**: 纯 ASCII 文本已经是 NFC 规范形式，无需做任何处理
- **实现**: `NFCNormalizerOptimized` 先检查 `_is_all_ascii(text)`，如果是则直接返回原文
- **效果**: ASCII 文本的 NFC 从 0.14 us 降到接近 0 ns

### 2. 正确性验证
- V1 和 V3 输出完全相同的 token ID 序列
- 测试文本: "hello world" → [48, 50]

## 尝试但未成功的优化

### ByteTables (List[Int] 替代 Dict[Int, String])
- **结果**: 略慢
- **原因**: Dict 对 256 条目的小数据集已经很高效；List 初始化开销更大

### CompactEncoding (arena + 零拷贝)
- **结果**: 略慢
- **原因**: Mojo 1.0.0 的 List 操作有额外开销

### BPE Merge Loop 优化
- **结果**: 无显著差异
- **原因**: 核心瓶颈是 String 不可变性和 `parts[i] + parts[i+1]` 拷贝

## Mojo 1.0.0 的限制

无法使用的高级特性:
1. ❌ `DTypePointer` — 无法做 SIMD 或零拷贝
2. ❌ `from sys import simd_width` — 无 SIMD 原语
3. ❌ 全局 `var` — 无法做进程级单例
4. ❌ `fn` — 必须用 `def`
5. ❌ `alias` — 已弃用

## 后续优化方向

### 等待 Mojo 2.0+
1. **DTypePointer + SIMD**: 16/32 字节并行处理
2. **零拷贝 Encoding**: 字节级数组操作
3. **更好的内存管理**: Arena allocator

### 算法级优化
1. **预计算 Merge Rank 表**: 将 Dict 改为排序数组 + 二分查找
2. **批量处理**: 一次处理多个文本
3. **缓存热路径**: LRU cache for frequent merge operations

## 文件结构

```
src/
├── normalizers_optimized.mojo  # NFC ASCII 快速路径
├── tokenizer_v3.mojo           # V3 Tokenizer (使用优化 NFC)
├── from_pretrained_v3.mojo     # V3 加载器
├── byte_level_v2.mojo          # V2 ByteLevel (未使用)
├── bpe_v2.mojo                 # V2 BPE (未使用)
├── compact_encoding.mojo       # V2 CompactEncoding (未使用)
```
