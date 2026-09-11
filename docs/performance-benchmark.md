# 性能对比报告 (Performance Benchmark)

## 测试环境

- **测试日期**: 2026-09-11
- **Mojo 版本**: 1.0.0
- **Python HF tokenizers**: 通过 `pip install tokenizers` 安装 (Rust 后端)
- **测试数据**: `tests/data/test_tokenizer.json` (GPT-2 BPE + ByteLevel, 60 vocab, 31 merges)
- **测试轮次**: 5000 轮，取平均值

## 测试用例

| 输入类型 | 示例 | 长度 |
|---------|------|------|
| Short | "hi", "ok", "go" | 2-3 chars |
| Medium | "hello world", "how are you" | 5-12 chars |
| Long | "the quick brown fox jumps over the lazy dog" | 30-50 chars |
| Mixed | 混合短中长文本 | 混合 |

## 性能对比

### 1. 编码 (Encode)

| 输入长度 | Mojo (us/call) | Python/Rust (us/call) | Mojo 相对速度 |
|---------|----------------|----------------------|---------------|
| Short (2-3 chars) | 33.52 | 2.36 | **14.2x 慢** |
| Medium (5-12 chars) | 36.35 | 4.22 | **8.6x 慢** |
| Long (30-50 chars) | 45.96 | 11.75 | **3.9x 慢** |

### 2. 解码 (Decode)

| 输入长度 | Mojo (us/call) | Python/Rust (us/call) | Mojo 相对速度 |
|---------|----------------|----------------------|---------------|
| Long (30-50 chars) | 32.89 | 2.85 | **11.5x 慢** |

### 3. 词表查找 (Vocab Lookup)

| 操作 | Mojo (ns/call) | Python/Rust (ns/call) | Mojo 相对速度 |
|------|----------------|----------------------|---------------|
| Single lookup | 0.69 | 92.90 | **134x 快** ⚡ |

### 4. 端到端 (Encode + Decode)

| 操作 | Mojo (us/call) | Python/Rust (us/call) | Mojo 相对速度 |
|------|----------------|----------------------|---------------|
| Roundtrip | 70.68 | 7.56 | **9.3x 慢** |

### 5. 总吞吐量

| 实现 | 15000 次 roundtrip 耗时 | 吞吐量 (roundtrips/sec) |
|------|------------------------|------------------------|
| **Mojo** | 1.06 秒 | ~14,200/sec |
| **Python/Rust** | 0.11 秒 | ~132,000/sec |
| **差距** | **9.3x** | Python 快 9.3 倍 |

## 瓶颈分析

### 1. BPE 编码算法 (主要瓶颈)

**当前实现**: O(n² × m) 复杂度

```mojo
# src/bpe.mojo: encode_word()
while n > 1:
    # 1. 线性扫描所有 pair 找最佳 merge → O(n)
    for i in range(n - 1):
        var pair = parts[i] + " " + parts[i + 1]  # String 拼接
        var r = self.merges.get(pair, -1)
        ...
    # 2. 重建 List → O(n)
    var new_parts = List[String]()
    for i in range(n):
        if i != best_idx + 1:
            new_parts.append(parts[i])
```

**HF Rust 实现**: O(n log n) 复杂度
- 使用 **优先队列 (Binary Heap)** 维护最佳 merge pair → O(log n) 提取
- 使用 **链表结构** 实现 O(1) 合并操作
- 使用 **切片/索引** 而非 String 分配

### 2. String 分配开销

```mojo
# 每次 pair 比较都创建新 String
var pair = parts[i] + " " + parts[i + 1]  # 3 次内存分配
```

### 3. 预分词 (Pre-tokenization)

预分词耗时约 32us，与输入长度无关（主要开销在 regex 匹配和字符串操作）。

### 4. 词表查找 (优势)

Mojo 的 `Dict` 实现非常高效，查找速度比 Python 快 **134 倍**。

## 优化建议

### 优先级 1: BPE 算法重构

**目标**: 将 O(n² × m) 降到 O(n log n)

1. **引入优先队列 (Min-Heap)**
   - 维护 (rank, pair_index) 的堆
   - 每次 merge 只需 O(log n) 提取最佳 pair

2. **使用链表/索引结构**
   - 替代 `List[String]` 的重建
   - 合并操作 O(1)

3. **预计算 pair rank**
   - 在 `from_pretrained` 加载时预计算所有可能的 pair rank
   - 避免运行时 `self.merges.get(pair)` 查询

### 优先级 2: 减少 String 分配

1. **使用 (start, end) 索引** 而非 String 切片
2. **复用缓冲区** 避免每轮循环分配

### 优先级 3: 利用 Mojo SIMD 优势

1. **批量 vocab 查找**: 使用 SIMD 并行查找多个 token
2. **UTF-8 编码加速**: 利用 SIMD 快速检测 UTF-8 边界

### 优先级 4: 预分词优化

1. **缓存 regex 编译结果**
2. **使用 slice view** 而非 String 分配

## 实施路线图

### Phase A: BPE 算法优化 (预计提升 4-6x)

```mojo
# 新的 encode_word 实现草案
struct BPESymbol:
    var start: Int  # 原始位置
    var end: Int
    var rank: Int   # 当前最佳 merge rank

struct BPEHeap:
    # Min-heap of (rank, symbol_index)
    var heap: List[Tuple[Int, Int]]
```

### Phase B: 内存优化 (预计提升 1.5-2x)

- 使用 Arena allocator 减少 GC 压力
- 预分配 buffer 避免动态扩容

### Phase C: SIMD 优化 (预计提升 1.2-1.5x)

- 批量 token 查找
- UTF-8 快速扫描

## 目标性能

| 指标 | 当前 | 目标 (Phase A+B) | 目标 (Phase A+B+C) |
|------|------|-----------------|-------------------|
| Encode (Long) | 45.96 us | ~12 us | ~8 us |
| End-to-End | 70.68 us | ~18 us | ~12 us |
| 相对 HF Rust | 3.9x 慢 | **1.0x** | **<1x** |

## 验证方法

每次优化后运行:
```bash
# 正确性验证
pixi run test

# 性能验证
mojo run -I src -I tests examples/bench_comprehensive.mojo
python3 scripts/bench_comprehensive.py
```

确保:
1. 所有测试通过 (正确性不变)
2. 性能指标达到目标
3. 内存分配减少 (可通过 profiling 验证)

---

**结论**: Mojo 实现的主要瓶颈在 BPE 编码算法的 O(n²) 复杂度和频繁的 String 分配。词表查找已经非常高效 (比 Python 快 134x)。通过算法重构和内存优化，预计可以达到与 HF Rust 相当的性能水平。
