# Mojo Tokenizer 性能分析

## 当前基线 (V1)
- **Encode**: ~4000 ns/call (250K encodes/sec)
- **Pre-tokenize**: ~800 ns/call
- **BPE overhead**: ~3200 ns/call

## V2 优化尝试结果

### 1. ByteTables (List[Int] 替代 Dict[Int, String])
- **结果**: 略慢
- **原因**: Dict 对于小数据集（256 entries）已经很高效，List 初始化开销更大

### 2. CompactEncoding (arena + List[Int])
- **结果**: 略慢
- **原因**: Mojo 1.0.0 的 List 操作有额外开销，arena 方案增加了 push 的复杂度

### 3. simple_gpt2_split_v2 (run-length)
- **结果**: 略快
- **原因**: 减少了中间 String 对象的创建

## Mojo 1.0.0 的限制

### 无法使用的技术
1. **DTypePointer**: Mojo 1.0.0 中不可用
2. **SIMD**: 没有 `from sys import simd_width`
3. **全局 var**: 不允许
4. **fn**: 已移除，必须用 `def`
5. **alias**: 已弃用，用 `comptime`

### String 操作的瓶颈
- Mojo String 是不可变的，每次 `+` 都创建新对象
- BPE merge 循环中 `parts[i] + parts[i+1]` 是最大开销
- `Dict.get()` 返回的是值的拷贝，不是引用

## 可行的优化方向

### 1. 跳过不必要的 NFC 规范化
对于纯 ASCII 文本，NFC 是恒等操作。可以检测并跳过。

### 2. 预计算 merge rank 表
将 merges Dict 改为更紧凑的数据结构（如 sorted array + binary search）。

### 3. 批量处理
一次处理多个文本，减少函数调用开销。

### 4. 等待 Mojo 工具链升级
Mojo 2.0+ 预计会有:
- DTypePointer 支持
- SIMD 原语
- 更好的内存管理

## 结论
在 Mojo 1.0.0 的限制下，性能优化空间有限。主要瓶颈是 String 不可变性和缺乏低级内存操作原语。真正的 10x 加速需要:
1. Mojo 2.0+ 的 DTypePointer/SIMD 支持
2. 重写 BPE merge 循环使用字节级操作
3. 零拷贝 Encoding 设计
