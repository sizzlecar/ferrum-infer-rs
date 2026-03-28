# Marlin INT4xFP16 集成复盘

## 目标

将 Marlin fused INT4xFP16 GEMM kernel（IST Austria 开源）集成到 ferrum-infer-rs，替代 dequant+cuBLAS 两步方案，实现 INT4 量化模型的高效推理。

## 最终结果

| 指标 | FP16 基线 | INT4 Marlin | 提升 |
|------|----------|-------------|------|
| Decode tok/s | 88.1 | **112.4** | +28% |
| TPOT | 11.35ms | **8.90ms** | -22% |
| 显存 | ~8GB | ~2.5GB | -69% |

GPU: RTX PRO 6000 (Blackwell, sm_120, compute 12.0), CUDA 13.1
模型: Qwen3-4B GPTQ-Int4 (JunHowie/Qwen3-4B-GPTQ-Int4)

## 问题时间线

### 问题 1: Marlin 编译需要 `--expt-relaxed-constexpr`

**现象**: nvcc 报错 `calling a constexpr __host__ function("ceildiv") from a __device__ function is not allowed`

**根因**: Marlin 的 `ceildiv` 函数声明为 `constexpr int ceildiv(int a, int b)` 没有 `__device__` 修饰。CUDA 13.1 对此更严格。

**解决**: 改为 `__host__ __device__ constexpr int ceildiv(...)`, 不需要 `--expt-relaxed-constexpr` 编译参数。

**教训**: 第三方 CUDA 代码在新版本编译器上可能有兼容性问题，应该先单独编译验证。

---

### 问题 2: `extern "C"` 缺失导致链接失败

**现象**: `undefined symbol: marlin_cuda`，linker 提示 `did you mean to declare marlin_cuda(...) as extern "C"?`

**根因**: `marlin_cuda_kernel.cu` 中的 `marlin_cuda()` 是 C++ 函数，有 name mangling。Rust FFI 用 `extern "C"` 声明，期望 C linkage。

**解决**: 在 .cu 文件中 `int marlin_cuda(...)` 前加 `extern "C"`。

**教训**: C++ CUDA 文件的函数需要显式 `extern "C"` 才能被 Rust FFI 调用。

---

### 问题 3: CUDA_ERROR_ILLEGAL_INSTRUCTION on Blackwell

**现象**: Marlin kernel 编译成功但运行时 `CUDA_ERROR_ILLEGAL_INSTRUCTION`。所有单独指令测试（mma、cp.async、lop3、96KB shared memory）都通过。

**诊断过程**:
1. ❌ 以为是 mma.sync 指令不兼容 → 测试通过
2. ❌ 以为是 cp.async.cg 不兼容 → 测试通过
3. ❌ 以为是 96KB shared memory 不支持 → 测试通过
4. ❌ 以为是 template + shared memory 组合问题 → 测试通过
5. ✅ `compute-sanitizer --show-backtrace device` 定位到 **line 77**: `cp_async4_stream` 函数

**根因**: Marlin 的 `cp_async4_stream()` 使用了 Ampere 特有的 L2 cache eviction policy 指令:
```cuda
"createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
"cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
```
这两条 PTX 指令在 Blackwell 的 JIT 编译器中无法翻译成有效机器码。

**解决**: 改为不带 cache hint 的普通 `cp.async.cg.shared.global`:
```cuda
"cp.async.cg.shared.global [%0], [%1], %2;\n"
```

**教训**:
- `compute-sanitizer --show-backtrace device -lineinfo` 是定位 CUDA kernel 崩溃的利器
- CUDA PTX 的前向兼容不是 100% 的——特定于某代架构的 PTX 指令可能在新架构上不被支持
- L2 cache hint 是性能优化，去掉只影响 ~1-2% 性能，但保证了跨架构兼容

---

### 问题 4: 编译架构选择

**现象**: `-arch=sm_80` 生成的 SASS 在 Blackwell 上不能运行。`-arch=sm_120` 可以编译但某些指令不兼容。

**解决**: 使用 `-arch=compute_80`（只生成 PTX，不生成 SASS）。GPU driver 在运行时 JIT 编译 PTX 为目标架构的原生代码。

**教训**: 对于需要跨架构兼容的 CUDA kernel，使用 `-arch=compute_XX`（PTX only）而非 `-arch=sm_XX`（SASS only）。

---

### 问题 5: Weight repacking 输出全是乱码

**现象**: Marlin kernel 不再崩溃，但输出全是垃圾 token。速度 90+ tok/s 说明 kernel 在正常计算。

**诊断过程**:

1. **隔离 Marlin vs 其他**: `FERRUM_NO_MARLIN=1` 走 dequant+cuBLAS → 输出正确（15.9 tok/s）。确认问题在 Marlin。

2. **Python 参考验证**: 用 Marlin 原版 Python 代码（`__init__.py` 的 `Layer.pack()`）生成 test data，直接传给 Marlin kernel → 输出正确（128.0, 0.0, 0.0...）。确认 kernel 本身没问题。

3. **Python 模拟 Rust 逻辑**: 在 Python 中完全复现 Rust 的 repacking 逻辑，与原版 Python 逐元素对比 → **完全匹配**。确认 Rust 逻辑正确。

4. **真实模型数据验证**: 用 Python 从 safetensors 加载 o_proj 的 GPTQ qweight，repack 后传给 Marlin → 输出正确（0 NaN, 0 Inf）。

5. **结论**: repacking 正确，问题在 **数据上传或 kernel 调用层面**。

**根因 A — Scale upload 的 transmute**:
```rust
// 错误: CudaSlice<u8> transmute 到 CudaSlice<f16>，len 字段变成字节数而非元素数
let scales = stream.clone_htod(bytemuck::cast_slice::<f16, u8>(&scales_data))?;
let scales_f16: CudaSlice<f16> = unsafe { std::mem::transmute(scales) };
```
`CudaSlice<u8>` 的 len=N*2（字节），transmute 后 `CudaSlice<f16>` 的 len 仍是 N*2（应该是 N）。

**解决**: 直接用 `clone_htod(&[half::f16])`（cudarc 的 f16 feature 支持 `DeviceRepr for half::f16`），不需要 bytemuck 或 transmute：
```rust
let scales_f16 = stream.clone_htod(&scales_data)?;
```

**根因 B — Stream 同步缺失**:
Marlin kernel 在 CudaDecodeRunner 的非阻塞 stream 上执行，但 Marlin 内部使用 CUDA Runtime API（`<<<>>>` 语法）。Runtime API 和 Driver API（cudarc）的 stream 语义不完全兼容，后续操作可能在 kernel 完成前读到 stale 数据。

**解决**: 在每次 `marlin_cuda()` 调用后执行 `stream.synchronize()`：
```rust
stream.synchronize()?;
```

**教训**:
- 永远不要用 `unsafe { transmute }` 在不同元素类型的 CudaSlice 之间转换——用 cudarc 原生支持的类型上传
- CUDA Runtime API 和 Driver API 混用时，stream 同步是必须的
- 先用 Python 参考实现验证数据正确性，再排查调用层问题

---

### 问题 6: Workspace 必须每次调用前清零

**现象**: 第一个 token 可能正确，后续 token 输出错误。

**根因**: Marlin 使用 workspace buffer 作为 mutex locks。第一次调用后 locks 变为非零值，第二次调用时 stale locks 导致 kernel 内部竞态。

**解决**: 在每次 `marlin_cuda()` 调用前执行 `cuMemsetD32(workspace, 0, len)`。

**教训**: 认真阅读第三方 kernel 的 workspace 使用说明。Marlin 的文档说 workspace entries must be "all zero"。

---

### 问题 7: Repacking 的多次错误迭代

在最终正确之前，repacking 经历了以下错误版本：

| 版本 | 错误 | 表现 |
|------|------|------|
| v1 | 没有 transpose [K,N]→[N,K] | 维度混乱，值差 8x |
| v2 | Tile 方向错（[N/16,...] 应该是 [K/16,...]） | 列偏移，线性递进 |
| v3 | 缺少 interleave [0,2,4,6,1,3,5,7] | 90 tok/s 但乱码 |
| v4 | 多做了一步 tile（tile+perm 双重置换） | 同上 |
| v5（正确） | kn[K,N] → tile[K/16,N*16] → perm → pack | 输出正确 |

**最终正确的 repacking 流程**:
```
1. Unpack GPTQ [K/8, N] → [K, N] (kn)
2. Tile: kn.reshape(K//16, 16, N//16, 16).permute(0,2,1,3).reshape(K//16, N*16)
3. Apply _perm (1024-element permutation WITH interleave)
4. Pack: 8 consecutive values per int32
```

**关键发现**: GPTQ 的 [K, N] 布局等价于 Marlin 的 `linear.weight.data.t()`，不需要额外转置。tile 步骤在 _perm **之前**执行，两者缺一不可。_perm 本身还要做一次 interleave: `perm.reshape(-1, 8)[:, [0,2,4,6,1,3,5,7]].ravel()`。

**教训**:
- 对于复杂的数据变换，应该从第一步就用 Python 参考实现验证，而非仅凭理解去写
- 最终的验证方法是：Python 生成参考数据 → 逐元素对比 Rust 输出 → 在 GPU 上测 kernel
- 不要在还没验证 repacking 正确性的情况下调试其他问题

---

## 文件清单

| 文件 | 修改内容 |
|------|---------|
| `kernels/marlin_cuda_kernel.cu` | IST-DASLab 原版 + extern "C" + ceildiv __device__ + 去掉 L2 cache hint |
| `src/marlin.rs` | FFI 声明 + marlin_gemm + repack_gptq_to_marlin + build_marlin_perm + repack_scales_to_marlin + workspace zeroing + stream sync |
| `src/weight_store.rs` | LinearWeight::Marlin 变体 |
| `src/cuda_decode.rs` | linear() dispatch 支持 Marlin |
| `build.rs` | nvcc 编译 Marlin → libmarlin.a，-arch=compute_80 |
| `qwen3.rs` | load_linear_weight 自动选择 Marlin（K%128==0, N%256==0） |
| `gptq_loader.rs` | GPTQ weight fusion (q/k/v → qkv, gate/up → gate_up) |

## 架构决策

1. **Marlin 编译为独立静态库**（不用 bindgen_cuda PTX）：因为 Marlin 使用 CUDA Runtime API（cudaFuncSetAttribute, <<<>>>），不能编译为 PTX。
2. **Feature gate `marlin`**：Marlin 需要 nvcc，不是所有环境都有。Mac/CI 用 `--features cuda`，GPU 环境用 `--features cuda,marlin`。
3. **PTX 前向兼容**（`-arch=compute_80`）：生成 PTX 让 driver JIT 到目标架构，避免绑定特定 SM 版本。
4. **每次调用后 stream sync**：Runtime/Driver API 混用的临时方案。未来可改为全 Driver API（cuLaunchKernel）消除 sync 开销。

## 待优化

1. **去掉 per-call sync**: 改 Marlin 为 cuLaunchKernel 调用，或确保 Runtime/Driver 共享同一 stream context。预期 INT4 Marlin 可达 200+ tok/s。
2. **Scale permutation 优化**: 当前在 CPU 上做，可以移到 GPU。
3. **lm_head 量化**: 当前 lm_head 未量化（保持 FP16）。量化后可进一步减少显存。
