# SM120 sparse FP4 attention: checked Host and GPU region producers

The versioned C ABI is in
[`sage-vsa-fp4`](../../../crates/ferrum-kernels/native/sage-vsa-fp4/abi.h).
Its Rust descriptors and allocation-free list validator are in
[`sage_vsa_fp4`](../../../crates/ferrum-native-ops/src/sage_vsa_fp4.rs).
The explicit `cuda-sage-vsa-fp4` feature selects this locked artifact. Its
managed provider is constructed with one `Arc<CheckedHostVsaPattern>` and an
explicit opaque-module allowance. This is an immutable Host-pattern profile;
pooled-QK computation and full FastH3 integration remain separate.
Source compilation alone establishes no GPU, quality or performance result.

Profile 1 fixes D128, 128×128 blocks, noncausal attention, unsmoothed Q,
zero delta-S, two-level P quantization, F32 accumulation and BF16 output.
Q/K/V are already quantized E2M1 with group-16 E4M3 scale bytes in the
original, distinct physical Q/K/V layouts. Ordinary contiguous E2M1 or the
NVFP4 matrix provider's scale layout are not interchangeable with these bytes.
K smoothing, FP8 prefix attention, pooled selection and compression remain
separate graph work; this entry cannot establish
their equivalence to the fixed FastVideo baseline.

The separate `quantize_*_v1` ABI prepares all six packed tensors from BF16
Q `[B,Q,H*128]` and K/V `[B,K,H*128]`. Positive Q/K lengths are multiples of
128; actual sequence-major element strides are explicit and checked. It does
not slice prefix queries, center K, compute pooled scores or provide a global
scale. Its unchanged upstream kernel bodies use group-16 amax/6 rounded to
E4M3, then inverse scale (zero maps to zero) and E2M1 RN satfinite conversion
under the source's fast-math policy. Q is row-major, K permutes rows within
groups of 32, and V transposes after quantizing groups along sequence rows.
Scale bytes use the original 64-row/4-column swizzle, distinct from GEMM.

Quantization Plan records three 1024-thread dispatches and 32768 bytes of
static shared memory for the V transpose; Q/K require no shared memory. It
checks compiled function limits/occupancy without enqueueing. All nine spans
have exact extents and input/output alias checks; there is no scratch tensor,
allocation, Host readback or implicit stream. A later-launch error may follow
an enqueued earlier launch, so all owners remain retained to completion or
quarantine. Attention and quantization acquire the same admitted Sage module
slot with the same source/config identity. The isolated quantization ABI has
compiled with CUDA 12.8.1 for SM120a. Subsequent RTX 5090 arithmetic and managed
consumer qualification, including the integrated FastH3 pipeline, is recorded
in the [video goal](../../../docs/goal-fast-h3-video-5090.zh.md#9-当前进度与下一步).
Those records distinguish operator subsets from full-pipeline and quality claims.

The fixed source is FastVideo revision
`866a3cdb91a85aef8561847365fc8564add9e4c8`, under
`fastvideo-kernel/attn_qat_infer/blackwell`. The deployment's CUTLASS headers
report 4.3.0; they have no Git metadata. Its `include/cutlass/version.h`
SHA256 is `9e055289347d9caa7284d8f11357fc2fd0188593820a08cdab286d1c243a4bec`.
This digest identifies the version header only. The compiler-observed full
header closure must be locked before building; a version string alone is
insufficient, and the existing NVFP4 GEMM CUTLASS 4.7.1 closure is different.

## Ownership and validation

Plan and Validate never enqueue. Plan checks the current device against the
declared SM120 device, uses the compiled `Traits::SharedStorage` size and
records 384 threads, the actual SM count, three H2D uploads and one memset.
The static scheduler's grid is exactly the SM count. Integer bounds cover
the final scheduler advance, LSE stride and CuTe's int32 shape arithmetic.
Every byte extent is exact, positive and addressable.

Q blocks and KV blocks are independent. Lists have `[B,H,Qblocks,KVblocks]`
indices and `[B,H,Qblocks]` counts; KV half-valid lengths have `[2*KVblocks]`
elements shared across batches/heads. Counts are positive, active indices
strictly descend and end in block 0. Each tile has 1–128 valid tokens at its
beginning. These restrictions make the first visited block nonempty and
exclude duplicate/out-of-range accesses. Inactive slots are never interpreted.
A full `[B,H,KVblocks,KVblocks]` list must be compacted **per batch/head**
when dense prefix query blocks are removed. A single pointer offset is wrong
for multiple heads. The caller supplies the already compacted lists in its
admitted staging; this ABI creates no extra host allocation for compaction.

`CheckedCpuBlockLists` proves content and extent only. Its borrowed storage
can be ordinary Rust memory and is therefore not automatically upload-ready.
Native Validate and Run require CUDA pinned-Host pointer kinds for all three
sources, check exact lengths/alignment, then read the complete active lists.
Run always repeats validation and copies those lists to its explicit device
**write** spans; arbitrary device I32 tensors cannot bypass validation. A
GPU-produced mask uses the separate region entry described below; there is no
boolean that skips checking external device lists.

All device spans must belong to the declared current device. Output, LSE,
zero delta, indices, counts and half-valid destinations are pairwise disjoint
and cannot overlap Q/K/V, scales or Host upload sources. Runtime pointer-kind
queries cannot establish allocation extents, immutability or resource leases:
the unsafe raw caller guarantees those facts. Core must retain pinned Host
staging, all device resources and native module ownership until completion,
including errors after the first submitted copy. Native code never allocates,
pins, synchronizes or chooses an implicit stream. Stream capture is rejected.

The managed operation publishes only BF16 Y. The original noncausal epilogue
comments out its normal LSE writes (`epilogue_tma_ws.h:148–165`), so the ABI's
LSE span is reserved in invocation scratch and never exposed or read back.
Scratch also holds zero-delta and all three metadata destinations, including
16-byte section padding. No uninitialized LSE is presented as a numerical
result, and this profile cannot provide logsumexp for an attention merge.

`CheckedHostVsaPattern` consumes three existing immutable `ManagedHostBytes`
with TransferStaging charges. Their physical padding remains charged by the
caller's Host account. Provider preparation uses the runtime's existing
bounded pinned-upload registry; it neither copies nor allocates another
Host payload. The provider, Core prepared retention and encoded command keep
the same owner. Preparation binds shape, profile and SHA256; encoding also
checks owner identity, so serializing a digest cannot create valid metadata.
Native upload errors use the existing command quarantine and retain all
payloads and device destinations. This first factory binds one pattern per
registry and does not implement a dynamic GPU mask producer.

The separate `region_topk` operation has an actual GPU producer and managed
consumer, qualified on SM120 for the bounded routing fixture below. It accepts
the six packed tensors plus contiguous original F32 scores `[B,H,Ktiles,Ktiles]`.
It does **not** yet compute pooled QK scores or quantize QKV. `CheckedVsaRegions`
retains two existing charged immutable Host allocations: contiguous
`[start,end,keep]` region triples and every key tile's half-valid counts. Regions
cover the video suffix without holes/overlap; the nonempty prefix is dense and
query blocks are exactly the suffix. Its digest binds shape, geometry bytes and
the public numerical identity. It does not certify any GPU score value.

`region_*_v1` Plan/Validate enqueue nothing. Run rechecks pinned Host contents,
uploads geometry into two admitted scratch spans, zeroes delta, generates each
head/query's lists, and calls the unchanged attention launch on the same explicit
stream. Indices/counts are private writes, never input ports. The shared provider
keeps the original module ledger, prepared owner/identity checks and command
completion/quarantine path for both producers. The region plan records two
uploads, one memset and two kernel dispatches. Scratch is the existing five
sections plus the region table, each aligned to 16 bytes; LSE stays unpublished.

The separate `region_kernel_*_v1` transport accepts the same checked immutable
Host geometry without page locking. Each bounded 2048-byte packet is copied by
value into a CUDA kernel argument and writes the original admitted scratch
span. No asynchronous command retains a pageable Host pointer. Its distinct
240-byte plan records the exact packet-kernel count; the provider reports those
launches plus selection/attention and one memset, with no DMA uploads. It uses
the same selection/attention helper, numerical profile, scratch accounting and
Core completion/quarantine owners. The original pinned `region_*_v1` ABI stays
available. CUDA parameters are copied from Host values as documented by
[NVIDIA](https://docs.nvidia.com/cuda/archive/12.3.1/pdf/CUDA_Driver_API.pdf);
the complete packet is below even the older 4096-byte parameter limit. The
[managed SM120 tests](../../../crates/ferrum-kernels/tests/cuda_sparse_fp4_attention.rs)
cover both producers, including geometry crossing a packet boundary, against
an independent Q=K=0, dyadic-V oracle. Complete 480p audio/video runs also
preserved their corresponding output hashes. These checks do not establish
general softmax equivalence or an end-to-end performance improvement.

Selection profile `region-topk.f32.ieee-radix-descending-index-ascending.v1` ranks
scores descending, breaks equal radix-key ties by smaller key index, puts +0
above -0, orders infinities normally and places every NaN first. This follows
the F32 conversion in the deployed Torch commit
[`7661cd9c6b84` TopKTypeConfig](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/cuda/SortingRadixSelect.cuh#L25).
It does not establish every top-k launch algorithm's cutoff tie choice. Integer
IEEE keys preserve subnormals despite the attention TU's fast-math flag.
`-inf` is legal; finite-domain numerical qualification does not prove upstream
scores always finite. Lists are compacted in descending key order. The first
kernel uses exact quadratic rank comparisons within each region; it establishes
a simple correctness boundary, with no speed claim. Torch `topk` does not supply
a stable tie policy here, so formal model comparison must count cutoff ties and
assess their selected-set/output effect, including tied real workloads.

CPU tests cover geometry, the independent sort oracle, nonfinite/tie behavior,
original Host padding/ownership and exact scratch extents. The managed GPU test
changes scores between invocations, includes multiple regions, 130 key tiles,
cutoff ties and nonfinite routing. Its Q=K=0 dyadic attention oracle only checks
selection/routing/ownership; it cannot establish general two-level-P arithmetic,
real pooled-score precision, end-to-end quality or performance.

The 2026-11-13 RTX 5090 run completed both this dynamic fixture and the original
Host fixture with process exit 0. Dynamic `(B,H,Qtiles,Ktiles)` cases
`(2,3,4,5)`, `(1,2,4,6)` and `(1,1,129,130)`, each with two score inputs, compared
5,275,648 BF16 values exactly with the independent uniform-attention oracle.
The second inputs had respectively 4, 0 and 129 cutoff-tied regions; this
exercises the declared tie policy without establishing Torch's selected set.
The earlier combined run retained exit 101 because the two fixtures declared
different runtime fingerprints for one live device; the rerun declares both
capabilities consistently and leaves the production capacity guard unchanged.
Receipts, source identities and all failed attempts are outside the repository
under `vsa-dynamic-r1-stable/validation-r3-r6.json` in the task evidence root.

The delta table's physical size is `[B,H,1,KV]`, despite logical shape
`[128,KV,H,B]`. Original `kernel_traits.h:143` defines its atom with stride
`(0,1)`, and `mainloop_tma_ws.h:238` builds the layout from that atom, ignoring
the nominal DS argument stride. For `BlockMean=false`, lines 537–543 select
query block 0. Plan verifies `cute::cosize(layout)` against the exact physical
byte span. This preserves `api.py:121–136` without claiming a 128-row buffer.

## Build the observed source closure

Keep sources, logs, depfiles, archives and receipts outside the repository.
Materialize a fresh source root with:

* `kernels/{abi.h,boundary.cu,boundary_checks.cuh,launch.cuh,region_abi.h,region_boundary.cuh,region_select.cuh}` from Ferrum;
* `kernels/{quantize_abi.h,quantize_boundary.cuh,quantize_core.cuh}` from Ferrum;
* `kernels/sage/params.h` from Ferrum's adapted parameter header;
* the ten unchanged upstream headers named in `upstream-headers.sha256` under
  `kernels/sage/`, preserving their notices;
* the deployment's CUTLASS headers under `cutlass/include` and
  `cutlass/tools/util/include`.

The parameter header removes unused Torch generator state. Original attention
numerical headers stay unchanged; the quantization header retains the original
numerical bodies with a namespace and explicit CUDA includes. The launch and
quantization wrappers use checked spans. Do not copy the Torch API, original
launch header, Torch quantization wrapper, generated source trees or binaries into
this repository. Ship the Apache and BSD license texts and NOTICE listed in
the package definition; CUTLASS/FlashAttention/Sage notices stay in source.

Use CUDA >=12.8 with `sm_120a`, C++17, O3, relaxed constexpr, extended lambda
and **fast math**, as declared by `source.seed.json`. The source's historical
CTA256/QBLKSIZE macros remain recorded; actual compiled traits use 384
threads. Torch's half-disabling macros are absent in this standalone build.
The seed is intentionally incomplete and is not a locked build plan.

Run NVCC `-M` on `kernels/boundary.cu` with the same include roots, defines,
architecture and numerical flags. Feed its actual depfile to the existing
Rust builder:

```text
ferrum-native-ops-builder lock-source-dependencies
  --definition <source.seed.json> --source-root <source>
  --depfile <boundary.d> --out-definition <observed.json> --out-plan <plan.json>
ferrum-native-ops-builder source-build
  --plan <plan.json> --source-root <source> --compute-capability sm_120
  --builder-sha <builder-source-id> --nvcc <nvcc> --cuda-toolkit-root <toolkit>
  --ccbin <host-cxx> --ar <ar> --object-cache <shared-native-cache> --out <build>
```

The normal source builder hashes the observed CUTLASS/Sage/Ferrum header
closure and toolchain inputs. Use its receipt and plan for packaging, with
`package-definitions/sage-vsa-fp4.json` and the current CPU-generated provider
catalog (`sage_vsa_fp4_catalog_declaration` example). Compilation and export
verification establish the source/ABI build boundary only. GPU tests must
subsequently cover rectangular Q/K, multiple heads/batches, partial tile
halves, sparse visitation, pinned staging lifetime and failure retirement,
then compare numerical results against the fixed implementation before any
model-quality or speed claim.
