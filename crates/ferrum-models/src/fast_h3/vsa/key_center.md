# Key centering

`FastH3KeyCenterConfig::from_layout` uses the checked tile layout's entire
`padded_rows()` sequence. It must consume the original tiled keys after the
unsmoothed Q/K/V pooling branch has been formed. Padding participates in this
mean; the valid-key gather belongs after the prefix Q/K FP8 roundtrip. The
fixed checkpoint uses 128-row tiles, so the logical extent equals the padded
tile buffer extent. This constructor does not generalize that equality to
other tile sizes.

The shared `operation.row_center.bf16` returns two admitted tensors:
centered BF16 `[B,S,C]` and mean BF16 `[B,1,C]`. Batches are independent. Its
explicit arithmetic profile uses eight ascending F32 lane accumulators,
the lane-combination strides 4/2/1, an F32 factor multiplication, then a BF16
mean conversion. The subtraction reads that BF16 mean and rounds to BF16.
Both CUDA launches use the existing retained module and stream; the mean is
an ordinary plan output rather than a hidden allocation or CPU readback.

The pinned Torch implementation uses F32 accumulation for BF16 mean, computes
the factor as `F32(num_output_elements) / F32(num_input_elements)`, and projects
the accumulated sum to the output dtype. See the pinned
[CUDA mean implementation](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/cuda/ReduceMomentKernel.cu)
and [MeanOps](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/SharedReduceOps.h).
Ferrum preserves these dtype boundaries and factor rounding. Its declared
eight-lane reduction is a separate numerical identity; Torch's reduction
tree has not been reproduced or qualified as bitwise equivalent.

The reusable managed fixture compares two changing inputs at non-aligned
batch/sequence/channel extents with an independent F64 mean oracle on exactly
summable dyadic data. Backend tests separately exercise BF16 mean rounding,
subnormals, non-finite propagation, alias rejection and foreign owners. These
checks do not establish full-model output quality or performance. The ignored
CUDA fixture must pass on the frozen source before claiming GPU qualification.
