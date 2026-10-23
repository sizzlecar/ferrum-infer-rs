//! Explicit FP32 scalar arithmetic and final BF16 boundaries. Affine dots use
//! ascending separate multiply/add; norm uses a bounded binary pairwise tree;
//! GELU/softmax use pinned libm. Attention has at most 256 FP32 accumulators.
//! These profiles are independent of vendor GEMM, CUDA reduction and Torch.
use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceCommandKind, ReferenceDeviceCommand, ReferenceDeviceRuntimeError,
};
use ferrum_runtime::{TensorPrimitiveBf16, TensorPrimitiveBf16Geometry};

pub(crate) struct ReferenceTensorPrimitiveLaunch {
    pub(crate) inputs: Vec<ReferenceBufferRegion>,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: TensorPrimitiveBf16Geometry,
}
impl ReferenceTensorPrimitiveLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let g = &self.shape;
        let expected = byte_extents(g)?;
        if self.inputs.len() != expected.len() || g.input_bytes != expected {
            return Err(invalid("input arity/extent differs"));
        }
        for (index, region) in self.inputs.iter().enumerate() {
            let dtype = if g.kind == TensorPrimitiveBf16::HalfRotary && index > 0 {
                ElementType::F32
            } else {
                ElementType::Bf16
            };
            region_check(region, expected[index], dtype)?;
            if self.output.overlaps(region) {
                return Err(invalid("output overlaps input"));
            }
        }
        region_check(&self.output, g.output_bytes, ElementType::Bf16)?;
        arithmetic(g)?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in self.inputs.iter().chain(std::iter::once(&self.output)) {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
    fn logical_work(&self) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(self.shape.elements(), arithmetic(&self.shape)?)
            .map_err(|e| invalid(e.to_string()))
    }
}
impl ReferenceDeviceCommand {
    pub(crate) fn tensor_primitive(
        launch: ReferenceTensorPrimitiveLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: launch.shape.kind.operation_id(),
            logical_work: Some(launch.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::TensorPrimitive { launch },
        })
    }
}
fn region_check(
    region: &ReferenceBufferRegion,
    bytes: u64,
    dtype: ElementType,
) -> Result<(), ReferenceDeviceRuntimeError> {
    region.validate_range()?;
    let scalar = if dtype == ElementType::F32 { 4 } else { 2 };
    if bytes > isize::MAX as u64
        || region.length_bytes as u64 != bytes
        || region.element_type != dtype
        || (region.offset_bytes - region.allocation.logical_offset_bytes) % scalar != 0
    {
        return Err(invalid("region dtype/extent/alignment differs"));
    }
    Ok(())
}
fn product(values: &[u64]) -> Result<u64, ReferenceDeviceRuntimeError> {
    values
        .iter()
        .try_fold(1_u64, |n, d| n.checked_mul(*d))
        .ok_or_else(|| invalid("arithmetic/extent overflow"))
}
fn byte_extents(g: &TensorPrimitiveBf16Geometry) -> Result<Vec<u64>, ReferenceDeviceRuntimeError> {
    use TensorPrimitiveBf16::*;
    if g.rows == 0
        || g.columns == 0
        || product(&[g.rows, g.columns, 2])? != g.output_bytes
        || g.output_bytes > isize::MAX as u64
    {
        return Err(invalid("invalid output geometry"));
    }
    let same = g.output_bytes;
    let head = || {
        g.head_dim >= 2 && g.head_dim <= 256 && g.head_dim % 2 == 0 && g.columns % g.head_dim == 0
    };
    match g.kind {
        AffineLinear | PatchProjection => {
            if g.inner == 0 {
                return Err(invalid("zero projection width"));
            }
            Ok(vec![
                product(&[g.rows, g.inner, 2])?,
                product(&[g.columns, g.inner, 2])?,
                product(&[g.columns, 2])?,
            ])
        }
        AffineLayerNorm => {
            if g.columns > 1 << 24 || !g.scalar.is_finite() || g.scalar <= 0.0 {
                return Err(invalid("invalid normalization parameters"));
            }
            Ok(vec![
                same,
                product(&[g.columns, 2])?,
                product(&[g.columns, 2])?,
            ])
        }
        HalfRotary => {
            if !head() {
                return Err(invalid("invalid rotary partition"));
            }
            let phase = product(&[g.rows, g.head_dim, 4])?;
            Ok(vec![same, phase, phase])
        }
        SegmentedAttention | RowScatterAdd => {
            if g.starts.is_empty() || g.starts.len() > 32 || g.starts.len() != g.lengths.len() {
                return Err(invalid("invalid run count"));
            }
            let mut end = 0;
            let mut count = 0u64;
            for (&start, &length) in g.starts.iter().zip(&g.lengths) {
                if length == 0 || start < end || (g.kind == SegmentedAttention && start != end) {
                    return Err(invalid("invalid run partition"));
                }
                end = start
                    .checked_add(length)
                    .ok_or_else(|| invalid("run overflow"))?;
                count = count
                    .checked_add(length)
                    .ok_or_else(|| invalid("run count overflow"))?;
                if end > g.rows {
                    return Err(invalid("run outside output"));
                }
            }
            if g.kind == SegmentedAttention {
                if end != g.rows || !head() || !g.scalar.is_finite() || g.scalar <= 0.0 {
                    return Err(invalid("invalid attention geometry"));
                }
                Ok(vec![same; 3])
            } else {
                Ok(vec![same, product(&[count, g.columns, 2])?])
            }
        }
        GeluErf | ReshapeCopy => Ok(vec![same]),
    }
}
/// Logical scalar arithmetic, not hardware FLOPs; special-value early exits
/// retain the nominal finite-path count. Library erf/exp calls each count one.
pub(crate) fn arithmetic(
    g: &TensorPrimitiveBf16Geometry,
) -> Result<u64, ReferenceDeviceRuntimeError> {
    use TensorPrimitiveBf16::*;
    byte_extents(g)?;
    let n = g.elements();
    match g.kind {
        AffineLinear | PatchProjection => product(&[
            n,
            g.inner
                .checked_mul(2)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| invalid("dot work overflow"))?,
        ]),
        AffineLayerNorm => product(&[
            g.rows,
            g.columns
                .checked_mul(8)
                .and_then(|n| n.checked_add(3))
                .ok_or_else(|| invalid("norm work overflow"))?,
        ]),
        GeluErf => product(&[n, 5]),
        HalfRotary => product(&[n / 2, 7]),
        RowScatterAdd => product(&[g.lengths.iter().sum(), g.columns]),
        ReshapeCopy => Ok(0),
        SegmentedAttention => {
            let dot = g.head_dim + g.head_dim.next_power_of_two();
            let update = 3 * g.head_dim + 6;
            let mut work = 0u64;
            for &len in &g.lengths {
                let per = product(&[len, dot])?
                    .checked_add(product(&[len - 1, update])?)
                    .and_then(|n| n.checked_add(g.head_dim + 1))
                    .ok_or_else(|| invalid("attention work overflow"))?;
                work = work
                    .checked_add(product(&[len, g.columns / g.head_dim, per])?)
                    .ok_or_else(|| invalid("attention work overflow"))?;
            }
            Ok(work)
        }
    }
}
fn read(s: &AllocationLocks<'_, 4>, role: usize, i: usize) -> f32 {
    bf16::from_bits(s.read_bits(role, i)).to_f32()
}
fn write(s: &mut AllocationLocks<'_, 4>, i: usize, value: f32) {
    s.write_bits(3, i, bf16::from_f32(value).to_bits());
}
fn sum(start: usize, len: usize, value: &impl Fn(usize) -> f32) -> f32 {
    if len == 1 {
        value(start)
    } else {
        let left = len / 2;
        sum(start, left, value) + sum(start + left, len - left, value)
    }
}
pub(super) fn execute(launch: &ReferenceTensorPrimitiveLaunch) {
    let g = &launch.shape;
    let first = &launch.inputs[0];
    let mut s = AllocationLocks::new([
        first,
        launch.inputs.get(1).unwrap_or(first),
        launch.inputs.get(2).unwrap_or(first),
        &launch.output,
    ]);
    let rows = g.rows as usize;
    let columns = g.columns as usize;
    use TensorPrimitiveBf16::*;
    match g.kind {
        AffineLinear | PatchProjection => {
            let inner = g.inner as usize;
            for row in 0..rows {
                for col in 0..columns {
                    let mut total = 0.0f32;
                    for k in 0..inner {
                        total += read(&s, 0, row * inner + k) * read(&s, 1, col * inner + k);
                    }
                    let dot = if g.kind == PatchProjection {
                        bf16::from_f32(total).to_f32()
                    } else {
                        total
                    };
                    let value = dot + read(&s, 2, col);
                    write(&mut s, row * columns + col, value);
                }
            }
        }
        AffineLayerNorm => {
            for row in 0..rows {
                let base = row * columns;
                let mean = sum(0, columns, &|c| read(&s, 0, base + c)) / columns as f32;
                let variance = sum(0, columns, &|c| {
                    let d = read(&s, 0, base + c) - mean;
                    d * d
                }) / columns as f32;
                let inv = 1.0f32 / (variance + g.scalar).sqrt();
                for col in 0..columns {
                    let normalized = (read(&s, 0, base + col) - mean) * inv;
                    let value = normalized * read(&s, 1, col) + read(&s, 2, col);
                    write(&mut s, base + col, value);
                }
            }
        }
        GeluErf => {
            for i in 0..rows * columns {
                let x = read(&s, 0, i);
                let value = (0.5 * x) * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
                write(&mut s, i, value);
            }
        }
        HalfRotary => {
            let d = g.head_dim as usize;
            let half = d / 2;
            for i in 0..rows * columns {
                let lane = i % d;
                let partner = if lane < half { i + half } else { i - half };
                let rotated = if lane < half {
                    -read(&s, 0, partner)
                } else {
                    read(&s, 0, partner)
                };
                let phase = (i / columns) * d + lane;
                let value = read(&s, 0, i) * s.read_f32(1, phase) + rotated * s.read_f32(2, phase);
                write(&mut s, i, value);
            }
        }
        RowScatterAdd => {
            let mut run = 0;
            let mut source = 0usize;
            for row in 0..rows {
                while run < g.starts.len() && row as u64 >= g.starts[run] + g.lengths[run] {
                    source += g.lengths[run] as usize;
                    run += 1;
                }
                for c in 0..columns {
                    let i = row * columns + c;
                    if run < g.starts.len() && row as u64 >= g.starts[run] {
                        let update = (source + row - g.starts[run] as usize) * columns + c;
                        let value = read(&s, 0, i) + read(&s, 1, update);
                        write(&mut s, i, value);
                    } else {
                        let bits = s.read_bits(0, i);
                        s.write_bits(3, i, bits);
                    }
                }
            }
        }
        ReshapeCopy => {
            for i in 0..rows * columns {
                let bits = s.read_bits(0, i);
                s.write_bits(3, i, bits);
            }
        }
        SegmentedAttention => attention(&mut s, g),
    }
}
fn attention(s: &mut AllocationLocks<'_, 4>, g: &TensorPrimitiveBf16Geometry) {
    let columns = g.columns as usize;
    let d = g.head_dim as usize;
    let tree = d.next_power_of_two();
    for (&start, &length) in g.starts.iter().zip(&g.lengths) {
        let start = start as usize;
        let end = start + length as usize;
        for row in start..end {
            for head in 0..columns / d {
                let base = row * columns + head * d;
                let mut acc = [0.0f32; 256];
                let mut maximum = 0.0;
                let mut denominator = 1.0;
                let mut poison = false;
                for key in start..end {
                    let kbase = key * columns + head * d;
                    let score = sum(0, tree, &|c| {
                        if c < d {
                            read(s, 0, base + c) * read(s, 1, kbase + c)
                        } else {
                            0.0
                        }
                    }) * g.scalar;
                    if !score.is_finite() {
                        poison = true;
                        break;
                    }
                    if key == start {
                        maximum = score;
                        for c in 0..d {
                            acc[c] = read(s, 2, kbase + c);
                        }
                    } else {
                        let next = maximum.max(score);
                        let old_weight = libm::expf(maximum - next);
                        let weight = libm::expf(score - next);
                        denominator = denominator * old_weight + weight;
                        for c in 0..d {
                            acc[c] = acc[c] * old_weight + weight * read(s, 2, kbase + c);
                        }
                        maximum = next;
                    }
                }
                let inv = 1.0f32 / denominator;
                for c in 0..d {
                    write(s, base + c, if poison { f32::NAN } else { acc[c] * inv });
                }
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("reference tensor primitive: {}", reason.into()))
}

#[cfg(test)]
mod tests;
