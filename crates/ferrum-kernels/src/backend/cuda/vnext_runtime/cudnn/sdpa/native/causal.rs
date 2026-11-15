//! Native graph pattern from NVIDIA cudnn-frontend v1.18.0 SDPA expansion.
//! All score/probability tensors are virtual. No score matrix is allocated;
//! the selected native engine's exact workspace must fit the caller's quota.
use super::*;

pub(super) static MASKED_VALUE: f32 = f32::NEG_INFINITY;

fn pointwise(
    b: &mut Builder<'_>,
    mode: c_int,
    compute: c_int,
    axis: Option<i64>,
    inputs: &[Descriptor],
    output: Descriptor,
) -> Result<Descriptor, String> {
    let d = b.create(0)?;
    b.set(d, 0, 14, &[mode])?;
    b.set(d, 1, TYPE_DATA, &[compute])?;
    if let Some(axis) = axis {
        b.set(d, 9, TYPE_I64, &[axis])?;
    }
    b.finish(d)?;
    let op = b.create(13)?;
    b.set(op, 750, TYPE_DESCRIPTOR, &[d])?;
    for (&tensor, attr) in inputs.iter().zip([751, 752, 758]) {
        b.set(op, attr, TYPE_DESCRIPTOR, &[tensor])?;
    }
    b.set(op, 753, TYPE_DESCRIPTOR, &[output])?;
    b.set(op, 754, 4, &[1.0_f32])?;
    b.set(op, 755, 4, &[1.0_f32])?;
    b.finish(op)?;
    Ok(op)
}
fn matmul(
    b: &mut Builder<'_>,
    a: Descriptor,
    rhs: Descriptor,
    output: Descriptor,
) -> Result<Descriptor, String> {
    let d = b.create(18)?;
    b.set(d, 1500, TYPE_DATA, &[0 as c_int])?;
    b.finish(d)?;
    let op = b.create(19)?;
    for (attr, value) in [(1520, a), (1521, rhs), (1522, output), (1523, d)] {
        b.set(op, attr, TYPE_DESCRIPTOR, &[value])?;
    }
    b.finish(op)?;
    Ok(op)
}
fn reduction(
    b: &mut Builder<'_>,
    mode: c_int,
    x: Descriptor,
    y: Descriptor,
) -> Result<Descriptor, String> {
    let d = b.create(21)?;
    b.set(d, 1600, 18, &[mode])?;
    b.set(d, 1601, TYPE_DATA, &[0 as c_int])?;
    b.finish(d)?;
    let op = b.create(22)?;
    for (attr, value) in [(1610, x), (1611, y), (1612, d)] {
        b.set(op, attr, TYPE_DESCRIPTOR, &[value])?;
    }
    b.finish(op)?;
    Ok(op)
}
pub(super) fn operations(
    b: &mut Builder<'_>,
    s: CudnnSdpaSignature,
) -> Result<Vec<Descriptor>, String> {
    let qrow = s.heads * s.head_dim;
    let kvrow = s.key_value_heads * s.head_dim;
    let qdim = [s.batch, s.heads, s.query_length, s.head_dim];
    let q = b.tensor(
        1,
        9,
        qdim,
        [s.query_batch_stride, s.head_dim, qrow, 1],
        16,
        false,
    )?;
    let k = b.tensor(
        2,
        9,
        [s.batch, s.key_value_heads, s.head_dim, s.key_value_length],
        [s.key_value_batch_stride, s.head_dim, 1, kvrow],
        16,
        false,
    )?;
    let v = b.tensor(
        3,
        9,
        [s.batch, s.key_value_heads, s.key_value_length, s.head_dim],
        [s.key_value_batch_stride, s.head_dim, kvrow, 1],
        16,
        false,
    )?;
    let o = b.tensor(
        4,
        9,
        qdim,
        [s.output_batch_stride, s.head_dim, qrow, 1],
        16,
        false,
    )?;
    let scale = b.tensor(5, 0, [1; 4], [1; 4], 4, true)?;
    let negative = b.tensor(8, 0, [1; 4], [1; 4], 4, true)?;
    let pair = s
        .query_length
        .checked_mul(s.key_value_length)
        .ok_or("virtual scores overflow")?;
    let plane = s.heads.checked_mul(pair).ok_or("virtual scores overflow")?;
    s.batch
        .checked_mul(plane)
        .filter(|n| *n <= isize::MAX as i64 / 4)
        .ok_or("virtual score geometry exceeds address space")?;
    let dims = [s.batch, s.heads, s.query_length, s.key_value_length];
    let strides = [plane, pair, s.key_value_length, 1];
    let small = [s.batch, s.heads, s.query_length, 1];
    let small_stride = [s.heads * s.query_length, s.query_length, 1, 1];
    let mut t = |uid, ty, reduce| {
        b.tensor_kind(
            uid,
            ty,
            if reduce { small } else { dims },
            if reduce { small_stride } else { strides },
            16,
            false,
            true,
        )
    };
    let scores = t(20, 0, false)?;
    let scaled = t(21, 0, false)?;
    let row = t(22, 4, false)?;
    let col = t(23, 4, false)?;
    let keep = t(24, 11, false)?;
    let masked = t(25, 0, false)?;
    let maximum = t(26, 0, true)?;
    let shifted = t(27, 0, false)?;
    let exp = t(28, 0, false)?;
    let sum = t(29, 0, true)?;
    let probability = t(30, 9, false)?;
    // Match frontend's virtual softmax statistics branch as well as P's BF16
    // tensor-core boundary. Vendor arithmetic is explicitly a separate profile.
    let log = t(31, 0, true)?;
    let stats = t(32, 0, true)?;
    Ok(vec![
        matmul(b, q, k, scores)?,
        pointwise(b, 1, 0, None, &[scores, scale], scaled)?,
        pointwise(b, 501, 4, Some(2), &[scaled], row)?,
        pointwise(b, 501, 4, Some(3), &[scaled], col)?,
        pointwise(b, 303, 11, None, &[row, col], keep)?,
        pointwise(b, 601, 0, None, &[scaled, negative, keep], masked)?,
        reduction(b, 3, masked, maximum)?,
        pointwise(b, 9, 0, None, &[masked, maximum], shifted)?,
        pointwise(b, 13, 0, None, &[shifted], exp)?,
        reduction(b, 0, exp, sum)?,
        pointwise(b, 15, 0, None, &[sum], log)?,
        pointwise(b, 0, 0, None, &[maximum, log], stats)?,
        pointwise(b, 6, 0, None, &[exp, sum], probability)?,
        matmul(b, probability, v, o)?,
    ])
}
