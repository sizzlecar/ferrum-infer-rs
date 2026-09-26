//! Lossless source-code reorder for the checked RN-F16 fragment ABI.
//! Layout is shared with the qualified CUDA diagnostic: N16/K32 metadata and
//! two warp fragments. This function grants no loading or execution authority.
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_interfaces::vnext::{RnF16FragmentPlanV1, RnF16FragmentSourceFormatV1, VNextError};
use half::f16;

#[cfg(test)]
mod tests;

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn format(plan: &RnF16FragmentPlanV1) -> GgufBlockFormat {
    match plan.source_format() {
        RnF16FragmentSourceFormatV1::Q4K => GgufBlockFormat::Q4K,
        RnF16FragmentSourceFormatV1::Q5K => GgufBlockFormat::Q5K,
        RnF16FragmentSourceFormatV1::Q6K => GgufBlockFormat::Q6K,
    }
}
fn half_at(block: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([block[offset], block[offset + 1]])).to_f32()
}
fn scale_min(s: &[u8], group: usize) -> (u8, u8) {
    if group < 4 {
        (s[group] & 63, s[group + 4] & 63)
    } else {
        (
            (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4),
            (s[group + 4] >> 4) | ((s[group] >> 6) << 4),
        )
    }
}
// These are physical code/metadata reads, independently checked against the
// real materializer, not an alternative F32/F16 oracle implementation.
fn coefficient(format: GgufBlockFormat, block: &[u8], i: usize) -> (f32, f32, u8) {
    if format == GgufBlockFormat::Q6K {
        let group = i % 128 / 32;
        let low = (block[i / 128 * 64 + group % 2 * 32 + i % 32] >> (4 * (group / 2))) & 15;
        let high = (block[128 + i / 128 * 32 + i % 32] >> (2 * group)) & 3;
        (
            half_at(block, 208) * f32::from(block[192 + i / 16] as i8),
            0.0,
            low | high << 4,
        )
    } else {
        let group = i / 32;
        let (a, z) = scale_min(&block[4..16], group);
        let start = if format == GgufBlockFormat::Q5K {
            48
        } else {
            16
        };
        let low = (block[start + i / 64 * 32 + i % 32] >> (4 * (i % 64 / 32))) & 15;
        let high = if format == GgufBlockFormat::Q5K {
            (block[16 + i % 32] >> group) & 1
        } else {
            0
        };
        (
            half_at(block, 0) * f32::from(a),
            half_at(block, 2) * f32::from(z),
            low | high << 4,
        )
    }
}
fn rounded(format: GgufBlockFormat, a: f32, z: f32, q: u8) -> f16 {
    f16::from_f32(if format == GgufBlockFormat::Q6K {
        a * (i32::from(q) - 32) as f32
    } else {
        a * f32::from(q) - z
    })
}

/// Pack consecutive, whole-row source slices as one logical N dimension.
/// In particular, gate and up do not receive separate N16 padding.
pub fn pack_rn_f16_fragments(
    plan: &RnF16FragmentPlanV1,
    ordered_sources: &[&[u8]],
) -> Result<Vec<u8>, VNextError> {
    pack_checked(plan, ordered_sources, None)
}

/// The dual materializer validates every real reconstructed coefficient against
/// the existing RN converter while packing; no second unpack/matrix pass.
pub(crate) fn pack_checked(
    plan: &RnF16FragmentPlanV1,
    ordered_sources: &[&[u8]],
    dense_rn: Option<&[u8]>,
) -> Result<Vec<u8>, VNextError> {
    let host_size =
        |n| usize::try_from(n).map_err(|_| invalid("RN fragment exceeds host address space"));
    let (n, k) = (host_size(plan.n())?, host_size(plan.k())?);
    let (row_bytes, source_bytes, dense_bytes, size) = (
        host_size(plan.source_row_bytes())?,
        host_size(plan.source_bytes())?,
        host_size(plan.dense_bytes())?,
        host_size(plan.packed_bytes())?,
    );
    if ordered_sources.is_empty() || dense_rn.is_some_and(|v| v.len() != dense_bytes) {
        return Err(invalid("RN fragment source group or dense RN span differs"));
    }
    let total = ordered_sources.iter().try_fold(0usize, |total, bytes| {
        if bytes.is_empty() || !bytes.len().is_multiple_of(row_bytes) {
            return Err(invalid(
                "RN fragment source slices must contain complete nonempty rows",
            ));
        }
        total
            .checked_add(bytes.len())
            .ok_or_else(|| invalid("RN fragment source span overflow"))
    })?;
    if total != source_bytes {
        return Err(invalid("RN fragment source span differs from checked plan"));
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(n)
        .map_err(|_| invalid("RN fragment row index allocation unavailable"))?;
    for source in ordered_sources {
        rows.extend(source.chunks_exact(row_bytes));
    }
    let format = format(plan);
    let [tiles, groups] = plan.packed_dimensions();
    let (tiles, groups) = (host_size(tiles)?, host_size(groups)?);
    // Byte arithmetic is supplied by the interfaces plan, not another policy.
    let cell_bytes = size / tiles / groups;
    let step = (cell_bytes - 128) / 2;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| invalid("RN fragment packed allocation unavailable"))?;
    bytes.resize(size, 0);
    for tile in 0..tiles {
        for group in 0..groups {
            let cell = &mut bytes[(tile * groups + group) * cell_bytes..][..cell_bytes];
            for local in 0..16 {
                let column = tile * 16 + local;
                if column >= n {
                    continue;
                }
                let block_start = group / 8 * format.block_bytes();
                let block = &rows[column][block_start..][..format.block_bytes()];
                let (a, z, _) = coefficient(format, block, group % 8 * 32);
                cell[local * 4..local * 4 + 4].copy_from_slice(&a.to_le_bytes());
                let second = if format == GgufBlockFormat::Q6K {
                    coefficient(format, block, group % 8 * 32 + 16).0
                } else {
                    z
                };
                cell[64 + local * 4..68 + local * 4].copy_from_slice(&second.to_le_bytes());
            }
            for fragment in 0..2 {
                for lane in 0..32 {
                    let mut low = 0_u32;
                    let mut high = [0_u8; 2];
                    for slot in 0..8 {
                        let local = lane / 4 + ((slot / 2) % 2) * 8;
                        let column = tile * 16 + local;
                        if column >= n {
                            continue;
                        }
                        let kk =
                            group * 32 + fragment * 16 + 2 * (lane % 4) + (slot / 4) * 8 + slot % 2;
                        let start = kk / 256 * format.block_bytes();
                        let (a, z, q) = coefficient(
                            format,
                            &rows[column][start..][..format.block_bytes()],
                            kk % 256,
                        );
                        let rn = rounded(format, a, z, q);
                        if !a.is_finite() || !z.is_finite() || !rn.is_finite() {
                            return Err(invalid(
                                "RN fragment coefficient nonfinite or overflows binary16",
                            ));
                        }
                        if let Some(dense) = dense_rn {
                            let offset = (column * k + kk) * 2;
                            if rn.to_bits()
                                != u16::from_le_bytes([dense[offset], dense[offset + 1]])
                            {
                                return Err(invalid("RN fragment reconstruction differs from original RN conversion"));
                            }
                        }
                        low |= u32::from(q & 15) << (slot * 4);
                        high[0] |= ((q >> 4) & 1) << slot;
                        high[1] |= ((q >> 5) & 1) << slot;
                    }
                    let base = 128 + fragment * step;
                    cell[base + lane * 4..base + lane * 4 + 4].copy_from_slice(&low.to_le_bytes());
                    if format != GgufBlockFormat::Q4K {
                        cell[base + 128 + lane] = high[0];
                    }
                    if format == GgufBlockFormat::Q6K {
                        cell[base + 160 + lane] = high[1];
                    }
                }
            }
        }
    }
    Ok(bytes)
}
