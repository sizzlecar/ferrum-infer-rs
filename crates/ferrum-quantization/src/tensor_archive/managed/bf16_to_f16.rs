//! Exact bit conversion for normal/zero BF16 blocks. No float scratch, extra
//! payload or new rounding boundary is introduced. Exceptional blocks delegate
//! to the existing scalar converter, including its finite-overflow rejection.
use super::diagnostics::Bf16ToF16BlockCounts;
use ferrum_runtime::ElementType;

const BLOCK_BYTES: usize = 16;

pub(super) fn convert(input: &[u8], output: &mut [u8]) -> Result<(), &'static str> {
    convert_counted::<false>(input, output, &mut Bf16ToF16BlockCounts::default())
}

pub(super) fn convert_counted<const COUNT: bool>(
    input: &[u8],
    output: &mut [u8],
    counts: &mut Bf16ToF16BlockCounts,
) -> Result<(), &'static str> {
    if input.len() != output.len() || !input.len().is_multiple_of(2) {
        return Err("source and target floating element counts differ");
    }
    #[cfg(all(
        target_endian = "little",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
        let complete = input.len() / BLOCK_BYTES * BLOCK_BYTES;
        for (source, destination) in input[..complete]
            .chunks_exact(BLOCK_BYTES)
            .zip(output[..complete].chunks_exact_mut(BLOCK_BYTES))
        {
            // SAFETY: both slices cover one complete block. These intrinsic
            // loads/stores explicitly permit unaligned addresses. SSE2 is an
            // x86_64 baseline; Advanced SIMD is an AArch64 baseline. Each
            // implementation checks every lane before writing anything.
            if COUNT {
                counts.simd_blocks_attempted += 1;
            }
            if !unsafe { normal_block(source.as_ptr(), destination.as_mut_ptr()) } {
                if COUNT {
                    counts.scalar_fallback_blocks += 1;
                }
                scalar(source, destination)?;
            } else if COUNT {
                counts.simd_blocks_converted += 1;
            }
        }
        if COUNT {
            counts.scalar_tail_elements += ((input.len() - complete) / 2) as u64;
        }
        scalar(&input[complete..], &mut output[complete..])
    }
    #[cfg(not(all(
        target_endian = "little",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    {
        if COUNT {
            counts.portable_scalar_elements += (input.len() / 2) as u64;
        }
        scalar(input, output)
    }
}

fn scalar(input: &[u8], output: &mut [u8]) -> Result<(), &'static str> {
    super::convert_floating_scalar(ElementType::Bf16, ElementType::F16, input, output, None)
}

// A BF16 normal with magnitude bits in [0x3880, 0x477f] has an exponent
// representable as a normal F16. Its 7 mantissa bits fit exactly in F16's 10;
// rebiasing is sign | ((magnitude - 0x3800) << 3), with no rounding. Zero keeps
// its sign. Subnormals, infinities, NaNs and overflow candidates use scalar.
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "sse2")]
unsafe fn normal_block(input: *const u8, output: *mut u8) -> bool {
    use std::arch::x86_64::*;
    // SAFETY: caller supplies 16 readable and 16 exclusively writable bytes;
    // loadu/storeu require no alignment. All arithmetic is integer lane-wise.
    unsafe {
        let words = _mm_loadu_si128(input.cast());
        let magnitude = _mm_and_si128(words, _mm_set1_epi16(0x7fff));
        let below = _mm_cmpgt_epi16(_mm_set1_epi16(0x3880), magnitude);
        let above = _mm_cmpgt_epi16(magnitude, _mm_set1_epi16(0x477f));
        let normal = _mm_andnot_si128(_mm_or_si128(below, above), _mm_set1_epi16(-1));
        let zero = _mm_cmpeq_epi16(magnitude, _mm_setzero_si128());
        if _mm_movemask_epi8(_mm_or_si128(normal, zero)) != 0xffff {
            return false;
        }
        let rebased = _mm_slli_epi16::<3>(_mm_sub_epi16(magnitude, _mm_set1_epi16(0x3800)));
        let sign = _mm_and_si128(words, _mm_set1_epi16(i16::MIN));
        let converted = _mm_or_si128(sign, _mm_and_si128(rebased, normal));
        _mm_storeu_si128(output.cast(), converted);
        true
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn normal_block(input: *const u8, output: *mut u8) -> bool {
    use std::arch::aarch64::*;
    // SAFETY: byte-vector LD1/ST1 accept unaligned pointers and access exactly
    // the complete block admitted by the caller. Reinterpret affects registers
    // only, never Rust references or the source/destination allocation layouts.
    unsafe {
        let words = vreinterpretq_u16_u8(vld1q_u8(input));
        let magnitude = vandq_u16(words, vdupq_n_u16(0x7fff));
        let normal = vandq_u16(
            vcgeq_u16(magnitude, vdupq_n_u16(0x3880)),
            vcleq_u16(magnitude, vdupq_n_u16(0x477f)),
        );
        let zero = vceqq_u16(magnitude, vdupq_n_u16(0));
        if vminvq_u16(vorrq_u16(normal, zero)) != u16::MAX {
            return false;
        }
        let rebased = vshlq_n_u16::<3>(vsubq_u16(magnitude, vdupq_n_u16(0x3800)));
        let sign = vandq_u16(words, vdupq_n_u16(0x8000));
        let converted = vorrq_u16(sign, vandq_u16(rebased, normal));
        vst1q_u8(output, vreinterpretq_u8_u16(converted));
        true
    }
}

#[cfg(test)]
mod tests;
