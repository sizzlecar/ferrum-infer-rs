//! Separable Lanczos-3 with the Pillow 13 integer pixel-rounding boundaries.
//! Coefficient transcendental evaluation uses pinned Rust libm; equivalence of
//! those coefficients to a particular deployed libc is not assumed. All image
//! and coefficient backings are charged before construction. Identity retains
//! the original owner and hidden RGB channels without premultiplication.
//!
//! References: Pillow 13.0.0 libImaging/Resample.c (coefficient normalization,
//! 22-bit fixed-point pixels, axis order) and Convert.c (RGBA premultiplication).

use super::*;
use ferrum_runtime::ManagedHostBuffer;

const FRACTION_BITS: u32 = 22;
const UNIT: i64 = 1 << FRACTION_BITS;

struct Axis {
    input: usize,
    output: usize,
    stride: usize,
    bytes: usize,
    scale: f64,
    filter_scale: f64,
    support: f64,
}

impl Axis {
    fn new(input: u32, output: u32) -> ImageInputResult<Self> {
        if input == 0 || output == 0 || input > i32::MAX as u32 || output > i32::MAX as u32 {
            return Err(ImageInputError::Limit("resampling axis"));
        }
        // PIL's default box endpoints are float32, before its double arithmetic.
        let scale = f64::from(input as f32) / f64::from(output);
        let filter_scale = scale.max(1.0);
        let support = 3.0 * filter_scale;
        let taps = (support.ceil() as usize)
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or(ImageInputError::Limit("resampling filter support"))?;
        let stride = taps
            .checked_mul(4)
            .and_then(|n| n.checked_add(8))
            .ok_or(ImageInputError::Limit("resampling coefficient stride"))?;
        let bytes = stride
            .checked_mul(output as usize)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(ImageInputError::Limit("resampling coefficient bytes"))?;
        Ok(Self {
            input: input as usize,
            output: output as usize,
            stride,
            bytes,
            scale,
            filter_scale,
            support,
        })
    }

    fn write(&self, target: &mut [u8]) -> ImageInputResult<()> {
        let inverse_scale = 1.0 / self.filter_scale;
        for output in 0..self.output {
            let center = (output as f64 + 0.5) * self.scale;
            // C truncation after +0.5, then endpoint clipping.
            let start = ((center - self.support + 0.5) as i64).max(0) as usize;
            let end = ((center + self.support + 0.5) as usize).min(self.input);
            if start >= end {
                return Err(ImageInputError::Invalid("empty resampling support"));
            }
            let count = end - start;
            let row = &mut target[output * self.stride..(output + 1) * self.stride];
            row[..4].copy_from_slice(&(start as u32).to_le_bytes());
            row[4..8].copy_from_slice(&(count as u32).to_le_bytes());
            let mut sum = 0.0;
            for source in start..end {
                sum += lanczos_weight((source as f64 - center + 0.5) * inverse_scale);
            }
            if !sum.is_finite() || sum == 0.0 {
                return Err(ImageInputError::Invalid(
                    "non-finite resampling normalization",
                ));
            }
            let mut absolute_sum = 0i64;
            for (index, source) in (start..end).enumerate() {
                let weight = lanczos_weight((source as f64 - center + 0.5) * inverse_scale) / sum;
                let fixed = (weight * UNIT as f64).round(); // ties away from zero
                if !fixed.is_finite() || fixed < f64::from(i32::MIN) || fixed > f64::from(i32::MAX)
                {
                    return Err(ImageInputError::Limit("resampling coefficient range"));
                }
                let fixed = fixed as i32;
                absolute_sum += i64::from(fixed).abs();
                row[8 + index * 4..12 + index * 4].copy_from_slice(&fixed.to_le_bytes());
            }
            if absolute_sum * 255 + UNIT / 2 > i64::from(i32::MAX) {
                return Err(ImageInputError::Limit("resampling accumulator range"));
            }
        }
        Ok(())
    }
}

pub(super) fn lanczos(
    host: &HostMemoryAccount,
    source: &ManagedRgba8Image,
    width: u32,
    height: u32,
    limits: ImageInputLimits,
) -> ImageInputResult<ManagedRgba8Image> {
    source.validate_parent(host)?;
    limits.rgba_bytes(source.width, source.height)?;
    let output_bytes = limits.rgba_bytes(width, height)?;
    if (source.width, source.height) == (width, height) {
        return Ok(source.clone());
    }
    let x = (source.width != width)
        .then(|| Axis::new(source.width, width))
        .transpose()?;
    let y = (source.height != height)
        .then(|| Axis::new(source.height, height))
        .transpose()?;
    let horizontal_first = !((i64::from(source.height) - i64::from(height)) > 0
        && (i64::from(source.height) - i64::from(height))
            > (i64::from(source.width) - i64::from(width)) * 2);
    let temporary_shape = if x.is_some() && y.is_some() {
        Some(if horizontal_first {
            (width, source.height)
        } else {
            (source.width, height)
        })
    } else {
        None
    };
    let temporary_bytes = temporary_shape
        .map(|(w, h)| limits.rgba_bytes(w, h))
        .transpose()?;
    // At most four bounded control reservations: two coefficient tables,
    // optional intermediate, and final output. No pixel-sized Vec exists.
    let mut requests = Vec::with_capacity(4);
    if let Some(axis) = &x {
        requests.push((HostMemoryClass::InputPreprocess, axis.bytes));
    }
    if let Some(axis) = &y {
        requests.push((HostMemoryClass::InputPreprocess, axis.bytes));
    }
    if let Some(bytes) = temporary_bytes {
        requests.push((HostMemoryClass::InputPreprocess, bytes));
    }
    requests.push((HostMemoryClass::InputPreprocess, output_bytes));
    let mut reservations = host.reserve_many(&requests)?.into_iter();
    let mut table = |axis: &Option<Axis>| -> ImageInputResult<Option<ManagedHostBytes>> {
        axis.as_ref()
            .map(|axis| {
                let mut bytes = reservations.next().unwrap().allocate_zeroed()?;
                axis.write(bytes.as_mut())?;
                Ok(bytes.freeze())
            })
            .transpose()
    };
    let tx = table(&x)?;
    let ty = table(&y)?;
    let temporary = temporary_bytes
        .map(|_| reservations.next().unwrap().allocate_zeroed())
        .transpose()?;
    let mut output = reservations.next().unwrap().allocate_zeroed()?;
    let source_bytes = source.bytes.bytes();
    let premultiply = source.source_has_alpha;
    match (x.as_ref(), y.as_ref(), temporary) {
        (Some(x), Some(y), Some(mut intermediate)) => {
            if horizontal_first {
                pass(
                    source_bytes,
                    source.width as usize,
                    source.height as usize,
                    &mut intermediate,
                    x,
                    tx.as_ref().unwrap().bytes(),
                    true,
                    premultiply,
                );
                pass(
                    intermediate.as_ref(),
                    width as usize,
                    source.height as usize,
                    &mut output,
                    y,
                    ty.as_ref().unwrap().bytes(),
                    false,
                    false,
                );
            } else {
                pass(
                    source_bytes,
                    source.width as usize,
                    source.height as usize,
                    &mut intermediate,
                    y,
                    ty.as_ref().unwrap().bytes(),
                    false,
                    premultiply,
                );
                pass(
                    intermediate.as_ref(),
                    source.width as usize,
                    height as usize,
                    &mut output,
                    x,
                    tx.as_ref().unwrap().bytes(),
                    true,
                    false,
                );
            }
        }
        (Some(x), None, None) => pass(
            source_bytes,
            source.width as usize,
            source.height as usize,
            &mut output,
            x,
            tx.as_ref().unwrap().bytes(),
            true,
            premultiply,
        ),
        (None, Some(y), None) => pass(
            source_bytes,
            source.width as usize,
            source.height as usize,
            &mut output,
            y,
            ty.as_ref().unwrap().bytes(),
            false,
            premultiply,
        ),
        _ => return Err(ImageInputError::Invalid("resampling axis plan")),
    }
    if premultiply {
        for pixel in output.as_mut().chunks_exact_mut(4) {
            let alpha = u32::from(pixel[3]);
            if alpha != 0 && alpha != 255 {
                for channel in &mut pixel[..3] {
                    *channel = (u32::from(*channel) * 255 / alpha).min(255) as u8;
                }
            }
        }
    }
    Ok(ManagedRgba8Image {
        bytes: output.freeze(),
        width,
        height,
        source_has_alpha: source.source_has_alpha,
    })
}

fn pass(
    source: &[u8],
    input_width: usize,
    input_height: usize,
    output: &mut ManagedHostBuffer,
    axis: &Axis,
    table: &[u8],
    horizontal: bool,
    premultiply: bool,
) {
    let width = if horizontal { axis.output } else { input_width };
    let height = if horizontal {
        input_height
    } else {
        axis.output
    };
    for y in 0..height {
        for x in 0..width {
            let position = if horizontal { x } else { y };
            let row = &table[position * axis.stride..(position + 1) * axis.stride];
            let start = u32::from_le_bytes(row[..4].try_into().unwrap()) as usize;
            let count = u32::from_le_bytes(row[4..8].try_into().unwrap()) as usize;
            for channel in 0..4 {
                let mut sum = UNIT / 2;
                for tap in 0..count {
                    let pixel = if horizontal {
                        y * input_width + start + tap
                    } else {
                        (start + tap) * input_width + x
                    };
                    let mut value = source[pixel * 4 + channel];
                    if premultiply && channel < 3 {
                        value = ((u32::from(value) * u32::from(source[pixel * 4 + 3]) + 127) / 255)
                            as u8;
                    }
                    let weight =
                        i32::from_le_bytes(row[8 + tap * 4..12 + tap * 4].try_into().unwrap());
                    sum += i64::from(value) * i64::from(weight);
                }
                output.as_mut()[(y * width + x) * 4 + channel] =
                    (sum >> FRACTION_BITS).clamp(0, 255) as u8;
            }
        }
    }
}

fn lanczos_weight(x: f64) -> f64 {
    if !(-3.0..3.0).contains(&x) {
        return 0.0;
    }
    fn sinc(x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            let phase = x * std::f64::consts::PI;
            libm::sin(phase) / phase
        }
    }
    sinc(x) * sinc(x / 3.0)
}

#[cfg(test)]
#[path = "resize_tests.rs"]
mod tests;
