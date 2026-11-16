//! Shared RGB8 bicubic antialias arithmetic for declared HF processors.
//! F64 normalized taps and U8 ties-to-even per axis; SIMD bitwise parity is not asserted.
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1. {
        ((1.5 * x - 2.5) * x) * x + 1.
    } else if x < 2. {
        ((-0.5 * x + 2.5) * x - 4.) * x + 2.
    } else {
        0.
    }
}
pub(super) fn axis_sample(
    index: usize,
    input: usize,
    output: usize,
    read: impl Fn(usize) -> u8,
) -> u8 {
    if input == output {
        return read(index);
    }
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.);
    let center = (index as f64 + 0.5) * scale;
    let support = 2. * filter_scale;
    let start = ((center - support + 0.5).floor() as i64).max(0) as usize;
    let end = ((center + support + 0.5).floor() as usize).min(input);
    let mut weight = 0.;
    let mut value = 0.;
    for source in start..end {
        let w = cubic((source as f64 - center + 0.5) / filter_scale);
        weight += w;
        value += f64::from(read(source)) * w;
    }
    (value / weight).round_ties_even().clamp(0., 255.) as u8
}

pub(super) fn smart_resize(
    ow: u32,
    oh: u32,
    factor: u64,
    minimum_pixels: u64,
    maximum_pixels: u64,
) -> anyhow::Result<(usize, usize)> {
    use anyhow::ensure;
    ensure!(
        ow > 0
            && oh > 0
            && factor > 0
            && factor <= 1 << 24
            && minimum_pixels > 0
            && minimum_pixels <= maximum_pixels
            && maximum_pixels <= 1 << 53,
        "invalid image resize geometry or area policy"
    );
    ensure!(
        f64::from(ow.max(oh)) / f64::from(ow.min(oh)) <= 200.,
        "HF image aspect ratio exceeds 200"
    );
    let factor = factor as f64;
    let (w, h) = (f64::from(ow), f64::from(oh));
    let (mut rw, mut rh) = (
        (w / factor).round_ties_even() * factor,
        (h / factor).round_ties_even() * factor,
    );
    if rw * rh > maximum_pixels as f64 {
        let beta = (w * h / maximum_pixels as f64).sqrt();
        rw = (w / beta / factor).floor().max(1.) * factor;
        rh = (h / beta / factor).floor().max(1.) * factor;
    } else if rw * rh < minimum_pixels as f64 {
        let beta = (minimum_pixels as f64 / (w * h)).sqrt();
        rw = (w * beta / factor).ceil() * factor;
        rh = (h * beta / factor).ceil() * factor;
    }
    ensure!(
        rw >= factor && rh >= factor && rw <= (1u64 << 24) as f64 && rh <= (1u64 << 24) as f64,
        "resized coordinates exceed exact F32 integer range"
    );
    let (rw, rh) = (rw as usize, rh as usize);
    Ok((rw, rh))
}
