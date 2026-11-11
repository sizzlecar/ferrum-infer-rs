//! Independent F64 statistics/correlation oracle. It reads original 5D source
//! tensors and selects their literal last temporal tap, not the packed source
//! adapter or provider geometry/index helpers.
use super::*;
fn read(source: &SafetensorsArchive, name: &str) -> Vec<f32> {
    source
        .tensor(name)
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn reflected(mut p: i64, n: usize) -> usize {
    while p < 0 || p >= n as i64 {
        p = if p < 0 { -p } else { 2 * n as i64 - 2 - p };
    }
    p as usize
}
fn conv(
    source: &SafetensorsArchive,
    module: &str,
    x: &[f32],
    shape: [usize; 4],
    stride: usize,
    padding: [usize; 4],
) -> (Vec<f32>, [usize; 4]) {
    let weight_name = format!("{module}.weight");
    let t = source.tensor(&weight_name).unwrap();
    let d = t.shape();
    let (oc, ic, kt, kh, kw) = (
        d[0] as usize,
        d[1] as usize,
        d[2] as usize,
        d[3] as usize,
        d[4] as usize,
    );
    assert_eq!(shape[1], ic);
    let w = read(source, &weight_name);
    let bias = read(source, &format!("{module}.bias"));
    let [batch, _, height, width] = shape;
    let out_h = (height + padding[0] + padding[1] - kh) / stride + 1;
    let out_w = (width + padding[2] + padding[3] - kw) / stride + 1;
    let mut out = vec![0.; batch * oc * out_h * out_w];
    for b in 0..batch {
        for o in 0..oc {
            for y in 0..out_h {
                for z in 0..out_w {
                    let mut sum = 0.0f64;
                    for c in 0..ic {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy =
                                    reflected((y * stride + ky) as i64 - padding[0] as i64, height);
                                let ix =
                                    reflected((z * stride + kx) as i64 - padding[2] as i64, width);
                                let weight = (((o * ic + c) * kt + (kt - 1)) * kh + ky) * kw + kx;
                                let input = ((b * ic + c) * height + iy) * width + ix;
                                sum += f64::from(x[input]) * f64::from(w[weight]);
                            }
                        }
                    }
                    let rounded = sum as f32;
                    out[((b * oc + o) * out_h + y) * out_w + z] = rounded + bias[o];
                }
            }
        }
    }
    (out, [batch, oc, out_h, out_w])
}
fn norm_silu(
    source: &SafetensorsArchive,
    module: &str,
    x: &[f32],
    shape: [usize; 4],
    groups: usize,
    eps: f64,
) -> Vec<f32> {
    let gamma = read(source, &format!("{module}.weight"));
    let beta = read(source, &format!("{module}.bias"));
    let [batch, channels, height, width] = shape;
    let spatial = height * width;
    let count = channels / groups * spatial;
    let mut out = vec![0.; x.len()];
    for group in 0..batch * groups {
        let start = group * count;
        let row = &x[start..start + count];
        let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / count as f64;
        let variance = row
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        let inverse = 1. / (variance + eps).sqrt();
        for i in 0..count {
            let channel = (start + i) / spatial % channels;
            let normalized = ((f64::from(row[i]) - mean) * inverse) as f32;
            let scaled = normalized * gamma[channel];
            let affine = scaled + beta[channel];
            out[start + i] = (f64::from(affine) / (1. + (-f64::from(affine)).exp())) as f32;
        }
    }
    out
}
pub(super) fn capture(
    c: &PH3ImageVaeEncoderVerticalConfig,
    source: &SafetensorsArchive,
    x: &[f32],
) -> Values {
    let mut out = Values::new();
    let value = |name: &str| {
        ProgramValueId::new(format!("value.p_h3_image.vae.encoder_vertical.{name}")).unwrap()
    };
    let mut save = |name: &str, v: &[f32]| {
        out.insert(value(name), v.to_vec());
    };
    let shape = [
        c.batch() as usize,
        c.architecture().in_channels() as usize,
        c.height() as usize,
        c.width() as usize,
    ];
    let prefix = format!("encoder.down.{}.block.0", c.block_level());
    let eps = c.architecture().epsilon();
    let epsilon = eps.numerator() as f64 / eps.denominator() as f64;
    let groups = c.architecture().norm_groups() as usize;
    let (stem, stem_shape) = conv(source, "encoder.conv_in", x, shape, 1, [1; 4]);
    save("stem", &stem);
    let norm1 = norm_silu(
        source,
        &format!("{prefix}.norm1"),
        &stem,
        stem_shape,
        groups,
        epsilon,
    );
    save("norm1", &norm1);
    let (conv1, block_shape) = conv(
        source,
        &format!("{prefix}.conv1"),
        &norm1,
        stem_shape,
        1,
        [1; 4],
    );
    save("conv1", &conv1);
    let (shortcut, shortcut_shape) = conv(
        source,
        &format!("{prefix}.nin_shortcut"),
        &stem,
        stem_shape,
        1,
        [0; 4],
    );
    save("shortcut", &shortcut);
    assert_eq!(shortcut_shape, block_shape);
    let norm2 = norm_silu(
        source,
        &format!("{prefix}.norm2"),
        &conv1,
        block_shape,
        groups,
        epsilon,
    );
    save("norm2", &norm2);
    let (conv2, shape) = conv(
        source,
        &format!("{prefix}.conv2"),
        &norm2,
        block_shape,
        1,
        [1; 4],
    );
    save("conv2", &conv2);
    let residual = conv2
        .iter()
        .zip(&shortcut)
        .map(|(x, y)| x + y)
        .collect::<Vec<_>>();
    save("residual", &residual);
    let (down, _) = conv(
        source,
        &format!("encoder.down.{}.downsample.conv", c.block_level()),
        &residual,
        shape,
        2,
        [0, 1, 0, 1],
    );
    out.insert(id(Family::OUTPUT_ID), down);
    out
}
