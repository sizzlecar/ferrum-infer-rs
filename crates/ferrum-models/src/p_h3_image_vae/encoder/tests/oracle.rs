//! Literal full-CNN oracle: original F32 5D last temporal tap, F64 spatial
//! correlation/group statistics, explicit F32 boundaries. No production
//! geometry/source packing/provider implementation is used for mathematics.
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
fn reflect(mut p: i64, n: usize) -> usize {
    while p < 0 || p >= n as i64 {
        p = if p < 0 { -p } else { 2 * n as i64 - 2 - p };
    }
    p as usize
}
fn conv(
    source: &SafetensorsArchive,
    name: &str,
    x: &[f32],
    shape: [usize; 4],
    stride: usize,
    pad: [usize; 4],
) -> (Vec<f32>, [usize; 4]) {
    let key = format!("{name}.weight");
    let t = source.tensor(&key).unwrap();
    let d = t.shape();
    let (out, inc, time, kh, kw) = (
        d[0] as usize,
        d[1] as usize,
        d[2] as usize,
        d[3] as usize,
        d[4] as usize,
    );
    assert_eq!(inc, shape[1]);
    let w = read(source, &key);
    let bias = read(source, &format!("{name}.bias"));
    let [batch, _, height, width] = shape;
    let oh = (height + pad[0] + pad[1] - kh) / stride + 1;
    let ow = (width + pad[2] + pad[3] - kw) / stride + 1;
    let mut y = vec![0.; batch * out * oh * ow];
    for b in 0..batch {
        for o in 0..out {
            for h in 0..oh {
                for z in 0..ow {
                    let mut sum = 0.0f64;
                    for c in 0..inc {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = reflect((h * stride + ky) as i64 - pad[0] as i64, height);
                                let ix = reflect((z * stride + kx) as i64 - pad[2] as i64, width);
                                sum += f64::from(x[((b * inc + c) * height + iy) * width + ix])
                                    * f64::from(
                                        w[(((o * inc + c) * time + time - 1) * kh + ky) * kw + kx],
                                    );
                            }
                        }
                    }
                    let rounded = sum as f32;
                    y[((b * out + o) * oh + h) * ow + z] = rounded + bias[o];
                }
            }
        }
    }
    (y, [batch, out, oh, ow])
}
fn norm(
    source: &SafetensorsArchive,
    name: &str,
    x: &[f32],
    shape: [usize; 4],
    a: &PH3ImageVaeEncoderArchitecture,
) -> Vec<f32> {
    let gamma = read(source, &format!("{name}.weight"));
    let beta = read(source, &format!("{name}.bias"));
    let [batch, channels, height, width] = shape;
    let groups = a.norm_groups() as usize;
    let spatial = height * width;
    let count = channels / groups * spatial;
    let eps = a.epsilon();
    let eps = eps.numerator() as f64 / eps.denominator() as f64;
    let mut y = vec![0.; x.len()];
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
            let n = ((f64::from(row[i]) - mean) * inverse) as f32;
            let scaled = n * gamma[channel];
            let affine = scaled + beta[channel];
            y[start + i] = (f64::from(affine) / (1. + (-f64::from(affine)).exp())) as f32;
        }
    }
    y
}
pub(super) fn capture(
    c: &PH3ImageVaeEncoderConfig,
    source: &SafetensorsArchive,
    x: &[f32],
) -> Values {
    let mut values = Values::new();
    let mut save = |name: &str, data: &[f32]| {
        values.insert(
            id::<ProgramValueId>(format!("value.p_h3_image.vae.encoder.{name}")),
            data.to_vec(),
        );
    };
    let a = c.architecture();
    let shape = [
        c.batch() as usize,
        a.in_channels() as usize,
        c.height() as usize,
        c.width() as usize,
    ];
    let (mut hidden, mut shape) = conv(source, "encoder.conv_in", x, shape, 1, [1; 4]);
    save("encoder.conv_in", &hidden);
    for (level, &channels) in a.channels().iter().enumerate() {
        for block in 0..a.layers_per_block() {
            let prefix = format!("encoder.down.{level}.block.{block}");
            let n1 = norm(source, &format!("{prefix}.norm1"), &hidden, shape, a);
            save(&format!("{prefix}.norm1"), &n1);
            let (h, s) = conv(source, &format!("{prefix}.conv1"), &n1, shape, 1, [1; 4]);
            save(&format!("{prefix}.conv1"), &h);
            let shortcut = if shape[1] != channels as usize {
                let (v, ss) = conv(
                    source,
                    &format!("{prefix}.nin_shortcut"),
                    &hidden,
                    shape,
                    1,
                    [0; 4],
                );
                assert_eq!(ss, s);
                save(&format!("{prefix}.nin_shortcut"), &v);
                v
            } else {
                hidden
            };
            let n2 = norm(source, &format!("{prefix}.norm2"), &h, s, a);
            save(&format!("{prefix}.norm2"), &n2);
            let (h, s) = conv(source, &format!("{prefix}.conv2"), &n2, s, 1, [1; 4]);
            save(&format!("{prefix}.conv2"), &h);
            hidden = h.iter().zip(&shortcut).map(|(a, b)| a + b).collect();
            shape = s;
            save(&format!("{prefix}.residual"), &hidden);
        }
        if a.spatial_downsample_factors()[level] > 1 || a.temporal_downsample_factors()[level] > 1 {
            let stride = a.spatial_downsample_factors()[level] as usize;
            let pad = if stride == 2 { [0, 1, 0, 1] } else { [0; 4] };
            let name = format!("encoder.down.{level}.downsample.conv");
            (hidden, shape) = conv(source, &name, &hidden, shape, stride, pad);
            save(&name, &hidden);
        }
    }
    hidden = norm(source, "encoder.norm_out", &hidden, shape, a);
    save("encoder.norm_out", &hidden);
    (hidden, shape) = conv(source, "encoder.conv_out", &hidden, shape, 1, [1; 4]);
    save("encoder.conv_out", &hidden);
    let (moments, _) = conv(source, "quant_conv", &hidden, shape, 1, [0; 4]);
    values.insert(id(Family::OUTPUT_ID), moments);
    values
}
