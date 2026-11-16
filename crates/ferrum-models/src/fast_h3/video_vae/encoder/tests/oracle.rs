//! Independent F64 dense NCTHW oracle; does not inspect emitted nodes,
//! operation geometry, source mappings or provider arithmetic.
use super::*;
struct Volume {
    shape: [usize; 5],
    data: Vec<f64>,
}
impl Volume {
    fn at(&self, n: usize, c: usize, t: usize, h: usize, w: usize) -> f64 {
        let [_, channels, time, height, width] = self.shape;
        self.data[((((n * channels + c) * time + t) * height + h) * width) + w]
    }
    fn conv(
        &self,
        weights: &Weights,
        name: &str,
        temporal_stride: usize,
        spatial_stride: usize,
        spatial_padding: [usize; 2],
    ) -> Self {
        let (s, w) = &weights[&format!("{name}.weight")];
        let bias = &weights[&format!("{name}.bias")].1;
        let [n, c, t, h, width] = self.shape;
        let k = s[2];
        let temporal = k - 1;
        let shape = [
            n,
            s[0],
            t.div_ceil(temporal_stride),
            (h + spatial_padding.iter().sum::<usize>() - k) / spatial_stride + 1,
            (width + spatial_padding.iter().sum::<usize>() - k) / spatial_stride + 1,
        ];
        let reflect = |p: isize, extent: usize| {
            if p < 0 {
                (-p) as usize
            } else if p as usize >= extent {
                2 * extent - 2 - p as usize
            } else {
                p as usize
            }
        };
        let mut data = Vec::with_capacity(shape.iter().product());
        for b in 0..n {
            for o in 0..s[0] {
                for z in 0..shape[2] {
                    for y in 0..shape[3] {
                        for x in 0..shape[4] {
                            let mut dot = 0f64;
                            for i in 0..c {
                                for dt in 0..k {
                                    for dy in 0..k {
                                        for dx in 0..k {
                                            let time = (z * temporal_stride + dt) as isize
                                                - temporal as isize;
                                            if time < 0 {
                                                continue;
                                            }
                                            let yy = reflect(
                                                (y * spatial_stride + dy) as isize
                                                    - spatial_padding[0] as isize,
                                                h,
                                            );
                                            let xx = reflect(
                                                (x * spatial_stride + dx) as isize
                                                    - spatial_padding[0] as isize,
                                                width,
                                            );
                                            let index = (((o * c + i) * k + dt) * k + dy) * k + dx;
                                            dot += self.at(b, i, time as usize, yy, xx)
                                                * f64::from(w[index]);
                                        }
                                    }
                                }
                            }
                            data.push(dot + f64::from(bias[o]));
                        }
                    }
                }
            }
        }
        Self { shape, data }
    }
    fn norm(&self, weights: &Weights, name: &str) -> Self {
        let gamma = &weights[&format!("{name}.weight")].1;
        let beta = &weights[&format!("{name}.bias")].1;
        let [n, c, t, h, w] = self.shape;
        let mut data = vec![0.; self.data.len()];
        for b in 0..n {
            for time in 0..t {
                for group in 0..2 {
                    let group_channels = group * (c / 2)..(group + 1) * (c / 2);
                    let values: Vec<_> = group_channels
                        .clone()
                        .flat_map(|channel| {
                            (0..h).flat_map(move |y| {
                                (0..w).map(move |x| self.at(b, channel, time, y, x))
                            })
                        })
                        .collect();
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    for channel in group_channels {
                        for y in 0..h {
                            for x in 0..w {
                                let affine = (self.at(b, channel, time, y, x) - mean)
                                    / (variance + f64::from(1e-6f32)).sqrt()
                                    * f64::from(gamma[channel])
                                    + f64::from(beta[channel]);
                                data[(((b * c + channel) * t + time) * h + y) * w + x] =
                                    affine / (1. + (-affine).exp());
                            }
                        }
                    }
                }
            }
        }
        Self {
            shape: self.shape,
            data,
        }
    }
    fn residual(self, weights: &Weights, name: &str, channels: usize) -> Self {
        let h = self.norm(weights, &format!("{name}.norm1")).conv(
            weights,
            &format!("{name}.conv1"),
            1,
            1,
            [1, 1],
        );
        let mut h = h.norm(weights, &format!("{name}.norm2")).conv(
            weights,
            &format!("{name}.conv2"),
            1,
            1,
            [1, 1],
        );
        let skip = if self.shape[1] == channels {
            self
        } else {
            self.conv(weights, &format!("{name}.conv_shortcut"), 1, 1, [0, 0])
        };
        assert_eq!(h.shape, skip.shape);
        for (v, x) in h.data.iter_mut().zip(skip.data) {
            *v += x;
        }
        h
    }
}
pub(super) fn encode(weights: &Weights, input: &[f32], g: FastH3VideoEncoderGeometry) -> Vec<f64> {
    encode_with_final_downsample(weights, input, g, true)
}
pub(in crate::fast_h3::video_vae) fn encode_with_final_downsample(
    weights: &Weights,
    input: &[f32],
    g: FastH3VideoEncoderGeometry,
    final_downsample: bool,
) -> Vec<f64> {
    let mut x = Volume {
        shape: [
            g.batch as usize,
            3,
            g.frames as usize,
            g.height as usize,
            g.width as usize,
        ],
        data: input.iter().map(|v| f64::from(*v)).collect(),
    };
    x = x.conv(weights, "encoder.conv_in", 1, 1, [1, 1]);
    for (stage, channels, stride, pad) in [(0, 2, 2, [0, 1]), (1, 4, 1, [0, 0])] {
        for block in 0..2 {
            x = x.residual(
                weights,
                &format!("encoder.down_blocks.{stage}.resnets.{block}"),
                channels,
            );
        }
        if stage == 0 || final_downsample {
            x = x.conv(
                weights,
                &format!("encoder.down_blocks.{stage}.downsamplers.0.conv"),
                2,
                stride,
                pad,
            );
        }
    }
    x.norm(weights, "encoder.norm_out")
        .conv(weights, "encoder.conv_out", 1, 1, [1, 1])
        .conv(weights, "quant_conv", 1, 1, [0, 0])
        .data
}
