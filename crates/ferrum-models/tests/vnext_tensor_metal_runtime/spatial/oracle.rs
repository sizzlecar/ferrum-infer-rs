use super::*;
fn list(a: &BTreeMap<AttributeId, SemanticValue>, name: &str) -> Vec<usize> {
    match &a[&id::<AttributeId>(name)] {
        SemanticValue::Integers(v) => v.iter().map(|&x| usize::try_from(x).unwrap()).collect(),
        other => panic!("{other:?}"),
    }
}
fn uint(a: &BTreeMap<AttributeId, SemanticValue>, name: &str) -> usize {
    match a[&id::<AttributeId>(name)] {
        SemanticValue::Unsigned(x) => x as usize,
        _ => panic!("unsigned {name}"),
    }
}
fn ratio(a: &BTreeMap<AttributeId, SemanticValue>, name: &str) -> CanonicalRational {
    match a[&id::<AttributeId>(name)] {
        SemanticValue::Rational(x) => x,
        _ => panic!("rational {name}"),
    }
}
fn scalar(a: &BTreeMap<AttributeId, SemanticValue>, name: &str) -> f32 {
    let r = ratio(a, name);
    (r.numerator() as f64 / r.denominator() as f64) as f32
}
pub(crate) fn check(
    kind: MetalTensorOperation,
    attrs: &BTreeMap<AttributeId, SemanticValue>,
    inputs: &[(&ProgramTensorSpec, &[f32])],
    output: &ProgramTensorSpec,
    actual: &[f32],
) {
    let x = inputs[0].1;
    let dims: Vec<_> = inputs[0].0.dimensions.iter().map(|&n| n as usize).collect();
    let [batch, channels, height, width] = dims.as_slice() else {
        panic!("NCHW input");
    };
    let (batch, channels, height, width) = (*batch, *channels, *height, *width);
    let spatial = height * width;
    match kind {
        MetalTensorOperation::SpatialGroupMean => {
            let (sh, sw, frames) = (
                uint(attrs, "scale_h"),
                uint(attrs, "scale_w"),
                uint(attrs, "frame_factor"),
            );
            let oc = output.dimensions[1] as usize;
            let group = channels * frames * sh * sw / oc;
            for n in 0..batch {
                for y in 0..height / sh {
                    for col in 0..width / sw {
                        let expanded: Vec<_> = (0..channels)
                            .flat_map(|c| {
                                (0..frames).flat_map(move |frame| {
                                    (0..sh).flat_map(move |dy| {
                                        (0..sw).map(move |dx| {
                                            if frame + 1 == frames {
                                                x[((n * channels + c) * height + y * sh + dy)
                                                    * width
                                                    + col * sw
                                                    + dx]
                                            } else {
                                                0.0
                                            }
                                        })
                                    })
                                })
                            })
                            .collect();
                        for (out, values) in expanded.chunks_exact(group).enumerate() {
                            let i = ((n * oc + out) * (height / sh) + y) * (width / sw) + col;
                            let sum = values[1..]
                                .iter()
                                .fold(values[0] as f64, |sum, &v| sum + v as f64);
                            let exact = sum / group as f64;
                            if !exact.is_finite() {
                                if exact.is_nan() {
                                    assert!(actual[i].is_nan());
                                } else {
                                    assert_eq!(actual[i] as f64, exact);
                                }
                            } else if group == 1 {
                                assert_eq!(actual[i].to_bits(), values[0].to_bits());
                            } else {
                                let nu = (group - 1) as f64 * 2_f64.powi(-24);
                                let reference = group as f64 * f64::EPSILON;
                                let error = (nu / (1.0 - nu) + reference / (1.0 - reference))
                                    * values.iter().map(|v| (*v as f64).abs()).sum::<f64>();
                                let lo = rounds(((sum - error) as f32) / group as f32);
                                let hi = rounds(((sum + error) as f32) / group as f32);
                                assert!(
                                    actual[i].is_finite() && actual[i] >= lo && actual[i] <= hi,
                                    "group mean {i}: {} outside [{lo},{hi}]",
                                    actual[i]
                                );
                            }
                        }
                    }
                }
            }
        }
        MetalTensorOperation::ChannelL2Norm | MetalTensorOperation::ChannelL2NormImage => {
            transforms::oracle::channel_l2(
                x,
                inputs[1].1,
                batch,
                channels,
                spatial,
                scalar(attrs, "epsilon"),
                scalar(attrs, "scale"),
                actual,
            );
        }
        MetalTensorOperation::SpatialAttention => {
            let c = channels / 3;
            let mut values = vec![vec![0.0; batch * spatial * c]; 3];
            for n in 0..batch {
                for role in 0..3 {
                    for channel in 0..c {
                        for s in 0..spatial {
                            values[role][(n * spatial + s) * c + channel] =
                                x[((n * 3 + role) * c + channel) * spatial + s];
                        }
                    }
                }
            }
            let mut result = vec![0.0; actual.len()];
            for n in 0..batch {
                for channel in 0..c {
                    for s in 0..spatial {
                        result[(n * spatial + s) * c + channel] =
                            actual[(n * c + channel) * spatial + s];
                    }
                }
            }
            attention::Case {
                sequence: spatial,
                heads: 1,
                kv_heads: 1,
                width: c,
                segments: vec![(spatial, true)],
                keys: vec![vec![(spatial, true)]; batch],
                scale: ratio(attrs, "scale"),
            }
            .check(&values, &result);
        }
        MetalTensorOperation::SpatialUpsampleGather => {
            let sh = uint(attrs, "scale_h");
            let sw = uint(attrs, "scale_w");
            let stride = uint(attrs, "channel_stride");
            let offsets = list(attrs, "channel_offsets");
            let oc = output.dimensions[1] as usize;
            let mut reference = vec![0.0_f32; actual.len()];
            for n in 0..batch {
                for out in 0..oc {
                    for y in 0..height {
                        for col in 0..width {
                            for dy in 0..sh {
                                for dx in 0..sw {
                                    let source =
                                        ((n * channels + out * stride + offsets[dy * sw + dx])
                                            * height
                                            + y)
                                            * width
                                            + col;
                                    let target = ((n * oc + out) * (height * sh) + y * sh + dy)
                                        * (width * sw)
                                        + col * sw
                                        + dx;
                                    reference[target] = x[source];
                                }
                            }
                        }
                    }
                }
            }
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        MetalTensorOperation::Conv2d => {
            let weights = inputs[1].1;
            let bias = inputs[2].1;
            let wd = &inputs[1].0.dimensions;
            let (oc, kh, kw) = (wd[0] as usize, wd[2] as usize, wd[3] as usize);
            let pad = list(attrs, "padding");
            let stride = list(attrs, "stride");
            let dilation = list(attrs, "dilation");
            let groups = uint(attrs, "groups");
            let (oh, ow) = (output.dimensions[2] as usize, output.dimensions[3] as usize);
            let inner = channels / groups * kh * kw;
            assert_eq!(actual.len(), batch * oc * oh * ow);
            for n in 0..batch {
                for out in 0..oc {
                    for y in 0..oh {
                        for col in 0..ow {
                            let mut sum = 0.0_f64;
                            let mut absolute = 0.0_f64;
                            for c in 0..channels / groups {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (y * stride[0] + ky * dilation[0]) as i64
                                            - pad[0] as i64;
                                        let ix = (col * stride[1] + kx * dilation[1]) as i64
                                            - pad[2] as i64;
                                        if iy >= 0
                                            && ix >= 0
                                            && iy < height as i64
                                            && ix < width as i64
                                        {
                                            let source_c =
                                                (out / (oc / groups)) * (channels / groups) + c;
                                            let input = x[((n * channels + source_c) * height
                                                + iy as usize)
                                                * width
                                                + ix as usize];
                                            let w = weights[((out * (channels / groups) + c) * kh
                                                + ky)
                                                * kw
                                                + kx];
                                            let product = input as f64 * w as f64;
                                            sum += product;
                                            absolute += product.abs();
                                        }
                                    }
                                }
                            }
                            let nu = (2 * inner) as f64 * 2_f64.powi(-24);
                            let reference_nu = (2 * inner) as f64 * f64::EPSILON;
                            let error =
                                (nu / (1.0 - nu) + reference_nu / (1.0 - reference_nu)) * absolute;
                            let lo = rounds(rounds((sum - error) as f32) + bias[out]);
                            let hi = rounds(rounds((sum + error) as f32) + bias[out]);
                            let i = ((n * oc + out) * oh + y) * ow + col;
                            assert!(
                                actual[i].is_finite() && actual[i] >= lo && actual[i] <= hi,
                                "conv[{n},{out},{y},{col}] {} outside [{lo},{hi}]",
                                actual[i]
                            );
                        }
                    }
                }
            }
        }
        _ => panic!("not a spatial oracle: {kind:?}"),
    }
}
