use super::super::tests::{buffer, config};
use super::*;
use ferrum_runtime::{AttributeId, ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
use std::collections::BTreeMap;
fn geometry(
    input: [u64; 4],
    weight: [u64; 4],
    stride: [i64; 2],
    pad: [i64; 2],
    dilation: [i64; 2],
    extra: [i64; 2],
    groups: u64,
) -> ConvTranspose2dF32Geometry {
    let spec = |dims: &[u64]| {
        ResolvedTensorSpec::new(
            dims.to_vec(),
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    };
    let mut output = [input[0], weight[1] * groups, 0, 0];
    for a in 0..2 {
        output[2 + a] = ((input[2 + a] as i64 - 1) * stride[a] - 2 * pad[a]
            + (weight[2 + a] as i64 - 1) * dilation[a]
            + extra[a]
            + 1) as u64;
    }
    let mut attrs = BTreeMap::new();
    for (name, values) in [
        ("stride", stride),
        ("padding", pad),
        ("dilation", dilation),
        ("output_padding", extra),
    ] {
        attrs.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(values.to_vec()),
        );
    }
    attrs.insert(
        AttributeId::new("groups").unwrap(),
        SemanticValue::Unsigned(groups),
    );
    ConvTranspose2dF32Geometry::new(
        &spec(&input),
        &spec(&weight),
        &spec(&[output[1]]),
        &spec(&output),
        &attrs,
    )
    .unwrap()
}
fn make(
    rt: &ReferenceDeviceRuntime,
    g: ConvTranspose2dF32Geometry,
    values: &[Vec<f32>; 3],
) -> ReferenceConvTranspose2dF32Launch {
    let inputs = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let bytes = v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
            buffer(
                rt,
                &format!("resource.transpose.input.{i}"),
                &bytes,
                16,
                ElementType::F32,
            )
            .region(0..bytes.len() as u64)
            .unwrap()
        })
        .collect();
    let output = buffer(
        rt,
        "resource.transpose.output",
        &vec![0xa5; g.elements() as usize * 4],
        16,
        ElementType::F32,
    )
    .region(0..g.elements() * 4)
    .unwrap();
    ReferenceConvTranspose2dF32Launch {
        inputs,
        output,
        geometry: g,
    }
}
fn values(g: ConvTranspose2dF32Geometry) -> [Vec<f32>; 3] {
    std::array::from_fn(|j| {
        (0..g.input_elements(j).unwrap())
            .map(|i| ((i * 7 + j as u64 * 3) % 17) as f32 / 16. - 0.5)
            .collect()
    })
}
// Independent adjoint scatter over input coordinates in F64, unlike the
// production inverse-coordinate output gather. Dyadic data make exact checks useful.
fn oracle(g: ConvTranspose2dF32Geometry, v: &[Vec<f32>; 3]) -> Vec<f32> {
    let [n, ci, ih, iw] = g.input();
    let [_, co, oh, ow] = g.output();
    let [_, cog, kh, kw] = g.weight();
    let [ph, pw] = g.padding();
    let [sh, sw] = g.stride();
    let [dh, dw] = g.dilation();
    let mut out = vec![0f64; g.elements() as usize];
    for b in 0..n {
        for c in 0..ci {
            for y in 0..ih {
                for x in 0..iw {
                    let value = f64::from(v[0][(((b * ci + c) * ih + y) * iw + x) as usize]);
                    for oc in 0..cog {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let oy = (y * sh + ky * dh) as i64 - ph as i64;
                                let ox = (x * sw + kx * dw) as i64 - pw as i64;
                                if oy >= 0 && ox >= 0 && oy < oh as i64 && ox < ow as i64 {
                                    let o = c / (ci / g.groups()) * cog + oc;
                                    out[(((b * co + o) * oh + oy as u64) * ow + ox as u64)
                                        as usize] += value
                                        * f64::from(
                                            v[1][(((c * cog + oc) * kh + ky) * kw + kx) as usize],
                                        );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(i, x)| (x + f64::from(v[2][(i as u64 / (oh * ow) % co) as usize])) as f32)
        .collect()
}
#[test]
fn conv_transpose2d_f32_managed_execution_matches_independent_adjoint_scatter() {
    let rt = ReferenceDeviceRuntime::new(config()).unwrap();
    for g in [
        geometry(
            [2, 4, 1, 3],
            [4, 3, 1, 9],
            [1, 5],
            [0, 2],
            [1, 1],
            [0, 0],
            2,
        ),
        geometry(
            [1, 2, 3, 4],
            [2, 3, 3, 3],
            [2, 3],
            [1, 1],
            [2, 1],
            [1, 2],
            1,
        ),
    ] {
        let v = values(g);
        let l = make(&rt, g, &v);
        let out = l.output.clone();
        let before = rt.snapshot().allocations;
        let command = ReferenceDeviceCommand::conv_transpose2d_f32_tensor(l).unwrap();
        let mut stream = rt.create_stream().unwrap();
        rt.submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), command)],
            false,
        )
        .unwrap();
        let actual = out
            .read()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, oracle(g, &v));
        assert_eq!(rt.snapshot().allocations, before);
    }
    assert_eq!(rt.snapshot().live_allocations, 0);
}
#[test]
fn conv_transpose2d_f32_rejects_alias_physical_extent_and_foreign_owner_before_writes() {
    let rt = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let g = geometry(
        [1, 2, 1, 4],
        [2, 2, 1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        1,
    );
    let v = values(g);
    for case in 0..4 {
        let mut l = make(&rt, g, &v);
        let out = l.output.clone();
        match case {
            0 => l.output = l.inputs[0].clone(),
            1 => l.inputs[0].element_type = ElementType::Bf16,
            2 => l.inputs[1].length_bytes -= 4,
            3 => l.output.length_bytes -= 4,
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::conv_transpose2d_f32_tensor(l).is_err());
        assert!(out.read().iter().all(|&x| x == 0xa5));
    }
    let mut l = make(&rt, g, &v);
    let out = l.output.clone();
    l.inputs[2] = make(&foreign, g, &v).inputs.pop().unwrap();
    let command = ReferenceDeviceCommand::conv_transpose2d_f32_tensor(l).unwrap();
    let mut stream = rt.create_stream().unwrap();
    assert!(rt
        .submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), command)],
            false
        )
        .is_err());
    assert_eq!(rt.snapshot().submissions, 0);
    assert!(out.read().iter().all(|&x| x == 0xa5));
}
