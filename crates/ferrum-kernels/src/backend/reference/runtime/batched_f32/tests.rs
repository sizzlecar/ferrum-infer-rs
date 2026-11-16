use super::super::tests::{buffer, config};
use super::*;
use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};
use std::collections::BTreeMap;
fn spec(shape: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        shape.to_vec(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn region(runtime: &ReferenceDeviceRuntime, name: &str, values: &[f32]) -> ReferenceBufferRegion {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    buffer(runtime, name, &bytes, 16, ElementType::F32)
        .region(0..bytes.len() as u64)
        .unwrap()
}
fn submit(runtime: &ReferenceDeviceRuntime, launch: ReferenceBatchedF32Launch) -> Vec<f32> {
    let output = launch.output.clone();
    let command = ReferenceDeviceCommand::batched_f32_tensor(launch).unwrap();
    let mut stream = runtime.create_stream().unwrap();
    runtime
        .submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), command)],
            false,
        )
        .unwrap();
    output
        .read()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
#[test]
fn batched_products_use_both_axes_and_both_rhs_layouts_without_broadcast() {
    let rt = ReferenceDeviceRuntime::new(config()).unwrap();
    let a = (0..2 * 3 * 5 * 7)
        .map(|i| (i % 17) as f32 / 8. - 1.)
        .collect::<Vec<_>>();
    let b = (0..2 * 3 * 7 * 11)
        .map(|i| (i % 13) as f32 / 16. - 0.5)
        .collect::<Vec<_>>();
    let mut transposed = vec![0.; b.len()];
    let mut expected = vec![0.; 2 * 3 * 5 * 11];
    for bg in 0..6 {
        for n in 0..11 {
            for k in 0..7 {
                transposed[(bg * 11 + n) * 7 + k] = b[(bg * 7 + k) * 11 + n];
            }
        }
        for m in 0..5 {
            for n in 0..11 {
                expected[(bg * 5 + m) * 11 + n] = (0..7)
                    .map(|k| {
                        f64::from(a[(bg * 5 + m) * 7 + k]) * f64::from(b[(bg * 7 + k) * 11 + n])
                    })
                    .sum::<f64>() as f32;
            }
        }
    }
    for transpose in [false, true] {
        let rhs = if transpose { &transposed } else { &b };
        let shape = if transpose {
            [2, 3, 11, 7]
        } else {
            [2, 3, 7, 11]
        };
        let g = BatchedF32Geometry::new(
            BatchedF32Op::Matmul,
            &[&spec(&[2, 3, 5, 7]), &spec(&shape)],
            &spec(&[2, 3, 5, 11]),
            &BatchedF32Geometry::matmul_attributes(transpose).unwrap(),
        )
        .unwrap();
        let actual = submit(
            &rt,
            ReferenceBatchedF32Launch {
                inputs: vec![
                    region(&rt, "resource.batched.a", &a),
                    region(&rt, "resource.batched.b", rhs),
                ],
                output: region(&rt, "resource.batched.y", &vec![f32::NAN; expected.len()]),
                geometry: g,
            },
        );
        assert_eq!(actual, expected);
    }
}
#[test]
fn scale_and_softmax_have_explicit_f32_nonfinite_and_underflow_boundaries() {
    let rt = ReferenceDeviceRuntime::new(config()).unwrap();
    let x = [
        0f32,
        1.,
        -1.,
        f32::NEG_INFINITY,
        2.,
        2.,
        2.,
        2.,
        0.,
        f32::NAN,
        0.,
        0.,
        0.,
        f32::INFINITY,
        0.,
        0.,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let t = spec(&[1, 1, 5, 4]);
    let g = BatchedF32Geometry::new(BatchedF32Op::Softmax, &[&t], &t, &BTreeMap::new()).unwrap();
    let y = submit(
        &rt,
        ReferenceBatchedF32Launch {
            inputs: vec![region(&rt, "resource.softmax.x", &x)],
            output: region(&rt, "resource.softmax.y", &[0.; 20]),
            geometry: g,
        },
    );
    let sum = 1f64 + 1f64.exp() + (-1f64).exp();
    for i in 0..3 {
        assert!((f64::from(y[i]) - f64::from(x[i]).exp() / sum).abs() < 1e-7);
    }
    assert_eq!(y[3].to_bits(), 0);
    assert_eq!(&y[4..8], &[0.25; 4]);
    assert!(y[8..].iter().all(|v| v.is_nan()));
    let t = spec(&[1, 1, 1, 4]);
    let g = BatchedF32Geometry::new(
        BatchedF32Op::Scale,
        &[&t],
        &t,
        &BatchedF32Geometry::scale_attributes(0.5).unwrap(),
    )
    .unwrap();
    let x = [f32::from_bits(2), -0., f32::MAX, f32::INFINITY];
    let y = submit(
        &rt,
        ReferenceBatchedF32Launch {
            inputs: vec![region(&rt, "resource.scale.x", &x)],
            output: region(&rt, "resource.scale.y", &[0.; 4]),
            geometry: g,
        },
    );
    assert_eq!(
        y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [
            1,
            0x80000000,
            (f32::MAX * 0.5).to_bits(),
            f32::INFINITY.to_bits()
        ]
    );
}
#[test]
fn batched_preflight_rejects_alias_wrong_storage_and_foreign_runtime_before_writes() {
    let rt = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let t = spec(&[1, 1, 2, 2]);
    let g = BatchedF32Geometry::new(BatchedF32Op::Softmax, &[&t], &t, &BTreeMap::new()).unwrap();
    for case in 0..4 {
        let input = region(&rt, "resource.invalid.x", &[1.; 4]);
        let mut l = ReferenceBatchedF32Launch {
            inputs: vec![input],
            output: region(&rt, "resource.invalid.y", &[19.; 4]),
            geometry: g,
        };
        match case {
            0 => l.output = l.inputs[0].clone(),
            1 => l.inputs[0].element_type = ElementType::I32,
            2 => l.output.length_bytes -= 4,
            3 => l.inputs[0].offset_bytes += 1,
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::batched_f32_tensor(l).is_err());
    }
    let l = ReferenceBatchedF32Launch {
        inputs: vec![region(&foreign, "resource.foreign.x", &[1.; 4])],
        output: region(&rt, "resource.foreign.y", &[19.; 4]),
        geometry: g,
    };
    let output = l.output.clone();
    let command = ReferenceDeviceCommand::batched_f32_tensor(l).unwrap();
    let mut stream = rt.create_stream().unwrap();
    assert!(rt
        .submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), command)],
            false
        )
        .is_err());
    assert_eq!(rt.snapshot().submissions, 0);
    assert!(output
        .read()
        .chunks_exact(4)
        .all(|b| b == 19f32.to_le_bytes()));
}
