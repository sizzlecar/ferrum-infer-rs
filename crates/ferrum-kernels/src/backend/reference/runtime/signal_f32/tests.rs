use super::super::tests::{buffer, config};
use super::*;
use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};
use std::collections::BTreeMap;
fn t(d: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        d.to_vec(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn make(
    rt: &ReferenceDeviceRuntime,
    g: SignalF32Geometry,
    values: &[Vec<f32>],
) -> ReferenceSignalF32Launch {
    let inputs = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
            buffer(
                rt,
                &format!("resource.signal.input.{i}"),
                &bytes,
                16,
                ElementType::F32,
            )
            .region(0..bytes.len() as u64)
            .unwrap()
        })
        .collect();
    let bytes = g.output_elements() * 4;
    let output = buffer(
        rt,
        "resource.signal.output",
        &vec![0xa5; bytes as usize],
        16,
        ElementType::F32,
    )
    .region(0..bytes)
    .unwrap();
    ReferenceSignalF32Launch {
        inputs,
        output,
        geometry: g,
    }
}
fn submit(rt: &ReferenceDeviceRuntime, l: ReferenceSignalF32Launch) -> Vec<f32> {
    let out = l.output.clone();
    let before = rt.snapshot().allocations;
    let command = ReferenceDeviceCommand::signal_f32_tensor(l).unwrap();
    let mut stream = rt.create_stream().unwrap();
    rt.submit_entries(
        &mut stream,
        vec![(DeviceCommandPhase::Compute, Some(0), command)],
        false,
    )
    .unwrap();
    assert_eq!(rt.snapshot().allocations, before);
    out.read()
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
#[test]
fn fir_replication_upsample_crop_and_downsample_match_independent_f64_expanded_signal() {
    let rt = ReferenceDeviceRuntime::new(config()).unwrap();
    let values = (0..12).map(|i| (i as f32 - 5.) / 8.).collect::<Vec<_>>();
    let filter = vec![-0.25, 0.5, 0.75];
    for transpose in [false, true] {
        let padding = [2, 1];
        let crop = if transpose { [2, 3] } else { [0, 0] };
        let width = if transpose { 8 } else { 2 };
        let g = SignalF32Geometry::new(
            SignalF32Op::FirResample,
            &[&t(&[2, 2, 1, 3]), &t(&[3])],
            &t(&[2, 2, 1, width]),
            &SignalF32Geometry::fir_attributes(transpose, 2, padding, crop).unwrap(),
        )
        .unwrap();
        let actual = submit(&rt, make(&rt, g, &[values.clone(), filter.clone()]));
        let mut expected = Vec::new();
        for channel in values.chunks_exact(3) {
            let padded = [
                channel[0], channel[0], channel[0], channel[1], channel[2], channel[2],
            ];
            if transpose {
                let mut full = vec![0f64; 13];
                for (j, &x) in padded.iter().enumerate() {
                    for (k, &w) in filter.iter().enumerate() {
                        full[j * 2 + k] += f64::from(x) * f64::from(w);
                    }
                }
                expected.extend(full[2..10].iter().map(|&x| x as f32));
            } else {
                for j in [0usize, 2] {
                    expected.push(
                        padded[j..j + 3]
                            .iter()
                            .zip(&filter)
                            .map(|(&x, &w)| f64::from(x) * f64::from(w))
                            .sum::<f64>() as f32,
                    );
                }
            }
        }
        assert_eq!(actual, expected);
    }
}
#[test]
fn snake_beta_uses_per_channel_log_parameters_and_clamp_preserves_in_range_bits() {
    let rt = ReferenceDeviceRuntime::new(config()).unwrap();
    let shape = t(&[2, 2, 1, 3]);
    let logs = vec![0., 2f32.ln()];
    let x = vec![-2., -0.5, 0., 0.25, 1.5, 2., -1., 0., 1., -0.25, 0.75, 2.5];
    let g = SignalF32Geometry::new(
        SignalF32Op::SnakeBetaLog,
        &[&shape, &t(&[2]), &t(&[2])],
        &shape,
        &BTreeMap::new(),
    )
    .unwrap();
    let actual = submit(&rt, make(&rt, g, &[x.clone(), logs.clone(), vec![0., 0.]]));
    for (i, (&x, &y)) in x.iter().zip(&actual).enumerate() {
        let alpha = f64::from(logs[(i / 3) % 2]).exp();
        let expected = f64::from(x) + (alpha * f64::from(x)).sin().powi(2) / (1. + 1e-9);
        assert!((f64::from(y) - expected).abs() < 3e-6);
    }
    let x = vec![
        -f32::INFINITY,
        -2.,
        -0.,
        0.,
        2.,
        f32::INFINITY,
        f32::from_bits(0x7fc12345),
        0.5,
    ];
    let shape = t(&[1, 1, 1, 8]);
    let g = SignalF32Geometry::new(
        SignalF32Op::Clamp,
        &[&shape],
        &shape,
        &SignalF32Geometry::clamp_attributes(-1., 1.).unwrap(),
    )
    .unwrap();
    let y = submit(&rt, make(&rt, g, &[x]));
    assert_eq!(
        y.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        [-1f32, -1., -0., 0., 1., 1., f32::from_bits(0x7fc12345), 0.5]
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
}
#[test]
fn direct_snake_alpha_preserves_independent_channels_including_zero_and_negative_alpha() {
    let rt = ReferenceDeviceRuntime::new(config()).unwrap();
    let shape = t(&[2, 3, 2, 3]);
    let alpha = vec![0.0, -0.5, 2.0];
    let x = (0..36)
        .map(|i| (i as f32 - 17.0) / 16.0)
        .collect::<Vec<_>>();
    let g = SignalF32Geometry::new(
        SignalF32Op::SnakeAlpha,
        &[&shape, &t(&[3])],
        &shape,
        &BTreeMap::new(),
    )
    .unwrap();
    let actual = submit(&rt, make(&rt, g, &[x.clone(), alpha.clone()]));
    for (i, (&x, &actual)) in x.iter().zip(&actual).enumerate() {
        let a = f64::from(alpha[i / 6 % 3]);
        let expected = f64::from(x) + (a * f64::from(x)).sin().powi(2) / (a + 1.0e-9);
        assert!((f64::from(actual) - expected).abs() <= 3.0e-7);
        if a == 0.0 {
            assert_eq!(actual.to_bits(), x.to_bits());
        }
    }
    assert!(SignalF32Geometry::new(
        SignalF32Op::SnakeAlpha,
        &[&shape, &t(&[1, 3, 1])],
        &shape,
        &BTreeMap::new()
    )
    .is_err());
    assert!(SignalF32Geometry::new(
        SignalF32Op::SnakeAlpha,
        &[&shape, &t(&[3]), &t(&[3])],
        &shape,
        &BTreeMap::new()
    )
    .is_err());
}

#[test]
fn signal_preflight_rejects_physical_mismatch_alias_and_foreign_owner() {
    let rt = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let shape = t(&[1, 1, 1, 4]);
    let g = SignalF32Geometry::new(
        SignalF32Op::Clamp,
        &[&shape],
        &shape,
        &SignalF32Geometry::clamp_attributes(-1., 1.).unwrap(),
    )
    .unwrap();
    let mut l = make(&rt, g, &[vec![0.; 4]]);
    l.output = l.inputs[0].clone();
    assert!(ReferenceDeviceCommand::signal_f32_tensor(l).is_err());
    let mut l = make(&rt, g, &[vec![0.; 4]]);
    l.inputs[0].element_type = ElementType::I32;
    assert!(ReferenceDeviceCommand::signal_f32_tensor(l).is_err());
    let mut l = make(&rt, g, &[vec![0.; 4]]);
    let output = l.output.clone();
    l.inputs = make(&foreign, g, &[vec![0.; 4]]).inputs;
    let command = ReferenceDeviceCommand::signal_f32_tensor(l).unwrap();
    let mut stream = rt.create_stream().unwrap();
    assert!(rt
        .submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), command)],
            false
        )
        .is_err());
    assert!(output.read().iter().all(|&x| x == 0xa5));
}
