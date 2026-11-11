use super::super::tests::{buffer, config};
use super::*;
use ferrum_runtime::{
    AttributeId, CanonicalRational, F32CnnOp, ResolvedTensorLayout, ResolvedTensorSpec,
    SemanticValue,
};
use std::collections::BTreeMap;
fn launch(runtime: &ReferenceDeviceRuntime, kind: F32CnnOp) -> ReferenceCnnF32Launch {
    let t = |shape: Vec<u64>| {
        ResolvedTensorSpec::new(shape, ElementType::F32, ResolvedTensorLayout::Contiguous).unwrap()
    };
    let x = t(vec![2, 4, 3, 5]);
    let output = x.clone();
    let (specs, attrs) = match kind {
        F32CnnOp::Conv2d => (
            vec![x, t(vec![4, 4, 1, 1]), t(vec![4])],
            BTreeMap::from([
                (
                    AttributeId::new("padding").unwrap(),
                    SemanticValue::Integers(vec![0; 4]),
                ),
                (
                    AttributeId::new("stride").unwrap(),
                    SemanticValue::Integers(vec![1; 2]),
                ),
                (
                    AttributeId::new("dilation").unwrap(),
                    SemanticValue::Integers(vec![1; 2]),
                ),
                (
                    AttributeId::new("groups").unwrap(),
                    SemanticValue::Unsigned(1),
                ),
                (
                    AttributeId::new("padding_mode").unwrap(),
                    SemanticValue::Text("reflect".into()),
                ),
            ]),
        ),
        F32CnnOp::GroupNormSilu => (
            vec![x, t(vec![4]), t(vec![4])],
            BTreeMap::from([
                (
                    AttributeId::new("groups").unwrap(),
                    SemanticValue::Unsigned(2),
                ),
                (
                    AttributeId::new("epsilon").unwrap(),
                    SemanticValue::Rational(CanonicalRational::new(1, 1_000_000).unwrap()),
                ),
            ]),
        ),
        F32CnnOp::ResidualAdd => (vec![x.clone(), x], BTreeMap::new()),
    };
    let geometry =
        F32CnnGeometry::new(kind, &specs.iter().collect::<Vec<_>>(), &output, &attrs).unwrap();
    let inputs = specs
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let len = t.minimum_storage_bytes().unwrap() as usize;
            buffer(
                runtime,
                &format!("resource.cnn.{}.input{i}", kind.operation_id()),
                &vec![0; len],
                16,
                ElementType::F32,
            )
            .region(0..u64::try_from(len).unwrap())
            .unwrap()
        })
        .collect();
    let len = output.minimum_storage_bytes().unwrap() as usize;
    let output = buffer(
        runtime,
        "resource.cnn.output",
        &vec![0xa5; len],
        16,
        ElementType::F32,
    )
    .region(0..u64::try_from(len).unwrap())
    .unwrap();
    ReferenceCnnF32Launch {
        inputs,
        output,
        geometry,
    }
}
#[test]
fn cnn_exact_f32_physical_ranges_and_noalias_reject_before_mutation() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in F32CnnOp::ALL {
        for case in 0..4 {
            let mut l = launch(&runtime, kind);
            let output = l.output.clone();
            match case {
                0 => l.inputs[0].element_type = ElementType::Bf16,
                1 => l.inputs[0].length_bytes -= 4,
                2 => l.output = l.inputs[0].clone(),
                3 => l.output.length_bytes -= 4,
                _ => unreachable!(),
            };
            assert!(ReferenceDeviceCommand::cnn_f32_tensor(l).is_err());
            assert!(output.read().iter().all(|b| *b == 0xa5));
        }
    }
    assert_eq!(runtime.snapshot().live_allocations, 0);
}
#[test]
fn foreign_last_cnn_parameter_preflight_retains_no_outputs_or_submitted_owners() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in F32CnnOp::ALL {
        let first = launch(&runtime, F32CnnOp::ResidualAdd);
        let output = first.output.clone();
        let mut last = launch(&runtime, kind);
        *last.inputs.last_mut().unwrap() = launch(&foreign, kind).inputs.pop().unwrap();
        let mut stream = runtime.create_stream().unwrap();
        assert!(runtime
            .submit_entries(
                &mut stream,
                vec![
                    (
                        DeviceCommandPhase::Compute,
                        Some(0),
                        ReferenceDeviceCommand::cnn_f32_tensor(first).unwrap()
                    ),
                    (
                        DeviceCommandPhase::Compute,
                        Some(1),
                        ReferenceDeviceCommand::cnn_f32_tensor(last).unwrap()
                    )
                ],
                false
            )
            .is_err());
        assert!(output.read().iter().all(|b| *b == 0xa5));
        assert_eq!(runtime.snapshot().submissions, 0);
        drop(output);
        drop(stream);
        assert_eq!(runtime.snapshot().live_allocations, 0);
        assert_eq!(foreign.snapshot().live_allocations, 0);
    }
}
