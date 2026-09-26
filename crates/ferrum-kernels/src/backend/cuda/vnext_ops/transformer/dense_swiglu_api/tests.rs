use super::*;
use ferrum_interfaces::execution_cost::{AlgorithmWorkKindV1, SelectedReplayAlgorithmTemplateV1};

#[test]
fn cublas_ffn_complete_sequence_keeps_api_calls_distinct_from_native_activation() {
    let identity = CublasHandleApiIdentity::fixture_identity();
    let build = |m| {
        Shape::new(m, 64, 96)
            .unwrap()
            .selected(SloStructuredCostCapture::HostSettledV1, Some(identity))
            .unwrap()
    };
    let a = build(3);
    let b = build(7);
    let kinds = a
        .algorithm_work()
        .unwrap()
        .unwrap()
        .entries()
        .iter()
        .map(|entry| entry.kind())
        .collect::<Vec<_>>();
    assert_eq!(
        kinds
            .iter()
            .filter(|k| **k == AlgorithmWorkKindV1::LibraryCall)
            .count(),
        2
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|k| **k == AlgorithmWorkKindV1::Kernel)
            .count(),
        1
    );

    a.validate_command(3, 3, 0).unwrap();
    b.validate_command(7, 3, 0).unwrap();
    assert_eq!(
        a.family_signature(),
        b.family_signature(),
        "M varies as numeric work, with fixed N/K API classes"
    );
    assert_ne!(a.algorithm_work().unwrap(), b.algorithm_work().unwrap());
    let template = SelectedReplayAlgorithmTemplateV1::from_selected(&a, 3, 3, 0).unwrap();
    template.validate_binding(&a).unwrap();
    assert!(
        template.validate_binding(&b).is_err(),
        "different M cannot borrow captured GemmEx parameters"
    );
    let shape = Shape::new(3, 64, 96).unwrap();
    assert!(shape
        .selected(SloStructuredCostCapture::Disabled, Some(identity))
        .is_none());
    assert!(shape
        .selected(SloStructuredCostCapture::HostSettledV1, None)
        .is_none());
    assert!(Shape::new(0, 64, 96).is_err());
    assert!(Shape::new(1, 64, i32::MAX as u64).is_err());
    assert!(Shape::new(u64::MAX, 64, 96).is_err());
}
