use super::*;

#[test]
fn causal_admission_retry_binds_the_original_immutable_request_owner() {
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1024,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1024,
                output_bytes: 1024,
                transfer_staging_bytes: 1024,
                weight_backing_bytes: 1024,
            },
        )
        .unwrap(),
        1,
    )
    .unwrap();
    let value = ProgramValueId::new("input.conditioning").unwrap();
    let schema = CausalInputSchema::new(vec![CausalTokenInputSpec {
        value_id: value.clone(),
        element_type: ElementType::F16,
        row_dimensions: vec![1],
        storage: CausalTokenInputStorage::ZeroOutsideSegments,
    }])
    .unwrap();
    let mut source = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 2)
        .unwrap();
    source
        .as_mut()
        .copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
    let source = source.freeze();
    let input = || {
        Arc::new(
            CausalRequestInputs::new(
                &host,
                &schema,
                3,
                vec![CausalTokenInput::new(
                    value.clone(),
                    vec![CausalInputSegment::new(1, 1, source.clone()).unwrap()],
                )],
            )
            .unwrap(),
        )
    };
    let original = input();
    let foreign = input(); // Same physical source bytes, different admission owner.
    let request = RequestId::new();
    let tokens = [TokenId::new(1), TokenId::new(2)];
    let admission =
        ExecutorPrefillAdmission::for_product_request(&request, &tokens, 3, 2, 0).unwrap();
    let binding = AdmissionBinding::from_input(admission, Some(Arc::clone(&original)));
    assert!(binding.matches(admission, Some(&original)));
    assert!(binding.matches(admission, Some(&Arc::clone(&original))));
    assert!(!binding.matches(admission, Some(&foreign)));
    assert!(!binding.matches(admission, None));
    drop((binding, original, foreign));
    drop(source);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
