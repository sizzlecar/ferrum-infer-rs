use super::*;

fn bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|bits| bits.to_le_bytes()).collect()
}

fn input(host: &HostMemoryAccount, bits: &[u16; 6]) -> ManagedHostTensor {
    let mut allocation = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 12)
        .unwrap();
    allocation.as_mut().copy_from_slice(&bytes(bits));
    ManagedHostTensor::new(
        id(INPUT_ID),
        input_spec(vec![2, 3], ElementType::Bf16),
        allocation.freeze(),
    )
    .unwrap()
}

fn check_exact(input_bits: [u16; 6], weight_bits: [u16; 6], expected: [u16; 4]) {
    // Same exact twenty-byte Host input/output budget as the F16 fixture.
    let host = host_account(20, 8);
    let fixture = build_fixture_with_precision(
        host.clone(),
        TinyTensorPrecision::Bf16,
        &bytes(&weight_bits),
    );
    let mut executor = executor(&fixture);
    let output = executor
        .execute(
            id("run.tensor.bf16"),
            id("request.tensor.bf16"),
            vec![input(&host, &input_bits)],
        )
        .unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].tensor().dimensions(), [2, 2]);
    assert_eq!(output[0].tensor().element_type(), ElementType::Bf16);
    assert_eq!(output[0].bytes().bytes(), bytes(&expected));
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    let snapshot = fixture.composition.runtime().snapshot();
    assert_eq!(snapshot.dense_linear_launches, 1);
    assert_eq!(snapshot.uploaded_bytes, 24); // 12 weights + 12 inputs.
    assert_eq!(snapshot.readback_bytes, 8);
    drop(executor);
    close_fixture(fixture);
    assert_eq!(output[0].bytes().bytes(), bytes(&expected));
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn bf16_rounds_both_halfway_directions_to_even_and_preserves_sign() {
    // X = [[1,1,0],[-1,1,0]], W = [[1,1/256,0],[1+1/128,1/256,0]].
    // Row zero dots are 1+1/256 and 1+3/256: respectively round down to
    // even 1 and up to even 1+2/128. Row one gives -255/256 and -1-1/256.
    check_exact(
        [0x3f80, 0x3f80, 0, 0xbf80, 0x3f80, 0],
        [0x3f80, 0x3b80, 0, 0x3f81, 0x3b80, 0],
        [0x3f80, 0x3f82, 0xbf7f, 0xbf80],
    );
}

#[test]
fn bf16_uses_fp32_accumulation_before_the_output_rounding() {
    // Two half-ULP contributions after 1 must survive in FP32. Rounding each
    // partial sum to BF16 would lose both and incorrectly produce 1, not 129/128.
    check_exact(
        [0x3f80, 0x3f80, 0x3f80, 0xbf80, 0xbf80, 0xbf80],
        [0x3f80, 0x3b80, 0x3b80, 0x4000, 0xc000, 0x3e80],
        [0x3f81, 0x3e80, 0xbf81, 0xbe80],
    );
}

#[test]
fn bf16_cancellation_and_fractional_outputs_use_independent_expected_bits() {
    // X = [[1,-1,1],[-1,1,-1]], W = [[1,1,1/2],[-3/2,-3/2,-1/4]].
    check_exact(
        [0x3f80, 0xbf80, 0x3f80, 0xbf80, 0x3f80, 0xbf80],
        [0x3f80, 0x3f80, 0x3f00, 0xbfc0, 0xbfc0, 0xbe80],
        [0x3f00, 0xbe80, 0xbf00, 0x3e80],
    );
}

#[test]
fn bf16_plan_rejects_f16_metadata_before_admission_despite_equal_byte_width() {
    let host = host_account(20, 8);
    let fixture = build_fixture_with_precision(
        host.clone(),
        TinyTensorPrecision::Bf16,
        &bytes(&[0x3f80; 6]),
    );
    let binding = fixture
        .resources
        .bind_tensor_execution(
            fixture.compilation.executable().execution_plan(),
            host.clone(),
        )
        .unwrap();
    let allocation = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 12)
        .unwrap()
        .freeze();
    let before = fixture.composition.runtime().snapshot();
    for spec in [
        input_spec(vec![2, 3], ElementType::F16),
        input_spec(vec![3, 2], ElementType::Bf16),
    ] {
        let declared = ManagedHostTensor::new(id(INPUT_ID), spec, allocation.clone()).unwrap();
        assert!(binding.bind_inputs(vec![declared]).is_err());
    }
    let after = fixture.composition.runtime().snapshot();
    assert_eq!(after.submissions, before.submissions);
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.dense_linear_launches, 0);
    drop((binding, allocation));
    close_fixture(fixture);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
