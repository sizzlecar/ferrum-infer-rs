use super::*;
use ferrum_runtime::{
    HostMemoryClassLimits, HostMemoryLimits, HostMemoryPressureScope, ProgramValueId,
};

fn account(total: usize, input: usize, output: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: input,
                output_bytes: output,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn spec(shape: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        shape.to_vec(),
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn bytes(host: &HostMemoryAccount, class: HostMemoryClass, values: &[u16]) -> ManagedHostBytes {
    let mut data = host.allocate_zeroed(class, 2 * values.len()).unwrap();
    for (pair, value) in data.as_mut().chunks_exact_mut(2).zip(values) {
        pair.copy_from_slice(&value.to_le_bytes());
    }
    data.freeze()
}
fn sample(host: &HostMemoryAccount, shape: &[u64], values: &[u16]) -> ManagedHostTensor {
    ManagedHostTensor::new(
        ProgramValueId::new("value.latent").unwrap(),
        spec(shape),
        bytes(host, HostMemoryClass::InputPreprocess, values),
    )
    .unwrap()
}
fn solver(
    host: HostMemoryAccount,
    tensor: ResolvedTensorSpec,
    sigmas: &[f32],
    terminal: f32,
) -> ManagedBf16Euler {
    ManagedBf16Euler::new(
        FlowMatchEulerSchedule::new(sigmas, terminal, 1.0).unwrap(),
        host,
        tensor,
        Bf16EulerProfile::CudaScalarBf16StagedRneV1,
    )
    .unwrap()
}
fn bits(data: &ManagedHostBytes) -> Vec<u16> {
    data.bytes()
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

#[test]
fn scalar_and_product_rounding_are_independently_observable() {
    // dt=-257/512 is exactly halfway between -1/2 and -129/256.
    // dtB=-1/2 (even); v=129/128 => y=-129/256, bits bf01.
    // Without dtB the exact product is -33153/65536, just past the
    // midpoint -33152/65536, and final B would instead be bf02.
    let host = account(6, 4, 2);
    let mut euler = solver(host.clone(), spec(&[1]), &[1.0], 255.0 / 512.0);
    let x = sample(&host, &[1], &[0]);
    let v = bytes(&host, HostMemoryClass::Output, &[0x3f81]);
    let (step, result) = euler.advance(&x, &spec(&[1]), &v).unwrap();
    assert_eq!(step.delta_sigma(), -257.0 / 512.0);
    assert_eq!(bits(result.bytes()), [0xbf01]);
    assert_eq!(bits(x.bytes()), [0]);
    assert_eq!(bits(&v), [0x3f81]);
    assert!(euler.is_complete());
    assert!(matches!(
        euler.advance(&x, &spec(&[1]), &v),
        Err(Bf16EulerError::Complete)
    ));

    // Here dt=-129/256 is already BF16. Its product with 129/128 is
    // -16641/32768. ProductB=-130/256; x=+130/256 cancels to +0.
    // Omitting only productB leaves -1/32768, bits b800.
    let host = account(6, 4, 2);
    let mut euler = solver(host.clone(), spec(&[1]), &[1.0], 127.0 / 256.0);
    let x = sample(&host, &[1], &[0x3f02]);
    let v = bytes(&host, HostMemoryClass::Output, &[0x3f81]);
    let (_, result) = euler.advance(&x, &spec(&[1]), &v).unwrap();
    assert_eq!(bits(result.bytes()), [0]);
    assert_eq!(
        euler.profile().id(),
        "flow_match_euler.bf16.cuda_scalar_staged.rne.v1"
    );
}

#[test]
fn exact_budget_supports_successive_steps_and_retains_output_backing() {
    let host = account(12, 8, 4);
    let mut euler = solver(host.clone(), spec(&[1, 2]), &[1.0, 0.5], 0.0);
    let x = sample(&host, &[1, 2], &[0x4000, 0xc000]); // [2,-2]
    let v = bytes(&host, HostMemoryClass::Output, &[0x3f80, 0xbf80]); // [1,-1]
    let (first, next) = euler.advance(&x, &spec(&[1, 2]), &v).unwrap();
    assert_eq!(first.index(), 0);
    assert_eq!(bits(next.bytes()), [0x3fc0, 0xbfc0]);
    assert_eq!(next.value_id(), x.value_id());
    assert!(host.owns(next.bytes()));
    assert_eq!(next.bytes().class(), HostMemoryClass::InputPreprocess);
    assert_eq!(host.usage().unwrap().committed_bytes(), 12);
    drop(x);
    let (second, result) = euler.advance(&next, &spec(&[1, 2]), &v).unwrap();
    assert_eq!(second.index(), 1);
    assert_eq!(bits(result.bytes()), [0x3f80, 0xbf80]);
    assert!(euler.is_complete());
    let retained = result.bytes().slice(2..4).unwrap();
    let alias = retained.clone();
    drop((result, next, v, euler));
    assert_eq!(host.usage().unwrap().committed_bytes(), 4);
    assert_eq!(bits(&retained), [0xbf80]);
    drop(retained);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 4);
    drop(alias);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn parent_and_input_class_capacity_fail_without_progress_or_mutation() {
    for (host, expected_scope) in [
        (account(5, 4, 2), HostMemoryPressureScope::Total),
        (
            account(6, 3, 2),
            HostMemoryPressureScope::Class(HostMemoryClass::InputPreprocess),
        ),
    ] {
        let mut euler = solver(host.clone(), spec(&[1]), &[1.0], 0.0);
        let x = sample(&host, &[1], &[0x4000]);
        let v = bytes(&host, HostMemoryClass::Output, &[0x3f80]);
        assert!(matches!(euler.advance(&x, &spec(&[1]), &v),
            Err(Bf16EulerError::Host(HostMemoryError::Capacity { scope, .. })) if scope == expected_scope));
        assert_eq!(euler.step_index(), 0);
        assert_eq!(bits(x.bytes()), [0x4000]);
        assert_eq!(bits(&v), [0x3f80]);
        assert_eq!(host.usage().unwrap().committed_bytes(), 4);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 4);
    }
}

#[test]
fn actual_account_shape_dtype_class_and_byte_extent_are_checked() {
    let host = account(48, 32, 16);
    let foreign = account(48, 32, 16); // equal limits are not equal authority
    let mut euler = solver(host.clone(), spec(&[1, 2]), &[1.0], 0.0);
    let x = sample(&host, &[1, 2], &[0, 0]);
    let v = bytes(&host, HostMemoryClass::Output, &[0, 0]);
    let wrong_shape = sample(&host, &[2, 1], &[0, 0]);
    assert!(matches!(
        euler.advance(&wrong_shape, &spec(&[1, 2]), &v),
        Err(Bf16EulerError::TensorMismatch {
            operand: Bf16EulerOperand::Sample
        })
    ));
    assert!(matches!(
        euler.advance(&x, &spec(&[2, 1]), &v),
        Err(Bf16EulerError::TensorMismatch {
            operand: Bf16EulerOperand::Velocity
        })
    ));
    let f16 = ResolvedTensorSpec::new(
        vec![1, 2],
        ElementType::F16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(matches!(
        euler.advance(&x, &f16, &v),
        Err(Bf16EulerError::TensorMismatch { .. })
    ));
    let foreign_sample = sample(&foreign, &[1, 2], &[0, 0]);
    let foreign_velocity = bytes(&foreign, HostMemoryClass::Output, &[0, 0]);
    assert!(matches!(
        euler.advance(&foreign_sample, &spec(&[1, 2]), &v),
        Err(Bf16EulerError::ForeignOwner {
            operand: Bf16EulerOperand::Sample
        })
    ));
    assert!(matches!(
        euler.advance(&x, &spec(&[1, 2]), &foreign_velocity),
        Err(Bf16EulerError::ForeignOwner {
            operand: Bf16EulerOperand::Velocity
        })
    ));
    assert!(matches!(
        euler.advance(&x, &spec(&[1, 2]), x.bytes()),
        Err(Bf16EulerError::WrongClass {
            operand: Bf16EulerOperand::Velocity,
            actual: HostMemoryClass::InputPreprocess
        })
    ));
    let short = v.slice(0..2).unwrap();
    assert!(matches!(
        euler.advance(&x, &spec(&[1, 2]), &short),
        Err(Bf16EulerError::ByteExtentMismatch { .. })
    ));
    assert_eq!(euler.step_index(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 12);
}

#[test]
fn nonfinite_inputs_and_final_cast_overflow_release_private_partial_output() {
    for bad in [0x7f80, 0xff80, 0x7fc1] {
        for operand in [Bf16EulerOperand::Sample, Bf16EulerOperand::Velocity] {
            let host = account(12, 8, 4);
            let mut euler = solver(host.clone(), spec(&[2]), &[1.0], 0.0);
            let x_bits = if operand == Bf16EulerOperand::Sample {
                [0x3f80, bad]
            } else {
                [0x3f80, 0x3f80]
            };
            let v_bits = if operand == Bf16EulerOperand::Velocity {
                [0x3f80, bad]
            } else {
                [0x3f80, 0x3f80]
            };
            let x = sample(&host, &[2], &x_bits);
            let v = bytes(&host, HostMemoryClass::Output, &v_bits);
            assert!(matches!(euler.advance(&x, &spec(&[2]), &v),
                Err(Bf16EulerError::NonFiniteInput { operand: actual, index: 1 }) if actual == operand));
            assert_eq!(bits(x.bytes()), x_bits);
            assert_eq!(bits(&v), v_bits);
            assert_eq!(euler.step_index(), 0);
            assert_eq!(host.usage().unwrap().committed_bytes(), 8);
            assert_eq!(host.usage().unwrap().allocated_bytes(), 8);
        }
    }
    // maxBF16 + 2^119 is finite FP32 but exactly at BF16's overflow
    // midpoint, so checking only the FP32 sum would incorrectly accept it.
    let host = account(6, 4, 2);
    let mut euler = solver(host.clone(), spec(&[1]), &[1.0], 0.5);
    let x = sample(&host, &[1], &[0x7f7f]);
    let v = bytes(&host, HostMemoryClass::Output, &[0xfb80]); // -2^120
    assert!(matches!(
        euler.advance(&x, &spec(&[1]), &v),
        Err(Bf16EulerError::NonFiniteResult { index: 0 })
    ));
    assert_eq!(euler.step_index(), 0);
    assert_eq!(bits(x.bytes()), [0x7f7f]);
    assert_eq!(host.usage().unwrap().committed_bytes(), 4);
}

#[test]
fn signed_zero_subnormal_midpoints_and_tiny_scalar_follow_declared_rne() {
    let host = account(24, 16, 8);
    let mut euler = solver(host.clone(), spec(&[4]), &[1.0], 0.5);
    let x = sample(&host, &[4], &[0x8000, 0, 0x8000, 0x8000]);
    let v = bytes(&host, HostMemoryClass::Output, &[0, 0, 0x0001, 0x8001]);
    let (_, result) = euler.advance(&x, &spec(&[4]), &v).unwrap();
    assert_eq!(bits(result.bytes()), [0x8000, 0, 0x8000, 0]);

    // The smallest FP32 interval rounds to negative BF16 zero. This
    // profile preserves that sign and still advances the declared interval.
    let host = account(6, 4, 2);
    let mut euler = solver(host.clone(), spec(&[1]), &[f32::from_bits(1)], 0.0);
    let x = sample(&host, &[1], &[0x8000]);
    let v = bytes(&host, HostMemoryClass::Output, &[0x3f80]);
    let (_, result) = euler.advance(&x, &spec(&[1]), &v).unwrap();
    assert_eq!(bits(result.bytes()), [0x8000]);
    assert!(euler.is_complete());
}

#[test]
fn unsupported_tensor_type_is_rejected_without_reserving_host_payload() {
    let host = account(6, 4, 2);
    let schedule = FlowMatchEulerSchedule::new(&[1.0], 0.0, 1.0).unwrap();
    let tensor =
        ResolvedTensorSpec::new(vec![1], ElementType::F32, ResolvedTensorLayout::Contiguous)
            .unwrap();
    assert!(matches!(
        ManagedBf16Euler::new(
            schedule,
            host.clone(),
            tensor,
            Bf16EulerProfile::CudaScalarBf16StagedRneV1
        ),
        Err(Bf16EulerError::InvalidTensorSpec)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
