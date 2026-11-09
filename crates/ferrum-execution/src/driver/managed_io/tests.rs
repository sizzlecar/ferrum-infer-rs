use super::*;

fn host(total: usize, alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: total,
                transfer_staging_bytes: total,
                output_bytes: total,
                weight_backing_bytes: total,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}

#[test]
fn managed_greedy_profile_rejects_processors_and_full_logits_explicitly() {
    let io = ManagedCausalIo {
        host: host(1024, 1),
        profile: ManagedCausalIoProfile::ManagedGreedyF32V1,
    };
    let greedy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: None,
    };
    io.validate_policy(&greedy).unwrap();
    assert!(io.validate_policy(&LogitsReturnPolicy::FullLogits).is_err());
    assert!(io
        .validate_policy(&LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(crate::model_executor::TokenSelectionMask::new(vec![
                1, 1, 1
            ])),
            repetition_penalty: None,
        })
        .is_err());
    assert!(
        io.validate_policy(&LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: Some(crate::model_executor::GreedyRepetitionPenalty::new(
                1.0,
                vec![1]
            )),
        })
        .is_err(),
        "even a neutral processor must be an explicit future profile"
    );
    assert_eq!(io.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn managed_greedy_scans_exact_f32_values_finite_domain_and_first_ties() {
    let cases: &[(&[f32], ExecutorSamplingOutput)] = &[
        (
            &[0.5, 0.50000006, -1.0],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(1)),
        ),
        (
            &[2.0, 2.0, 1.0],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(0)),
        ),
        (
            &[f32::NAN, f32::INFINITY, -3.0, f32::NEG_INFINITY],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(2)),
        ),
        (
            &[f32::NAN, f32::NEG_INFINITY],
            ExecutorSamplingOutput::NoSelectableFiniteLogit,
        ),
        (
            &[-0.0, 0.0, f32::from_bits(1)],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(2)),
        ),
        (
            &[-0.0, 0.0],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(1)),
        ),
        (
            &[0.0, -0.0],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(0)),
        ),
    ];
    for (values, expected) in cases {
        let host = host(values.len() * 4, 1);
        let mut owner = host
            .allocate_zeroed(HostMemoryClass::Output, values.len() * 4)
            .unwrap();
        for (destination, value) in owner.as_mut().chunks_exact_mut(4).zip(values.iter()) {
            destination.copy_from_slice(&value.to_le_bytes());
        }
        let owner = owner.freeze();
        assert_eq!(
            &select_finite_logits(owner.bytes(), values.len(), ElementType::F32).unwrap(),
            expected
        );
        assert_eq!(host.usage().unwrap().allocated_bytes(), values.len() * 4);
        drop(owner);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    assert!(select_finite_logits(&[0; 3], 1, ElementType::F32).is_err());
    assert!(select_finite_logits(&[], 1, ElementType::F32).is_err());
}

#[test]
fn managed_io_query_counts_each_physical_allocation_and_full_prefill_row() {
    let host = host(2 << 20, 16);
    let rows = [
        ManagedCausalIoWork {
            immediate_tokens: 2,
            mask_upload: true,
            output: ManagedCausalReadback::F32Logits,
        },
        ManagedCausalIoWork {
            immediate_tokens: 1,
            mask_upload: true,
            output: ManagedCausalReadback::GreedyToken,
        },
    ];
    let report = requirements(
        &host,
        248320,
        ManagedCausalIoProfile::ManagedGreedyF32V1,
        &rows,
    )
    .unwrap();
    let physical = |n| host.backing_bytes(n).unwrap();
    assert_eq!(
        report.input_bytes,
        physical(8) + physical(4) + 2 * (physical(248320) + physical(8) + physical(4))
    );
    assert_eq!(report.output_bytes, physical(993280) + physical(4));
    assert_eq!(report.total_bytes, report.input_bytes + report.output_bytes);
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        0,
        "query grants no allocation authority"
    );
    assert!(requirements(&host, 3, ManagedCausalIoProfile::ManagedGreedyF32V1, &[]).is_err());
    assert!(requirements(
        &host,
        3,
        ManagedCausalIoProfile::ManagedGreedyF32V1,
        &[ManagedCausalIoWork {
            immediate_tokens: usize::MAX,
            mask_upload: true,
            output: ManagedCausalReadback::GreedyToken,
        }]
    )
    .is_err());
}

#[test]
fn full_prefill_owner_remains_accounted_until_its_real_consumer_releases_it() {
    let host = host(993280, 1);
    let mut logits = host
        .allocate_zeroed(HostMemoryClass::Output, 993280)
        .unwrap();
    logits.as_mut()[4 * 248319..].copy_from_slice(&1.0000001f32.to_le_bytes());
    let logits = logits.freeze();
    let consumer = logits.clone();
    drop(logits);
    assert!(host.allocate_zeroed(HostMemoryClass::Output, 1).is_err());
    assert_eq!(
        select_finite_logits(consumer.bytes(), 248320, ElementType::F32).unwrap(),
        ExecutorSamplingOutput::GreedyToken(TokenId::new(248319))
    );
    assert_eq!(host.usage().unwrap().allocated_bytes(), 993280);
    drop(consumer);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn f16_logits_use_the_actual_half_bytes_and_original_finite_total_order() {
    let cases: &[(&[u16], ExecutorSamplingOutput)] = &[
        (
            &[0x8000, 0x0000],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(1)),
        ),
        (
            &[0x0000, 0x8000],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(0)),
        ),
        (
            &[0x3c00, 0x3c01, 0x3c01],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(1)),
        ),
        (
            &[0x8001, 0x0000, 0x0001],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(2)),
        ),
        (
            &[0x7e01, 0x7c00, 0xfc00, 0xc200],
            ExecutorSamplingOutput::GreedyToken(TokenId::new(3)),
        ),
        (
            &[0x7e01, 0x7c00, 0xfc00],
            ExecutorSamplingOutput::NoSelectableFiniteLogit,
        ),
    ];
    for (bits, expected) in cases {
        let host = host(bits.len() * 2, 1);
        let mut bytes = host
            .allocate_zeroed(HostMemoryClass::Output, bits.len() * 2)
            .unwrap();
        for (slot, value) in bytes.as_mut().chunks_exact_mut(2).zip(bits.iter()) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        let owner = bytes.freeze();
        assert_eq!(
            &select_finite_logits(owner.bytes(), bits.len(), ElementType::F16).unwrap(),
            expected
        );
        assert_eq!(host.usage().unwrap().allocated_bytes(), bits.len() * 2);
        drop(owner);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    assert!(select_finite_logits(&[0; 3], 1, ElementType::F16).is_err());
    assert!(select_finite_logits(&[0; 2], 1, ElementType::Bf16).is_err());
}

#[test]
fn f16_profile_budget_is_half_head_storage_and_rejects_cross_profile_estimates() {
    let host = host(2 << 20, 16);
    let work = [ManagedCausalIoWork {
        immediate_tokens: 1,
        mask_upload: true,
        output: ManagedCausalReadback::F16Logits,
    }];
    let profile = ManagedCausalIoProfile::ManagedGreedyF16V1;
    let report = requirements(&host, 248320, profile, &work).unwrap();
    assert_eq!(report.output_bytes, host.backing_bytes(496640).unwrap());
    assert!(requirements(
        &host,
        248320,
        ManagedCausalIoProfile::ManagedGreedyF32V1,
        &work
    )
    .is_err());
    assert!(requirements(
        &host,
        248320,
        profile,
        &[ManagedCausalIoWork {
            immediate_tokens: 1,
            mask_upload: true,
            output: ManagedCausalReadback::F32Logits,
        }]
    )
    .is_err());
    profile.validate_logits_type(ElementType::F16).unwrap();
    assert!(profile.validate_logits_type(ElementType::F32).is_err());
    assert!(ManagedCausalIoProfile::ManagedGreedyF32V1
        .validate_logits_type(ElementType::F16)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn mixed_managed_output_demand_uses_the_original_finality_rule() {
    use crate::token_execution::ExecutedTokenWork;
    let decode = ExecutedTokenWork::Decode { position: 2 };
    let intermediate = ExecutedTokenWork::Prefill {
        chunk: PrefillChunk::new(0, 1, 2).unwrap(),
        capacity_probe_count: 0,
    };
    let final_prefill = ExecutedTokenWork::Prefill {
        chunk: PrefillChunk::new(1, 1, 2).unwrap(),
        capacity_probe_count: 0,
    };
    assert_eq!(
        managed_product_mode(VNextExecutionWaveKind::Mixed, &[intermediate, decode]),
        VNextProductOutputMode::GreedyToken
    );
    assert_eq!(
        managed_product_mode(VNextExecutionWaveKind::Mixed, &[final_prefill, decode]),
        VNextProductOutputMode::FullLogits
    );
    assert_eq!(
        managed_product_mode(VNextExecutionWaveKind::Prefill, &[intermediate]),
        VNextProductOutputMode::FullLogits
    );
}
