//! Provider declarations versus real CUDA commands and unchanged output/state.
use super::attention_cost_route::{compare, ExpectedRoute};
use super::*;
#[path = "head_cost_route/family.rs"]
mod head_family;

#[path = "q8_ffn_cost_family.rs"]
mod q8_ffn_family;

#[path = "cuda_ffn_replay.rs"]
mod ffn_replay;

#[test]
fn selected_cuda_q8_ffn_complete_future_route_matches_real_scalar_and_packed_outputs() {
    // These are the two existing opt-in policies, not a new numerical scope.
    for input_sum in [false, true] {
        let kind = AttentionKind::GatedDeltaHadamardF16;
        let definition = q8_ffn_family::Q8FfnFamily {
            base: Family::new(kind),
            policy: if input_sum {
                q8_ffn_family::FfnPolicy::Q8InputSum
            } else {
                q8_ffn_family::FfnPolicy::Q8
            },
        };
        let states = definition.base.states();
        let profile = definition.base.profile_id();
        let family = TypedFamilyRegistration::new(definition)
            .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
            .unwrap();
        let fixture =
            Fixture::from_prepared_family(kind, family, states, false, None, 1, BTreeMap::new());
        for lengths in [&[1][..], &[3][..], &[2, 3][..], &[4, 4][..]] {
            let baseline =
                super::full_cost_route::run_with_output(&fixture, lengths, false, "node.ffn").0;
            let actual =
                super::full_cost_route::run_with_output(&fixture, lengths, true, "node.ffn").0;
            assert_eq!(baseline, actual);
            let values = actual
                .iter()
                .flat_map(|row| row.chunks_exact(2))
                .map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
                .collect::<Vec<_>>();
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(
                values.iter().any(|value| *value != 0.0),
                "real FFN output must be nondegenerate"
            );
        }
    }
}

#[test]
fn selected_cuda_gdn_cost_matches_scalar_and_packed_real_waves() {
    for rows in [&[4][..], &[2, 3][..]] {
        compare(AttentionKind::GatedDelta, rows, ExpectedRoute::GatedDelta);
    }
}

#[test]
fn selected_cuda_gdn_transformed_and_q8_cost_matches_real_waves() {
    for kind in [
        AttentionKind::GatedDeltaHadamardF16,
        AttentionKind::GatedDeltaHadamardF32,
        AttentionKind::GatedDeltaQ8Projections,
    ] {
        compare(kind, &[2, 3], ExpectedRoute::GatedDelta);
    }
}

#[test]
fn selected_cuda_causal_cost_matches_scalar_packed_and_multi_page_waves() {
    for rows in [&[4][..], &[2, 3][..], &[129, 2][..]] {
        compare(
            AttentionKind::Causal,
            rows,
            ExpectedRoute::Causal { int8: false },
        );
    }
}

#[test]
fn selected_cuda_int8_causal_cost_stays_unknown_while_real_execution_succeeds() {
    compare(
        AttentionKind::CausalInt8,
        &[2, 3],
        ExpectedRoute::Causal { int8: true },
    );
}

#[test]
fn selected_cuda_head_range_proof_matches_real_scalar_packed_and_prefill_waves() {
    use head_family::{Head, HeadFamily};
    for workspace in [false, true] {
        let definition = HeadFamily {
            base: Family::new(AttentionKind::GatedDelta),
            head: Head::Strict,
        };
        let states = definition.base.states();
        let profile = definition.base.profile_id();
        let family = TypedFamilyRegistration::new(definition)
            .prepare_with_profile(
                &serde_json::to_value(AttentionKind::GatedDelta).unwrap(),
                &id(profile),
            )
            .unwrap();
        let fixture = Fixture::from_prepared_family_with_mode(
            AttentionKind::GatedDelta,
            family,
            states,
            if workspace {
                FixtureExecutionMode::WorkspaceOnly
            } else {
                FixtureExecutionMode::Eager
            },
            None,
            8,
            BTreeMap::new(),
        );
        let rows: &[&[usize]] = if workspace {
            &[&[1], &[4]]
        } else {
            &[&[1], &[1, 1], &[2, 3]]
        };
        for lengths in rows {
            let baseline = super::full_cost_route::run(&fixture, lengths, false, false).0;
            let actual = super::full_cost_route::run(&fixture, lengths, true, false).0;
            assert_eq!(baseline, actual);
        }
    }
}

#[test]
fn selected_cuda_native_head_capture_matches_real_future_and_preserves_output() {
    use ferrum_types::SloStructuredCostCapture;
    use head_family::{Head, HeadFamily};
    let build = |capture| {
        let definition = HeadFamily {
            base: Family::new(AttentionKind::GatedDelta),
            head: Head::Strict,
        };
        let states = definition.base.states();
        let profile = definition.base.profile_id();
        let family = TypedFamilyRegistration::new(definition)
            .prepare_with_profile(
                &serde_json::to_value(AttentionKind::GatedDelta).unwrap(),
                &id(profile),
            )
            .unwrap();
        let (runtime, registry, materializers, catalog) =
            ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
                0, id(format!("device.cuda.native-head-capture.{capture:?}")),
                ferrum_types::AttentionExecutionPolicy::Portable, None, capture,
            ).unwrap().into_parts();
        let materializer =
            ferrum_kernels::backend::cuda::vnext_ops::cuda_weight_materializer_selection(&family)
                .unwrap();
        Fixture::from_prepared_family_with_composition(
            AttentionKind::GatedDelta,
            family,
            states,
            FixtureExecutionMode::Eager,
            None,
            8,
            BTreeMap::new(),
            (runtime, registry, materializers, materializer, catalog),
        )
    };
    let disabled = build(SloStructuredCostCapture::Disabled);
    let enabled = build(SloStructuredCostCapture::HostSettledV1);
    // Non-unit ranges exercise actual last-token selection and participant
    // fallback without inventing physical proof for a plan-only packed query.
    for rows in [&[1][..], &[4][..], &[2, 3][..]] {
        let reference = super::full_cost_route::run_with_head_statistics(&disabled, rows, false);
        let observed = super::full_cost_route::run_with_head_statistics(&enabled, rows, true);
        assert_eq!(
            reference, observed,
            "passive capture changes real head output"
        );
        for bytes in observed {
            let values = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(values.len(), head_family::OUTPUTS as usize);
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(values.iter().any(|v| *v != 0.0));
        }
    }
}

#[test]
fn selected_cuda_native_ffn_capture_matches_current_future_work_and_preserves_real_output() {
    use ferrum_types::SloStructuredCostCapture;
    for policy in [
        q8_ffn_family::FfnPolicy::Strict,
        q8_ffn_family::FfnPolicy::Q8,
        q8_ffn_family::FfnPolicy::Q8InputSum,
        q8_ffn_family::FfnPolicy::StreamMmq,
    ] {
        let build = |capture| {
            let kind = AttentionKind::GatedDeltaHadamardF16;
            let definition = q8_ffn_family::Q8FfnFamily {
                base: Family::new(kind),
                policy,
            };
            let states = definition.base.states();
            let profile = definition.base.profile_id();
            let family = TypedFamilyRegistration::new(definition)
                .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
                .unwrap();
            let (runtime, registry, materializers, catalog) =
                ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
                    0, id(format!("device.cuda.ffn-selected.{policy:?}.{capture:?}")),
                    ferrum_types::AttentionExecutionPolicy::Portable, None, capture,
                ).unwrap().into_parts();
            let materializer =
                ferrum_kernels::backend::cuda::vnext_ops::cuda_weight_materializer_selection(
                    &family,
                )
                .unwrap();
            Fixture::from_prepared_family_with_composition(
                kind,
                family,
                states,
                FixtureExecutionMode::Eager,
                None,
                8,
                BTreeMap::new(),
                (runtime, registry, materializers, materializer, catalog),
            )
        };
        let disabled = build(SloStructuredCostCapture::Disabled);
        let enabled = build(SloStructuredCostCapture::HostSettledV1);
        for rows in [&[1][..], &[3][..], &[2, 3][..], &[4, 4][..]] {
            let baseline = super::full_cost_route::run_with_ffn_statistics(&disabled, rows, false);
            let observed = super::full_cost_route::run_with_ffn_statistics(&enabled, rows, true);
            assert_eq!(baseline, observed, "capture must preserve real FFN output");
            let values = observed
                .iter()
                .flat_map(|bytes| bytes.chunks_exact(2))
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect::<Vec<_>>();
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(values.iter().any(|v| *v != 0.0));
        }
    }
}

#[test]
fn selected_cuda_residual2_ffn_capture_matches_current_future_work_and_preserves_real_output() {
    use ferrum_types::SloStructuredCostCapture;
    for policy in [q8_ffn_family::FfnPolicy::Residual2M2To8] {
        let build = |capture| {
            let kind = AttentionKind::GatedDeltaHadamardF16;
            let definition = q8_ffn_family::Q8FfnFamily {
                base: Family::new(kind),
                policy,
            };
            let states = definition.base.states();
            let profile = definition.base.profile_id();
            let family = TypedFamilyRegistration::new(definition)
                .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
                .unwrap();
            let (runtime, registry, materializers, catalog) =
                ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
                    0, id(format!("device.cuda.ffn-selected.{policy:?}.{capture:?}")),
                    ferrum_types::AttentionExecutionPolicy::Portable, None, capture,
                ).unwrap().into_parts();
            let materializer =
                ferrum_kernels::backend::cuda::vnext_ops::cuda_weight_materializer_selection(
                    &family,
                )
                .unwrap();
            Fixture::from_prepared_family_with_composition(
                kind,
                family,
                states,
                FixtureExecutionMode::Eager,
                None,
                8,
                BTreeMap::new(),
                (runtime, registry, materializers, materializer, catalog),
            )
        };
        let disabled = build(SloStructuredCostCapture::Disabled);
        let enabled = build(SloStructuredCostCapture::HostSettledV1);
        for rows in [
            &[1][..],
            &[2][..],
            &[3][..],
            &[2, 3][..],
            &[3, 4][..],
            &[4, 4][..],
            &[9][..],
        ] {
            let baseline = super::full_cost_route::run_with_ffn_statistics(&disabled, rows, false);
            let observed = super::full_cost_route::run_with_ffn_statistics(&enabled, rows, true);
            assert_eq!(baseline, observed, "capture must preserve real FFN output");
            let values = observed
                .iter()
                .flat_map(|bytes| bytes.chunks_exact(2))
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect::<Vec<_>>();
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(values.iter().any(|v| *v != 0.0));
        }
    }
}

#[test]
fn selected_cuda_native_gdn_capture_matches_current_future_work_and_preserves_real_output() {
    use ferrum_types::SloStructuredCostCapture;
    for kind in [
        AttentionKind::GatedDelta,
        AttentionKind::GatedDeltaHadamardF16,
        AttentionKind::GatedDeltaHadamardF32,
        AttentionKind::GatedDeltaQ8Projections,
    ] {
        let build = |capture| {
            let definition = Family::new(kind);
            let states = definition.states();
            let profile = definition.profile_id();
            let family = TypedFamilyRegistration::new(definition)
                .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
                .unwrap();
            let (runtime, registry, materializers, catalog) =
                ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
                    0, id(format!("device.cuda.gdn-selected.{kind:?}.{capture:?}")),
                    ferrum_types::AttentionExecutionPolicy::Portable, None, capture,
                ).unwrap().into_parts();
            let materializer =
                ferrum_kernels::backend::cuda::vnext_ops::cuda_weight_materializer_selection(
                    &family,
                )
                .unwrap();
            Fixture::from_prepared_family_with_composition(
                kind,
                family,
                states,
                FixtureExecutionMode::Eager,
                None,
                8,
                BTreeMap::new(),
                (runtime, registry, materializers, materializer, catalog),
            )
        };
        let disabled = build(SloStructuredCostCapture::Disabled);
        let enabled = build(SloStructuredCostCapture::HostSettledV1);
        for rows in [&[1][..], &[3][..], &[2, 3][..], &[4, 4][..]] {
            let baseline = super::full_cost_route::run_with_gdn_statistics(&disabled, rows, false);
            let observed = super::full_cost_route::run_with_gdn_statistics(&enabled, rows, true);
            assert_eq!(baseline, observed, "capture must preserve real GDN output");
            let values: Vec<f32> = if kind.activation_type() == ElementType::F32 {
                observed
                    .iter()
                    .flat_map(|row| row.chunks_exact(4))
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            } else {
                observed
                    .iter()
                    .flat_map(|row| row.chunks_exact(2))
                    .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect()
            };
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(values.iter().any(|v| *v != 0.0));
        }
    }
}

#[test]
fn selected_cuda_native_gdn_current_evidence_survives_real_capture_and_replay() {
    use ferrum_types::SloStructuredCostCapture;
    for kind in [
        AttentionKind::GatedDelta,
        AttentionKind::GatedDeltaQ8Projections,
    ] {
        let build = |capture| {
            let definition = Family::new(kind);
            let states = definition.states();
            let profile = definition.profile_id();
            let family = TypedFamilyRegistration::new(definition)
                .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
                .unwrap();
            let (runtime, registry, materializers, catalog)=ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
                0,id(format!("device.cuda.gdn-replay.{kind:?}.{capture:?}")),
                ferrum_types::AttentionExecutionPolicy::Portable,None,capture).unwrap().into_parts();
            let materializer =
                ferrum_kernels::backend::cuda::vnext_ops::cuda_weight_materializer_selection(
                    &family,
                )
                .unwrap();
            Fixture::from_prepared_family_with_composition(
                kind,
                family,
                states,
                FixtureExecutionMode::Replay,
                None,
                1,
                BTreeMap::new(),
                (runtime, registry, materializers, materializer, catalog),
            )
        };
        let enabled = build(SloStructuredCostCapture::HostSettledV1);
        let disabled = build(SloStructuredCostCapture::Disabled);
        let tokens: Arc<[u32]> = Arc::from([3, 4, 5, 6]);
        let observed = enabled.admit("gdn-capture-on", Arc::clone(&tokens));
        let baseline = disabled.admit("gdn-capture-off", Arc::clone(&tokens));
        for turn in 0..4 {
            let actual = if turn < 2 {
                enabled.execute(&observed, Arc::clone(&tokens), turn..turn + 1)
            } else {
                enabled.execute_replayed_with_selected(
                    &observed,
                    Arc::clone(&tokens),
                    turn..turn + 1,
                )
            };
            let expected = if turn < 2 {
                disabled.execute(&baseline, Arc::clone(&tokens), turn..turn + 1)
            } else {
                disabled.execute_replayed(&baseline, Arc::clone(&tokens), turn..turn + 1)
            };
            actual.assert_state_nonzero();
            actual.assert_same(
                &expected,
                "current graph evidence must preserve different token/history state",
            );
        }
        observed.try_complete().unwrap();
        baseline.try_complete().unwrap();
    }
}

#[path = "causal_selected.rs"]
mod causal_selected;

#[path = "embedding_cost_route.rs"]
mod embedding_selected;
