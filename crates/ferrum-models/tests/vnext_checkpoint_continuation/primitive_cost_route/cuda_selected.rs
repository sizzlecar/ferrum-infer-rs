//! Existing primitive program, real resources and original output oracle.
use super::*;
use ferrum_types::SloStructuredCostCapture;

fn build(
    hadamard: bool,
    vocabulary: u64,
    capture: SloStructuredCostCapture,
    replay: bool,
) -> (Fixture, WeightSchema) {
    let kind = if hadamard {
        AttentionKind::GatedDeltaHadamardF16
    } else {
        AttentionKind::GatedDelta
    };
    let definition = PrimitiveFamily {
        base: Family::new(kind),
        vocabulary,
        linear: LinearWeight::Q4K,
        graph_attention: replay,
    };
    let inputs = definition.additional_inputs();
    let states = if replay {
        definition.base.states()
    } else {
        vec![]
    };
    let profile = definition.base.profile_id();
    let family = TypedFamilyRegistration::new(definition)
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
        .unwrap();
    let schema = family.weight_schema().clone();
    let (runtime, registry, materializers, catalog) =
        ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
            0,
            id(format!(
                "device.cuda.primitive-selected.{hadamard}.{vocabulary}.{capture:?}.{replay}"
            )),
            ferrum_types::AttentionExecutionPolicy::Portable,
            None,
            capture,
        )
        .unwrap()
        .into_parts();
    let materializer =
        ferrum_kernels::backend::cuda::vnext_ops::cuda_weight_materializer_selection(&family)
            .unwrap();
    (
        Fixture::from_prepared_family_with_composition(
            kind,
            family,
            states,
            if replay {
                FixtureExecutionMode::Replay
            } else {
                FixtureExecutionMode::Eager
            },
            None,
            3,
            inputs,
            (runtime, registry, materializers, materializer, catalog),
        ),
        schema,
    )
}

#[test]
fn selected_cuda_embedding_argmax_current_future_and_outputs_match_capture_off() {
    for (hadamard, vocabulary) in [(false, 17), (true, 8192)] {
        let (enabled, schema) = build(
            hadamard,
            vocabulary,
            SloStructuredCostCapture::HostSettledV1,
            false,
        );
        let (disabled, control) = build(
            hadamard,
            vocabulary,
            SloStructuredCostCapture::Disabled,
            false,
        );
        let actual = execute(
            &enabled,
            &schema,
            hadamard,
            vocabulary,
            LinearWeight::Q4K,
            &[2, 3],
            PrimitiveExecutionStage::Eager,
            Some(true),
            0,
        );
        let expected = execute(
            &disabled,
            &control,
            hadamard,
            vocabulary,
            LinearWeight::Q4K,
            &[2, 3],
            PrimitiveExecutionStage::Eager,
            Some(false),
            0,
        );
        assert_eq!(
            actual, expected,
            "capture must preserve complete embedding/argmax/residual/linear outputs"
        );
    }
}

#[test]
fn selected_cuda_embedding_argmax_current_work_binds_real_graph_and_updated_inputs() {
    for hadamard in [false, true] {
        let vocabulary = 8192;
        let (enabled, schema) = build(
            hadamard,
            vocabulary,
            SloStructuredCostCapture::HostSettledV1,
            true,
        );
        let (disabled, control) = build(
            hadamard,
            vocabulary,
            SloStructuredCostCapture::Disabled,
            true,
        );
        for turn in 0..4 {
            let stage = match turn {
                0 => PrimitiveExecutionStage::GraphWarmup,
                1 => PrimitiveExecutionStage::GraphCapture,
                _ => PrimitiveExecutionStage::DirectReplay,
            };
            let actual = execute(
                &enabled,
                &schema,
                hadamard,
                vocabulary,
                LinearWeight::Q4K,
                &[3],
                stage,
                Some(true),
                turn,
            );
            let expected = execute(
                &disabled,
                &control,
                hadamard,
                vocabulary,
                LinearWeight::Q4K,
                &[3],
                stage,
                Some(false),
                turn,
            );
            assert_eq!(
                actual, expected,
                "captured graph must consume new token IDs and logits, without changing output"
            );
        }
    }
}
