//! Real admitted head dispatch, independently known representable inputs, and
//! actual CUDA replay. General quantization error is covered by the primitive
//! actual-pack/F64 fixture, not inferred from these exactly representable rows.
use ferrum_runtime::*;
use half::f16;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

// This target uses only part of the shared checkpoint fixture.
#[allow(dead_code)]
#[path = "vnext_checkpoint_continuation/family.rs"]
mod legacy_family;
use legacy_family::AttentionKind;

#[path = "vnext_cuda_q6_head/family.rs"]
mod family;
#[cfg(feature = "cuda-upstream-q6-f32-linear")]
use family::MAX_TOKENS;
use family::{Family, Head, Weights, OUTPUTS, PROFILE};

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}

#[test]
fn q6_head_fixture_declares_last_token_and_exact_input_oracle() {
    for head in [Head::Q6, Head::Dense, Head::Q6Split, Head::Q6Hadamard] {
        let definition = Family::new(head);
        let family = TypedFamilyRegistration::new(definition)
            .prepare_with_profile(&serde_json::to_value(head).unwrap(), &id(PROFILE))
            .unwrap();
        let nodes = &family.program().blocks()[0].nodes;
        let attention = nodes
            .iter()
            .find(|n| n.id.as_str() == "node.attention")
            .unwrap();
        assert_eq!(
            attention.operation_id.as_str(),
            CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID
        );
        assert_eq!(attention.outputs, [id::<ProgramValueId>("value.attention")]);
        assert!(!family.program().states().is_empty());
        let head_node = nodes.iter().find(|n| n.id.as_str() == "node.head").unwrap();
        assert_eq!(head_node.inputs[0].as_str(), "value.embedding");
        assert!(head_node.attributes.contains_key(&id("hidden_size")));
        assert_eq!(
            (OUTPUTS * 4) % 16,
            0,
            "physical head rows permit packed adjacency"
        );
        assert_eq!(
            family
                .weight_schema()
                .components
                .iter()
                .filter(|component| component.id.as_str().starts_with("component.head"))
                .count(),
            head.parts()
        );
        let source = Weights::new(family.weight_schema(), None, false);
        for component in &family.weight_schema().components {
            source.component(component).unwrap();
        }
        let a = source.expected(3);
        let b = source.expected(4);
        assert_eq!(a.len(), OUTPUTS as usize);
        assert!(a.iter().all(|v| v.is_finite()));
        assert_ne!(a, b, "last-token mistakes must be observable");
    }
}

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_weight_materializer_selection, CudaVNextComposition},
    vnext_runtime::CudaDeviceRuntime as Runtime,
};
#[cfg(feature = "cuda-upstream-q6-f32-linear")]
use std::sync::Arc;
#[cfg(feature = "cuda-upstream-q6-f32-linear")]
#[path = "vnext_cuda_q6_head/checks.rs"]
mod checks;
#[cfg(feature = "cuda-upstream-q6-f32-linear")]
#[path = "vnext_cuda_q6_head/runtime.rs"]
mod runtime;

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
#[test]
#[ignore = "requires exclusive CUDA and the actual production Q6 F32 native artifact"]
fn q6_head_actual_provider_last_token_batch_and_replay() {
    checks::verify();
}

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
#[test]
#[ignore = "requires exclusive CUDA and the actual production Q6 F32 native artifact"]
fn q6_head_actual_provider_markers_isolate_rows_and_poison_leaf() {
    checks::markers();
}

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
#[test]
#[ignore = "requires exclusive CUDA and the actual production Q6 F32 native artifact"]
fn q6_head_actual_provider_two_lanes_share_retained_validation() {
    checks::shared_plan_lanes();
}

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
#[path = "vnext_cuda_q6_head/facts.rs"]
mod facts;

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
#[test]
#[ignore = "requires exclusive CUDA and the actual production Q6 F32 native artifact"]
fn q6_head_execution_facts_actual_geometry_and_current_rows() {
    facts::geometry_and_rows();
}

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
#[test]
#[ignore = "requires exclusive CUDA and the actual production Q6 F32 native artifact"]
fn q6_head_execution_facts_original_capture_and_changed_replay_rows() {
    facts::replay();
}

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
#[test]
#[ignore = "requires exclusive CUDA and the actual production Q6 F32 native artifact"]
fn q6_head_execution_facts_off_capture_is_not_retroactively_described() {
    facts::off_capture();
}
