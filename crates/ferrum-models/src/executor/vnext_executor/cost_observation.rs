//! Cold-path identity and explicit passive actual dispatch observation.
//! Neither path changes timing/capture or physical execution policy.

pub(super) mod dispatch;
pub(super) mod domain;

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod product_tests;

#[cfg(test)]
#[path = "../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod identity_contract;

use ferrum_interfaces::{
    execution_cost::{
        CostIdentityUnknownReason, ExecutorCostIdentityAvailability,
        ExecutorCostIdentityComponents, EXECUTOR_COST_IDENTITY_SCHEMA,
    },
    vnext::{
        DeviceCostHardwareIdentityAvailability, ModelArtifactSourceRole, ResolvedModelPlan,
        ResolvedModelSources,
    },
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::Arc;

use super::{PreparedProductionModel, VNextExecutorConfig};

/// Length-delimited fields prevent ambiguous concatenations. All hashing and
/// serialization happens once during construction, never in a wave callback.
struct IdentityHasher(Sha256);

impl IdentityHasher {
    fn new(domain: &str) -> Self {
        let mut result = Self(Sha256::new());
        result.bytes("schema", &EXECUTOR_COST_IDENTITY_SCHEMA.to_le_bytes());
        result.bytes("domain", domain.as_bytes());
        result
    }

    fn bytes(&mut self, name: &str, value: &[u8]) {
        self.0.update((name.len() as u64).to_le_bytes());
        self.0.update(name.as_bytes());
        self.0.update((value.len() as u64).to_le_bytes());
        self.0.update(value);
    }

    fn serialized<T: Serialize + ?Sized>(
        &mut self,
        name: &str,
        value: &T,
    ) -> Result<(), CostIdentityUnknownReason> {
        let bytes = serde_json::to_vec(value)
            .map_err(|_| CostIdentityUnknownReason::MissingExecutionConfig)?;
        self.bytes(name, &bytes);
        Ok(())
    }

    fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

fn parse_sha256(value: &str) -> Result<[u8; 32], CostIdentityUnknownReason> {
    if value.len() != 64 {
        return Err(CostIdentityUnknownReason::InvalidSourceEvidence);
    }
    let mut digest = [0; 32];
    for (output, pair) in digest.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        let high = (pair[0] as char)
            .to_digit(16)
            .ok_or(CostIdentityUnknownReason::InvalidSourceEvidence)?;
        let low = (pair[1] as char)
            .to_digit(16)
            .ok_or(CostIdentityUnknownReason::InvalidSourceEvidence)?;
        *output = ((high << 4) | low) as u8;
    }
    Ok(digest)
}

fn model_content_identity(
    sources: &ResolvedModelSources,
) -> Result<[u8; 32], CostIdentityUnknownReason> {
    let mut hash = IdentityHasher::new("ferrum.executor.cost.model-content");
    for role in ModelArtifactSourceRole::ALL {
        let source = sources.for_role(role);
        if source.files.is_empty() {
            return Err(CostIdentityUnknownReason::MissingModelContent);
        }
        hash.bytes("role", role.as_str().as_bytes());
        let mut files: Vec<_> = source.files.iter().collect();
        files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        let mut previous = None;
        for file in files {
            let path = file.relative_path.as_str();
            if path.is_empty()
                || path.starts_with('/')
                || path.contains('\\')
                || path
                    .split('/')
                    .any(|part| part.is_empty() || part == "." || part == "..")
                || previous == Some(path)
            {
                return Err(CostIdentityUnknownReason::InvalidSourceEvidence);
            }
            hash.bytes("relative-path", path.as_bytes());
            hash.bytes("size", &file.size_bytes.to_le_bytes());
            hash.bytes("content-sha256", &parse_sha256(&file.sha256)?);
            previous = Some(path);
        }
    }
    // canonical_location / requested model / revision are provenance, not
    // content identity. Moving identical artifacts must preserve this digest.
    Ok(hash.finish())
}

fn numerical_identity(
    family: &str,
    program: &str,
    numerical: &str,
) -> Result<[u8; 32], CostIdentityUnknownReason> {
    let mut hash = IdentityHasher::new("ferrum.executor.cost.numerical-policy");
    for (name, value) in [
        ("family", family),
        ("program", program),
        ("numerical", numerical),
    ] {
        hash.bytes(name, &parse_sha256(value)?);
    }
    Ok(hash.finish())
}

fn execution_config_identity(
    plan: &ResolvedModelPlan,
    config: &VNextExecutorConfig,
) -> Result<[u8; 32], CostIdentityUnknownReason> {
    let mut hash = IdentityHasher::new("ferrum.executor.cost.execution-config");
    hash.bytes(
        "canonical-wave-schema",
        &ferrum_interfaces::execution_cost::CANONICAL_WAVE_COST_SCHEMA.to_le_bytes(),
    );
    let payload = plan.execution_plan().payload();
    // Explicit projection: not the full resolved plan (which carries source
    // paths and requested product settings), not a random runtime run ID.
    hash.serialized("schema", &payload.schema())?;
    hash.serialized("family", payload.family_id())?;
    hash.bytes(
        "runtime-implementation",
        payload
            .device_runtime_implementation_fingerprint()
            .as_bytes(),
    );
    hash.bytes(
        "catalog",
        payload.capability_catalog_fingerprint().as_bytes(),
    );
    hash.bytes("policy", config.runtime_policy.fingerprint_str().as_bytes());
    hash.serialized("scheduled-token-cap", &payload.maximum_scheduled_tokens())?;
    hash.serialized("weights-plan", payload.execution_weights())?;
    hash.serialized("weight-format", payload.weight_format())?;
    hash.serialized("quantization", payload.quantization_formats())?;
    hash.serialized("retention", payload.retained_completion_values())?;
    hash.serialized("outputs", payload.terminal_output_resources())?;
    // Nodes include selected provider implementations/contracts and bindings;
    // memory includes compiled capacity, workspace and reusable execution.
    hash.serialized("nodes", payload.nodes())?;
    hash.serialized("memory", payload.memory())?;
    hash.serialized("maximum-model-tokens", &config.maximum_model_tokens)?;
    hash.serialized(
        "reusable-enabled",
        &config.device_reusable_execution_enabled,
    )?;
    let chunks: Vec<_> = config
        .reusable_execution_prefill_chunks
        .iter()
        .map(|chunk| {
            (
                chunk.tokens_processed(),
                chunk.tokens_to_process(),
                chunk.total_prompt_tokens(),
            )
        })
        .collect();
    hash.serialized("reusable-chunks", &chunks)?;
    // Completion instrumentation is part of the measured execution population.
    // Keep existing unprofiled fingerprints stable; never borrow their models
    // for a profile which enqueues device events or changes capture prefixes.
    if config.device_timing_mode != super::DeviceTimingMode::Off {
        hash.serialized("device-timing-mode", &config.device_timing_mode)?;
        hash.serialized("device-timing-profile", &config.device_timing_profile)?;
    }
    // Preserve old default identities; new dynamic-sample policy is a distinct
    // cost boundary population and cannot borrow an old policy profile.
    if !config.structured_actual_capture.is_legacy() {
        hash.serialized(
            "structured-actual-capture",
            &config.structured_actual_capture,
        )?;
    }
    Ok(hash.finish())
}

pub(super) fn cache_identity(
    prepared: &PreparedProductionModel,
    plan: &ResolvedModelPlan,
    config: &VNextExecutorConfig,
    family_fingerprint: &str,
    program_fingerprint: &str,
    hardware: &DeviceCostHardwareIdentityAvailability,
) -> ExecutorCostIdentityAvailability {
    let model = model_content_identity(prepared.sources().resolved_sources());
    let numerical = prepared
        .family()
        .numerical_profile()
        .fingerprint()
        .map_err(|_| CostIdentityUnknownReason::MissingNumericalPolicy)
        .and_then(|numerical| {
            numerical_identity(family_fingerprint, program_fingerprint, &numerical)
        });
    let execution = execution_config_identity(plan, config);
    let reason = model
        .as_ref()
        .err()
        .or(numerical.as_ref().err())
        .or(execution.as_ref().err())
        .copied();
    let partial = ExecutorCostIdentityComponents {
        model_weights: model.ok(),
        numerical_policy: numerical.ok(),
        execution_config: execution.ok(),
        device_runtime: hardware_digest(hardware),
    };
    if let Some(reason) = reason {
        return ExecutorCostIdentityAvailability::Unknown {
            reason,
            partial: Some(Arc::new(partial)),
        };
    }
    // Only the runtime's explicit hardware evidence can fill this component.
    // Its digest includes scope and runtime version; missing evidence remains
    // Unknown rather than falling back to DeviceDescriptor capacity or name.
    partial.into_availability()
}

fn hardware_digest(hardware: &DeviceCostHardwareIdentityAvailability) -> Option<[u8; 32]> {
    match hardware {
        DeviceCostHardwareIdentityAvailability::Known(identity) => Some(*identity.fingerprint()),
        DeviceCostHardwareIdentityAvailability::Unknown(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{FileFingerprint, ResolvedModelSource};

    #[test]
    fn actual_capture_policy_binds_execution_identity_and_preserves_legacy_default() {
        use ferrum_types::{
            DataType, Device, EngineConfig, ModelInfo, ModelType, SloCostObservationConfig,
            SloStructuredActualCapturePolicy,
        };

        // Reuse the core compiler's resolved CPU model and its actual runtime.
        // This tests the product config-to-identity path, not GPU execution or
        // the acceptability of an old profile under a different sample regime.
        let fixture = identity_contract::fixture();
        let info = ModelInfo {
            model_id: "identity-contract".into(),
            model_type: ModelType::Custom("identity-contract".into()),
            num_parameters: 0,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 16,
            max_sequence_length: 64,
            dtype: DataType::FP16,
            device: Device::CPU,
            version: None,
            license: None,
            metadata: Default::default(),
        };
        let omitted =
            serde_json::to_value(SloCostObservationConfig::structured_whole_wave_v2()).unwrap();
        assert!(omitted.get("structured_actual_capture").is_none());
        let mut explicit_legacy = omitted.clone();
        explicit_legacy["structured_actual_capture"] =
            serde_json::to_value(SloStructuredActualCapturePolicy::LegacyEveryWave).unwrap();
        let mut consumer = omitted.clone();
        consumer["structured_actual_capture"] =
            serde_json::to_value(SloStructuredActualCapturePolicy::ConsumerDrivenV1).unwrap();
        let resolve = |wire| {
            let mut engine = EngineConfig::default();
            engine.backend.device = Device::CPU;
            engine.backend.enable_reusable_execution = false;
            engine.scheduler.max_running_requests = 1;
            engine.batching.max_num_batched_tokens = 8;
            engine.scheduler.slo.cost_observation = serde_json::from_value(wire).unwrap();
            engine.scheduler.slo.cost_observation.validate().unwrap();
            VNextExecutorConfig::from_engine_config(&engine, &info, fixture.runtime.as_ref())
                .unwrap()
        };
        let omitted = resolve(omitted);
        let legacy = resolve(explicit_legacy);
        let consumer = resolve(consumer);
        assert_eq!(
            omitted.runtime_policy.fingerprint_str(),
            consumer.runtime_policy.fingerprint_str(),
            "observation policy must not change physical scheduling or memory policy"
        );
        assert_eq!(omitted.maximum_model_tokens, consumer.maximum_model_tokens);
        assert_eq!(
            omitted.device_reusable_execution_enabled,
            consumer.device_reusable_execution_enabled
        );
        let original = execution_config_identity(&fixture.resolved, &omitted).unwrap();
        assert_eq!(
            execution_config_identity(&fixture.resolved, &legacy).unwrap(),
            original
        );
        assert_ne!(
            execution_config_identity(&fixture.resolved, &consumer).unwrap(),
            original,
            "identical plan and runtime must not alias different actual-sample policies"
        );
        // Close the fixture's real resources rather than relying on a detached
        // fake plan or leaving a device-account claim behind after the test.
        let mut close_checks = 0;
        identity_contract::close_plan_runtime(fixture.plan_resources, &mut close_checks);
    }

    #[test]
    fn completion_profile_identity_is_declared_before_sink_attachment() {
        use ferrum_types::{
            DataType, Device, EngineConfig, ModelInfo, ModelType, ObservabilityProfileDetail as D,
            ProfileEntrypoint,
        };
        let fixture = identity_contract::fixture();
        let info = ModelInfo {
            model_id: "profile-identity-contract".into(),
            model_type: ModelType::Custom("fixture".into()),
            num_parameters: 0,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 16,
            max_sequence_length: 64,
            dtype: DataType::FP16,
            device: Device::CPU,
            version: None,
            license: None,
            metadata: Default::default(),
        };
        let resolve = |entrypoint, detail, journal: bool| {
            let mut engine = EngineConfig::default();
            engine.backend.device = Device::CPU;
            engine.backend.enable_reusable_execution = false;
            engine.scheduler.max_running_requests = 1;
            engine.batching.max_num_batched_tokens = 8;
            engine.runtime.profile_entrypoint = Some(entrypoint);
            engine.runtime.profile_detail = detail;
            engine.runtime.profile_jsonl = journal.then(|| "logical-profile-path".into());
            VNextExecutorConfig::from_engine_config(&engine, &info, fixture.runtime.as_ref())
                .unwrap()
        };
        let off = resolve(ProfileEntrypoint::Run, D::Off, false);
        let off_id = execution_config_identity(&fixture.resolved, &off).unwrap();
        let mut basic_id = None;
        for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
            for journal in [false, true] {
                let config = resolve(entrypoint, D::Basic, journal);
                assert_eq!(
                    config.device_timing_mode,
                    super::super::DeviceTimingMode::Completion
                );
                assert_eq!(
                    config.runtime_policy.fingerprint_str(),
                    off.runtime_policy.fingerprint_str()
                );
                let id = execution_config_identity(&fixture.resolved, &config).unwrap();
                assert_ne!(
                    id, off_id,
                    "instrumented and uninstrumented profiles cannot share cost artifacts"
                );
                if let Some(expected) = basic_id {
                    assert_eq!(id, expected);
                } else {
                    basic_id = Some(id);
                }
            }
            let off_journal = resolve(entrypoint, D::Off, true);
            assert_eq!(
                execution_config_identity(&fixture.resolved, &off_journal).unwrap(),
                off_id
            );
            for detail in [D::Replay, D::Kernel, D::Verify, D::Full] {
                assert!(!resolve(entrypoint, detail, true)
                    .device_timing_mode
                    .guarded_completion_compatible());
            }
        }
        let mut close_checks = 0;
        identity_contract::close_plan_runtime(fixture.plan_resources, &mut close_checks);
    }

    fn sources() -> ResolvedModelSources {
        let source = ResolvedModelSource {
            canonical_location: "/a/model".into(),
            resolved_revision: "revision".into(),
            files: vec![
                FileFingerprint {
                    relative_path: "b.bin".into(),
                    size_bytes: 3,
                    sha256: "ab".repeat(32),
                },
                FileFingerprint {
                    relative_path: "a.bin".into(),
                    size_bytes: 8,
                    sha256: "cd".repeat(32),
                },
            ],
        };
        ResolvedModelSources {
            semantic: source.clone(),
            tokenizer: source.clone(),
            weights: source,
        }
    }

    #[test]
    fn moving_identical_sources_and_reordering_files_preserves_identity() {
        let original = sources();
        let mut moved = original.clone();
        moved.weights.canonical_location = "/different/cache/path".into();
        moved.weights.resolved_revision = "another-label-same-content".into();
        moved.weights.files.reverse();
        moved.tokenizer.files[0].sha256.make_ascii_uppercase();
        assert_eq!(
            model_content_identity(&original),
            model_content_identity(&moved)
        );
    }

    #[test]
    fn semantic_tokenizer_weight_content_and_layout_changes_invalidate_identity() {
        let original = sources();
        let baseline = model_content_identity(&original).unwrap();
        let mut changed = original.clone();
        changed.semantic.files[0].sha256 = "01".repeat(32);
        assert_ne!(model_content_identity(&changed).unwrap(), baseline);
        changed = original.clone();
        changed.tokenizer.files[0].sha256 = "02".repeat(32);
        assert_ne!(model_content_identity(&changed).unwrap(), baseline);
        changed = original.clone();
        changed.weights.files[0].size_bytes += 1;
        assert_ne!(model_content_identity(&changed).unwrap(), baseline);
        changed = original;
        changed.weights.files[0].relative_path = "renamed.bin".into();
        assert_ne!(model_content_identity(&changed).unwrap(), baseline);
    }

    #[test]
    fn malformed_missing_or_duplicate_source_evidence_is_unknown() {
        let mut malformed = sources();
        malformed.weights.files[0].sha256 = "unverified".into();
        assert_eq!(
            model_content_identity(&malformed),
            Err(CostIdentityUnknownReason::InvalidSourceEvidence)
        );
        malformed = sources();
        malformed
            .weights
            .files
            .push(malformed.weights.files[0].clone());
        assert_eq!(
            model_content_identity(&malformed),
            Err(CostIdentityUnknownReason::InvalidSourceEvidence)
        );
        malformed = sources();
        malformed.weights.files.clear();
        assert_eq!(
            model_content_identity(&malformed),
            Err(CostIdentityUnknownReason::MissingModelContent)
        );
        malformed = sources();
        malformed.weights.files[0].relative_path = "../external.bin".into();
        assert_eq!(
            model_content_identity(&malformed),
            Err(CostIdentityUnknownReason::InvalidSourceEvidence)
        );
    }

    #[test]
    fn actual_numeric_program_and_profile_are_separate_identity_inputs() {
        let one = "01".repeat(32);
        let two = "02".repeat(32);
        let base = numerical_identity(&one, &one, &one).unwrap();
        assert_ne!(base, numerical_identity(&two, &one, &one).unwrap());
        assert_ne!(base, numerical_identity(&one, &two, &one).unwrap());
        assert_ne!(base, numerical_identity(&one, &one, &two).unwrap());
    }

    #[test]
    fn hash_field_boundaries_and_domains_are_unambiguous() {
        let mut first = IdentityHasher::new("a");
        first.bytes("x", b"ab");
        first.bytes("y", b"c");
        let mut second = IdentityHasher::new("a");
        second.bytes("x", b"a");
        second.bytes("y", b"bc");
        assert_ne!(first.finish(), second.finish());
        assert_ne!(
            IdentityHasher::new("a").finish(),
            IdentityHasher::new("b").finish()
        );
    }

    #[test]
    fn runtime_hardware_evidence_is_required_to_complete_cached_identity() {
        use ferrum_interfaces::vnext::{DeviceCostHardwareIdentity, MetalHostBootDeviceEvidence};
        let partial = ExecutorCostIdentityComponents {
            model_weights: Some([1; 32]),
            numerical_policy: Some([2; 32]),
            execution_config: Some([3; 32]),
            device_runtime: hardware_digest(&Default::default()),
        };
        assert!(matches!(
            partial.clone().into_availability(),
            ExecutorCostIdentityAvailability::Unknown {
                reason: CostIdentityUnknownReason::MissingHardwareIdentity,
                ..
            }
        ));
        let known = DeviceCostHardwareIdentityAvailability::Known(Arc::new(
            DeviceCostHardwareIdentity::metal_host_boot(MetalHostBootDeviceEvidence {
                boot_session_uuid: [1; 16],
                registry_id: std::num::NonZeroU64::new(4).unwrap(),
                device_name: "Injected device",
                unified_memory: true,
                os_build: "Build1",
                host_model: "Host1",
                runtime_implementation_fingerprint:
                    "abababababababababababababababababababababababababababababababab",
            })
            .unwrap(),
        ));
        let complete = ExecutorCostIdentityComponents {
            device_runtime: hardware_digest(&known),
            ..partial
        };
        match complete.into_availability() {
            ExecutorCostIdentityAvailability::Known(identity) => {
                assert_eq!(identity.device_runtime, hardware_digest(&known).unwrap())
            }
            other => panic!("expected complete injected identity: {other:?}"),
        }
    }
}
