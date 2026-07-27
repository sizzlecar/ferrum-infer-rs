//! CUDA vNext determinism evidence collection.
//!
//! The command is intentionally a product binary entrypoint rather than a
//! test-only harness. It resolves the same immutable model package and creates
//! the same concrete executor as `run` and `serve`, while keeping evidence
//! assembly separate from the external bounded-runner receipt.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use clap::Args;
#[cfg(feature = "cuda")]
use ferrum_models::{
    VNextDeterminismExecutionMode, VNextDeterminismExecutionSpec, VNextDeterminismPhase,
};
#[cfg(any(feature = "cuda", test))]
use ferrum_models::{
    VNextDeterminismInitialState, VNextDeterminismParticipantSpec, VNextDeterminismWorkspacePoison,
    MAX_VNEXT_DETERMINISM_PARTICIPANTS,
};
use ferrum_types::{FerrumError, Result};

const PRIMARY_MODEL_KEYS: [&str; 3] = ["m1-qwen35-4b", "m2-qwen35-35b-a3b", "m3-qwen3-30b-a3b"];
#[cfg(feature = "cuda")]
const EXECUTIONS_PER_MODE: usize = 6;
#[cfg(any(feature = "cuda", test))]
const EXPECTED_CASES: usize = 72;

#[derive(Args, Clone, Debug)]
pub struct VNextDeterminismCommand {
    /// Immutable three-model lock generated from the checked-in release catalog.
    #[arg(long, value_name = "PATH")]
    pub models_lock: PathBuf,

    /// Existing artifact root containing `hardware-probe/probe.json`.
    #[arg(long, value_name = "DIR")]
    pub artifact_root: PathBuf,

    /// Primary model binding in `MODEL_KEY=/absolute/model/directory` form.
    #[arg(
        long = "model",
        value_name = "MODEL_KEY=DIR",
        action = clap::ArgAction::Append
    )]
    pub models: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelBinding {
    key: String,
    directory: PathBuf,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShapePhase {
    Prefill,
    Decode,
}

#[cfg(any(feature = "cuda", test))]
impl ShapePhase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }

    #[cfg(feature = "cuda")]
    const fn execution_phase(self) -> VNextDeterminismPhase {
        match self {
            Self::Prefill => VNextDeterminismPhase::Prefill,
            Self::Decode => VNextDeterminismPhase::Decode,
        }
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct ParticipantFixture {
    token_ids: Vec<u32>,
    immediate_start: usize,
}

#[cfg(any(feature = "cuda", test))]
impl ParticipantFixture {
    fn immediate_end(&self) -> usize {
        self.token_ids.len()
    }

    fn to_spec(&self) -> Result<VNextDeterminismParticipantSpec> {
        VNextDeterminismParticipantSpec::new(
            self.token_ids.clone(),
            self.immediate_start..self.immediate_end(),
            self.immediate_end().saturating_add(8),
        )
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct ShapeFixture {
    phase: ShapePhase,
    partition: &'static str,
    participants: Vec<ParticipantFixture>,
}

#[cfg(any(feature = "cuda", test))]
impl ShapeFixture {
    #[cfg(feature = "cuda")]
    fn execution_spec(
        &self,
        initial_state: VNextDeterminismInitialState,
        workspace_poison: VNextDeterminismWorkspacePoison,
        mode: VNextDeterminismExecutionMode,
    ) -> Result<VNextDeterminismExecutionSpec> {
        VNextDeterminismExecutionSpec::new(
            self.phase.execution_phase(),
            self.participants
                .iter()
                .map(ParticipantFixture::to_spec)
                .collect::<Result<Vec<_>>>()?,
            initial_state,
            workspace_poison,
            mode,
        )
    }
}

pub async fn execute(command: VNextDeterminismCommand) -> Result<()> {
    let bindings = validate_command(&command)?;
    #[cfg(feature = "cuda")]
    {
        return cuda::collect(command, bindings).await;
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = bindings;
        Err(FerrumError::unsupported(
            "ferrum vnext-determinism requires a binary built with the cuda feature",
        ))
    }
}

fn validate_command(command: &VNextDeterminismCommand) -> Result<Vec<ModelBinding>> {
    require_regular_file(&command.models_lock, "--models-lock")?;
    if !command.artifact_root.is_dir() {
        return Err(FerrumError::invalid_parameter(format!(
            "--artifact-root is not a directory: {}",
            command.artifact_root.display()
        )));
    }
    require_regular_file(
        &command.artifact_root.join("hardware-probe/probe.json"),
        "hardware-probe/probe.json",
    )?;
    parse_model_bindings(&command.models, true)
}

fn require_regular_file(path: &Path, label: &str) -> Result<()> {
    let metadata = path.symlink_metadata().map_err(|error| {
        FerrumError::invalid_parameter(format!(
            "{label} is not a readable regular file at {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(FerrumError::invalid_parameter(format!(
            "{label} must be a real regular file: {}",
            path.display()
        )));
    }
    Ok(())
}

fn parse_model_bindings(values: &[String], require_directories: bool) -> Result<Vec<ModelBinding>> {
    let expected = PRIMARY_MODEL_KEYS.into_iter().collect::<BTreeSet<_>>();
    let mut indexed = BTreeMap::new();
    for value in values {
        let (key, raw_directory) = value.split_once('=').ok_or_else(|| {
            FerrumError::invalid_parameter(format!(
                "--model must use MODEL_KEY=DIR form, got {value:?}"
            ))
        })?;
        if key.is_empty() || raw_directory.is_empty() || !expected.contains(key) {
            return Err(FerrumError::invalid_parameter(format!(
                "--model has an unknown key or empty directory: {value:?}"
            )));
        }
        let directory = PathBuf::from(raw_directory);
        if require_directories && (!directory.is_absolute() || !directory.is_dir()) {
            return Err(FerrumError::invalid_parameter(format!(
                "--model {key} must name an existing absolute directory: {}",
                directory.display()
            )));
        }
        if indexed.insert(key.to_owned(), directory).is_some() {
            return Err(FerrumError::invalid_parameter(format!(
                "--model duplicates primary model key {key}"
            )));
        }
    }
    let actual = indexed.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual != expected {
        let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
        return Err(FerrumError::invalid_parameter(format!(
            "--model must bind exactly the three primary CUDA models; missing {missing:?}"
        )));
    }
    Ok(PRIMARY_MODEL_KEYS
        .iter()
        .map(|key| ModelBinding {
            key: (*key).to_owned(),
            directory: indexed
                .remove(*key)
                .expect("exact primary model set was checked"),
        })
        .collect())
}

#[cfg(any(feature = "cuda", test))]
fn release_shape_fixtures() -> Vec<ShapeFixture> {
    let prefill = |partition, token_count, immediate_start| ShapeFixture {
        phase: ShapePhase::Prefill,
        partition,
        participants: vec![ParticipantFixture {
            token_ids: deterministic_tokens(0, token_count),
            immediate_start,
        }],
    };
    let decode = |partition, participant_count| ShapeFixture {
        phase: ShapePhase::Decode,
        partition,
        participants: (0..participant_count)
            .map(|participant| ParticipantFixture {
                token_ids: deterministic_tokens(participant, 9),
                immediate_start: 8,
            })
            .collect(),
    };
    vec![
        prefill("single_token", 1, 0),
        prefill("multi_token", 4, 0),
        prefill("chunk_boundary", 8, 4),
        decode("c1", 1),
        decode("multi_participant", 4),
        decode("c32", MAX_VNEXT_DETERMINISM_PARTICIPANTS),
    ]
}

#[cfg(any(feature = "cuda", test))]
fn deterministic_tokens(participant: usize, count: usize) -> Vec<u32> {
    let base = 100_u32.saturating_add(
        u32::try_from(participant)
            .unwrap_or(u32::MAX)
            .saturating_mul(16),
    );
    (0..count)
        .map(|offset| base.saturating_add(u32::try_from(offset).unwrap_or(u32::MAX)))
        .collect()
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::sync::Arc;

    use ferrum_engine::vnext_determinism::{
        create_cuda_vnext_determinism_collector, CudaVNextDeterminismCollector,
    };
    use ferrum_interfaces::vnext::{
        CapabilityCatalog, ContractVersion, ExecutionDeterminismEvidenceDenominator,
        ProviderReplayEquivalence, ResolvedModelPlan, SubmissionWaveDeterminismArtifactExecution,
        SubmissionWaveDeterminismArtifactWitness, VNextError,
    };
    use ferrum_models::vnext::{
        open_registered_colocated_safetensors, resolve_registered_model_from_sources,
        PreparedProductionModel,
    };
    use ferrum_types::{Device, EngineConfig, ModelId};
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use uuid::Uuid;

    use super::*;

    const ARTIFACT_TYPE: &str = "runtime_vnext_cuda_determinism_collector";

    #[derive(Debug, Serialize)]
    struct TokenShapeArtifact {
        partition: String,
        participant_count: usize,
        immediate_tokens: Vec<usize>,
        source_start_tokens: Vec<usize>,
        source_end_tokens: Vec<usize>,
    }

    impl TokenShapeArtifact {
        fn from_fixture(fixture: &ShapeFixture) -> Self {
            Self {
                partition: fixture.partition.to_owned(),
                participant_count: fixture.participants.len(),
                immediate_tokens: fixture
                    .participants
                    .iter()
                    .map(|participant| {
                        participant
                            .immediate_end()
                            .saturating_sub(participant.immediate_start)
                    })
                    .collect(),
                source_start_tokens: fixture
                    .participants
                    .iter()
                    .map(|participant| participant.immediate_start)
                    .collect(),
                source_end_tokens: fixture
                    .participants
                    .iter()
                    .map(ParticipantFixture::immediate_end)
                    .collect(),
            }
        }
    }

    #[derive(Debug, Serialize)]
    struct InitializationArtifact {
        input_sha256: String,
        rng_sha256: String,
        initial_state_kind: String,
        initial_state_sha256: String,
        workspace_poison: String,
    }

    #[derive(Clone, Debug, Serialize)]
    struct CoverageTargetArtifact {
        operation_id: String,
        operation_version: ContractVersion,
        operation_fingerprint: String,
        provider_id: String,
        provider_version: ContractVersion,
        provider_implementation_fingerprint: String,
        provider_execution_contract_fingerprint: String,
        replay_equivalence: String,
        witness_plan_fingerprint: String,
        node_ids: Vec<String>,
    }

    #[derive(Debug, Serialize)]
    struct ComparisonArtifact {
        kind: String,
        ordinal: usize,
        left_execution_id: String,
        right_execution_id: String,
        relation: &'static str,
        first_mismatch: Option<String>,
    }

    #[derive(Debug, Serialize)]
    struct CaseArtifact {
        schema_version: u32,
        case_id: String,
        denominator_fingerprint: String,
        binary_sha256: String,
        device_runtime_implementation_fingerprint: String,
        device_fingerprint: String,
        model_key: String,
        resolved_plan_fingerprint: String,
        plan_hash: String,
        phase: String,
        token_shape: TokenShapeArtifact,
        dtype: String,
        quantization: String,
        initialization: InitializationArtifact,
        coverage_targets: Vec<CoverageTargetArtifact>,
        executions: Vec<SubmissionWaveDeterminismArtifactExecution>,
        comparisons: Vec<ComparisonArtifact>,
        first_mismatch: Option<String>,
    }

    struct PendingCase {
        case_id: String,
        model_key: String,
        phase: String,
        token_shape: TokenShapeArtifact,
        dtype: String,
        quantization: String,
        initial_state_kind: String,
        workspace_poison: String,
        executions: Vec<SubmissionWaveDeterminismArtifactExecution>,
        comparisons: Vec<ComparisonArtifact>,
    }

    #[derive(Debug, Serialize)]
    struct CaseProgressArtifact<'a> {
        schema_version: u32,
        artifact_type: &'static str,
        status: &'static str,
        case_id: &'a str,
        model_key: &'a str,
        phase: &'a str,
        token_shape: &'a TokenShapeArtifact,
        dtype: &'a str,
        quantization: &'a str,
        initial_state_kind: &'a str,
        workspace_poison: &'a str,
        initialization_identity:
            &'a ferrum_interfaces::vnext::SubmissionWaveDeterminismArtifactInitializationIdentity,
        execution_count: usize,
        comparison_count: usize,
        canonical_witness_count: usize,
        canonical_witnesses_sha256: String,
        replayed_segment_count: usize,
    }

    struct CollectedModel {
        key: String,
        directory: PathBuf,
        plan: ResolvedModelPlan,
        dtype: String,
        quantization: String,
        cases: Vec<PendingCase>,
    }

    #[derive(Debug, Deserialize)]
    struct HardwareProbeIdentity {
        schema_version: u32,
        fingerprint: String,
    }

    #[derive(Debug, Serialize)]
    struct FileReference {
        path: String,
        sha256: String,
        size_bytes: u64,
    }

    #[derive(Debug, Serialize)]
    struct DenominatorReference {
        path: String,
        sha256: String,
        size_bytes: u64,
        fingerprint: String,
    }

    #[derive(Debug, Serialize)]
    struct CollectorModelSummary {
        model_key: String,
        model_dir: String,
        resolved_plan_fingerprint: String,
        plan_hash: String,
        dtype: String,
        quantization: String,
        case_count: usize,
    }

    #[derive(Debug, Serialize)]
    struct CollectorManifest {
        schema_version: u32,
        artifact_type: &'static str,
        status: &'static str,
        backend: &'static str,
        models_lock: FileReference,
        hardware_probe: FileReference,
        device_fingerprint: String,
        binary: FileReference,
        denominator: DenominatorReference,
        models: Vec<CollectorModelSummary>,
        cases: Vec<FileReference>,
        case_count: usize,
        execution_count: usize,
        comparison_count: usize,
        pass_line: String,
    }

    #[derive(Debug, Serialize)]
    struct RejectionArtifact {
        schema_version: u32,
        artifact_type: &'static str,
        status: &'static str,
        failure_class: &'static str,
        message: String,
    }

    pub(super) async fn collect(
        command: VNextDeterminismCommand,
        bindings: Vec<ModelBinding>,
    ) -> Result<()> {
        match collect_inner(&command, &bindings).await {
            Ok(pass_line) => {
                println!("{pass_line}");
                Ok(())
            }
            Err(error) => {
                let rejection = RejectionArtifact {
                    schema_version: 1,
                    artifact_type: ARTIFACT_TYPE,
                    status: "reject",
                    failure_class: "collector_failure",
                    message: error.to_string(),
                };
                let rejection_path = command.artifact_root.join("collector.reject.json");
                let _ = write_json_exclusive(&rejection_path, &rejection);
                Err(error)
            }
        }
    }

    async fn collect_inner(
        command: &VNextDeterminismCommand,
        bindings: &[ModelBinding],
    ) -> Result<String> {
        reject_existing_outputs(&command.artifact_root)?;
        let models_lock = file_reference(
            &command.artifact_root,
            &command.models_lock,
            "models.lock.json",
        )?;
        let probe_path = command.artifact_root.join("hardware-probe/probe.json");
        let hardware_probe = read_hardware_probe(&probe_path)?;
        let hardware_probe_ref = file_reference(
            &command.artifact_root,
            &probe_path,
            "hardware-probe/probe.json",
        )?;
        let current_exe = std::env::current_exe().map_err(|error| {
            FerrumError::io(format!("cannot resolve current ferrum binary: {error}"))
        })?;
        let binary = absolute_file_reference(&current_exe)?;
        let progress_root = command.artifact_root.join("collector-progress");
        let progress_cases = progress_root.join("cases");
        fs::create_dir(&progress_root).map_err(|error| {
            FerrumError::io(format!(
                "cannot create determinism progress directory {}: {error}",
                progress_root.display()
            ))
        })?;
        fs::create_dir(&progress_cases).map_err(|error| {
            FerrumError::io(format!(
                "cannot create determinism progress case directory {}: {error}",
                progress_cases.display()
            ))
        })?;

        let mut canonical_catalog: Option<CapabilityCatalog> = None;
        let mut canonical_catalog_fingerprint: Option<String> = None;
        let mut collected_models = Vec::with_capacity(bindings.len());
        for (model_ordinal, binding) in bindings.iter().enumerate() {
            println!(
                "FERRUM VNEXT DETERMINISM PROGRESS model={} stage=load ordinal={}/{}",
                binding.key,
                model_ordinal + 1,
                bindings.len()
            );
            let sources = Arc::new(
                open_registered_colocated_safetensors(&binding.directory).map_err(|error| {
                    FerrumError::model(format!(
                        "cannot open registered source for {} at {}: {error}",
                        binding.key,
                        binding.directory.display()
                    ))
                })?,
            );
            let registration = resolve_registered_model_from_sources(sources.as_ref())
                .and_then(|registration| registration.into_required())
                .map_err(|error| {
                    FerrumError::model(format!(
                        "cannot require vNext registration for {}: {error}",
                        binding.key
                    ))
                })?;
            let prepared = registration
                .prepare_from_sources(sources)
                .map_err(|error| {
                    FerrumError::model(format!(
                        "cannot prepare registered vNext model {}: {error}",
                        binding.key
                    ))
                })?;
            let capabilities = prepared.model_capabilities()?;
            let dtype = prepared.descriptor().execution_dtype().to_string();
            let quantization = capabilities
                .quantization
                .unwrap_or_else(|| "none".to_owned());
            let mut engine = determinism_engine_config(binding, &prepared);
            engine.backend.dtype = prepared.descriptor().execution_dtype();
            let collector = create_cuda_vnext_determinism_collector(&engine, &prepared, 0)
                .map_err(|error| {
                    FerrumError::backend(format!(
                        "cannot create CUDA determinism collector for {}: {error}",
                        binding.key
                    ))
                })?;
            let catalog_fingerprint = collector
                .capability_catalog()
                .fingerprint()
                .map_err(vnext_backend_error)?;
            match canonical_catalog_fingerprint.as_deref() {
                Some(expected) if expected != catalog_fingerprint => {
                    return Err(FerrumError::backend(format!(
                        "CUDA capability catalog drifted between primary models: expected {expected}, got {catalog_fingerprint} for {}",
                        binding.key
                    )));
                }
                None => {
                    canonical_catalog = Some(collector.capability_catalog().clone());
                    canonical_catalog_fingerprint = Some(catalog_fingerprint);
                }
                Some(_) => {}
            }
            let plan = collector.resolved_model_plan().clone();
            collector.prepare().await.map_err(|error| {
                FerrumError::backend(format!(
                    "cannot prepare CUDA determinism collector for {}: {error}",
                    binding.key
                ))
            })?;
            println!(
                "FERRUM VNEXT DETERMINISM PROGRESS model={} stage=prepared",
                binding.key
            );
            let cases = collect_model_cases(
                &collector,
                &binding.key,
                &dtype,
                &quantization,
                model_ordinal * 24,
                &progress_cases,
            )
            .await?;
            drop(collector);
            drop(prepared);
            println!(
                "FERRUM VNEXT DETERMINISM PROGRESS model={} stage=released cases={}",
                binding.key,
                cases.len()
            );
            collected_models.push(CollectedModel {
                key: binding.key.clone(),
                directory: binding.directory.clone(),
                plan,
                dtype,
                quantization,
                cases,
            });
        }

        let catalog = canonical_catalog.ok_or_else(|| {
            FerrumError::internal("CUDA determinism collector produced no capability catalog")
        })?;
        let plan_refs = collected_models
            .iter()
            .map(|model| (model.key.as_str(), &model.plan))
            .collect::<Vec<_>>();
        let denominator = ExecutionDeterminismEvidenceDenominator::from_catalog_and_resolved_plans(
            &catalog, &plan_refs,
        )
        .map_err(vnext_backend_error)?;
        let denominator_bytes = denominator.to_json().map_err(vnext_backend_error)?;
        let denominator_fingerprint = denominator.fingerprint().map_err(vnext_backend_error)?;
        let device_runtime_fingerprint = denominator
            .coverage()
            .device_runtime_implementation_fingerprint()
            .to_owned();

        let stage = command.artifact_root.join(format!(
            ".vnext-determinism-stage-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        fs::create_dir(&stage).map_err(|error| {
            FerrumError::io(format!(
                "cannot create determinism staging directory {}: {error}",
                stage.display()
            ))
        })?;
        let stage_cases = stage.join("cases");
        fs::create_dir(&stage_cases).map_err(|error| {
            FerrumError::io(format!(
                "cannot create determinism case staging directory {}: {error}",
                stage_cases.display()
            ))
        })?;

        let staged = stage_collection(
            &stage,
            collected_models,
            &denominator,
            &denominator_bytes,
            &denominator_fingerprint,
            &device_runtime_fingerprint,
            &hardware_probe.fingerprint,
            models_lock,
            hardware_probe_ref,
            binary,
        );
        let manifest = match staged {
            Ok(manifest) => manifest,
            Err(error) => {
                let _ = fs::remove_dir_all(&stage);
                return Err(error);
            }
        };
        publish_stage(&stage, &command.artifact_root)?;
        let pass_line = manifest.pass_line.clone();
        Ok(pass_line)
    }

    fn determinism_engine_config(
        binding: &ModelBinding,
        prepared: &PreparedProductionModel,
    ) -> EngineConfig {
        let mut engine = EngineConfig::default();
        engine.model.model_id = ModelId::new(binding.key.clone());
        engine.backend.device = Device::CUDA(0);
        engine.backend.dtype = prepared.descriptor().execution_dtype();
        engine.backend.enable_reusable_execution = true;
        engine.scheduler.max_running_requests = MAX_VNEXT_DETERMINISM_PARTICIPANTS;
        engine.batching.max_batch_size = MAX_VNEXT_DETERMINISM_PARTICIPANTS;
        engine.batching.max_num_batched_tokens = engine
            .batching
            .max_num_batched_tokens
            .max(MAX_VNEXT_DETERMINISM_PARTICIPANTS);
        engine.runtime.model_path = Some(binding.directory.display().to_string());
        engine
    }

    async fn collect_model_cases(
        collector: &CudaVNextDeterminismCollector,
        model_key: &str,
        dtype: &str,
        quantization: &str,
        completed_before_model: usize,
        progress_cases: &Path,
    ) -> Result<Vec<PendingCase>> {
        let mut cases = Vec::with_capacity(24);
        for fixture in release_shape_fixtures() {
            for (initial_state, initial_state_kind) in [
                (VNextDeterminismInitialState::Zero, "zero"),
                (VNextDeterminismInitialState::Nonzero, "nonzero"),
            ] {
                let zero = collect_case(
                    collector,
                    model_key,
                    &fixture,
                    dtype,
                    quantization,
                    initial_state,
                    initial_state_kind,
                    VNextDeterminismWorkspacePoison::Zero,
                    "00",
                )
                .await?;
                write_case_progress(progress_cases, &zero)?;
                print_case_progress(completed_before_model + cases.len() + 1, &zero);
                let a5 = collect_case(
                    collector,
                    model_key,
                    &fixture,
                    dtype,
                    quantization,
                    initial_state,
                    initial_state_kind,
                    VNextDeterminismWorkspacePoison::A5,
                    "a5",
                )
                .await?;
                write_case_progress(progress_cases, &a5)?;
                print_case_progress(completed_before_model + cases.len() + 2, &a5);
                ensure_poison_equivalence(&zero, &a5)?;
                cases.extend([zero, a5]);
            }
        }
        Ok(cases)
    }

    fn write_case_progress(progress_cases: &Path, case: &PendingCase) -> Result<()> {
        let first = case.executions.first().ok_or_else(|| {
            FerrumError::internal(format!(
                "determinism case {} contains no execution",
                case.case_id
            ))
        })?;
        let witness_bytes = serde_json::to_vec(first.witnesses())
            .map_err(|error| FerrumError::serialization(error.to_string()))?;
        let progress = CaseProgressArtifact {
            schema_version: 1,
            artifact_type: "runtime_vnext_cuda_determinism_case_progress",
            status: "case_comparisons_pass",
            case_id: &case.case_id,
            model_key: &case.model_key,
            phase: &case.phase,
            token_shape: &case.token_shape,
            dtype: &case.dtype,
            quantization: &case.quantization,
            initial_state_kind: &case.initial_state_kind,
            workspace_poison: &case.workspace_poison,
            initialization_identity: first.initialization_identity(),
            execution_count: case.executions.len(),
            comparison_count: case.comparisons.len(),
            canonical_witness_count: first.witnesses().len(),
            canonical_witnesses_sha256: format!("{:x}", Sha256::digest(&witness_bytes)),
            replayed_segment_count: case
                .executions
                .iter()
                .find(|execution| execution.mode() == "replay")
                .map(|execution| execution.replayed_segments().len())
                .unwrap_or(0),
        };
        write_json_exclusive(
            &progress_cases.join(format!("{}.json", case.case_id)),
            &progress,
        )
    }

    fn print_case_progress(completed: usize, case: &PendingCase) {
        println!(
            "FERRUM VNEXT DETERMINISM PROGRESS case={} complete={}/{}",
            case.case_id, completed, EXPECTED_CASES
        );
    }

    async fn collect_case(
        collector: &CudaVNextDeterminismCollector,
        model_key: &str,
        fixture: &ShapeFixture,
        dtype: &str,
        quantization: &str,
        initial_state: VNextDeterminismInitialState,
        initial_state_kind: &str,
        workspace_poison: VNextDeterminismWorkspacePoison,
        workspace_poison_label: &str,
    ) -> Result<PendingCase> {
        let case_id = format!(
            "{model_key}.{}.{}.{}.{}",
            fixture.phase.as_str(),
            fixture.partition,
            initial_state_kind,
            workspace_poison_label
        );
        let mut executions = Vec::with_capacity(EXECUTIONS_PER_MODE * 2);
        for (mode, mode_label) in [
            (VNextDeterminismExecutionMode::Eager, "eager"),
            (VNextDeterminismExecutionMode::Replayed, "replay"),
        ] {
            for repeat in 0..EXECUTIONS_PER_MODE {
                let spec = fixture.execution_spec(initial_state, workspace_poison, mode)?;
                let execution_id = format!("{mode_label}-{repeat:02}");
                let evidence = collector.collect_execution(&spec).await.map_err(|error| {
                    FerrumError::backend(format!(
                        "determinism execution failed for case {case_id} execution {execution_id}: {error}"
                    ))
                })?;
                executions.push(
                    evidence
                        .into_artifact_execution(execution_id.clone())
                        .map_err(|error| {
                            FerrumError::backend(format!(
                                "determinism artifact projection failed for case {case_id} execution {execution_id}: {error}"
                            ))
                        })?,
                );
            }
        }
        executions.sort_by(|left, right| left.execution_id().cmp(right.execution_id()));
        let comparisons = compare_case_executions(&executions)?;
        Ok(PendingCase {
            case_id,
            model_key: model_key.to_owned(),
            phase: fixture.phase.as_str().to_owned(),
            token_shape: TokenShapeArtifact::from_fixture(fixture),
            dtype: dtype.to_owned(),
            quantization: quantization.to_owned(),
            initial_state_kind: initial_state_kind.to_owned(),
            workspace_poison: workspace_poison_label.to_owned(),
            executions,
            comparisons,
        })
    }

    fn compare_case_executions(
        executions: &[SubmissionWaveDeterminismArtifactExecution],
    ) -> Result<Vec<ComparisonArtifact>> {
        let by_id = executions
            .iter()
            .map(|execution| (execution.execution_id(), execution))
            .collect::<BTreeMap<_, _>>();
        let mut comparisons = Vec::with_capacity(15);
        for (kind, left_mode, right_mode) in [
            ("eager_eager", "eager", "eager"),
            ("eager_replay", "eager", "replay"),
            ("replay_replay", "replay", "replay"),
        ] {
            for ordinal in 0..5 {
                let left_id = format!("{left_mode}-{ordinal:02}");
                let right_ordinal = if kind == "eager_replay" {
                    ordinal
                } else {
                    ordinal + 1
                };
                let right_id = format!("{right_mode}-{right_ordinal:02}");
                let left = by_id.get(left_id.as_str()).ok_or_else(|| {
                    FerrumError::internal(format!(
                        "determinism comparison lacks execution {left_id}"
                    ))
                })?;
                let right = by_id.get(right_id.as_str()).ok_or_else(|| {
                    FerrumError::internal(format!(
                        "determinism comparison lacks execution {right_id}"
                    ))
                })?;
                ensure_execution_equivalence(left, right, kind)?;
                if kind == "replay_replay" && left.replayed_segments() != right.replayed_segments()
                {
                    return Err(FerrumError::backend(format!(
                        "{kind} replay shape mismatch between {left_id} and {right_id}"
                    )));
                }
                comparisons.push(ComparisonArtifact {
                    kind: kind.to_owned(),
                    ordinal,
                    left_execution_id: left_id,
                    right_execution_id: right_id,
                    relation: "bitwise_equal",
                    first_mismatch: None,
                });
            }
        }
        Ok(comparisons)
    }

    fn ensure_execution_equivalence(
        left: &SubmissionWaveDeterminismArtifactExecution,
        right: &SubmissionWaveDeterminismArtifactExecution,
        comparison_kind: &str,
    ) -> Result<()> {
        if left.restore_sha256() != right.restore_sha256()
            || left.initialization_identity() != right.initialization_identity()
        {
            return Err(FerrumError::backend(format!(
                "{comparison_kind} restored different input/RNG/initial-state bytes between {} and {}",
                left.execution_id(),
                right.execution_id()
            )));
        }
        if left.witnesses().len() != right.witnesses().len() {
            return Err(FerrumError::backend(format!(
                "{comparison_kind} witness cardinality differs between {} ({}) and {} ({})",
                left.execution_id(),
                left.witnesses().len(),
                right.execution_id(),
                right.witnesses().len()
            )));
        }
        for (left_witness, right_witness) in left.witnesses().iter().zip(right.witnesses()) {
            if left_witness != right_witness {
                return Err(witness_mismatch(
                    comparison_kind,
                    left.execution_id(),
                    right.execution_id(),
                    left_witness,
                    right_witness,
                ));
            }
        }
        Ok(())
    }

    fn witness_mismatch(
        comparison_kind: &str,
        left_execution_id: &str,
        right_execution_id: &str,
        left: &SubmissionWaveDeterminismArtifactWitness,
        right: &SubmissionWaveDeterminismArtifactWitness,
    ) -> FerrumError {
        FerrumError::backend(format!(
            "{comparison_kind} first witness mismatch between {left_execution_id} and {right_execution_id}: left=({},{},{},{},{},{},{},{},{},{}) right=({},{},{},{},{},{},{},{},{},{})",
            left.kind(),
            left.semantic_id(),
            left.node_id(),
            left.resource_id(),
            left.access(),
            left.participant_index(),
            left.logical_offset_bytes(),
            left.length_bytes(),
            left.element_type(),
            left.raw_sha256(),
            right.kind(),
            right.semantic_id(),
            right.node_id(),
            right.resource_id(),
            right.access(),
            right.participant_index(),
            right.logical_offset_bytes(),
            right.length_bytes(),
            right.element_type(),
            right.raw_sha256(),
        ))
    }

    fn ensure_poison_equivalence(zero: &PendingCase, a5: &PendingCase) -> Result<()> {
        let zero_execution = zero.executions.first().ok_or_else(|| {
            FerrumError::internal("zero-poison determinism case contains no execution")
        })?;
        let a5_execution = a5.executions.first().ok_or_else(|| {
            FerrumError::internal("a5-poison determinism case contains no execution")
        })?;
        ensure_execution_equivalence(zero_execution, a5_execution, "workspace_poison").map_err(
            |error| {
                FerrumError::backend(format!(
                    "workspace poison changed {} versus {}: {error}",
                    zero.case_id, a5.case_id
                ))
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_collection(
        stage: &Path,
        collected_models: Vec<CollectedModel>,
        denominator: &ExecutionDeterminismEvidenceDenominator,
        denominator_bytes: &[u8],
        denominator_fingerprint: &str,
        device_runtime_fingerprint: &str,
        device_fingerprint: &str,
        models_lock: FileReference,
        hardware_probe: FileReference,
        binary: FileReference,
    ) -> Result<CollectorManifest> {
        let denominator_path = stage.join("denominator.json");
        write_bytes_exclusive(&denominator_path, denominator_bytes)?;
        let denominator_ref = DenominatorReference {
            path: "denominator.json".to_owned(),
            sha256: file_sha256(&denominator_path)?,
            size_bytes: file_size(&denominator_path)?,
            fingerprint: denominator_fingerprint.to_owned(),
        };
        if denominator_ref.sha256 != denominator_ref.fingerprint {
            return Err(FerrumError::internal(
                "typed denominator fingerprint differs from exact serialized bytes",
            ));
        }

        let binary_sha256 = binary.sha256.clone();
        let mut case_refs = Vec::with_capacity(EXPECTED_CASES);
        let mut model_summaries = Vec::with_capacity(collected_models.len());
        let mut execution_count = 0;
        let mut comparison_count = 0;
        for model in collected_models {
            let identity = denominator
                .coverage()
                .models()
                .iter()
                .find(|identity| identity.model_key() == model.key)
                .ok_or_else(|| {
                    FerrumError::internal(format!(
                        "typed denominator lacks model identity {}",
                        model.key
                    ))
                })?;
            let targets = coverage_targets(denominator, &model.key)?;
            let case_count = model.cases.len();
            model_summaries.push(CollectorModelSummary {
                model_key: model.key.clone(),
                model_dir: model.directory.display().to_string(),
                resolved_plan_fingerprint: identity.resolved_plan_fingerprint().to_owned(),
                plan_hash: identity.plan_hash().as_str().to_owned(),
                dtype: model.dtype,
                quantization: model.quantization,
                case_count,
            });
            for pending in model.cases {
                let first = pending.executions.first().ok_or_else(|| {
                    FerrumError::internal(format!(
                        "determinism case {} contains no executions",
                        pending.case_id
                    ))
                })?;
                let initialization_input_sha256 =
                    first.initialization_identity().input_sha256().to_owned();
                let initialization_rng_sha256 =
                    first.initialization_identity().rng_sha256().to_owned();
                let initialization_state_sha256 = first
                    .initialization_identity()
                    .initial_state_sha256()
                    .to_owned();
                let case = CaseArtifact {
                    schema_version: 1,
                    case_id: pending.case_id.clone(),
                    denominator_fingerprint: denominator_fingerprint.to_owned(),
                    binary_sha256: binary_sha256.clone(),
                    device_runtime_implementation_fingerprint: device_runtime_fingerprint
                        .to_owned(),
                    device_fingerprint: device_fingerprint.to_owned(),
                    model_key: pending.model_key,
                    resolved_plan_fingerprint: identity.resolved_plan_fingerprint().to_owned(),
                    plan_hash: identity.plan_hash().as_str().to_owned(),
                    phase: pending.phase,
                    token_shape: pending.token_shape,
                    dtype: pending.dtype,
                    quantization: pending.quantization,
                    initialization: InitializationArtifact {
                        input_sha256: initialization_input_sha256,
                        rng_sha256: initialization_rng_sha256,
                        initial_state_kind: pending.initial_state_kind,
                        initial_state_sha256: initialization_state_sha256,
                        workspace_poison: pending.workspace_poison,
                    },
                    coverage_targets: targets.clone(),
                    executions: pending.executions,
                    comparisons: pending.comparisons,
                    first_mismatch: None,
                };
                execution_count += case.executions.len();
                comparison_count += case.comparisons.len();
                let relative = format!("cases/{}.json", case.case_id);
                let staged_path = stage.join(&relative);
                write_json_exclusive(&staged_path, &case)?;
                case_refs.push(FileReference {
                    path: relative,
                    sha256: file_sha256(&staged_path)?,
                    size_bytes: file_size(&staged_path)?,
                });
            }
        }
        case_refs.sort_by(|left, right| left.path.cmp(&right.path));
        if case_refs.len() != EXPECTED_CASES {
            return Err(FerrumError::internal(format!(
                "determinism collector produced {} cases, expected {EXPECTED_CASES}",
                case_refs.len()
            )));
        }
        let pass_line = format!(
            "FERRUM VNEXT DETERMINISM COLLECTOR PASS: {}",
            stage
                .parent()
                .expect("stage is inside artifact root")
                .display()
        );
        let manifest = CollectorManifest {
            schema_version: 1,
            artifact_type: ARTIFACT_TYPE,
            status: "pass",
            backend: "cuda",
            models_lock,
            hardware_probe,
            device_fingerprint: device_fingerprint.to_owned(),
            binary,
            denominator: denominator_ref,
            models: model_summaries,
            cases: case_refs,
            case_count: EXPECTED_CASES,
            execution_count,
            comparison_count,
            pass_line,
        };
        write_json_exclusive(&stage.join("collector.json"), &manifest)?;
        Ok(manifest)
    }

    fn coverage_targets(
        denominator: &ExecutionDeterminismEvidenceDenominator,
        model_key: &str,
    ) -> Result<Vec<CoverageTargetArtifact>> {
        let mut targets = Vec::new();
        let mut replay_equivalence = None;
        for requirement in denominator.coverage().provider_requirements() {
            let Some(selection) = requirement
                .model_selections()
                .iter()
                .find(|selection| selection.model_key() == model_key)
            else {
                continue;
            };
            let evidence = denominator
                .provider_evidence()
                .iter()
                .find(|evidence| {
                    evidence.model_key() == model_key
                        && evidence.operation_id() == requirement.operation_id()
                        && evidence.provider_id() == requirement.provider_id()
                })
                .ok_or_else(|| {
                    FerrumError::internal(format!(
                        "typed denominator lacks provider evidence for {model_key}/{}/{}",
                        requirement.operation_id(),
                        requirement.provider_id()
                    ))
                })?;
            match replay_equivalence {
                None => replay_equivalence = Some(requirement.replay_equivalence()),
                Some(expected) if expected != requirement.replay_equivalence() => {
                    return Err(FerrumError::unsupported(format!(
                        "model {model_key} mixes replay equivalence contracts inside one full-program determinism wave"
                    )));
                }
                Some(_) => {}
            }
            targets.push(CoverageTargetArtifact {
                operation_id: requirement.operation_id().to_string(),
                operation_version: requirement.operation_version(),
                operation_fingerprint: requirement.operation_fingerprint().to_owned(),
                provider_id: requirement.provider_id().to_string(),
                provider_version: requirement.provider_version(),
                provider_implementation_fingerprint: requirement
                    .provider_implementation_fingerprint()
                    .to_owned(),
                provider_execution_contract_fingerprint: requirement
                    .provider_execution_contract_fingerprint()
                    .to_string(),
                replay_equivalence: requirement.replay_equivalence().as_str().to_owned(),
                witness_plan_fingerprint: evidence.witness_plan_fingerprint().to_owned(),
                node_ids: selection
                    .node_ids()
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            });
        }
        if targets.is_empty() {
            return Err(FerrumError::internal(format!(
                "typed denominator selected no provider targets for model {model_key}"
            )));
        }
        if replay_equivalence != Some(ProviderReplayEquivalence::BitwiseEagerEquivalent) {
            return Err(FerrumError::unsupported(format!(
                "model {model_key} does not authorize bitwise eager/replay comparison for every selected provider"
            )));
        }
        targets.sort_by(|left, right| {
            (&left.operation_id, &left.provider_id).cmp(&(&right.operation_id, &right.provider_id))
        });
        Ok(targets)
    }

    fn reject_existing_outputs(root: &Path) -> Result<()> {
        for relative in [
            "denominator.json",
            "cases",
            "collector.json",
            "collector.reject.json",
            "collector-progress",
        ] {
            let path = root.join(relative);
            if path.exists() {
                return Err(FerrumError::invalid_parameter(format!(
                    "determinism output already exists and will not be overwritten: {}",
                    path.display()
                )));
            }
        }
        Ok(())
    }

    fn read_hardware_probe(path: &Path) -> Result<HardwareProbeIdentity> {
        let bytes = fs::read(path).map_err(|error| {
            FerrumError::io(format!(
                "cannot read CUDA hardware probe {}: {error}",
                path.display()
            ))
        })?;
        let probe: HardwareProbeIdentity = serde_json::from_slice(&bytes).map_err(|error| {
            FerrumError::serialization(format!(
                "cannot decode CUDA hardware probe {}: {error}",
                path.display()
            ))
        })?;
        if probe.schema_version != 1 || !is_sha256(&probe.fingerprint) {
            return Err(FerrumError::invalid_parameter(
                "CUDA hardware probe identity is not schema v1 with a lowercase SHA256",
            ));
        }
        Ok(probe)
    }

    fn publish_stage(stage: &Path, root: &Path) -> Result<()> {
        for relative in ["denominator.json", "cases", "collector.json"] {
            let source = stage.join(relative);
            let destination = root.join(relative);
            fs::rename(&source, &destination).map_err(|error| {
                FerrumError::io(format!(
                    "cannot publish determinism artifact {} to {}: {error}",
                    source.display(),
                    destination.display()
                ))
            })?;
        }
        fs::remove_dir(stage).map_err(|error| {
            FerrumError::io(format!(
                "cannot remove empty determinism staging directory {}: {error}",
                stage.display()
            ))
        })?;
        Ok(())
    }

    fn file_reference(root: &Path, path: &Path, expected_relative: &str) -> Result<FileReference> {
        let canonical_root = root.canonicalize().map_err(|error| {
            FerrumError::io(format!(
                "cannot canonicalize artifact root {}: {error}",
                root.display()
            ))
        })?;
        let canonical_path = path.canonicalize().map_err(|error| {
            FerrumError::io(format!("cannot canonicalize {}: {error}", path.display()))
        })?;
        let relative = canonical_path
            .strip_prefix(&canonical_root)
            .map_err(|_| {
                FerrumError::invalid_parameter(format!(
                    "artifact input must be inside artifact root: {}",
                    path.display()
                ))
            })?
            .to_string_lossy()
            .replace('\\', "/");
        if relative != expected_relative {
            return Err(FerrumError::invalid_parameter(format!(
                "artifact input path must be {expected_relative}, got {relative}"
            )));
        }
        Ok(FileReference {
            path: relative,
            sha256: file_sha256(path)?,
            size_bytes: file_size(path)?,
        })
    }

    fn absolute_file_reference(path: &Path) -> Result<FileReference> {
        Ok(FileReference {
            path: path.display().to_string(),
            sha256: file_sha256(path)?,
            size_bytes: file_size(path)?,
        })
    }

    fn file_size(path: &Path) -> Result<u64> {
        path.metadata()
            .map(|metadata| metadata.len())
            .map_err(|error| FerrumError::io(format!("cannot stat {}: {error}", path.display())))
    }

    fn file_sha256(path: &Path) -> Result<String> {
        let mut file = File::open(path)
            .map_err(|error| FerrumError::io(format!("cannot open {}: {error}", path.display())))?;
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = file.read(&mut buffer).map_err(|error| {
                FerrumError::io(format!("cannot hash {}: {error}", path.display()))
            })?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
        Ok(format!("{digest:x}"))
    }

    fn write_json_exclusive(path: &Path, value: &impl Serialize) -> Result<()> {
        let mut bytes = serde_json::to_vec_pretty(value)
            .map_err(|error| FerrumError::serialization(error.to_string()))?;
        bytes.push(b'\n');
        write_bytes_exclusive(path, &bytes)
    }

    fn write_bytes_exclusive(path: &Path, bytes: &[u8]) -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| {
                FerrumError::io(format!("cannot create {}: {error}", path.display()))
            })?;
        file.write_all(bytes).map_err(|error| {
            FerrumError::io(format!("cannot write {}: {error}", path.display()))
        })?;
        file.sync_all()
            .map_err(|error| FerrumError::io(format!("cannot sync {}: {error}", path.display())))
    }

    fn is_sha256(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    }

    fn vnext_backend_error(error: VNextError) -> FerrumError {
        FerrumError::backend(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_bindings_are_exact_and_canonical() {
        let values = vec![
            "m3-qwen3-30b-a3b=/models/m3".to_owned(),
            "m1-qwen35-4b=/models/m1".to_owned(),
            "m2-qwen35-35b-a3b=/models/m2".to_owned(),
        ];
        let bindings = parse_model_bindings(&values, false).unwrap();
        assert_eq!(
            bindings
                .iter()
                .map(|binding| binding.key.as_str())
                .collect::<Vec<_>>(),
            PRIMARY_MODEL_KEYS
        );
    }

    #[test]
    fn model_bindings_reject_missing_duplicate_and_unknown_models() {
        let missing = vec![
            "m1-qwen35-4b=/models/m1".to_owned(),
            "m2-qwen35-35b-a3b=/models/m2".to_owned(),
        ];
        assert!(parse_model_bindings(&missing, false).is_err());

        let duplicate = vec![
            "m1-qwen35-4b=/models/m1".to_owned(),
            "m1-qwen35-4b=/models/other".to_owned(),
            "m2-qwen35-35b-a3b=/models/m2".to_owned(),
            "m3-qwen3-30b-a3b=/models/m3".to_owned(),
        ];
        assert!(parse_model_bindings(&duplicate, false).is_err());

        let unknown = vec![
            "m1-qwen35-4b=/models/m1".to_owned(),
            "m2-qwen35-35b-a3b=/models/m2".to_owned(),
            "llama=/models/llama".to_owned(),
        ];
        assert!(parse_model_bindings(&unknown, false).is_err());
    }

    #[test]
    fn release_shapes_cover_the_exact_bounded_partition_matrix() {
        let fixtures = release_shape_fixtures();
        assert_eq!(fixtures.len(), 6);
        assert_eq!(
            fixtures
                .iter()
                .map(|fixture| (fixture.phase.as_str(), fixture.partition))
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                ("prefill", "single_token"),
                ("prefill", "multi_token"),
                ("prefill", "chunk_boundary"),
                ("decode", "c1"),
                ("decode", "multi_participant"),
                ("decode", "c32"),
            ])
        );
        assert_eq!(
            fixtures
                .iter()
                .find(|fixture| fixture.partition == "c32")
                .unwrap()
                .participants
                .len(),
            MAX_VNEXT_DETERMINISM_PARTICIPANTS
        );
        assert!(fixtures.iter().all(|fixture| fixture
            .participants
            .iter()
            .all(|participant| participant.to_spec().is_ok())));
    }

    #[test]
    fn case_denominator_is_three_models_times_exact_fixture_cross_product() {
        let states = [
            VNextDeterminismInitialState::Zero,
            VNextDeterminismInitialState::Nonzero,
        ];
        let poisons = [
            VNextDeterminismWorkspacePoison::Zero,
            VNextDeterminismWorkspacePoison::A5,
        ];
        assert_eq!(
            PRIMARY_MODEL_KEYS.len()
                * release_shape_fixtures().len()
                * states.len()
                * poisons.len(),
            EXPECTED_CASES
        );
    }
}
