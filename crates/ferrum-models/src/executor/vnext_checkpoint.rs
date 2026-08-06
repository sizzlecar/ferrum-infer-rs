use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ferrum_interfaces::vnext::{
    CompletionReadbackBatchRequest, CompletionReadbackOutput, ExecutionPlan, HostTransferLayout,
    ProgramPlanCompileOptions, ProgramValueId, ResourceWorkShape, RetainedCompletionValue, RunId,
    TokenSpanWork,
};
use ferrum_types::{
    FerrumError, RequestId, Result, TokenId, VNextCheckpointCaptureConfig,
    VNextTeacherForcingConfig,
};
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};

const MAX_CHECKPOINT_VALUES: usize = 63;
const MAX_PREFILL_WAVES: usize = 16;
const MAX_DECODE_WAVES: usize = 512;
const CHECKPOINT_SCHEMA_VERSION: u32 = 3;
const TEACHER_FORCING_CHECKPOINT_SCHEMA_VERSION: u32 = 4;
const TEACHER_PROMPT_MANIFEST_FILE: &str = "teacher-prompt.json";

pub(super) struct VNextCheckpointSelection {
    output_dir: PathBuf,
    value_ids: Vec<ProgramValueId>,
    maximum_prefill_waves: usize,
    maximum_decode_waves: usize,
    capture_product_output: bool,
    teacher_forcing: Option<VNextTeacherForcingConfig>,
}

impl VNextCheckpointSelection {
    pub(super) fn from_config(
        config: Option<&VNextCheckpointCaptureConfig>,
    ) -> Result<Option<Self>> {
        let Some(config) = config else {
            return Ok(None);
        };
        if config.output_dir.as_os_str().is_empty() {
            return Err(FerrumError::config(
                "vNext checkpoint output directory cannot be empty",
            ));
        }
        if (config.value_ids.is_empty() && !config.capture_product_output)
            || config.value_ids.len() > MAX_CHECKPOINT_VALUES
        {
            return Err(FerrumError::config(format!(
                "vNext checkpoint requires product output capture or 1..={MAX_CHECKPOINT_VALUES} values"
            )));
        }
        if config.maximum_prefill_waves == 0 || config.maximum_prefill_waves > MAX_PREFILL_WAVES {
            return Err(FerrumError::config(format!(
                "vNext checkpoint prefill wave count must be in 1..={MAX_PREFILL_WAVES}"
            )));
        }
        if config.maximum_decode_waves > MAX_DECODE_WAVES {
            return Err(FerrumError::config(format!(
                "vNext checkpoint decode wave count must be in 0..={MAX_DECODE_WAVES}"
            )));
        }
        if let Some(teacher) = &config.teacher_forcing {
            teacher.validate().map_err(FerrumError::config)?;
            if !config.capture_product_output {
                return Err(FerrumError::config(
                    "vNext checkpoint teacher forcing requires product-output capture",
                ));
            }
            if config.maximum_prefill_waves != 1
                || config.maximum_decode_waves != teacher.token_count().saturating_sub(1)
            {
                return Err(FerrumError::config(format!(
                    "vNext checkpoint teacher forcing requires one prefill wave and {} decode waves",
                    teacher.token_count().saturating_sub(1)
                )));
            }
        }
        let value_ids = config
            .value_ids
            .iter()
            .map(|value| {
                ProgramValueId::new(value.clone()).map_err(|error| {
                    FerrumError::config(format!(
                        "invalid vNext checkpoint value {value:?}: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if value_ids.iter().collect::<BTreeSet<_>>().len() != value_ids.len() {
            return Err(FerrumError::config(
                "vNext checkpoint values contain duplicates",
            ));
        }
        let mut value_ids = value_ids;
        value_ids.sort();
        Ok(Some(Self {
            output_dir: config.output_dir.clone(),
            value_ids,
            maximum_prefill_waves: config.maximum_prefill_waves,
            maximum_decode_waves: config.maximum_decode_waves,
            capture_product_output: config.capture_product_output,
            teacher_forcing: config.teacher_forcing.clone(),
        }))
    }

    pub(super) fn retain_in(&self, options: &mut ProgramPlanCompileOptions) {
        for value_id in &self.value_ids {
            options.retain_completion_value(value_id.clone());
        }
    }

    pub(super) fn bind(
        self,
        plan: &ExecutionPlan,
        model_id: String,
        family_fingerprint: String,
        program_fingerprint: String,
        run_id: &RunId,
        vocabulary_size: usize,
    ) -> Result<VNextCheckpointCapture> {
        if let Some(teacher) = &self.teacher_forcing {
            if let Some(token) = teacher
                .token_ids()
                .iter()
                .find(|token| usize::try_from(token.get()).map_or(true, |id| id >= vocabulary_size))
            {
                return Err(FerrumError::config(format!(
                    "vNext checkpoint teacher token {} is outside vocabulary {vocabulary_size}",
                    token.get()
                )));
            }
        }
        prepare_empty_output_directory(&self.output_dir)?;
        let checkpoints = self
            .value_ids
            .iter()
            .map(|value_id| plan.completion_checkpoint(value_id).cloned())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| FerrumError::model(error.to_string()))?;
        let capture = VNextCheckpointCapture {
            output_dir: self.output_dir,
            checkpoints,
            maximum_prefill_waves: self.maximum_prefill_waves,
            maximum_decode_waves: self.maximum_decode_waves,
            capture_product_output: self.capture_product_output,
            teacher_forcing: self.teacher_forcing.map(VNextTeacherForcingCapture::new),
            next_prefill_wave: AtomicUsize::new(0),
            next_decode_wave: AtomicUsize::new(0),
            armed: AtomicBool::new(false),
            plan_id: plan.payload().plan_id().to_string(),
            plan_hash: plan.plan_hash().to_string(),
            model_id,
            family_fingerprint,
            program_fingerprint,
            run_id: run_id.to_string(),
        };
        capture.write_plan_manifest()?;
        Ok(capture)
    }
}

pub(super) struct VNextCheckpointCapture {
    output_dir: PathBuf,
    checkpoints: Vec<RetainedCompletionValue>,
    maximum_prefill_waves: usize,
    maximum_decode_waves: usize,
    capture_product_output: bool,
    teacher_forcing: Option<VNextTeacherForcingCapture>,
    next_prefill_wave: AtomicUsize,
    next_decode_wave: AtomicUsize,
    armed: AtomicBool,
    plan_id: String,
    plan_hash: String,
    model_id: String,
    family_fingerprint: String,
    program_fingerprint: String,
    run_id: String,
}

struct VNextTeacherForcingCapture {
    config: VNextTeacherForcingConfig,
    token_ids_sha256: String,
    state: Mutex<VNextTeacherForcingState>,
}

#[derive(Default)]
struct VNextTeacherForcingState {
    owner_request_id: Option<RequestId>,
    prompt_token_ids: Option<Vec<u32>>,
    next_token_index: usize,
}

impl VNextTeacherForcingCapture {
    fn new(config: VNextTeacherForcingConfig) -> Self {
        let token_ids_sha256 = config.token_ids_sha256();
        Self {
            config,
            token_ids_sha256,
            state: Mutex::new(VNextTeacherForcingState::default()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VNextCheckpointWaveKind {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum VNextCheckpointProductOutputMode {
    FullLogits,
    GreedyToken,
}

impl VNextCheckpointWaveKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VNextCheckpointClaim {
    kind: VNextCheckpointWaveKind,
    capture_index: usize,
}

impl VNextCheckpointClaim {
    const fn new(kind: VNextCheckpointWaveKind, capture_index: usize) -> Self {
        Self {
            kind,
            capture_index,
        }
    }

    fn manifest_file_name(self) -> String {
        match self.kind {
            VNextCheckpointWaveKind::Prefill => {
                format!("wave-{:04}.json", self.capture_index)
            }
            VNextCheckpointWaveKind::Decode => {
                format!("decode-wave-{:04}.json", self.capture_index)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct VNextTeacherForcedDecision {
    token_index: usize,
    token_id: TokenId,
}

impl VNextTeacherForcedDecision {
    pub(super) const fn new(token_index: usize, token_id: TokenId) -> Self {
        Self {
            token_index,
            token_id,
        }
    }

    pub(super) const fn token_index(self) -> usize {
        self.token_index
    }

    pub(super) const fn token_id(self) -> TokenId {
        self.token_id
    }
}

impl VNextCheckpointCapture {
    pub(super) fn arm(&self) {
        self.armed.store(true, Ordering::Release);
    }

    pub(super) fn claim_prefill_wave(
        &self,
        participant_count: usize,
        request_id: Option<&RequestId>,
        token_ids: Option<&[u32]>,
        is_final_prefill: bool,
    ) -> Result<Option<VNextCheckpointClaim>> {
        self.claim_wave(
            VNextCheckpointWaveKind::Prefill,
            participant_count,
            request_id,
            token_ids,
            is_final_prefill,
        )
    }

    pub(super) fn claim_decode_wave(
        &self,
        participant_count: usize,
        request_id: Option<&RequestId>,
        token_ids: Option<&[u32]>,
    ) -> Result<Option<VNextCheckpointClaim>> {
        self.claim_wave(
            VNextCheckpointWaveKind::Decode,
            participant_count,
            request_id,
            token_ids,
            true,
        )
    }

    fn claim_wave(
        &self,
        kind: VNextCheckpointWaveKind,
        participant_count: usize,
        request_id: Option<&RequestId>,
        token_ids: Option<&[u32]>,
        is_final_prefill: bool,
    ) -> Result<Option<VNextCheckpointClaim>> {
        if !self.armed.load(Ordering::Acquire) {
            return Ok(None);
        }
        if let Some(teacher) = &self.teacher_forcing {
            return self.claim_teacher_forced_wave(
                teacher,
                kind,
                participant_count,
                request_id,
                token_ids,
                is_final_prefill,
            );
        }
        let (next_wave, maximum_waves) = match kind {
            VNextCheckpointWaveKind::Prefill => {
                (&self.next_prefill_wave, self.maximum_prefill_waves)
            }
            VNextCheckpointWaveKind::Decode => (&self.next_decode_wave, self.maximum_decode_waves),
        };
        Ok(next_wave
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < maximum_waves).then_some(current + 1)
            })
            .ok()
            .map(|capture_index| VNextCheckpointClaim::new(kind, capture_index)))
    }

    fn claim_teacher_forced_wave(
        &self,
        teacher: &VNextTeacherForcingCapture,
        kind: VNextCheckpointWaveKind,
        participant_count: usize,
        request_id: Option<&RequestId>,
        token_ids: Option<&[u32]>,
        is_final_prefill: bool,
    ) -> Result<Option<VNextCheckpointClaim>> {
        if participant_count != 1 {
            return Err(FerrumError::request_validation(format!(
                "vNext checkpoint teacher forcing requires one participant, got {participant_count}"
            )));
        }
        let request_id = request_id.ok_or_else(|| {
            FerrumError::internal("vNext checkpoint teacher-forced wave has no request identity")
        })?;
        let token_ids = token_ids.ok_or_else(|| {
            FerrumError::internal("vNext checkpoint teacher-forced wave has no token history")
        })?;
        if token_ids.is_empty() {
            return Err(FerrumError::request_validation(
                "vNext checkpoint teacher-forced wave has an empty token history",
            ));
        }
        let mut state = teacher.state.lock();
        if let Some(owner) = &state.owner_request_id {
            if owner != request_id {
                return Err(FerrumError::request_validation(format!(
                    "vNext checkpoint teacher-forced owner is `{owner}`, got `{request_id}`"
                )));
            }
        }

        match kind {
            VNextCheckpointWaveKind::Prefill if !is_final_prefill => {
                if state.owner_request_id.is_some() || state.next_token_index != 0 {
                    return Err(FerrumError::request_validation(
                        "vNext checkpoint teacher forcing observed prefill after final prefill",
                    ));
                }
                if state.prompt_token_ids.is_some() {
                    return Err(FerrumError::internal(
                        "vNext checkpoint teacher forcing retained a prompt before final prefill",
                    ));
                }
                Ok(None)
            }
            VNextCheckpointWaveKind::Prefill => {
                if state.owner_request_id.is_some()
                    || state.prompt_token_ids.is_some()
                    || state.next_token_index != 0
                {
                    return Err(FerrumError::request_validation(
                        "vNext checkpoint teacher forcing observed an extra final prefill wave",
                    ));
                }
                state.owner_request_id = Some(request_id.clone());
                state.prompt_token_ids = Some(token_ids.to_vec());
                state.next_token_index = 1;
                Ok(Some(VNextCheckpointClaim::new(
                    VNextCheckpointWaveKind::Prefill,
                    0,
                )))
            }
            VNextCheckpointWaveKind::Decode => {
                if state.owner_request_id.is_none() || state.next_token_index == 0 {
                    return Err(FerrumError::request_validation(
                        "vNext checkpoint teacher-forced decode arrived before final prefill",
                    ));
                }
                let token_index = state.next_token_index;
                if token_index >= teacher.config.token_count() {
                    return Err(FerrumError::request_validation(
                        "vNext checkpoint teacher forcing observed an extra decode wave",
                    ));
                }
                let prompt_token_ids = state.prompt_token_ids.as_deref().ok_or_else(|| {
                    FerrumError::internal(
                        "vNext checkpoint teacher-forced decode has no canonical prompt",
                    )
                })?;
                let expected_tokens =
                    prompt_token_ids
                        .len()
                        .checked_add(token_index)
                        .ok_or_else(|| {
                            FerrumError::internal(
                                "vNext checkpoint teacher-forced history length overflows usize",
                            )
                        })?;
                let prompt_matches = token_ids.starts_with(prompt_token_ids);
                let forced_prefix_matches =
                    token_ids
                        .get(prompt_token_ids.len()..)
                        .is_some_and(|suffix| {
                            suffix.len() == token_index
                                && suffix
                                    .iter()
                                    .copied()
                                    .eq(teacher.config.token_ids()[..token_index]
                                        .iter()
                                        .map(|token| token.get()))
                        });
                if token_ids.len() != expected_tokens || !prompt_matches || !forced_prefix_matches {
                    return Err(FerrumError::request_validation(format!(
                        "vNext checkpoint teacher-forced decode token history differs at decision {token_index}"
                    )));
                }
                state.next_token_index += 1;
                Ok(Some(VNextCheckpointClaim::new(
                    VNextCheckpointWaveKind::Decode,
                    token_index - 1,
                )))
            }
        }
    }

    pub(super) fn teacher_forced_decision(
        &self,
        claim: VNextCheckpointClaim,
    ) -> Result<Option<VNextTeacherForcedDecision>> {
        let Some(teacher) = &self.teacher_forcing else {
            return Ok(None);
        };
        let token_index = match claim.kind {
            VNextCheckpointWaveKind::Prefill => 0,
            VNextCheckpointWaveKind::Decode => {
                claim.capture_index.checked_add(1).ok_or_else(|| {
                    FerrumError::internal("vNext teacher-forced token index overflow")
                })?
            }
        };
        let token_id = teacher
            .config
            .token_ids()
            .get(token_index)
            .copied()
            .ok_or_else(|| {
                FerrumError::internal(format!(
                    "vNext teacher-forced claim index {token_index} exceeds configured history"
                ))
            })?;
        Ok(Some(VNextTeacherForcedDecision::new(token_index, token_id)))
    }

    pub(super) fn checkpoints(&self) -> &[RetainedCompletionValue] {
        &self.checkpoints
    }

    pub(super) const fn captures_product_output(&self) -> bool {
        self.capture_product_output
    }

    fn schema_version(&self) -> u32 {
        if self.teacher_forcing.is_some() {
            TEACHER_FORCING_CHECKPOINT_SCHEMA_VERSION
        } else {
            CHECKPOINT_SCHEMA_VERSION
        }
    }

    pub(super) fn readback_batches(
        &self,
        plan: &ExecutionPlan,
        token_spans: &[&TokenSpanWork],
    ) -> Result<Vec<CompletionReadbackBatchRequest>> {
        self.checkpoints
            .iter()
            .map(|checkpoint| {
                let requests = token_spans
                    .iter()
                    .enumerate()
                    .map(|(participant_index, token_span)| {
                        let participant_index = u32::try_from(participant_index).map_err(|_| {
                            FerrumError::backend("vNext checkpoint participant index exceeds u32")
                        })?;
                        let work = ResourceWorkShape::single((*token_span).clone())
                            .map_err(|error| FerrumError::backend(error.to_string()))?;
                        plan.completion_checkpoint_readback_for_work(
                            checkpoint.value_id(),
                            participant_index,
                            &work,
                        )
                        .map_err(|error| FerrumError::backend(error.to_string()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                CompletionReadbackBatchRequest::new(requests)
                    .map_err(|error| FerrumError::backend(error.to_string()))
            })
            .collect()
    }

    pub(super) fn checkpoint_for_output(
        &self,
        output: &CompletionReadbackOutput,
    ) -> Option<&RetainedCompletionValue> {
        self.checkpoints.iter().find(|checkpoint| {
            checkpoint.producer_node_id() == output.request().node_id()
                && checkpoint.resource_id() == output.request().resource_id()
        })
    }

    pub(super) fn write_output(
        &self,
        claim: VNextCheckpointClaim,
        request_id: &RequestId,
        token_span: &TokenSpanWork,
        checkpoint: &RetainedCompletionValue,
        output: &CompletionReadbackOutput,
    ) -> Result<VNextCheckpointArtifactRecord> {
        if checkpoint.producer_node_id() != output.request().node_id()
            || checkpoint.resource_id() != output.request().resource_id()
            || checkpoint.logical_offset_bytes() != output.request().logical_offset_bytes()
            || checkpoint.tensor().element_type() != output.request().output_layout().element_type()
        {
            return Err(FerrumError::internal(
                "vNext checkpoint output does not match its retained semantic value",
            ));
        }
        let stem = checkpoint_file_stem(
            claim,
            output.request().participant_index(),
            checkpoint.value_id(),
        );
        let raw_file = format!("{stem}.bin");
        write_new_file(&self.output_dir.join(&raw_file), output.bytes())?;
        Ok(VNextCheckpointArtifactRecord {
            value: checkpoint.clone(),
            participant_index: output.request().participant_index(),
            request_id: request_id.to_string(),
            token_span: token_span.clone(),
            output_layout: output.request().output_layout(),
            raw_file,
            raw_bytes: u64::try_from(output.bytes().len()).unwrap_or(u64::MAX),
            raw_sha256: output.sha256().to_owned(),
        })
    }

    pub(super) fn write_product_output(
        &self,
        claim: VNextCheckpointClaim,
        request_id: &RequestId,
        token_span: &TokenSpanWork,
        output_mode: VNextCheckpointProductOutputMode,
        output: &CompletionReadbackOutput,
    ) -> Result<VNextCheckpointProductOutputRecord> {
        if !self.capture_product_output {
            return Err(FerrumError::internal(
                "vNext product-output checkpoint was not configured",
            ));
        }
        let stem =
            product_output_file_stem(claim, output.request().participant_index(), output_mode);
        let raw_file = format!("{stem}.bin");
        write_new_file(&self.output_dir.join(&raw_file), output.bytes())?;
        Ok(VNextCheckpointProductOutputRecord {
            output_mode,
            node_id: output.request().node_id().to_string(),
            resource_id: output.request().resource_id().to_string(),
            logical_offset_bytes: output.request().logical_offset_bytes(),
            participant_index: output.request().participant_index(),
            request_id: request_id.to_string(),
            token_span: token_span.clone(),
            output_layout: output.request().output_layout(),
            raw_file,
            raw_bytes: u64::try_from(output.bytes().len()).unwrap_or(u64::MAX),
            raw_sha256: output.sha256().to_owned(),
        })
    }

    pub(super) fn finish_wave(
        &self,
        claim: VNextCheckpointClaim,
        participant_count: usize,
        completion_fingerprint: &str,
        receipt_fingerprint: &str,
        mut records: Vec<VNextCheckpointArtifactRecord>,
        mut product_outputs: Vec<VNextCheckpointProductOutputRecord>,
    ) -> Result<()> {
        let teacher_forced_decision = self.teacher_forced_decision(claim)?;
        let mut teacher_prompt_manifest = None;
        let expected_records = self
            .checkpoints
            .len()
            .checked_mul(participant_count)
            .ok_or_else(|| {
                FerrumError::internal("vNext checkpoint record count overflows usize")
            })?;
        if records.len() != expected_records {
            return Err(FerrumError::internal(format!(
                "vNext checkpoint wave produced {} records, expected {expected_records}",
                records.len()
            )));
        }
        let observed = records
            .iter()
            .map(|record| (record.value.value_id().clone(), record.participant_index))
            .collect::<BTreeSet<_>>();
        if observed.len() != expected_records {
            return Err(FerrumError::internal(
                "vNext checkpoint wave contains duplicate semantic participant records",
            ));
        }
        let expected_product_outputs = usize::from(self.capture_product_output)
            .checked_mul(participant_count)
            .ok_or_else(|| {
                FerrumError::internal("vNext product-output record count overflows usize")
            })?;
        if product_outputs.len() != expected_product_outputs {
            return Err(FerrumError::internal(format!(
                "vNext checkpoint wave produced {} product outputs, expected {expected_product_outputs}",
                product_outputs.len()
            )));
        }
        let observed_product_participants = product_outputs
            .iter()
            .map(|record| record.participant_index)
            .collect::<BTreeSet<_>>();
        if observed_product_participants.len() != expected_product_outputs {
            return Err(FerrumError::internal(
                "vNext checkpoint wave contains duplicate product-output participant records",
            ));
        }
        if let Some(decision) = teacher_forced_decision {
            if participant_count != 1 || product_outputs.len() != 1 {
                return Err(FerrumError::internal(
                    "vNext teacher-forced checkpoint wave must contain one product participant",
                ));
            }
            let product = &product_outputs[0];
            if product.output_mode != VNextCheckpointProductOutputMode::FullLogits {
                return Err(FerrumError::internal(
                    "vNext teacher-forced checkpoint wave must persist full logits",
                ));
            }
            let teacher = self.teacher_forcing.as_ref().ok_or_else(|| {
                FerrumError::internal("vNext teacher-forced decision has no capture contract")
            })?;
            let state = teacher.state.lock();
            let owner = state.owner_request_id.as_ref().ok_or_else(|| {
                FerrumError::internal("vNext teacher-forced checkpoint has no request owner")
            })?;
            let prompt_token_ids = state.prompt_token_ids.as_ref().ok_or_else(|| {
                FerrumError::internal("vNext teacher-forced checkpoint has no canonical prompt")
            })?;
            if product.request_id != owner.to_string() {
                return Err(FerrumError::internal(format!(
                    "vNext teacher-forced checkpoint product request {} differs from owner {owner}",
                    product.request_id
                )));
            }
            if decision.token_index >= teacher.config.token_count() {
                return Err(FerrumError::internal(
                    "vNext teacher-forced checkpoint decision exceeds configured history",
                ));
            }
            let mut expected_history = prompt_token_ids.clone();
            expected_history.extend(
                teacher.config.token_ids()[..decision.token_index]
                    .iter()
                    .map(|token| token.get()),
            );
            validate_teacher_forced_token_span(&expected_history, &product.token_span)?;
            for record in &records {
                if record.request_id != owner.to_string() {
                    return Err(FerrumError::internal(format!(
                        "vNext teacher-forced checkpoint record request {} differs from owner {owner}",
                        record.request_id
                    )));
                }
                validate_teacher_forced_token_span(&expected_history, &record.token_span)?;
            }
            if claim.kind == VNextCheckpointWaveKind::Prefill {
                teacher_prompt_manifest = Some(VNextTeacherPromptManifest {
                    schema_version: 1,
                    encoding: "u32-le",
                    request_id: owner.to_string(),
                    token_count: prompt_token_ids.len(),
                    token_ids_sha256: token_ids_sha256(prompt_token_ids),
                    token_ids: prompt_token_ids.clone(),
                });
            }
        }
        records.sort_by(|left, right| {
            left.value
                .value_id()
                .cmp(right.value.value_id())
                .then_with(|| left.participant_index.cmp(&right.participant_index))
        });
        product_outputs.sort_by_key(|record| record.participant_index);
        if let Some(prompt_manifest) = &teacher_prompt_manifest {
            let bytes = serde_json::to_vec_pretty(prompt_manifest).map_err(|error| {
                FerrumError::internal(format!(
                    "serialize vNext checkpoint teacher prompt: {error}"
                ))
            })?;
            write_new_file(&self.output_dir.join(TEACHER_PROMPT_MANIFEST_FILE), &bytes)?;
        }
        let manifest = VNextCheckpointWaveManifest {
            schema_version: self.schema_version(),
            capture_index: claim.capture_index,
            plan_id: &self.plan_id,
            plan_hash: &self.plan_hash,
            model_id: &self.model_id,
            family_fingerprint: &self.family_fingerprint,
            program_fingerprint: &self.program_fingerprint,
            run_id: &self.run_id,
            wave_kind: claim.kind.as_str(),
            participant_count,
            completion_fingerprint,
            receipt_fingerprint,
            teacher_forced_decision,
            records: &records,
            product_outputs: &product_outputs,
        };
        let bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| {
            FerrumError::internal(format!("serialize vNext checkpoint wave: {error}"))
        })?;
        write_new_file(&self.output_dir.join(claim.manifest_file_name()), &bytes)
    }

    fn write_plan_manifest(&self) -> Result<()> {
        let teacher_forcing =
            self.teacher_forcing
                .as_ref()
                .map(|teacher| VNextCheckpointTeacherForcingManifest {
                    mode: "canonical-history",
                    encoding: "u32-le",
                    token_count: teacher.config.token_count(),
                    token_ids_sha256: &teacher.token_ids_sha256,
                    prompt_file: TEACHER_PROMPT_MANIFEST_FILE,
                });
        let manifest = VNextCheckpointPlanManifest {
            schema_version: self.schema_version(),
            plan_id: &self.plan_id,
            plan_hash: &self.plan_hash,
            model_id: &self.model_id,
            family_fingerprint: &self.family_fingerprint,
            program_fingerprint: &self.program_fingerprint,
            run_id: &self.run_id,
            maximum_prefill_waves: self.maximum_prefill_waves,
            maximum_decode_waves: self.maximum_decode_waves,
            capture_product_output: self.capture_product_output,
            teacher_forcing,
            checkpoints: &self.checkpoints,
        };
        let bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| {
            FerrumError::internal(format!("serialize vNext checkpoint plan: {error}"))
        })?;
        write_new_file(&self.output_dir.join("plan.json"), &bytes)
    }
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct VNextCheckpointArtifactRecord {
    value: RetainedCompletionValue,
    participant_index: u32,
    request_id: String,
    token_span: TokenSpanWork,
    output_layout: HostTransferLayout,
    raw_file: String,
    raw_bytes: u64,
    raw_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct VNextCheckpointProductOutputRecord {
    output_mode: VNextCheckpointProductOutputMode,
    node_id: String,
    resource_id: String,
    logical_offset_bytes: u64,
    participant_index: u32,
    request_id: String,
    token_span: TokenSpanWork,
    output_layout: HostTransferLayout,
    raw_file: String,
    raw_bytes: u64,
    raw_sha256: String,
}

#[derive(Serialize)]
struct VNextCheckpointPlanManifest<'a> {
    schema_version: u32,
    plan_id: &'a str,
    plan_hash: &'a str,
    model_id: &'a str,
    family_fingerprint: &'a str,
    program_fingerprint: &'a str,
    run_id: &'a str,
    maximum_prefill_waves: usize,
    maximum_decode_waves: usize,
    capture_product_output: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    teacher_forcing: Option<VNextCheckpointTeacherForcingManifest<'a>>,
    checkpoints: &'a [RetainedCompletionValue],
}

#[derive(Serialize)]
struct VNextCheckpointTeacherForcingManifest<'a> {
    mode: &'static str,
    encoding: &'static str,
    token_count: usize,
    token_ids_sha256: &'a str,
    prompt_file: &'static str,
}

#[derive(Serialize)]
struct VNextTeacherPromptManifest {
    schema_version: u32,
    encoding: &'static str,
    request_id: String,
    token_count: usize,
    token_ids_sha256: String,
    token_ids: Vec<u32>,
}

#[derive(Serialize)]
struct VNextCheckpointWaveManifest<'a> {
    schema_version: u32,
    capture_index: usize,
    plan_id: &'a str,
    plan_hash: &'a str,
    model_id: &'a str,
    family_fingerprint: &'a str,
    program_fingerprint: &'a str,
    run_id: &'a str,
    wave_kind: &'static str,
    participant_count: usize,
    completion_fingerprint: &'a str,
    receipt_fingerprint: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    teacher_forced_decision: Option<VNextTeacherForcedDecision>,
    records: &'a [VNextCheckpointArtifactRecord],
    product_outputs: &'a [VNextCheckpointProductOutputRecord],
}

fn token_ids_sha256(token_ids: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token_id in token_ids {
        digest.update(token_id.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn validate_teacher_forced_token_span(
    expected_history: &[u32],
    token_span: &TokenSpanWork,
) -> Result<()> {
    let immediate_range = token_span.immediate_token_range();
    let start = usize::try_from(immediate_range.start).map_err(|_| {
        FerrumError::internal("vNext teacher-forced token-span start exceeds usize")
    })?;
    let end = usize::try_from(immediate_range.end)
        .map_err(|_| FerrumError::internal("vNext teacher-forced token-span end exceeds usize"))?;
    let fit_input_tokens = usize::try_from(token_span.fit_input_tokens()).map_err(|_| {
        FerrumError::internal("vNext teacher-forced token-span fit ceiling exceeds usize")
    })?;
    let expected =
        TokenSpanWork::from_token_ids_with_fit(expected_history, start..end, fit_input_tokens)
            .map_err(|error| {
                FerrumError::internal(format!(
                    "reconstruct vNext teacher-forced token-span evidence: {error}"
                ))
            })?;
    if expected != *token_span {
        return Err(FerrumError::internal(format!(
            "vNext teacher-forced token-span fingerprint {} differs from canonical history {}",
            token_span.fingerprint(),
            expected.fingerprint()
        )));
    }
    Ok(())
}

fn prepare_empty_output_directory(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(FerrumError::config(
                    "vNext checkpoint output path must be a real directory",
                ));
            }
            if fs::read_dir(path)
                .map_err(|error| checkpoint_io_error("inspect output directory", error))?
                .next()
                .is_some()
            {
                return Err(FerrumError::config(
                    "vNext checkpoint output directory must be empty",
                ));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(path)
                .map_err(|error| checkpoint_io_error("create output directory", error))?;
        }
        Err(error) => return Err(checkpoint_io_error("inspect output directory", error)),
    }
    Ok(())
}

fn checkpoint_file_stem(
    claim: VNextCheckpointClaim,
    participant_index: u32,
    value_id: &ProgramValueId,
) -> String {
    let slug = value_id
        .as_str()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    let digest = format!("{:x}", Sha256::digest(value_id.as_str().as_bytes()));
    let kind_prefix = match claim.kind {
        VNextCheckpointWaveKind::Prefill => "",
        VNextCheckpointWaveKind::Decode => "decode-",
    };
    format!(
        "{kind_prefix}capture-{:04}-participant-{participant_index:04}-{slug}-{}",
        claim.capture_index,
        &digest[..12]
    )
}

fn product_output_file_stem(
    claim: VNextCheckpointClaim,
    participant_index: u32,
    output_mode: VNextCheckpointProductOutputMode,
) -> String {
    let kind_prefix = match claim.kind {
        VNextCheckpointWaveKind::Prefill => "",
        VNextCheckpointWaveKind::Decode => "decode-",
    };
    let mode = match output_mode {
        VNextCheckpointProductOutputMode::FullLogits => "full-logits",
        VNextCheckpointProductOutputMode::GreedyToken => "greedy-token",
    };
    format!(
        "{kind_prefix}product-output-{:04}-participant-{participant_index:04}-{mode}",
        claim.capture_index
    )
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| checkpoint_io_error("create evidence file", error))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| checkpoint_io_error("write evidence file", error))
}

fn checkpoint_io_error(context: &'static str, error: std::io::Error) -> FerrumError {
    FerrumError::internal(format!("vNext checkpoint {context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(
        maximum_prefill_waves: usize,
        maximum_decode_waves: usize,
    ) -> VNextCheckpointCapture {
        VNextCheckpointCapture {
            output_dir: PathBuf::new(),
            checkpoints: Vec::new(),
            maximum_prefill_waves,
            maximum_decode_waves,
            capture_product_output: false,
            teacher_forcing: None,
            next_prefill_wave: AtomicUsize::new(0),
            next_decode_wave: AtomicUsize::new(0),
            armed: AtomicBool::new(false),
            plan_id: "plan.test".to_owned(),
            plan_hash: "plan-hash".to_owned(),
            model_id: "model.test".to_owned(),
            family_fingerprint: "family-hash".to_owned(),
            program_fingerprint: "program-hash".to_owned(),
            run_id: "run.test".to_owned(),
        }
    }

    #[test]
    fn absent_capture_config_preserves_the_default_path() {
        assert!(VNextCheckpointSelection::from_config(None)
            .unwrap()
            .is_none());
    }

    #[test]
    fn capture_config_is_canonical_and_rejects_invalid_bounds() {
        let directory = tempfile::tempdir().unwrap();
        let config = VNextCheckpointCaptureConfig {
            output_dir: directory.path().join("capture"),
            value_ids: vec!["value.z".to_owned(), "value.a".to_owned()],
            maximum_prefill_waves: 2,
            maximum_decode_waves: 64,
            capture_product_output: false,
            teacher_forcing: None,
        };
        let selection = VNextCheckpointSelection::from_config(Some(&config))
            .unwrap()
            .unwrap();
        assert_eq!(
            selection
                .value_ids
                .iter()
                .map(ProgramValueId::as_str)
                .collect::<Vec<_>>(),
            ["value.a", "value.z"]
        );

        let duplicate = VNextCheckpointCaptureConfig {
            value_ids: vec!["value.a".to_owned(), "value.a".to_owned()],
            ..config.clone()
        };
        assert!(VNextCheckpointSelection::from_config(Some(&duplicate)).is_err());
        let zero_waves = VNextCheckpointCaptureConfig {
            maximum_prefill_waves: 0,
            ..config
        };
        assert!(VNextCheckpointSelection::from_config(Some(&zero_waves)).is_err());
        let excessive_decode_waves = VNextCheckpointCaptureConfig {
            maximum_prefill_waves: 1,
            maximum_decode_waves: MAX_DECODE_WAVES + 1,
            ..zero_waves
        };
        assert!(VNextCheckpointSelection::from_config(Some(&excessive_decode_waves)).is_err());
    }

    #[test]
    fn product_output_only_capture_does_not_change_compile_options() {
        let directory = tempfile::tempdir().unwrap();
        let config = VNextCheckpointCaptureConfig {
            output_dir: directory.path().join("capture"),
            value_ids: Vec::new(),
            maximum_prefill_waves: 1,
            maximum_decode_waves: 64,
            capture_product_output: true,
            teacher_forcing: None,
        };
        let selection = VNextCheckpointSelection::from_config(Some(&config))
            .unwrap()
            .unwrap();
        let mut options =
            ProgramPlanCompileOptions::new(std::collections::BTreeMap::new()).unwrap();
        let unchanged = options.clone();

        selection.retain_in(&mut options);

        assert_eq!(options, unchanged);
        assert!(selection.value_ids.is_empty());
        assert!(selection.capture_product_output);
    }

    #[test]
    fn capture_only_claims_bounded_prefill_and_decode_waves_after_startup_arm() {
        let capture = capture(2, 1);
        let request_id = RequestId::new();
        assert_eq!(
            capture
                .claim_prefill_wave(1, Some(&request_id), None, true)
                .unwrap(),
            None
        );
        assert_eq!(
            capture
                .claim_decode_wave(1, Some(&request_id), None)
                .unwrap(),
            None
        );
        capture.arm();
        assert_eq!(
            capture
                .claim_prefill_wave(1, Some(&request_id), None, true)
                .unwrap(),
            Some(VNextCheckpointClaim::new(
                VNextCheckpointWaveKind::Prefill,
                0
            ))
        );
        assert_eq!(
            capture
                .claim_prefill_wave(1, Some(&request_id), None, true)
                .unwrap(),
            Some(VNextCheckpointClaim::new(
                VNextCheckpointWaveKind::Prefill,
                1
            ))
        );
        assert_eq!(
            capture
                .claim_prefill_wave(1, Some(&request_id), None, true)
                .unwrap(),
            None
        );
        assert_eq!(
            capture
                .claim_decode_wave(1, Some(&request_id), None)
                .unwrap(),
            Some(VNextCheckpointClaim::new(
                VNextCheckpointWaveKind::Decode,
                0
            ))
        );
        assert_eq!(
            capture
                .claim_decode_wave(1, Some(&request_id), None)
                .unwrap(),
            None
        );
    }

    #[test]
    fn teacher_forcing_claims_final_prefill_then_same_owner_decode() {
        let mut capture = capture(1, 1);
        capture.capture_product_output = true;
        capture.teacher_forcing = Some(VNextTeacherForcingCapture::new(
            VNextTeacherForcingConfig::new(vec![TokenId::new(11690), TokenId::new(369)]).unwrap(),
        ));
        let owner = RequestId::new();
        let other = RequestId::new();
        capture.arm();

        assert!(capture
            .claim_prefill_wave(1, Some(&owner), Some(&[101]), false)
            .unwrap()
            .is_none());
        assert!(capture
            .claim_prefill_wave(2, Some(&owner), Some(&[101, 102]), true)
            .is_err());
        let prefill = capture
            .claim_prefill_wave(1, Some(&owner), Some(&[101, 102]), true)
            .unwrap()
            .unwrap();
        let decision = capture.teacher_forced_decision(prefill).unwrap().unwrap();
        assert_eq!(decision.token_index(), 0);
        assert_eq!(decision.token_id(), TokenId::new(11690));

        assert!(capture
            .claim_decode_wave(1, Some(&other), Some(&[101, 102, 11690]))
            .is_err());
        assert!(capture
            .claim_decode_wave(1, Some(&owner), Some(&[101, 102, 369]))
            .is_err());
        let decode = capture
            .claim_decode_wave(1, Some(&owner), Some(&[101, 102, 11690]))
            .unwrap()
            .unwrap();
        let decision = capture.teacher_forced_decision(decode).unwrap().unwrap();
        assert_eq!(decision.token_index(), 1);
        assert_eq!(decision.token_id(), TokenId::new(369));
        assert!(capture
            .claim_decode_wave(1, Some(&owner), Some(&[101, 102, 11690, 369]))
            .is_err());
        assert!(capture
            .claim_prefill_wave(1, Some(&owner), Some(&[101, 102]), false)
            .is_err());
        assert!(capture
            .claim_prefill_wave(1, Some(&owner), Some(&[101, 102]), true)
            .is_err());
    }

    #[test]
    fn decode_artifact_names_do_not_collide_with_legacy_prefill_names() {
        let prefill = VNextCheckpointClaim::new(VNextCheckpointWaveKind::Prefill, 0);
        let decode = VNextCheckpointClaim::new(VNextCheckpointWaveKind::Decode, 0);
        let value_id = ProgramValueId::new("value.output.logits").unwrap();

        assert_eq!(prefill.manifest_file_name(), "wave-0000.json");
        assert_eq!(decode.manifest_file_name(), "decode-wave-0000.json");
        assert_ne!(
            checkpoint_file_stem(prefill, 0, &value_id),
            checkpoint_file_stem(decode, 0, &value_id)
        );
        assert_ne!(
            product_output_file_stem(prefill, 0, VNextCheckpointProductOutputMode::FullLogits),
            product_output_file_stem(decode, 0, VNextCheckpointProductOutputMode::FullLogits)
        );
    }

    #[test]
    fn evidence_directory_and_files_are_create_once() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("capture");
        prepare_empty_output_directory(&output).unwrap();
        let evidence = output.join("evidence.bin");
        write_new_file(&evidence, b"first").unwrap();
        assert!(write_new_file(&evidence, b"replacement").is_err());
        assert!(prepare_empty_output_directory(&output).is_err());
        assert_eq!(fs::read(evidence).unwrap(), b"first");
    }
}
