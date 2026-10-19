//! A single diagnostic observation of an original participant's retained row.
//! No request registry, execution policy, device owner, or replay authority.
use super::*;
use ferrum_types::RetainedOutputCaptureConfig;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write};

pub(super) struct RetainedOutputCapture {
    config: RetainedOutputCaptureConfig,
    available: AtomicBool,
}

pub(super) struct CaptureAttempt {
    owner: Arc<RetainedOutputCapture>,
    participant: usize,
    expected: CompletionReadbackRequest,
    metadata: Value,
    finished: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum Selection {
    None,
    Unique { index: usize, history_matches: bool },
    Ambiguous,
}

struct Input<'a> {
    span: &'a TokenSpanWork,
    tokens: &'a [u32],
    prompt_tokens: u64,
    product: bool,
}

fn token_digest(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn select<'a>(
    config: &RetainedOutputCaptureConfig,
    inputs: impl Iterator<Item = Input<'a>>,
) -> Selection {
    let mut selected = Selection::None;
    for (index, input) in inputs.enumerate() {
        // No hashing or allocation for an unrelated span, origin, or prompt length.
        if !input.product
            || input.span.immediate_token_range() != config.source_span
            || input.tokens.len() as u64 != config.source_span.end
            || input.prompt_tokens != config.prompt_token_count as u64
            || input.tokens.len() < config.prompt_token_count
        {
            continue;
        }
        if token_digest(&input.tokens[..config.prompt_token_count]) != config.prompt_token_sha256 {
            continue;
        }
        if selected != Selection::None {
            return Selection::Ambiguous;
        }
        selected = Selection::Unique {
            index,
            history_matches: token_digest(input.tokens) == config.consumed_history_sha256,
        };
    }
    selected
}

impl RetainedOutputCapture {
    pub(super) fn new(config: RetainedOutputCaptureConfig) -> Result<Self> {
        config.validate().map_err(FerrumError::config)?;
        let capture = Self {
            config,
            available: AtomicBool::new(true),
        };
        // Bad diagnostic storage disables capture, not the product request.
        let setup = (|| -> std::io::Result<()> {
            fs::create_dir_all(&capture.config.output_dir)?;
            if fs::read_dir(&capture.config.output_dir)?.next().is_some() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "capture directory must be empty",
                ));
            }
            capture.write_report(json!({"status":"not_captured", "reason":"target_not_observed", "selector":capture.config}))
        })();
        if let Err(error) = setup {
            capture.available.store(false, Ordering::Release);
            tracing::warn!(%error, "retained output diagnostic disabled: storage setup failed");
        }
        Ok(capture)
    }

    fn write_report(&self, report: Value) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(&report)?;
        let temporary = self.config.output_dir.join("report.next.json");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        drop(file);
        fs::rename(temporary, self.config.output_dir.join("report.json"))
    }

    fn record(&self, report: Value) {
        if let Err(error) = self.write_report(report) {
            self.available.store(false, Ordering::Release);
            tracing::warn!(%error, "retained output diagnostic failed to write its report");
        }
    }

    fn select_once<'a>(&self, inputs: impl Iterator<Item = Input<'a>>) -> Option<Selection> {
        if !self.available.load(Ordering::Acquire) {
            return None;
        }
        let selected = select(&self.config, inputs);
        if selected == Selection::None {
            return None;
        }
        self.available
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| selected)
    }

    pub(super) fn try_claim<R: DeviceRuntime>(
        self: &Arc<Self>,
        participants: &[VNextExecutionParticipant<'_, R>],
        modes: &ProductReadbackModes,
        readbacks: &VNextTerminalReadbacks,
        io: &VNextIoBinding,
    ) -> Option<CaptureAttempt> {
        let selected = self.select_once(participants.iter().map(|p| Input {
            span: p.span,
            tokens: p.tokens,
            prompt_tokens: p.sequence.product_prompt_tokens,
            product: p.sequence.request_origin == ExecutorRequestOrigin::Product,
        }))?;
        // One claim for this executor instance. Cancellation or a replacement
        // incarnation never re-arms it; no later caller can inherit the claim.
        let Selection::Unique {
            index,
            history_matches,
        } = selected
        else {
            self.record(json!({"status":"ambiguous", "reason":"multiple_original_participants_match_prompt_at_span"}));
            return None;
        };
        let participant = &participants[index];
        let binding = &participant.sequence.active_binding;
        let mut metadata = json!({
            "status":"claimed", "selector":self.config,
            "request_id":participant.sequence.request_id().to_string(),
            "active_sequence_fingerprint":binding.fingerprint(),
            "sequence_authority":binding.sequence_authority(),
            "activation_epoch":binding.activation_epoch(),
            "participant_index":index, "token_span":participant.span,
            "consumed_history_sha256":token_digest(participant.tokens),
            "consumed_token_ids":participant.tokens,
            "native_kernel_launch_facts":{"status":"unavailable_at_this_diagnostic_seam"},
        });
        let unavailable = if !history_matches {
            Some("history_mismatch")
        } else if participant.sequence.replayed_output_tokens != 0 {
            Some("recomputed_incarnation_unsupported")
        } else if !matches!(readbacks, VNextTerminalReadbacks::Batch(_))
            || modes.uniform().is_none()
        {
            Some("nonuniform_or_checkpoint_readback_unsupported")
        } else if io.output_layout.element_type() != ElementType::F32 {
            Some("non_f32_output_unsupported")
        } else {
            None
        };
        if let Some(reason) = unavailable {
            metadata["status"] = json!(reason);
            self.record(metadata);
            return None;
        }
        let Ok(participant_index) = u32::try_from(index) else {
            metadata["status"] = json!("unavailable");
            metadata["reason"] = json!("participant index exceeds typed readback bounds");
            self.record(metadata);
            return None;
        };
        let expected = CompletionReadbackRequest::new(
            io.output_node_id.clone(),
            participant_index,
            io.output_resource_id.clone(),
            io.output_offset_bytes,
            io.output_layout,
        );
        let expected = match expected {
            Ok(request)
                if request
                    .output_layout()
                    .byte_len()
                    .is_ok_and(|bytes| bytes <= self.config.maximum_readback_bytes) =>
            {
                request
            }
            other => {
                metadata["status"] = json!("unavailable");
                metadata["reason"] = json!(format!("bounded retained row unavailable: {other:?}"));
                self.record(metadata);
                return None;
            }
        };
        metadata["cohort"] = json!(participants
            .iter()
            .map(|p| json!({
                "request_id":p.sequence.request_id().to_string(),
                "active_sequence_fingerprint":p.sequence.active_binding.fingerprint(),
                "sequence_authority":p.sequence.active_binding.sequence_authority(),
                "activation_epoch":p.sequence.active_binding.activation_epoch(),
                "token_span":p.span, "consumed_history_sha256":token_digest(p.tokens),
            }))
            .collect::<Vec<_>>());
        metadata["sampling_policy"] = match participant.output_role.logits_policy() {
            Some(LogitsReturnPolicy::GreedyArgmax {
                token_mask,
                repetition_penalty,
            }) => json!({
                "kind":"greedy_argmax",
                "mask":token_mask.as_ref().map(|mask| json!({
                    "fingerprint":mask.fingerprint, "length":mask.len(),
                    "invalid_token_ids":mask.valid_token_mask.iter().enumerate().filter_map(|(i, flag)| (*flag == 0).then_some(i)).collect::<Vec<_>>(),
                })),
                "repetition":repetition_penalty.as_ref().map(|penalty| json!({"penalty":penalty.penalty(),"token_ids":penalty.token_ids()})),
            }),
            Some(LogitsReturnPolicy::FullLogits) => {
                json!({"kind":"host_sampling_not_observed_here"})
            }
            None => json!({"kind":"no_sampling"}),
        };
        Some(CaptureAttempt {
            owner: Arc::clone(self),
            participant: index,
            expected,
            metadata,
            finished: false,
        })
    }
}

impl CaptureAttempt {
    pub(super) fn observe_submission(
        &mut self,
        submission: &SubmittedOperationReceipt,
        observation: &SubmissionObservation,
    ) {
        self.metadata["submission_observation"] = json!({
            "submission":submission,
            "work":match observation.work() {
                Some(work) if work.submission() == submission => json!({"status":"captured","value":work}),
                Some(_) => json!({"status":"unavailable","reason":"submission_mismatch"}),
                None => json!({"status":"unavailable","reason":"not_captured"}),
            },
            "attribution":match observation.attribution() {
                Some(attribution) if attribution.batch_identity() == submission.batch_identity()
                    && attribution.submission_fingerprint() == submission.fingerprint() => json!({"status":"captured","value":attribution}),
                Some(_) => json!({"status":"unavailable","reason":"submission_mismatch"}),
                None => json!({"status":"unavailable","reason":"not_captured"}),
            },
        });
        self.metadata["native_kernel_launch_facts"] = match observation.device_facts() {
            Some(facts) if facts.submission() == submission => json!({
                "status":"observed", "submission":facts.submission(), "coverage":facts.coverage(),
                "device":facts.device(), "projection_rows":facts.projection_rows(),
            }),
            Some(_) => json!({"status":"unavailable","reason":"submission_mismatch"}),
            None => json!({"status":"unavailable","reason":"not_captured"}),
        };
    }

    pub(super) fn observe_prepared_dispatch<R: DeviceRuntime>(
        &mut self,
        dispatch: &PreparedDispatch<R>,
    ) {
        self.metadata["prepared_dispatch_path"] = json!(match dispatch {
            PreparedDispatch::Ordinary { .. } => "ordinary",
            PreparedDispatch::Resident { .. } => "resident",
        });
    }

    pub(super) fn extra_request(
        &mut self,
        plan: &ExecutionPlan,
    ) -> Option<CompletionExtraReadbackRequest> {
        match CompletionExtraReadbackRequest::new(
            plan,
            self.expected.clone(),
            self.owner.config.maximum_readback_bytes,
        ) {
            Ok(request) => Some(request),
            Err(error) => {
                self.fail("unavailable", &error.to_string());
                None
            }
        }
    }

    pub(super) fn fail(&mut self, status: &str, reason: &str) {
        if self.finished {
            return;
        }
        self.metadata["status"] = json!(status);
        self.metadata["reason"] = json!(reason);
        self.owner.record(self.metadata.take());
        self.finished = true;
    }

    pub(super) fn finish(
        &mut self,
        primary: &CompletionReadbackBatchReceipt,
        extra: Option<&CompletionReadbackReceipt>,
        products: &[ExecutorSamplingOutput],
    ) {
        if self.finished {
            return;
        }
        let disposition = extra.map(|receipt| receipt.disposition()).or_else(|| {
            primary.dispositions().iter().find(|disposition| {
                matches!(disposition, CompletionReadbackDisposition::Succeeded(output) if output.request() == &self.expected)
            })
        });
        let Some(CompletionReadbackDisposition::Succeeded(output)) = disposition else {
            self.fail(
                "readback_failed",
                "original retained row was not successfully read",
            );
            return;
        };
        if output.request() != &self.expected
            || output.bytes().len() as u64 > self.owner.config.maximum_readback_bytes
        {
            self.fail(
                "readback_mismatch",
                "original participant output differs from its claim",
            );
            return;
        }
        if let Some(extra) = extra {
            if extra.completion() != primary.completion() {
                self.fail(
                    "readback_mismatch",
                    "extra output names another terminal completion",
                );
                return;
            }
        }
        self.metadata["completion"] = json!(primary.completion());
        self.metadata["primary_readback_fingerprint"] = json!(primary.fingerprint());
        self.metadata["readback_request"] = json!(output.request());
        self.metadata["raw_sha256"] = json!(output.sha256());
        self.metadata["raw_bytes"] = json!(output.bytes().len());
        self.metadata["all_finite"] = json!(output
            .bytes()
            .chunks_exact(4)
            .all(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")).is_finite()));
        self.metadata["selected_token"] = json!(match products.get(self.participant) {
            Some(ExecutorSamplingOutput::GreedyToken(token)) => Some(token.get()),
            _ => None,
        });
        let written = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.owner.config.output_dir.join("logits.f32le.bin"))
            .and_then(|mut file| file.write_all(output.bytes()));
        if let Err(error) = written {
            self.fail("io_failed", &error.to_string());
            return;
        }
        self.metadata["raw_file"] = json!("logits.f32le.bin");
        self.metadata["status"] = json!("captured");
        self.owner.record(self.metadata.take());
        self.finished = true;
    }
}

impl Drop for CaptureAttempt {
    fn drop(&mut self) {
        self.fail(
            "not_captured",
            "original attempt ended or was cancelled before capture completed",
        );
    }
}

/// Preserve every original observation variant for the existing wave guard.
/// The optional receipt is pure host evidence and grants no retirement rights.
pub(super) fn split_extra_observation(
    observation: CompletionReadbackWithExtraObservation,
    extra: &Mutex<Option<CompletionReadbackReceipt>>,
) -> CompletionReadbackBatchObservation {
    match observation {
        CompletionReadbackWithExtraObservation::Terminal(receipt) => {
            let (primary, additional) = receipt.into_parts();
            *extra.lock() = Some(additional);
            CompletionReadbackBatchObservation::Terminal(primary)
        }
        CompletionReadbackWithExtraObservation::Pending => {
            CompletionReadbackBatchObservation::Pending
        }
        CompletionReadbackWithExtraObservation::Indeterminate(failures) => {
            CompletionReadbackBatchObservation::Indeterminate(failures)
        }
        CompletionReadbackWithExtraObservation::SubmissionIndeterminate => {
            CompletionReadbackBatchObservation::SubmissionIndeterminate
        }
        CompletionReadbackWithExtraObservation::ObservationPanicked => {
            CompletionReadbackBatchObservation::ObservationPanicked
        }
        CompletionReadbackWithExtraObservation::Quarantined(receipt) => {
            CompletionReadbackBatchObservation::Quarantined(receipt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(tokens: &[u32]) -> RetainedOutputCaptureConfig {
        RetainedOutputCaptureConfig {
            output_dir: "unused".into(),
            prompt_token_count: 2,
            prompt_token_sha256: token_digest(&tokens[..2]),
            source_span: 3..4,
            consumed_history_sha256: token_digest(tokens),
            maximum_readback_bytes: 1024,
        }
    }

    fn input<'a>(tokens: &'a [u32], span: &'a TokenSpanWork) -> Input<'a> {
        Input {
            span,
            tokens,
            prompt_tokens: 2,
            product: true,
        }
    }

    #[test]
    fn retained_row_selection_requires_actual_span_and_consumed_history() {
        let tokens = [2, 3, 7, 11];
        let spec = config(&tokens);
        let span = TokenSpanWork::from_token_ids(&tokens, 3..4).unwrap();
        let wrong_span = TokenSpanWork::from_token_ids(&tokens, 2..3).unwrap();
        assert_eq!(
            select(&spec, [input(&tokens, &wrong_span)].into_iter()),
            Selection::None
        );
        assert_eq!(
            select(&spec, [input(&tokens, &span)].into_iter()),
            Selection::Unique {
                index: 0,
                history_matches: true
            }
        );
        let changed = [2, 3, 7, 12];
        let changed_span = TokenSpanWork::from_token_ids(&changed, 3..4).unwrap();
        assert_eq!(
            select(
                &spec,
                [Input {
                    span: &changed_span,
                    tokens: &changed,
                    prompt_tokens: 2,
                    product: true
                }]
                .into_iter()
            ),
            Selection::Unique {
                index: 0,
                history_matches: false
            }
        );
        // Two original participants remain ambiguous even if one has changed history.
        assert_eq!(
            select(
                &spec,
                [
                    input(&tokens, &span),
                    Input {
                        span: &changed_span,
                        tokens: &changed,
                        prompt_tokens: 2,
                        product: true
                    }
                ]
                .into_iter()
            ),
            Selection::Ambiguous
        );
    }

    #[test]
    fn retained_row_selection_ignores_startup_and_a_different_prompt() {
        let tokens = [2, 3, 7, 11];
        let spec = config(&tokens);
        let span = TokenSpanWork::from_token_ids(&tokens, 3..4).unwrap();
        assert_eq!(
            select(
                &spec,
                [Input {
                    span: &span,
                    tokens: &tokens,
                    prompt_tokens: 2,
                    product: false
                }]
                .into_iter()
            ),
            Selection::None
        );
        let other = [2, 4, 7, 11];
        let other_span = TokenSpanWork::from_token_ids(&other, 3..4).unwrap();
        assert_eq!(
            select(
                &spec,
                [Input {
                    span: &other_span,
                    tokens: &other,
                    prompt_tokens: 2,
                    product: true
                }]
                .into_iter()
            ),
            Selection::None
        );
    }

    #[test]
    fn retained_row_changed_history_cannot_rearm_for_a_replacement() {
        let directory = tempfile::tempdir().unwrap();
        let tokens = [2, 3, 7, 11];
        let mut spec = config(&tokens);
        spec.output_dir = directory.path().to_owned();
        let capture = RetainedOutputCapture::new(spec).unwrap();
        let changed = [2, 3, 7, 12];
        let old_span = TokenSpanWork::from_token_ids(&changed, 3..4).unwrap();
        assert_eq!(
            capture.select_once([input(&changed, &old_span)].into_iter()),
            Some(Selection::Unique {
                index: 0,
                history_matches: false
            })
        );
        let replacement_span = TokenSpanWork::from_token_ids(&tokens, 3..4).unwrap();
        assert_eq!(
            capture.select_once([input(&tokens, &replacement_span)].into_iter()),
            None
        );
    }

    #[test]
    fn retained_row_storage_failure_preserves_existing_files_and_product_startup() {
        let directory = tempfile::tempdir().unwrap();
        let existing = directory.path().join("report.json");
        fs::write(&existing, b"owned by another run").unwrap();
        let tokens = [2, 3, 7, 11];
        let mut spec = config(&tokens);
        spec.output_dir = directory.path().to_owned();
        let capture = RetainedOutputCapture::new(spec).unwrap();
        let span = TokenSpanWork::from_token_ids(&tokens, 3..4).unwrap();
        assert_eq!(
            capture.select_once([input(&tokens, &span)].into_iter()),
            None
        );
        assert_eq!(fs::read(existing).unwrap(), b"owned by another run");
    }
}
