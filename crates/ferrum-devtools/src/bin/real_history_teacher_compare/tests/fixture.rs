//! Small typed capture fixtures. They model evidence contracts, not inference.
use super::super::*;
use ferrum_interfaces::vnext::*;

pub(super) const BASE_LOGITS: [f32; 4] = [0.0, 2.0, 1.0, -1.0];

fn source_identity(location: &str) -> ProductModelSourceIdentity {
    let original = OriginalModelSource {
        kind: ModelSourceKind::LocalDirectory,
        location: location.into(),
        requested_revision: None,
    };
    let resolved = |file: &str| ResolvedModelSource {
        canonical_location: location.into(),
        resolved_revision: "fixture-content".into(),
        files: vec![FileFingerprint {
            relative_path: file.into(),
            size_bytes: 1,
            sha256: sha256(file.as_bytes()),
        }],
    };
    let binding = |role, file: &str, content| {
        ProductModelArtifactBinding::new(role, file, sha256(file.as_bytes()), content).unwrap()
    };
    ProductModelSourceIdentity::new(
        "fixture-model",
        "fixture-model",
        OriginalModelSources {
            semantic: original.clone(),
            tokenizer: original.clone(),
            weights: original,
        },
        ResolvedModelSources {
            semantic: resolved("config.json"),
            tokenizer: resolved("tokenizer.json"),
            weights: resolved("weights.gguf"),
        },
        binding(ModelArtifactSourceRole::Semantic, "config.json", None),
        binding(ModelArtifactSourceRole::Tokenizer, "tokenizer.json", None),
        binding(
            ModelArtifactSourceRole::Tokenizer,
            "tokenizer.json",
            Some(sha256(b"template")),
        ),
        None,
    )
    .unwrap()
}

fn identity(owner: usize, wave: usize, node: usize) -> UnvalidatedExecutionIdentityParts {
    UnvalidatedExecutionIdentityParts {
        version: EXECUTION_IDENTITY_VERSION,
        run_id: RunId::new("run.fixture").unwrap(),
        request_id: RequestIdentity::new(format!("request.diagnostic.request-{owner}")).unwrap(),
        sequence: (wave * 16 + node * 4 + owner + 1) as u64,
        plan_id: Some(PlanId::new("plan.fixture").unwrap()),
        plan_hash: Some(serde_json::from_value(json!(sha256(b"plan"))).unwrap()),
        frame_id: Some(ExecutionFrameId::try_from((wave + 1) as u64).unwrap()),
        node_invocation_id: Some(NodeInvocationId::try_from((wave * 2 + node + 1) as u64).unwrap()),
        node_id: Some(NodeId::new(format!("node.{node}")).unwrap()),
        operation_id: Some(OperationId::new(format!("operation.{node}")).unwrap()),
        provider_id: Some(ProviderId::new("provider.fixture").unwrap()),
        device_id: Some(DeviceId::new("device.fixture").unwrap()),
        resource_pool_id: None,
        resource_pool_identity_fingerprint: None,
        provisioning_run_id: None,
        provisioning_request_id: None,
        transaction_id: None,
        active_sequence_slot: Some(owner as u32),
        admission_generation: Some(1),
        activation_epoch: Some(1),
        runtime_implementation_fingerprint: Some(sha256(b"runtime")),
        active_sequence_fingerprint: Some(sha256(format!("owner-{owner}").as_bytes())),
        completed_sequence_fingerprint: None,
        aborted_sequence_fingerprint: None,
        resource_id: None,
        resource_generation: None,
        resource_batch_fingerprint: None,
        span_id: SpanId::new(format!("span.{wave}.{node}.{owner}")).unwrap(),
        parent_span_id: None,
        async_links: vec![],
    }
}

fn completion(owners: &[usize], wave: usize) -> receipt::Completion {
    let mut nodes = Vec::new();
    for node_index in 0..2 {
        let participants = owners
            .iter()
            .enumerate()
            .map(|(logical, &owner)| {
                let identity = identity(owner, wave, node_index);
                ExecutionIdentityEnvelope::new(identity.clone().into()).unwrap();
                receipt::Participant {
                    participant_index: node_index * owners.len() + logical,
                    node_key: receipt::NodeKey {
                        sequence_authority: receipt::Authority {
                            sparse_id: owner as u32,
                            generation: 1,
                        },
                        request_authority: receipt::Authority {
                            sparse_id: owner as u32,
                            generation: 1,
                        },
                        frame_id: serde_json::to_value(identity.frame_id).unwrap(),
                        node_id: format!("node.{node_index}"),
                    },
                    identity,
                }
            })
            .collect();
        let mut node = receipt::Node {
            node_index,
            node_id: format!("node.{node_index}"),
            operation_id: format!("operation.{node_index}"),
            provider_id: "provider.fixture".into(),
            provider_implementation_fingerprint: sha256(b"provider"),
            provider_execution_semantics: ProviderExecutionSemantics::bitwise_eager_only(),
            work_shape_fingerprint: sha256(format!("shape.{wave}.{node_index}").as_bytes()),
            participants,
            fingerprint: String::new(),
        };
        node.fingerprint = receipt::node_fingerprint(&node).unwrap();
        nodes.push(node);
    }
    let participants: Vec<_> = nodes
        .iter()
        .flat_map(|node| node.participants.clone())
        .collect();
    let batch_identity = receipt::Batch {
        // Other runtime work may consume IDs between teacher waves.
        batch_step_id: json!(wave * 3 + 7),
        batch_invocation_id: json!(wave * 5 + 9),
        plan_id: "plan.fixture".into(),
        plan_hash: sha256(b"plan"),
        device_id: "device.fixture".into(),
        runtime_implementation_fingerprint: sha256(b"runtime"),
        lane_id: json!(1),
        claimed_backing_fingerprint: sha256(b"backing"),
        nodes,
        participants,
        fingerprint: sha256(format!("compiled-root.{wave}.{owners:?}").as_bytes()),
    };
    let mut submission = receipt::Submission {
        slot_id: json!(wave + 1),
        batch_identity,
        participants: vec![],
        fingerprint: String::new(),
    };
    submission.fingerprint = receipt::submission_fingerprint(&submission).unwrap();
    submission.participants = submission
        .batch_identity
        .participants
        .iter()
        .map(|part| receipt::SubmissionParticipant {
            slot_id: submission.slot_id.clone(),
            participant_index: part.participant_index,
            identity: part.identity.clone(),
            batch_submission_fingerprint: submission.fingerprint.clone(),
        })
        .collect();
    let mut completion = receipt::Completion {
        submission,
        disposition: json!({"status":"succeeded"}),
        fence_timing: json!({"timing_mode":"off","device_execution":{"status":"not_requested"},"blocking_wait_host_ns":{"status":"not_requested"}}),
        submission_timing: json!({"status":"not_requested"}),
        participants: vec![],
        fingerprint: String::new(),
    };
    completion.fingerprint = receipt::completion_fingerprint(&completion).unwrap();
    completion.participants = completion
        .submission
        .participants
        .iter()
        .cloned()
        .map(|submission| receipt::CompletionParticipant {
            submission,
            disposition: json!({"status":"succeeded"}),
            batch_completion_fingerprint: completion.fingerprint.clone(),
        })
        .collect();
    completion
}

fn raw_file(directory: &Path, file: String, bytes: &[u8]) -> VNextTeacherRawArtifact {
    fs::write(directory.join(&file), bytes).unwrap();
    VNextTeacherRawArtifact {
        file,
        bytes: bytes.len() as u64,
        sha256: sha256(bytes),
    }
}

fn external_file(directory: &Path, name: &str, bytes: &[u8]) -> VNextTeacherFileIdentity {
    let path = directory.join(name);
    fs::write(&path, bytes).unwrap();
    VNextTeacherFileIdentity {
        path: path.to_string_lossy().into_owned(),
        bytes: bytes.len() as u64,
        sha256: sha256(bytes),
    }
}

pub(super) struct Arm {
    pub directory: PathBuf,
    pub manifest: VNextTeacherCaptureManifest,
}

impl Arm {
    pub fn set_raw_dtype(&mut self, dtype: &str) {
        let bits: &[u16] = match dtype {
            "f16" => &[0x0000, 0x4000, 0x3c00, 0xbc00],
            "bf16" => &[0x0000, 0x4000, 0x3f80, 0xbf80],
            _ => panic!("fixture conversion expects f16 or bf16"),
        };
        let bytes: Vec<_> = bits.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        for wave in &mut self.manifest.waves {
            for raw in &mut wave.readbacks {
                let file = raw_file(
                    &self.directory,
                    raw.raw_artifact.as_ref().unwrap().file.clone(),
                    &bytes,
                );
                raw.byte_count = bytes.len();
                raw.sha256 = file.sha256.clone();
                raw.raw_artifact = Some(file);
                raw.request["output_layout"]["element_type"] = json!(dtype);
            }
            wave.receipt_fingerprint =
                receipt::readback_fingerprint(&wave.completion_fingerprint, &wave.readbacks)
                    .unwrap();
        }
        self.persist();
    }
    pub fn persist(&self) {
        fs::write(
            self.directory.join("manifest.json"),
            serde_json::to_vec_pretty(&self.manifest).unwrap(),
        )
        .unwrap();
    }

    pub fn edit_completion(
        &mut self,
        wave_index: usize,
        edit: impl FnOnce(&mut receipt::Completion),
    ) {
        let wave = &mut self.manifest.waves[wave_index];
        let file = wave.completion_receipt.as_ref().unwrap();
        let mut completion: receipt::Completion =
            serde_json::from_slice(&fs::read(self.directory.join(&file.file)).unwrap()).unwrap();
        edit(&mut completion);
        wave.completion_receipt = Some(raw_file(
            &self.directory,
            file.file.clone(),
            &serde_json::to_vec(&completion).unwrap(),
        ));
        self.persist();
    }

    /// A raw-consistent candidate numerical difference, not post-readback editing.
    pub fn set_logits(&mut self, owner: &str, decision: usize, logits: &[f32]) {
        let row = self
            .manifest
            .decisions
            .iter_mut()
            .find(|row| row.evidence.owner_id == owner && row.evidence.decision_index == decision)
            .unwrap();
        let bytes: Vec<_> = logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let file = raw_file(&self.directory, row.logits.file.clone(), &bytes);
        row.logits.bytes = file.bytes;
        row.logits.sha256 = file.sha256;
        let wave = &mut self.manifest.waves[row.evidence.wave_index];
        let raw = wave
            .readbacks
            .iter_mut()
            .find(|raw| raw.participant_index == row.evidence.participant_index)
            .unwrap();
        let file = raw_file(
            &self.directory,
            raw.raw_artifact.as_ref().unwrap().file.clone(),
            &bytes,
        );
        raw.sha256 = file.sha256.clone();
        raw.byte_count = bytes.len();
        raw.raw_artifact = Some(file);
        wave.receipt_fingerprint =
            receipt::readback_fingerprint(&wave.completion_fingerprint, &wave.readbacks).unwrap();
        self.persist();
    }

    fn wave(&mut self, owner_indices: &[usize], decision: usize, prompt_end: Option<usize>) {
        let index = self.manifest.waves.len();
        let completed = completion(owner_indices, index);
        let completion_receipt = raw_file(
            &self.directory,
            format!("completion-{index}.json"),
            &serde_json::to_vec(&completed).unwrap(),
        );
        let mut wave = VNextTeacherWaveEvidence {
            wave_index: index,
            kind: if decision == 0 { "prefill" } else { "decode" }.into(),
            participant_count: owner_indices.len(),
            completion_fingerprint: completed.fingerprint,
            receipt_fingerprint: String::new(),
            completion_receipt: Some(completion_receipt),
            readbacks: vec![],
            participants: vec![],
        };
        for (logical, &owner_index) in owner_indices.iter().enumerate() {
            let owner = &self.manifest.owners[owner_index];
            let mut history = owner.prompt_token_ids.clone();
            history.extend_from_slice(&owner.teacher_token_ids[..decision]);
            if let Some(end) = prompt_end {
                history.truncate(end);
            }
            let start = self
                .manifest
                .waves
                .iter()
                .rev()
                .flat_map(|wave| &wave.participants)
                .find(|part| part.owner_id == owner.owner_id)
                .map_or(0, |part| part.immediate_end);
            let part = VNextTeacherWaveParticipant {
                owner_id: owner.owner_id.clone(),
                request_id: format!("request-{owner_index}"),
                participant_index: logical,
                cache_id: format!("cache-{owner_index}"),
                history_tokens: history.len(),
                history_sha256: token_digest(&history),
                immediate_start: start,
                immediate_end: history.len(),
            };
            let bytes: Vec<_> = BASE_LOGITS
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            let raw = raw_file(
                &self.directory,
                format!("readback-{index}-{logical}.bin"),
                &bytes,
            );
            wave.readbacks.push(VNextTeacherReadbackEvidence {
                participant_index: logical,
                request: serde_json::to_value(receipt::ReadbackRequest {
                    node_id: "node.1".into(),
                    participant_index: logical as u32,
                    resource_id: "resource.logits".into(),
                    expected_usage: "activations".into(),
                    logical_offset_bytes: 0,
                    output_layout: receipt::Layout {
                        element_type: "f32".into(),
                        element_count: 4,
                    },
                })
                .unwrap(),
                byte_count: bytes.len(),
                sha256: raw.sha256.clone(),
                raw_artifact: Some(raw),
            });
            if prompt_end.is_none_or(|end| end == owner.prompt_token_ids.len()) {
                let file = raw_file(
                    &self.directory,
                    format!("logits-{owner_index}-{decision}.f32le"),
                    &bytes,
                );
                self.manifest.decisions.push(VNextTeacherDecisionRecord {
                    evidence: VNextTeacherDecisionEvidence {
                        owner_id: part.owner_id.clone(),
                        decision_index: decision,
                        teacher_token_id: owner.teacher_token_ids[decision],
                        history_tokens: part.history_tokens,
                        history_sha256: part.history_sha256.clone(),
                        wave_index: index,
                        participant_index: logical,
                        request_id: part.request_id.clone(),
                        cache_id: part.cache_id.clone(),
                    },
                    logits: VNextTeacherLogitArtifact {
                        file: file.file,
                        encoding: "f32-le".into(),
                        elements: 4,
                        bytes: file.bytes,
                        sha256: file.sha256,
                    },
                });
            }
            wave.participants.push(part);
        }
        wave.receipt_fingerprint =
            receipt::readback_fingerprint(&wave.completion_fingerprint, &wave.readbacks).unwrap();
        self.manifest.waves.push(wave);
    }
}

pub(super) struct Pair {
    pub root: tempfile::TempDir,
    pub reference: Arm,
    pub candidate: Arm,
}

impl Pair {
    pub fn new() -> Self {
        Self::with_width(4)
    }

    pub fn with_width(width: usize) -> Self {
        Self::with_modes(width, "serial", "batched")
    }

    pub fn with_batched_width(width: usize) -> Self {
        Self::with_modes(width, "batched", "batched")
    }

    pub fn with_serial_width(width: usize) -> Self {
        Self::with_modes(width, "serial", "serial")
    }

    fn with_modes(width: usize, reference_mode: &str, candidate_mode: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        let owners: Vec<_> = (0..width)
            .map(|owner| {
                let prompt_token_ids = vec![owner as u32 % 4, 1, 2];
                let teacher_token_ids = vec![3, 2, 3];
                VNextTeacherOwnerRecord {
                    owner_id: format!("owner-{owner}"),
                    prompt_token_ids_sha256: token_digest(&prompt_token_ids),
                    teacher_token_ids_sha256: token_digest(&teacher_token_ids),
                    prompt_token_ids,
                    teacher_token_ids,
                }
            })
            .collect();
        let history = json!({"schema_version":1,"owners":owners.iter().map(|owner|json!({"owner_id":owner.owner_id,"prompt_token_ids":owner.prompt_token_ids,"teacher_token_ids":owner.teacher_token_ids})).collect::<Vec<_>>()});
        let history_file = external_file(
            root.path(),
            "history.json",
            &serde_json::to_vec(&history).unwrap(),
        );
        let binary = external_file(
            root.path(),
            "fixture-binary",
            b"typed fixture binary provenance; not executable evidence",
        );
        let make_arm = |mode: &str, label: &str| {
            let directory = root.path().join(label);
            fs::create_dir(&directory).unwrap();
            let mut arm = Arm {
                directory,
                manifest: VNextTeacherCaptureManifest {
                    schema_version: 1,
                    artifact_type: REAL_HISTORY_TEACHER_CAPTURE_TYPE.into(),
                    mode: mode.into(),
                    output_policy: "unmodified_full_logits_before_sampling".into(),
                    identity: Some(VNextTeacherCaptureIdentity {
                        model_id: "fixture-model".into(),
                        model_source: serde_json::to_value(source_identity(&format!(
                            "/fixtures/{mode}/model"
                        )))
                        .unwrap(),
                        numerical_profile: "fixture.exact".into(),
                        kv_storage: "fp16".into(),
                        family_fingerprint: sha256(b"family"),
                        program_fingerprint: sha256(b"program"),
                        resolved_plan_fingerprint: sha256(b"resolved"),
                        binary: binary.clone(),
                        history_file: history_file.clone(),
                    }),
                    vocabulary_size: 4,
                    configuration: json!({"sequence_slots":width,"max_model_len":16,"prefill_chunk_tokens":2,
                        "backend":{"backend_options":{"model_path":format!("/fixtures/{mode}/model"),"numerical_route":"fixture.exact"}}}),
                    owners: owners.clone(),
                    waves: vec![],
                    decisions: vec![],
                    complete: true,
                    errors: vec![],
                },
            };
            for owner in 0..width {
                arm.wave(&[owner], 0, Some(1));
                arm.wave(&[owner], 0, Some(3));
            }
            for decision in 1..3 {
                if mode == "serial" {
                    for owner in 0..width {
                        arm.wave(&[owner], decision, None);
                    }
                } else {
                    arm.wave(&(0..width).collect::<Vec<_>>(), decision, None);
                }
            }
            arm.persist();
            arm
        };
        let reference = make_arm(reference_mode, "reference");
        let candidate = make_arm(candidate_mode, "candidate");
        Self {
            root,
            reference,
            candidate,
        }
    }

    pub fn args(&self) -> Args {
        Args {
            reference_dir: self.reference.directory.clone(),
            candidate_dir: self.candidate.directory.clone(),
            reference_binary: None,
            candidate_binary: None,
            history_file: None,
            implementation_transition: None,
            numerical_profile_transition: None,
            comparison_shape: ComparisonShape::SerialToBatched,
            require_bitwise_logits: false,
            native_work_audit: None,
            output_file: self.root.path().join("report.json"),
            budgets: Budgets {
                mean_delta_nll_limit: 0.01,
                max_delta_nll_limit: 0.1,
                mean_kl_limit: 0.001,
                max_kl_limit: 0.01,
            },
        }
    }
    pub fn compare(&self) -> (Value, i32) {
        compare(&self.args())
    }
}
