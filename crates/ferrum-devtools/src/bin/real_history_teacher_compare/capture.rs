use super::*;
use ferrum_interfaces::vnext::{ModelArtifactSourceRole, ProductModelSourceIdentity};

pub(super) struct CheckedCapture {
    pub directory: PathBuf,
    pub manifest: VNextTeacherCaptureManifest,
    pub manifest_sha256: String,
    pub decisions: BTreeMap<(String, usize), VNextTeacherDecisionRecord>,
    pub receipts: BTreeMap<usize, receipt::CheckedReceipt>,
    pub external_files: Value,
}

fn check_owner(owner: &VNextTeacherOwnerRecord, vocabulary: usize) -> Result<()> {
    ensure!(
        !owner.owner_id.trim().is_empty()
            && !owner.prompt_token_ids.is_empty()
            && owner.teacher_token_ids.len() >= 2,
        "owner has no real prompt or decode history"
    );
    ensure!(
        owner.prompt_token_ids_sha256 == token_digest(&owner.prompt_token_ids)
            && owner.teacher_token_ids_sha256 == token_digest(&owner.teacher_token_ids),
        "owner token SHA-256 differs"
    );
    ensure!(
        owner
            .prompt_token_ids
            .iter()
            .chain(&owner.teacher_token_ids)
            .all(|token| (*token as usize) < vocabulary),
        "owner token is outside vocabulary"
    );
    Ok(())
}

pub(super) fn source_content(value: &Value) -> Result<Value> {
    let source: ProductModelSourceIdentity =
        serde_json::from_value(value.clone()).context("parse product source identity")?;
    source.validate().map_err(anyhow::Error::msg)?;
    let mut roles = BTreeMap::new();
    for role in ModelArtifactSourceRole::ALL {
        let resolved = source.resolved_sources.for_role(role);
        ensure!(
            !resolved.files.is_empty(),
            "source role {} has no content inventory",
            role.as_str()
        );
        let mut files: Vec<_> = resolved
            .files
            .iter()
            .map(|file| (file.sha256.clone(), file.size_bytes))
            .collect();
        files.sort();
        roles.insert(role.as_str(), files);
    }
    let binding = |binding: &ferrum_interfaces::vnext::ProductModelArtifactBinding| json!({"role":binding.role,"container_sha256":binding.container_sha256,"content_sha256":binding.content_sha256});
    Ok(
        json!({"roles":roles,"semantic_config":binding(&source.semantic_config),"tokenizer":binding(&source.tokenizer),
        "template":binding(&source.template),"weight_config":source.weight_config.as_ref().map(binding)}),
    )
}

fn provenance(
    manifest: &VNextTeacherCaptureManifest,
    binary: Option<&Path>,
    history: Option<&Path>,
) -> Result<Value> {
    let identity = manifest
        .identity
        .as_ref()
        .context("capture lacks model/runtime identity")?;
    ensure!(
        !identity.numerical_profile.is_empty()
            && identity.numerical_profile != "auto"
            && identity.kv_storage == "fp16",
        "capture lacks exact profile/F16 KV policy"
    );
    for digest in [
        &identity.family_fingerprint,
        &identity.program_fingerprint,
        &identity.resolved_plan_fingerprint,
    ] {
        ensure!(
            files::canonical_sha(digest),
            "capture lacks a complete model/program/resolved-plan fingerprint"
        );
    }
    source_content(&identity.model_source)?;
    let verified_binary = files::external(&identity.binary, binary)?;
    let verified_history = files::external(&identity.history_file, history)?;
    #[derive(Deserialize)]
    struct HistoryOwner {
        owner_id: String,
        prompt_token_ids: Vec<u32>,
        teacher_token_ids: Vec<u32>,
    }
    #[derive(Deserialize)]
    struct History {
        schema_version: u32,
        owners: Vec<HistoryOwner>,
    }
    let path = history.unwrap_or_else(|| Path::new(&identity.history_file.path));
    let source: History = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        source.schema_version == 1 && source.owners.len() == manifest.owners.len(),
        "external teacher history has another schema or owner inventory"
    );
    let mut seen = BTreeSet::new();
    for owner in source.owners {
        ensure!(
            seen.insert(owner.owner_id.clone()),
            "external teacher history repeats owner"
        );
        let recorded = manifest
            .owners
            .iter()
            .find(|candidate| candidate.owner_id == owner.owner_id)
            .context("external teacher history has another owner")?;
        ensure!(
            owner.prompt_token_ids == recorded.prompt_token_ids
                && owner.teacher_token_ids == recorded.teacher_token_ids,
            "external teacher history differs from captured canonical tokens"
        );
    }
    Ok(
        json!({"binary":verified_binary,"history":verified_history,"model_source_file_inventory":"validated_capture_identity_not_reopened_model_files"}),
    )
}

#[derive(Default)]
struct OwnerProgress {
    committed: usize,
    decode_steps: usize,
    request: Option<String>,
    cache: Option<String>,
    authority: Option<(String, receipt::Authority, receipt::Authority)>,
}

fn wave_history(
    manifest: &VNextTeacherCaptureManifest,
    wave: &VNextTeacherWaveEvidence,
    progress: &mut BTreeMap<String, OwnerProgress>,
    checked: &receipt::CheckedReceipt,
) -> Result<()> {
    ensure!(
        wave.participant_count > 0 && wave.participant_count == wave.participants.len(),
        "wave logical participant inventory differs"
    );
    let expected_width = match wave.kind.as_str() {
        "prefill" => 1,
        "decode" if manifest.mode == "serial" => 1,
        "decode" => manifest.owners.len(),
        _ => anyhow::bail!("unsupported teacher wave kind"),
    };
    ensure!(
        wave.participant_count == expected_width,
        "actual wave width differs from declared serial/batched execution"
    );
    if wave.kind == "decode" {
        ensure!(
            manifest.owners.iter().all(|owner| progress
                .get(&owner.owner_id)
                .is_some_and(|state| state.committed >= owner.prompt_token_ids.len())),
            "decode started before all real prompt states were committed"
        );
    }
    let mut owners = BTreeSet::new();
    let mut indices = BTreeSet::new();
    for part in &wave.participants {
        ensure!(
            owners.insert(part.owner_id.clone())
                && indices.insert(part.participant_index)
                && part.participant_index < wave.participant_count,
            "wave repeats/misindexes owner"
        );
        let owner = manifest
            .owners
            .iter()
            .find(|owner| owner.owner_id == part.owner_id)
            .context("wave contains undeclared owner")?;
        let authority = checked
            .owners
            .get(&owner.owner_id)
            .context("actual receipt omits logical owner")?;
        for (other_id, other) in progress.iter().filter(|(id, _)| *id != &owner.owner_id) {
            ensure!(
                other.request.as_ref() != Some(&part.request_id)
                    && other.cache.as_ref() != Some(&part.cache_id)
                    && other
                        .authority
                        .as_ref()
                        .is_none_or(|prior| prior.0 != authority.0
                            && prior.1 != authority.1
                            && prior.2 != authority.2),
                "owners {} and {} reuse request/cache/physical sequence authority",
                other_id,
                owner.owner_id
            );
        }
        let state = progress.entry(owner.owner_id.clone()).or_default();
        ensure!(
            !part.request_id.is_empty() && !part.cache_id.is_empty(),
            "wave lacks actual request/cache identity"
        );
        if let Some(request) = &state.request {
            ensure!(
                request == &part.request_id,
                "owner request identity changed"
            );
        }
        if let Some(cache) = &state.cache {
            ensure!(cache == &part.cache_id, "owner cache identity changed");
        }
        if let Some(prior) = &state.authority {
            ensure!(
                prior == authority,
                "owner's actual physical authority changed"
            );
        }
        ensure!(
            part.immediate_start == state.committed
                && part.immediate_end == part.history_tokens
                && part.immediate_end > part.immediate_start,
            "wave token frontier skipped/replayed work"
        );
        let mut expected = owner.prompt_token_ids.clone();
        if wave.kind == "prefill" {
            ensure!(
                state.decode_steps == 0
                    && state.committed < owner.prompt_token_ids.len()
                    && part.immediate_end <= owner.prompt_token_ids.len(),
                "prefill appeared after final prompt/decode"
            );
            expected.truncate(part.immediate_end);
        } else {
            state.decode_steps += 1;
            ensure!(
                state.decode_steps < owner.teacher_token_ids.len()
                    && part.immediate_end == state.committed + 1,
                "decode is not one canonical next-token step"
            );
            expected.extend_from_slice(&owner.teacher_token_ids[..state.decode_steps]);
        }
        ensure!(
            part.history_tokens == expected.len() && part.history_sha256 == token_digest(&expected),
            "actual wave history differs from canonical owner prefix"
        );
        state.committed = part.immediate_end;
        state.request = Some(part.request_id.clone());
        state.cache = Some(part.cache_id.clone());
        state.authority = Some(authority.clone());
    }
    Ok(())
}

fn decision(
    directory: &Path,
    manifest: &VNextTeacherCaptureManifest,
    record: &VNextTeacherDecisionRecord,
    valid_waves: &BTreeSet<usize>,
) -> Result<()> {
    let row = &record.evidence;
    let owner = manifest
        .owners
        .iter()
        .find(|owner| owner.owner_id == row.owner_id)
        .context("decision has undeclared owner")?;
    ensure!(
        row.decision_index < owner.teacher_token_ids.len(),
        "decision exceeds teacher history"
    );
    let mut history = owner.prompt_token_ids.clone();
    history.extend_from_slice(&owner.teacher_token_ids[..row.decision_index]);
    ensure!(
        row.teacher_token_id == owner.teacher_token_ids[row.decision_index]
            && row.history_tokens == history.len()
            && row.history_sha256 == token_digest(&history),
        "decision target/history binding differs"
    );
    ensure!(
        valid_waves.contains(&row.wave_index),
        "decision has no valid actual wave receipt"
    );
    let wave = manifest
        .waves
        .get(row.wave_index)
        .context("decision wave missing")?;
    ensure!(
        wave.kind
            == if row.decision_index == 0 {
                "prefill"
            } else {
                "decode"
            },
        "decision phase differs from actual wave"
    );
    let part = wave
        .participants
        .iter()
        .find(|part| part.participant_index == row.participant_index)
        .context("decision physical participant missing")?;
    ensure!(
        part.owner_id == row.owner_id
            && part.request_id == row.request_id
            && part.cache_id == row.cache_id
            && part.history_tokens == row.history_tokens
            && part.history_sha256 == row.history_sha256,
        "decision differs from actual owner/cache/history"
    );
    let readback = wave
        .readbacks
        .iter()
        .find(|readback| readback.participant_index == row.participant_index)
        .context("decision raw readback missing")?;
    ensure!(
        record.logits.elements == manifest.vocabulary_size,
        "decision drops vocabulary values"
    );
    let actual = files::logits(directory, &record.logits)?;
    let raw = files::raw(directory, readback, manifest.vocabulary_size)?;
    ensure!(actual.iter().zip(&raw).all(|(a,b)|a.to_bits()==b.to_bits()),"saved f32 logits differ from independent raw dtype conversion; possible sampling/masking/modification");
    Ok(())
}

pub(super) fn validate(
    directory: &Path,
    arm: &str,
    binary: Option<&Path>,
    history: Option<&Path>,
    errors: &mut Vec<Issue>,
) -> Option<CheckedCapture> {
    let loaded = (|| -> Result<_> {
        let bytes = fs::read(directory.join("manifest.json"))?;
        let manifest: VNextTeacherCaptureManifest = serde_json::from_slice(&bytes)?;
        Ok((manifest, sha256(&bytes)))
    })();
    let (manifest, manifest_sha256) = match loaded {
        Ok(value) => value,
        Err(error) => {
            issue(errors, format!("{arm}/manifest"), error);
            return None;
        }
    };
    let basic = (|| -> Result<()> {
        ensure!(
            manifest.schema_version == REAL_HISTORY_TEACHER_CAPTURE_SCHEMA
                && manifest.artifact_type == REAL_HISTORY_TEACHER_CAPTURE_TYPE,
            "unsupported teacher capture schema/type"
        );
        ensure!(
            manifest.complete && manifest.errors.is_empty(),
            "capture incomplete or recorded failures: {:?}",
            manifest.errors
        );
        ensure!(
            manifest.output_policy == "unmodified_full_logits_before_sampling"
                && manifest.vocabulary_size > 0,
            "capture lacks raw full-logits policy/vocabulary"
        );
        ensure!(
            matches!(manifest.mode.as_str(), "serial" | "batched") && !manifest.owners.is_empty(),
            "capture has no declared execution mode/owners"
        );
        ensure!(
            manifest.configuration.is_object(),
            "capture has no actual fixed configuration"
        );
        Ok(())
    })();
    if let Err(error) = basic {
        issue(errors, format!("{arm}/contract"), error);
    }
    let external_files = match provenance(&manifest, binary, history) {
        Ok(value) => value,
        Err(error) => {
            issue(errors, format!("{arm}/provenance"), error);
            Value::Null
        }
    };
    let mut owner_ids = BTreeSet::new();
    for owner in &manifest.owners {
        if !owner_ids.insert(owner.owner_id.clone()) {
            issue(
                errors,
                format!("{arm}/owner"),
                format!("duplicate owner {}", owner.owner_id),
            );
        }
        if let Err(error) = check_owner(owner, manifest.vocabulary_size) {
            issue(errors, format!("{arm}/owner/{}", owner.owner_id), error);
        }
    }
    if manifest.owners.first().is_some_and(|first| {
        manifest
            .owners
            .iter()
            .any(|owner| owner.teacher_token_ids.len() != first.teacher_token_ids.len())
    }) {
        issue(
            errors,
            format!("{arm}/owners"),
            "canonical continuations have different lengths",
        );
    }
    let mut receipts = BTreeMap::new();
    let mut progress = BTreeMap::new();
    let mut valid_waves = BTreeSet::new();
    let mut completed_operations = BTreeSet::new();
    for (index, wave) in manifest.waves.iter().enumerate() {
        let checked = (|| -> Result<_> {
            ensure!(
                wave.wave_index == index,
                "physical wave index is missing/reordered"
            );
            let checked = receipt::validate(directory, wave)?;
            ensure!(
                completed_operations.insert(wave.completion_fingerprint.clone()),
                "actual physical completion was reused for another teacher wave"
            );
            if let Some(previous) = receipts.values().next_back() {
                let previous: &receipt::CheckedReceipt = previous;
                // The dedicated teacher collector awaits each real wave. The
                // runtime allocates both IDs monotonically; unrelated work may
                // introduce gaps, so contiguity is deliberately not required.
                ensure!(
                    checked.batch_step_id > previous.batch_step_id
                        && checked.batch_invocation_id > previous.batch_invocation_id,
                    "actual physical teacher wave order was replayed or reversed"
                );
            }
            if let Some(first) = receipts.values().next() {
                let first: &receipt::CheckedReceipt = first;
                ensure!(
                    checked.plan_id == first.plan_id
                        && checked.plan_hash == first.plan_hash
                        && checked.runtime == first.runtime
                        && checked.run_id == first.run_id
                        && checked.node_signature == first.node_signature,
                    "physical plan/runtime/provider set changed between waves"
                );
                ensure!(
                    checked.output_binding == first.output_binding,
                    "product output binding changed between physical waves"
                );
            }
            wave_history(&manifest, wave, &mut progress, &checked)?;
            for readback in &wave.readbacks {
                files::raw(directory, readback, manifest.vocabulary_size)?;
            }
            Ok(checked)
        })();
        match checked {
            Ok(checked) => {
                receipts.insert(index, checked);
                valid_waves.insert(index);
            }
            Err(error) => issue(errors, format!("{arm}/wave/{index}"), error),
        }
    }
    for owner in &manifest.owners {
        if !progress.get(&owner.owner_id).is_some_and(|state| {
            state.decode_steps + 1 == owner.teacher_token_ids.len()
                && state.committed
                    == owner.prompt_token_ids.len() + owner.teacher_token_ids.len() - 1
        }) {
            issue(
                errors,
                format!("{arm}/owner/{}/completion", owner.owner_id),
                "actual physical waves do not cover the complete teacher history",
            );
        }
    }
    let mut decisions = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for record in &manifest.decisions {
        let key = (
            record.evidence.owner_id.clone(),
            record.evidence.decision_index,
        );
        if !seen.insert(key.clone()) {
            issue(
                errors,
                format!("{arm}/decision/{}/{}", key.0, key.1),
                "duplicate decision",
            );
            continue;
        }
        match decision(directory, &manifest, record, &valid_waves) {
            Ok(()) => {
                decisions.insert(key, record.clone());
            }
            Err(error) => issue(errors, format!("{arm}/decision/{}/{}", key.0, key.1), error),
        }
    }
    for owner in &manifest.owners {
        for index in 0..owner.teacher_token_ids.len() {
            if !seen.contains(&(owner.owner_id.clone(), index)) {
                issue(
                    errors,
                    format!("{arm}/decision/{}/{index}", owner.owner_id),
                    "canonical decision missing",
                );
            }
        }
    }
    Some(CheckedCapture {
        directory: directory.to_owned(),
        manifest,
        manifest_sha256,
        decisions,
        receipts,
        external_files,
    })
}

/// The only non-execution configuration difference allowed is a model source
/// location already accounted for by the validated content inventory.
fn comparison_configuration(
    manifest: &VNextTeacherCaptureManifest,
) -> Result<(Value, Option<String>)> {
    let mut configuration = manifest.configuration.clone();
    let Some(location) = configuration.pointer("/backend/backend_options/model_path") else {
        return Ok((configuration, None));
    };
    let path = location
        .as_str()
        .filter(|path| !path.trim().is_empty())
        .context("configured model_path is not a nonempty path")?
        .to_owned();
    let identity = manifest
        .identity
        .as_ref()
        .context("configuration source identity missing")?;
    let source: ProductModelSourceIdentity = serde_json::from_value(identity.model_source.clone())?;
    source.validate().map_err(anyhow::Error::msg)?;
    let covered = ModelArtifactSourceRole::ALL.iter().any(|&role| {
        let original = source.original_sources.for_role(role);
        let resolved = source.resolved_sources.for_role(role);
        path == original.location
            || path == resolved.canonical_location
            || resolved.files.iter().any(|file| {
                Path::new(&resolved.canonical_location).join(&file.relative_path)
                    == Path::new(&path)
            })
    });
    ensure!(
        covered,
        "configured model_path is not covered by the captured source inventory"
    );
    // Preserve field presence, and retain the original value in the report.
    *configuration
        .pointer_mut("/backend/backend_options/model_path")
        .expect("checked path") = json!({"location_bound_by_model_source_content":true});
    Ok((configuration, Some(path)))
}

pub(super) fn compare_identity(
    reference: &CheckedCapture,
    candidate: &CheckedCapture,
    transition: Option<&transition::DeclaredTransition>,
    numerical_transition: Option<&numerical_transition::DeclaredNumericalTransition>,
    shape: ComparisonShape,
) -> Result<Value> {
    ensure!(
        transition.is_none() || numerical_transition.is_none(),
        "implementation and numerical profile transitions are mutually exclusive"
    );
    let a = &reference.manifest;
    let b = &candidate.manifest;
    match shape {
        ComparisonShape::SerialToSerial => ensure!(
            a.mode == "serial" && b.mode == "serial",
            "comparison requires two real serial captures"
        ),
        ComparisonShape::SerialToBatched => ensure!(
            a.mode == "serial" && b.mode == "batched",
            "comparison requires real serial reference and batched candidate"
        ),
        ComparisonShape::BatchedToBatched => ensure!(
            a.mode == "batched" && b.mode == "batched",
            "comparison requires two real batched captures"
        ),
    }
    ensure!(
        a.owners.len() == b.owners.len()
            && !b.owners.is_empty()
            && (shape != ComparisonShape::SerialToBatched || b.owners.len() > 1),
        "comparison must cover the same owners (serial-to-batched requires multiple owners)"
    );
    ensure!(
        a.vocabulary_size == b.vocabulary_size,
        "actual vocabulary/configuration differs between arms"
    );
    let owners = |manifest: &VNextTeacherCaptureManifest| {
        manifest
            .owners
            .iter()
            .map(|owner| {
                (
                    owner.owner_id.clone(),
                    (
                        owner.prompt_token_ids.clone(),
                        owner.teacher_token_ids.clone(),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    ensure!(
        owners(a) == owners(b),
        "paired owner token histories differ"
    );
    let x = a.identity.as_ref().context("reference identity missing")?;
    let y = b.identity.as_ref().context("candidate identity missing")?;
    ensure!(
        x.kv_storage == y.kv_storage
            && (numerical_transition.is_some()
                || (x.numerical_profile == y.numerical_profile
                    && x.family_fingerprint == y.family_fingerprint
                    && x.program_fingerprint == y.program_fingerprint)),
        "model/numerical/KV/resolved plan identity differs"
    );
    ensure!(
        source_content(&x.model_source)? == source_content(&y.model_source)?,
        "semantic/tokenizer/weight source content differs"
    );
    let (mut reference_configuration, reference_model_path) = comparison_configuration(a)?;
    let (mut candidate_configuration, candidate_model_path) = comparison_configuration(b)?;
    if let Some(declaration) = numerical_transition {
        declaration
            .normalize_configuration(&mut reference_configuration, &mut candidate_configuration)?;
    }
    ensure!(
        reference_configuration == candidate_configuration,
        "actual vocabulary/configuration differs between arms"
    );
    let r = reference
        .receipts
        .values()
        .next()
        .context("no valid reference physical receipt")?;
    let c = candidate
        .receipts
        .values()
        .next()
        .context("no valid candidate physical receipt")?;
    let execution_qualification = if let Some(declaration) = numerical_transition {
        declaration.qualify(reference, candidate, r, c)?
    } else if let Some(declaration) = transition {
        declaration.qualify(reference, candidate, r, c)?
    } else {
        ensure!(
            x.resolved_plan_fingerprint == y.resolved_plan_fingerprint,
            "model/numerical/KV/resolved plan identity differs"
        );
        ensure!(
            r.plan_id == c.plan_id
                && r.plan_hash == c.plan_hash
                && r.runtime == c.runtime
                && r.node_signature == c.node_signature,
            "actual executed plan/provider/runtime differs between arms"
        );
        json!({"mode":"strict_fixed_plan","ordered_nodes_and_execution_semantics_matched":true})
    };
    if numerical_transition.is_none() {
        // The numerical declaration checked its exact optional resource-name
        // mapping above. Strict and implementation modes never receive it.
        ensure!(
            r.output_binding == c.output_binding,
            "product output binding differs between arms"
        );
    }
    Ok(
        json!({"reference":reference.external_files,"candidate":candidate.external_files,
        "model_source_content_matched":true,"source_location_difference":x.model_source!=y.model_source,
        "reference_model_label":x.model_id,"candidate_model_label":y.model_id,
        "reference_binary_sha256":x.binary.sha256,"candidate_binary_sha256":y.binary.sha256,
        "same_binary":x.binary.sha256==y.binary.sha256,"fixed_configuration_matched":true,
        "execution_qualification":execution_qualification,
        "configuration_source_locations":{"reference_model_path":reference_model_path,"candidate_model_path":candidate_model_path,
            "normalization":if numerical_transition.is_some() {"model_path_bound_by_source_content_and_exact_declared_numerical_execution.require"} else {"only_backend.backend_options.model_path_bound_by_source_content"}},
        "numerical_profile":if x.numerical_profile==y.numerical_profile {Some(&x.numerical_profile)} else {None},
        "numerical_profiles":{"reference":x.numerical_profile,"candidate":y.numerical_profile},
        "kv_storage":x.kv_storage,
        "receipt_hashes_recomputed":["node_identity","submission","completion","readback"],
        "compiled_batch_root":"captured_opaque_fingerprint_topology_seed_preimage_not_in_schema",
        "history_receipt_binding":"producer_verified_tokens_plus_unique_monotonically_ordered_physical_completions",
        "history_tokens_independently_reconstructed_from_receipt":false,
        "comparison_shape":shape,
        "logical_decode_width":{"reference":if a.mode=="serial" {1} else {a.owners.len()},"candidate":if b.mode=="serial" {1} else {b.owners.len()}},
        "fixed_product_output_binding":if r.output_binding==c.output_binding {Some(&r.output_binding)} else {None},
        "product_output_bindings":{"reference":r.output_binding,"candidate":c.output_binding},
        "full_vocabulary_raw_conversion_verified":true}),
    )
}
