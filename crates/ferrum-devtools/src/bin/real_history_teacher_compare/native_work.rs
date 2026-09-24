//! Optional, bounded actual native inventory audit after the existing capture
//! and identity validators. This proves node dispatch coverage, not PSO traces.
use super::*;
#[path = "native_work/scope.rs"]
mod scope;
#[path = "native_work/wire.rs"]
mod wire;
const FILE_LIMIT: u64 = 16 * 1024 * 1024;

fn bounded_file(path: &Path) -> Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= FILE_LIMIT,
        "native audit input exceeds file bound"
    );
    let mut bytes = Vec::new();
    file.take(FILE_LIMIT + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= FILE_LIMIT,
        "native audit file grew past bound"
    );
    Ok(bytes)
}
fn bounded_artifact(directory: &Path, artifact: &VNextTeacherRawArtifact) -> Result<Vec<u8>> {
    ensure!(
        artifact.bytes <= FILE_LIMIT,
        "native audit artifact exceeds file bound"
    );
    files::artifact(directory, &artifact.file, artifact.bytes, &artifact.sha256)
}

type WaveKey = (String, Vec<(String, usize, usize, String)>);
fn wave_key(wave: &VNextTeacherWaveEvidence) -> WaveKey {
    let mut participants: Vec<_> = wave
        .participants
        .iter()
        .map(|p| {
            (
                p.owner_id.clone(),
                p.immediate_start,
                p.immediate_end,
                p.history_sha256.clone(),
            )
        })
        .collect();
    participants.sort();
    (wave.kind.clone(), participants)
}
#[derive(Serialize)]
struct NodeAudit {
    node_id: String,
    node_index: usize,
    eligible_weights: bool,
    in_shared_tail_scope: bool,
    physical_rows: u64,
    compute_dispatches: u64,
    family_signature: [u8; 32],
    work: wire::Work,
}
struct CheckedWave {
    index: usize,
    receipt: String,
    nodes: Vec<NodeAudit>,
    other_routes: Vec<Value>,
}

pub(super) fn audit(
    args: &Args,
    path: &Path,
    reference: &capture::CheckedCapture,
    candidate: &capture::CheckedCapture,
) -> Result<Value> {
    ensure!(args.comparison_shape == ComparisonShape::BatchedToBatched
        && args.require_bitwise_logits && args.numerical_profile_transition.is_none(),
        "native shared-tail scope requires batched-to-batched, required bits and no numerical migration");
    let width = reference.manifest.owners.len();
    ensure!(
        (5..=7).contains(&width) && candidate.manifest.owners.len() == width,
        "native shared-tail audit requires actual owner width 5..7"
    );
    let (provenance, leaves) = scope::load(path, [reference, candidate])?;
    let a = arm(reference, &leaves, false)?;
    let b = arm(candidate, &leaves, true)?;
    ensure!(
        a.keys().eq(b.keys()),
        "native audit arms have different actual wave histories"
    );
    let mut waves = Vec::new();
    let mut decode_waves = 0;
    let mut hits = 0;
    let mut unchanged = 0;
    for (key, old) in &a {
        let new = &b[key];
        ensure!(
            old.other_routes == new.other_routes,
            "non-target native operation route changed"
        );
        ensure!(
            old.nodes.len() == new.nodes.len(),
            "FFN node inventory differs between arms"
        );
        for (x, y) in old.nodes.iter().zip(&new.nodes) {
            ensure!(
                x.node_id == y.node_id
                    && x.physical_rows == y.physical_rows
                    && x.in_shared_tail_scope == y.in_shared_tail_scope,
                "paired FFN scope/work differs"
            );
            if y.in_shared_tail_scope {
                hits += 1;
            } else {
                unchanged += 1;
                ensure!(
                    x.compute_dispatches == y.compute_dispatches && x.work == y.work,
                    "out-of-scope FFN dispatch count changed"
                );
            }
        }
        decode_waves += usize::from(key.0 == "decode");
        waves.push(json!({"kind":key.0,"participants":key.1.len(),
            "reference_wave_index":old.index,"candidate_wave_index":new.index,
            "reference_receipt_fingerprint":old.receipt,"candidate_receipt_fingerprint":new.receipt,
            "reference_nodes":old.nodes,"candidate_nodes":new.nodes}));
    }
    let expected_decode = reference.manifest.owners[0].teacher_token_ids.len() - 1;
    ensure!(
        decode_waves == expected_decode && hits > 0,
        "native audit does not cover the complete actual decode history with target hits"
    );
    Ok(json!({"schema_version":1,"passed":true,"scope":provenance,
        "coverage":"completed_backend_node_dispatch_inventory_not_per_pso_trace",
        "physical_decode_width":width,"decode_waves":decode_waves,"all_waves":waves.len(),
        "eligible_layer_count":leaves.values().filter(|leaf|leaf.eligible_weights).count(),
        "all_ffn_layer_count":leaves.len(),"hit_node_waves_per_arm":hits,
        "unchanged_node_waves_per_arm":unchanged,"weight_nodes":leaves,
        "waves":waves,"quality_budgets_replaced":false,"performance_evidence":false}))
}

fn arm(
    capture: &capture::CheckedCapture,
    leaves: &BTreeMap<String, scope::LeafSet>,
    candidate: bool,
) -> Result<BTreeMap<WaveKey, CheckedWave>> {
    let path = capture.directory.join("native-work-manifest.json");
    ensure!(
        fs::canonicalize(&path)?.starts_with(fs::canonicalize(&capture.directory)?),
        "native index escapes capture directory"
    );
    let index_bytes = bounded_file(&path)?;
    let index: wire::Index = serde_json::from_slice(&index_bytes)?;
    ensure!(
        index.schema_version == 1
            && index.artifact_type == "ferrum.teacher.completed_native_work_index"
            && index.complete
            && index.errors.is_empty()
            && index.file_limit_bytes == FILE_LIMIT,
        "native index is incomplete or unsupported"
    );
    ensure!(
        serde_json::to_value(&index.identity)? == serde_json::to_value(&capture.manifest.identity)?,
        "native index identity differs from actual teacher manifest"
    );
    ensure!(
        index.waves.len() == capture.manifest.waves.len(),
        "native index wave coverage incomplete"
    );
    let mut files_seen = BTreeSet::new();
    let mut result = BTreeMap::new();
    for (indexed, wave) in index.waves.iter().zip(&capture.manifest.waves) {
        ensure!(
            capture.receipts.contains_key(&wave.wave_index),
            "native audit has no validated receipt"
        );
        ensure!(
            indexed.wave_index == wave.wave_index
                && indexed.completion_fingerprint == wave.completion_fingerprint
                && indexed.receipt_fingerprint == wave.receipt_fingerprint
                && files_seen.insert(indexed.artifact.file.clone()),
            "native index wave repeated, reordered or rebound"
        );
        let bytes = bounded_artifact(&capture.directory, &indexed.artifact)?;
        let record: wire::Wave = serde_json::from_slice(&bytes)?;
        let receipt_file = wave
            .completion_receipt
            .as_ref()
            .context("missing completion")?;
        let completed: receipt::Completion =
            serde_json::from_slice(&bounded_artifact(&capture.directory, receipt_file)?)?;
        // The original validator already checked every authority and readback.
        // Reuse its exact typed receipt, and join rather than reconstruct hashes.
        let checked = checked_wave(wave, &completed, &record, leaves, candidate)?;
        ensure!(
            result.insert(wave_key(wave), checked).is_none(),
            "duplicate actual native wave history"
        );
    }
    Ok(result)
}

fn checked_wave(
    wave: &VNextTeacherWaveEvidence,
    completed: &receipt::Completion,
    record: &wire::Wave,
    leaves: &BTreeMap<String, scope::LeafSet>,
    candidate: bool,
) -> Result<CheckedWave> {
    let submission = &completed.submission;
    let batch = &submission.batch_identity;
    ensure!(
        record.schema_version == 1
            && record.artifact_type == "ferrum.teacher.completed_native_work"
            && record.coverage == "backend_reported_only_no_inferred_node_coverage"
            && record.wave_index == wave.wave_index
            && record.kind == wave.kind
            && record.participant_count == wave.participant_count
            && record.completion_fingerprint == wave.completion_fingerprint
            && record.receipt_fingerprint == wave.receipt_fingerprint
            && record.submission_fingerprint == submission.fingerprint
            && record.batch_identity_fingerprint == batch.fingerprint
            && completed.fingerprint == record.completion_fingerprint
            && completed.disposition == json!({"status":"succeeded"}),
        "native record is cross-wave, rebound or not successfully completed"
    );
    // This scope is the Metal eager candidate. Never infer logical nodes from
    // a replay command. Another backend requires a separate replay-aware scope.
    ensure!(
        record.device.replayed_segments.is_empty() && record.device.graph_evidence.is_none(),
        "replay/graph attribution is outside this native audit scope"
    );
    let rows = wave.participants.iter().try_fold(0_u64, |sum, p| {
        let count = p
            .immediate_end
            .checked_sub(p.immediate_start)
            .filter(|&n| n > 0)
            .context("empty native participant span")?;
        sum.checked_add(u64::try_from(count)?)
            .context("native rows overflow")
    })?;
    ensure!(rows > 0, "native wave has no physical rows");
    if wave.kind == "decode" {
        ensure!(
            rows == wave.participant_count as u64
                && wave
                    .participants
                    .iter()
                    .all(|p| p.immediate_end - p.immediate_start == 1),
            "decode native rows differ from real one-token participants"
        );
    }
    let actual_ffn: BTreeSet<_> = batch
        .nodes
        .iter()
        .filter(|n| n.operation_id == "operation.dense_swiglu")
        .map(|n| n.node_id.clone())
        .collect();
    ensure!(
        actual_ffn == leaves.keys().cloned().collect(),
        "actual FFN nodes differ from complete GGUF inventory"
    );
    let commands = &record.device.commands;
    ensure!(
        !commands.is_empty()
            && commands
                .windows(2)
                .all(|w| w[0].command_index < w[1].command_index),
        "native command indices repeated or reordered"
    );
    let mut observed = BTreeMap::<String, &wire::Command>::new();
    let mut other_routes = Vec::new();
    for command in commands {
        ensure!(
            command.execution_path == "eager" && command.reusable_graph_node_count.is_none(),
            "native command is not the declared eager scope"
        );
        ensure!(
            !command.native_op_id.is_empty()
                && matches!(
                    command.command_phase.as_str(),
                    "initialization" | "dynamic_binding" | "compute" | "result_binding"
                )
                && matches!(
                    command.batching_form.as_str(),
                    "scalar" | "packed" | "participant_loop"
                )
                && (command.compute_dispatch_count > 0 || command.transfer_command_count > 0),
            "invalid native command metadata"
        );
        if let Some(s) = &command.statistical_evidence {
            ensure!(
                s.schema_version == 1
                    && s.token_count == command.token_count
                    && s.compute_dispatches == command.compute_dispatch_count
                    && s.transfer_commands == command.transfer_command_count,
                "selected statistics differ from actual native command"
            );
        }
        let Some(node_index) = command.node_index else {
            continue;
        };
        let node = batch
            .nodes
            .get(node_index as usize)
            .context("native command points outside actual nodes")?;
        ensure!(
            node.node_index == node_index as usize
                && command.participant_count > 0
                && command
                    .participant_start
                    .checked_add(command.participant_count)
                    .is_some_and(|end| end as usize <= node.participants.len()),
            "native command participant range differs from actual node"
        );
        if leaves.contains_key(&node.node_id) {
            ensure!(
                observed.insert(node.node_id.clone(), command).is_none(),
                "duplicate native FFN node command"
            );
        } else {
            other_routes.push(json!({"node_id":node.node_id,"phase":command.command_phase,
                "native_op_id":command.native_op_id,"batching_form":command.batching_form,
                "participant_start":command.participant_start,"participant_count":command.participant_count,
                "token_count":command.token_count,"compute":command.compute_dispatch_count,
                "transfers":command.transfer_command_count}));
        }
    }
    ensure!(
        observed.len() == leaves.len(),
        "missing native FFN node command"
    );
    let mut nodes = Vec::new();
    for (id, leaf) in leaves {
        let command = observed[id];
        ensure!(
            command.command_phase == "compute"
                && command.native_op_id == "vnext_dense_swiglu"
                && command.participant_start == 0
                && command.participant_count as usize == wave.participant_count
                && command.token_count == rows
                && command.transfer_command_count == 0
                && command.batching_form
                    == if wave.participant_count == 1 {
                        "scalar"
                    } else {
                        "packed"
                    },
            "FFN command does not cover actual full physical batch"
        );
        let in_scope = leaf.eligible_weights && (5..=7).contains(&rows);
        if (5..=7).contains(&rows) {
            ensure!(
                command.compute_dispatch_count == if candidate && in_scope { 6 } else { 4 },
                "FFN native dispatch count does not match declared target/control route"
            );
        }
        let stats = command
            .statistical_evidence
            .as_ref()
            .context("missing actual FFN selected work evidence")?;
        let (logical, inner) = leaf.work(rows)?;
        if (5..=7).contains(&rows) {
            ensure!(
                stats.work.logical_units == logical
                    && stats.work.inner_work_units == inner
                    && stats.work.padded_units >= logical
                    && stats.work.grid_blocks > 0
                    && stats.work.staged_weight_bytes == 0
                    && stats.work.host_to_device_bytes == 0
                    && stats.work.device_to_host_bytes == 0
                    && stats.work.device_to_device_bytes == 0
                    && stats.work.fill_bytes == 0,
                "FFN actual work differs from full gate/up/activation/down ABI"
            );
        }
        nodes.push(NodeAudit {
            node_id: id.clone(),
            node_index: command.node_index.unwrap() as usize,
            eligible_weights: leaf.eligible_weights,
            in_shared_tail_scope: in_scope,
            physical_rows: rows,
            compute_dispatches: command.compute_dispatch_count,
            family_signature: stats.family_signature,
            work: stats.work.clone(),
        });
    }
    Ok(CheckedWave {
        index: wave.wave_index,
        receipt: wave.receipt_fingerprint.clone(),
        nodes,
        other_routes,
    })
}
