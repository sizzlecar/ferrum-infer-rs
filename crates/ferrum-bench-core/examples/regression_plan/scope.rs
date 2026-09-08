//! Refine only proven coordinated version edits from immutable Git snapshots.
use super::{git, resolve_revision};
use ferrum_bench_core::release_regression::dependency_change::{
    product_dependency_paths, validation_dependency_paths,
};
use ferrum_bench_core::release_regression::source_change::{
    bench_release_exports_only, boxed_command_dispatch_only, cli_run_reasoning_mask_only,
    cli_serve_host_capability_only, core_cuda_build_host_only, cuda_diagnostic_loader_only,
    cuda_optional_exports_only, gptq_format_validation_only, homebrew_documentation_only,
    journal_initialization_only, legacy_metal_submission_only, model_template_wiring_only,
    reasoning_controls_exports_only, reasoning_descriptor_metadata_only,
    reasoning_effort_declaration_only, run_ready_capability_only, rust_validation_only,
    windows_process_memory_only,
};
use ferrum_bench_core::release_regression::source_change::{
    protocol_observability_unchanged, ProtocolObservabilityKind,
};
use ferrum_bench_core::release_regression::{
    analyze_paths, coordinated_version_paths, ChangeArea, Impact,
};
use semver::Version;
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

pub(super) struct Analysis {
    pub impact: Impact,
    pub version_refinement: Value,
    pub dependency_refinement: Value,
    pub content_refinement: Value,
}

fn cargo_snapshot(repo: &Path, revision: &str) -> Result<BTreeMap<String, String>, String> {
    let tree = git(
        repo,
        &["ls-tree", "-r", "--name-only", "-z", revision, "--"],
    )?;
    let mut snapshot = BTreeMap::new();
    // Decode only Cargo inputs; an unrelated non-UTF-8 asset does not affect
    // whether these manifests form a coordinated version update.
    for path in tree.split(|byte| *byte == 0).filter(|path| {
        *path == b"Cargo.toml" || *path == b"Cargo.lock" || path.ends_with(b"/Cargo.toml")
    }) {
        let path = std::str::from_utf8(path).map_err(|_| "non-UTF-8 Cargo manifest path")?;
        let object = format!("{revision}:{path}");
        let bytes = git(repo, &["cat-file", "blob", &object])?;
        let text =
            String::from_utf8(bytes).map_err(|_| format!("non-UTF-8 Cargo input: {path}"))?;
        snapshot.insert(path.into(), text);
    }
    Ok(snapshot)
}

fn refine_content(repo: &Path, base: &str, candidate: &str, impact: &mut Impact) -> Value {
    enum ContentProof {
        Homebrew,
        Tests,
        ReleaseExports,
        LegacyMetalSubmission,
        ReadyCapability,
        ModelTemplateWiring,
        GptqFormatValidation,
        ReasoningMetadata,
        CommandAllocation,
        CliCapability,
        CudaHost,
        CoreBuild,
        ObservabilityHost,
        ProtocolObservability,
    }
    let mut rust_paths = Vec::new();
    let mut release_tool_paths = Vec::new();
    let mut installation_paths = Vec::new();
    let mut legacy_submission_paths = Vec::new();
    let mut ready_capability_paths = Vec::new();
    let mut model_template_paths = Vec::new();
    let mut gptq_format_paths = Vec::new();
    let mut reasoning_metadata_paths = Vec::new();
    let mut command_allocation_paths = Vec::new();
    let mut cli_capability_paths = Vec::new();
    let mut cuda_host_paths = Vec::new();
    let mut core_build_paths = Vec::new();
    let mut observability_host_paths = Vec::new();
    let mut protocol_observability_paths = Vec::new();
    let mut unresolved = Vec::new();
    for entry in &mut impact.paths {
        let core_build = matches!(
            entry.path.as_str(),
            "crates/ferrum-kernels/build.rs" | "crates/ferrum-kernels/build_support/host.rs"
        );
        let readme = matches!(entry.path.as_str(), "README.md" | "README_zh.md");
        let rust = entry.path.starts_with("crates/")
            && entry.path.contains("/src/")
            && entry.path.ends_with(".rs")
            && entry.areas != [ChangeArea::Validation];
        if !readme && !rust && !core_build {
            continue;
        }
        let comparison = (|| -> Result<Option<ContentProof>, String> {
            if core_build {
                let read = |revision: &str, path: &str| {
                    String::from_utf8(git(
                        repo,
                        &["cat-file", "blob", &format!("{revision}:{path}")],
                    )?)
                    .map_err(|_| "non-UTF-8 core build scope input".to_owned())
                };
                return Ok(core_cuda_build_host_only(
                    &read(base, "crates/ferrum-kernels/build.rs")?,
                    &read(candidate, "crates/ferrum-kernels/build.rs")?,
                    &read(candidate, "crates/ferrum-kernels/build_support/host.rs")?,
                )?
                .then_some(ContentProof::CoreBuild));
            }
            let read = |revision| {
                String::from_utf8(git(
                    repo,
                    &["cat-file", "blob", &format!("{revision}:{}", entry.path)],
                )?)
                .map_err(|_| "non-UTF-8 scope input".to_owned())
            };
            let before = read(base)?;
            let after = read(candidate)?;
            if readme {
                Ok(homebrew_documentation_only(&before, &after)?.then_some(ContentProof::Homebrew))
            } else if rust_validation_only(&before, &after)? {
                Ok(Some(ContentProof::Tests))
            } else if (entry.path == "crates/ferrum-bench-core/src/jsonl_journal.rs"
                && journal_initialization_only(&before, &after)?)
                || (entry.path == "crates/ferrum-types/src/process_memory.rs"
                    && windows_process_memory_only(&before, &after)?)
            {
                Ok(Some(ContentProof::ObservabilityHost))
            } else if match entry.path.as_str() {
                "crates/ferrum-types/src/requests.rs" => protocol_observability_unchanged(
                    &before,
                    &after,
                    ProtocolObservabilityKind::Requests,
                )?,
                "crates/ferrum-engine/src/continuous_engine/sequence.rs" => {
                    protocol_observability_unchanged(
                        &before,
                        &after,
                        ProtocolObservabilityKind::Sequence,
                    )?
                }
                "crates/ferrum-engine/src/continuous_engine/inner/completion.rs" => {
                    protocol_observability_unchanged(
                        &before,
                        &after,
                        ProtocolObservabilityKind::Completion,
                    )?
                }
                _ => false,
            } {
                Ok(Some(ContentProof::ProtocolObservability))
            } else if entry.path == "crates/ferrum-bench-core/src/lib.rs"
                && bench_release_exports_only(&before, &after)?
            {
                Ok(Some(ContentProof::ReleaseExports))
            } else if entry.path == "crates/ferrum-kernels/src/backend/metal/mod.rs"
                && legacy_metal_submission_only(&before, &after)?
            {
                Ok(Some(ContentProof::LegacyMetalSubmission))
            } else if entry.path == "crates/ferrum-cli/src/main.rs"
                && boxed_command_dispatch_only(&before, &after)?
            {
                Ok(Some(ContentProof::CommandAllocation))
            } else if (matches!(
                entry.path.as_str(),
                "crates/ferrum-kernels/src/lib.rs"
                    | "crates/ferrum-kernels/src/backend/cuda/mod.rs"
            ) && cuda_optional_exports_only(&before, &after)?)
                || (entry.path == "crates/ferrum-kernels/src/backend/cuda/fa2_ffi.rs"
                    && cuda_diagnostic_loader_only(&before, &after)?)
            {
                Ok(Some(ContentProof::CudaHost))
            } else if match entry.path.as_str() {
                "crates/ferrum-types/src/lib.rs" => {
                    reasoning_controls_exports_only(&before, &after)?
                }
                "crates/ferrum-models/src/vnext/mod.rs" => {
                    reasoning_descriptor_metadata_only(&before, &after)?
                }
                "crates/ferrum-models/src/vnext/gpt_oss/mod.rs" => {
                    reasoning_effort_declaration_only(&before, &after)?
                }
                _ => false,
            } {
                Ok(Some(ContentProof::ReasoningMetadata))
            } else if matches!(
                entry.path.as_str(),
                "crates/ferrum-models/src/vnext/qwen35.rs"
                    | "crates/ferrum-models/src/vnext/qwen3_moe/mod.rs"
            ) && model_template_wiring_only(&before, &after)?
            {
                Ok(Some(ContentProof::ModelTemplateWiring))
            } else if entry.path == "crates/ferrum-models/src/vnext/qwen3_moe/config.rs"
                && gptq_format_validation_only(&before, &after)?
            {
                Ok(Some(ContentProof::GptqFormatValidation))
            } else if (entry.path == "crates/ferrum-cli/src/commands/run.rs"
                && cli_run_reasoning_mask_only(&before, &after)?)
                || (entry.path == "crates/ferrum-cli/src/commands/serve.rs"
                    && cli_serve_host_capability_only(&before, &after)?)
            {
                Ok(Some(ContentProof::CliCapability))
            } else if entry.path == "crates/ferrum-cli/src/commands/run.rs" {
                let template = String::from_utf8(git(
                    repo,
                    &[
                        "cat-file",
                        "blob",
                        &format!("{candidate}:crates/ferrum-server/src/chat_template.rs"),
                    ],
                )?)
                .map_err(|_| "non-UTF-8 candidate template scope input".to_owned())?;
                Ok(run_ready_capability_only(&before, &after, &template)?
                    .then_some(ContentProof::ReadyCapability))
            } else {
                Ok(None)
            }
        })();
        match comparison {
            Ok(Some(ContentProof::Homebrew)) => {
                entry.areas = vec![ChangeArea::Build];
                entry.reason = "only explicit Homebrew installation blocks changed; model commands and performance text outside those blocks are unchanged; require distribution installation checks".into();
                installation_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::Tests)) => {
                entry.areas = vec![ChangeArea::Validation];
                entry.reason = "Rust AST production tokens are unchanged after removing only explicit top-level cfg(test) modules".into();
                rust_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::ReleaseExports)) => {
                entry.areas = vec![ChangeArea::Build, ChangeArea::Validation];
                entry.reason = "bench-core root AST only adds reviewed release_candidate/release_regression module exports; every existing item, including observability APIs, is unchanged".into();
                release_tool_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::LegacyMetalSubmission)) => {
                for area in &mut entry.areas {
                    if *area == ChangeArea::Kernel {
                        *area = ChangeArea::BackendSubmission;
                    }
                }
                entry.execution_paths = Some(vec!["legacy-model-executor".into()]);
                entry.reason = "AST changes are confined to MetalContext submission/completion and its checked-sync API; shared state, operator bodies and Backend trait implementations are unchanged; independent production-plan queues remain outside this path's reach".into();
                legacy_submission_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::ReadyCapability)) => {
                entry.areas = vec![ChangeArea::Template];
                entry.reason = "ready JSONL only appends the declared reasoning capability through a proven scalar observation and existing Option borrow; original fields, call positions/arguments and all other production AST remain unchanged; retain protocol and capability regression".into();
                ready_capability_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::ModelTemplateWiring)) => {
                entry.areas = vec![
                    ChangeArea::Download,
                    ChangeArea::Template,
                    ChangeArea::Termination,
                ];
                entry.reason = "AST changes only wire immutable standalone template sources and validate their selected name; metadata inputs, model configuration, weights, logical programs and execution remain unchanged".into();
                model_template_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::GptqFormatValidation)) => {
                entry.areas = vec![ChangeArea::Download];
                entry.reason = "AST changes only the exact GPTQ v1 source format acceptance guards; every parsed quantization value, model parameter, conversion, validation and caller is unchanged".into();
                gptq_format_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::ReasoningMetadata)) => {
                entry.areas = vec![
                    ChangeArea::Template,
                    ChangeArea::Termination,
                    ChangeArea::Structured,
                    ChangeArea::Tools,
                ];
                entry.reason = "AST only adds explicit reasoning capability metadata, pure accessors/declaration or its public module exports; existing model parameters, tensor ownership, programs and execution remain unchanged; retain shared protocol coverage".into();
                reasoning_metadata_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::CommandAllocation)) => {
                entry.areas = vec![ChangeArea::Build];
                entry.reason = "AST preserves startup, every command dispatch arm and its arguments, configuration and error handling; only independent command futures move to heap allocation; retain installation and entrypoint startup checks".into();
                command_allocation_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::CliCapability)) => {
                entry.areas = vec![
                    ChangeArea::Build,
                    ChangeArea::Template,
                    ChangeArea::Termination,
                    ChangeArea::Structured,
                    ChangeArea::Tools,
                ];
                entry.reason = "AST changes only the declared-template initial token mask or Unix-only diagnostic capability and reasoning help; other CLI configuration, scheduling, model loading and device execution remain unchanged".into();
                cli_capability_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::CudaHost)) => {
                entry.areas = vec![ChangeArea::Build];
                entry.reason = "AST only excludes unavailable optional multi-device exports on Windows or returns an explicit unsupported diagnostic loader there; every existing Unix loader, export and numerical implementation is unchanged".into();
                cuda_host_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::CoreBuild)) => {
                entry.areas = vec![ChangeArea::Build];
                entry.reason = "AST proves the reviewed host tool/link/cache identity changes preserve every core CUDA source/header, device codegen flag and build control flow; the same candidate host module preserves Unix tool invocation".into();
                core_build_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::ObservabilityHost)) => {
                for area in &mut entry.areas {
                    if *area == ChangeArea::Observability {
                        *area = ChangeArea::ObservabilityContract;
                    }
                }
                entry.reason = "complete production AST comparison confines the change to journal truncate/open initialization or the explicit Windows resident-memory sampler; retain host sink, lifecycle and accounting contracts; model event acquisition and Unix sampling remain unchanged".into();
                observability_host_paths.push(entry.path.clone());
            }
            Ok(Some(ContentProof::ProtocolObservability)) => {
                entry
                    .areas
                    .retain(|area| *area != ChangeArea::Observability);
                entry.reason = "closed protocol AST transformations preserve existing metadata, model event acquisition and sink lifecycle code; all independently required protocol, scheduler and KV checks remain in scope".into();
                protocol_observability_paths.push(entry.path.clone());
            }
            Ok(None) => {}
            Err(reason) => unresolved.push(json!({"path": entry.path, "reason": reason})),
        }
    }
    impact.product_contract_changed = impact.paths.iter().any(|entry| {
        matches!(entry.path.as_str(), "README.md" | "README_zh.md")
            && !installation_paths.contains(&entry.path)
    });
    impact.areas = ChangeArea::ALL
        .into_iter()
        .filter(|area| impact.paths.iter().any(|entry| entry.areas.contains(area)))
        .collect();
    json!({
        "rust_validation_paths": rust_paths,
        "release_tool_export_paths": release_tool_paths,
        "homebrew_installation_paths": installation_paths,
        "legacy_metal_submission_paths": legacy_submission_paths,
        "ready_capability_paths": ready_capability_paths,
        "model_template_paths": model_template_paths,
        "gptq_format_paths": gptq_format_paths,
        "reasoning_metadata_paths": reasoning_metadata_paths,
        "command_allocation_paths": command_allocation_paths,
        "cli_capability_paths": cli_capability_paths,
        "cuda_host_paths": cuda_host_paths,
        "core_build_paths": core_build_paths,
        "observability_host_paths": observability_host_paths,
        "protocol_observability_paths": protocol_observability_paths,
        "unresolved": unresolved,
    })
}

pub(super) fn analyze(repo: &Path, base: &str, candidate: &str, paths: &[String]) -> Analysis {
    let mut impact = analyze_paths(paths);
    let content_refinement = refine_content(repo, base, candidate, &mut impact);
    if !paths
        .iter()
        .any(|path| path == "Cargo.toml" || path == "Cargo.lock" || path.ends_with("/Cargo.toml"))
    {
        return Analysis {
            impact,
            version_refinement: json!({"applied": false, "reason": "no Cargo inputs changed"}),
            dependency_refinement: json!({"applied": false, "reason": "no Cargo inputs changed"}),
            content_refinement,
        };
    }
    // Both refinements use one immutable snapshot read, including unchanged
    // member manifests and the complete lockfile. Source paths remain in union.
    let snapshots = (|| {
        Ok::<_, String>((
            cargo_snapshot(repo, base)?,
            cargo_snapshot(repo, candidate)?,
        ))
    })();
    let version = snapshots
        .as_ref()
        .map_err(Clone::clone)
        .and_then(|(before, after)| coordinated_version_paths(before, after));
    let (version_refinement, dependency_refinement) = match version {
        Ok(version_paths) => {
            for entry in &mut impact.paths {
                if version_paths.contains(&entry.path) {
                    entry.areas = vec![ChangeArea::Build];
                    entry.reason = "verified coordinated version metadata update; preserve build, installation and release baseline checks without inferring inference-code changes".into();
                }
            }
            (
                json!({"applied": true, "paths": version_paths}),
                json!({"applied": false, "reason": "coordinated version refinement already handled Cargo changes"}),
            )
        }
        Err(version_reason) => {
            let dependency = snapshots
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|(before, after)| validation_dependency_paths(before, after));
            let dependency_refinement = match dependency {
                Ok(refinement) => {
                    let build = refinement.coordinated_version
                        || !refinement.validation_runtime_dependencies.is_empty()
                        || !refinement.private_tool_members.is_empty();
                    for entry in &mut impact.paths {
                        if refinement.paths.contains(&entry.path) {
                            entry.areas = if build {
                                vec![ChangeArea::Build, ChangeArea::Validation]
                            } else {
                                vec![ChangeArea::Validation]
                            };
                            entry.reason = if build {
                                "verified development/release-tool dependency changes with preserved registry identities and explicitly isolated private-tool feature edges; retain build and installation checks, while production source changes retain their own impact"
                            } else {
                                "verified development dependency changes without altered normal/build dependencies or existing registry resolution"
                            }.into();
                        }
                    }
                    json!({"applied": true, "analysis": refinement})
                }
                Err(reason) => {
                    let product = snapshots
                        .as_ref()
                        .map_err(Clone::clone)
                        .and_then(|(before, after)| product_dependency_paths(before, after));
                    if let Ok(refinement) = &product {
                        for entry in &mut impact.paths {
                            if let Some(areas) = refinement.paths.get(&entry.path) {
                                entry.areas = areas.clone();
                                entry.reason = "reviewed product dependency feature/edge transition; complete remaining Cargo snapshots and registry identities verified; retain the dependency's declared protocol, installation and process-observation responsibilities".into();
                            } else if refinement.validation.paths.contains(&entry.path) {
                                entry.areas = vec![ChangeArea::Build, ChangeArea::Validation];
                                entry.reason = "remaining coordinated metadata and validation dependency changes verified after the separately classified product dependency transitions".into();
                            }
                        }
                        json!({"applied": true, "analysis": refinement})
                    } else {
                        for entry in &mut impact.paths {
                            if entry.path == "crates/ferrum-devtools/Cargo.toml" {
                                entry.areas = ChangeArea::ALL.to_vec();
                                entry.reason = "private tool manifest isolation could not be proven from complete Cargo snapshots; retain full impact".into();
                            }
                        }
                        json!({"applied": false, "reason": reason, "product_reason": product.err(), "fallback": "conservative path impact retained"})
                    }
                }
            };
            (
                json!({"applied": false, "reason": version_reason}),
                dependency_refinement,
            )
        }
    };
    impact.areas = ChangeArea::ALL
        .into_iter()
        .filter(|area| impact.paths.iter().any(|entry| entry.areas.contains(area)))
        .collect();
    Analysis {
        impact,
        version_refinement,
        dependency_refinement,
        content_refinement,
    }
}

/// The repository's formal release tag convention is vMAJOR.MINOR.PATCH.
/// Require the newest reachable formal tag as the base, so HEAD~1 cannot omit
/// earlier PRs in the same release. Tags identify the comparison interval only;
/// neither their presence nor their object IDs establish behavior correctness.
pub(super) fn validate_release_base(
    repo: &Path,
    base: &str,
    candidate: &str,
) -> Result<String, String> {
    let bytes = git(
        repo,
        &[
            "for-each-ref",
            "--format=%(refname)",
            "--merged",
            candidate,
            "refs/tags/",
        ],
    )?;
    let tags = String::from_utf8(bytes).map_err(|_| "non-UTF-8 release tag listing")?;
    let mut latest: Option<(Version, String, String)> = None;
    for reference in tags.lines() {
        let Some(name) = reference.strip_prefix("refs/tags/v") else {
            continue;
        };
        let Ok(version) = Version::parse(name) else {
            continue;
        };
        if !version.pre.is_empty() || !version.build.is_empty() {
            continue;
        }
        let commit = resolve_revision(repo, reference)?;
        // Allow a historical candidate to be checked after its formal tag exists.
        if commit == candidate {
            continue;
        }
        if latest
            .as_ref()
            .is_none_or(|(current, _, _)| version > *current)
        {
            latest = Some((version, reference.to_owned(), commit));
        }
    }
    let (version, reference, commit) = latest.ok_or("no previous reachable formal release tag found; fetch release tags before release planning")?;
    if base != commit {
        return Err(format!("release base must resolve to latest reachable formal tag {reference}; a shorter diff can omit earlier release changes"));
    }
    // Git lookup uses the unambiguous full ref. Delivery and GitHub's release
    // API consume the tag name; the exact commit remains separate provenance.
    Ok(format!("v{version}"))
}
