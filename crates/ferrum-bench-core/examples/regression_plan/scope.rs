//! Refine only proven coordinated version edits from immutable Git snapshots.
use super::{git, resolve_revision};
use ferrum_bench_core::release_regression::dependency_change::validation_dependency_paths;
use ferrum_bench_core::release_regression::source_change::{
    bench_release_exports_only, homebrew_documentation_only, rust_validation_only,
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
    }
    let mut rust_paths = Vec::new();
    let mut release_tool_paths = Vec::new();
    let mut installation_paths = Vec::new();
    let mut unresolved = Vec::new();
    for entry in &mut impact.paths {
        let readme = matches!(entry.path.as_str(), "README.md" | "README_zh.md");
        let rust = entry.path.starts_with("crates/")
            && entry.path.contains("/src/")
            && entry.path.ends_with(".rs")
            && entry.areas != [ChangeArea::Validation];
        if !readme && !rust {
            continue;
        }
        let comparison = (|| -> Result<Option<ContentProof>, String> {
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
            } else if entry.path == "crates/ferrum-bench-core/src/lib.rs"
                && bench_release_exports_only(&before, &after)?
            {
                Ok(Some(ContentProof::ReleaseExports))
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
    json!({"rust_validation_paths": rust_paths, "release_tool_export_paths": release_tool_paths, "homebrew_installation_paths": installation_paths, "unresolved": unresolved})
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
                        || !refinement.validation_runtime_dependencies.is_empty();
                    for entry in &mut impact.paths {
                        if refinement.paths.contains(&entry.path) {
                            entry.areas = if build {
                                vec![ChangeArea::Build, ChangeArea::Validation]
                            } else {
                                vec![ChangeArea::Validation]
                            };
                            entry.reason = if build {
                                "verified development/release-tool dependency changes with unchanged existing registry resolution; retain build and installation checks, while production source changes retain their own impact"
                            } else {
                                "verified development dependency changes without altered normal/build dependencies or existing registry resolution"
                            }.into();
                        }
                    }
                    json!({"applied": true, "analysis": refinement})
                }
                Err(reason) => {
                    json!({"applied": false, "reason": reason, "fallback": "conservative path impact retained"})
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
    let (_, reference, commit) = latest.ok_or("no previous reachable formal release tag found; fetch release tags before release planning")?;
    if base != commit {
        return Err(format!("release base must resolve to latest reachable formal tag {reference}; a shorter diff can omit earlier release changes"));
    }
    Ok(reference)
}
