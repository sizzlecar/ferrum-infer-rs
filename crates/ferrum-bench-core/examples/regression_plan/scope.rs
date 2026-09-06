//! Refine only proven coordinated version edits from immutable Git snapshots.
use super::{git, resolve_revision};
use ferrum_bench_core::release_regression::{
    analyze_paths, coordinated_version_paths, ChangeArea, Impact,
};
use semver::Version;
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

pub(super) struct Analysis {
    pub impact: Impact,
    pub version_refinement: Value,
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

pub(super) fn analyze(repo: &Path, base: &str, candidate: &str, paths: &[String]) -> Analysis {
    let mut impact = analyze_paths(paths);
    if !paths
        .iter()
        .any(|path| path == "Cargo.toml" || path == "Cargo.lock" || path.ends_with("/Cargo.toml"))
    {
        return Analysis {
            impact,
            version_refinement: json!({"applied": false, "reason": "no Cargo inputs changed"}),
        };
    }
    let refinement = (|| {
        let before = cargo_snapshot(repo, base)?;
        let after = cargo_snapshot(repo, candidate)?;
        coordinated_version_paths(&before, &after)
    })();
    let version_refinement = match refinement {
        Ok(version_paths) => {
            for entry in &mut impact.paths {
                if version_paths.contains(&entry.path) {
                    entry.areas = vec![ChangeArea::Build];
                    entry.reason = "verified coordinated version metadata update; preserve build, installation and release baseline checks without inferring inference-code changes".into();
                }
            }
            impact.areas = ChangeArea::ALL
                .into_iter()
                .filter(|area| impact.paths.iter().any(|entry| entry.areas.contains(area)))
                .collect();
            json!({"applied": true, "paths": version_paths})
        }
        Err(reason) => {
            json!({"applied": false, "reason": reason, "fallback": "conservative path impact retained"})
        }
    };
    Analysis {
        impact,
        version_refinement,
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
