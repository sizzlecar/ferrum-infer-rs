//! Generate a reviewable regression plan from a complete Git diff and product catalog.
//! Planning does not execute checks or authorize a release.
#[path = "regression_plan/native_artifacts.rs"]
mod native_artifacts;
#[path = "regression_plan/readme.rs"]
mod readme;
#[path = "regression_plan/scope.rs"]
mod scope;

#[cfg(test)]
use ferrum_bench_core::release_regression::analyze_paths;
use ferrum_bench_core::release_regression::contracts::contract_check_descriptors;
use ferrum_bench_core::release_regression::distribution::distribution_check_descriptors;
use ferrum_bench_core::release_regression::model_schedule::{
    model_check_descriptors, model_task_schedule,
};
use ferrum_bench_core::release_regression::performance::{
    performance_check_descriptors, performance_task_schedule,
};
use ferrum_bench_core::release_regression::submission::submission_check_descriptors;
use ferrum_bench_core::release_regression::{plan, Impact, PlanInput};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "regression_plan --catalog PATH --base REV [--candidate REV] \
    [--stage pull_request|release|nightly] [--repo PATH] [--output PATH] [--summary PATH]\n\
    Uses git diff --no-renames to include both sides of renames.\n\
    Release base must resolve to the latest reachable formal vMAJOR.MINOR.PATCH tag. Output is a plan, not passing evidence.\n\
    Missing coverage is recorded in gaps; it never authorizes promotion.";

#[derive(Debug, PartialEq, Eq)]
struct Args {
    catalog: PathBuf,
    base: String,
    candidate: String,
    stage: String,
    repo: PathBuf,
    output: Option<PathBuf>,
    summary: Option<PathBuf>,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<Args>, String> {
    let args: Vec<_> = args.into_iter().collect();
    if args == ["--help"] || args == ["-h"] {
        return Ok(None);
    }
    let mut values = BTreeMap::new();
    let mut iter = args.into_iter();
    while let Some(key) = iter.next() {
        if !matches!(
            key.as_str(),
            "--catalog"
                | "--base"
                | "--candidate"
                | "--stage"
                | "--repo"
                | "--output"
                | "--summary"
        ) {
            return Err(format!("unknown option {key:?}\n{USAGE}"));
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if value.is_empty() || value.starts_with("--") {
            return Err(format!("missing value for {key}"));
        }
        if values.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate option {key}"));
        }
    }
    let catalog = values.remove("--catalog").ok_or("--catalog is required")?;
    let base = values.remove("--base").ok_or("--base is required")?;
    let candidate = values
        .remove("--candidate")
        .unwrap_or_else(|| "HEAD".into());
    let stage = values.remove("--stage").unwrap_or_else(|| "release".into());
    if !matches!(stage.as_str(), "pull_request" | "release" | "nightly") {
        return Err(format!("invalid --stage {stage:?}"));
    }
    Ok(Some(Args {
        catalog: catalog.into(),
        base,
        candidate,
        stage,
        repo: values
            .remove("--repo")
            .map(PathBuf::from)
            .unwrap_or_else(|| ".".into()),
        output: values.remove("--output").map(PathBuf::from),
        summary: values.remove("--summary").map(PathBuf::from),
    }))
}

fn git(repo: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .args(args)
        .output()
        .map_err(|error| format!("run git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(output.stdout)
}

fn resolve_revision(repo: &Path, revision: &str) -> Result<String, String> {
    let expression = format!("{revision}^{{commit}}");
    let bytes = git(
        repo,
        &["rev-parse", "--verify", "--end-of-options", &expression],
    )?;
    let value = String::from_utf8(bytes).map_err(|_| "git returned non-UTF-8 revision")?;
    let value = value.trim();
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("git returned an invalid revision".into());
    }
    Ok(value.into())
}

fn changed_paths(bytes: &[u8]) -> Result<Vec<String>, String> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let bytes = bytes
        .strip_suffix(&[0])
        .ok_or("truncated NUL-separated Git diff")?;
    bytes
        .split(|byte| *byte == 0)
        .map(|path| {
            if path.is_empty() {
                return Err("empty path in Git diff".into());
            }
            String::from_utf8(path.to_vec())
                .map_err(|_| "non-UTF-8 changed path cannot be classified safely".into())
        })
        .collect()
}

fn changed_paths_between(repo: &Path, base: &str, candidate: &str) -> Result<Vec<String>, String> {
    changed_paths(&git(
        repo,
        &[
            "diff",
            "--no-ext-diff",
            "--no-renames",
            "--name-only",
            "-z",
            base,
            candidate,
            "--",
        ],
    )?)
}

fn plan_input(catalog: Value, stage: &str, impact: Impact) -> Result<PlanInput, String> {
    let mut input = catalog
        .as_object()
        .cloned()
        .ok_or("catalog must be an object")?;
    for key in input.keys() {
        if !matches!(
            key.as_str(),
            "profiles" | "quick_start_profile_ids" | "required_targets" | "checks"
        ) {
            return Err(format!("unknown catalog field {key:?}"));
        }
    }
    input.insert("stage".into(), json!(stage));
    input.insert(
        "impact".into(),
        serde_json::to_value(impact).map_err(|error| error.to_string())?,
    );
    let mut input: PlanInput = serde_json::from_value(Value::Object(input))
        .map_err(|error| format!("invalid product catalog: {error}"))?;
    for descriptor in model_check_descriptors()
        .into_iter()
        .chain(contract_check_descriptors())
        .chain(distribution_check_descriptors())
        .chain(submission_check_descriptors(&input.required_targets))
        .chain(performance_check_descriptors(&input.required_targets))
    {
        if input.checks.iter().any(|check| check.id == descriptor.id) {
            return Err(format!(
                "catalog cannot replace built-in checker {}",
                descriptor.id
            ));
        }
        input.checks.push(descriptor);
    }
    Ok(input)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("write {}: {error}", path.display()))
}

fn render_summary(document: &Value) -> String {
    let plan = &document["plan"];
    let count = |key: &str| plan[key].as_array().map_or(0, Vec::len);
    format!(
        "## Regression plan\n\nPlanning only; no model or GPU execution is certified.\n\n\
         Stage: `{}`. Required behaviors: {}. Selected profiles: {}. Unresolved gaps: {}.\n\n\
         Missing estimates remain unknown. Review scope, selections and gaps before allocating hardware.\n\n\
         <details><summary>Complete plan and provenance</summary>\n\n```json\n{}\n```\n\n</details>\n",
        document["stage"].as_str().unwrap_or("unknown"), count("obligations"), count("selected"), count("gaps"),
        serde_json::to_string_pretty(document).expect("JSON value serializes")
    )
}

fn run(args: Args) -> Result<(), String> {
    let base = resolve_revision(&args.repo, &args.base)?;
    let candidate = resolve_revision(&args.repo, &args.candidate)?;
    let release_base_tag = if args.stage == "release" {
        Some(scope::validate_release_base(&args.repo, &base, &candidate)?)
    } else {
        None
    };
    let paths = changed_paths_between(&args.repo, &base, &candidate)?;
    let mut analysis = scope::analyze(&args.repo, &base, &candidate, &paths);
    let catalog_bytes = fs::read(&args.catalog)
        .map_err(|error| format!("read {}: {error}", args.catalog.display()))?;
    let mut catalog = serde_json::from_slice(&catalog_bytes)
        .map_err(|error| format!("read product catalog JSON: {error}"))?;
    let reviews = readme::take_reviews(&mut catalog)?;
    let readme_reviews = readme::apply_reviews(&mut analysis.impact, &reviews, |path| {
        Ok((
            git(&args.repo, &["cat-file", "blob", &format!("{base}:{path}")])?,
            git(
                &args.repo,
                &["cat-file", "blob", &format!("{candidate}:{path}")],
            )?,
        ))
    })?;
    let native_artifact_refinement = if args.stage == "release" {
        native_artifacts::refine(&args.repo, &base, &candidate, &mut analysis.impact)
    } else {
        json!({"applied": false, "reason": "artifact host projection only applies to formal release plans"})
    };
    let input = plan_input(catalog, &args.stage, analysis.impact)?;
    let plan = plan(&input)?;
    let document = json!({
        "schema_version": 2,
        "stage": args.stage,
        "provenance": {"base": base, "candidate": candidate, "changed_paths": paths,
            "catalog_sha256": format!("{:x}", Sha256::digest(&catalog_bytes)),
            "release_base_tag": release_base_tag, "version_refinement": analysis.version_refinement,
            "dependency_refinement": analysis.dependency_refinement,
            "content_refinement": analysis.content_refinement,
            "native_artifact_refinement": native_artifact_refinement,
            "readme_reviews_applied": readme_reviews.applied,
            "readme_reviews_unmatched": readme_reviews.unmatched},
        "model_tasks": model_task_schedule(&plan),
        "performance_tasks": performance_task_schedule(&plan),
        "plan": plan,
    });
    let mut bytes = serde_json::to_vec_pretty(&document).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    if let Some(output) = &args.output {
        write_new(output, &bytes)?;
    } else {
        io::stdout()
            .write_all(&bytes)
            .map_err(|error| error.to_string())?;
    }
    if let Some(summary) = &args.summary {
        write_new(summary, render_summary(&document).as_bytes())?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_args(env::args().skip(1)) {
        Ok(None) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("regression plan: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("regression plan: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Args>, String> {
        parse_args(args.iter().map(|value| (*value).into()))
    }

    #[test]
    fn requires_explicit_release_base_and_catalog() {
        assert!(parse(&[]).is_err());
        assert!(parse(&["--catalog", "catalog.json"]).is_err());
        assert!(parse(&["--base", "v1"]).is_err());
        assert!(parse(&[
            "--catalog",
            "catalog.json",
            "--base",
            "v1",
            "--stage",
            "unknown"
        ])
        .is_err());
        let args = parse(&["--catalog", "catalog.json", "--base", "v1"])
            .unwrap()
            .unwrap();
        assert_eq!(args.stage, "release");
        assert_eq!(args.candidate, "HEAD");
    }

    #[test]
    fn rejects_ambiguous_or_unknown_arguments() {
        for args in [
            vec!["--catalog", "a", "--base", "v1", "--base", "v2"],
            vec!["--catalog", "--base", "v1"],
            vec!["--catalog", "a", "--base", "v1", "--skip-gaps", "yes"],
        ] {
            assert!(parse(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn nul_paths_preserve_whitespace_and_both_rename_sides() {
        assert_eq!(
            changed_paths(b"crates/old.rs\0docs/new name\n.rs\0").unwrap(),
            ["crates/old.rs", "docs/new name\n.rs"]
        );
        assert!(changed_paths(b"crates/source.rs").is_err());
        assert!(changed_paths(b"\xff\0").is_err());
        assert!(changed_paths(b"a\0\0").is_err());
        assert!(changed_paths(b"").unwrap().is_empty());
    }

    #[test]
    fn catalog_cannot_supply_a_precomputed_impact_or_success_claim() {
        for field in ["impact", "stage", "passed", "status"] {
            let mut catalog =
                json!({"profiles": [], "quick_start_profile_ids": [], "required_targets": []});
            catalog[field] = json!(true);
            assert!(
                plan_input(catalog, "release", analyze_paths(Vec::<String>::new())).is_err(),
                "{field}"
            );
        }
    }

    #[test]
    fn builtin_cpu_contracts_are_executable_bindings_and_cannot_be_shadowed() {
        let empty = json!({"profiles": [], "quick_start_profile_ids": [], "required_targets": []});
        let input = plan_input(
            empty.clone(),
            "pull_request",
            analyze_paths(Vec::<String>::new()),
        )
        .unwrap();
        for descriptor in contract_check_descriptors()
            .into_iter()
            .chain(distribution_check_descriptors())
        {
            assert!(input.checks.contains(&descriptor));
            let mut catalog = empty.clone();
            catalog["checks"] = json!([descriptor]);
            assert!(
                plan_input(catalog, "pull_request", analyze_paths(Vec::<String>::new()))
                    .unwrap_err()
                    .contains("cannot replace built-in checker")
            );
        }
    }

    #[test]
    fn catalog_cannot_replace_a_builtin_model_checker() {
        let mut check = serde_json::to_value(&model_check_descriptors()[0]).unwrap();
        check["entrypoints"] = json!([]);
        let catalog = json!({"profiles": [], "quick_start_profile_ids": [],
            "required_targets": [], "checks": [check]});
        assert!(
            plan_input(catalog, "pull_request", analyze_paths(Vec::<String>::new()))
                .unwrap_err()
                .contains("cannot replace built-in checker")
        );
    }

    struct GitFixture(PathBuf);

    impl GitFixture {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = env::temp_dir().join(format!(
                "ferrum-regression-plan-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir(&path).unwrap();
            let fixture = Self(path);
            fixture.git(&["init", "--quiet"]);
            fixture.git(&["config", "user.name", "Regression fixture"]);
            fixture.git(&["config", "user.email", "fixture@example.invalid"]);
            fixture.git(&["config", "commit.gpgsign", "false"]);
            fixture.git(&["config", "core.hooksPath", "/dev/null"]);
            fixture
        }

        fn git(&self, args: &[&str]) -> Vec<u8> {
            git(&self.0, args).unwrap()
        }

        fn commit(&self) -> String {
            self.git(&["add", "."]);
            self.git(&["commit", "--quiet", "-m", "fixture change"]);
            resolve_revision(&self.0, "HEAD").unwrap()
        }
    }

    impl Drop for GitFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn actual_git_diff_preserves_removed_source_and_all_release_commits() {
        let repo = GitFixture::new();
        fs::create_dir_all(repo.0.join("crates/ferrum-engine/src")).unwrap();
        fs::create_dir(repo.0.join("docs")).unwrap();
        let source = "crates/ferrum-engine/src/sequence.rs";
        fs::write(repo.0.join(source), "original source").unwrap();
        let base = repo.commit();
        fs::rename(repo.0.join(source), repo.0.join("docs/renamed source.md")).unwrap();
        repo.commit();
        fs::write(repo.0.join("docs/later.md"), "another release change").unwrap();
        let candidate = repo.commit();
        let paths = changed_paths_between(&repo.0, &base, &candidate).unwrap();
        assert!(
            paths.iter().any(|path| path == source),
            "removed code is still in the release diff"
        );
        assert!(paths.iter().any(|path| path == "docs/renamed source.md"));
        assert!(paths.iter().any(|path| path == "docs/later.md"));
        let impact = serde_json::to_value(analyze_paths(&paths)).unwrap();
        let docs_only = serde_json::to_value(analyze_paths(["docs/later.md"])).unwrap();
        assert_ne!(
            impact, docs_only,
            "renaming source into docs must retain code impact"
        );
        assert!(resolve_revision(&repo.0, "missing-release").is_err());
        assert!(changed_paths_between(&repo.0, &candidate, &candidate)
            .unwrap()
            .is_empty());
    }

    fn workspace_fixture(repo: &GitFixture) {
        fs::create_dir_all(repo.0.join("crates/ferrum-engine/src")).unwrap();
        fs::write(
            repo.0.join("Cargo.toml"),
            r#"[workspace]
members = ["crates/ferrum-engine"]
resolver = "2"
[workspace.package]
version = "1.2.3"
[workspace.dependencies]
ferrum-engine = { path = "crates/ferrum-engine", version = "1.2.3" }
"#,
        )
        .unwrap();
        fs::write(
            repo.0.join("crates/ferrum-engine/Cargo.toml"),
            "[package]\nname = \"ferrum-engine\"\nversion.workspace = true\nedition = \"2021\"\n",
        )
        .unwrap();
        fs::write(
            repo.0.join("crates/ferrum-engine/src/lib.rs"),
            "pub fn forward() {}\n",
        )
        .unwrap();
        fs::write(
            repo.0.join("Cargo.lock"),
            "version = 4\n[[package]]\nname = \"ferrum-engine\"\nversion = \"1.2.3\"\n",
        )
        .unwrap();
    }

    fn bump_fixture(repo: &GitFixture) {
        use ferrum_bench_core::release_candidate::version::{plan_version_update, MemberManifest};
        let members = [MemberManifest {
            package_name: "ferrum-engine".into(),
            manifest_path: "crates/ferrum-engine/Cargo.toml".into(),
            text: fs::read_to_string(repo.0.join("crates/ferrum-engine/Cargo.toml")).unwrap(),
        }];
        let plan = plan_version_update(
            &fs::read_to_string(repo.0.join("Cargo.toml")).unwrap(),
            &members,
            &fs::read_to_string(repo.0.join("Cargo.lock")).unwrap(),
            "1.2.4",
        )
        .unwrap();
        fs::write(repo.0.join("Cargo.toml"), plan.workspace_manifest).unwrap();
        fs::write(repo.0.join("Cargo.lock"), plan.lockfile).unwrap();
        for member in plan.members {
            fs::write(repo.0.join(member.manifest_path), member.text).unwrap();
        }
    }

    #[test]
    fn version_refinement_reads_git_content_and_retains_build_validation() {
        use ferrum_bench_core::release_regression::ChangeArea;
        let repo = GitFixture::new();
        workspace_fixture(&repo);
        let base = repo.commit();
        bump_fixture(&repo);
        let candidate = repo.commit();
        let paths = changed_paths_between(&repo.0, &base, &candidate).unwrap();
        // Current checkout edits must not change the declared Git comparison.
        fs::write(repo.0.join("Cargo.toml"), "uncommitted unrelated text").unwrap();
        let analysis = scope::analyze(&repo.0, &base, &candidate, &paths);
        assert_eq!(analysis.version_refinement["applied"], true);
        assert!(analysis.impact.areas.contains(&ChangeArea::Build));
        for unrelated in [
            ChangeArea::Kernel,
            ChangeArea::Architecture,
            ChangeArea::Termination,
        ] {
            assert!(!analysis.impact.areas.contains(&unrelated));
        }
    }

    #[test]
    fn version_refinement_cannot_hide_earlier_runtime_changes_or_added_build_flags() {
        use ferrum_bench_core::release_regression::ChangeArea;
        let repo = GitFixture::new();
        workspace_fixture(&repo);
        let base = repo.commit();
        repo.git(&["tag", "-a", "v1.2.3", "-m", "formal release"]);
        fs::write(
            repo.0.join("crates/ferrum-engine/src/lib.rs"),
            "pub fn forward() { panic!(\"regression\"); }\n",
        )
        .unwrap();
        let only_last_pr = repo.commit();
        repo.git(&["tag", "v1.2.4-rc.1"]);
        bump_fixture(&repo);
        let candidate = repo.commit();
        let paths = changed_paths_between(&repo.0, &base, &candidate).unwrap();
        let analysis = scope::analyze(&repo.0, &base, &candidate, &paths);
        assert_eq!(analysis.version_refinement["applied"], true);
        assert!(analysis.impact.areas.contains(&ChangeArea::Kernel));
        assert!(analysis.impact.areas.contains(&ChangeArea::Termination));
        assert!(scope::validate_release_base(&repo.0, &only_last_pr, &candidate).is_err());
        assert_eq!(
            scope::validate_release_base(&repo.0, &base, &candidate).unwrap(),
            "v1.2.3"
        );
        let manifest = repo.0.join("Cargo.toml");
        let mut text = fs::read_to_string(&manifest).unwrap();
        text.push_str("\n[profile.release]\nopt-level = 1\n");
        fs::write(manifest, text).unwrap();
        let changed_flags = repo.commit();
        let paths = changed_paths_between(&repo.0, &base, &changed_flags).unwrap();
        let analysis = scope::analyze(&repo.0, &base, &changed_flags, &paths);
        assert_eq!(analysis.version_refinement["applied"], false);
        let root = analysis
            .impact
            .paths
            .iter()
            .find(|entry| entry.path == "Cargo.toml")
            .unwrap();
        assert!(root.areas.contains(&ChangeArea::Kernel));
        assert!(root.areas.contains(&ChangeArea::Architecture));
    }

    #[test]
    fn release_base_uses_reachable_formal_tags_and_rejects_truncated_intervals() {
        let repo = GitFixture::new();
        fs::write(repo.0.join("source"), "first release").unwrap();
        let older = repo.commit();
        repo.git(&["tag", "v1.2.3"]);
        fs::write(repo.0.join("source"), "next release").unwrap();
        let previous = repo.commit();
        repo.git(&["tag", "v1.3.0"]);
        fs::write(repo.0.join("source"), "candidate").unwrap();
        let candidate = repo.commit();
        repo.git(&["tag", "v1.4.0-rc.1"]);
        // A future/unreachable tag and component asset tags are not the base.
        fs::write(repo.0.join("future"), "separate future commit").unwrap();
        repo.commit();
        repo.git(&["tag", "v99.0.0"]);
        repo.git(&["tag", "ferrum-native-cuda12.4-sm89-v6"]);
        assert_eq!(
            scope::validate_release_base(&repo.0, &previous, &candidate).unwrap(),
            "v1.3.0"
        );
        assert!(scope::validate_release_base(&repo.0, &older, &candidate).is_err());
        assert!(scope::validate_release_base(&repo.0, &candidate, &candidate).is_err());
    }

    #[test]
    fn generated_release_plan_exposes_a_formal_tag_name_for_delivery_consumers() {
        use ferrum_bench_core::release_candidate::staging::validate_version_progression;
        use ferrum_bench_core::release_regression::{Backend, ExecutionTarget, ModelProfile};
        use ferrum_types::{ModelOutputProtocol, ModelReasoningProtocol};
        let repo = GitFixture::new();
        workspace_fixture(&repo);
        let base = repo.commit();
        repo.git(&["tag", "-a", "v1.2.3", "-m", "formal release"]);
        bump_fixture(&repo);
        let candidate = repo.commit();
        // A branch with the same short name must not change the verified base.
        repo.git(&["branch", "v1.2.3", &candidate]);
        let profile = ModelProfile {
            id: "quick-start".into(),
            model: "fixture".into(),
            target: ExecutionTarget {
                architecture: "dense".into(),
                protocol: ModelOutputProtocol::Text,
                precision: "f32".into(),
                backend: Backend::Cpu,
                execution_path: "production-plan-runtime".into(),
            },
            available: true,
            estimate: None,
            reasoning_protocol: ModelReasoningProtocol::None,
        };
        let targets =
            [Backend::Cpu, Backend::Metal, Backend::Cuda].map(|backend| ExecutionTarget {
                backend,
                ..profile.target.clone()
            });
        let catalog = repo.0.join("catalog.json");
        fs::write(
            &catalog,
            serde_json::to_vec(&json!({
                "profiles": [profile], "quick_start_profile_ids": ["quick-start"],
                "required_targets": targets
            }))
            .unwrap(),
        )
        .unwrap();
        let output = repo.0.join("release-plan.json");
        run(Args {
            catalog,
            base: "refs/tags/v1.2.3".into(),
            candidate: candidate.clone(),
            stage: "release".into(),
            repo: repo.0.clone(),
            output: Some(output.clone()),
            summary: None,
        })
        .unwrap();
        let document: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        let tag = document["provenance"]["release_base_tag"].as_str().unwrap();
        assert_eq!(tag, "v1.2.3");
        // Consume the generated document using the delivery version contract.
        validate_version_progression(tag.strip_prefix('v').unwrap(), "1.2.4").unwrap();
        assert_eq!(
            resolve_revision(&repo.0, &format!("refs/tags/{tag}")).unwrap(),
            base
        );
        assert_eq!(document["provenance"]["base"], base);
        assert_eq!(document["provenance"]["candidate"], candidate);
    }

    #[test]
    fn content_scope_uses_committed_rust_and_preserves_actual_metal_changes() {
        use ferrum_bench_core::release_regression::ChangeArea;
        let repo = GitFixture::new();
        fs::create_dir_all(repo.0.join("crates/ferrum-cli/src/commands")).unwrap();
        fs::create_dir_all(repo.0.join("crates/ferrum-kernels/src/backend/metal")).unwrap();
        let run = "crates/ferrum-cli/src/commands/run.rs";
        let metal = "crates/ferrum-kernels/src/backend/metal/mod.rs";
        fs::write(repo.0.join(run), "pub fn run() {}\n").unwrap();
        fs::write(
            repo.0.join(metal),
            "pub fn synchronize() -> bool { true }\n",
        )
        .unwrap();
        let base = repo.commit();
        fs::write(repo.0.join(run), "pub fn run() {}\n#[cfg(test)] mod tests { #[test] fn boundary() { assert_eq!(1 + 1, 2); } }\n").unwrap();
        let test_candidate = repo.commit();
        let paths = changed_paths_between(&repo.0, &base, &test_candidate).unwrap();
        let analysis = scope::analyze(&repo.0, &base, &test_candidate, &paths);
        assert_eq!(analysis.impact.areas, [ChangeArea::Validation]);
        assert_eq!(
            analysis.content_refinement["rust_validation_paths"],
            json!([run])
        );
        fs::write(
            repo.0.join(metal),
            "pub fn synchronize() -> bool { false }\n",
        )
        .unwrap();
        let candidate = repo.commit();
        // Uncommitted text must not replace the candidate's production AST.
        fs::write(
            repo.0.join(run),
            "pub fn run() { panic!(\"dirty unrelated source\"); }\n",
        )
        .unwrap();
        let paths = changed_paths_between(&repo.0, &base, &candidate).unwrap();
        let analysis = scope::analyze(&repo.0, &base, &candidate, &paths);
        assert!(analysis.impact.areas.contains(&ChangeArea::Kernel));
        assert!(!analysis.impact.areas.contains(&ChangeArea::Architecture));
        assert!(analysis.impact.areas.contains(&ChangeArea::Validation));
        assert!(!analysis.impact.areas.contains(&ChangeArea::Download));
    }

    #[test]
    fn dev_dependency_git_snapshot_refinement_does_not_infer_changed_kernels() {
        use ferrum_bench_core::release_regression::ChangeArea;
        let repo = GitFixture::new();
        workspace_fixture(&repo);
        let base = repo.commit();
        let member = repo.0.join("crates/ferrum-engine/Cargo.toml");
        let mut text = fs::read_to_string(&member).unwrap();
        text.push_str("[dev-dependencies]\nfixture_test = '1'\n");
        fs::write(member, text).unwrap();
        let lock = repo.0.join("Cargo.lock");
        let mut text = fs::read_to_string(&lock).unwrap();
        text.push_str("dependencies = ['fixture_test']\n[[package]]\nname = 'fixture_test'\nversion = '1.0.0'\nsource = 'registry+https://example.invalid/index'\nchecksum = 'fixture'\n");
        fs::write(lock, text).unwrap();
        let candidate = repo.commit();
        let paths = changed_paths_between(&repo.0, &base, &candidate).unwrap();
        let analysis = scope::analyze(&repo.0, &base, &candidate, &paths);
        assert_eq!(
            analysis.dependency_refinement["applied"], true,
            "{:?}",
            analysis.dependency_refinement
        );
        assert_eq!(analysis.impact.areas, [ChangeArea::Validation]);
    }

    #[test]
    fn private_tool_manifest_only_change_cannot_keep_validation_scope_when_isolation_fails() {
        use ferrum_bench_core::release_regression::ChangeArea;
        let repo = GitFixture::new();
        workspace_fixture(&repo);
        let root_path = repo.0.join("Cargo.toml");
        let root = fs::read_to_string(&root_path).unwrap().replace(
            "members = [\"crates/ferrum-engine\"]",
            "members = [\"crates/ferrum-engine\", \"crates/ferrum-devtools\"]",
        );
        fs::write(root_path, root).unwrap();
        fs::create_dir_all(repo.0.join("crates/ferrum-devtools")).unwrap();
        let path = "crates/ferrum-devtools/Cargo.toml";
        let manifest = "[package]\nname='ferrum-devtools'\nversion.workspace=true\npublish=false\n[[bin]]\nname='release_delivery'\npath='src/bin/release_delivery.rs'\n[[bin]]\nname='contract_checks'\npath='src/bin/contract_checks.rs'\n";
        fs::write(repo.0.join(path), manifest).unwrap();
        let lock_path = repo.0.join("Cargo.lock");
        let mut lock = fs::read_to_string(&lock_path).unwrap();
        lock.push_str("[[package]]\nname='ferrum-devtools'\nversion='1.2.3'\n");
        fs::write(lock_path, lock).unwrap();
        let base = repo.commit();
        fs::write(
            repo.0.join(path),
            manifest.replace("publish=false", "publish=true"),
        )
        .unwrap();
        let candidate = repo.commit();
        let paths = changed_paths_between(&repo.0, &base, &candidate).unwrap();
        assert_eq!(paths, [path]);
        let analysis = scope::analyze(&repo.0, &base, &candidate, &paths);
        assert_eq!(analysis.dependency_refinement["applied"], false);
        assert_eq!(analysis.impact.areas, ChangeArea::ALL);
    }

    #[test]
    fn readme_review_uses_immutable_git_content_and_rejects_a_later_claim() {
        use ferrum_bench_core::release_regression::{Backend, ExecutionTarget, ModelProfile};
        use ferrum_types::{ModelOutputProtocol, ModelReasoningProtocol};
        let repo = GitFixture::new();
        let before = "# Product\nInstall the package.\n";
        let after = "# Product\nInstall the package and use its API.\n";
        fs::write(repo.0.join("README.md"), before).unwrap();
        let base = repo.commit();
        fs::write(repo.0.join("README.md"), after).unwrap();
        let candidate = repo.commit();
        let review = readme::ReadmeReview {
            path: "README.md".into(),
            before_sha256: format!("{:x}", Sha256::digest(before.as_bytes())),
            after_sha256: format!("{:x}", Sha256::digest(after.as_bytes())),
            areas: vec![ferrum_bench_core::release_regression::ChangeArea::Build],
            rationale: "Reviewed the installation description.".into(),
        };
        let profile = ModelProfile {
            id: "quick-start".into(),
            model: "fixture".into(),
            target: ExecutionTarget {
                architecture: "dense".into(),
                protocol: ModelOutputProtocol::Text,
                precision: "f32".into(),
                backend: Backend::Cpu,
                execution_path: "production-plan-runtime".into(),
            },
            available: true,
            estimate: None,
            reasoning_protocol: ModelReasoningProtocol::None,
        };
        let targets =
            [Backend::Cpu, Backend::Metal, Backend::Cuda].map(|backend| ExecutionTarget {
                backend,
                ..profile.target.clone()
            });
        let catalog = repo.0.join("catalog.json");
        fs::write(
            &catalog,
            serde_json::to_vec(&json!({
                "profiles": [profile], "quick_start_profile_ids": ["quick-start"], "required_targets": targets,
                "readme_reviews": [review]
            }))
            .unwrap(),
        )
        .unwrap();
        // Worktree edits cannot replace the candidate's reviewed blob.
        fs::write(
            repo.0.join("README.md"),
            "Supports an unreviewed new model.",
        )
        .unwrap();
        let output = repo.0.join("plan.json");
        run(Args {
            catalog: catalog.clone(),
            base: base.clone(),
            candidate,
            stage: "pull_request".into(),
            repo: repo.0.clone(),
            output: Some(output.clone()),
            summary: None,
        })
        .unwrap();
        let document: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(
            document["provenance"]["readme_reviews_applied"],
            json!([review])
        );
        assert_eq!(
            document["plan"]["impact"]["product_contract_changed"],
            false
        );
        assert!(
            !document["plan"]["gaps"].as_array().unwrap().is_empty(),
            "documentation review does not supply missing execution coverage"
        );
        // Once the changed claim is committed, the previous review is stale.
        fs::remove_file(&output).unwrap();
        let later = repo.commit();
        run(Args {
            catalog,
            base,
            candidate: later,
            stage: "pull_request".into(),
            repo: repo.0.clone(),
            output: Some(output.clone()),
            summary: None,
        })
        .unwrap();
        let later: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(later["plan"]["impact"]["product_contract_changed"], true);
        assert_eq!(later["provenance"]["readme_reviews_applied"], json!([]));
        assert_eq!(
            later["provenance"]["readme_reviews_unmatched"][0]["path"],
            "README.md"
        );
        assert!(later["plan"]["gaps"]
            .as_array()
            .unwrap()
            .contains(&json!({"kind":"product_contract_review"})));
    }

    #[test]
    fn homebrew_only_readme_changes_require_installation_while_new_models_still_require_review() {
        use ferrum_bench_core::release_regression::ChangeArea;
        let repo = GitFixture::new();
        let before = "# Product\n## Quick Start\nInstall Ferrum:\n```bash\nbrew install ferrum\n```\n```bash\nferrum run stable-model\n```\n## Installation\nHomebrew:\n```bash\nbrew install ferrum\n```\n";
        fs::write(repo.0.join("README.md"), before).unwrap();
        let base = repo.commit();
        let installation = before.replace(
            "brew install ferrum",
            "brew trust --formula owner/project/ferrum\nbrew install ferrum",
        );
        fs::write(repo.0.join("README.md"), &installation).unwrap();
        let candidate = repo.commit();
        let paths = changed_paths_between(&repo.0, &base, &candidate).unwrap();
        let analysis = scope::analyze(&repo.0, &base, &candidate, &paths);
        assert_eq!(analysis.impact.areas, [ChangeArea::Build]);
        assert!(!analysis.impact.product_contract_changed);
        fs::write(
            repo.0.join("README.md"),
            installation.replace("stable-model", "new-model"),
        )
        .unwrap();
        let new_model = repo.commit();
        let paths = changed_paths_between(&repo.0, &base, &new_model).unwrap();
        assert!(
            scope::analyze(&repo.0, &base, &new_model, &paths)
                .impact
                .product_contract_changed
        );
    }
}
