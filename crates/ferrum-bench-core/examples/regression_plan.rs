//! Generate a reviewable regression plan from a complete Git diff and product catalog.
//! Planning does not execute checks or authorize a release.
use ferrum_bench_core::release_regression::{analyze_paths, plan, PlanInput};
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
    Release base must be the previous formal release. Output is a plan, not passing evidence.\n\
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

fn plan_input(catalog: Value, stage: &str, paths: &[String]) -> Result<PlanInput, String> {
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
        serde_json::to_value(analyze_paths(paths)).map_err(|error| error.to_string())?,
    );
    serde_json::from_value(Value::Object(input))
        .map_err(|error| format!("invalid product catalog: {error}"))
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
    let paths = changed_paths_between(&args.repo, &base, &candidate)?;
    let catalog_bytes = fs::read(&args.catalog)
        .map_err(|error| format!("read {}: {error}", args.catalog.display()))?;
    let catalog = serde_json::from_slice(&catalog_bytes)
        .map_err(|error| format!("read product catalog JSON: {error}"))?;
    let input = plan_input(catalog, &args.stage, &paths)?;
    let plan = plan(&input)?;
    let document = json!({
        "schema_version": 1,
        "stage": args.stage,
        "provenance": {"base": base, "candidate": candidate, "changed_paths": paths,
            "catalog_sha256": format!("{:x}", Sha256::digest(&catalog_bytes))},
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
            assert!(plan_input(catalog, "release", &[]).is_err(), "{field}");
        }
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
}
