//! Release-artifact reach, not a proof that external CUDA source is equivalent.
//! The caller invokes this only for release plans; PR/nightly keep path impact.
use ferrum_bench_core::release_regression::{
    source_change::rust_validation_only, ChangeArea, Impact,
};
use quote::ToTokens;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

const WORKFLOW: &str = ".github/workflows/release-cuda.yml";
const WINDOWS_WORKFLOW: &str = ".github/workflows/release-windows.yml";
const DELIVERY: &str = ".github/workflows/release-delivery.yml";
const BUNDLE: &str = "native-operators/cuda/source-bundles/ferrum-native-cuda-v1.json";
const NATIVE: &str = "crates/ferrum-kernels/src/native_ops.rs";
const BUILDER: &str = "crates/ferrum-native-ops-builder/src/source_build.rs";
const CI: &str = ".github/workflows/ci.yml";
const LINUX: &str = "linux-x86_64-cuda-sm89";
const WINDOWS: &str = "windows-x86_64-cuda-sm89";
const OPERATORS: &[&str] = &[
    "marlin",
    "vllm-marlin",
    "vllm-moe-marlin",
    "vllm-paged-attention-v2",
];
const CONSUMERS: &[&str] = &[
    "crates/ferrum-kernels/src/backend/cuda/vnext_ops/transformer/causal_attention.rs",
    "crates/ferrum-kernels/src/backend/cuda/vnext_ops/transformer/gpt_oss_moe.rs",
    "crates/ferrum-kernels/src/backend/cuda/vnext_ops/transformer/moe.rs",
    "crates/ferrum-kernels/src/backend/cuda/vnext_ops/transformer/moe_routed.rs",
];
const NEEDLES: &[&str] = &[
    "CUDA_NATIVE_SOURCE_BUNDLE_ID",
    "native-operators/cuda/",
    "ferrum-native-cuda-sources-",
];

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn relative(value: &str) -> bool {
    !value.is_empty()
        && !value.contains(['\\', '\0', '\n', '\r', ':'])
        && value.split('/').all(|p| !matches!(p, "" | "." | ".."))
}
fn source_paths() -> BTreeSet<String> {
    std::iter::once(BUNDLE.to_string())
        .chain(OPERATORS.iter().flat_map(|operator| {
            [
                format!("native-operators/cuda/source-definitions/{operator}.json"),
                format!("native-operators/cuda/source-locks/{operator}.plan.json"),
            ]
        }))
        .collect()
}

#[derive(Clone, Default)]
struct Snapshot {
    files: BTreeMap<String, String>,
    consumers: BTreeSet<String>,
}
impl Snapshot {
    fn get(&self, path: &str) -> Result<&str, String> {
        self.files
            .get(path)
            .map(String::as_str)
            .ok_or_else(|| format!("missing immutable input {path}"))
    }
}

fn read(repo: &Path, revision: &str, path: &str) -> Result<String, String> {
    String::from_utf8(super::git(
        repo,
        &["cat-file", "blob", &format!("{revision}:{path}")],
    )?)
    .map_err(|_| format!("non-UTF-8 artifact-scope input {path}"))
}
fn snapshot(repo: &Path, revision: &str) -> Result<Snapshot, String> {
    // Git grep reads this commit, never the working tree. The known bundle
    // declaration guarantees at least one result in supported repositories.
    let found = super::git(
        repo,
        &[
            "grep", "-l", "-z", "-e", NEEDLES[0], "-e", NEEDLES[1], "-e", NEEDLES[2], revision,
            "--", "*.rs", "*.yml", "*.yaml", "*.sh", "*.ps1", "*.toml", "*.cu", "*.cuh", "*.h",
            "*.hpp", "*.cpp",
        ],
    )?;
    let mut consumers = BTreeSet::new();
    for name in found.split(|b| *b == 0).filter(|name| !name.is_empty()) {
        let name = std::str::from_utf8(name).map_err(|_| "non-UTF-8 Git consumer path")?;
        let path = name
            .strip_prefix(&format!("{revision}:"))
            .ok_or("Git consumer has another revision")?;
        if !relative(path) {
            return Err("invalid native consumer path".into());
        }
        // These are planner/validation targets, not an inference artifact.
        if matches!(
            path,
            "crates/ferrum-bench-core/src/release_regression/impact.rs"
                | "crates/ferrum-bench-core/examples/regression_plan/native_artifacts.rs"
                | "crates/ferrum-kernels/tests/native_ops.rs"
        ) {
            continue;
        }
        consumers.insert(path.to_string());
    }
    let paths: BTreeSet<_> = source_paths()
        .into_iter()
        .chain([
            WORKFLOW.to_string(),
            WINDOWS_WORKFLOW.to_string(),
            DELIVERY.to_string(),
            CI.to_string(),
            NATIVE.to_string(),
            BUILDER.to_string(),
        ])
        .chain(CONSUMERS.iter().map(|path| (*path).to_string()))
        .chain(consumers.iter().cloned())
        .collect();
    let files = paths
        .into_iter()
        .map(|path| Ok((path.clone(), read(repo, revision, &path)?)))
        .collect::<Result<_, String>>()?;
    Ok(Snapshot { files, consumers })
}

fn yaml(source: &str) -> Result<Value, String> {
    serde_yaml::from_str(source).map_err(|e| format!("release workflow YAML: {e}"))
}
fn mandatory_steps(job: &Value) -> Result<&[Value], String> {
    if job.get("if").is_some() || job.get("continue-on-error").is_some_and(|v| v != false) {
        return Err("artifact producer must not be optional or tolerate failure".into());
    }
    job["steps"]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| "artifact producer has no steps".into())
}
fn step<'a>(job: &'a Value, name: &str) -> Result<&'a str, String> {
    let mut matches = mandatory_steps(job)?.iter().filter(|s| s["name"] == name);
    let value = matches
        .next()
        .ok_or_else(|| format!("missing mandatory staging step {name}"))?;
    if matches.next().is_some()
        || value.get("if").is_some()
        || value.get("continue-on-error").is_some_and(|v| v != false)
    {
        return Err(format!("ambiguous or optional staging step {name}"));
    }
    value["run"]
        .as_str()
        .ok_or_else(|| format!("staging step {name} has no command"))
}
fn require_fragments(script: &str, fragments: &[&str]) -> Result<(), String> {
    if fragments.iter().any(|fragment| !script.contains(fragment)) {
        return Err("reviewed artifact consumption command changed".into());
    }
    Ok(())
}

#[path = "native_artifacts/topology.rs"]
mod topology;
use topology::producer_jobs;

fn recipes(before: &Snapshot, after: &Snapshot) -> Result<Value, String> {
    // Source-input reuse is justified only under an unchanged, resolved
    // producer/consumer graph. A topology migration must retain kernel scope;
    // do not infer PowerShell or Actions expression equivalence from similar names.
    for path in [WORKFLOW, WINDOWS_WORKFLOW, DELIVERY] {
        if yaml(before.get(path)?)? != yaml(after.get(path)?)? {
            return Err(format!(
                "artifact producer or consumer workflow changed: {path}"
            ));
        }
    }
    let delivery = yaml(after.get(DELIVERY)?)?;
    let windows_workflow = yaml(after.get(WINDOWS_WORKFLOW)?)?;
    let (linux, windows) =
        producer_jobs(&yaml(after.get(WORKFLOW)?)?, &windows_workflow, &delivery)?;
    let before = yaml(before.get(WORKFLOW)?)?;
    let after = yaml(after.get(WORKFLOW)?)?;
    for key in ["env", "defaults"] {
        if before.get(key) != after.get(key) {
            return Err(format!("global release {key} changed"));
        }
    }
    for trigger in ["workflow_call", "workflow_dispatch"] {
        let inputs = before["on"][trigger]["inputs"]
            .as_object()
            .ok_or("release inputs missing")?;
        for (name, value) in inputs {
            if after["on"][trigger]["inputs"].get(name) != Some(value) {
                return Err(format!("Linux release input {trigger}/{name} changed"));
            }
        }
        let current = after["on"][trigger]["inputs"]
            .as_object()
            .ok_or("release inputs missing")?;
        if before["jobs"].get(WINDOWS).is_some() {
            if inputs != current {
                return Err("existing Windows staging input contract changed".into());
            }
        } else if current.keys().any(|name| {
            !inputs.contains_key(name)
                && !matches!(
                    name.as_str(),
                    "windows_launcher_url" | "windows_launcher_sha256"
                )
        }) {
            return Err("unreviewed new staging input".into());
        }
    }
    if before["jobs"].get(LINUX) != after["jobs"].get(LINUX) || linux.is_null() {
        return Err("Linux artifact recipe/toolchain changed".into());
    }
    mandatory_steps(&linux)?;
    let url = after["env"]["NATIVE_OPERATOR_SET_ARCHIVE_URL"]
        .as_str()
        .ok_or("native archive URL absent")?;
    let hash = after["env"]["NATIVE_OPERATOR_SET_ARCHIVE_SHA256"]
        .as_str()
        .ok_or("native archive SHA absent")?;
    if !url.starts_with("https://github.com/")
        || !url.contains("/releases/download/")
        || url.contains(['$', '?', '#'])
        || !sha(hash)
    {
        return Err("Linux native archive is not pinned by URL and SHA256".into());
    }
    let materialize = step(&linux, "Materialize pinned CUDA native operator set")?;
    require_fragments(
        materialize,
        &[
            "$NATIVE_OPERATOR_SET_ARCHIVE_URL",
            "$NATIVE_OPERATOR_SET_ARCHIVE_SHA256",
            "sha256sum --check --strict",
            "inputs/native-operator-set.lock.json",
            "FERRUM_NATIVE_OPERATOR_SET_LOCK=$lock",
        ],
    )?;
    let linux_text = serde_json::to_string(&linux).unwrap();
    if NEEDLES.iter().any(|needle| linux_text.contains(needle))
        || linux_text.contains("source-build")
        || linux_text.contains("ferrum-native-ops-builder")
    {
        return Err("Linux artifact now consumes native source inputs".into());
    }
    require_fragments(
        step(&linux, "Build release CUDA sm89 binary exactly once")?,
        &[
            "$FERRUM_NATIVE_OPERATOR_SET_LOCK",
            "cargo build --release --locked -p ferrum-cli --bin ferrum",
            "cuda,vllm-moe-marlin,vllm-paged-attn-v2",
        ],
    )?;
    require_fragments(
        step(&windows, "Build the four native CUDA operators with MSVC")?,
        &[
            "$ErrorActionPreference = 'Stop'",
            "$PSNativeCommandUseErrorActionPreference = $true",
            BUNDLE,
            "$distribution.archive.sha256",
            "& $builder lock-source",
            "& $builder source-build",
            "& $builder package",
            "& $builder assemble-set",
            "FERRUM_NATIVE_OPERATOR_SET_LOCK=$lock",
            "if ($failedOperators.Count -gt 0)",
            "throw (\"Native CUDA operator builds failed:",
        ],
    )?;
    require_fragments(
        step(
            &windows,
            "Build the selected Windows executable exactly once",
        )?,
        &[
            "FERRUM_NATIVE_OPERATOR_SET_LOCK",
            "cargo build --release --locked -p ferrum-cli --bin ferrum",
            "cuda,vllm-moe-marlin,vllm-paged-attn-v2",
        ],
    )?;
    require_fragments(
        step(
            &windows,
            "Pack the exact release EXE and redistributable runtime bytes",
        )?,
        &[
            "portable pack",
            "portable inspect",
            "--extract-only",
            "staged_bytes_only",
            "x86_64-pc-windows-msvc",
        ],
    )?;
    Ok(
        json!({"linux_job": LINUX, "linux_job_sha256": digest(linux_text.as_bytes()),
        "linux_native_archive": {"url": url, "sha256": hash},
        "linux_toolchain": linux["container"], "global_env": after["env"], "global_defaults": after["defaults"],
        "windows_job": WINDOWS, "windows_ci_job": "stage-cuda / Stage Windows x86_64 CUDA sm89",
        "windows_recipe_check": "unchanged local caller, callee and delivery bindings; literal CUDA backend",
        "windows_job_sha256": digest(serde_json::to_string(&windows).unwrap().as_bytes()),
        "windows_workflow_sha256": digest(serde_json::to_string(&windows_workflow).unwrap().as_bytes()),
        "delivery_workflow_sha256": digest(serde_json::to_string(&delivery).unwrap().as_bytes())}),
    )
}

fn bundle(snapshot: &Snapshot) -> Result<Value, String> {
    let value: Value = serde_json::from_str(snapshot.get(BUNDLE)?)
        .map_err(|e| format!("native descriptor: {e}"))?;
    let members = value["members"]
        .as_array()
        .ok_or("native source members absent")?;
    let mut paths = Vec::new();
    for member in members {
        let path = member["path"].as_str().ok_or("native member path absent")?;
        if !relative(path)
            || !member["sha256"].as_str().is_some_and(sha)
            || member["size_bytes"].as_u64().is_none_or(|size| size == 0)
        {
            return Err("invalid native member identity".into());
        }
        paths.push(path);
    }
    let member_sha = digest(&serde_json::to_vec(members).unwrap());
    if paths.is_empty()
        || paths.windows(2).any(|p| p[0] >= p[1])
        || value["schema_version"] != 1
        || value["member_set_sha256"] != member_sha
        || value["bundle_id"] != format!("ferrum-native-cuda-v1+sha256.{member_sha}")
        || !value["archive"]["sha256"].as_str().is_some_and(sha)
        || value["archive"]["size_bytes"]
            .as_u64()
            .is_none_or(|size| size == 0)
        || value["archive"]["file_name"] != value["distribution"]["asset_name"]
    {
        return Err("native descriptor identities are inconsistent".into());
    }
    for operator in OPERATORS {
        for (directory, suffix, field) in [
            ("source-definitions", "json", "source_package_revision"),
            ("source-locks", "plan.json", "source_package"),
        ] {
            let path = format!("native-operators/cuda/{directory}/{operator}.{suffix}");
            let input: Value = serde_json::from_str(snapshot.get(&path)?)
                .map_err(|e| format!("native input {path}: {e}"))?;
            let revision = if field == "source_package" {
                &input[field]["revision"]
            } else {
                &input[field]
            };
            if revision != &value["bundle_id"] {
                return Err(format!("native input {path} binds another source bundle"));
            }
        }
    }
    Ok(value)
}

fn identity_only(
    before: &str,
    after: &str,
    old_id: &Value,
    new_id: &Value,
) -> Result<bool, String> {
    fn normalized(source: &str, expected: &Value) -> Result<String, String> {
        let mut file = syn::parse_file(source).map_err(|e| e.to_string())?;
        let mut count = 0;
        for item in &mut file.items {
            if let syn::Item::Const(item) = item {
                if item.ident != "CUDA_NATIVE_SOURCE_BUNDLE_ID" {
                    continue;
                }
                count += 1;
                if !matches!(&*item.expr, syn::Expr::Lit(value) if matches!(&value.lit, syn::Lit::Str(value) if Some(value.value().as_str()) == expected.as_str()))
                {
                    return Err("live native bundle identity differs from source descriptor".into());
                }
                item.expr = Box::new(syn::parse_str("\"artifact-source-identity\"").unwrap());
            }
        }
        if count != 1 {
            return Err("ambiguous native bundle declaration".into());
        }
        Ok(file.into_token_stream().to_string())
    }
    rust_validation_only(&normalized(before, old_id)?, &normalized(after, new_id)?)
}

fn consumers(before: &Snapshot, after: &Snapshot) -> Result<(), String> {
    for path in before.consumers.union(&after.consumers) {
        if matches!(path.as_str(), WORKFLOW | WINDOWS_WORKFLOW | DELIVERY | CI) {
            for snapshot in [before, after] {
                let workflow = yaml(snapshot.get(path)?)?;
                let mut globals = workflow.clone();
                globals
                    .as_object_mut()
                    .ok_or("workflow is not a mapping")?
                    .remove("jobs");
                if NEEDLES
                    .iter()
                    .any(|needle| serde_json::to_string(&globals).unwrap().contains(needle))
                {
                    return Err("native source input escaped the Windows job scope".into());
                }
                let jobs = workflow["jobs"]
                    .as_object()
                    .ok_or("consumer workflow jobs absent")?;
                for (name, job) in jobs {
                    let text = serde_json::to_string(job).unwrap();
                    if NEEDLES.iter().any(|needle| text.contains(needle))
                        && !topology::windows_consumer(path, name, job)
                    {
                        return Err(format!(
                            "native source consumer {path}:{name} is not Windows staging"
                        ));
                    }
                }
            }
            continue;
        }
        if path == NATIVE {
            continue;
        }
        if CONSUMERS.contains(&path.as_str()) {
            if !rust_validation_only(before.get(path)?, after.get(path)?)? {
                return Err(format!("live native consumer changed: {path}"));
            }
            continue;
        }
        // The package builder embeds these inputs only in its canonical tests.
        if path == "crates/ferrum-native-ops-builder/src/lib.rs" {
            for snapshot in [before, after] {
                let mut file = syn::parse_file(snapshot.get(path)?).map_err(|e| e.to_string())?;
                file.items.retain(|item| !matches!(item, syn::Item::Mod(module)
                    if module.attrs.iter().any(|attr| matches!(&attr.meta, syn::Meta::List(meta)
                        if meta.path.is_ident("cfg") && matches!(meta.tokens.to_string().as_str(), "test" | "all (test , unix)")))));
                let production = file.into_token_stream().to_string();
                if NEEDLES.iter().any(|needle| production.contains(needle)) {
                    return Err(
                        "native builder library now consumes source inputs in production".into(),
                    );
                }
            }
            continue;
        }
        return Err(format!("unreviewed native source consumer: {path}"));
    }
    Ok(())
}

fn prove(before: &Snapshot, after: &Snapshot) -> Result<Value, String> {
    let recipe = recipes(before, after)?;
    consumers(before, after)?;
    let old = bundle(before)?;
    let new = bundle(after)?;
    if !identity_only(
        before.get(NATIVE)?,
        after.get(NATIVE)?,
        &old["bundle_id"],
        &new["bundle_id"],
    )? {
        return Err("native provider/runtime logic changed beyond source identity".into());
    }
    let member_map = |value: &Value| {
        value["members"]
            .as_array()
            .unwrap()
            .iter()
            .map(|member| (member["path"].as_str().unwrap().to_string(), member.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    let previous = member_map(&old);
    let current = member_map(&new);
    let members: BTreeSet<_> = previous.keys().chain(current.keys()).cloned().collect();
    let changed: Vec<_> = members.into_iter().filter(|path| previous.get(path) != current.get(path))
        .map(|path| json!({"path": path, "before": previous.get(&path), "after": current.get(&path)})).collect();
    let paths: Vec<_> = source_paths().into_iter().chain([NATIVE.to_string(), BUILDER.to_string()])
        .map(|path| Ok(json!({"path": path, "before_sha256": digest(before.get(&path)?.as_bytes()), "after_sha256": digest(after.get(&path)?.as_bytes())})))
        .collect::<Result<_, String>>()?;
    Ok(
        json!({"recipe": recipe, "source_before": old, "source_after": new, "changed_members": changed,
        "input_identities": paths,
        "scope": "release artifact consumers; external source arithmetic is not asserted equivalent",
        "windows_checks": "mandatory candidate staging builds/packages/links native operators and verifies portable bytes; startup may be explicitly deferred without a system driver",
        "local_evidence": "separate local native/model/installer acceptance is not attributed to formal CI",
        "required_gate": "windows_assets and verify_ci must require this candidate's successful Windows staging attempt; this helper does not execute or semantically prove the shell producer",
        "builder_scope": "Linux's unchanged artifact recipe never runs source-build; source producer changes remain the separate Windows staging responsibility, not an assertion of arithmetic equivalence"}),
    )
}

fn project(impact: &mut Impact) -> Vec<String> {
    let mut eligible = source_paths();
    eligible.extend([NATIVE.to_string(), BUILDER.to_string()]);
    let mut paths = Vec::new();
    for entry in &mut impact.paths {
        if eligible.contains(&entry.path) {
            entry.areas = vec![ChangeArea::Build];
            entry.reason = "release-only artifact host scope: unchanged Linux native archive consumption; current source inputs are compiled by independently required Windows staging, with driverless startup explicitly deferred".into();
            paths.push(entry.path.clone());
        }
    }
    impact.areas = ChangeArea::ALL
        .into_iter()
        .filter(|area| impact.paths.iter().any(|entry| entry.areas.contains(area)))
        .collect();
    paths
}

/// This changes only contributing source-input paths, never the global Kernel
/// rules. Unsupported recipes/consumers leave the complete impact untouched.
pub(super) fn refine(repo: &Path, base: &str, candidate: &str, impact: &mut Impact) -> Value {
    let proof = (|| {
        let base = super::resolve_revision(repo, base)?;
        let candidate = super::resolve_revision(repo, candidate)?;
        let proof = prove(&snapshot(repo, &base)?, &snapshot(repo, &candidate)?)?;
        Ok::<_, String>((base, candidate, proof))
    })();
    let (base, candidate, proof) = match proof {
        Ok(value) => value,
        Err(reason) => return json!({"applied": false, "reason": reason}),
    };
    let paths = project(impact);
    json!({"applied": !paths.is_empty(), "base": base, "candidate": candidate, "paths": paths, "proof": proof})
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_bench_core::release_regression::analyze_paths;
    use std::fs;

    fn fixture(windows: bool, source: &[u8]) -> Snapshot {
        // Exercise the actual reviewed workflow, without pinning a commit or
        // inventing success logs. The source inventory itself is deliberately small.
        let mut workflow = yaml(include_str!(
            "../../../../.github/workflows/release-cuda.yml"
        ))
        .unwrap();
        if !windows {
            workflow["jobs"].as_object_mut().unwrap().remove(WINDOWS);
            for trigger in ["workflow_call", "workflow_dispatch"] {
                for input in ["windows_launcher_url", "windows_launcher_sha256"] {
                    workflow["on"][trigger]["inputs"]
                        .as_object_mut()
                        .unwrap()
                        .remove(input);
                }
            }
        }
        let members =
            json!([{"path": "kernel.cu", "sha256": digest(source), "size_bytes": source.len()}]);
        let member_sha = digest(&serde_json::to_vec(&members).unwrap());
        let id = format!("ferrum-native-cuda-v1+sha256.{member_sha}");
        let descriptor = json!({"schema_version": 1, "bundle_id": id,
            "member_set_sha256": member_sha, "members": members,
            "archive": {"file_name": "sources.tar.gz", "sha256": digest(b"archive fixture"), "size_bytes": 10},
            "distribution": {"asset_name": "sources.tar.gz"}});
        let mut snapshot = Snapshot::default();
        snapshot.files.insert(WORKFLOW.into(), workflow.to_string());
        snapshot.files.insert(
            WINDOWS_WORKFLOW.into(),
            include_str!("../../../../.github/workflows/release-windows.yml").into(),
        );
        snapshot.files.insert(
            DELIVERY.into(),
            include_str!("../../../../.github/workflows/release-delivery.yml").into(),
        );
        snapshot.files.insert(
            CI.into(),
            json!({"jobs": {"quality": {"runs-on": "ubuntu-latest", "steps": []}}}).to_string(),
        );
        snapshot.files.insert(BUNDLE.into(), descriptor.to_string());
        snapshot.files.insert(NATIVE.into(), format!("pub const CUDA_NATIVE_SOURCE_BUNDLE_ID: &str = {id:?}; fn provider() -> u32 {{ 1 }}"));
        snapshot
            .files
            .insert(BUILDER.into(), "fn source_build() {}".into());
        snapshot
            .consumers
            .extend([WINDOWS_WORKFLOW.into(), NATIVE.into()]);
        for path in CONSUMERS {
            snapshot.files.insert(
                (*path).into(),
                "fn identity() { cache(CUDA_NATIVE_SOURCE_BUNDLE_ID); }".into(),
            );
            snapshot.consumers.insert((*path).into());
        }
        for operator in OPERATORS {
            snapshot.files.insert(
                format!("native-operators/cuda/source-definitions/{operator}.json"),
                json!({"source_package_revision": id}).to_string(),
            );
            snapshot.files.insert(
                format!("native-operators/cuda/source-locks/{operator}.plan.json"),
                json!({"source_package": {"revision": id}}).to_string(),
            );
        }
        snapshot
    }

    fn change_workflow(snapshot: &mut Snapshot, change: impl FnOnce(&mut Value)) {
        change_file(snapshot, WORKFLOW, change);
    }

    fn change_file(snapshot: &mut Snapshot, path: &str, change: impl FnOnce(&mut Value)) {
        let mut workflow = yaml(snapshot.get(path).unwrap()).unwrap();
        change(&mut workflow);
        snapshot.files.insert(path.into(), workflow.to_string());
    }

    fn commit(repo: &Path, snapshot: &Snapshot) -> String {
        for (path, contents) in &snapshot.files {
            let path = repo.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        }
        super::super::git(repo, &["add", "."]).unwrap();
        super::super::git(
            repo,
            &[
                "-c",
                "user.name=Artifact scope fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=",
                "commit",
                "--quiet",
                "-m",
                "artifact inputs",
            ],
        )
        .unwrap();
        super::super::resolve_revision(repo, "HEAD").unwrap()
    }

    #[test]
    fn immutable_artifact_inputs_preserve_independent_kernel_and_protocol_impact() {
        let repo = tempfile::tempdir().unwrap();
        super::super::git(repo.path(), &["init", "--quiet"]).unwrap();
        let before = fixture(true, b"old native device arithmetic");
        let after = fixture(true, b"different native device arithmetic");
        let base = commit(repo.path(), &before);
        let candidate = commit(repo.path(), &after);
        // An unrelated dirty recipe must not be mistaken for the accepted commit.
        fs::write(repo.path().join(WORKFLOW), "not workflow YAML").unwrap();
        let kernel = "crates/ferrum-kernels/kernels/rms_norm.cu";
        let protocol = "crates/ferrum-types/src/reasoning_controls.rs";
        let unknown = "crates/ferrum-kernels/src/new_runtime.rs";
        let mut impact = analyze_paths([BUNDLE, NATIVE, BUILDER, kernel, protocol, unknown]);
        let original = impact.clone();
        let result = refine(repo.path(), &base, &candidate, &mut impact);
        assert_eq!(result["applied"], true, "{result}");
        assert_eq!(
            result["proof"]["changed_members"].as_array().unwrap().len(),
            1
        );
        assert_eq!(impact.unknown_paths, original.unknown_paths);
        for path in [kernel, protocol, unknown] {
            assert_eq!(
                impact.paths.iter().find(|entry| entry.path == path),
                original.paths.iter().find(|entry| entry.path == path)
            );
        }
        assert!(impact.areas.contains(&ChangeArea::Kernel));
        assert!(impact.areas.contains(&ChangeArea::Tools));
        assert_eq!(
            impact
                .paths
                .iter()
                .find(|entry| entry.path == BUNDLE)
                .unwrap()
                .areas,
            [ChangeArea::Build]
        );

        // A new consumer under an examples directory is not automatically validation.
        let mut unknown_consumer = after;
        unknown_consumer.files.insert(
            "crates/ferrum-bench-core/examples/new_native_consumer.rs".into(),
            "fn main() { consume(CUDA_NATIVE_SOURCE_BUNDLE_ID); }".into(),
        );
        let candidate = commit(repo.path(), &unknown_consumer);
        let mut impact = original.clone();
        assert_eq!(
            refine(repo.path(), &base, &candidate, &mut impact)["applied"],
            false
        );
        assert_eq!(impact, original);
    }

    #[test]
    fn linux_recipe_archive_toolchain_and_future_windows_changes_retain_kernel_scope() {
        let before = fixture(true, b"old");
        let after = fixture(true, b"new");
        assert!(prove(&before, &after).is_ok());
        for change in [
            |value: &mut Value| {
                value["env"]["NATIVE_OPERATOR_SET_ARCHIVE_SHA256"] = json!(digest(b"other archive"))
            },
            |value: &mut Value| {
                value["jobs"][LINUX]["container"]["image"] = json!("different-toolchain")
            },
            |value: &mut Value| {
                value["jobs"][LINUX]["steps"][0]["run"] =
                    json!("source-build native-operators/cuda/source-definitions/new.json")
            },
            |value: &mut Value| value["jobs"][WINDOWS]["continue-on-error"] = json!(true),
        ] {
            let mut changed = after.clone();
            change_workflow(&mut changed, change);
            assert!(prove(&before, &changed).is_err());
        }
        let existing = fixture(true, b"old");
        assert!(prove(&existing, &after).is_ok());
        for change in [
            |value: &mut Value| value["jobs"][WINDOWS]["uses"] = json!("unreviewed/producer@main"),
            |value: &mut Value| {
                value["on"]["workflow_call"]["inputs"]["windows_launcher_url"]["default"] =
                    json!("different input")
            },
        ] {
            let mut changed = after.clone();
            change_workflow(&mut changed, change);
            assert!(prove(&existing, &changed).is_err());
        }
    }

    #[test]
    fn provider_logic_inventory_and_test_only_consumer_boundaries_are_strict() {
        let before = fixture(true, b"old");
        let after = fixture(true, b"new");
        let mut changed = after.clone();
        *changed.files.get_mut(NATIVE).unwrap() = changed
            .get(NATIVE)
            .unwrap()
            .replace("{ 1 }", "{ side_effect(); 1 }");
        assert!(prove(&before, &changed).is_err());
        let mut changed = after.clone();
        changed
            .files
            .get_mut(CONSUMERS[0])
            .unwrap()
            .push_str("fn dispatch() { arithmetic(); }");
        assert!(prove(&before, &changed).is_err());
        let mut changed = after.clone();
        let mut descriptor: Value = serde_json::from_str(changed.get(BUNDLE).unwrap()).unwrap();
        descriptor["members"][0]["size_bytes"] = json!(999);
        changed.files.insert(BUNDLE.into(), descriptor.to_string());
        assert!(prove(&before, &changed).is_err());
        let builder_lib = "crates/ferrum-native-ops-builder/src/lib.rs";
        let mut before = before;
        let mut after = after;
        for snapshot in [&mut before, &mut after] {
            snapshot.consumers.insert(builder_lib.into());
            snapshot.files.insert(builder_lib.into(), "#[cfg(all(test, unix))] mod tests { const SOURCE: &str = \"native-operators/cuda/fixture\"; } fn production() {}".into());
        }
        assert!(prove(&before, &after).is_ok());
        after
            .files
            .get_mut(builder_lib)
            .unwrap()
            .push_str("const PRODUCTION: &str = \"native-operators/cuda/fixture\";");
        assert!(prove(&before, &after).is_err());
    }

    #[test]
    fn unresolved_or_optional_producers_never_authorize_reuse() {
        type Mutation = (&'static str, fn(&mut Value));
        let changes: &[Mutation] = &[
            (DELIVERY, |v| {
                v["jobs"]["stage-cuda-linux"]["with"]["platform"] = json!("windows")
            }),
            (DELIVERY, |v| {
                v["jobs"]["stage-cuda"]["with"]["backend"] = json!("cpu")
            }),
            (DELIVERY, |v| {
                v["jobs"]["stage-cuda"]["uses"] = json!("elsewhere/windows@main")
            }),
            (DELIVERY, |v| {
                v["jobs"]["publish"]["needs"] = json!(["stage-cpu-windows"])
            }),
            (DELIVERY, |v| {
                v["jobs"]["cuda-models"]["continue-on-error"] = json!(true)
            }),
            (WORKFLOW, |v| {
                v["jobs"][LINUX]["if"] = json!("inputs.platform == 'windows'")
            }),
            (WINDOWS_WORKFLOW, |v| {
                v["jobs"]["build"]["if"] = json!("false")
            }),
            (WINDOWS_WORKFLOW, |v| {
                v["jobs"]["build"]["runs-on"] = json!("ubuntu-latest")
            }),
            (WINDOWS_WORKFLOW, |v| v["env"]["BACKEND"] = json!("cpu")),
            (WINDOWS_WORKFLOW, |v| {
                for step in v["jobs"]["build"]["steps"].as_array_mut().unwrap() {
                    if step["name"] == "Build the four native CUDA operators with MSVC" {
                        step["if"] = json!("inputs.backend != 'cuda'");
                    }
                }
            }),
        ];
        for &(path, change) in changes {
            let mut before = fixture(true, b"old");
            let mut after = fixture(true, b"new");
            // Even an unchanged graph is insufficient when its selected
            // producer is optional or consumes another platform's artifact.
            change_file(&mut before, path, change);
            change_file(&mut after, path, change);
            assert!(prove(&before, &after).is_err(), "{path}");
        }
        assert!(prove(&fixture(false, b"old"), &fixture(true, b"new")).is_err());
        let mut missing_callee = fixture(true, b"old");
        missing_callee.files.remove(WINDOWS_WORKFLOW);
        assert!(prove(&missing_callee, &fixture(true, b"new")).is_err());
    }
}
