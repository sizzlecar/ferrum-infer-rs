use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Program {
    pub program: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OrchestralSpec {
    pub program: PathBuf,
    /// JSON configuration (also valid YAML); only whole-value placeholders render.
    pub config_template: PathBuf,
    #[serde(default)]
    pub tool_result_format: OrchestralToolResultFormat,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OrchestralToolResultFormat {
    #[default]
    Json,
    Yaml,
    TextParts,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum AgentSpec {
    Pi(Program),
    Orchestral(OrchestralSpec),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Server {
    /// OpenAI base URL including /v1, on this machine only.
    pub base_url: String,
    pub model: String,
    pub context_window: u32,
    pub max_tokens: u32,
    pub request_timeout_secs: u64,
    pub reasoning: bool,
    pub thinking: String,
    #[serde(default)]
    pub sampling_params: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Validation {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub cwd: PathBuf,
    pub timeout_secs: u64,
    pub expected_initial_exit_code: i32,
    pub protected_paths: Vec<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Task {
    pub id: String,
    pub workdir: PathBuf,
    pub prompt_file: PathBuf,
    pub timeout_secs: u64,
    pub validation: Validation,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub schema_version: u32,
    pub run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pi: Option<Program>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentSpec>,
    pub server: Server,
    pub tasks: Vec<Task>,
}

fn canonical(base: &Path, path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    };
    path.canonicalize()
        .with_context(|| format!("resolve {}", path.display()))
}

pub(crate) fn disjoint(a: &Path, b: &Path) -> bool {
    !a.starts_with(b) && !b.starts_with(a)
}

fn ensure_owned_regular_file(workdir: &Path, declared: &Path) -> Result<()> {
    let mut prefix = PathBuf::new();
    let mut inside = false;
    for component in declared.components() {
        prefix.push(component);
        if inside {
            ensure!(
                !fs::symlink_metadata(&prefix)?.file_type().is_symlink(),
                "candidate-owned protected input traverses symlink {}",
                prefix.display()
            );
        } else if prefix.has_root() && prefix.canonicalize()? == workdir {
            // System aliases above the canonical workdir do not change ownership.
            inside = true;
        }
    }
    ensure!(
        inside && fs::symlink_metadata(declared)?.is_file(),
        "candidate-owned protected input must be a regular file: {}",
        declared.display()
    );
    Ok(())
}

impl Manifest {
    pub(crate) fn pi_program(&self) -> Result<&Program> {
        match (&self.pi, &self.agent) {
            (Some(pi), None) | (None, Some(AgentSpec::Pi(pi))) => Ok(pi),
            _ => anyhow::bail!("this manifest does not select Pi"),
        }
    }

    pub(crate) fn orchestral(&self) -> Option<&OrchestralSpec> {
        match &self.agent {
            Some(AgentSpec::Orchestral(spec)) => Some(spec),
            _ => None,
        }
    }

    pub(crate) fn program(&self) -> &Path {
        match &self.agent {
            Some(AgentSpec::Orchestral(spec)) => &spec.program,
            _ => &self.pi_program().expect("validated Pi selection").program,
        }
    }

    fn validate_selection(&self) -> Result<()> {
        ensure!(
            matches!(
                (self.schema_version, &self.pi, &self.agent),
                (1, Some(_), None) | (2, None, Some(_))
            ),
            "schema 1 requires only pi; schema 2 requires only an explicit agent"
        );
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let base = path.parent().context("manifest parent")?;
        let mut m: Self = serde_json::from_slice(&fs::read(&path)?)?;
        m.validate_selection()?;
        ensure!(!m.tasks.is_empty(), "no tasks");
        ferrum_bench_core::BenchmarkRequestCorrelation::new(
            m.run_id.clone(),
            "task".into(),
            0,
            ferrum_bench_core::BenchmarkPhase::Measured,
            0,
        )
        .map_err(anyhow::Error::msg)?;
        let url = reqwest::Url::parse(&m.server.base_url)?;
        ensure!(
            url.scheme() == "http"
                && matches!(
                    url.host_str(),
                    Some("127.0.0.1" | "localhost" | "[::1]" | "::1")
                ),
            "server base_url must use HTTP loopback; remote inference is not allowed"
        );
        ensure!(
            url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
                && url.path().trim_end_matches('/') == "/v1",
            "expected unauthenticated /v1 base URL"
        );
        ensure!(
            m.server.context_window > m.server.max_tokens
                && m.server.max_tokens > 0
                && m.server.request_timeout_secs > 0
                && !m.server.model.is_empty(),
            "invalid server budgets/model"
        );
        ensure!(
            ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
                .contains(&m.server.thinking.as_str()),
            "invalid pi thinking level"
        );
        for key in m.server.sampling_params.keys() {
            ensure!(
                ![
                    "model",
                    "messages",
                    "tools",
                    "stream",
                    "max_tokens",
                    "max_completion_tokens"
                ]
                .contains(&key.as_str()),
                "sampling_params cannot replace {key}"
            );
        }
        match (&mut m.pi, &mut m.agent) {
            (Some(pi), None) | (None, Some(AgentSpec::Pi(pi))) => {
                pi.program = canonical(base, &pi.program)?;
            }
            (None, Some(AgentSpec::Orchestral(spec))) => {
                spec.program = canonical(base, &spec.program)?;
                spec.config_template = canonical(base, &spec.config_template)?;
                let _: Value = serde_json::from_slice(&fs::read(&spec.config_template)?)
                    .context("Orchestral config template must be JSON (valid YAML)")?;
            }
            _ => unreachable!("selection validated"),
        }
        let mut ids = BTreeSet::new();
        for task in &mut m.tasks {
            ensure!(
                !task.id.is_empty()
                    && task
                        .id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
                "unsafe task id"
            );
            ensure!(ids.insert(task.id.clone()), "duplicate task id");
            ensure!(
                task.timeout_secs > 0 && task.validation.timeout_secs > 0,
                "task/validator timeout must be explicit and positive"
            );
            ensure!(
                task.validation.expected_initial_exit_code == 1,
                "validator must distinguish semantic failure (1) from compilation/infrastructure failure (2)"
            );
            task.workdir = canonical(base, &task.workdir)?;
            ensure!(task.workdir.is_dir(), "task workdir is not a directory");
            task.prompt_file = canonical(base, &task.prompt_file)?;
            ensure!(
                !fs::read_to_string(&task.prompt_file)?.trim().is_empty(),
                "empty task prompt"
            );
            task.validation.program = canonical(base, &task.validation.program)?;
            task.validation.cwd = canonical(base, &task.validation.cwd)?;
            ensure!(
                !task.validation.protected_paths.is_empty(),
                "validator protection paths required"
            );
            for p in &mut task.validation.protected_paths {
                let resolved = canonical(base, p)?;
                if resolved.starts_with(&task.workdir) {
                    let declared = if p.is_absolute() {
                        p.clone()
                    } else {
                        base.join(&*p)
                    };
                    ensure_owned_regular_file(&task.workdir, &declared)?;
                }
                *p = resolved;
            }
        }
        for (i, task) in m.tasks.iter().enumerate() {
            if let Some(spec) = m.orchestral() {
                for input in [&spec.program, &spec.config_template] {
                    ensure!(
                        disjoint(&task.workdir, input),
                        "Orchestral input overlaps candidate"
                    );
                }
            }
            ensure!(
                disjoint(&task.workdir, &path),
                "manifest must be outside agent workdir"
            );
            for other in &m.tasks[i + 1..] {
                ensure!(
                    disjoint(&task.workdir, &other.workdir),
                    "agent workdirs overlap"
                );
            }
            for (other_idx, other) in m.tasks.iter().enumerate() {
                for p in [&other.validation.program, &other.prompt_file] {
                    ensure!(
                        disjoint(&task.workdir, p),
                        "agent workdir overlaps protected input {}",
                        p.display()
                    );
                }
                for p in &other.validation.protected_paths {
                    let owned_file = i == other_idx
                        && p != &task.workdir
                        && p.starts_with(&task.workdir)
                        && p.is_file();
                    ensure!(
                        disjoint(&task.workdir, p) || owned_file,
                        "agent workdir overlaps protected input {}",
                        p.display()
                    );
                }
            }
        }
        Ok(m)
    }
}

pub(crate) fn hash_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Snapshot canonical protected paths without following symlinks into user directories.
pub(crate) fn snapshot(paths: &[PathBuf]) -> Result<BTreeMap<PathBuf, String>> {
    fn visit(path: &Path, entries: &mut BTreeMap<PathBuf, String>) -> Result<()> {
        let meta = fs::symlink_metadata(path)?;
        ensure!(
            !meta.file_type().is_symlink(),
            "protected symlink {}",
            path.display()
        );
        if meta.is_dir() {
            entries.insert(path.to_owned(), "directory".into());
            for entry in fs::read_dir(path)? {
                visit(&entry?.path(), entries)?;
            }
        } else if meta.is_file() {
            entries.insert(path.to_owned(), hash_file(path)?);
        } else {
            anyhow::bail!("unsupported protected file type {}", path.display());
        }
        Ok(())
    }
    let mut result = BTreeMap::new();
    for path in paths {
        // Manifest loading canonicalizes these paths. A later directory-to-symlink
        // replacement must not redirect an individually protected file's lookup.
        for ancestor in path.ancestors().skip(1) {
            ensure!(
                !fs::symlink_metadata(ancestor)?.file_type().is_symlink(),
                "protected symlink ancestor {}",
                ancestor.display()
            );
        }
        visit(path, &mut result)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn legacy_pi_selection_round_trips_and_schema_two_is_explicit() {
        let legacy = json!({"schema_version":1,"run_id":"legacy","pi":{"program":"pi","args":["script.js"]},
            "server":{"base_url":"http://127.0.0.1:8001/v1","model":"local","context_window":4096,
                "max_tokens":512,"request_timeout_secs":30,"reasoning":false,"thinking":"off","sampling_params":{}},"tasks":[]});
        let parsed: Manifest = serde_json::from_value(legacy.clone()).unwrap();
        parsed.validate_selection().unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), legacy);
        assert_eq!(parsed.pi_program().unwrap().args, ["script.js"]);
        let mut explicit = legacy.clone();
        explicit["schema_version"] = json!(2);
        explicit.as_object_mut().unwrap().remove("pi");
        explicit["agent"] =
            json!({"kind":"orchestral","program":"orchestral","config_template":"agent.json"});
        let selected: Manifest = serde_json::from_value(explicit.clone()).unwrap();
        selected.validate_selection().unwrap();
        assert!(selected.orchestral().is_some() && selected.pi_program().is_err());
        explicit["pi"] = legacy["pi"].clone();
        assert!(serde_json::from_value::<Manifest>(explicit)
            .unwrap()
            .validate_selection()
            .is_err());
        let mut no_agent = legacy;
        no_agent["schema_version"] = json!(2);
        assert!(serde_json::from_value::<Manifest>(no_agent)
            .unwrap()
            .validate_selection()
            .is_err());
    }

    #[test]
    fn orchestral_tool_result_format_is_explicit_and_legacy_defaults_to_json() {
        let mut value = json!({"program":"orchestral","config_template":"agent.json"});
        let legacy: OrchestralSpec = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(legacy.tool_result_format, OrchestralToolResultFormat::Json);
        value["tool_result_format"] = json!("yaml");
        let selected: OrchestralSpec = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(
            selected.tool_result_format,
            OrchestralToolResultFormat::Yaml
        );
        assert_eq!(serde_json::to_value(selected).unwrap(), value);
        value["tool_result_format"] = json!("auto");
        assert!(serde_json::from_value::<OrchestralSpec>(value).is_err());
    }

    #[test]
    fn protection_detects_edits_additions_and_deletions() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("contract.rs");
        fs::write(&file, "original").unwrap();
        let paths = [dir.path().canonicalize().unwrap()];
        let before = snapshot(&paths).unwrap();
        fs::write(&file, "changed").unwrap();
        assert_ne!(before, snapshot(&paths).unwrap());
        fs::write(&file, "original").unwrap();
        fs::write(dir.path().join("extra.rs"), "extra").unwrap();
        assert_ne!(before, snapshot(&paths).unwrap());
        fs::remove_file(&file).unwrap();
        assert_ne!(before, snapshot(&paths).unwrap());
    }

    #[test]
    fn adjacent_directories_are_disjoint_but_parent_input_is_not() {
        assert!(disjoint(Path::new("/runs/a"), Path::new("/runs/ab")));
        assert!(!disjoint(
            Path::new("/runs/a"),
            Path::new("/runs/a/contract")
        ));
        assert!(!disjoint(Path::new("/runs/a"), Path::new("/runs")));
    }

    struct ProtectedFilesFixture {
        _root: tempfile::TempDir,
        path: PathBuf,
        manifest: Manifest,
    }

    impl ProtectedFilesFixture {
        fn new() -> Self {
            let root = tempfile::tempdir().unwrap();
            let base = root.path().canonicalize().unwrap();
            let inputs = base.join("inputs");
            fs::create_dir(&inputs).unwrap();
            let program = base.join("program-stub");
            fs::write(&program, "not executed by manifest loading").unwrap();
            let tasks = ["first", "second"]
                .into_iter()
                .map(|id| {
                    let workdir = base.join("work").join(id);
                    fs::create_dir_all(workdir.join("tests")).unwrap();
                    fs::write(workdir.join("tests/repro.rs"), "original regression").unwrap();
                    let prompt_file = inputs.join(format!("{id}.txt"));
                    fs::write(
                        &prompt_file,
                        "Repair the regression without changing tests.",
                    )
                    .unwrap();
                    let contract = inputs.join(format!("{id}-contract.rs"));
                    fs::write(&contract, "independent contract").unwrap();
                    Task {
                        id: id.into(),
                        workdir,
                        prompt_file,
                        timeout_secs: 30,
                        validation: Validation {
                            program: program.clone(),
                            args: Vec::new(),
                            cwd: inputs.clone(),
                            timeout_secs: 10,
                            expected_initial_exit_code: 1,
                            protected_paths: vec![contract],
                        },
                    }
                })
                .collect();
            Self {
                _root: root,
                path: base.join("manifest.json"),
                manifest: Manifest {
                    schema_version: 2,
                    run_id: "protected-files".into(),
                    pi: None,
                    agent: Some(AgentSpec::Pi(Program {
                        program,
                        args: Vec::new(),
                    })),
                    server: Server {
                        base_url: "http://127.0.0.1:8001/v1".into(),
                        model: "local".into(),
                        context_window: 4096,
                        max_tokens: 512,
                        request_timeout_secs: 30,
                        reasoning: false,
                        thinking: "off".into(),
                        sampling_params: BTreeMap::new(),
                    },
                    tasks,
                },
            }
        }

        fn repro(&self, task: usize) -> PathBuf {
            self.manifest.tasks[task].workdir.join("tests/repro.rs")
        }

        fn load(&self) -> Result<Manifest> {
            fs::write(&self.path, serde_json::to_vec(&self.manifest)?)?;
            Manifest::load(&self.path)
        }
    }

    #[test]
    fn protected_files_accept_each_owners_file_and_preserve_external_contracts() {
        let mut fixture = ProtectedFilesFixture::new();
        for i in 0..fixture.manifest.tasks.len() {
            let relative = fixture
                .repro(i)
                .strip_prefix(fixture.path.parent().unwrap())
                .unwrap()
                .to_owned();
            fixture.manifest.tasks[i]
                .validation
                .protected_paths
                .push(relative);
        }
        let loaded = fixture.load().unwrap();
        for (i, task) in loaded.tasks.iter().enumerate() {
            assert_eq!(
                task.validation.protected_paths[0],
                fixture.manifest.tasks[i].validation.protected_paths[0]
            );
            assert_eq!(task.validation.protected_paths[1], fixture.repro(i));
            assert_eq!(snapshot(&task.validation.protected_paths).unwrap().len(), 2);
        }
    }

    #[test]
    fn protected_files_reject_candidate_directories_and_ancestors() {
        let mut fixture = ProtectedFilesFixture::new();
        let workdir = fixture.manifest.tasks[0].workdir.clone();
        for path in [
            workdir.join("tests"),
            workdir.clone(),
            workdir.parent().unwrap().to_owned(),
        ] {
            fixture.manifest.tasks[0].validation.protected_paths = vec![path.clone()];
            assert!(fixture.load().is_err(), "accepted {}", path.display());
        }
    }

    #[test]
    fn protected_files_reject_another_tasks_candidate_file() {
        for (owner, other) in [(0, 1), (1, 0)] {
            let mut fixture = ProtectedFilesFixture::new();
            let repro = fixture.repro(owner);
            fixture.manifest.tasks[other].validation.protected_paths = vec![repro];
            assert!(
                fixture.load().is_err(),
                "accepted task {owner} file for task {other}"
            );
        }
    }

    #[test]
    fn protected_files_do_not_relax_prompt_program_or_manifest_isolation() {
        let mut fixture = ProtectedFilesFixture::new();
        let repro = fixture.repro(0);
        fixture.manifest.tasks[0]
            .validation
            .protected_paths
            .push(repro.clone());
        let prompt = fixture.manifest.tasks[0].prompt_file.clone();
        fixture.manifest.tasks[0].prompt_file = repro.clone();
        assert!(fixture.load().is_err());
        fixture.manifest.tasks[0].prompt_file = prompt;
        let program = fixture.manifest.tasks[0].validation.program.clone();
        fixture.manifest.tasks[0].validation.program = repro;
        assert!(fixture.load().is_err());
        fixture.manifest.tasks[0].validation.program = program;
        fixture.path = fixture.manifest.tasks[0].workdir.join("manifest.json");
        assert!(fixture.load().is_err());
    }

    #[test]
    fn protected_files_snapshot_detects_owned_file_edits_and_deletion() {
        let mut fixture = ProtectedFilesFixture::new();
        let repro = fixture.repro(0);
        fixture.manifest.tasks[0]
            .validation
            .protected_paths
            .push(repro.clone());
        let loaded = fixture.load().unwrap();
        let paths = &loaded.tasks[0].validation.protected_paths;
        let frozen = snapshot(paths).unwrap();
        fs::write(&repro, "changed regression").unwrap();
        assert_ne!(snapshot(paths).unwrap(), frozen);
        fs::write(&repro, "original regression").unwrap();
        assert_eq!(snapshot(paths).unwrap(), frozen);
        fs::remove_file(&repro).unwrap();
        assert!(snapshot(paths).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn protected_files_reject_owned_symlink_at_load_and_after_snapshot() {
        let mut fixture = ProtectedFilesFixture::new();
        let repro = fixture.repro(0);
        fixture.manifest.tasks[0]
            .validation
            .protected_paths
            .push(repro.clone());
        let loaded = fixture.load().unwrap();
        let paths = &loaded.tasks[0].validation.protected_paths;
        let frozen = snapshot(paths).unwrap();
        let target = repro.with_file_name("same-content.rs");
        fs::write(&target, "original regression").unwrap();
        fs::remove_file(&repro).unwrap();
        std::os::unix::fs::symlink(&target, &repro).unwrap();
        assert_eq!(hash_file(&repro).unwrap(), frozen[&repro]);
        assert!(snapshot(paths).is_err());
        assert!(fixture.load().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn protected_files_reject_owned_symlink_ancestors_and_later_redirection() {
        let mut fixture = ProtectedFilesFixture::new();
        let repro = fixture.repro(0);
        fixture.manifest.tasks[0]
            .validation
            .protected_paths
            .push(repro.clone());
        let loaded = fixture.load().unwrap();
        let paths = &loaded.tasks[0].validation.protected_paths;
        let frozen = snapshot(paths).unwrap();
        let tests = repro.parent().unwrap();
        let target = tests.with_file_name("real-tests");
        fs::rename(tests, &target).unwrap();
        std::os::unix::fs::symlink(&target, tests).unwrap();
        assert!(fs::symlink_metadata(&repro).unwrap().is_file());
        assert_eq!(hash_file(&repro).unwrap(), frozen[&repro]);
        assert!(fixture.load().is_err());
        assert!(snapshot(paths).is_err());

        // Replacing the link with another ordinary directory changes the frozen file.
        fs::remove_file(tests).unwrap();
        fs::create_dir(tests).unwrap();
        fs::write(&repro, "redirected regression").unwrap();
        assert_ne!(snapshot(paths).unwrap(), frozen);
    }

    #[cfg(unix)]
    #[test]
    fn protected_files_allow_aliases_above_the_owning_workdir() {
        let mut fixture = ProtectedFilesFixture::new();
        let root = fixture.path.parent().unwrap();
        let alias = root.join("work-alias");
        std::os::unix::fs::symlink(root.join("work"), &alias).unwrap();
        fixture.manifest.tasks[0]
            .validation
            .protected_paths
            .push(alias.join("first/tests/repro.rs"));
        let loaded = fixture.load().unwrap();
        assert_eq!(
            loaded.tasks[0].validation.protected_paths[1],
            fixture.repro(0)
        );
        snapshot(&loaded.tasks[0].validation.protected_paths).unwrap();
    }
}
