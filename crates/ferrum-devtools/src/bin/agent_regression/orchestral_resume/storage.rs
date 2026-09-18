//! Filesystem boundaries for the explicitly selected private session only.
use super::*;
use std::collections::BTreeMap;

pub(super) fn read_json(path: &Path) -> Result<Value> {
    ensure!(
        fs::symlink_metadata(path)?.is_file(),
        "evidence must be a regular file: {}",
        path.display()
    );
    serde_json::from_slice(&fs::read(path)?).with_context(|| format!("parse {}", path.display()))
}

pub(super) fn new_report(path: &Path, source: &Path, workdir: &Path) -> Result<PathBuf> {
    let name = path
        .file_name()
        .context("new report must have a final path component")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let path = parent.canonicalize()?.join(name);
    let source = source.canonicalize()?;
    let workdir = workdir.canonicalize()?;
    ensure!(
        config::disjoint(&path, &source) && config::disjoint(&path, &workdir),
        "new report must be disjoint from source report and task workspace"
    );
    fs::create_dir(&path).context("create exclusive new report directory")?;
    Ok(path)
}

pub(super) struct CopiedSession {
    pub directory: PathBuf,
    pub session_path: PathBuf,
    pub hashes: BTreeMap<String, String>,
    pub session_id: String,
    pub prior_run_id: String,
}

fn is_owned(name: &str, value: &Value, session: &str, run: &str) -> bool {
    if name.starts_with("session-") {
        value.as_array().is_some_and(|records| {
            !records.is_empty()
                && records
                    .iter()
                    .all(|record| record["session_id"] == session && record["run_id"] == run)
        })
    } else if name.starts_with("run-") {
        value
            .pointer("/run/registration/request/run/spec")
            .is_some_and(|spec| spec["session_id"] == session && spec["run_id"] == run)
    } else if name.starts_with("generic-checkpoint-") {
        value
            .pointer("/registration/request/run/spec")
            .is_some_and(|spec| spec["session_id"] == session && spec["run_id"] == run)
    } else if name.starts_with("effect-") {
        value.as_array().is_some_and(|records| {
            !records.is_empty() && records.iter().all(|record| record["key"]["run_id"] == run)
        })
    } else {
        false
    }
}

pub(super) fn copy_session(
    prior: &orchestral_evidence::Evidence,
    journal: &Path,
) -> Result<CopiedSession> {
    ensure!(prior.complete(), "cannot copy an incomplete source session");
    let session = prior.session_id.as_deref().context("missing session")?;
    let run = prior.run_id.as_deref().context("missing run")?;
    let source_session = prior
        .session_path
        .as_ref()
        .context("missing Session path")?;
    let source_directory = source_session
        .parent()
        .context("missing Session directory")?;
    ensure!(
        prior.run_path.as_ref().and_then(|path| path.parent()) == Some(source_directory),
        "source public journals are split across directories"
    );
    let hash = format!(
        "{:x}",
        <sha2::Sha256 as sha2::Digest>::digest(session.as_bytes())
    );
    let directory = journal.join("sessions").join(hash);
    fs::create_dir_all(&directory)?;
    let mut hashes = BTreeMap::new();
    for entry in fs::read_dir(source_directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 journal filename"))?;
        ensure!(
            entry.file_type()?.is_file(),
            "selected journal contains a non-regular entry"
        );
        if name == ".writer.lock" {
            continue;
        }
        ensure!(name.ends_with(".json"), "unknown selected journal file");
        let value = read_json(&entry.path())?;
        ensure!(
            is_owned(&name, &value, session, run),
            "journal file is foreign or unsupported: {name}"
        );
        let bytes = fs::read(entry.path())?;
        let destination = directory.join(&name);
        fs::write(&destination, bytes)?;
        let hash = config::hash_file(&entry.path())?;
        ensure!(
            config::hash_file(&destination)? == hash,
            "journal copy changed bytes"
        );
        hashes.insert(name, hash);
    }
    Ok(CopiedSession {
        session_path: directory.join(source_session.file_name().context("Session filename")?),
        directory,
        hashes,
        session_id: session.into(),
        prior_run_id: run.into(),
    })
}

pub(super) fn check_copied_session(copied: &CopiedSession) -> Result<PathBuf> {
    // Do not silently ignore a second layout/session created by the new process.
    let sessions = copied.directory.parent().context("session parent")?;
    let root = sessions.parent().context("journal root")?;
    for (parent, expected) in [(root, sessions), (sessions, copied.directory.as_path())] {
        ensure!(
            fs::symlink_metadata(parent)?.is_dir(),
            "journal layout changed type"
        );
        let entries = fs::read_dir(parent)?.collect::<std::io::Result<Vec<_>>>()?;
        ensure!(
            entries.len() == 1 && entries[0].path() == expected && entries[0].file_type()?.is_dir(),
            "unexpected journal layout or foreign session"
        );
    }
    for (name, expected) in &copied.hashes {
        let path = copied.directory.join(name);
        ensure!(
            fs::symlink_metadata(&path)?.is_file(),
            "copied journal changed file type"
        );
        if path != copied.session_path {
            ensure!(
                config::hash_file(&path)? == *expected,
                "previous journal bytes changed: {name}"
            );
        }
    }
    let mut new_runs = Vec::new();
    let mut new_files = Vec::new();
    for entry in fs::read_dir(&copied.directory)? {
        let entry = entry?;
        ensure!(
            entry.file_type()?.is_file(),
            "new journal entry is not regular"
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 journal filename"))?;
        if copied.hashes.contains_key(&name) || name == ".writer.lock" {
            continue;
        }
        ensure!(
            name.ends_with(".json")
                && (name.starts_with("run-") || name.starts_with("generic-checkpoint-")),
            "unexpected new journal or tool effect: {name}"
        );
        let value = read_json(&entry.path())?;
        if name.starts_with("run-") {
            new_runs.push(entry.path());
        }
        new_files.push((name, value));
    }
    ensure!(new_runs.len() == 1, "expected exactly one new Run journal");
    let run = read_json(&new_runs[0])?;
    let new_id = run
        .pointer("/run/registration/request/run/spec/run_id")
        .and_then(Value::as_str)
        .context("new run identity")?;
    ensure!(
        new_id != copied.prior_run_id,
        "new process reused prior Run"
    );
    for (name, value) in new_files {
        ensure!(
            is_owned(&name, &value, &copied.session_id, new_id),
            "new journal belongs to another session/run"
        );
    }
    Ok(new_runs.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finished_journal(directory: &Path) -> orchestral_evidence::Evidence {
        fs::create_dir(directory).unwrap();
        let inline = |text: &str| json!({"body":{"kind":"inline","value":text}});
        let events = [
            json!({"type":"run_accepted","session_id":"s","spec_digest":"digest"}),
            json!({"type":"run_started"}),
            json!({"type":"output_committed","content":[inline("done")]}),
            json!({"type":"delivery_committed","delivery":{"run_id":"r","spec_digest":"digest","final_response":inline("done")}}),
        ];
        let run = json!({"schema_version":1,"run":{
            "registration":{"request":{"run":{"spec":{"protocol_version":{"major":1},"session_id":"s","run_id":"r","input":[inline("initial")]},"spec_digest":"digest"}},"execution":{"session_id":"s","run_id":"r","spec_digest":"digest"}},
            "records":events.into_iter().enumerate().map(|(i,payload)|json!({"event":{"event_id":format!("run-{i}"),"run_id":"r","run_seq":i+1,"payload":payload}})).collect::<Vec<_>>()
        }});
        let session = json!([
            {"session_seq":1,"session_id":"s","run_id":"r","event_id":"input","payload":{"type":"run_input_committed","message":{"role":"user","content":[{"type":"text","text":"initial"}]}}},
            {"session_seq":2,"session_id":"s","run_id":"r","event_id":"output","payload":{"type":"run_output_committed","request_id":"request","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"usage":{"input_tokens":5,"output_tokens":1}}}
        ]);
        write_json(directory.join("run-old.json"), &run).unwrap();
        write_json(directory.join("session-owned.json"), &session).unwrap();
        fs::write(directory.join(".writer.lock"), b"old lease").unwrap();
        let evidence = orchestral_evidence::read(directory, "s");
        assert!(evidence.complete(), "{:?}", evidence.errors);
        evidence
    }

    #[test]
    fn copied_session_is_byte_exact_and_retained_runs_cannot_change_or_duplicate() {
        let root = tempfile::tempdir().unwrap();
        let prior = finished_journal(&root.path().join("source"));
        let source_before = config::hash_file(prior.run_path.as_ref().unwrap()).unwrap();
        let copied = copy_session(&prior, &root.path().join("new")).unwrap();
        assert!(!copied.directory.join(".writer.lock").exists());
        assert_eq!(
            config::hash_file(&copied.directory.join("run-old.json")).unwrap(),
            source_before
        );
        assert!(
            check_copied_session(&copied).is_err(),
            "missing new Run must fail"
        );
        let mut run = read_json(&copied.directory.join("run-old.json")).unwrap();
        run["run"]["registration"]["request"]["run"]["spec"]["run_id"] = json!("new");
        write_json(copied.directory.join("run-new.json"), &run).unwrap();
        assert!(check_copied_session(&copied).is_ok());
        write_json(copied.directory.join("run-extra.json"), &run).unwrap();
        assert!(check_copied_session(&copied).is_err());
        fs::remove_file(copied.directory.join("run-extra.json")).unwrap();
        fs::write(copied.directory.join("run-old.json"), b"changed").unwrap();
        assert!(check_copied_session(&copied).is_err());
        assert_eq!(
            config::hash_file(prior.run_path.as_ref().unwrap()).unwrap(),
            source_before
        );
    }

    #[test]
    fn copy_rejects_foreign_private_checkpoint_in_the_selected_shard() {
        let root = tempfile::tempdir().unwrap();
        let prior = finished_journal(&root.path().join("source"));
        let checkpoint = json!({"registration":{"request":{"run":{"spec":{"session_id":"foreign","run_id":"r"}}}}});
        write_json(
            root.path().join("source/generic-checkpoint-foreign.json"),
            &checkpoint,
        )
        .unwrap();
        assert!(copy_session(&prior, &root.path().join("new")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn copy_rejects_symlinks_without_reading_their_target() {
        let root = tempfile::tempdir().unwrap();
        let prior = finished_journal(&root.path().join("source"));
        std::os::unix::fs::symlink(
            "missing-private-file",
            root.path().join("source/effect-link.json"),
        )
        .unwrap();
        assert!(copy_session(&prior, &root.path().join("new")).is_err());
    }

    #[test]
    fn copied_private_files_require_exact_session_and_run_ownership() {
        let run = json!({"run":{"registration":{"request":{"run":{"spec":{"session_id":"s","run_id":"r"}}}}}});
        assert!(is_owned("run-hash.json", &run, "s", "r"));
        assert!(!is_owned("run-hash.json", &run, "other", "r"));
        let effect = json!([{"key":{"run_id":"r"}},{"key":{"run_id":"foreign"}}]);
        assert!(!is_owned("effect-hash.json", &effect, "s", "r"));
        assert!(!is_owned("unknown.json", &run, "s", "r"));
    }

    #[test]
    fn reports_cannot_overwrite_or_nest_in_original_evidence() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        let workdir = root.path().join("work");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&workdir).unwrap();
        assert!(new_report(&source, &source, &workdir).is_err());
        assert!(new_report(&source.join("child"), &source, &workdir).is_err());
        assert!(new_report(&workdir.join("child"), &source, &workdir).is_err());
        let output = root.path().join("new");
        assert!(new_report(&output, &source, &workdir).is_ok());
        assert!(new_report(&output, &source, &workdir).is_err());
    }
}
