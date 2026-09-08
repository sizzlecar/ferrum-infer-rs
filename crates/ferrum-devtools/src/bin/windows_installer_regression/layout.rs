//! The launcher selects an immutable, complete version payload through current.json.
use super::evidence::{self, FileRecord, Identity};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Current {
    schema_version: u32,
    version_dir: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Selection {
    pub version_dir: String,
    pub directory: PathBuf,
    pub manifest: Identity,
    pub core_path: PathBuf,
    pub core: Identity,
    pub launcher_path: PathBuf,
    pub launcher: Identity,
    pub pointer: Identity,
}

pub fn version_dir(source: &Path) -> Result<String> {
    Ok(format!(
        "{}-{}",
        evidence::payload_version(source)?,
        evidence::digest(&source.join("ferrum-portable.json"))?.sha256
    ))
}

pub fn inspect(install: &Path, source: &Path, expected_launcher: &Identity) -> Result<Selection> {
    let pointer_path = install.join("current.json");
    let pointer = evidence::digest(&pointer_path)?;
    let current: Current = serde_json::from_slice(&fs::read(&pointer_path)?)?;
    let expected = version_dir(source)?;
    ensure!(
        current.schema_version == 1 && current.version_dir == expected,
        "current.json does not select the accepted version/manifest"
    );
    let directory = install.join("versions").join(&expected);
    ensure!(
        evidence::regular(&install.join("versions"))?.is_dir()
            && evidence::regular(&directory)?.is_dir(),
        "version payload is not a regular directory"
    );
    let records = evidence::payload(source)?;
    evidence::verify_files(&directory, &records)?;
    ensure!(
        evidence::payload(&directory)? == records,
        "installed version has extra or changed payload files"
    );
    let launcher_path = install.join("ferrum.exe");
    let launcher = evidence::digest(&launcher_path)?;
    ensure!(
        &launcher == expected_launcher,
        "stable launcher bytes changed or were replaced by a core binary"
    );
    ensure!(
        evidence::digest(&pointer_path)? == pointer,
        "current.json changed while checking the selected payload"
    );
    let core_path = directory.join("ferrum.exe");
    Ok(Selection {
        version_dir: expected,
        manifest: evidence::digest(&directory.join("ferrum-portable.json"))?,
        core: evidence::digest(&core_path)?,
        core_path,
        launcher_path,
        launcher,
        pointer,
        directory,
    })
}

pub fn records_for_version(records: &[FileRecord], version_dir: &str) -> Vec<String> {
    records
        .iter()
        .map(|record| format!("versions/{version_dir}/{}", record.path))
        .collect()
}

pub fn core_record(records: &[FileRecord]) -> Result<&FileRecord> {
    records
        .iter()
        .find(|record| record.path == "ferrum.exe")
        .context("payload has no core executable")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn source(root: &Path, version: &str, bytes: &[u8]) {
        fs::create_dir_all(root).unwrap();
        fs::write(root.join("ferrum.exe"), bytes).unwrap();
        let id = evidence::digest(&root.join("ferrum.exe")).unwrap();
        fs::write(root.join("ferrum-portable.json"),serde_json::to_vec(&json!({"build":{"version":version},"files":[{"path":"ferrum.exe","sha256":id.sha256,"size_bytes":id.size_bytes}]})).unwrap()).unwrap();
    }
    #[test]
    fn selection_binds_launcher_and_complete_version_without_removing_old_payload() {
        let temp = tempfile::tempdir().unwrap();
        let install = temp.path().join("installed");
        fs::create_dir_all(install.join("versions")).unwrap();
        fs::write(install.join("ferrum.exe"), b"launcher").unwrap();
        let launcher = evidence::digest(&install.join("ferrum.exe")).unwrap();
        let mut paths = Vec::new();
        for (version, bytes) in [("0.8.8", &b"old core"[..]), ("0.8.9", &b"new core"[..])] {
            let input = temp.path().join(version);
            source(&input, version, bytes);
            let id = version_dir(&input).unwrap();
            let target = install.join("versions").join(&id);
            fs::create_dir(&target).unwrap();
            for file in ["ferrum.exe", "ferrum-portable.json"] {
                fs::copy(input.join(file), target.join(file)).unwrap();
            }
            fs::write(
                install.join("current.json"),
                serde_json::to_vec(&Current {
                    schema_version: 1,
                    version_dir: id,
                })
                .unwrap(),
            )
            .unwrap();
            let selected = inspect(&install, &input, &launcher).unwrap();
            assert_eq!(
                selected.core.sha256,
                evidence::digest(&input.join("ferrum.exe")).unwrap().sha256
            );
            paths.push((input, target));
        }
        assert_eq!(
            fs::read(paths[0].1.join("ferrum.exe")).unwrap(),
            b"old core"
        );
        assert!(inspect(&install, &paths[0].0, &launcher).is_err());
        fs::write(paths[1].1.join("ferrum.exe"), b"bad core").unwrap();
        assert!(inspect(&install, &paths[1].0, &launcher).is_err());
    }
    #[test]
    fn pointer_rejects_traversal_unknown_fields_and_wrong_manifest() {
        for value in [
            json!({"schema_version":1,"version_dir":"../escape"}),
            json!({"schema_version":1,"version_dir":"x","extra":true}),
        ] {
            let temp = tempfile::tempdir().unwrap();
            source(temp.path(), "0.8.9", b"core");
            fs::write(
                temp.path().join("current.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
            assert!(inspect(
                temp.path(),
                temp.path(),
                &evidence::digest(&temp.path().join("ferrum.exe")).unwrap()
            )
            .is_err());
        }
    }
}
