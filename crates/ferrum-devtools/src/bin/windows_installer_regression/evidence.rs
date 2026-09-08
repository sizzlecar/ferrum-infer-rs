use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::Path,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryValue {
    pub kind: u32,
    pub bytes: Vec<u8>,
}
impl RegistryValue {
    pub fn text(&self) -> Result<String> {
        ensure!(
            matches!(self.kind, 1 | 2) && self.bytes.len() % 2 == 0,
            "registry value is not a UTF-16 string"
        );
        let mut words: Vec<_> = self
            .bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        ensure!(
            words.pop() == Some(0) && !words.contains(&0),
            "registry string termination is not canonical"
        );
        Ok(String::from_utf16(&words)?)
    }
}

pub fn appended_path(before: Option<&RegistryValue>, app: &str) -> Result<RegistryValue> {
    let mut text = before
        .map(RegistryValue::text)
        .transpose()?
        .unwrap_or_default();
    if !text.is_empty() && !text.ends_with(';') {
        text.push(';');
    }
    text.push_str(app);
    Ok(RegistryValue {
        kind: before.map_or(1, |p| p.kind),
        bytes: text
            .encode_utf16()
            .chain([0])
            .flat_map(u16::to_le_bytes)
            .collect(),
    })
}

pub fn comparable(value: &str) -> String {
    let mut value = value
        .trim()
        .trim_matches('"')
        .replace('/', "\\")
        .to_lowercase();
    while value.len() > 3 && value.ends_with('\\') {
        value.pop();
    }
    value
}

pub fn paths_overlap(left: &Path, right: &Path) -> Result<bool> {
    fn normalized(path: &Path) -> Result<String> {
        let path = path.to_str().context("non-Unicode protected directory")?;
        let path = if let Some(tail) = path.strip_prefix("\\\\?\\UNC\\") {
            format!("\\\\{tail}")
        } else {
            path.strip_prefix("\\\\?\\").unwrap_or(path).to_owned()
        };
        Ok(comparable(&path))
    }
    let (left, right) = (normalized(left)?, normalized(right)?);
    Ok(left == right
        || left.starts_with(&format!("{}\\", right.trim_end_matches('\\')))
        || right.starts_with(&format!("{}\\", left.trim_end_matches('\\'))))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    pub sha256: String,
    pub size_bytes: u64,
}
pub fn absent(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(false), // A dangling link is still an existing user-owned entry.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(error) => Err(error.into()),
    }
}
pub fn regular(path: &Path) -> Result<fs::Metadata> {
    let meta = fs::symlink_metadata(path)?;
    ensure!(
        !meta.file_type().is_symlink(),
        "symlink is outside verification scope: {}",
        path.display()
    );
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        ensure!(
            meta.file_attributes() & 0x400 == 0,
            "reparse point is outside verification scope: {}",
            path.display()
        );
    }
    Ok(meta)
}
pub fn digest(path: &Path) -> Result<Identity> {
    ensure!(
        regular(path)?.is_file(),
        "not a regular file: {}",
        path.display()
    );
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = vec![0; 65536];
    let mut size_bytes = 0;
    loop {
        let n = file.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
        size_bytes += n as u64;
    }
    Ok(Identity {
        sha256: format!("{:x}", hash.finalize()),
        size_bytes,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRecord {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}
#[derive(Deserialize)]
struct Manifest {
    files: Vec<FileRecord>,
}

pub fn payload_version(root: &Path) -> Result<String> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("ferrum-portable.json"))?)?;
    let text = manifest["build"]["version"]
        .as_str()
        .context("payload version is missing")?;
    let version = semver::Version::parse(text)?;
    ensure!(
        version.pre.is_empty() && version.build.is_empty() && version.to_string() == text,
        "payload requires a canonical formal version"
    );
    Ok(text.to_owned())
}

pub fn verify_upgrade_versions(previous: &str, current: &str) -> Result<()> {
    ensure!(
        semver::Version::parse(previous)? < semver::Version::parse(current)?,
        "upgrade requires a newer executable version; same-version reinstall is a separate check"
    );
    Ok(())
}

pub fn payload(root: &Path) -> Result<Vec<FileRecord>> {
    let manifest = root.join("ferrum-portable.json");
    let parsed: Manifest = serde_json::from_slice(&fs::read(&manifest)?)?;
    ensure!(
        !parsed.files.is_empty() && parsed.files.len() <= 128,
        "invalid payload inventory size"
    );
    let mut names = BTreeSet::from(["ferrum-portable.json".to_owned()]);
    for row in &parsed.files {
        let parts: Vec<_> = row.path.split('/').collect();
        ensure!(
            parts.len() == 1 || parts.len() == 2 && parts[0] == "licenses",
            "invalid payload path"
        );
        ensure!(
            parts.iter().all(|p| !p.is_empty()
                && p.is_ascii()
                && !matches!(*p, "." | "..")
                && !p.ends_with(['.', ' '])
                && !p
                    .chars()
                    .any(|c| c.is_control() || "<>:\"\\|?*".contains(c))),
            "unsafe payload path"
        );
        ensure!(
            names.insert(row.path.to_ascii_lowercase()),
            "duplicate payload path"
        );
        ensure!(
            row.sha256.len() == 64
                && row
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid payload SHA"
        );
    }
    ensure!(names.contains("ferrum.exe"), "payload omits ferrum.exe");
    let identity = digest(&manifest)?;
    let mut records = parsed.files;
    records.push(FileRecord {
        path: "ferrum-portable.json".into(),
        sha256: identity.sha256,
        size_bytes: identity.size_bytes,
    });
    let observed = inventory(root)?;
    ensure!(
        observed.files.len() == records.len(),
        "accepted payload has unlisted files"
    );
    Ok(records)
}
pub fn verify_files(root: &Path, records: &[FileRecord]) -> Result<()> {
    ensure!(regular(root)?.is_dir(), "payload root is not a directory");
    for row in records {
        let mut path = root.to_path_buf();
        for part in row.path.split('/') {
            path.push(part);
            regular(&path)?;
        }
        let actual = digest(&path)?;
        ensure!(
            actual.sha256 == row.sha256 && actual.size_bytes == row.size_bytes,
            "payload bytes differ: {}",
            row.path
        );
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Inventory {
    pub files: BTreeMap<String, Identity>,
    pub directories: BTreeSet<String>,
}
pub fn inventory(root: &Path) -> Result<Inventory> {
    fn visit(root: &Path, dir: &Path, result: &mut Inventory) -> Result<()> {
        ensure!(regular(dir)?.is_dir(), "not a directory: {}", dir.display());
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            let name = path
                .strip_prefix(root)?
                .to_str()
                .context("non-Unicode inventory path")?
                .replace('\\', "/");
            let meta = regular(&path)?;
            if meta.is_dir() {
                result.directories.insert(name);
                visit(root, &path, result)?;
            } else {
                result.files.insert(name, digest(&path)?);
            }
        }
        Ok(())
    }
    let mut result = Inventory {
        files: BTreeMap::new(),
        directories: BTreeSet::new(),
    };
    visit(root, root, &mut result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn value(kind: u32, text: &str) -> RegistryValue {
        RegistryValue {
            kind,
            bytes: text
                .encode_utf16()
                .chain([0])
                .flat_map(u16::to_le_bytes)
                .collect(),
        }
    }
    #[test]
    fn path_append_preserves_prefix_type_and_distinguishes_absence() {
        for (original, expected) in [
            ("A;B", "A;B;C:\\Ferrum"),
            ("A;", "A;C:\\Ferrum"),
            ("", "C:\\Ferrum"),
        ] {
            for kind in [1, 2] {
                assert_eq!(
                    appended_path(Some(&value(kind, original)), "C:\\Ferrum").unwrap(),
                    value(kind, expected)
                );
            }
        }
        assert_ne!(None, Some(value(1, "")));
        assert_ne!(value(1, "%USERPROFILE%"), value(2, "%USERPROFILE%"));
        assert_eq!(comparable(" \"C:/Ferrum/\" "), comparable("c:\\ferrum"));
        assert_ne!(comparable("C:\\OtherFerrum"), comparable("C:\\Ferrum"));
    }

    #[test]
    fn upgrade_requires_a_newer_formal_payload_version() {
        assert!(verify_upgrade_versions("0.8.9", "0.8.10").is_ok());
        assert!(verify_upgrade_versions("0.8.9", "0.8.9").is_err());
        assert!(verify_upgrade_versions("0.9.0", "0.8.10").is_err());
        let root = tempfile::tempdir().unwrap();
        for text in ["0.8.10", "0.8.10-rc.1", "0.8.10+local", "v0.8.10", ""] {
            fs::write(
                root.path().join("ferrum-portable.json"),
                serde_json::to_vec(&serde_json::json!({"build":{"version":text}})).unwrap(),
            )
            .unwrap();
            assert_eq!(payload_version(root.path()).is_ok(), text == "0.8.10");
        }
    }
    #[test]
    fn payload_requires_exact_manifest_and_file_bytes() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("ferrum.exe"), b"payload").unwrap();
        let id = digest(&root.path().join("ferrum.exe")).unwrap();
        fs::write(root.path().join("ferrum-portable.json"),serde_json::to_vec(&serde_json::json!({"files":[{"path":"ferrum.exe","sha256":id.sha256,"size_bytes":id.size_bytes}]})).unwrap()).unwrap();
        let records = payload(root.path()).unwrap();
        verify_files(root.path(), &records).unwrap();
        fs::write(root.path().join("ferrum.exe"), b"PAYLOAD").unwrap();
        assert!(verify_files(root.path(), &records).is_err());
        fs::write(root.path().join("ferrum.exe"), b"payload").unwrap();
        fs::write(root.path().join("ferrum-portable.json"), b"{}").unwrap();
        assert!(verify_files(root.path(), &records).is_err());
    }

    #[test]
    fn external_inventory_detects_changed_bytes_and_empty_directories() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("empty")).unwrap();
        let model = root.path().join("模型 weights.bin");
        fs::write(&model, b"original").unwrap();
        let before = inventory(root.path()).unwrap();
        fs::write(&model, b"replaced").unwrap();
        assert_ne!(inventory(root.path()).unwrap(), before);
        fs::write(&model, b"original").unwrap();
        assert_eq!(inventory(root.path()).unwrap(), before);
        fs::remove_dir(root.path().join("empty")).unwrap();
        assert_ne!(inventory(root.path()).unwrap(), before);
        assert!(paths_overlap(
            Path::new("\\\\?\\C:\\Users\\User"),
            Path::new("C:\\Users\\User\\Programs\\Ferrum")
        )
        .unwrap());
        assert!(!paths_overlap(Path::new("C:\\Models"), Path::new("C:\\Models-other")).unwrap());
        assert!(paths_overlap(Path::new("C:\\"), Path::new("C:\\Programs\\Ferrum")).unwrap());
    }
}
