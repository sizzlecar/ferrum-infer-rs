use serde::Deserialize;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Current {
    schema_version: u32,
    version_dir: String,
}

fn version_directory(value: &str) -> bool {
    if value.len() > 128 {
        return false;
    }
    let Some((version, digest)) = value.split_once('-') else {
        return false;
    };
    let numbers: Vec<_> = version.split('.').collect();
    numbers.len() == 3
        && numbers.iter().all(|part| {
            !part.is_empty()
                && (part.len() == 1 || !part.starts_with('0'))
                && part.bytes().all(|b| b.is_ascii_digit())
                && part.parse::<u64>().is_ok()
        })
        && digest.len() == 64
        && digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn plain(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes()
            & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
            != 0
        {
            return false;
        }
    }
    !metadata.file_type().is_symlink()
}

pub(super) fn resolve(root: &Path) -> Result<PathBuf, String> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_DELETE, FILE_SHARE_READ,
        };
        // Atomic replacement can publish a new pointer while this reader finishes
        // reading the old file. In-place writers cannot change our open record.
        options
            .share_mode(FILE_SHARE_READ | FILE_SHARE_DELETE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let path = root.join("current.json");
    let before = fs::symlink_metadata(&path).map_err(|e| format!("read current.json: {e}"))?;
    if !before.is_file() || !plain(&before) {
        return Err("current.json must be a regular file".into());
    }
    let file = options
        .open(&path)
        .map_err(|e| format!("open current.json: {e}"))?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || !plain(&metadata) || metadata.len() > 4096 {
        return Err("current.json must be a regular file of at most 4096 bytes".into());
    }
    let mut bytes = Vec::new();
    file.take(4097)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 4096 {
        return Err("current.json exceeds 4096 bytes".into());
    }
    let current: Current =
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid current.json: {e}"))?;
    if current.schema_version != 1 || !version_directory(&current.version_dir) {
        return Err("current.json has an unsupported schema or unsafe version_dir".into());
    }
    let versions = root.join("versions");
    let directory = versions.join(current.version_dir);
    let binary = directory.join("ferrum.exe");
    for (path, is_directory) in [(&versions, true), (&directory, true), (&binary, false)] {
        let metadata =
            fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if !plain(&metadata)
            || if is_directory {
                !metadata.is_dir()
            } else {
                !metadata.is_file()
            }
        {
            return Err(
                "version payload must use ordinary directories and a regular ferrum.exe".into(),
            );
        }
    }
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    let program = binary.canonicalize().map_err(|e| e.to_string())?;
    if !program.starts_with(root.join("versions")) {
        return Err("version payload escapes the installation directory".into());
    }
    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn version_pointer_selects_one_immutable_payload_and_rejects_unsafe_records() {
        let root = tempfile::tempdir().unwrap();
        let id = format!("0.8.9-{}", "a".repeat(64));
        let payload = root.path().join("versions").join(&id);
        fs::create_dir_all(&payload).unwrap();
        fs::write(payload.join("ferrum.exe"), b"fixture, never executed").unwrap();
        let path = root.path().join("current.json");
        let valid = json!({"schema_version":1,"version_dir":id});
        fs::write(&path, serde_json::to_vec(&valid).unwrap()).unwrap();
        assert_eq!(
            resolve(root.path()).unwrap(),
            payload.join("ferrum.exe").canonicalize().unwrap()
        );
        for bad in [
            "../escape",
            "C:\\outside",
            "0.8.9-short",
            "0.8.09-aaaaaaaa",
            "versions/other",
            "NUL",
            "0.8.9:stream",
        ] {
            fs::write(
                &path,
                json!({"schema_version":1,"version_dir":bad}).to_string(),
            )
            .unwrap();
            assert!(resolve(root.path()).is_err(), "accepted {bad}");
        }
        let mut unknown = valid.clone();
        unknown["binary"] = json!("outside.exe");
        for bad in [
            unknown.to_string(),
            valid.to_string() + "{}",
            " ".repeat(4097),
        ] {
            fs::write(&path, bad).unwrap();
            assert!(resolve(root.path()).is_err());
        }
        fs::write(&path, valid.to_string()).unwrap();
        fs::remove_file(payload.join("ferrum.exe")).unwrap();
        assert!(resolve(root.path()).is_err());
    }
}
