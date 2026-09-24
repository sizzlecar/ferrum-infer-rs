use super::*;
use std::path::Path;

pub(super) fn outputs<'a>(
    cmd: &'a CalibrateSloCommand,
    manifest: &'a manifest::Manifest,
) -> Vec<&'a Path> {
    let mut result = vec![cmd.observations.as_path(), cmd.out.as_path()];
    if let Some(directory) = &cmd.export_http_inputs {
        result.push(directory.as_path());
    }
    if let Some((profile, source)) = manifest.validation_model.destinations() {
        result.extend([profile, source]);
    }
    if let Some(reference) = &manifest.reference {
        result.extend([
            reference.artifact_path.as_path(),
            reference.frozen_plan_path.as_path(),
        ]);
    }
    result
}

pub(super) fn validate(paths: impl IntoIterator<Item = impl AsRef<Path>>) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for path in paths {
        let path = path.as_ref();
        match std::fs::symlink_metadata(path) {
            Ok(_) => {
                return Err(FerrumError::config(format!(
                    "calibration output already exists: {}",
                    path.display()
                )))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(FerrumError::config(format!(
                    "inspect calibration output {}: {error}",
                    path.display()
                )))
            }
        }
        if !seen.insert(identity(path)?) {
            return Err(FerrumError::config(
                "calibration output paths alias one another",
            ));
        }
    }
    Ok(())
}

pub(super) fn distinct(paths: impl IntoIterator<Item = impl AsRef<Path>>) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for path in paths {
        if !seen.insert(identity(path.as_ref())?) {
            return Err(FerrumError::config(
                "calibration cut, live export and report destinations must be distinct",
            ));
        }
    }
    Ok(())
}

fn identity(path: &Path) -> Result<PathBuf> {
    let leaf = path
        .file_name()
        .ok_or_else(|| FerrumError::config("calibration output needs a filename"))?;
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(parent
        .canonicalize()
        .map_err(|error| {
            FerrumError::config(format!("calibration output parent must exist: {error}"))
        })?
        .join(leaf))
}
