//! A successful compiler must produce a new artifact before any cache/stamp publication.
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Old cache entries can be internally consistent while certifying stale bytes.
/// Include this contract in the input identity; never migrate pre-contract PTX.
pub const OUTPUT_CONTRACT: &str = "core-ptx-staged-output-v1";
static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, PartialEq, Eq)]
pub struct ArtifactIdentity {
    pub sha256: String,
    pub size_bytes: u64,
}

struct Stage(PathBuf, bool);
impl Stage {
    fn new(out_dir: &Path) -> io::Result<Self> {
        loop {
            let sequence = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
            let path = out_dir.join(format!(".core-ptx-{}-{sequence}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path, true)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }
}
impl Drop for Stage {
    fn drop(&mut self) {
        if self.1 {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

fn regular_file(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(true),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "PTX artifact/stamp is not a regular file: {}",
                path.display()
            ),
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn artifact_identity(path: &Path) -> io::Result<ArtifactIdentity> {
    if !regular_file(path)? {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("compiler did not create {}", path.display()),
        ));
    }
    let bytes = fs::read(path)?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let directive = |name: &str| {
        text.lines()
            .any(|line| line.split_ascii_whitespace().next() == Some(name))
    };
    if !directive(".version") || !directive(".target") || !directive(".address_size") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "compiler output lacks PTX directives",
        ));
    }
    Ok(ArtifactIdentity {
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        size_bytes: bytes.len() as u64,
    })
}

fn bound_stamp(signature: &str, artifact: &ArtifactIdentity) -> String {
    format!(
        "{signature}\nartifact.sha256={}\nartifact.size_bytes={}",
        artifact.sha256, artifact.size_bytes
    )
}

/// A source signature alone cannot prove which bytes remain in OUT_DIR.
pub fn local_artifact_matches(
    out_dir: &Path,
    file_name: &str,
    signature: &str,
) -> io::Result<bool> {
    let artifact = artifact_identity(&out_dir.join(file_name))?;
    let stamp = out_dir.join(format!("{file_name}.stamp"));
    if !regular_file(&stamp)? {
        return Ok(false);
    }
    Ok(fs::read_to_string(stamp)? == bound_stamp(signature, &artifact))
}

/// The shared cache already checked its spec and payload SHA. Bind its restored
/// bytes locally too, so interruption or a later overwrite cannot become a hit.
pub fn record_restored_artifact(
    out_dir: &Path,
    file_name: &str,
    signature: &str,
) -> io::Result<()> {
    let artifact = artifact_identity(&out_dir.join(file_name))?;
    let stage = Stage::new(out_dir)?;
    let staged_stamp = stage.0.join("input.stamp");
    fs::write(&staged_stamp, bound_stamp(signature, &artifact))?;
    fs::rename(staged_stamp, out_dir.join(format!("{file_name}.stamp")))
}

/// The compiler receives a previously nonexistent explicit output path. Its
/// success alone is insufficient: the new file must be regular, nonempty PTX.
/// The cache receives only these verified bytes, before replacing local outputs.
pub fn compile_and_publish(
    out_dir: &Path,
    file_name: &str,
    signature: &str,
    compile: impl FnOnce(&Path) -> io::Result<()>,
    publish_cache: impl FnOnce(&Path) -> io::Result<()>,
) -> io::Result<ArtifactIdentity> {
    if Path::new(file_name)
        .file_name()
        .and_then(|name| name.to_str())
        != Some(file_name)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "PTX filename must be a basename",
        ));
    }
    let mut stage = Stage::new(out_dir)?;
    let generated = stage.0.join(file_name);
    compile(&generated)?;
    let identity = artifact_identity(&generated)?;
    let destination = out_dir.join(file_name);
    let stamp = out_dir.join(format!("{file_name}.stamp"));
    let previous = stage.0.join("previous.ptx");
    let had_previous = regular_file(&destination)?;
    regular_file(&stamp)?;
    if had_previous {
        fs::copy(&destination, &previous)?;
    }
    let staged_stamp = stage.0.join("input.stamp");
    fs::write(&staged_stamp, bound_stamp(signature, &identity))?;
    publish_cache(&generated)?;
    if let Err(error) = replace_pair(
        &generated,
        &destination,
        &staged_stamp,
        &stamp,
        had_previous.then_some(previous.as_path()),
        |from, to| fs::rename(from, to),
    ) {
        if previous.exists() {
            // If filesystem failure also prevented rollback, retain the backup
            // for recovery instead of deleting the only previous artifact copy.
            stage.1 = false;
            return Err(io::Error::new(
                error.kind(),
                format!(
                    "{error}; previous artifact retained at {}",
                    previous.display()
                ),
            ));
        }
        return Err(error);
    }
    Ok(identity)
}

// Each replacement is atomic on its filesystem. If replacing the stamp fails,
// restore the previous artifact rather than leaving it under the old stamp.
fn replace_pair(
    generated: &Path,
    destination: &Path,
    staged_stamp: &Path,
    stamp: &Path,
    previous: Option<&Path>,
    mut rename: impl FnMut(&Path, &Path) -> io::Result<()>,
) -> io::Result<()> {
    rename(generated, destination)?;
    if let Err(error) = rename(staged_stamp, stamp) {
        let rollback = match previous {
            Some(previous) => fs::rename(previous, destination),
            None => fs::remove_file(destination),
        };
        return match rollback {
            Ok(()) => Err(error),
            Err(rollback) => Err(io::Error::other(format!(
                "PTX stamp replacement failed: {error}; artifact rollback also failed: {rollback}"
            ))),
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    include!("core_ptx_tests.rs");
}
