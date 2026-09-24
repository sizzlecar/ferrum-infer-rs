//! Bounded-buffer publication of a single owned JSON artifact.
//!
//! This is separate from append-only JSONL journals. The record can own a
//! retention lease: it stays alive through serialization, flush and publication,
//! without creating a second JSON tree or a whole-artifact byte buffer.

use std::{
    fs::{File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use serde::Serialize;

static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

/// Stream one JSON value and atomically replace the destination on success.
///
/// The caller bounds the record's retained storage; this writer uses an 8 KiB
/// buffer regardless of encoded size. A failed write/flush/rename preserves an
/// existing artifact, removes the temporary file and drops the owned record.
/// The parent directory must exist. This does not promise crash durability.
pub fn write_json_owned_record<T: Serialize>(path: &Path, record: T) -> io::Result<()> {
    let (file, mut temporary) = Temporary::create(path)?;
    let mut writer = BufWriter::with_capacity(8 * 1024, file);
    serde_json::to_writer(&mut writer, &record).map_err(io::Error::other)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    drop(writer);
    std::fs::rename(temporary.path.as_ref().expect("unpublished file"), path)?;
    temporary.path = None;
    drop(record);
    Ok(())
}

struct Temporary {
    path: Option<PathBuf>,
}

impl Temporary {
    fn create(destination: &Path) -> io::Result<(File, Self)> {
        let parent = destination
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        // Bounded retries handle leftover names from an earlier process with
        // the same PID. create_new never follows an existing temporary symlink.
        for _ in 0..16 {
            let serial = NEXT_TEMPORARY
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .map_err(|_| io::Error::other("JSON artifact temporary identity exhausted"))?;
            let path = parent.join(format!(".ferrum-json-{}-{serial}.tmp", std::process::id()));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok((file, Self { path: Some(path) })),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "JSON artifact temporary names are occupied",
        ))
    }
}

impl Drop for Temporary {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests;
