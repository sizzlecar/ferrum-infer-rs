use super::{error, OfflineSloError, OfflineSloReport};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::Path,
};

/// Read a regular file with a growth-safe byte bound.
pub fn read_bounded(path: &Path, maximum_bytes: usize) -> Result<Vec<u8>, OfflineSloError> {
    let file = std::fs::File::open(path).map_err(|e| error(format!("{}: {e}", path.display())))?;
    let metadata = file.metadata().map_err(|e| error(e.to_string()))?;
    if !metadata.is_file() || metadata.len() > maximum_bytes as u64 {
        return Err(error(format!(
            "{} is not a bounded regular file",
            path.display()
        )));
    }
    let read_limit = (maximum_bytes as u64)
        .checked_add(1)
        .ok_or_else(|| error("file limit overflow"))?;
    let mut bytes = Vec::new();
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|e| error(e.to_string()))?;
    if bytes.len() > maximum_bytes {
        return Err(error("input grew beyond its byte limit"));
    }
    Ok(bytes)
}

/// Never overwrite an input, symlink, hard-link alias, or earlier output.
/// On a write error the new partial output remains for diagnosis; it is never
/// silently presented as complete, and existing paths are never removed.
pub fn write_new(path: &Path, report: &OfflineSloReport) -> Result<(), OfflineSloError> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| error(format!("create new {}: {e}", path.display())))?;
    let mut writer = std::io::BufWriter::new(file);
    let write = (|| -> Result<(), Box<dyn std::error::Error>> {
        serde_json::to_writer_pretty(&mut writer, report)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    })();
    write.map_err(|e| {
        error(format!(
            "write {}; partial new output retained: {e}",
            path.display()
        ))
    })
}
