//! Calibration-only bounded source stream. It is created immediately so a
//! failed run leaves its actual partial records. Only finish returns a digest
//! receipt; an open/error file must never be used as a completed source.
use super::*;
use std::fs::{File, OpenOptions};

pub(super) struct RawSource {
    file: File,
    path: PathBuf,
    bytes: u64,
    limit: u64,
    hash: Sha256,
}
impl RawSource {
    pub fn create(path: &Path, limit: u64) -> Result<Self, ExportError> {
        let path = files::destination(path)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            file,
            path,
            bytes: 0,
            limit,
            hash: Sha256::new(),
        })
    }
    pub fn record(&mut self, value: &impl Serialize) -> Result<(), ExportError> {
        serde_json::to_writer(&mut *self, value)?;
        self.write_all(b"\n")?;
        Ok(())
    }
    pub fn prefix_digest(&self) -> [u8; 32] {
        self.hash.clone().finalize().into()
    }
    pub fn finish(mut self) -> Result<PublishedFile, ExportError> {
        self.flush()?;
        self.file.sync_all()?;
        let digest = self.hash.finalize();
        Ok(PublishedFile {
            path: self.path,
            bytes: self.bytes,
            sha256: format!("{digest:x}"),
            digest: digest.into(),
        })
    }
}
impl Write for RawSource {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.limit.saturating_sub(self.bytes) {
            return Err(std::io::Error::other(
                "selected calibration raw byte limit exceeded",
            ));
        }
        let n = self.file.write(bytes)?;
        self.hash.update(&bytes[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}
