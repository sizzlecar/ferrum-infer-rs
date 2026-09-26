//! Calibration-only bounded source stream. It is created immediately so a
//! failed run leaves its actual partial records. Only finish returns a digest
//! receipt; an open/error file must never be used as a completed source.
use super::*;
use std::{
    fs::{File, OpenOptions},
    io::BufWriter,
};

// Bounded per-source memory; JSON fragments share a write buffer while each
// completed record retains its original immediate-read/error boundary.
const RAW_SOURCE_WRITE_BUFFER_BYTES: usize = 64 * 1024;

pub(in crate::continuous_engine::inner::cost_observation::profile_export) struct RawSource {
    file: BufWriter<File>,
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
            file: BufWriter::with_capacity(RAW_SOURCE_WRITE_BUFFER_BYTES, file),
            path,
            bytes: 0,
            limit,
            hash: Sha256::new(),
        })
    }
    pub fn record(&mut self, value: &impl Serialize) -> Result<(), ExportError> {
        serde_json::to_writer(&mut *self, value)?;
        self.write_all(b"\n")?;
        // The caller may immediately read the source prefix. Do not delay a
        // file-write error until a later record, phase freeze, or finish.
        self.flush()?;
        Ok(())
    }
    pub fn prefix_digest(&self) -> [u8; 32] {
        self.hash.clone().finalize().into()
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn finish(mut self) -> Result<PublishedFile, ExportError> {
        self.flush()?;
        self.file.get_ref().sync_all()?;
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

#[cfg(test)]
mod tests {
    use super::*;

    struct SourceFile(PathBuf);
    impl Drop for SourceFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[derive(Serialize)]
    struct Record {
        label: &'static str,
        values: Vec<u64>,
    }

    #[test]
    fn buffered_raw_records_are_immediately_readable_with_original_bytes_and_hashes() {
        let file = SourceFile(std::env::temp_dir().join(format!(
            "ferrum-buffered-raw-{}.jsonl",
            uuid::Uuid::new_v4()
        )));
        let records = [
            Record {
                label: "首次\n\"quoted\"",
                values: vec![0, u64::MAX],
            },
            Record {
                label: "many serializer fragments across buffer boundaries",
                values: (0..RAW_SOURCE_WRITE_BUFFER_BYTES as u64 / 2).collect(),
            },
            Record {
                label: "small final record",
                values: vec![7],
            },
        ];
        let encoded = records
            .iter()
            .map(|record| {
                let mut bytes = serde_json::to_vec(record).unwrap();
                bytes.push(b'\n');
                bytes
            })
            .collect::<Vec<_>>();
        assert!(encoded[1].len() > RAW_SOURCE_WRITE_BUFFER_BYTES);
        let limit = encoded.iter().map(Vec::len).sum::<usize>() as u64;
        let mut source = RawSource::create(&file.0, limit).unwrap();
        let mut expected = Vec::new();
        for (record, bytes) in records.iter().zip(encoded) {
            source.record(record).unwrap();
            expected.extend(bytes);
            // No explicit caller flush/finish: a successful record must already
            // expose the entire exact prefix to the calibration freeze reader.
            assert_eq!(std::fs::read(&file.0).unwrap(), expected);
            assert_eq!(source.bytes(), expected.len() as u64);
            assert_eq!(
                source.prefix_digest(),
                <[u8; 32]>::from(Sha256::digest(&expected))
            );
        }
        let published = source.finish().unwrap();
        assert_eq!(published.bytes, limit);
        assert_eq!(std::fs::read(&published.path).unwrap(), expected);
        assert_eq!(
            published.digest,
            <[u8; 32]>::from(Sha256::digest(&expected))
        );
        assert_eq!(published.sha256, format!("{:x}", Sha256::digest(&expected)));
    }
}
