use super::*;
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Debug, Clone, Serialize)]
pub(in crate::commands::calibrate_slo) struct FrozenPlanReceipt {
    pub path: PathBuf,
    pub sha256: [u8; 32],
    pub bytes: u64,
}

pub(super) fn bounded_json(value: &impl Serialize, maximum: usize) -> Result<Vec<u8>> {
    struct Bounded {
        bytes: Vec<u8>,
        maximum: usize,
    }
    impl Write for Bounded {
        fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
            if self
                .bytes
                .len()
                .checked_add(input.len())
                .is_none_or(|len| len > self.maximum)
            {
                return Err(std::io::Error::other(
                    "frozen plan exceeds declared byte bound",
                ));
            }
            let required = self.bytes.len() + input.len();
            if required > self.bytes.capacity() {
                let capacity = self
                    .bytes
                    .capacity()
                    .saturating_mul(2)
                    .max(4096)
                    .max(required)
                    .min(self.maximum);
                self.bytes
                    .try_reserve_exact(capacity - self.bytes.len())
                    .map_err(std::io::Error::other)?;
            }
            self.bytes.extend_from_slice(input);
            Ok(input.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut out = Bounded {
        bytes: Vec::new(),
        maximum,
    };
    serde_json::to_writer(&mut out, value)
        .map_err(|error| invalid(format!("serialize frozen plan: {error}")))?;
    Ok(out.bytes)
}

pub(super) fn publish(
    destination: &Path,
    value: &impl Serialize,
    maximum: usize,
) -> Result<FrozenPlanReceipt> {
    let bytes = bounded_json(value, maximum)?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()
        .map_err(io_error)?;
    let leaf = destination
        .file_name()
        .ok_or_else(|| invalid("frozen plan needs a filename"))?;
    let destination = parent.join(leaf);
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nonce = NEXT
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| invalid("frozen plan temporary identifier exhausted"))?;
    let path = parent.join(format!(
        ".ferrum-reference-plan-{}-{nonce}.tmp",
        std::process::id()
    ));
    // Construct cleanup ownership only after create_new succeeds: an existing
    // colliding temporary file belongs to someone else and must be preserved.
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .map_err(io_error)?;
    struct Temporary(PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let temporary = Temporary(path);
    file.write_all(&bytes).map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    std::fs::hard_link(&temporary.0, &destination).map_err(io_error)?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(io_error)?;
    Ok(FrozenPlanReceipt {
        path: destination,
        sha256: Sha256::digest(&bytes).into(),
        bytes: bytes.len() as u64,
    })
}

pub(super) fn verify(receipt: &FrozenPlanReceipt) -> Result<()> {
    let mut file = File::open(&receipt.path).map_err(io_error)?;
    if file.metadata().map_err(io_error)?.len() != receipt.bytes {
        return Err(invalid("persisted frozen plan size changed"));
    }
    let mut hash = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(io_error)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .filter(|n| *n <= receipt.bytes)
            .ok_or_else(|| invalid("persisted frozen plan grew"))?;
        hash.update(&buffer[..count]);
    }
    if total != receipt.bytes || <[u8; 32]>::from(hash.finalize()) != receipt.sha256 {
        return Err(invalid(
            "persisted frozen plan no longer matches its pre-trial bytes",
        ));
    }
    Ok(())
}

fn io_error(error: std::io::Error) -> FerrumError {
    invalid(format!("frozen plan I/O: {error}"))
}
