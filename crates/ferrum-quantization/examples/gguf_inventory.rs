//! Inspect local GGUF descriptors, or an HTTP range containing the full header.
//! Usage: gguf_inventory FILE [--file-size FULL_ARTIFACT_BYTES]

use std::fs::File;
use std::io::{Cursor, Write};
use std::path::PathBuf;

use ferrum_quantization::gguf::GgufInventory;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: gguf_inventory FILE [--file-size FULL_ARTIFACT_BYTES]")?;
    let file = File::open(&path)?;
    let local_file_bytes = file.metadata()?.len();
    let (declared_file_bytes, file_length_source) = match args.next() {
        None => (local_file_bytes, "local_file"),
        Some(option) if option == "--file-size" => {
            let size = args.next().ok_or("--file-size needs a byte count")?;
            let size: u64 = size.to_str().ok_or("invalid byte count")?.parse()?;
            if size < local_file_bytes || args.next().is_some() {
                return Err(
                    "declared size must cover the local header; unexpected arguments are rejected"
                        .into(),
                );
            }
            (size, "caller_declared")
        }
        Some(_) => return Err("expected --file-size FULL_ARTIFACT_BYTES".into()),
    };
    // SAFETY: this read-only diagnostic requires the input to remain unchanged
    // for its duration, just like GgufFile::open. No tensor is materialized.
    let bytes = unsafe { memmap2::Mmap::map(&file)? };
    let inventory = GgufInventory::read(&mut Cursor::new(&bytes[..]), declared_file_bytes)?;
    let report = serde_json::json!({
        "file": path,
        "local_file_bytes": local_file_bytes,
        "file_length_source": file_length_source,
        "payload_verified": false,
        "tensor_payloads_materialized": false,
        "inventory": inventory,
    });
    let mut output = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut output, &report)?;
    writeln!(output)?;
    Ok(())
}
