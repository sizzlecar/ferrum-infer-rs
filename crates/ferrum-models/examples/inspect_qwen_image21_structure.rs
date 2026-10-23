//! Inspect detached Transformer metadata; no safetensors payload is opened.
//!
//! Usage: inspect_qwen_image21_structure CONFIG INDEX SHARD HEADER_JSON FILE_BYTES [SHARD HEADER_JSON FILE_BYTES ...]
//! SHARD is the package-relative safetensors name in INDEX. FILE_BYTES is the
//! separately observed whole-file length, including the length word and header.
//! HEADER_JSON must preserve the entire original header, including padding;
//! do not trim or reserialize it after extracting the file's header range.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use ferrum_models::qwen_image21::{HeaderEvidence, QwenImage21Structure, StructureLimits};

fn read_metadata(path: &Path, limit: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > limit {
        return Err(format!(
            "{} is not a non-empty bounded metadata file",
            path.display()
        )
        .into());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.is_empty() || bytes.len() as u64 > limit {
        return Err(format!(
            "{} changed outside the metadata read bounds",
            path.display()
        )
        .into());
    }
    Ok(bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let usage = "usage: inspect_qwen_image21_structure CONFIG INDEX SHARD HEADER_JSON FILE_BYTES [SHARD HEADER_JSON FILE_BYTES ...]";
    if arguments.len() == 1 && (arguments[0] == "--help" || arguments[0] == "-h") {
        println!(
            "{usage}\nStructure only; payload authenticity and executability are not established."
        );
        return Ok(());
    }
    let limits = StructureLimits::default();
    if arguments.len() < 5
        || (arguments.len() - 2) % 3 != 0
        || (arguments.len() - 2) / 3 > limits.maximum_shards()
    {
        return Err(usage.into());
    }
    let config = read_metadata(Path::new(&arguments[0]), limits.metadata_file_bytes())?;
    let index = read_metadata(Path::new(&arguments[1]), limits.metadata_file_bytes())?;
    let mut remaining = limits
        .total_metadata_bytes()
        .checked_sub((config.len() + index.len()) as u64)
        .ok_or("metadata total exceeds limit")?;
    let mut stored = Vec::new();
    for shard in arguments[2..].chunks_exact(3) {
        let name = shard[0].to_str().ok_or("SHARD must be UTF-8")?.to_owned();
        let header = read_metadata(
            Path::new(&shard[1]),
            remaining.min(limits.metadata_file_bytes()),
        )?;
        remaining -= header.len() as u64;
        let file_bytes: u64 = shard[2]
            .to_str()
            .ok_or("FILE_BYTES must be an unsigned integer")?
            .parse()?;
        stored.push((name, header, file_bytes));
    }
    let evidence = stored
        .iter()
        .map(|(name, header, bytes)| HeaderEvidence::new(name, header, *bytes))
        .collect::<ferrum_types::Result<Vec<_>>>()?;
    let structure = QwenImage21Structure::inspect(&config, &index, &evidence, limits)?;
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, &structure)?;
    writeln!(stdout)?;
    Ok(())
}
