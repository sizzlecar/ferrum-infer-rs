//! Read-only original source6/source7 diagnostics; not a profile importer.
use ferrum_scheduler::implementations::continuous::cost_profile::{
    diagnose_structured_service_owners_v6, CostProfileLoadLimits,
};
use std::{
    io::{BufRead, BufReader, Read},
    num::NonZeroUsize,
    path::PathBuf,
};

#[path = "inspect_service_owners/owner_blocks.rs"]
mod owner_blocks;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(
        args.next()
            .ok_or("usage: inspect_service_owners SOURCE6_OR_SOURCE7.jsonl")?,
    );
    if args.next().is_some() {
        return Err("usage: inspect_service_owners SOURCE6_OR_SOURCE7.jsonl".into());
    }
    let limits = CostProfileLoadLimits {
        max_file_bytes: NonZeroUsize::new(256 * 1024 * 1024).unwrap(),
        max_samples: NonZeroUsize::new(131_072).unwrap(),
        max_total_shape_rows: NonZeroUsize::new(1_048_576).unwrap(),
        ..CostProfileLoadLimits::default()
    };
    let mut header = Vec::new();
    BufReader::new(std::fs::File::open(&path)?)
        .take(8 * 1024 * 1024 + 1)
        .read_until(b'\n', &mut header)?;
    if header.len() > 8 * 1024 * 1024 || header.last() != Some(&b'\n') {
        return Err("incomplete or oversized source header".into());
    }
    let version = serde_json::from_slice::<serde_json::Value>(&header)?["schema_version"].as_u64();
    match version {
        Some(6) => println!(
            "{}",
            serde_json::to_string_pretty(&diagnose_structured_service_owners_v6(&path, &limits)?)?
        ),
        Some(7) => println!(
            "{}",
            serde_json::to_string_pretty(&owner_blocks::diagnose(&path, &limits)?)?
        ),
        _ => {
            return Err(
                "expected original source schema 6 or 7; profiles are not source journals".into(),
            )
        }
    }
    Ok(())
}
