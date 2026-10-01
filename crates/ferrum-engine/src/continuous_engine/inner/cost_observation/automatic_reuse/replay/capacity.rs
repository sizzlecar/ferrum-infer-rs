//! Preflight the actual serial replay shape, not N copies of its full history.
use super::*;

pub(super) fn replay_peak_bytes(
    raw: &[u8],
    numeric: usize,
    retained: usize,
    transaction: &ColdTransaction,
) -> Result<usize> {
    let mut maximum = 0;
    for line in raw.split_inclusive(|byte| *byte == b'\n') {
        transaction.poll()?;
        if line.len() > 8 * 1024 * 1024 {
            return Err(CacheMiss::Capacity);
        }
        if line.last() != Some(&b'\n') {
            return Err(CacheMiss::Corrupt);
        }
        maximum = maximum.max(line.len());
    }
    if maximum == 0 {
        return Err(CacheMiss::Corrupt);
    }
    peak(raw.len() as u64, maximum, numeric, retained)?
        .checked_add(codec::DECODER_RETAINED)
        .ok_or(CacheMiss::Capacity)
}

pub(super) fn replay_peak(
    path: &Path,
    journal_bytes: u64,
    numeric_and_discovery: usize,
    retained: usize,
    transaction: &ColdTransaction,
) -> Result<usize> {
    let maximum_record = largest_record(path, journal_bytes, transaction)?;
    peak(
        journal_bytes,
        maximum_record,
        numeric_and_discovery,
        retained,
    )
}
fn peak(
    journal_bytes: u64,
    maximum_record: usize,
    numeric: usize,
    retained: usize,
) -> Result<usize> {
    // read_bounded retains the original byte vector; replay splits borrowed
    // lines and the collector hashes (never stores) encoded history. Two raw
    // copies cover its Vec allocation/growth headroom. Typed record, canonical
    // Value and canonical bytes coexist for ONE actual line, not all history.
    // Collector numeric/discovery allocation already includes Fit workspaces.
    // Three declared populations conservatively cover collector plus checkpoint
    // and import/envelope transition; qualified model backing is Arc-shared.
    // Metadata's typed decode/encode is separately bounded by the real encoder.
    usize::try_from(journal_bytes)
        .ok()
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(maximum_record.checked_mul(12)?))
        .and_then(|n| n.checked_add(numeric.checked_mul(3)?))
        .and_then(|n| {
            n.checked_add(file::structured_owner_block_metadata_maximum_bytes().checked_mul(12)?)
        })
        .and_then(|n| n.checked_add(retained))
        .and_then(|n| n.checked_add(8192))
        .ok_or(CacheMiss::Capacity)
}
fn largest_record(path: &Path, declared: u64, transaction: &ColdTransaction) -> Result<usize> {
    transaction.poll()?;
    if declared == 0 || regular_bytes(path)? != declared {
        return Err(CacheMiss::Corrupt);
    }
    let mut file = File::open(path).map_err(io)?;
    let mut buffer = [0u8; 8192];
    let mut total = 0u64;
    let mut line = 0usize;
    let mut maximum = 0usize;
    loop {
        transaction.poll()?;
        let count = file.read(&mut buffer).map_err(io)?;
        if count == 0 {
            break;
        }
        total = total.checked_add(count as u64).ok_or(CacheMiss::Capacity)?;
        if total > declared {
            return Err(CacheMiss::Corrupt);
        }
        for byte in &buffer[..count] {
            line = line.checked_add(1).ok_or(CacheMiss::Capacity)?;
            // The original V7/V8 replay enforces this same per-record limit.
            if line > 8 * 1024 * 1024 {
                return Err(CacheMiss::Capacity);
            }
            if *byte == b'\n' {
                maximum = maximum.max(line);
                line = 0;
            }
        }
    }
    if total != declared || line != 0 || maximum == 0 {
        return Err(CacheMiss::Corrupt);
    }
    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn automatic_reuse_serial_replay_charges_history_once_and_largest_record_separately() {
        // Same bounded canonical-record sizes, different history lengths.
        // This tests allocation inventory, not qualification of fake samples.
        let numeric = 136 * 1024 * 1024;
        let small = peak(30 * 1024 * 1024, 32 * 1024, numeric, 1024).unwrap();
        let long = peak(68 * 1024 * 1024, 32 * 1024, numeric, 1024).unwrap();
        assert_eq!(long - small, 2 * 38 * 1024 * 1024);
        let wide = peak(68 * 1024 * 1024, 64 * 1024, numeric, 1024).unwrap();
        assert_eq!(wide - long, 12 * 32 * 1024);
        assert_eq!(peak(u64::MAX, 1, numeric, 0), Err(CacheMiss::Capacity));
        assert_eq!(peak(1, 1, usize::MAX, 0), Err(CacheMiss::Capacity));
        assert_eq!(peak(1, 1, 1, usize::MAX), Err(CacheMiss::Capacity));
    }
    #[test]
    fn automatic_reuse_record_scan_preserves_incomplete_and_exact_file_boundaries() {
        let path =
            std::env::temp_dir().join(format!("ferrum-reuse-records-{}", uuid::Uuid::new_v4()));
        let limits = CacheLimits {
            maximum_bytes: 1024,
            maximum_source_bytes: 1024,
            maximum_sources: 1,
            maximum_retained_bytes: 1024 * 1024,
            maximum_samples: 1,
            maximum_rows: 1,
            maximum_duration: Duration::from_secs(30),
        };
        let tx = ColdTransaction::new(limits).unwrap();
        fs::write(&path, b"{}\n{\"record\":1}\n").unwrap();
        let bytes = regular_bytes(&path).unwrap();
        assert_eq!(largest_record(&path, bytes, &tx), Ok(13));
        assert_eq!(
            largest_record(&path, bytes - 1, &tx),
            Err(CacheMiss::Corrupt)
        );
        fs::write(&path, b"{}\n{\"record\":1}").unwrap();
        assert_eq!(
            largest_record(&path, regular_bytes(&path).unwrap(), &tx),
            Err(CacheMiss::Corrupt)
        );
        fs::remove_file(path).unwrap();
    }
}
