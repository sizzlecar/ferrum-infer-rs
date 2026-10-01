//! Cache byte-storage feasibility only. These tests do not qualify a model.
use flate2::{bufread::GzDecoder, write::GzEncoder, Compression};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    fs::File,
    io::{self, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

const CACHE_LIMIT: u64 = 256 * 1024 * 1024;
const IMPORT_LIMIT: u64 = 256 * 1024 * 1024;
const RECORD_LIMIT: u64 = 8 * 1024 * 1024;

struct ChargedWriter<W> {
    inner: W,
    used: Rc<Cell<u64>>,
    limit: u64,
    stored: u64,
    hash: Sha256,
}
impl<W: Write> Write for ChargedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self.used.get().checked_add(bytes.len() as u64);
        let Some(next) = next.filter(|n| *n <= self.limit) else {
            return Err(io::Error::other("original codec disk capacity"));
        };
        // Charge before the write, retaining failed/partial-write reservations.
        self.used.set(next);
        self.inner.write_all(bytes)?;
        self.stored += bytes.len() as u64;
        self.hash.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
fn writer<W>(inner: W, used: Rc<Cell<u64>>, limit: u64) -> ChargedWriter<W> {
    ChargedWriter {
        inner,
        used,
        limit,
        stored: 0,
        hash: Sha256::new(),
    }
}

fn decoded_receipt<R: BufRead>(
    reader: R,
    limit: u64,
    prefix: u64,
) -> io::Result<(u64, [u8; 32], [u8; 32])> {
    let mut decoder = GzDecoder::new(reader);
    let mut buffer = [0u8; 8192];
    let mut total = 0u64;
    let mut line = 0u64;
    let mut raw_hash = Sha256::new();
    let mut prefix_hash = Sha256::new();
    loop {
        let n = decoder.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        let end = total
            .checked_add(n as u64)
            .ok_or_else(|| io::Error::other("decoded overflow"))?;
        if end > limit {
            return Err(io::Error::other("original decoded capacity"));
        }
        let prefix_n = prefix.saturating_sub(total).min(n as u64) as usize;
        prefix_hash.update(&buffer[..prefix_n]);
        raw_hash.update(&buffer[..n]);
        for byte in &buffer[..n] {
            line += 1;
            if line > RECORD_LIMIT {
                return Err(io::Error::other("original record capacity"));
            }
            if *byte == b'\n' {
                line = 0;
            }
        }
        total = end;
    }
    if line != 0 || prefix > total || !decoder.into_inner().fill_buf()?.is_empty() {
        return Err(io::Error::other(
            "incomplete original or trailing stored bytes",
        ));
    }
    Ok((
        total,
        raw_hash.finalize().into(),
        prefix_hash.finalize().into(),
    ))
}

#[test]
fn original_codec_enforces_emitted_disk_and_decoded_original_limits() {
    let input = b"{\"original\":12345}\n".repeat(1000);
    let used = Rc::new(Cell::new(0));
    let mut gzip = GzEncoder::new(writer(Vec::new(), used.clone(), 4096), Compression::fast());
    gzip.write_all(&input).unwrap();
    let encoded = gzip.finish().unwrap();
    assert_eq!(encoded.stored, used.get());
    assert_eq!(encoded.stored, encoded.inner.len() as u64);
    let exact = decoded_receipt(encoded.inner.as_slice(), input.len() as u64, 19).unwrap();
    assert_eq!(exact.1, <[u8; 32]>::from(Sha256::digest(&input)));
    assert_eq!(exact.2, <[u8; 32]>::from(Sha256::digest(&input[..19])));
    assert!(decoded_receipt(encoded.inner.as_slice(), input.len() as u64 - 1, 19).is_err());
    assert!(decoded_receipt(&encoded.inner[..encoded.inner.len() - 1], IMPORT_LIMIT, 19).is_err());
    let mut corrupted = encoded.inner.clone();
    let crc = corrupted.len() - 8;
    corrupted[crc] ^= 1;
    assert!(decoded_receipt(corrupted.as_slice(), IMPORT_LIMIT, 19).is_err());
    let mut trailing = encoded.inner;
    trailing.push(0);
    assert!(decoded_receipt(trailing.as_slice(), IMPORT_LIMIT, 19).is_err());
    let used = Rc::new(Cell::new(0));
    let mut too_small = GzEncoder::new(writer(Vec::new(), used.clone(), 10), Compression::fast());
    assert!(too_small
        .write_all(&input)
        .and_then(|()| too_small.finish().map(|_| ()))
        .is_err());
    assert!(
        used.get() <= 10,
        "trailer and delayed output cannot bypass the same charge"
    );
}

fn hash_file(path: &Path) -> ([u8; 32], u64) {
    let mut source = File::open(path).unwrap();
    let mut digest = Sha256::new();
    let mut total = 0;
    let mut bytes = [0u8; 8192];
    loop {
        let n = source.read(&mut bytes).unwrap();
        if n == 0 {
            break;
        }
        total += n as u64;
        digest.update(&bytes[..n]);
    }
    (digest.finalize().into(), total)
}

#[test]
#[ignore = "requires an original source path; storage feasibility, not model qualification"]
fn original_codec_five_concurrent_sources_preserve_every_raw_byte() {
    let source = PathBuf::from(
        std::env::var_os("FERRUM_ORIGINAL_CODEC_SOURCE").expect("original canonical source path"),
    );
    let (raw_sha, raw_bytes) = hash_file(&source);
    assert!(raw_bytes > 0 && raw_bytes <= IMPORT_LIMIT);
    let prefix_bytes = BufReader::new(File::open(&source).unwrap())
        .read_until(b'\n', &mut Vec::new())
        .unwrap() as u64;
    let mut raw_prefix = vec![0; prefix_bytes as usize];
    File::open(&source)
        .unwrap()
        .read_exact(&mut raw_prefix)
        .unwrap();
    let prefix_sha: [u8; 32] = Sha256::digest(&raw_prefix).into();
    let directory = tempfile::tempdir().unwrap();
    // The same five source contents intentionally isolate byte storage; they
    // are not five independent calibration populations or replayed models.
    let used = Rc::new(Cell::new(0));
    let paths: Vec<_> = (0..5)
        .map(|n| directory.path().join(format!("source-{n}.gz")))
        .collect();
    let mut encoders: Vec<_> = paths
        .iter()
        .map(|path| {
            GzEncoder::new(
                writer(File::create(path).unwrap(), used.clone(), CACHE_LIMIT),
                Compression::fast(),
            )
        })
        .collect();
    let started = Instant::now();
    let mut maximum_five_writer_chunk_ns = 0;
    let mut reader = File::open(&source).unwrap();
    let mut buffer = [0u8; 8192];
    loop {
        let n = reader.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        let chunk_started = Instant::now();
        for encoder in &mut encoders {
            encoder.write_all(&buffer[..n]).unwrap();
        }
        maximum_five_writer_chunk_ns =
            maximum_five_writer_chunk_ns.max(chunk_started.elapsed().as_nanos());
    }
    let mut receipts = Vec::new();
    for encoder in encoders {
        let writer = encoder.finish().unwrap();
        writer.inner.sync_all().unwrap();
        receipts.push((writer.stored, <[u8; 32]>::from(writer.hash.finalize())));
    }
    let encoded_ns = started.elapsed().as_nanos();
    let decode_started = Instant::now();
    let mut decoded_remaining = IMPORT_LIMIT;
    for (path, expected) in paths.iter().zip(&receipts) {
        let actual = hash_file(path);
        assert_eq!((actual.1, actual.0), *expected);
        assert_eq!(
            decoded_receipt(
                BufReader::new(File::open(path).unwrap()),
                decoded_remaining,
                prefix_bytes
            )
            .unwrap(),
            (raw_bytes, raw_sha, prefix_sha)
        );
        decoded_remaining = decoded_remaining.checked_sub(raw_bytes).unwrap();
    }
    assert_eq!(used.get(), receipts.iter().map(|r| r.0).sum::<u64>());
    println!(
        "{}",
        serde_json::json!({
            "scope":"five byte-identical original streams; storage feasibility only",
            "source":source,"original_bytes":raw_bytes,"original_sha256":raw_sha,
            "original_prefix_bytes":prefix_bytes,"original_prefix_sha256":prefix_sha,
            "copies":5,"uncompressed_total":raw_bytes.checked_mul(5).unwrap(),
            "stored_total":used.get(),"fixed_disk_cap":CACHE_LIMIT,
            "transaction_decoded_cap":IMPORT_LIMIT,"decoded_remaining":decoded_remaining,
            "codec_state_memory_bound_validated":false,
            "model_qualification_or_transaction_replay_validated":false,
            "encoding_ns":encoded_ns,"maximum_five_writer_chunk_ns":maximum_five_writer_chunk_ns,
            "decode_and_verify_ns":decode_started.elapsed().as_nanos()
        })
    );
}
