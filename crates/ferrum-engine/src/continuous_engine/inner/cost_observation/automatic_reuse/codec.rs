//! Cache representation only: original source bytes and membership never change.
use super::*;
use flate2::{bufread::GzDecoder, write::GzEncoder, Compression};
use std::io::{BufRead, BufReader};

// Cargo.lock's flate2 1.1.5 / miniz_oxide 0.8.9 Rust backend owns fixed
// arrays: 65536 codes, 33026 dictionary, 2*65536 hash chains, 85196 local
// output, 4320 Huffman, plus flate2's 32768 output Vec, fixed gzip header,
// and struct/Box overhead. 512 KiB conservatively covers their total (<400
// KiB), including construction overlap. No original history is retained here.
pub(super) const ENCODER_RETAINED: usize = 512 * 1024;
// InflateState's 32 KiB dictionary + fixed Huffman state, 8 KiB BufReader,
// 8 KiB output scratch and fixed gzip header (optional header fields rejected).
pub(super) const DECODER_RETAINED: usize = 128 * 1024;

#[derive(Debug)]
struct WriteFailure(CacheMiss);
impl std::fmt::Display for WriteFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl std::error::Error for WriteFailure {}
fn write_error(reason: CacheMiss) -> std::io::Error {
    std::io::Error::other(WriteFailure(reason))
}
fn map_write(reason: std::io::Error) -> CacheMiss {
    reason
        .get_ref()
        .and_then(|e| e.downcast_ref::<WriteFailure>())
        .map_or(CacheMiss::Io, |e| e.0)
}

struct ChargedFile {
    file: File,
    session: CacheSession,
    index: usize,
    bytes: u64,
    hash: Sha256,
    disabled: bool,
}
impl Write for ChargedFile {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.disabled {
            return Err(write_error(CacheMiss::Dirty));
        }
        self.session
            .0
            .charge(bytes.len() as u64)
            .map_err(write_error)?;
        self.file.write_all(bytes)?;
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| write_error(CacheMiss::Capacity))?;
        self.hash.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}
enum Stream {
    Gzip(GzEncoder<ChargedFile>),
    Raw(ChargedFile),
}
pub(super) struct Writer {
    stream: Option<Stream>,
    session: CacheSession,
    reserved: usize,
}
impl Writer {
    pub fn new(session: CacheSession, index: usize, path: &Path, compressed: bool) -> Result<Self> {
        let reserved = if compressed { ENCODER_RETAINED } else { 0 };
        {
            let mut ledger = session.0.ledger.lock();
            let next = ledger
                .codec_retained
                .checked_add(reserved)
                .ok_or(CacheMiss::Capacity)?;
            if next > session.0.limits.maximum_retained_bytes {
                return Err(CacheMiss::Capacity);
            }
            ledger.codec_retained = next;
        }
        let mut out = Self {
            stream: None,
            session: session.clone(),
            reserved,
        };
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .map_err(io)?;
        let charged = ChargedFile {
            file,
            session,
            index,
            bytes: 0,
            hash: Sha256::new(),
            disabled: false,
        };
        out.stream = Some(if compressed {
            Stream::Gzip(GzEncoder::new(charged, Compression::fast()))
        } else {
            Stream::Raw(charged)
        });
        Ok(out)
    }
    pub fn append(&mut self, bytes: &[u8]) -> Result<()> {
        match self.stream.as_mut().ok_or(CacheMiss::Dirty)? {
            Stream::Gzip(writer) => writer.write_all(bytes),
            Stream::Raw(writer) => writer.write_all(bytes),
        }
        .map_err(map_write)
    }
    pub fn finish(mut self) -> Result<ArtifactReceipt> {
        let mut stream = self.stream.take().ok_or(CacheMiss::Dirty)?;
        // Keep custody on failure so Drop disables any implicit gzip retry.
        if let Stream::Gzip(writer) = &mut stream {
            if let Err(error) = writer.try_finish() {
                self.stream = Some(stream);
                return Err(map_write(error));
            }
        }
        let writer = match stream {
            Stream::Gzip(writer) => writer.finish().map_err(map_write)?,
            Stream::Raw(writer) => writer,
        };
        writer.file.sync_all().map_err(io)?;
        sync_directory(&self.session.0.generation_path)?;
        Ok(ArtifactReceipt {
            index: writer.index,
            bytes: writer.bytes,
            sha256: writer.hash.finalize().into(),
        })
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        if let Some(Stream::Gzip(writer)) = self.stream.as_mut() {
            writer.get_mut().disabled = true;
        }
        self.stream.take();
        let mut ledger = self.session.0.ledger.lock();
        ledger.codec_retained = ledger
            .codec_retained
            .checked_sub(self.reserved)
            .expect("codec memory ownership");
    }
}

/// Decode one complete stored artifact under the same cold transaction. The
/// full raw receipt is checked even when the chosen model uses an earlier cut.
pub(super) fn decode(
    session: &CacheSession,
    path: &Path,
    stored: ArtifactReceipt,
    encoding: JournalEncoding,
    maximum_raw: usize,
    retained: usize,
    transaction: &ColdTransaction,
) -> Result<Vec<u8>> {
    transaction.poll()?;
    let (raw_bytes, raw_sha) = encoding.original(stored);
    let length = usize::try_from(raw_bytes).map_err(|_| CacheMiss::Capacity)?;
    let resident = session.0.ledger.lock().codec_retained;
    if length == 0
        || length > maximum_raw
        || raw_bytes > session.0.limits.maximum_source_bytes
        || length
            .checked_mul(2)
            .and_then(|n| n.checked_add(DECODER_RETAINED))
            .and_then(|n| n.checked_add(retained))
            .and_then(|n| n.checked_add(resident))
            .is_none_or(|n| n > session.0.limits.maximum_retained_bytes)
    {
        return Err(CacheMiss::Capacity);
    }
    store::verify_file(path, &stored, transaction)?;
    let mut reader = BufReader::with_capacity(8192, File::open(path).map_err(io)?);
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| CacheMiss::Capacity)?;
    if output.capacity() != length {
        return Err(CacheMiss::Capacity);
    }
    let mut digest = Sha256::new();
    let mut line = 0usize;
    fn drain(
        reader: &mut impl Read,
        output: &mut Vec<u8>,
        length: usize,
        digest: &mut Sha256,
        line: &mut usize,
        tx: &ColdTransaction,
    ) -> Result<()> {
        let mut chunk = [0u8; 8192];
        loop {
            tx.poll()?;
            let n = reader.read(&mut chunk).map_err(|_| CacheMiss::Corrupt)?;
            if n == 0 {
                break;
            }
            if output.len().checked_add(n).is_none_or(|n| n > length) {
                return Err(CacheMiss::Capacity);
            }
            for byte in &chunk[..n] {
                *line += 1;
                if *line > 8 * 1024 * 1024 {
                    return Err(CacheMiss::Capacity);
                }
                if *byte == b'\n' {
                    *line = 0;
                }
            }
            digest.update(&chunk[..n]);
            output.extend_from_slice(&chunk[..n]);
        }
        Ok(())
    }
    match encoding {
        JournalEncoding::IdentityV1 => drain(
            &mut reader,
            &mut output,
            length,
            &mut digest,
            &mut line,
            transaction,
        )?,
        JournalEncoding::GzipV1 { .. } => {
            // Only our fixed header grammar is admitted, bounding header memory.
            let header = reader.fill_buf().map_err(io)?;
            if header.len() < 10 || header[..4] != [0x1f, 0x8b, 8, 0] {
                return Err(CacheMiss::Corrupt);
            }
            let mut decoder = GzDecoder::new(reader);
            drain(
                &mut decoder,
                &mut output,
                length,
                &mut digest,
                &mut line,
                transaction,
            )?;
            reader = decoder.into_inner();
            if !reader.fill_buf().map_err(io)?.is_empty() {
                return Err(CacheMiss::Corrupt);
            }
        }
    }
    if output.len() != length || line != 0 || <[u8; 32]>::from(digest.finalize()) != raw_sha {
        return Err(CacheMiss::Corrupt);
    }
    transaction.poll()?;
    Ok(output)
}
