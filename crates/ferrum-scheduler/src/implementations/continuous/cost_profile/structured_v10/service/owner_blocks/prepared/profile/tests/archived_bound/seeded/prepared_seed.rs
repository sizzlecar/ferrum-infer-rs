//! An original pretraffic source8 header can prove a cold declaration even
//! when its human-readable log did not print the universe digest. No later
//! source7 observation is used to construct or enlarge this seed.
use super::*;
use crate::implementations::continuous::cost_profile::StructuredPreparedOwnerBlockHeaderV8;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OriginalPreparedHeader {
    pub path: PathBuf,
    pub whole_sha256: [u8; 32],
    /// Includes the original canonical newline, as the source journal does.
    pub header_sha256: [u8; 32],
}

pub(super) fn verify(
    proof: &OriginalPreparedHeader,
    expected: &DeclaredAlgorithmUniverseV1,
    original: &StructuredServiceHeaderV7,
) {
    const HEADER_MAX: u64 = 8 * 1024 * 1024;
    let input = std::fs::File::open(&proof.path).unwrap();
    assert!(input.metadata().unwrap().is_file());
    let mut first = Vec::new();
    BufReader::new(input)
        .take(HEADER_MAX + 1)
        .read_until(b'\n', &mut first)
        .unwrap();
    assert!(!first.is_empty() && first.len() as u64 <= HEADER_MAX);
    assert_eq!(first.last(), Some(&b'\n'));
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&first)),
        proof.header_sha256
    );
    let header: StructuredPreparedOwnerBlockHeaderV8 = serde_json::from_slice(&first).unwrap();
    assert_eq!(
        record_bytes_v7(&header).unwrap(),
        first,
        "original canonical source8 header"
    );
    header.validate().unwrap();
    let mut input = std::fs::File::open(&proof.path).unwrap();
    assert!(input.metadata().unwrap().len() <= header.maximum_file_bytes);
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let count = input.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    assert_eq!(<[u8; 32]>::from(hash.finalize()), proof.whole_sha256);
    assert_eq!(header.fingerprint, original.fingerprint);
    assert_eq!(header.monotonic_domain, original.monotonic_domain);
    assert!(header.opening.monotonic_ns < original.opening.monotonic_ns);
    assert!(header.opening.wall_unix_ns < original.opening.wall_unix_ns);
    let prepared = header
        .declaration
        .population
        .nonnegative_envelope
        .as_ref()
        .unwrap();
    let live = original.declaration.nonnegative_envelope.as_ref().unwrap();
    assert_eq!(prepared.workload_domain, live.workload_domain);
    let declared = prepared
        .algorithm_universe
        .as_ref()
        .expect("original declared cold universe");
    assert_eq!(
        declared, expected,
        "exact original seed, never a union with later observations"
    );
    assert_eq!(
        declared.workload_domain_signature(),
        live.workload_domain.sha256()
    );
    eprintln!(
        "SEEDED_CANDIDATE_ORIGINAL_PREPARED_PROOF {}",
        json!({
            "source8_whole_sha256":proof.whole_sha256,
            "source8_header_sha256":proof.header_sha256,
            "cold_universe_signature":declared.signature(),
            "cold_algorithm_count":declared.algorithm_count(),
            "source8_opening":header.opening,
            "source7_opening":original.opening,
            "boundary":"only this original pretraffic declared subset; no claim that later uncaptured startup inventory was complete"
        })
    );
}
