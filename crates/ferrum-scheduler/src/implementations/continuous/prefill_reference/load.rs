use super::*;
use std::{fs::File, io::Read};

pub fn load_prefill_reference(
    path: &Path,
    fingerprint: &ExecutionFingerprint,
    expected_protocol_sha256: [u8; 32],
    limits: &SloPrefillReferenceLimits,
) -> Result<Arc<LoadedPrefillReference>, ReferenceError> {
    limits
        .validate()
        .map_err(|_| ReferenceError::Limit("configuration"))?;
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(ReferenceError::Evidence("artifact must be a regular file"));
    }
    // Same bounded regular-file pattern as cost_profile. No metadata size is
    // trusted as permission to allocate or read an unbounded stream.
    let mut bytes = Vec::new();
    file.take(limits.max_file_bytes.get() as u64 + 1)
        .read_to_end(&mut bytes)?;
    let mut loaded = parse(&bytes, fingerprint, expected_protocol_sha256, limits)?;
    loaded.source_path = Some(path.canonicalize()?);
    Ok(Arc::new(loaded))
}

pub fn load_prefill_reference_bytes(
    bytes: &[u8],
    fingerprint: &ExecutionFingerprint,
    expected_protocol_sha256: [u8; 32],
    limits: &SloPrefillReferenceLimits,
) -> Result<Arc<LoadedPrefillReference>, ReferenceError> {
    parse(bytes, fingerprint, expected_protocol_sha256, limits).map(Arc::new)
}

fn parse(
    bytes: &[u8],
    fingerprint: &ExecutionFingerprint,
    expected_protocol_sha256: [u8; 32],
    limits: &SloPrefillReferenceLimits,
) -> Result<LoadedPrefillReference, ReferenceError> {
    limits
        .validate()
        .map_err(|_| ReferenceError::Limit("configuration"))?;
    if bytes.len() > limits.max_file_bytes.get() {
        return Err(ReferenceError::Limit("file bytes"));
    }
    #[derive(Deserialize)]
    struct Schema {
        schema_version: u32,
    }
    let schema: Schema = serde_json::from_slice(bytes)?;
    if schema.schema_version == PREFILL_REFERENCE_SCHEMA_V2 {
        let value: ReferenceCalibrationV2 = serde_json::from_slice(bytes)?;
        let (evidence, spec) = value.into_evidence()?;
        return build::compile_partitioned(
            &evidence,
            fingerprint,
            expected_protocol_sha256,
            limits,
            Sha256::digest(bytes).into(),
            Some(&spec),
        );
    }
    let artifact: ReferenceCalibrationV1 = serde_json::from_slice(bytes)?;
    build::compile(
        &artifact,
        fingerprint,
        expected_protocol_sha256,
        limits,
        Sha256::digest(bytes).into(),
    )
}
