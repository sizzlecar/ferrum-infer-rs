//! Full-geometry ATen linear replay, independent of Ferrum candidate outputs.
//! File segments refer to immutable existing artifacts/checkpoint shards. Rust
//! verifies their bytes before and after the child; this is local integrity
//! evidence, not protection against a malicious concurrently changing source.
use super::native;
use anyhow::{ensure, Context, Result};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBytes};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

const PROGRAM: native::NativeSource = native::NativeSource {
    filename: "torch_linear_oracle.cpp",
    text: include_str!("../native/torch_linear_oracle.cpp"),
    required_header: "ATen/ops/linear.h",
    cache_prefix: "torch-linear",
};
pub const MAX_WEIGHT_BYTES: u64 = 64 * 1024 * 1024;
const CLEARED_MATH_ENVIRONMENT: [&str; 7] = [
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "TORCH_BLAS_PREFER_CUBLASLT",
    "TORCH_BLAS_PREFER_HIPBLASLT",
    "NVIDIA_TF32_OVERRIDE",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLAS_WORKSPACE_SIZE",
    "CUBLASLT_WORKSPACE_SIZE",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MathPolicy {
    LibraryDefaults,
    NoReducedPrecision,
}
impl MathPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LibraryDefaults => "library_defaults",
            Self::NoReducedPrecision => "no_reduced_precision",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileSegment {
    pub path: PathBuf,
    pub offset: u64,
    pub bytes: u64,
    pub sha256: String,
}
impl FileSegment {
    pub fn verify(&self) -> Result<()> {
        ensure!(
            self.path.is_absolute() && (1..=MAX_WEIGHT_BYTES).contains(&self.bytes),
            "oracle file segment needs an absolute path and bounded nonempty extent"
        );
        ensure!(
            self.sha256.len() == 64
                && self
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "oracle segment SHA must be lowercase SHA256"
        );
        let mut file = File::open(&self.path)?;
        let end = self
            .offset
            .checked_add(self.bytes)
            .context("oracle segment overflow")?;
        ensure!(
            file.metadata()?.is_file() && end <= file.metadata()?.len(),
            "oracle segment exceeds regular file"
        );
        file.seek(SeekFrom::Start(self.offset))?;
        let mut remaining = self.bytes;
        let mut buffer = [0_u8; 64 * 1024];
        let mut digest = Sha256::new();
        while remaining != 0 {
            let count = usize::try_from(remaining.min(buffer.len() as u64))?;
            file.read_exact(&mut buffer[..count])?;
            digest.update(&buffer[..count]);
            remaining -= count as u64;
        }
        ensure!(
            format!("{:x}", digest.finalize()) == self.sha256,
            "oracle segment SHA mismatch"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub rows: u64,
    pub in_features: u64,
    pub out_features: u64,
    pub policy: MathPolicy,
    pub input: FileSegment,
    pub weight: FileSegment,
}
impl Case {
    pub fn output_bytes(&self) -> Result<u64> {
        ensure!(
            (1..=16).contains(&self.rows)
                && (1..=65536).contains(&self.in_features)
                && (1..=65536).contains(&self.out_features),
            "linear geometry exceeds declared scope"
        );
        let input = self
            .rows
            .checked_mul(self.in_features)
            .and_then(|n| n.checked_mul(2))
            .context("linear input size overflow")?;
        let weight = self
            .out_features
            .checked_mul(self.in_features)
            .and_then(|n| n.checked_mul(2))
            .context("linear weight size overflow")?;
        ensure!(
            weight <= MAX_WEIGHT_BYTES && self.input.bytes == input && self.weight.bytes == weight,
            "linear input/full OI weight extent mismatch"
        );
        self.rows
            .checked_mul(self.out_features)
            .and_then(|n| n.checked_mul(2))
            .context("linear output size overflow")
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub operation: String,
    pub torch_version: String,
    pub torch_config: String,
    pub cxx11_abi: u8,
    pub device: u32,
    pub policy: MathPolicy,
    pub rows: u64,
    pub in_features: u64,
    pub out_features: u64,
    pub dtype: String,
    pub input_path: PathBuf,
    pub input_offset: u64,
    pub weight_path: PathBuf,
    pub weight_offset: u64,
    /// Raw enum/bool ordinal from the actual version's ATen Context header.
    pub bf16_reduction_option: i32,
    pub allow_tf32: bool,
    pub float32_matmul_precision: i32,
    pub blas_preferred_backend: i32,
    pub deterministic_algorithms: bool,
    pub cuda_runtime_version: i64,
    pub loaded_libraries: Vec<PathBuf>,
}
pub struct Output {
    pub metadata: Metadata,
    pub bytes: ManagedHostBytes,
    pub library_sha256: BTreeMap<PathBuf, String>,
}

pub struct Oracle {
    pub native: native::Oracle,
}
impl Oracle {
    pub fn prepare(config: native::Config) -> Result<Self> {
        Ok(Self {
            native: native::Oracle::prepare_source(config, PROGRAM)?,
        })
    }

    pub fn linear(&mut self, case: &Case, host: &HostMemoryAccount) -> Result<Output> {
        let bytes = case.output_bytes()?;
        ensure!(
            bytes <= self.native.config.maximum_payload_bytes,
            "linear output exceeds configured payload bound"
        );
        case.input.verify()?;
        case.weight.verify()?;
        let temporary = tempfile::tempdir()?;
        let output_path = temporary.path().join("linear.bin");
        let stderr_path = temporary.path().join("linear.stderr");
        let mut command = self.native.config.child(&self.native.executable)?;
        // These are explicit test policies, not inherited hidden math controls.
        for name in CLEARED_MATH_ENVIRONMENT {
            command.env_remove(name);
        }
        command
            .arg(self.native.config.device.to_string())
            .arg(case.policy.as_str())
            .arg(case.rows.to_string())
            .arg(case.in_features.to_string())
            .arg(case.out_features.to_string())
            .arg(&case.input.path)
            .arg(case.input.offset.to_string())
            .arg(&case.weight.path)
            .arg(case.weight.offset.to_string());
        native::run(
            &mut command,
            &output_path,
            &stderr_path,
            self.native.config.timeout_seconds,
            16 + native::METADATA_LIMIT + 8 + bytes,
        )?;
        case.input.verify()?;
        case.weight.verify()?;
        let mut output = parse(&output_path, &self.native.config, case, host)?;
        output.library_sha256 = self
            .native
            .validate_mapped_libraries(&output.metadata.loaded_libraries)?;
        ensure!(
            output.library_sha256.keys().any(|path| path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(
                    |name| name.starts_with("libcublas.so.") || name.starts_with("libcublasLt.so.")
                )),
            "linear oracle did not map an identifiable cuBLAS library"
        );
        Ok(output)
    }
}

fn parse(
    path: &Path,
    config: &native::Config,
    case: &Case,
    host: &HostMemoryAccount,
) -> Result<Output> {
    let expected = case.output_bytes()?;
    ensure!(
        expected <= config.maximum_payload_bytes,
        "linear output exceeds configured bound"
    );
    let mut file = File::open(path)?;
    let mut header = [0_u8; 16];
    file.read_exact(&mut header)?;
    ensure!(
        &header[..8] == b"FLINOR01" && u32::from_le_bytes(header[8..12].try_into()?) == 1,
        "invalid linear oracle protocol/version"
    );
    let len = u32::from_le_bytes(header[12..].try_into()?) as u64;
    ensure!(
        len <= native::METADATA_LIMIT && file.metadata()?.len() == 16 + len + 8 + expected,
        "linear oracle metadata/extent mismatch, truncated or trailing data"
    );
    let mut encoded = vec![0; usize::try_from(len)?];
    file.read_exact(&mut encoded)?;
    let metadata: Metadata = serde_json::from_slice(&encoded)?;
    ensure!(
        metadata.operation == "aten.linear.no_bias"
            && metadata.dtype == "BF16"
            && metadata.torch_version == config.expected_torch_version
            && metadata.cxx11_abi == config.cxx11_abi
            && metadata.device == config.device
            && metadata.policy == case.policy
            && (metadata.rows, metadata.in_features, metadata.out_features)
                == (case.rows, case.in_features, case.out_features)
            && metadata.input_path == case.input.path
            && metadata.input_offset == case.input.offset
            && metadata.weight_path == case.weight.path
            && metadata.weight_offset == case.weight.offset,
        "linear oracle operation/deployment/input identity mismatch"
    );
    ensure!(
        metadata.loaded_libraries.len() <= 1024
            && metadata.loaded_libraries.iter().all(|p| p.is_absolute()),
        "invalid linear mapped-library identity"
    );
    let mut count = [0_u8; 8];
    file.read_exact(&mut count)?;
    ensure!(
        u64::from_le_bytes(count) == expected,
        "linear output payload count mismatch"
    );
    let mut storage = host.allocate_zeroed(HostMemoryClass::Output, usize::try_from(expected)?)?;
    file.read_exact(storage.as_mut())?;
    Ok(Output {
        metadata,
        bytes: storage.freeze(),
        library_sha256: BTreeMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

    fn config() -> native::Config {
        serde_json::from_value(serde_json::json!({
            "torch_root":"/explicit/torch","library_directories":["/explicit/torch/lib"],
            "executable":{"mode":"prebuilt","path":"/explicit/oracle","sha256":"0".repeat(64)},
            "cxx11_abi":1,"expected_torch_version":"test-version","expected_cuda_runtime_soname":"libcudart.so.13",
            "deployment_profile":"test-only","device":0,"timeout_seconds":30,"maximum_payload_bytes":1024
        })).unwrap()
    }
    fn case() -> Case {
        let segment = |path| FileSegment {
            path: PathBuf::from(path),
            offset: 0,
            bytes: 4,
            sha256: "0".repeat(64),
        };
        Case {
            rows: 1,
            in_features: 2,
            out_features: 1,
            policy: MathPolicy::LibraryDefaults,
            input: segment("/inputs/x"),
            weight: segment("/inputs/w"),
        }
    }
    fn host(bytes: usize) -> HostMemoryAccount {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                bytes,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 0,
                    output_bytes: bytes,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
    }
    fn frame(case: &Case) -> Vec<u8> {
        let metadata = serde_json::json!({
            "operation":"aten.linear.no_bias","torch_version":"test-version","torch_config":"fixture","cxx11_abi":1,"device":0,
            "policy":case.policy,"rows":case.rows,"in_features":case.in_features,"out_features":case.out_features,"dtype":"BF16",
            "input_path":case.input.path,"input_offset":case.input.offset,"weight_path":case.weight.path,"weight_offset":case.weight.offset,
            "bf16_reduction_option":0,"allow_tf32":false,"float32_matmul_precision":0,"blas_preferred_backend":0,
            "deterministic_algorithms":false,"cuda_runtime_version":13000,"loaded_libraries":[]
        });
        let encoded = serde_json::to_vec(&metadata).unwrap();
        let mut frame = b"FLINOR01".to_vec();
        frame.extend(1u32.to_le_bytes());
        frame.extend((encoded.len() as u32).to_le_bytes());
        frame.extend(encoded);
        frame.extend(2u64.to_le_bytes());
        frame.extend(0x3f80u16.to_le_bytes());
        frame
    }
    #[test]
    fn linear_protocol_is_distinct_strict_and_managed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("linear.bin");
        let case = case();
        let bytes = frame(&case);
        std::fs::write(&path, &bytes).unwrap();
        let host = host(2);
        let output = parse(&path, &config(), &case, &host).unwrap();
        assert_eq!(output.bytes.bytes(), 0x3f80u16.to_le_bytes());
        assert!(host.owns(&output.bytes));
        drop(output);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        let mut changed = case.clone();
        changed.policy = MathPolicy::NoReducedPrecision;
        assert!(parse(&path, &config(), &changed, &host).is_err());
        changed = case.clone();
        changed.weight.offset = 8;
        assert!(parse(&path, &config(), &changed, &host).is_err());
        for malformed in [
            bytes[..bytes.len() - 1].to_vec(),
            [bytes.as_slice(), &[0]].concat(),
            [b"FRNGOR01".as_slice(), &bytes[8..]].concat(),
        ] {
            std::fs::write(&path, malformed).unwrap();
            assert!(parse(&path, &config(), &case, &host).is_err());
        }
        std::fs::write(&path, bytes).unwrap();
        let too_small = self::host(1);
        assert!(parse(&path, &config(), &case, &too_small).is_err());
        assert_eq!(too_small.usage().unwrap().committed_bytes(), 0);
    }
    #[test]
    fn original_file_segments_are_checked_without_copying_a_weight_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard");
        std::fs::write(&path, [9, 8, 1, 2, 3, 4, 7]).unwrap();
        let mut segment = FileSegment {
            path,
            offset: 2,
            bytes: 4,
            sha256: native::bytes_sha(&[1, 2, 3, 4]),
        };
        segment.verify().unwrap();
        segment.offset = 3;
        assert!(segment.verify().is_err());
        segment.offset = u64::MAX;
        assert!(segment.verify().is_err());
        let mut invalid = case();
        invalid.out_features = 65536;
        assert!(invalid.output_bytes().is_err());
        invalid = case();
        invalid.rows = 0;
        assert!(invalid.output_bytes().is_err());
    }
}
