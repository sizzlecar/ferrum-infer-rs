//! Independent ATen propagation from original inputs through every block port.
//! No candidate or Reference intermediate is an input to this oracle.
use super::{artifacts, comparison, linear::MathPolicy, native, source};
use anyhow::{ensure, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21AttentionLayout, QwenImage21Block0Broadcast, QwenImage21Block0Config,
    QwenImage21Block0Family as Block,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::{
    HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits, ManagedHostBytes,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};

#[path = "torch_block_oracle/replay.rs"]
mod replay;
#[cfg(test)]
#[path = "torch_block_oracle/tests.rs"]
mod tests;

const PROGRAM: native::NativeSource = native::NativeSource {
    filename: "torch_block_oracle.cpp",
    text: include_str!("../native/torch_block_oracle.cpp"),
    required_header: "ATen/ops/linear.h",
    cache_prefix: "torch-block",
};
const MAX_WEIGHT_BYTES: u64 = 128 * 1024 * 1024;
const MAX_TOTAL_WEIGHT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_OUTPUT_BYTES: u64 = 16 * 1024 * 1024;
const ARITHMETIC: &str = "bf16-boundaries.f32-two-pass-norm.f32-dense-softmax.identity-rope.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Geometry {
    rows: u64,
    heads: u64,
    head_dim: u64,
    mlp_width: u64,
    epsilon_f32_bits: u32,
    scale_f32_bits: u32,
}
impl Geometry {
    fn hidden(self) -> Result<u64> {
        self.heads
            .checked_mul(self.head_dim)
            .context("block hidden dimension overflow")
    }
    fn validate(self) -> Result<()> {
        ensure!(
            (1..=16).contains(&self.rows)
                && (1..=512).contains(&self.heads)
                && (2..=256).contains(&self.head_dim)
                && self.head_dim.is_power_of_two()
                && (1..=65536).contains(&self.mlp_width)
                && self.hidden()? <= 65536,
            "block geometry exceeds declared oracle scope"
        );
        for bits in [self.epsilon_f32_bits, self.scale_f32_bits] {
            let value = f32::from_bits(bits);
            ensure!(
                value.is_finite() && value > 0.0,
                "block epsilon/scale must be finite positive F32"
            );
        }
        let weights = self.weight_shapes()?;
        let mut total = 0_u64;
        for shape in weights {
            let bytes = tensor_bytes(&shape, 2)?;
            ensure!(bytes <= MAX_WEIGHT_BYTES, "block weight exceeds 128 MiB");
            total = total
                .checked_add(bytes)
                .context("block weight total overflow")?;
        }
        ensure!(
            total <= MAX_TOTAL_WEIGHT_BYTES,
            "block weights exceed total 512 MiB budget"
        );
        ensure!(
            self.output_bytes()? <= MAX_OUTPUT_BYTES,
            "block output exceeds 16 MiB"
        );
        Ok(())
    }
    fn input_shapes(self) -> Result<[Vec<u64>; 6]> {
        let matrix = vec![self.rows, self.hidden()?];
        Ok([
            matrix.clone(),
            matrix.clone(),
            matrix.clone(),
            matrix.clone(),
            matrix,
            vec![self.rows, self.head_dim / 2, 2],
        ])
    }
    fn weight_shapes(self) -> Result<[Vec<u64>; 9]> {
        let h = self.hidden()?;
        Ok([
            vec![h, h],
            vec![h, h],
            vec![self.head_dim],
            vec![self.head_dim],
            vec![h, h],
            vec![h, h],
            vec![self.mlp_width, h],
            vec![self.mlp_width, h],
            vec![h, self.mlp_width],
        ])
    }
    fn boundaries(self) -> Result<Vec<Boundary>> {
        Block::OUTPUT_IDS
            .iter()
            .enumerate()
            .map(|(index, id)| {
                let shape = vec![
                    self.rows,
                    if (16..=18).contains(&index) {
                        self.mlp_width
                    } else {
                        self.hidden()?
                    },
                ];
                Ok(Boundary {
                    value_id: (*id).into(),
                    bytes: tensor_bytes(&shape, 2)?,
                    shape,
                })
            })
            .collect()
    }
    fn output_bytes(self) -> Result<u64> {
        self.boundaries()?.iter().try_fold(0_u64, |sum, b| {
            sum.checked_add(b.bytes)
                .context("block output sum overflow")
        })
    }
}
fn tensor_bytes(shape: &[u64], element: u64) -> Result<u64> {
    ensure!(
        !shape.is_empty() && !shape.contains(&0),
        "tensor shape must be nonempty positive"
    );
    shape.iter().try_fold(element, |n, d| {
        n.checked_mul(*d).context("tensor byte overflow")
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SegmentLocation {
    path: PathBuf,
    offset: u64,
    bytes: u64,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Segment {
    location: SegmentLocation,
    sha256: String,
}
impl Segment {
    fn verify(&self) -> Result<()> {
        let l = &self.location;
        ensure!(
            l.path.is_absolute()
                && (1..=MAX_WEIGHT_BYTES).contains(&l.bytes)
                && self.sha256.len() == 64
                && self
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid bounded block source segment"
        );
        let mut file = File::open(&l.path)?;
        let end = l
            .offset
            .checked_add(l.bytes)
            .context("source extent overflow")?;
        ensure!(
            file.metadata()?.is_file() && end <= file.metadata()?.len(),
            "source segment exceeds file"
        );
        file.seek(SeekFrom::Start(l.offset))?;
        let mut remaining = l.bytes;
        let mut buffer = [0_u8; 64 * 1024];
        let mut hash = Sha256::new();
        while remaining > 0 {
            let n = usize::try_from(remaining.min(buffer.len() as u64))?;
            file.read_exact(&mut buffer[..n])?;
            hash.update(&buffer[..n]);
            remaining -= n as u64;
        }
        ensure!(
            format!("{:x}", hash.finalize()) == self.sha256,
            "block source SHA mismatch"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    geometry: Geometry,
    policy: MathPolicy,
    inputs: [Segment; 6],
    weights: [Segment; 9],
}
impl Case {
    fn validate(&self) -> Result<()> {
        self.geometry.validate()?;
        for (i, shape) in self.geometry.input_shapes()?.iter().enumerate() {
            ensure!(
                self.inputs[i].location.bytes == tensor_bytes(shape, if i == 5 { 4 } else { 2 })?,
                "original block input extent differs"
            );
        }
        for (i, shape) in self.geometry.weight_shapes()?.iter().enumerate() {
            ensure!(
                self.weights[i].location.bytes == tensor_bytes(shape, 2)?,
                "complete block weight extent differs"
            );
        }
        Ok(())
    }
    fn sources(&self) -> impl Iterator<Item = &Segment> {
        self.inputs.iter().chain(self.weights.iter())
    }
    fn verify_sources(&self) -> Result<()> {
        self.validate()?;
        for segment in self.sources() {
            segment.verify()?;
        }
        // This independent oracle is qualified only for the original identity
        // RoPE probe. It still computes every adjacent pair rather than taking
        // a candidate's rotated boundary as an input.
        let mut file = File::open(&self.inputs[5].location.path)?;
        file.seek(SeekFrom::Start(self.inputs[5].location.offset))?;
        for _ in 0..self.inputs[5].location.bytes / 8 {
            let mut pair = [0_u8; 8];
            file.read_exact(&mut pair)?;
            ensure!(
                pair[..4] == 1_f32.to_le_bytes() && pair[4..] == 0_f32.to_le_bytes(),
                "block oracle requires declared identity RoPE probe"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Boundary {
    value_id: String,
    shape: Vec<u64>,
    bytes: u64,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    operation: String,
    arithmetic: String,
    torch_version: String,
    torch_config: String,
    cxx11_abi: u8,
    device: u32,
    policy: MathPolicy,
    geometry: Geometry,
    sources: Vec<SegmentLocation>,
    boundaries: Vec<Boundary>,
    bf16_reduction_option: i32,
    allow_tf32: bool,
    float32_matmul_precision: i32,
    blas_preferred_backend: i32,
    deterministic_algorithms: bool,
    cuda_runtime_version: i64,
    loaded_libraries: Vec<PathBuf>,
}
struct Output {
    metadata: Metadata,
    bytes: ManagedHostBytes,
    library_sha256: BTreeMap<PathBuf, String>,
}
struct Oracle {
    native: native::Oracle,
}
impl Oracle {
    fn prepare(config: native::Config) -> Result<Self> {
        Ok(Self {
            native: native::Oracle::prepare_source(config, PROGRAM)?,
        })
    }
    fn run(&mut self, case: &Case, host: &HostMemoryAccount) -> Result<Output> {
        case.verify_sources()?;
        let bytes = case.geometry.output_bytes()?;
        ensure!(
            bytes <= self.native.config.maximum_payload_bytes,
            "block output exceeds configured payload budget"
        );
        let temporary = tempfile::tempdir()?;
        let path = temporary.path().join("block.bin");
        let mut command = self.native.config.child(&self.native.executable)?;
        for name in [
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "TORCH_BLAS_PREFER_CUBLASLT",
            "TORCH_BLAS_PREFER_HIPBLASLT",
            "NVIDIA_TF32_OVERRIDE",
            "CUBLAS_WORKSPACE_CONFIG",
            "CUBLAS_WORKSPACE_SIZE",
            "CUBLASLT_WORKSPACE_SIZE",
        ] {
            command.env_remove(name);
        }
        let g = case.geometry;
        command.args([
            self.native.config.device.to_string(),
            case.policy.as_str().into(),
            g.rows.to_string(),
            g.heads.to_string(),
            g.head_dim.to_string(),
            g.mlp_width.to_string(),
            g.epsilon_f32_bits.to_string(),
            g.scale_f32_bits.to_string(),
        ]);
        for source in case.sources() {
            command
                .arg(&source.location.path)
                .arg(source.location.offset.to_string());
        }
        let execution = native::run(
            &mut command,
            &path,
            &temporary.path().join("stderr.txt"),
            self.native.config.timeout_seconds,
            16 + native::METADATA_LIMIT + 8 + bytes,
        );
        // Check even after a child failure; never substitute a successful exit
        // for immutable inputs or a complete numerical comparison.
        let unchanged = case.verify_sources();
        execution?;
        unchanged?;
        let mut output = parse(&path, &self.native.config, case, host)?;
        output.library_sha256 = self
            .native
            .validate_mapped_libraries(&output.metadata.loaded_libraries)?;
        ensure!(
            output.library_sha256.keys().any(|p| p
                .file_name()
                .and_then(|p| p.to_str())
                .is_some_and(
                    |n| n.starts_with("libcublas.so.") || n.starts_with("libcublasLt.so.")
                )),
            "block oracle did not map identifiable cuBLAS"
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
    case.validate()?;
    let expected = case.geometry.output_bytes()?;
    ensure!(
        expected <= config.maximum_payload_bytes,
        "block output exceeds configured payload budget"
    );
    let mut file = File::open(path)?;
    let mut header = [0_u8; 16];
    file.read_exact(&mut header)?;
    ensure!(
        &header[..8] == b"FBLKOR01" && u32::from_le_bytes(header[8..12].try_into()?) == 1,
        "invalid block oracle protocol"
    );
    let len = u32::from_le_bytes(header[12..].try_into()?) as u64;
    ensure!(
        len <= native::METADATA_LIMIT && file.metadata()?.len() == 16 + len + 8 + expected,
        "block frame extent/trailing data mismatch"
    );
    let mut encoded = vec![0; usize::try_from(len)?];
    file.read_exact(&mut encoded)?;
    let metadata: Metadata = serde_json::from_slice(&encoded)?;
    ensure!(
        metadata.operation == "aten.qwen_image21.block0"
            && metadata.arithmetic == ARITHMETIC
            && metadata.torch_version == config.expected_torch_version
            && metadata.cxx11_abi == config.cxx11_abi
            && metadata.device == config.device
            && metadata.policy == case.policy
            && metadata.geometry == case.geometry
            && metadata.sources
                == case
                    .sources()
                    .map(|s| s.location.clone())
                    .collect::<Vec<_>>()
            && metadata.boundaries == case.geometry.boundaries()?,
        "block oracle deployment/source/boundary identity mismatch"
    );
    ensure!(
        !metadata.allow_tf32 && metadata.cuda_runtime_version > 0,
        "block F32 attention requires observed TF32 disabled and actual CUDA runtime"
    );
    ensure!(
        case.policy != MathPolicy::NoReducedPrecision || metadata.bf16_reduction_option == 1,
        "deployed NoReducedPrecision must report DisallowReducedPrecisionAllowSplitK"
    );
    ensure!(
        metadata.loaded_libraries.len() <= 1024
            && metadata.loaded_libraries.iter().all(|p| p.is_absolute()),
        "invalid block mapped-library identities"
    );
    let mut count = [0_u8; 8];
    file.read_exact(&mut count)?;
    ensure!(
        u64::from_le_bytes(count) == expected,
        "block payload count differs"
    );
    let mut storage = host.allocate_zeroed(HostMemoryClass::Output, usize::try_from(expected)?)?;
    file.read_exact(storage.as_mut())?;
    Ok(Output {
        metadata,
        bytes: storage.freeze(),
        library_sha256: BTreeMap::new(),
    })
}
