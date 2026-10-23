//! Artifact identity and an explicitly limited Reference differential. These
//! thresholds are declared before execution, not inferred from observed error.
use super::{
    artifacts,
    backend::Backend,
    probe::{Inputs, OutputSpec},
    Family,
};
use anyhow::{bail, Context, Result};
use ferrum_models::qwen_image21::QwenImage21Block0Config;
use ferrum_quantization::SafetensorsArchive;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const MANIFEST_SCHEMA: &str = "ferrum.qwen_image21.block0-artifacts.v1";
pub const ABSOLUTE_TOLERANCE: f64 = 0.02;
pub const RELATIVE_TOLERANCE: f64 = 0.02;
const MANIFEST_LIMIT: usize = 2 * 1024 * 1024;

pub fn backend_name(backend: &Backend) -> &'static str {
    match backend {
        Backend::Reference => "reference",
        Backend::Cuda(_) => "cuda",
    }
}

pub fn declaration(with_reference: bool) -> Value {
    json!({"mode":if with_reference {"reference_differential"} else {"execution_smoke_only"},
        "finite_required_at_all_21_boundaries":true,
        "absolute_tolerance":ABSOLUTE_TOLERANCE,"relative_tolerance":RELATIVE_TOLERANCE,
        "criterion":"for every finite BF16 value: abs(candidate-reference) <= 0.02 + 0.02*abs(reference), evaluated in F64",
        "reference_identity":"exact canonical config, config-file SHA-256, all six input SHA-256 and all nine selected weight SHA-256; all reference output files checked against their recorded digests",
        "scope":"Reference artifact differential only; no independent whole-block oracle, Torch/vLLM parity or full-domain numerical qualification",
        "without_reference":"execution and finite-output smoke only; no numerical verification claim"})
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorDigest {
    pub value_id: String,
    pub file: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
    pub sha256: String,
}
impl TensorDigest {
    pub fn from_spec(spec: &OutputSpec, bytes: &[u8]) -> Result<Self> {
        if bytes.len() != spec.byte_len {
            bail!("output bytes differ from exact declared shape");
        }
        Ok(Self {
            value_id: spec.value_id.clone(),
            file: spec.file.clone(),
            dtype: "BF16".into(),
            shape: spec.shape.clone(),
            bytes: u64::try_from(bytes.len())?,
            sha256: artifacts::sha256(bytes),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightDigest {
    tensor: String,
    dtype: String,
    shape: Vec<u64>,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    canonical_config: QwenImage21Block0Config,
    config_file_sha256: String,
    pub inputs: Vec<TensorDigest>,
    weights: Vec<WeightDigest>,
}
impl Identity {
    pub fn new(
        config: &QwenImage21Block0Config,
        config_bytes: &[u8],
        inputs: &Inputs,
        source: &SafetensorsArchive,
    ) -> Result<Self> {
        let specs = config.input_tensor_specs()?;
        let mut input_evidence = Vec::new();
        for (index, (name, bytes)) in Family::INPUT_IDS.iter().zip(&inputs.bytes).enumerate() {
            let spec = &specs[&ferrum_runtime::ProgramValueId::new(*name)?];
            if spec.byte_len()? != bytes.len() as u64 {
                bail!("input payload differs from shape");
            }
            input_evidence.push(TensorDigest {
                value_id: (*name).into(),
                file: format!(
                    "input-{index:02}.{}",
                    if index == 5 { "f32" } else { "bf16" }
                ),
                dtype: if index == 5 { "F32" } else { "BF16" }.into(),
                shape: spec.dimensions.clone(),
                bytes: u64::try_from(bytes.len())?,
                sha256: artifacts::sha256(bytes),
            });
        }
        let mut weights = Vec::new();
        for (index, name) in Family::WEIGHT_NAMES.iter().enumerate() {
            let h = config.transformer().hidden_size();
            let m = config.transformer().mlp_hidden_size();
            let shape = match index {
                2 | 3 => vec![config.transformer().head_dim()],
                6 | 7 => vec![m, h],
                8 => vec![h, m],
                _ => vec![h, h],
            };
            let tensor = source.tensor(name)?;
            if tensor.dtype() != safetensors::Dtype::BF16 || tensor.shape() != shape {
                bail!("block-0 weight {name} has wrong dtype/shape");
            }
            weights.push(WeightDigest {
                tensor: (*name).into(),
                dtype: "BF16".into(),
                shape,
                bytes: u64::try_from(tensor.bytes().len())?,
                sha256: artifacts::sha256(tensor.bytes()),
            });
        }
        Ok(Self {
            canonical_config: config.clone(),
            config_file_sha256: artifacts::sha256(config_bytes),
            inputs: input_evidence,
            weights,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub backend: String,
    pub identity: Identity,
    pub outputs: Vec<TensorDigest>,
}

/// Holds the already checked reference independently from candidate outputs.
pub struct Reference {
    directory: PathBuf,
    outputs: Vec<Vec<u8>>,
}
impl Reference {
    pub fn load(directory: &Path, identity: &Identity, specs: &[OutputSpec]) -> Result<Self> {
        let directory = directory
            .canonicalize()
            .context("reference directory must exist")?;
        if !directory.is_dir() {
            bail!("reference path is not a directory");
        }
        let manifest: Manifest =
            serde_json::from_slice(&read(&directory, "manifest.json", MANIFEST_LIMIT)?)?;
        if manifest.schema != MANIFEST_SCHEMA
            || manifest.backend != "reference"
            || &manifest.identity != identity
        {
            bail!("reference manifest schema/backend/config/input/weight identity mismatch");
        }
        let result: Value =
            serde_json::from_slice(&read(&directory, "result.json", MANIFEST_LIMIT)?)?;
        if result["validation"]["passed"] != true
            || result["cleanup"]["managed_host_committed_bytes"] != 0
            || result["cleanup"]["managed_host_allocated_bytes"] != 0
        {
            bail!(
                "reference run lacks a successful finite validation and zero-Host cleanup receipt"
            );
        }
        for input in &identity.inputs {
            let bytes = read(&directory, &input.file, usize::try_from(input.bytes)?)?;
            if bytes.len() as u64 != input.bytes || artifacts::sha256(&bytes) != input.sha256 {
                bail!("reference input file differs from its identity digest");
            }
        }
        if manifest.outputs.len() != specs.len() {
            bail!("reference must contain every output boundary");
        }
        let mut outputs = Vec::new();
        for (record, spec) in manifest.outputs.iter().zip(specs) {
            if record.value_id != spec.value_id
                || record.file != spec.file
                || record.shape != spec.shape
                || record.bytes != spec.byte_len as u64
                || record.dtype != "BF16"
            {
                bail!("reference output descriptor differs from declared complete boundary");
            }
            let bytes = read(&directory, &record.file, spec.byte_len)?;
            if bytes.len() != spec.byte_len || artifacts::sha256(&bytes) != record.sha256 {
                bail!("reference output file differs from recorded bytes/digest");
            }
            if bytes.chunks_exact(2).any(|v| !decode(v).is_finite()) {
                bail!("reference contains a non-finite BF16 output");
            }
            outputs.push(bytes);
        }
        Ok(Self { directory, outputs })
    }
    pub fn directory(&self) -> &Path {
        &self.directory
    }
    pub fn output(&self, index: usize) -> &[u8] {
        &self.outputs[index]
    }
}

fn read(directory: &Path, name: &str, limit: usize) -> Result<Vec<u8>> {
    // Names have already been matched to the current fixed manifest descriptors.
    // Check canonical containment as well so cross-directory symlinks fail.
    let path = directory.join(name).canonicalize()?;
    if !path.starts_with(directory) {
        bail!("reference file escapes its directory");
    }
    artifacts::read_bounded(&path, limit)
}

fn decode(bytes: &[u8]) -> f64 {
    f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16) as f64
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    value_id: String,
    elements: usize,
    compared_elements: usize,
    nonfinite_count: usize,
    failure_count: usize,
    maximum_absolute_error: Option<f64>,
    mean_absolute_error: Option<f64>,
}
pub fn compare(spec: &OutputSpec, actual: &[u8], reference: Option<&[u8]>) -> Result<Comparison> {
    if actual.len() != spec.byte_len
        || spec.byte_len % 2 != 0
        || actual.is_empty()
        || reference.is_some_and(|expected| expected.len() != actual.len())
    {
        bail!("comparison requires the complete, exact BF16 boundary");
    }
    let mut result = Comparison {
        value_id: spec.value_id.clone(),
        elements: actual.len() / 2,
        compared_elements: 0,
        nonfinite_count: 0,
        failure_count: 0,
        maximum_absolute_error: reference.map(|_| 0.0),
        mean_absolute_error: None,
    };
    let mut total_error = 0.0;
    for (index, actual) in actual.chunks_exact(2).enumerate() {
        let actual = decode(actual);
        let expected = reference.map(|bytes| decode(&bytes[index * 2..index * 2 + 2]));
        if !actual.is_finite() || expected.is_some_and(|value| !value.is_finite()) {
            result.nonfinite_count += 1;
            result.failure_count += 1;
            continue;
        }
        if let Some(expected) = expected {
            let error = (actual - expected).abs();
            let allowed = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * expected.abs();
            result.compared_elements += 1;
            total_error += error;
            result.maximum_absolute_error =
                Some(result.maximum_absolute_error.unwrap_or(0.0).max(error));
            if error > allowed {
                result.failure_count += 1;
            }
        }
    }
    if result.compared_elements != 0 {
        result.mean_absolute_error = Some(total_error / result.compared_elements as f64);
    }
    Ok(result)
}
pub fn report(ports: Vec<Comparison>, with_reference: bool) -> Value {
    let failures = ports.iter().map(|port| port.failure_count).sum::<usize>();
    let nonfinite = ports.iter().map(|port| port.nonfinite_count).sum::<usize>();
    json!({"mode":if with_reference {"reference_differential"} else {"execution_smoke_only"},
        "passed":failures==0,"numerical_verification":if with_reference {"Reference differential only"} else {"not performed"},
        "absolute_tolerance":ABSOLUTE_TOLERANCE,"relative_tolerance":RELATIVE_TOLERANCE,
        "failure_count":failures,"nonfinite_count":nonfinite,"ports":ports})
}
