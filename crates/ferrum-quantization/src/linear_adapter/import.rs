use super::{config::PeftConfig, invalid, LinearAdapterSource, Result};
use crate::{SafetensorsArchive, SafetensorsTensor};
use ferrum_runtime::{ElementType, RetainedHostMemoryRegion};
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

pub struct LowRankMatrix {
    pub(super) a: RetainedHostMemoryRegion,
    pub(super) b: RetainedHostMemoryRegion,
    pub(super) a_type: ElementType,
    pub(super) b_type: ElementType,
    pub(super) rank: u64,
    pub(super) input: u64,
    pub(super) output: u64,
    pub(super) multiplier: f32,
}

/// Borrowed encoded storage for explicit interoperability/export. Access does
/// not scan, convert or copy a payload; consumers must admit retained owners
/// through the prepared adapter before reading/exporting their values.
pub struct LinearAdapterMatrixView<'a> {
    pub element_type: ElementType,
    pub dimensions: [u64; 2],
    pub bytes: &'a [u8],
}
impl LowRankMatrix {
    pub fn a(&self) -> LinearAdapterMatrixView<'_> {
        LinearAdapterMatrixView {
            element_type: self.a_type,
            dimensions: [self.rank, self.input],
            bytes: self.a.bytes(),
        }
    }
    pub fn b(&self) -> LinearAdapterMatrixView<'_> {
        LinearAdapterMatrixView {
            element_type: self.b_type,
            dimensions: [self.output, self.rank],
            bytes: self.b.bytes(),
        }
    }
    pub const fn rank(&self) -> u64 {
        self.rank
    }
    pub const fn dimensions(&self) -> [u64; 2] {
        [self.output, self.input]
    }
    pub const fn multiplier(&self) -> f32 {
        self.multiplier
    }
}

pub struct ImportedLinearAdapter {
    pub(super) base_model: String,
    pub(super) matrices: BTreeMap<String, LowRankMatrix>,
    architecture_hint: Option<String>,
}
impl ImportedLinearAdapter {
    /// No payload conversion, payload scan, device work or implicit download.
    pub fn open(source: &LinearAdapterSource) -> Result<Self> {
        let (archive, base_model, scaling, peft) = match source {
            LinearAdapterSource::Peft {
                directory,
                base_model,
            } => {
                let config =
                    PeftConfig::parse(&read_config(&directory.join("adapter_config.json"))?)?;
                if let (Some(declared), Some(explicit)) = (&config.base_model, base_model) {
                    if declared != explicit {
                        return Err(invalid("explicit base model contradicts PEFT declaration"));
                    }
                }
                let base = config
                    .base_model
                    .clone()
                    .or_else(|| base_model.clone())
                    .ok_or_else(|| {
                        invalid("PEFT config has no base model; provide an explicit binding")
                    })?;
                (
                    SafetensorsArchive::open_single_file(directory, "adapter_model.safetensors")?,
                    base,
                    config.scaling,
                    Some(config),
                )
            }
            LinearAdapterSource::Diffusers {
                file,
                base_model,
                scaling,
            } => {
                let name = file
                    .file_name()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| invalid("adapter file needs a UTF-8 safetensors basename"))?;
                (
                    SafetensorsArchive::open_single_file(
                        file.parent().unwrap_or(Path::new(".")),
                        name,
                    )?,
                    base_model.clone(),
                    *scaling,
                    None,
                )
            }
        };
        if base_model.trim().is_empty() || base_model.chars().any(char::is_control) {
            return Err(invalid(
                "base model binding must be nonempty and contain no control characters",
            ));
        }
        let mut pairs: BTreeMap<String, [Option<String>; 2]> = BTreeMap::new();
        let mut naming = None;
        for key in archive.tensor_names() {
            let (module, side, named) = parse_key(key)?;
            if naming
                .replace(named)
                .is_some_and(|previous| previous != named)
            {
                return Err(invalid("mixed named and unnamed adapters are unsupported"));
            }
            let pair = pairs.entry(module.to_owned()).or_default();
            if pair[side].replace(key.to_owned()).is_some() {
                return Err(invalid(format!("duplicate adapter matrix for {module:?}")));
            }
        }
        let mut matrices = BTreeMap::new();
        for (module, pair) in pairs {
            let [Some(a), Some(b)] = pair else {
                return Err(invalid(format!("target {module:?} is missing LoRA A or B")));
            };
            let a = archive.tensor(&a)?;
            let b = archive.tensor(&b)?;
            let ([rank, input], [output, b_rank]) = (a.shape(), b.shape()) else {
                return Err(invalid(format!(
                    "target {module:?} requires two rank-2 matrices"
                )));
            };
            if *rank == 0 || *input == 0 || *output == 0 || rank != b_rank {
                return Err(invalid(format!(
                    "target {module:?} has inconsistent A/B dimensions"
                )));
            }
            if let Some(config) = &peft {
                if config.rank != *rank
                    || !config.targets.iter().any(|target| {
                        module == *target
                            || module
                                .strip_suffix(target)
                                .is_some_and(|s| s.ends_with('.'))
                    })
                {
                    return Err(invalid(format!(
                        "target {module:?} differs from PEFT rank/target declaration"
                    )));
                }
            }
            matrices.insert(
                module,
                LowRankMatrix {
                    a: a.retained_host_memory().clone(),
                    b: b.retained_host_memory().clone(),
                    a_type: float_type(&a)?,
                    b_type: float_type(&b)?,
                    rank: *rank,
                    input: *input,
                    output: *output,
                    multiplier: scaling.multiplier(*rank)?,
                },
            );
        }
        if matrices.is_empty() {
            return Err(invalid(
                "adapter contains no complete ordinary LoRA targets",
            ));
        }
        let architecture_hint = validate_metadata(&archive, &base_model, &matrices)?;
        Ok(Self {
            base_model,
            matrices,
            architecture_hint,
        })
    }
    pub fn base_model(&self) -> &str {
        &self.base_model
    }
    pub fn matrices(&self) -> &BTreeMap<String, LowRankMatrix> {
        &self.matrices
    }
    pub fn architecture_hint(&self) -> Option<&str> {
        self.architecture_hint.as_deref()
    }
}

fn parse_key(key: &str) -> Result<(&str, usize, bool)> {
    let (module, side, named) = [
        (".lora_A.weight", 0, false),
        (".lora_B.weight", 1, false),
        (".lora_A.default.weight", 0, true),
        (".lora_B.default.weight", 1, true),
    ]
    .into_iter()
    .find_map(|(suffix, side, named)| key.strip_suffix(suffix).map(|module| (module, side, named)))
    .ok_or_else(|| {
        invalid(format!(
            "unsupported adapter tensor {key:?}; no keys may be ignored"
        ))
    })?;
    let module = module.strip_prefix("base_model.model.").unwrap_or(module);
    let module = module
        .strip_prefix("transformer.")
        .or_else(|| module.strip_prefix("diffusion_model."))
        .unwrap_or(module);
    if module.is_empty() || module.split('.').any(str::is_empty) {
        return Err(invalid(format!("invalid adapter module {key:?}")));
    }
    Ok((module, side, named))
}
fn float_type(tensor: &SafetensorsTensor<'_>) -> Result<ElementType> {
    match tensor.element_type() {
        Some(kind @ (ElementType::F16 | ElementType::Bf16 | ElementType::F32)) => Ok(kind),
        _ => Err(invalid(
            "ordinary LoRA matrices require F16, BF16 or F32 storage",
        )),
    }
}
fn read_config(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(invalid)?;
    if file.metadata().map_err(invalid)?.len() > 1024 * 1024 {
        return Err(invalid("adapter config exceeds 1 MiB metadata limit"));
    }
    let mut bytes = Vec::new();
    file.take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(invalid)?;
    if bytes.is_empty() || bytes.len() > 1024 * 1024 {
        return Err(invalid("empty or oversized adapter config"));
    }
    Ok(bytes)
}
fn validate_metadata(
    archive: &SafetensorsArchive,
    base_model: &str,
    matrices: &BTreeMap<String, LowRankMatrix>,
) -> Result<Option<String>> {
    let mut architecture = None;
    for metadata in archive.header_metadata()?.values() {
        if let Some(hint) = metadata.get("ss_base_model_version") {
            if architecture
                .replace(hint.clone())
                .is_some_and(|previous| previous != *hint)
            {
                return Err(invalid("conflicting adapter architecture declarations"));
            }
        }
        for key in ["ss_network_args", "lora_adapter_metadata"] {
            if let Some(raw) = metadata.get(key) {
                let value: serde_json::Value = serde_json::from_str(raw).map_err(invalid)?;
                if !value.is_object() {
                    return Err(invalid("adapter metadata configuration must be an object"));
                }
                if value.get("peft_type").is_some_and(|v| *v != "LORA") {
                    return Err(invalid("unsupported adapter metadata PEFT type"));
                }
                for flag in [
                    "use_dora",
                    "use_rslora",
                    "use_qalora",
                    "lora_bias",
                    "fan_in_fan_out",
                ] {
                    if value.get(flag).is_some_and(|v| !v.is_null() && *v != false) {
                        return Err(invalid(format!(
                            "unsupported adapter metadata variant {flag}"
                        )));
                    }
                }
                if let Some(declared) = value
                    .get("base_model_name_or_path")
                    .filter(|v| !v.is_null())
                {
                    if declared.as_str() != Some(base_model) {
                        return Err(invalid("base model contradicts adapter metadata"));
                    }
                }
                let rank = value
                    .get("r")
                    .filter(|v| !v.is_null())
                    .map(|v| {
                        v.as_u64()
                            .filter(|r| *r > 0)
                            .ok_or_else(|| invalid("invalid metadata rank"))
                    })
                    .transpose()?;
                let alpha = value
                    .get("lora_alpha")
                    .filter(|v| !v.is_null())
                    .map(|v| {
                        v.as_f64()
                            .map(|n| n as f32)
                            .ok_or_else(|| invalid("invalid metadata alpha"))
                    })
                    .transpose()?;
                check_declared_scaling(matrices, rank, alpha)?;
                for field in ["rank_pattern", "alpha_pattern"] {
                    if value.get(field).is_some_and(|v| {
                        !v.is_null() && !v.as_object().is_some_and(|m| m.is_empty())
                    }) {
                        return Err(invalid(format!("unsupported adapter metadata {field}")));
                    }
                }
            }
        }
        let rank = metadata
            .get("ss_network_dim")
            .map(|rank| rank.parse::<u64>().map_err(invalid))
            .transpose()?;
        let alpha = metadata
            .get("ss_network_alpha")
            .map(|alpha| alpha.parse::<f32>().map_err(invalid))
            .transpose()?;
        check_declared_scaling(matrices, rank, alpha)?;
    }
    Ok(architecture)
}

fn check_declared_scaling(
    matrices: &BTreeMap<String, LowRankMatrix>,
    rank: Option<u64>,
    alpha: Option<f32>,
) -> Result<()> {
    if rank == Some(0) || alpha.is_some_and(|alpha| !alpha.is_finite() || alpha <= 0.0) {
        return Err(invalid(
            "metadata rank/alpha is outside ordinary LoRA's domain",
        ));
    }
    for matrix in matrices.values() {
        if rank.is_some_and(|rank| rank != matrix.rank)
            || alpha.is_some_and(|alpha| alpha / matrix.rank as f32 != matrix.multiplier)
        {
            return Err(invalid(
                "explicit rank/scaling contradicts adapter metadata",
            ));
        }
    }
    Ok(())
}
