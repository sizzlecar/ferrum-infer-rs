//! Per-interval PDD output parameters, derived from the original widened
//! channel-major heads. The source owns two admitted F32 buffers; it has no
//! global fusion cursor, cache, device allocator, or request state.
use std::ops::Range;

use anyhow::{ensure, Context};
use ferrum_runtime::*;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{FastH3PddPlan, FastH3TransformerPackage, FastH3WeightStorage};

mod family;
mod program;
pub use family::FastH3PddProjectionFamily;

const ARITHMETIC_PROFILE: &str = "fast_h3.pdd.channel_major.host_f32_separate_mul_add.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3OutputModality {
    Video,
    Audio,
}

#[derive(Clone, Copy, Debug)]
pub struct FastH3PddProjectionLimits {
    pub maximum_rows: u64,
    pub maximum_metadata_bytes: usize,
    pub maximum_source_bytes: u64,
    pub maximum_fused_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PddParameter {
    pub(super) original: WeightComponentSpec,
    pub(super) source_file: String,
    pub(super) source_sha256: String,
    pub(super) fused: WeightComponentSpec,
}

/// Complete derivation identity. F64 integration weights normalize before
/// their F32 cast; raw source dtype, content digest, interval and accumulation
/// profile remain part of the prepared family. Different intervals cannot
/// silently share the same Stored source declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3PddProjectionConfig {
    pub(super) rows: u64,
    pub(super) hidden: u64,
    pub(super) output: u64,
    pub(super) grid_size: usize,
    pub(super) evaluation: usize,
    pub(super) modality: FastH3OutputModality,
    pub(super) heads: Range<usize>,
    pub(super) integration_weight_bits: Vec<u64>,
    pub(super) coefficient_bits: Vec<u32>,
    pub(super) parameters: [PddParameter; 2],
    pub(super) arithmetic_profile: String,
}

#[derive(Serialize)]
struct Derivation<'a> {
    grid_size: usize,
    evaluation: usize,
    modality: FastH3OutputModality,
    heads: &'a Range<usize>,
    integration_weight_bits: &'a [u64],
    coefficient_bits: &'a [u32],
}
impl FastH3PddProjectionConfig {
    fn derivation(&self) -> Derivation<'_> {
        Derivation {
            grid_size: self.grid_size,
            evaluation: self.evaluation,
            modality: self.modality,
            heads: &self.heads,
            integration_weight_bits: &self.integration_weight_bits,
            coefficient_bits: &self.coefficient_bits,
        }
    }
    pub fn evaluation(&self) -> usize {
        self.evaluation
    }
    pub fn modality(&self) -> FastH3OutputModality {
        self.modality
    }
    pub fn heads(&self) -> Range<usize> {
        self.heads.clone()
    }
    pub fn rows(&self) -> u64 {
        self.rows
    }
    pub fn hidden_size(&self) -> u64 {
        self.hidden
    }
    pub fn output_size(&self) -> u64 {
        self.output
    }
    pub fn coefficient_bits(&self) -> &[u32] {
        &self.coefficient_bits
    }
    pub(in crate::fast_h3) fn validate_schedule(
        &self,
        plan: &FastH3PddPlan,
    ) -> Result<(), VNextError> {
        let step = plan
            .steps()
            .get(self.evaluation)
            .ok_or_else(|| invalid("evaluation is outside the sampling plan"))?;
        let (modality, width) = match self.modality {
            FastH3OutputModality::Video => (&step.video, plan.video_head_width()),
            FastH3OutputModality::Audio => (&step.audio, plan.audio_head_width()),
        };
        if self.grid_size != plan.fine_grid().len() - 1
            || self.output != width as u64
            || self.heads != step.heads
            || self.integration_weight_bits
                != modality
                    .integration_weights()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            || self.coefficient_bits
                != modality
                    .normalized_head_weights()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
        {
            return Err(invalid(
                "projection interval or coefficients differ from the sampling plan",
            ));
        }
        Ok(())
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(program::build(self)?.1)
    }
    pub(super) fn validate(&self, limits: FastH3PddProjectionLimits) -> Result<(), VNextError> {
        validate_limits(limits)?;
        if self.rows == 0
            || self.rows > limits.maximum_rows
            || [self.rows, self.hidden, self.output]
                .iter()
                .any(|&v| v == 0 || v > i32::MAX as u64)
            || self.grid_size < 2
            || self.grid_size > i32::MAX as usize
            || self.evaluation >= self.grid_size
            || self.heads.start >= self.heads.end
            || self.heads.end > self.grid_size
            || self.coefficient_bits.len() != self.heads.len()
            || self.integration_weight_bits.len() != self.heads.len()
            || self.arithmetic_profile != ARITHMETIC_PROFILE
        {
            return Err(invalid(
                "invalid projection geometry, interval, arithmetic or capacity",
            ));
        }
        let mut total = 0.0_f64;
        for &bits in &self.integration_weight_bits {
            let weight = f64::from_bits(bits);
            if !weight.is_finite() || weight >= 0.0 {
                return Err(invalid("integration weights must be finite negative"));
            }
            total += weight;
        }
        if !total.is_finite() || total >= 0.0 {
            return Err(invalid("invalid integration total"));
        }
        for (&bits, &coefficient) in self
            .integration_weight_bits
            .iter()
            .zip(&self.coefficient_bits)
        {
            let expected = (f64::from_bits(bits) / total) as f32;
            if !expected.is_finite() || expected <= 0.0 || expected.to_bits() != coefficient {
                return Err(invalid(
                    "head coefficient differs from F64 normalization then F32 cast",
                ));
            }
        }
        let widened = self
            .output
            .checked_mul(self.grid_size as u64)
            .filter(|&n| n <= i32::MAX as u64)
            .ok_or_else(|| invalid("widened head extent overflow"))?;
        let mut source_bytes = 0u64;
        let mut fused_bytes = 0u64;
        for (index, p) in self.parameters.iter().enumerate() {
            let original_shape = if index == 0 {
                vec![widened, self.hidden]
            } else {
                vec![widened]
            };
            let fused_shape = if index == 0 {
                vec![self.output, self.hidden]
            } else {
                vec![self.output]
            };
            if p.original.dimensions != original_shape
                || p.fused.dimensions != fused_shape
                || p.original.role != WeightComponentRole::Values
                || !p.original.required
                || p.original.external_names.len() != 1
                || !matches!(
                    p.original.encoding,
                    WeightEncoding::Dense {
                        element_type: ElementType::F32 | ElementType::F16 | ElementType::Bf16
                    }
                )
                || !super::nvfp4_scalars::valid_source_file(&p.source_file)
                || p.source_sha256.len() != 64
                || !p
                    .source_sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                || p.fused != derived_component(p, &self.derivation())?
            {
                return Err(invalid(
                    "projection parameter identity, shape or provenance differs",
                ));
            }
            source_bytes = source_bytes
                .checked_add(p.original.physical_bytes()?)
                .ok_or_else(|| invalid("source extent overflow"))?;
            fused_bytes = fused_bytes
                .checked_add(p.fused.physical_bytes()?)
                .ok_or_else(|| invalid("fused extent overflow"))?;
        }
        if self.parameters[0].original.id == self.parameters[1].original.id
            || self.parameters[0].original.external_names
                == self.parameters[1].original.external_names
            || source_bytes > limits.maximum_source_bytes
            || fused_bytes > limits.maximum_fused_bytes
        {
            return Err(invalid("source alias or projection byte capacity exceeded"));
        }
        self.rows
            .checked_mul(self.hidden.max(self.output))
            .and_then(|n| n.checked_mul(4))
            .filter(|&n| n <= isize::MAX as u64)
            .ok_or_else(|| invalid("activation extent overflow"))?;
        if serde_json::to_vec(self)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > limits.maximum_metadata_bytes
        {
            return Err(invalid("projection metadata capacity exceeded"));
        }
        Ok(())
    }
}

/// Fused weights survive the original archive and retain only the admitted
/// WeightBacking owners. Raw mapped source bytes are read without an expanded
/// F32 copy. Each prepare call is immutable and binds exactly one modality and
/// evaluation; a batch adapter must route each member to its own interval.
pub struct FastH3PddProjectionSource {
    parameters: [PddParameter; 2],
    buffers: [ManagedHostBytes; 2],
}
impl FastH3PddProjectionSource {
    pub fn prepare(
        package: &FastH3TransformerPackage,
        plan: &FastH3PddPlan,
        evaluation: usize,
        modality: FastH3OutputModality,
        rows: u64,
        host: &HostMemoryAccount,
        limits: FastH3PddProjectionLimits,
    ) -> anyhow::Result<(FastH3PddProjectionConfig, Self)> {
        validate_limits(limits)?;
        ensure!(
            rows > 0 && rows <= limits.maximum_rows && rows <= i32::MAX as u64,
            "PDD projection rows exceed capacity"
        );
        let step = plan
            .steps()
            .get(evaluation)
            .context("PDD evaluation out of range")?;
        let c = package.config();
        let grid_size = plan.fine_grid().len() - 1;
        ensure!(
            c.video_head_width() == plan.video_head_width()
                && c.audio_head_width() == plan.audio_head_width()
                && c.video_projection_rows() / c.video_head_width() == grid_size,
            "PDD schedule and package output dimensions differ"
        );
        let (prefix, output, weights) = match modality {
            FastH3OutputModality::Video => ("proj_out", c.video_head_width(), &step.video),
            FastH3OutputModality::Audio => ("audio_proj_out", c.audio_head_width(), &step.audio),
        };
        let coefficient_bits: Vec<_> = weights
            .normalized_head_weights()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let integration_weight_bits: Vec<_> = weights
            .integration_weights()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let mut originals = Vec::with_capacity(2);
        let mut source_bytes = 0u64;
        for suffix in ["weight", "bias"] {
            let weight = package
                .weight(&format!("{prefix}.{suffix}"))
                .context("missing PDD output parameter")?;
            let FastH3WeightStorage::Dense(original) = &weight.storage else {
                anyhow::bail!("PDD output parameter must remain dense")
            };
            ensure!(
                weight.compute_element_type == ElementType::F32,
                "PDD output precision must be F32"
            );
            source_bytes = source_bytes
                .checked_add(original.physical_bytes()?)
                .context("PDD source extent overflow")?;
            originals.push(original.clone());
        }
        let fused_bytes = output
            .checked_mul(c.hidden_size())
            .and_then(|n| n.checked_add(output))
            .and_then(|n| n.checked_mul(4))
            .context("PDD fused extent overflow")?;
        ensure!(
            source_bytes <= limits.maximum_source_bytes
                && (fused_bytes as u64) <= limits.maximum_fused_bytes,
            "PDD source or fused byte capacity exceeded"
        );
        // Source limits precede payload hashing or parameter arithmetic.
        let mut payloads = Vec::with_capacity(2);
        let mut parameters = Vec::with_capacity(2);
        for (index, original) in originals.into_iter().enumerate() {
            let payload = package.component(&original)?;
            ensure!(
                payload.source_files().len() == 1,
                "PDD parameter must have one original source"
            );
            let mut p = PddParameter {
                fused: WeightComponentSpec {
                    id: WeightId::new("derived.fast_h3.pdd.pending")?,
                    role: WeightComponentRole::Values,
                    external_names: original.external_names.clone(),
                    dimensions: if index == 0 {
                        vec![output as u64, c.hidden_size() as u64]
                    } else {
                        vec![output as u64]
                    },
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::F32,
                    },
                    required: true,
                },
                original,
                source_file: payload.source_file().into(),
                source_sha256: format!("{:x}", Sha256::digest(payload.bytes())),
            };
            p.fused = derived_component(
                &p,
                &Derivation {
                    grid_size,
                    evaluation,
                    modality,
                    heads: &step.heads,
                    integration_weight_bits: &integration_weight_bits,
                    coefficient_bits: &coefficient_bits,
                },
            )?;
            parameters.push(p);
            payloads.push(payload);
        }
        let config = FastH3PddProjectionConfig {
            rows,
            hidden: c.hidden_size() as u64,
            output: output as u64,
            grid_size,
            evaluation,
            modality,
            heads: step.heads.clone(),
            integration_weight_bits,
            coefficient_bits,
            parameters: parameters
                .try_into()
                .map_err(|_| invalid("expected two parameters"))?,
            arithmetic_profile: ARITHMETIC_PROFILE.into(),
        };
        config.validate(limits)?;
        let mut buffers = [
            host.allocate_zeroed(
                HostMemoryClass::WeightBacking,
                config.parameters[0].fused.physical_bytes()?.try_into()?,
            )?,
            host.allocate_zeroed(
                HostMemoryClass::WeightBacking,
                config.parameters[1].fused.physical_bytes()?.try_into()?,
            )?,
        ];
        for (payload, buffer) in payloads.iter().zip(&mut buffers) {
            fuse_parameter(
                payload,
                &config.heads,
                &config.coefficient_bits,
                buffer.as_mut(),
            )?;
        }
        let source = Self {
            parameters: config.parameters.clone(),
            buffers: buffers.map(ManagedHostBuffer::freeze),
        };
        Ok((config, source))
    }
}
impl WeightComponentSource for FastH3PddProjectionSource {
    fn component<'s>(
        &'s self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'s>, VNextError> {
        let index = self
            .parameters
            .iter()
            .position(|p| &p.fused == component)
            .ok_or_else(|| invalid("undeclared fused PDD parameter"))?;
        WeightComponentPayload::from_managed_host(
            component,
            component.external_names.clone(),
            vec![self.parameters[index].source_file.clone()],
            component.dimensions.clone(),
            ElementType::F32,
            self.buffers[index].clone(),
        )
    }
}

fn derived_component(
    p: &PddParameter,
    derivation: &Derivation<'_>,
) -> Result<WeightComponentSpec, VNextError> {
    let identity = serde_json::to_vec(&(
        ARITHMETIC_PROFILE,
        &p.original,
        &p.source_file,
        &p.source_sha256,
        derivation,
        &p.fused.dimensions,
    ))
    .map_err(|e| invalid(e.to_string()))?;
    Ok(WeightComponentSpec {
        id: WeightId::new(format!(
            "derived.fast_h3.pdd.{:x}",
            Sha256::digest(identity)
        ))?,
        role: WeightComponentRole::Values,
        external_names: p.original.external_names.clone(),
        dimensions: p.fused.dimensions.clone(),
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F32,
        },
        required: true,
    })
}
fn fuse_parameter(
    payload: &WeightComponentPayload<'_>,
    heads: &Range<usize>,
    coefficients: &[u32],
    destination: &mut [u8],
) -> anyhow::Result<()> {
    let dtype = payload.element_type();
    let scalar_bytes = match dtype {
        ElementType::F32 => 4,
        ElementType::Bf16 | ElementType::F16 => 2,
        _ => anyhow::bail!("PDD source must be floating"),
    };
    let head_elements = destination.len() / 4;
    ensure!(
        !destination.is_empty()
            && destination.len() % 4 == 0
            && heads.start < heads.end
            && heads.len() == coefficients.len()
            && heads
                .end
                .checked_mul(head_elements)
                .and_then(|n| n.checked_mul(scalar_bytes))
                .is_some_and(|n| n <= payload.bytes().len()),
        "invalid PDD fusion byte extent"
    );
    for (element, dst) in destination.chunks_exact_mut(4).enumerate() {
        let mut sum = 0.0_f32;
        for (head, &coefficient) in heads.clone().zip(coefficients) {
            let offset = (head * head_elements + element) * scalar_bytes;
            let bytes = &payload.bytes()[offset..offset + scalar_bytes];
            let value = match dtype {
                ElementType::F32 => f32::from_le_bytes(bytes.try_into()?),
                ElementType::Bf16 => {
                    bf16::from_bits(u16::from_le_bytes(bytes.try_into()?)).to_f32()
                }
                ElementType::F16 => f16::from_bits(u16::from_le_bytes(bytes.try_into()?)).to_f32(),
                _ => unreachable!(),
            };
            // These separate RN32 operations are an explicit host profile,
            // not a bit-parity assertion about Torch's CUDA einsum reduction.
            let product = value * f32::from_bits(coefficient);
            sum += product;
            ensure!(
                value.is_finite() && product.is_finite() && sum.is_finite(),
                "PDD parameter fusion produced a non-finite value"
            );
        }
        dst.copy_from_slice(&sum.to_le_bytes());
    }
    Ok(())
}
fn validate_limits(limits: FastH3PddProjectionLimits) -> Result<(), VNextError> {
    if limits.maximum_rows == 0
        || limits.maximum_metadata_bytes == 0
        || limits.maximum_source_bytes == 0
        || limits.maximum_fused_bytes == 0
    {
        return Err(invalid("PDD projection capacities must be positive"));
    }
    Ok(())
}
pub(super) fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 PDD projection: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
