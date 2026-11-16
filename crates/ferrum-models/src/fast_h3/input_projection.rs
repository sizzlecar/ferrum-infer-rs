//! Released FastH3 input projections, independently of text refinement and
//! modality packing. Video/audio parameters compute in F32; context parameters
//! compute in BF16. Projected text is explicitly not a refined-text result.
use super::{FastH3PackedLayout, FastH3SegmentKind, FastH3TransformerPackage, FastH3WeightStorage};
use ferrum_quantization::ManagedDenseArchiveSource;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod family;
mod packing;
mod program;
pub use family::{FastH3InputPackingFamily, FastH3InputProjectionFamily};
pub use packing::FastH3InputPackingConfig;

#[derive(Clone, Copy, Debug)]
pub struct FastH3InputLimits {
    pub maximum_rows: u64,
    pub maximum_segments: usize,
    pub maximum_metadata_bytes: usize,
    pub maximum_stored_bytes: u64,
}
impl FastH3InputLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.maximum_rows == 0
            || self.maximum_segments == 0
            || self.maximum_metadata_bytes == 0
            || self.maximum_stored_bytes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(())
    }
}
/// Text projection is a cold request dependency; media projection runs on each
/// denoising evaluation. A phase owns only its selected sources and values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3InputProjectionPhase {
    Media,
    Text,
}
impl FastH3InputProjectionPhase {
    fn streams(self) -> &'static [usize] {
        match self {
            Self::Media => &[0, 1],
            Self::Text => &[2],
        }
    }
    pub const fn profile_id(self) -> &'static str {
        match self {
            Self::Media => FastH3InputProjectionFamily::MEDIA_PROFILE_ID,
            Self::Text => FastH3InputProjectionFamily::TEXT_PROFILE_ID,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3InputProjectionConfig {
    phase: FastH3InputProjectionPhase,
    rows: Vec<u64>,
    widths: Vec<u64>,
    hidden: u64,
    components: Vec<WeightComponentSpec>,
}
impl FastH3InputProjectionConfig {
    pub fn media_from_package(
        p: &FastH3TransformerPackage,
        layout: &FastH3PackedLayout,
        limits: FastH3InputLimits,
    ) -> Result<Self, VNextError> {
        Self::media_batch_from_package(p, layout, 1, limits)
    }
    /// Per-modality inputs are request-major, with the same layout per member.
    /// Projection is row-local and all members retain the original parameters.
    pub fn media_batch_from_package(
        p: &FastH3TransformerPackage,
        layout: &FastH3PackedLayout,
        batches: u64,
        limits: FastH3InputLimits,
    ) -> Result<Self, VNextError> {
        if batches == 0 {
            return Err(invalid("media batch must be nonempty"));
        }
        Self::with_rows(
            p,
            FastH3InputProjectionPhase::Media,
            counts(layout)?[..2]
                .iter()
                .map(|&rows| {
                    rows.checked_mul(batches)
                        .ok_or_else(|| invalid("media batch rows overflow"))
                })
                .collect::<Result<_, _>>()?,
            limits,
        )
    }
    pub fn text_from_package(
        p: &FastH3TransformerPackage,
        text_rows: u64,
        limits: FastH3InputLimits,
    ) -> Result<Self, VNextError> {
        Self::with_rows(p, FastH3InputProjectionPhase::Text, vec![text_rows], limits)
    }
    fn with_rows(
        p: &FastH3TransformerPackage,
        phase: FastH3InputProjectionPhase,
        rows: Vec<u64>,
        limits: FastH3InputLimits,
    ) -> Result<Self, VNextError> {
        if rows.len() != phase.streams().len() {
            return Err(invalid("phase row inventory differs"));
        }
        let config = p.config();
        let all_widths = [
            config.video_head_width() as u64,
            config.audio_channels() as u64,
            config.text_dim() as u64,
        ];
        let modules = ["proj_in", "audio_proj_in", "context_embedder"];
        let mut components = Vec::with_capacity(2 * rows.len());
        for &i in phase.streams() {
            for suffix in ["weight", "bias"] {
                let name = format!("{}.{suffix}", modules[i]);
                let weight = p
                    .weight(&name)
                    .ok_or_else(|| invalid(format!("missing {name}")))?;
                let FastH3WeightStorage::Dense(source) = &weight.storage else {
                    return Err(invalid("input projection requires dense source"));
                };
                let dtype = if i < 2 {
                    ElementType::F32
                } else {
                    ElementType::Bf16
                };
                if weight.compute_element_type != dtype {
                    return Err(invalid("input projection compute dtype differs"));
                }
                let mut c = source.clone();
                c.encoding = WeightEncoding::Dense {
                    element_type: dtype,
                };
                components.push(c);
            }
        }
        let c = Self {
            phase,
            rows,
            widths: phase.streams().iter().map(|&i| all_widths[i]).collect(),
            hidden: config.hidden_size() as u64,
            components,
        };
        c.validate(limits)?;
        Ok(c)
    }
    pub fn phase(&self) -> FastH3InputProjectionPhase {
        self.phase
    }
    /// Media is [video,audio]; Text has exactly one row count/width.
    pub fn rows(&self) -> &[u64] {
        &self.rows
    }
    pub fn input_widths(&self) -> &[u64] {
        &self.widths
    }
    pub fn hidden_size(&self) -> u64 {
        self.hidden
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(program::build(self)?.1)
    }
    fn check_inventory(&self) -> Result<(), VNextError> {
        let n = self.phase.streams().len();
        if self.rows.len() != n || self.widths.len() != n || self.components.len() != 2 * n {
            return Err(invalid(
                "phase must declare only its complete selected input/source inventory",
            ));
        }
        Ok(())
    }
    fn validate(&self, limits: FastH3InputLimits) -> Result<(), VNextError> {
        limits.validate()?;
        self.check_inventory()?;
        check_rows(&self.rows, limits)?;
        if self
            .widths
            .iter()
            .chain(std::iter::once(&self.hidden))
            .any(|&n| n == 0 || n > i32::MAX as u64)
        {
            return Err(invalid("input geometry exceeds affine bounds"));
        }
        let mut bytes = 0u64;
        for (i, c) in self.components.iter().enumerate() {
            let local = i / 2;
            let stream = self.phase.streams()[local];
            let shape = if i % 2 == 0 {
                vec![self.hidden, self.widths[local]]
            } else {
                vec![self.hidden]
            };
            let dtype = if stream < 2 {
                ElementType::F32
            } else {
                ElementType::Bf16
            };
            if c.dimensions != shape
                || c.role != WeightComponentRole::Values
                || !c.required
                || c.external_names.len() != 1
                || c.encoding
                    != (WeightEncoding::Dense {
                        element_type: dtype,
                    })
            {
                return Err(invalid("input source shape, role or compute dtype differs"));
            }
            bytes = bytes
                .checked_add(c.physical_bytes()?)
                .ok_or_else(|| invalid("source bytes overflow"))?;
        }
        if bytes > limits.maximum_stored_bytes
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("input source or metadata capacity exceeded"));
        }
        self.schema()?
            .validate(&id(FastH3InputProjectionFamily::FAMILY_ID)?)?;
        for s in program::build(self)?.1.values() {
            s.byte_len()?;
        }
        Ok(())
    }
    fn parameter_index(&self, local: usize) -> usize {
        self.phase.streams()[local / 2] * 2 + local % 2
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        self.check_inventory()?;
        Ok(WeightSchema {
            format_id: id("weight-format.safetensors.dense")?,
            layout_id: id("weight-layout.fast_h3.input_projection.f32_media_bf16_text.v1")?,
            version: ContractVersion::new(1, 0),
            components: self.components.clone(),
            tensors: self
                .components
                .iter()
                .enumerate()
                .map(|(local, c)| {
                    Ok(WeightTensorSpec {
                        id: id(&format!(
                            "weight.fast_h3.input_projection.{}",
                            self.parameter_index(local)
                        ))?,
                        dimensions: c.dimensions.clone(),
                        logical_element_type: c.physical_element_type(),
                        physical_layout: PhysicalWeightLayout::Dense {
                            component_id: c.id.clone(),
                        },
                        required: true,
                    })
                })
                .collect::<Result<_, VNextError>>()?,
        })
    }
}

/// Uses the existing managed source promotion; original BF16 media weights
/// are explicitly converted to F32 in caller-admitted WeightBacking buffers.
pub struct FastH3InputProjectionSource<'a> {
    source: ManagedDenseArchiveSource<'a>,
    components: Vec<WeightComponentSpec>,
}
impl<'a> FastH3InputProjectionSource<'a> {
    pub fn new(
        package: &'a FastH3TransformerPackage,
        c: &FastH3InputProjectionConfig,
        host: &'a HostMemoryAccount,
        limits: FastH3InputLimits,
    ) -> Result<Self, VNextError> {
        if FastH3InputProjectionConfig::with_rows(package, c.phase, c.rows.clone(), limits)? != *c {
            return Err(invalid("input source differs from package"));
        }
        Ok(Self {
            source: ManagedDenseArchiveSource::new(package.dense_archive(), host),
            components: c.components.clone(),
        })
    }
}
impl WeightComponentSource for FastH3InputProjectionSource<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if !self.components.contains(c) {
            return Err(invalid("undeclared input source component"));
        }
        self.source.component(c)
    }
}

fn counts(layout: &FastH3PackedLayout) -> Result<[u64; 3], VNextError> {
    let mut rows = [0u64; 3];
    for s in layout.segments() {
        let i = stream(s.kind);
        rows[i] = rows[i]
            .checked_add(s.packed_rows.len().try_into().map_err(invalid)?)
            .ok_or_else(|| invalid("input row count overflow"))?;
    }
    Ok(rows)
}
fn stream(kind: FastH3SegmentKind) -> usize {
    match kind {
        FastH3SegmentKind::Text => 2,
        FastH3SegmentKind::ReferenceAudio | FastH3SegmentKind::TargetAudio => 1,
        _ => 0,
    }
}
fn check_rows(rows: &[u64], limits: FastH3InputLimits) -> Result<(), VNextError> {
    if rows.iter().any(|&r| r == 0 || r > i32::MAX as u64)
        || rows
            .iter()
            .try_fold(0u64, |n, &r| n.checked_add(r))
            .is_none_or(|n| n > limits.maximum_rows)
    {
        return Err(invalid("input rows exceed declared capacity"));
    }
    Ok(())
}
fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 input: {e}"),
    }
}
