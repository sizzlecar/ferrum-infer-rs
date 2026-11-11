use anyhow::{ensure, Context, Result};
use ferrum_models::p_h3_image::{
    PH3ImageModality, PH3ImageReferenceImage, PH3ImageRequestLayout, PH3ImageTransformerConfig,
};
use ferrum_quantization::{SafetensorsArchive, SafetensorsTensor};
use half::bf16;
use serde::Deserialize;
use std::io::Write;

/// Actual first-forward inputs, not a seed-based reconstruction. Unmasked B1
/// image editing/generation with the model's original 1/1 schedule only.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Capture {
    pub schema: String,
    pub phase: String,
    pub target_shape: [u64; 5],
    pub reference_shape: Option<[u64; 5]>,
    pub audio_shape: [u64; 4],
    pub text_tags: Vec<u32>,
    pub mod_segments: Vec<[u64; 3]>,
    pub timestep: f64,
    pub sigma: f64,
    pub shift_video: f64,
    pub shift_audio: f64,
    pub audio_scale: f64,
    pub denoise_mask_present: bool,
    pub audio_denoise_mask_present: bool,
    pub checkpoint_sha256: String,
    pub pdd_head_count: u64,
    pub block_replacements_present: bool,
    pub layout_segments: Vec<(u64, u64, String)>,
    pub img_update: Vec<bool>,
    pub audio_update: Vec<bool>,
    #[serde(default)]
    pub provenance: serde_json::Value,
}
impl Capture {
    pub fn routing(
        &self,
        t: &PH3ImageTransformerConfig,
        basis_rows: u64,
    ) -> Result<(Vec<PH3ImageModality>, Vec<u64>)> {
        ensure!(
            self.schema == "ferrum.ph3.dit_input.v1" && self.phase == "refined",
            "requires the declared refined six-port capture format"
        );
        ensure!(
            self.pdd_head_count == 1 && !self.block_replacements_present,
            "only one original PDD head without block replacement is supported"
        );
        ensure!(self.shift_video==1. && self.shift_audio==1. && self.audio_scale==1.
            && !self.denoise_mask_present && !self.audio_denoise_mask_present,
            "only actual unmasked unit audio/video scale is admitted; output sign mapping would otherwise differ");
        ensure!(
            self.sigma.is_finite()
                && self.timestep.is_finite()
                && self.sigma > 0.
                && self.sigma <= 1.
                && (self.timestep / 1000. - self.sigma).abs() <= 1e-6,
            "invalid actual sigma/timestep identity"
        );
        ensure!(
            self.checkpoint_sha256.len() == 64
                && self
                    .checkpoint_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit()),
            "checkpoint SHA must have 64 hex digits"
        );
        ensure!(
            self.target_shape[0..3] == [1, t.in_channels(), 1]
                && self.audio_shape == [1, t.audio_in_channels(), 2, 2],
            "target/audio extents differ from the declared one-image model"
        );
        let target = self.target()?;
        let reference = self.reference(t)?;
        let tags = self
            .text_tags
            .iter()
            .map(|tag| match tag {
                0 => Ok(PH3ImageModality::Image),
                1 => Ok(PH3ImageModality::Text),
                _ => anyhow::bail!("text prefix tag must be image or text"),
            })
            .collect::<Result<Vec<_>>>()?;
        let layout = PH3ImageRequestLayout::new(t, &tags, target, reference.as_slice())?;
        let text_end = tags.len() as u64;
        let audio = layout.audio_rows();
        let video = layout.target_rows();
        let mut segments = vec![(0, text_end, "text".to_owned())];
        if reference.is_some() {
            segments.push((text_end, audio.start as u64, "ref_img".to_owned()));
        }
        segments.push((audio.start as u64, audio.end as u64, "audio".to_owned()));
        segments.push((video.start as u64, video.end as u64, "video".to_owned()));
        ensure!(
            self.layout_segments == segments,
            "actual layout segments differ from checked native packing"
        );
        let ref_rows = audio.start - tags.len();
        ensure!(
            self.img_update.len() == layout.image_row_count()
                && self
                    .img_update
                    .iter()
                    .enumerate()
                    .all(|(i, v)| *v == (i >= ref_rows))
                && self.audio_update == [true; 4],
            "actual update masks differ from reference/target/audio packing"
        );
        let mut times = Vec::with_capacity(layout.rows());
        let mut end = 0;
        for &[a, b, parameter] in &self.mod_segments {
            ensure!(
                a == end && b > a && b <= layout.rows() as u64 && parameter / 3 < basis_rows,
                "modulation segments must exactly partition rows with an existing time basis"
            );
            for row in a..b {
                ensure!(
                    layout
                        .modality(usize::try_from(row)?)
                        .context("row outside layout")? as u32
                        == u32::try_from(parameter % 3)?,
                    "captured modality differs from native checked packing"
                );
                times.push(parameter / 3);
            }
            end = b;
        }
        ensure!(
            end == layout.rows() as u64,
            "modulation segments omit packed rows"
        );
        Ok((tags, times))
    }
    pub fn target(&self) -> Result<PH3ImageReferenceImage> {
        PH3ImageReferenceImage::new(
            self.target_shape[3].try_into()?,
            self.target_shape[4].try_into()?,
        )
    }
    pub fn reference(
        &self,
        t: &PH3ImageTransformerConfig,
    ) -> Result<Option<PH3ImageReferenceImage>> {
        self.reference_shape
            .map(|s| {
                ensure!(
                    s[0..3] == [1, t.in_channels(), 1],
                    "reference shape differs"
                );
                PH3ImageReferenceImage::new(s[3].try_into()?, s[4].try_into()?)
            })
            .transpose()
    }
}
pub(super) fn scalar(t: &SafetensorsTensor<'_>, index: usize) -> Result<f32> {
    let value = match t.dtype() {
        safetensors::Dtype::BF16 => {
            let at = index.checked_mul(2).context("index overflow")?;
            let b = t
                .bytes()
                .get(at..at + 2)
                .context("tensor index exceeds extent")?;
            bf16::from_bits(u16::from_le_bytes(b.try_into()?)).to_f32()
        }
        safetensors::Dtype::F32 => {
            let at = index.checked_mul(4).context("index overflow")?;
            let b = t
                .bytes()
                .get(at..at + 4)
                .context("tensor index exceeds extent")?;
            f32::from_le_bytes(b.try_into()?)
        }
        _ => anyhow::bail!("capture must preserve F32 or BF16, no implicit precision conversion"),
    };
    ensure!(value.is_finite(), "capture contains nonfinite value");
    Ok(value)
}
pub(super) fn finite(t: &SafetensorsTensor<'_>) -> Result<()> {
    let count = t
        .shape()
        .iter()
        .try_fold(1usize, |n, v| n.checked_mul(usize::try_from(*v).ok()?))
        .context("tensor extent overflow")?;
    for i in 0..count {
        scalar(t, i)?;
    }
    Ok(())
}

/// Checks the actual final-layer F32 -> returned BF16 boundary while consuming
/// the same streaming sign/packing adapter used for the comparison payload.
pub(super) struct TerminalCheck<'t, 'a, W> {
    raw: &'t SafetensorsTensor<'a>,
    inner: W,
    index: usize,
}
impl<'t, 'a, W: Write> TerminalCheck<'t, 'a, W> {
    pub fn new(raw: &'t SafetensorsTensor<'a>, inner: W) -> Self {
        Self {
            raw,
            inner,
            index: 0,
        }
    }
    pub fn finish(self) -> Result<()> {
        ensure!(
            self.raw.dtype() == safetensors::Dtype::F32
                && self.index.checked_mul(4) == Some(self.raw.bytes().len()),
            "terminal comparison did not cover exact raw F32 output"
        );
        Ok(())
    }
}
impl<W: Write> Write for TerminalCheck<'_, '_, W> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        let checked = || -> Result<()> {
            ensure!(
                b.len() == 4,
                "terminal adapter requires one complete F32 element"
            );
            let actual = f32::from_le_bytes(b.try_into()?);
            let expected = bf16::from_f32(scalar(self.raw, self.index)?).to_f32();
            ensure!(
                actual.to_bits() == expected.to_bits(),
                "final F32 rounding/sign/packing disagrees with returned BF16 at element {}",
                self.index
            );
            Ok(())
        };
        checked()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        self.inner.write_all(b)?;
        self.index += 1;
        Ok(4)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Stream the exact sign/layout adapter only. No rescale, quantization, or
/// trigonometric reconstruction; the native result remains positive BF16 rows.
pub(super) fn reference_image(
    t: &SafetensorsTensor<'_>,
    shape: [u64; 5],
    patch: [u64; 3],
    mut out: impl Write,
) -> Result<()> {
    ensure!(
        t.dtype() == safetensors::Dtype::BF16
            && t.shape() == shape
            && shape[0] == 1
            && shape[2] == 1
            && patch[0] == 1
            && patch[1] > 0
            && patch[2] > 0
            && shape[3] % patch[1] == 0
            && shape[4] % patch[2] == 0,
        "returned image cannot map exactly to native patch rows"
    );
    let [_, c, _, h, w] = shape;
    let [_, ph, pw] = patch;
    for y in (0..h).step_by(ph.try_into()?) {
        for x in (0..w).step_by(pw.try_into()?) {
            for channel in 0..c {
                for dy in 0..ph {
                    for dx in 0..pw {
                        let index = ((channel * h + y + dy) * w + x + dx).try_into()?;
                        out.write_all(&(-scalar(t, index)?).to_le_bytes())?;
                    }
                }
            }
        }
    }
    Ok(())
}
pub(super) fn reference_audio(
    t: &SafetensorsTensor<'_>,
    shape: [u64; 4],
    mut out: impl Write,
) -> Result<()> {
    ensure!(
        t.dtype() == safetensors::Dtype::BF16
            && t.shape() == shape
            && shape[0] == 1
            && shape[2] == 2,
        "returned audio shape/dtype differs"
    );
    let [_, channels, stereo, time] = shape;
    for ear in 0..stereo {
        for step in 0..time {
            for channel in 0..channels {
                let index = ((channel * stereo + ear) * time + step).try_into()?;
                out.write_all(&(-scalar(t, index)?).to_le_bytes())?;
            }
        }
    }
    Ok(())
}
pub(super) fn open_archive(path: &std::path::Path, maximum: usize) -> Result<SafetensorsArchive> {
    let m = path.metadata()?;
    ensure!(
        m.is_file() && m.len() > 0 && m.len() <= u64::try_from(maximum)?,
        "capture exceeds input file budget"
    );
    SafetensorsArchive::open_single_file(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new(".")),
        path.file_name()
            .and_then(|s| s.to_str())
            .context("capture basename must be UTF-8")?,
    )
    .map_err(Into::into)
}

pub(super) const PACKED_ID: &str = "value.p_h3_image.dit_forward.packed";
pub(super) const MOD1_ID: &str =
    "value.p_h3_image.curve_stack.layer0.value.p_h3_image.curve_block.mod1";
pub(super) const FIRST_BLOCK_TAPS: [(&str, &str); 2] = [
    ("packed", PACKED_ID),
    (
        "block0_output",
        "value.p_h3_image.curve_stack.layer0.value.p_h3_image.curve_block.output",
    ),
];
pub(super) const FIRST_ATTENTION_TAPS: [(&str, &str); 2] = [
    ("block0_mod1", MOD1_ID),
    (
        "block0_residual1",
        "value.p_h3_image.curve_stack.layer0.value.p_h3_image.curve_block.residual1",
    ),
];
pub(super) const FIRST_ATTENTION_OUTPUT_TAPS: [(&str, &str); 3] = [
    ("block0_mod1", MOD1_ID),
    (
        "block0_attention_out",
        "value.p_h3_image.curve_stack.layer0.value.p_h3_image.curve_block.attention_out",
    ),
    (
        "block0_adaln_rows",
        "value.p_h3_image.curve_stack.layer0.value.p_h3_image.curve_block.adaln_rows",
    ),
];
pub(super) const FIRST_QKV_TAPS: [(&str, &str); 2] = [
    ("block0_mod1", MOD1_ID),
    (
        "block0_qkv",
        "value.p_h3_image.curve_stack.layer0.value.p_h3_image.curve_block.qkv",
    ),
];
pub(super) const FIRST_ATTENTION_CORE_TAPS: [(&str, &str); 2] = [
    ("block0_mod1", MOD1_ID),
    (
        "block0_attention",
        "value.p_h3_image.curve_stack.layer0.value.p_h3_image.curve_block.attention",
    ),
];

pub(super) fn taps(
    boundary: super::InputBoundary,
    enabled: bool,
    selection: super::FirstBlockTaps,
) -> &'static [(&'static str, &'static str)] {
    if !enabled {
        &[]
    } else if selection != super::FirstBlockTaps::Boundaries {
        let selected = match selection {
            super::FirstBlockTaps::Attention => &FIRST_ATTENTION_TAPS[..],
            super::FirstBlockTaps::AttentionOutput => &FIRST_ATTENTION_OUTPUT_TAPS[..],
            super::FirstBlockTaps::Qkv => &FIRST_QKV_TAPS[..],
            super::FirstBlockTaps::AttentionCore => &FIRST_ATTENTION_CORE_TAPS[..],
            super::FirstBlockTaps::Boundaries => unreachable!(),
        };
        if boundary == super::InputBoundary::Mod1 {
            // Supplied mod1 is an input receipt, never a duplicate readback.
            &selected[1..]
        } else {
            selected
        }
    } else if boundary == super::InputBoundary::Packed {
        // The supplied packed bytes already have an input receipt; no duplicate readback.
        &FIRST_BLOCK_TAPS[1..]
    } else {
        &FIRST_BLOCK_TAPS
    }
}

/// A diagnostic tap keeps the exact original compiler-declared extent. In
/// particular, compact AdaLN rows are not token-by-hidden activation matrices.
pub(super) fn tap_tensor_spec(
    options: &ferrum_runtime::ProgramPlanCompileOptions,
    value: &str,
) -> Result<ferrum_runtime::ResolvedTensorSpec> {
    use ferrum_runtime::{ElementType, ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec};
    let id = ProgramValueId::new(value)?;
    let spec = options
        .tensor_specs()
        .get(&id)
        .context("tap tensor spec missing")?;
    ensure!(
        spec.element_type == ElementType::Bf16
            && spec.dimensions.len() == 2
            && spec.dimensions.iter().all(|&n| n > 0)
            && spec.layout == ResolvedTensorLayout::Contiguous,
        "tap must be an original contiguous BF16 rank2 tensor"
    );
    Ok(ResolvedTensorSpec::new(
        spec.dimensions.clone(),
        spec.element_type,
        spec.layout.clone(),
    )?)
}

pub(super) fn input_bindings(
    boundary: super::InputBoundary,
) -> &'static [(&'static str, &'static str)] {
    use ferrum_models::p_h3_image::PH3ImageCurveBlockFamily as Block;
    match boundary {
        super::InputBoundary::Prepared => &[
            (
                "image_patches",
                "value.p_h3_image.dit_forward.image_patches_f32",
            ),
            (
                "audio_patches",
                "value.p_h3_image.dit_forward.audio_rows_f32",
            ),
            ("text", "value.p_h3_image.token_refiner.output"),
            ("time_basis", Block::BASIS_ID),
            ("rope_cos", Block::COSINE_ID),
            ("rope_sin", Block::SINE_ID),
        ],
        super::InputBoundary::Packed => &[
            ("packed", PACKED_ID),
            ("time_basis", Block::BASIS_ID),
            ("rope_cos", Block::COSINE_ID),
            ("rope_sin", Block::SINE_ID),
        ],
        super::InputBoundary::Mod1 => &[
            ("block0_mod1", MOD1_ID),
            (
                "image_patches",
                "value.p_h3_image.dit_forward.image_patches_f32",
            ),
            (
                "audio_patches",
                "value.p_h3_image.dit_forward.audio_rows_f32",
            ),
            ("text", "value.p_h3_image.token_refiner.output"),
            ("time_basis", Block::BASIS_ID),
            ("rope_cos", Block::COSINE_ID),
            ("rope_sin", Block::SINE_ID),
        ],
    }
}

/// Backward dependency closure with one internal activation promoted to input.
/// Retained nodes are cloned unchanged in original order; all weight references
/// survive, including now-unused projection weights, for the original shared ABI.
pub(super) fn cut_at_input(
    source: &ferrum_runtime::ModelProgram,
    family: ferrum_runtime::ModelFamilyId,
    cutoff: &ferrum_runtime::ProgramValueId,
    spec: ferrum_runtime::ProgramTensorSpec,
    taps: &[ferrum_runtime::ProgramValueId],
) -> Result<ferrum_runtime::ModelProgram> {
    use std::collections::BTreeSet;
    ensure!(
        source.states().is_empty() && source.checkpoint_inputs().is_none(),
        "activation cutoff only supports this stateless tensor program"
    );
    let observed = append_tap_outputs(source, family.clone(), taps)?;
    ensure!(
        source
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .any(|n| n.outputs.contains(cutoff)),
        "cutoff must be an original internal output"
    );
    ensure!(
        !observed.outputs().contains(cutoff),
        "cutoff cannot also be a diagnostic output"
    );
    let mut needed = observed.outputs().iter().cloned().collect::<BTreeSet<_>>();
    let mut retained = BTreeSet::new();
    for node in source
        .blocks()
        .iter()
        .rev()
        .flat_map(|b| b.nodes.iter().rev())
    {
        if node.outputs.contains(cutoff) {
            ensure!(
                node.outputs
                    .iter()
                    .all(|o| o == cutoff || !needed.contains(o)),
                "cutoff producer has another required output"
            );
            continue;
        }
        if node.outputs.iter().any(|o| needed.contains(o)) {
            retained.insert(node.id.clone());
            for output in &node.outputs {
                needed.remove(output);
            }
            needed.extend(node.inputs.iter().cloned());
        }
    }
    ensure!(
        needed.contains(cutoff),
        "requested outputs do not depend on cutoff"
    );
    let weights = source
        .weights()
        .iter()
        .map(|w| w.value_id.clone())
        .collect::<BTreeSet<_>>();
    let inputs = std::iter::once(cutoff.clone())
        .chain(
            source
                .inputs()
                .iter()
                .filter(|id| needed.contains(*id))
                .cloned(),
        )
        .collect::<Vec<_>>();
    ensure!(
        needed
            .iter()
            .all(|id| inputs.contains(id) || weights.contains(id)),
        "cutoff leaves an undeclared dependency"
    );
    let mut specs = source
        .input_tensor_specs()
        .iter()
        .filter(|(id, _)| inputs.contains(id))
        .map(|(id, spec)| (id.clone(), spec.clone()))
        .collect::<std::collections::BTreeMap<_, _>>();
    specs.insert(cutoff.clone(), spec);
    let blocks = source
        .blocks()
        .iter()
        .filter_map(|b| {
            let nodes = b
                .nodes
                .iter()
                .filter(|n| retained.contains(&n.id))
                .cloned()
                .collect::<Vec<_>>();
            (!nodes.is_empty()).then(|| ferrum_runtime::ProgramBlock {
                id: b.id.clone(),
                nodes,
            })
        })
        .collect();
    Ok(ferrum_runtime::ModelProgram::new(
        family,
        inputs,
        blocks,
        vec![],
        source.weights().to_vec(),
        observed.outputs().to_vec(),
    )?
    .with_input_tensor_specs(specs)?)
}

/// Only output membership changes. No operation, weight, state, input ABI,
/// ordering or arithmetic is reconstructed by this diagnostic.
pub(super) fn append_tap_outputs(
    source: &ferrum_runtime::ModelProgram,
    family: ferrum_runtime::ModelFamilyId,
    taps: &[ferrum_runtime::ProgramValueId],
) -> Result<ferrum_runtime::ModelProgram> {
    use std::collections::BTreeSet;
    let produced = source
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .flat_map(|n| &n.outputs)
        .collect::<BTreeSet<_>>();
    let mut outputs = source.outputs().to_vec();
    for tap in taps {
        ensure!(
            produced.contains(tap) && !outputs.contains(tap),
            "tap must be one distinct internal operation output"
        );
        outputs.push(tap.clone());
    }
    let mut program = ferrum_runtime::ModelProgram::new(
        family,
        source.inputs().to_vec(),
        source.blocks().to_vec(),
        source.states().to_vec(),
        source.weights().to_vec(),
        outputs,
    )?
    .with_input_tensor_specs(source.input_tensor_specs().clone())?;
    if let Some(inputs) = source.checkpoint_inputs() {
        program = program.with_checkpoint_inputs(inputs.clone())?;
    }
    Ok(program)
}

#[derive(Clone, serde::Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ObservationConfig {
    source_family_fingerprint: String,
    source_config: serde_json::Value,
    taps: Vec<ferrum_runtime::ProgramValueId>,
    input_boundary: super::InputBoundary,
    cutoff: Option<ferrum_runtime::ProgramValueId>,
    dense_linear_source: super::DenseLinearSource,
    #[serde(default, skip_serializing_if = "super::FirstBlockTaps::is_boundaries")]
    first_block_taps: super::FirstBlockTaps,
}
struct ObservedFamily {
    source: ferrum_runtime::PreparedModelFamily,
    config: ObservationConfig,
    family: ferrum_runtime::ModelFamilyId,
    metadata: ferrum_runtime::ExternalModelMetadataId,
    profile: ferrum_runtime::NumericalExecutionProfile,
    program: ferrum_runtime::ModelProgram,
}
fn observation_error(message: impl Into<String>) -> ferrum_runtime::VNextError {
    ferrum_runtime::VNextError::InvalidExecutionPlan {
        reason: format!("actual DiT observation: {}", message.into()),
    }
}
impl ferrum_runtime::ModelFamilyProvider for ObservedFamily {
    type Config = ObservationConfig;
    fn family_id(&self) -> &ferrum_runtime::ModelFamilyId {
        &self.family
    }
    fn external_metadata_ids(
        &self,
    ) -> std::collections::BTreeSet<ferrum_runtime::ExternalModelMetadataId> {
        std::collections::BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &Self::Config,
    ) -> std::result::Result<(), ferrum_runtime::VNextError> {
        if config != &self.config
            || serde_json::to_value(config).map_err(|e| observation_error(e.to_string()))? != *raw
        {
            return Err(observation_error(
                "source fingerprint/config and explicit taps must match this diagnostic",
            ));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &Self::Config,
    ) -> std::result::Result<ferrum_runtime::ExternalModelMetadataId, ferrum_runtime::VNextError>
    {
        self.validate_config_identity(raw, config)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(
        &self,
        raw: &serde_json::Value,
    ) -> std::result::Result<Self::Config, ferrum_runtime::VNextError> {
        let c =
            serde_json::from_value(raw.clone()).map_err(|e| observation_error(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(
        &self,
        _: &Self::Config,
    ) -> std::result::Result<ferrum_runtime::ModelWeights, ferrum_runtime::VNextError> {
        Ok(self.source.weights().clone())
    }
    fn numerical_profiles(
        &self,
        _: &Self::Config,
    ) -> std::result::Result<ferrum_runtime::FamilyNumericalProfiles, ferrum_runtime::VNextError>
    {
        ferrum_runtime::FamilyNumericalProfiles::new(
            &self.family,
            ferrum_runtime::ContractVersion::new(1, 0),
            vec![self.profile.clone()],
            vec![self.profile.id.clone()],
        )
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &ferrum_runtime::NumericalExecutionProfile,
    ) -> std::result::Result<ferrum_runtime::ModelProgram, ferrum_runtime::VNextError> {
        if config != &self.config || profile != &self.profile {
            return Err(observation_error("unrecognized observation profile"));
        }
        Ok(self.program.clone())
    }
    fn semantic_metadata(
        &self,
        _: &Self::Config,
    ) -> std::result::Result<ferrum_runtime::ModelSemanticMetadata, ferrum_runtime::VNextError>
    {
        Ok(self.source.metadata().clone())
    }
}

pub(super) fn observed_family(
    source: &ferrum_runtime::PreparedModelFamily,
    capture_first_block: bool,
    first_block_taps: super::FirstBlockTaps,
    boundary: super::InputBoundary,
    options: &ferrum_runtime::ProgramPlanCompileOptions,
    source_policy: super::DenseLinearSource,
) -> Result<ferrum_runtime::PreparedModelFamily> {
    ensure!(
        capture_first_block || first_block_taps == super::FirstBlockTaps::Boundaries,
        "attention taps require enabled capture"
    );
    if !capture_first_block
        && boundary == super::InputBoundary::Prepared
        && source_policy == super::DenseLinearSource::Archive
    {
        return Ok(source.clone());
    }
    let taps = taps(boundary, capture_first_block, first_block_taps)
        .iter()
        .map(|(_, value)| ferrum_runtime::ProgramValueId::new(*value))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    prepare_variant(
        source,
        taps,
        boundary,
        Some(options),
        source_policy,
        first_block_taps,
    )
}
pub(super) fn prepare_observation(
    source: &ferrum_runtime::PreparedModelFamily,
    taps: Vec<ferrum_runtime::ProgramValueId>,
    selection: super::FirstBlockTaps,
) -> Result<ferrum_runtime::PreparedModelFamily> {
    prepare_variant(
        source,
        taps,
        super::InputBoundary::Prepared,
        None,
        super::DenseLinearSource::Archive,
        selection,
    )
}
pub(super) fn installation_family(
    source: &ferrum_runtime::PreparedModelFamily,
    policy: super::DenseLinearSource,
) -> Result<ferrum_runtime::PreparedModelFamily> {
    if policy == super::DenseLinearSource::Archive {
        return Ok(source.clone());
    }
    prepare_variant(
        source,
        vec![],
        super::InputBoundary::Prepared,
        None,
        policy,
        super::FirstBlockTaps::Boundaries,
    )
}
fn prepare_variant(
    source: &ferrum_runtime::PreparedModelFamily,
    taps: Vec<ferrum_runtime::ProgramValueId>,
    boundary: super::InputBoundary,
    options: Option<&ferrum_runtime::ProgramPlanCompileOptions>,
    source_policy: super::DenseLinearSource,
    first_block_taps: super::FirstBlockTaps,
) -> Result<ferrum_runtime::PreparedModelFamily> {
    use ferrum_runtime::{
        ElementType, ExternalModelMetadataId, ModelFamilyId, ModelFamilyRegistration,
        NumericalProfileId, TypedFamilyRegistration,
    };
    ensure!(
        !taps.is_empty()
            || boundary != super::InputBoundary::Prepared
            || source_policy != super::DenseLinearSource::Archive,
        "observed family requires explicit nonempty taps"
    );
    ensure!(
        taps.iter()
            .all(|tap| source.numerical_profile().boundaries.get(tap) == Some(&ElementType::Bf16)),
        "selected original hidden boundaries must be BF16"
    );
    let cutoff = match boundary {
        super::InputBoundary::Prepared => None,
        super::InputBoundary::Packed => Some(PACKED_ID),
        super::InputBoundary::Mod1 => Some(MOD1_ID),
    }
    .map(ferrum_runtime::ProgramValueId::new)
    .transpose()?;
    let config = ObservationConfig {
        source_family_fingerprint: source.fingerprint()?,
        source_config: source.canonical_config().clone(),
        taps,
        input_boundary: boundary,
        cutoff: cutoff.clone(),
        dense_linear_source: source_policy,
        first_block_taps,
    };
    let name = match (boundary, first_block_taps) {
        (super::InputBoundary::Packed, super::FirstBlockTaps::Boundaries) => "packed_input",
        (super::InputBoundary::Packed, super::FirstBlockTaps::Attention) => {
            "packed_input_attention"
        }
        (super::InputBoundary::Packed, super::FirstBlockTaps::AttentionOutput) => {
            "packed_input_attention_output"
        }
        (super::InputBoundary::Packed, super::FirstBlockTaps::Qkv) => "packed_input_qkv",
        (super::InputBoundary::Packed, super::FirstBlockTaps::AttentionCore) => {
            "packed_input_attention_core"
        }
        (super::InputBoundary::Mod1, super::FirstBlockTaps::Boundaries) => "mod1_input",
        (super::InputBoundary::Mod1, super::FirstBlockTaps::Attention) => "mod1_input_attention",
        (super::InputBoundary::Mod1, super::FirstBlockTaps::AttentionOutput) => {
            "mod1_input_attention_output"
        }
        (super::InputBoundary::Mod1, super::FirstBlockTaps::Qkv) => "mod1_input_qkv",
        (super::InputBoundary::Mod1, super::FirstBlockTaps::AttentionCore) => {
            "mod1_input_attention_core"
        }
        (super::InputBoundary::Prepared, super::FirstBlockTaps::Boundaries) => "first_block",
        (super::InputBoundary::Prepared, super::FirstBlockTaps::Attention) => {
            "first_block_attention"
        }
        (super::InputBoundary::Prepared, super::FirstBlockTaps::AttentionOutput) => {
            "first_block_attention_output"
        }
        (super::InputBoundary::Prepared, super::FirstBlockTaps::Qkv) => "first_block_qkv",
        (super::InputBoundary::Prepared, super::FirstBlockTaps::AttentionCore) => {
            "first_block_attention_core"
        }
    };
    let suffix = if source_policy == super::DenseLinearSource::Bf16Roundtrip {
        ".bf16_roundtrip"
    } else {
        ""
    };
    let family = ModelFamilyId::new(format!(
        "family.diagnostic.p_h3_image.actual_dit.{name}{suffix}"
    ))?;
    let program = if let Some(cutoff) = cutoff {
        let spec = options
            .context("activation input needs original compile specs")?
            .tensor_specs()
            .get(&cutoff)
            .context("original activation spec is missing")?
            .clone();
        ensure!(
            spec.element_type == ElementType::Bf16
                && spec.dimensions.len() == 2
                && spec.layout == ferrum_runtime::ResolvedTensorLayout::Contiguous,
            "activation input must be original contiguous BF16 rank2"
        );
        let program = cut_at_input(
            source.program(),
            family.clone(),
            &cutoff,
            spec,
            &config.taps,
        )?;
        let expected = input_bindings(boundary)
            .iter()
            .map(|(_, id)| ferrum_runtime::ProgramValueId::new(*id))
            .collect::<std::result::Result<std::collections::BTreeSet<_>, _>>()?;
        ensure!(
            program
                .inputs()
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                == expected,
            "activation dependency closure differs from the exact selected input ABI"
        );
        program
    } else {
        append_tap_outputs(source.program(), family.clone(), &config.taps)?
    };
    let mut profile = source.numerical_profile().clone();
    profile.family_id = family.clone();
    profile.id = NumericalProfileId::new(format!("{}.{name}_observation{suffix}.v1", profile.id))
        .map_err(anyhow::Error::msg)?;
    let nodes = program
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .collect::<Vec<_>>();
    let outputs = nodes
        .iter()
        .flat_map(|n| &n.outputs)
        .collect::<std::collections::BTreeSet<_>>();
    let operations = nodes
        .iter()
        .map(|n| &n.operation_id)
        .collect::<std::collections::BTreeSet<_>>();
    profile.boundaries.retain(|id, _| outputs.contains(id));
    profile
        .operations
        .retain(|op| operations.contains(&op.operation_id));
    profile.validate_program(&program)?;
    let profile_id = profile.id.clone();
    let raw = serde_json::to_value(&config)?;
    let provider = ObservedFamily {
        source: source.clone(),
        config,
        family,
        metadata: ExternalModelMetadataId::new(format!(
            "metadata.diagnostic.p_h3_image.actual_dit.{name}{suffix}"
        ))?,
        profile,
        program,
    };
    Ok(TypedFamilyRegistration::new(provider).prepare_with_profile(&raw, &profile_id)?)
}

pub(super) fn execution_options(
    source: &ferrum_runtime::ProgramPlanCompileOptions,
    family: &ferrum_runtime::PreparedModelFamily,
    boundary: super::InputBoundary,
) -> Result<ferrum_runtime::ProgramPlanCompileOptions> {
    use ferrum_runtime::ProgramPlanCompileOptions;
    if boundary == super::InputBoundary::Prepared {
        return Ok(source.clone());
    }
    // The current adapter declares only specs plus arena packing. Reject future
    // hints rather than silently dropping a provider/materializer/retention rule.
    ensure!(
        *source
            == ProgramPlanCompileOptions::new(source.tensor_specs().clone())?
                .with_weight_arena_packing(source.weight_arena_packing()),
        "activation cutoff cannot discard nondefault compile policies"
    );
    let program = family.program();
    let values = program
        .inputs()
        .iter()
        .chain(
            program
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .flat_map(|n| &n.outputs),
        )
        .collect::<std::collections::BTreeSet<_>>();
    let specs = source
        .tensor_specs()
        .iter()
        .filter(|(id, _)| values.contains(id))
        .map(|(id, spec)| (id.clone(), spec.clone()))
        .collect();
    Ok(ProgramPlanCompileOptions::new(specs)?
        .with_weight_arena_packing(source.weight_arena_packing()))
}
