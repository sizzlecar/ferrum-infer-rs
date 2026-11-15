//! Actual block norm weights and compact AdaLN tables for one declared request
//! evaluation. Time routing and token modality are independent inputs.
use super::{
    FastH3ConditioningSchedule, FastH3PackedLayout, FastH3SegmentKind, FastH3TransformerPackage,
    FastH3WeightStorage,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod routing;
pub use routing::FastH3ConditioningRole;
use routing::Routing;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3ModulationStage {
    Attention,
    FeedForward,
}
impl FastH3ModulationStage {
    fn kind(self) -> RowSelectedRmsModulationKind {
        match self {
            Self::Attention => RowSelectedRmsModulationKind::Normalize,
            Self::FeedForward => RowSelectedRmsModulationKind::Residual,
        }
    }
    fn norm(self) -> usize {
        match self {
            Self::Attention => 1,
            Self::FeedForward => 2,
        }
    }
}
/// The text presentation can contain vision tokens. A caller must supply the
/// actual tokenizer/presentation tags; there is deliberately no default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3TextTokenTag {
    Vision,
    Text,
}
impl FastH3TextTokenTag {
    fn modality(self) -> u64 {
        match self {
            Self::Vision => 0,
            Self::Text => 1,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3TextTagRun {
    pub rows: u64,
    pub tag: FastH3TextTokenTag,
}

#[derive(Clone, Copy, Debug)]
pub struct FastH3ModulationLimits {
    pub maximum_rows: u64,
    pub maximum_time_rows: usize,
    pub maximum_metadata_bytes: usize,
    pub maximum_stored_bytes: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    rows: u64,
    time_row: usize,
    modality: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3ModulationConfig {
    block: usize,
    stage: FastH3ModulationStage,
    rows: u64,
    width: u64,
    times: Vec<f32>,
    routing: Routing,
    epsilon: CanonicalRational,
    gamma: WeightComponentSpec,
}
impl FastH3ModulationConfig {
    pub fn from_package(
        package: &FastH3TransformerPackage,
        block: usize,
        stage: FastH3ModulationStage,
        layout: &FastH3PackedLayout,
        schedule: &FastH3ConditioningSchedule,
        evaluation: usize,
        text_tags: &[FastH3TextTagRun],
        limits: FastH3ModulationLimits,
    ) -> Result<Self, VNextError> {
        let gamma = gamma(package, block, stage)?;
        let mut runs = Vec::<Run>::new();
        let mut saw_text = false;
        for segment in layout.segments() {
            let time_row = schedule
                .segment_time_index(evaluation, segment.kind)
                .ok_or_else(|| invalid("segment has no declared conditioning time"))?;
            let rows = u64::try_from(segment.packed_rows.len()).map_err(invalid)?;
            if segment.kind == FastH3SegmentKind::Text {
                if saw_text || text_tags.is_empty() {
                    return Err(invalid("requires actual text-presentation tag runs"));
                }
                saw_text = true;
                let total = text_tags
                    .iter()
                    .try_fold(0u64, |n, r| n.checked_add(r.rows).filter(|_| r.rows > 0))
                    .ok_or_else(|| invalid("invalid text tag extent"))?;
                if total != rows {
                    return Err(invalid(
                        "text tags must cover exactly the text-presentation segment",
                    ));
                }
                for tag in text_tags {
                    push_run(&mut runs, tag.rows, time_row, tag.tag.modality())?;
                }
            } else {
                let modality = match segment.kind {
                    FastH3SegmentKind::ReferenceAudio | FastH3SegmentKind::TargetAudio => 2,
                    _ => 0,
                };
                push_run(&mut runs, rows, time_row, modality)?;
            }
        }
        if !saw_text {
            return Err(invalid("request has no text-presentation segment"));
        }
        let c = Self {
            block,
            stage,
            rows: layout.sequence_rows().try_into().map_err(invalid)?,
            width: package.config().hidden_size().try_into().map_err(invalid)?,
            times: schedule.timesteps().to_vec(),
            routing: Routing::Scheduled { runs },
            epsilon: CanonicalRational::from_decimal_str(&package.config().norm_eps().to_string())?,
            gamma,
        };
        c.validate(limits)?;
        Ok(c)
    }
    pub fn block(&self) -> usize {
        self.block
    }
    /// Concatenate checked request rows in batch order while preserving each
    /// request's evaluation and text/vision/audio routing. All requests share
    /// one original block table and norm weight; no per-token table expansion.
    /// Attention batch geometry and scheduler admission remain caller contracts.
    pub fn concatenate_requests(
        requests: &[Self],
        limits: FastH3ModulationLimits,
    ) -> Result<Self, VNextError> {
        let first = requests
            .first()
            .ok_or_else(|| invalid("empty modulation request batch"))?;
        first.validate(limits)?;
        let mut combined = first.clone();
        combined.rows = 0;
        combined.routing = Routing::Scheduled { runs: Vec::new() };
        for request in requests {
            request.validate(limits)?;
            if request.block != first.block
                || request.stage != first.stage
                || request.width != first.width
                || request.gamma != first.gamma
                || request.epsilon != first.epsilon
                || !request
                    .times
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(first.times.iter().map(|v| v.to_bits()))
            {
                return Err(invalid("batched modulation requests must share the original block/table interpretation"));
            }
            combined.rows = combined
                .rows
                .checked_add(request.rows)
                .filter(|n| *n <= limits.maximum_rows)
                .ok_or_else(|| invalid("batched modulation rows exceed capacity"))?;
            let Routing::Scheduled { runs } = &request.routing else {
                return Err(invalid(
                    "scheduled concatenation requires scheduled routing",
                ));
            };
            let Routing::Scheduled {
                runs: combined_runs,
            } = &mut combined.routing
            else {
                unreachable!()
            };
            for run in runs {
                push_run(combined_runs, run.rows, run.time_row, run.modality)?;
            }
        }
        combined.validate(limits)?;
        Ok(combined)
    }
    pub fn rows(&self) -> u64 {
        self.rows
    }
    pub fn width(&self) -> u64 {
        self.width
    }
    pub(in crate::fast_h3) fn normalization_epsilon(&self) -> CanonicalRational {
        self.epsilon
    }
    pub fn time_values(&self) -> &[f32] {
        &self.times
    }
    pub fn stage(&self) -> FastH3ModulationStage {
        self.stage
    }
    pub fn parameter_row_runs(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.routing.parameter_row_runs()
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(self.program()?.1)
    }
    /// Import the original contiguous Time component's BF16 [time,18H] bytes
    /// as [3*time,6H]. This admits one compact InputPreprocess copy, never a
    /// per-token expanded table or a relabelled Output-class allocation.
    pub fn managed_table(
        &self,
        host: &HostMemoryAccount,
        bytes: &[u8],
    ) -> Result<ManagedHostTensor, VNextError> {
        if !matches!(self.routing, Routing::Scheduled { .. }) {
            return Err(invalid(
                "role routing requires an explicitly selected live conditioning table",
            ));
        }
        let spec = self.table_spec()?;
        if spec.byte_len()? != bytes.len() as u64 {
            return Err(invalid(
                "table extent differs from declared time/modality rows",
            ));
        }
        let mut owned = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
            .map_err(invalid)?;
        owned.as_mut().copy_from_slice(bytes);
        ManagedHostTensor::new(
            id(FastH3ModulationFamily::TABLE_ID)?,
            ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout)?,
            owned.freeze(),
        )
    }
    fn table_spec(&self) -> Result<ProgramTensorSpec, VNextError> {
        Ok(tensor(vec![
            self.routing.table_rows(self.times.len())?,
            self.width
                .checked_mul(6)
                .ok_or_else(|| invalid("table width overflows"))?,
        ]))
    }
    fn attributes(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        let mut a = RowSelectedRmsModulationProfile::Block8192Vec8Warp16Fma.attributes()?;
        let (shift, scale) = match self.stage {
            FastH3ModulationStage::Attention => (0, self.width),
            FastH3ModulationStage::FeedForward => (3 * self.width, 4 * self.width),
        };
        let (lengths, indices): (Vec<_>, Vec<_>) = self
            .parameter_row_runs()
            .map(|(n, i)| {
                Ok((
                    i64::try_from(n).map_err(invalid)?,
                    i64::try_from(i).map_err(invalid)?,
                ))
            })
            .collect::<Result<Vec<_>, VNextError>>()?
            .into_iter()
            .unzip();
        for (k, v) in [
            ("epsilon", SemanticValue::Rational(self.epsilon)),
            ("run_lengths", SemanticValue::Integers(lengths)),
            ("parameter_rows", SemanticValue::Integers(indices)),
            ("shift_column", SemanticValue::Unsigned(shift)),
            ("scale_column", SemanticValue::Unsigned(scale)),
        ] {
            a.insert(AttributeId::new(k)?, v);
        }
        if self.stage == FastH3ModulationStage::FeedForward {
            a.insert(
                AttributeId::new("gate_column")?,
                SemanticValue::Unsigned(2 * self.width),
            );
        }
        Ok(a)
    }
    fn validate(&self, limits: FastH3ModulationLimits) -> Result<(), VNextError> {
        if limits.maximum_rows == 0
            || limits.maximum_time_rows == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
            || self.rows == 0
            || self.rows > limits.maximum_rows
            || !(1..=8192).contains(&self.width)
            || self.times.is_empty()
            || self.times.len() > limits.maximum_time_rows
            || self
                .times
                .iter()
                .any(|t| !t.is_finite() || !(0.0..=1.0).contains(t))
            || !self.times.windows(2).all(|w| w[0] < w[1])
        {
            return Err(invalid("invalid modulation geometry, times or capacity"));
        }
        self.routing.validate(self.rows, self.times.len())?;
        if self.gamma.dimensions != [self.width]
            || self.gamma.role != WeightComponentRole::Values
            || self.gamma.encoding
                != (WeightEncoding::Dense {
                    element_type: ElementType::Bf16,
                })
            || !self.gamma.required
            || self.gamma.external_names.len() != 1
        {
            return Err(invalid("requires original dense BF16 norm weight"));
        }
        if self.gamma.physical_bytes()? > limits.maximum_stored_bytes
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("source or metadata capacity exceeded"));
        }
        let x = ResolvedTensorSpec::new(
            vec![self.rows, self.width],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let t = self.table_spec()?;
        let t = ResolvedTensorSpec::new(t.dimensions, t.element_type, t.layout)?;
        let gamma = ResolvedTensorSpec::new(
            vec![self.width],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let mut inputs = vec![&x, &t, &gamma];
        let mut outputs = vec![&x];
        if self.stage == FastH3ModulationStage::FeedForward {
            inputs.push(&x);
            outputs.push(&x);
        }
        RowSelectedRmsModulationGeometry::new(
            self.stage.kind(),
            &inputs,
            &outputs,
            &self.attributes()?,
        )?;
        self.schema()?
            .validate(&id(FastH3ModulationFamily::FAMILY_ID)?)
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.safetensors.dense")?,
            layout_id: id("weight-layout.fast_h3.modulation.gamma.v1")?,
            version: ContractVersion::new(1, 0),
            components: vec![self.gamma.clone()],
            tensors: vec![WeightTensorSpec {
                id: id("weight.fast_h3.modulation.gamma")?,
                dimensions: vec![self.width],
                logical_element_type: ElementType::Bf16,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: self.gamma.id.clone(),
                },
                required: true,
            }],
        })
    }
    fn program(
        &self,
    ) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
        let input: ProgramValueId = id(FastH3ModulationFamily::INPUT_ID)?;
        let table: ProgramValueId = id(FastH3ModulationFamily::TABLE_ID)?;
        let gamma: ProgramValueId = id("value.fast_h3.modulation.gamma")?;
        let output: ProgramValueId = id(FastH3ModulationFamily::OUTPUT_ID)?;
        let mut specs = BTreeMap::from([
            (input.clone(), tensor(vec![self.rows, self.width])),
            (table.clone(), self.table_spec()?),
        ]);
        let mut inputs = vec![input.clone(), table.clone(), gamma.clone()];
        let mut external = vec![input, table];
        let mut outputs = vec![output.clone()];
        if self.stage == FastH3ModulationStage::FeedForward {
            let branch: ProgramValueId = id(FastH3ModulationFamily::BRANCH_ID)?;
            specs.insert(branch.clone(), tensor(vec![self.rows, self.width]));
            external.push(branch.clone());
            inputs.push(branch);
            outputs.insert(0, id(FastH3ModulationFamily::HIDDEN_ID)?);
        }
        let input_specs = specs.clone();
        for v in &outputs {
            specs.insert(v.clone(), tensor(vec![self.rows, self.width]));
        }
        let program = ModelProgram::new(
            id(FastH3ModulationFamily::FAMILY_ID)?,
            external,
            vec![ProgramBlock {
                id: "block.fast_h3.modulation".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fast_h3.modulation")?,
                    operation_id: id(self.stage.kind().operation_id())?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs,
                    outputs: outputs.clone(),
                    attributes: self.attributes()?,
                }],
            }],
            vec![],
            vec![WeightReference {
                weight_id: id("weight.fast_h3.modulation.gamma")?,
                value_id: gamma,
                tensor: tensor(vec![self.width]),
            }],
            outputs,
        )?
        .with_input_tensor_specs(input_specs)?;
        Ok((program, specs))
    }
}
fn push_run(
    runs: &mut Vec<Run>,
    rows: u64,
    time_row: usize,
    modality: u64,
) -> Result<(), VNextError> {
    if let Some(last) = runs
        .last_mut()
        .filter(|last| last.time_row == time_row && last.modality == modality)
    {
        last.rows = last
            .rows
            .checked_add(rows)
            .ok_or_else(|| invalid("run extent overflow"))?;
    } else {
        if runs.len() == ROW_SELECTED_BF16_MAX_RUNS {
            return Err(invalid("request exceeds bounded row-routing capacity"));
        }
        runs.push(Run {
            rows,
            time_row,
            modality,
        });
    }
    Ok(())
}
fn gamma(
    package: &FastH3TransformerPackage,
    block: usize,
    stage: FastH3ModulationStage,
) -> Result<WeightComponentSpec, VNextError> {
    if block >= package.config().layers() {
        return Err(invalid("block outside checkpoint"));
    }
    let weight = package
        .weight(&format!(
            "transformer_blocks.{block}.norm{}.weight",
            stage.norm()
        ))
        .ok_or_else(|| invalid("missing norm weight"))?;
    let FastH3WeightStorage::Dense(c) = &weight.storage else {
        return Err(invalid("norm weight must remain dense"));
    };
    if weight.compute_element_type != ElementType::Bf16 {
        return Err(invalid("norm compute precision differs"));
    }
    Ok(c.clone())
}
pub struct FastH3ModulationSource<'a> {
    package: &'a FastH3TransformerPackage,
    component: WeightComponentSpec,
}
impl<'a> FastH3ModulationSource<'a> {
    pub fn new(
        package: &'a FastH3TransformerPackage,
        c: &FastH3ModulationConfig,
        limits: FastH3ModulationLimits,
    ) -> Result<Self, VNextError> {
        c.validate(limits)?;
        if gamma(package, c.block, c.stage)? != c.gamma
            || package.config().hidden_size() as u64 != c.width
            || CanonicalRational::from_decimal_str(&package.config().norm_eps().to_string())?
                != c.epsilon
        {
            return Err(invalid("modulation source differs from checkpoint"));
        }
        Ok(Self {
            package,
            component: c.gamma.clone(),
        })
    }
}
impl WeightComponentSource for FastH3ModulationSource<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if c != &self.component {
            return Err(invalid("undeclared norm component"));
        }
        self.package.dense_archive().component(c)
    }
}

pub struct FastH3ModulationFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3ModulationLimits,
}
impl FastH3ModulationFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.modulation";
    pub const PROFILE_ID: &'static str = "fast_h3.modulation.block8192_vec8_warp16_fma.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.modulation.input";
    pub const TABLE_ID: &'static str = "value.fast_h3.modulation.table";
    pub const BRANCH_ID: &'static str = "value.fast_h3.modulation.branch";
    pub const HIDDEN_ID: &'static str = "value.fast_h3.modulation.hidden";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.modulation.output";
    pub fn new(limits: FastH3ModulationLimits) -> Result<Self, VNextError> {
        if limits.maximum_rows == 0
            || limits.maximum_time_rows == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.modulation")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3ModulationFamily {
    type Config = FastH3ModulationConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(invalid)? != *raw {
            return Err(invalid("noncanonical modulation configuration"));
        }
        c.validate(self.limits)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        c.validate(self.limits)?;
        Ok(ModelWeights::Stored(c.schema()?))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        c.validate(self.limits)?;
        let (program, specs) = c.program()?;
        let p = NumericalExecutionProfile {
            id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
            family_id: self.id.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id(Self::OUTPUT_ID)?,
            boundaries: program
                .outputs()
                .iter()
                .map(|v| (v.clone(), specs[v].element_type))
                .collect(),
            states: vec![],
            kv_storage: vec![],
            operations: vec![NumericalOperationContract {
                operation_id: id(c.stage.kind().operation_id())?,
                version: ContractVersion::new(1, 0),
                multiplication_type: Some(ElementType::F32),
                accumulation_type: Some(ElementType::F32),
                staged_arithmetic: None,
                composite_arithmetic: None,
            }],
        };
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![p],
            vec![NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown modulation profile"));
        }
        Ok(c.program()?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(reason: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 modulation: {reason}"),
    }
}
