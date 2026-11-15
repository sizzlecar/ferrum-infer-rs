use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    rows: u64,
    time: usize,
}
/// One request evaluation's time routing. Modality tags never index this table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3OutputNormConfig {
    pub(super) times: Vec<f32>,
    rows: u64,
    pub(super) hidden: u64,
    pub(super) epsilon: CanonicalRational,
    pub(super) gamma: WeightComponentSpec,
    runs: Vec<Run>,
}
impl FastH3OutputNormConfig {
    pub fn from_package(
        p: &FastH3TransformerPackage,
        layout: &FastH3PackedLayout,
        schedule: &FastH3ConditioningSchedule,
        evaluation: usize,
        limits: FastH3OutputConditioningLimits,
    ) -> Result<Self, VNextError> {
        let runs = schedule_runs(
            schedule,
            evaluation,
            layout
                .segments()
                .iter()
                .map(|s| (s.kind, s.packed_rows.len() as u64)),
        )?;
        let c = Self {
            times: schedule.timesteps().to_vec(),
            rows: layout.sequence_rows().try_into().map_err(invalid)?,
            hidden: p.config().hidden_size() as u64,
            epsilon: CanonicalRational::from_decimal_str(&p.config().final_norm_eps().to_string())?,
            gamma: dense(p, "norm_out.norm.weight")?,
            runs,
        };
        c.validate(limits)?;
        Ok(c)
    }
    pub fn rows(&self) -> u64 {
        self.rows
    }
    pub fn hidden_size(&self) -> u64 {
        self.hidden
    }
    pub fn time_values(&self) -> &[f32] {
        &self.times
    }
    pub fn parameter_row_runs(&self) -> impl Iterator<Item = (u64, usize)> + '_ {
        self.runs.iter().map(|r| (r.rows, r.time))
    }
    pub(in crate::fast_h3) fn validate_schedule(
        &self,
        schedule: &FastH3ConditioningSchedule,
        evaluation: usize,
        segments: impl IntoIterator<Item = (crate::fast_h3::FastH3SegmentKind, u64)>,
    ) -> Result<(), VNextError> {
        if !same_times(&self.times, schedule.timesteps())
            || self.runs != schedule_runs(schedule, evaluation, segments)?
        {
            return Err(invalid(
                "final normalization routing differs from the sampling evaluation",
            ));
        }
        Ok(())
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(self.program()?.1)
    }
    pub(super) fn validate(
        &self,
        limits: FastH3OutputConditioningLimits,
    ) -> Result<(), VNextError> {
        check_times(&self.times, limits)?;
        if self.rows == 0
            || self.rows > limits.maximum_rows
            || self.hidden == 0
            || self.hidden > HEAD_RMS_NORM_BF16_MAX_HEAD_DIM
            || self.runs.is_empty()
            || self.runs.len() > ROW_SELECTED_BF16_MAX_RUNS
            || self.runs.windows(2).any(|w| w[0].time == w[1].time)
            || self
                .runs
                .iter()
                .any(|r| r.rows == 0 || r.time >= self.times.len())
            || self
                .runs
                .iter()
                .try_fold(0u64, |n, r| n.checked_add(r.rows))
                != Some(self.rows)
        {
            return Err(invalid("invalid final norm geometry or time routing"));
        }
        check_component(&self.gamma, &[self.hidden])?;
        check_capacity(self, std::slice::from_ref(&self.gamma), limits)?;
        self.schema()?;
        let x = resolved(vec![self.rows, self.hidden], ElementType::Bf16)?;
        let g = resolved(vec![self.hidden], ElementType::Bf16)?;
        let table = resolved(
            vec![self.times.len() as u64, 2 * self.hidden],
            ElementType::Bf16,
        )?;
        HeadRmsNormBf16Geometry::new(&x, &g, &x, self.hidden, self.epsilon)?;
        RowSelectedBf16Geometry::new(
            RowSelectedBf16Kind::Modulation,
            &[&x, &table],
            &x,
            &self.routing()?,
        )?;
        for s in self.program()?.1.values() {
            s.byte_len()?;
        }
        Ok(())
    }
    fn routing(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([
            (
                id("run_lengths")?,
                SemanticValue::Integers(
                    self.runs
                        .iter()
                        .map(|r| i64::try_from(r.rows).map_err(invalid))
                        .collect::<Result<_, _>>()?,
                ),
            ),
            (
                id("parameter_rows")?,
                SemanticValue::Integers(
                    self.runs
                        .iter()
                        .map(|r| i64::try_from(r.time).map_err(invalid))
                        .collect::<Result<_, _>>()?,
                ),
            ),
            (id("shift_column")?, SemanticValue::Unsigned(0)),
            (id("scale_column")?, SemanticValue::Unsigned(self.hidden)),
        ]))
    }
    pub(super) fn schema(&self) -> Result<WeightSchema, VNextError> {
        schema("output_norm", std::slice::from_ref(&self.gamma))
    }
    pub(super) fn program(
        &self,
    ) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
        type F = FastH3OutputNormFamily;
        let input: ProgramValueId = id(F::INPUT_ID)?;
        let table: ProgramValueId = id(F::TABLE_ID)?;
        let norm: ProgramValueId = id("value.fast_h3.output_norm.rms")?;
        let output: ProgramValueId = id(F::OUTPUT_ID)?;
        let shape = vec![self.rows, self.hidden];
        let inputs = BTreeMap::from([
            (input.clone(), tensor(shape.clone(), ElementType::Bf16)),
            (
                table.clone(),
                tensor(
                    vec![self.times.len() as u64, 2 * self.hidden],
                    ElementType::Bf16,
                ),
            ),
        ]);
        let mut specs = inputs.clone();
        specs.insert(norm.clone(), tensor(shape.clone(), ElementType::Bf16));
        specs.insert(output.clone(), tensor(shape, ElementType::Bf16));
        let nodes = vec![
            node(
                "output_norm",
                "rms",
                HEAD_RMS_NORM_BF16_OPERATION_ID,
                vec![input.clone(), id("value.fast_h3.output_norm.weight0")?],
                norm.clone(),
                BTreeMap::from([
                    (id("head_dim")?, SemanticValue::Unsigned(self.hidden)),
                    (id("epsilon")?, SemanticValue::Rational(self.epsilon)),
                ]),
            )?,
            node(
                "output_norm",
                "modulate",
                ROW_SELECTED_MODULATION_BF16_OPERATION_ID,
                vec![norm, table.clone()],
                output.clone(),
                self.routing()?,
            )?,
        ];
        Ok((
            ModelProgram::new(
                id(F::FAMILY_ID)?,
                vec![input, table],
                vec![ProgramBlock {
                    id: "block.fast_h3.output_norm".into(),
                    nodes,
                }],
                vec![],
                references("output_norm", std::slice::from_ref(&self.gamma))?,
                vec![output],
            )?
            .with_input_tensor_specs(inputs)?,
            specs,
        ))
    }
}

fn schedule_runs(
    schedule: &FastH3ConditioningSchedule,
    evaluation: usize,
    segments: impl IntoIterator<Item = (crate::fast_h3::FastH3SegmentKind, u64)>,
) -> Result<Vec<Run>, VNextError> {
    let mut runs = Vec::<Run>::new();
    for (kind, rows) in segments {
        if rows == 0 {
            return Err(invalid("empty final conditioning segment"));
        }
        let time = schedule
            .segment_time_index(evaluation, kind)
            .ok_or_else(|| invalid("segment has no declared final conditioning time"))?;
        if let Some(last) = runs.last_mut().filter(|r| r.time == time) {
            last.rows = last
                .rows
                .checked_add(rows)
                .ok_or_else(|| invalid("routing extent overflow"))?;
        } else {
            if runs.len() == ROW_SELECTED_BF16_MAX_RUNS {
                return Err(invalid("final row routing exceeds capacity"));
            }
            runs.push(Run { rows, time });
        }
    }
    Ok(runs)
}
