use super::*;

/// Request-independent projection of the immutable schedule's embeddings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3OutputTimeConfig {
    pub(super) times: Vec<f32>,
    pub(super) hidden: u64,
    pub(super) embedding: u64,
    pub(super) components: [WeightComponentSpec; 2],
}
pub(super) fn components(
    p: &FastH3TransformerPackage,
) -> Result<[WeightComponentSpec; 2], VNextError> {
    Ok([
        dense(p, "norm_out.linear.weight")?,
        dense(p, "norm_out.linear.bias")?,
    ])
}
impl FastH3OutputTimeConfig {
    pub fn from_package(
        p: &FastH3TransformerPackage,
        schedule: &FastH3ConditioningSchedule,
        limits: FastH3OutputConditioningLimits,
    ) -> Result<Self, VNextError> {
        if p.config().adaln_rank().is_some() {
            return Err(invalid(
                "output time projection requires full-rank SiLU path",
            ));
        }
        let c = Self {
            times: schedule.timesteps().to_vec(),
            hidden: p.config().hidden_size() as u64,
            embedding: p.config().time_embed_dim() as u64,
            components: components(p)?,
        };
        c.validate(limits)?;
        Ok(c)
    }
    pub fn time_values(&self) -> &[f32] {
        &self.times
    }
    pub fn hidden_size(&self) -> u64 {
        self.hidden
    }
    pub fn embedding_dim(&self) -> u64 {
        self.embedding
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(self.program()?.1)
    }
    /// Admit an independent input copy of the existing live time embedding.
    /// Original Output owners remain held by the table, including after close.
    /// Config/schedule binding is an adapter assertion, not file attestation.
    pub fn embedding_input(
        &self,
        table: &FastH3ConditioningTable,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor, VNextError> {
        if table.configuration().embedding_dim() != self.embedding
            || table.configuration().hidden_size() != self.hidden
            || !same_times(table.timesteps(), &self.times)
        {
            return Err(invalid("embedding width or exact schedule rows differ"));
        }
        managed_copy(
            host,
            FastH3OutputTimeFamily::INPUT_ID,
            vec![self.times.len() as u64, self.embedding],
            ElementType::F32,
            table.outputs()[0].bytes().bytes(),
        )
    }
    pub(super) fn validate(
        &self,
        limits: FastH3OutputConditioningLimits,
    ) -> Result<(), VNextError> {
        check_times(&self.times, limits)?;
        if self.hidden == 0
            || self.embedding == 0
            || self.hidden > i32::MAX as u64 / 2
            || self.embedding > i32::MAX as u64
            || self.times.len() > i32::MAX as usize
        {
            return Err(invalid("output time geometry exceeds affine bounds"));
        }
        check_component(&self.components[0], &[2 * self.hidden, self.embedding])?;
        check_component(&self.components[1], &[2 * self.hidden])?;
        check_capacity(self, &self.components, limits)?;
        self.schema()?;
        for s in self.program()?.1.values() {
            s.byte_len()?;
        }
        Ok(())
    }
    pub(super) fn schema(&self) -> Result<WeightSchema, VNextError> {
        schema("output_time", &self.components)
    }
    pub(super) fn program(
        &self,
    ) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
        type F = FastH3OutputTimeFamily;
        let input: ProgramValueId = id(F::INPUT_ID)?;
        let activated: ProgramValueId = id("value.fast_h3.output_time.activated")?;
        let cast: ProgramValueId = id("value.fast_h3.output_time.bf16")?;
        let output: ProgramValueId = id(F::OUTPUT_ID)?;
        let rows = self.times.len() as u64;
        let inputs = BTreeMap::from([(
            input.clone(),
            tensor(vec![rows, self.embedding], ElementType::F32),
        )]);
        let mut specs = inputs.clone();
        specs.insert(
            activated.clone(),
            tensor(vec![rows, self.embedding], ElementType::F32),
        );
        specs.insert(
            cast.clone(),
            tensor(vec![rows, self.embedding], ElementType::Bf16),
        );
        specs.insert(
            output.clone(),
            tensor(vec![rows, 2 * self.hidden], ElementType::Bf16),
        );
        let nodes = vec![
            node(
                "output_time",
                "activate",
                SILU_F32_OPERATION_ID,
                vec![input.clone()],
                activated.clone(),
                BTreeMap::new(),
            )?,
            node(
                "output_time",
                "cast",
                CAST_F32_TO_BF16_OPERATION_ID,
                vec![activated],
                cast.clone(),
                BTreeMap::new(),
            )?,
            node(
                "output_time",
                "project",
                AFFINE_LINEAR_BF16_OPERATION_ID,
                vec![
                    cast,
                    id("value.fast_h3.output_time.weight0")?,
                    id("value.fast_h3.output_time.weight1")?,
                ],
                output.clone(),
                BTreeMap::from([
                    (id("in_features")?, SemanticValue::Unsigned(self.embedding)),
                    (
                        id("out_features")?,
                        SemanticValue::Unsigned(2 * self.hidden),
                    ),
                ]),
            )?,
        ];
        Ok((
            ModelProgram::new(
                id(F::FAMILY_ID)?,
                vec![input],
                vec![ProgramBlock {
                    id: "block.fast_h3.output_time".into(),
                    nodes,
                }],
                vec![],
                references("output_time", &self.components)?,
                vec![output],
            )?
            .with_input_tensor_specs(inputs)?,
            specs,
        ))
    }
}
