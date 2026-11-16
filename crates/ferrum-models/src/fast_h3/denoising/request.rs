//! Model stage adaptation over the existing continuation and Host account.
//! The caller executes ordinary Core components and establishes completion.
use super::*;
use ferrum_execution::TensorExecutionOutput;
mod batch;
pub use batch::{FastH3DenoisingBatch, FastH3DenoisingBatchPlan};

pub struct FastH3DenoisingRequest {
    config: FastH3DenoisingConfig,
    schema_fingerprint: String,
    batch_regions: Option<Arc<CheckedVsaRegions>>,
    schedule: FastH3ConditioningSchedule,
    tables: Arc<[FastH3ConditioningTable]>,
    output_table: Arc<FastH3OutputConditioningTable>,
    continuation: FastH3ManagedContinuation,
    plan: Arc<FastH3PddPlan>,
    host: HostMemoryAccount,
    text: ManagedHostTensor,
    rotary: [ManagedHostTensor; 2],
}
impl FastH3DenoisingRequest {
    pub fn projected_text_input(
        config: &FastH3TextRefinerConfig,
        output: &TensorExecutionOutput,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor, VNextError> {
        if output.value_id().as_str() != FastH3InputProjectionFamily::PROJECTED_TEXT_ID
            || output.tensor().element_type() != ElementType::Bf16
            || output.tensor().dimensions() != [config.rows()?, config.hidden_size()]
        {
            return Err(invalid(
                "refiner requires the exact completed context projection",
            ));
        }
        copy_output(host, output, id(FastH3TextRefinerFamily::INPUT_ID)?)
    }
    /// The original live refiner result is copied once into its own admitted
    /// input owner. Unit noise and encoded reference prefixes are explicit
    /// caller inputs; this adapter never generates or substitutes them.
    pub fn initialize(
        config: FastH3DenoisingConfig,
        limits: FastH3DenoisingLimits,
        request: RequestIdentity,
        plan: Arc<FastH3PddPlan>,
        layout: &FastH3PackedLayout,
        host: HostMemoryAccount,
        video_noise: &ManagedHostTensor,
        audio_noise: &ManagedHostTensor,
        refined_text: &TensorExecutionOutput,
        rotary: [ManagedHostTensor; 2],
        tables: Arc<[FastH3ConditioningTable]>,
        output_table: Arc<FastH3OutputConditioningTable>,
    ) -> anyhow::Result<Self> {
        let built = compose(&config, limits)?;
        Self::initialize_checked(
            config,
            built.schema.fingerprint()?,
            &built.options,
            request,
            plan,
            layout,
            host,
            video_noise,
            audio_noise,
            refined_text,
            rotary,
            tables,
            output_table,
        )
    }
    /// Reuse the actual cold B1 plan's validated geometry and source contract.
    /// Request-local payloads and schedule are still checked below; no graph is
    /// recomposed merely because another prompt shares this declared layout.
    #[allow(clippy::too_many_arguments)]
    pub fn initialize_from_plan(
        prepared: &FastH3DenoisingBatchPlan,
        request: RequestIdentity,
        plan: Arc<FastH3PddPlan>,
        layout: &FastH3PackedLayout,
        host: HostMemoryAccount,
        video_noise: &ManagedHostTensor,
        audio_noise: &ManagedHostTensor,
        refined_text: &TensorExecutionOutput,
        rotary: [ManagedHostTensor; 2],
        tables: Arc<[FastH3ConditioningTable]>,
        output_table: Arc<FastH3OutputConditioningTable>,
    ) -> anyhow::Result<Self> {
        Self::initialize_checked(
            prepared.config.clone(),
            prepared.schema_fingerprint.clone(),
            &prepared.options,
            request,
            plan,
            layout,
            host,
            video_noise,
            audio_noise,
            refined_text,
            rotary,
            tables,
            output_table,
        )?
        .with_batch_regions(prepared.regions.clone())
        .map_err(Into::into)
    }
    #[allow(clippy::too_many_arguments)]
    fn initialize_checked(
        config: FastH3DenoisingConfig,
        schema_fingerprint: String,
        options: &ProgramPlanCompileOptions,
        request: RequestIdentity,
        plan: Arc<FastH3PddPlan>,
        layout: &FastH3PackedLayout,
        host: HostMemoryAccount,
        video_noise: &ManagedHostTensor,
        audio_noise: &ManagedHostTensor,
        refined_text: &TensorExecutionOutput,
        rotary: [ManagedHostTensor; 2],
        tables: Arc<[FastH3ConditioningTable]>,
        output_table: Arc<FastH3OutputConditioningTable>,
    ) -> anyhow::Result<Self> {
        if config.backbone.batches() != 1 {
            return Err(invalid(
                "a continuation requires its independent single-request configuration",
            )
            .into());
        }
        let expected = &options.tensor_specs()[&id(FastH3DenoisingFamily::TEXT_INPUT_ID)?];
        if refined_text.value_id().as_str() != FastH3TextRefinerFamily::OUTPUT_ID
            || refined_text.tensor().element_type() != expected.element_type
            || refined_text.tensor().dimensions() != expected.dimensions
            || refined_text.tensor().layout() != &expected.layout
        {
            return Err(invalid("requires exact live refined-text output").into());
        }
        let text = copy_output(
            &host,
            refined_text,
            id(FastH3DenoisingFamily::TEXT_INPUT_ID)?,
        )?;
        for (input, name) in rotary.iter().zip([
            FastH3DenoisingFamily::COSINE_ID,
            FastH3DenoisingFamily::SINE_ID,
        ]) {
            let expected = &options.tensor_specs()[&id(name)?];
            if input.value_id().as_str() != name
                || !host.owns(input.bytes())
                || input.tensor().dimensions() != expected.dimensions
                || input.tensor().element_type() != expected.element_type
                || input.tensor().layout() != &expected.layout
            {
                return Err(invalid("RoPE input geometry or ownership differs").into());
            }
        }
        let schedule = FastH3ConditioningSchedule::all_modalities(&plan)?;
        // Validate every live table and current schedule before starting solver
        // state; these temporary role inputs drop immediately after admission.
        drop(
            config
                .backbone
                .managed_role_tables(&tables, &schedule, &[0], &host)?,
        );
        if layout.sequence_rows() as u64 != config.backbone.packed_rows() {
            return Err(invalid("continuation layout differs from denoising plan").into());
        }
        let continuation = FastH3ManagedContinuation::initialize(
            request,
            plan.clone(),
            layout,
            host.clone(),
            video_noise,
            audio_noise,
        )?;
        for (sample, name) in [
            (continuation.video(), FastH3DenoisingFamily::VIDEO_INPUT_ID),
            (continuation.audio(), FastH3DenoisingFamily::AUDIO_INPUT_ID),
        ] {
            let expected = &options.tensor_specs()[&id(name)?];
            if sample.tensor().dimensions() != expected.dimensions
                || sample.tensor().element_type() != expected.element_type
                || sample.tensor().layout() != &expected.layout
            {
                return Err(invalid("initial media shape differs from denoising plan").into());
            }
        }
        Ok(Self {
            config,
            schema_fingerprint,
            batch_regions: None,
            schedule,
            tables,
            output_table,
            continuation,
            plan,
            host,
            text,
            rotary,
        })
    }
    pub fn request(&self) -> &RequestIdentity {
        self.continuation.request()
    }
    /// Retain the original checked native geometry used by this request's
    /// source/factory. This declares batch eligibility without assigning a
    /// permanent slot, copying metadata payloads or changing solver progress.
    pub fn with_batch_regions(
        mut self,
        regions: Arc<CheckedVsaRegions>,
    ) -> Result<Self, VNextError> {
        if self.batch_regions.is_some()
            || regions.digest() != self.config.backbone.region_digest()
            || regions.shape().batches() != 1
        {
            return Err(invalid(
                "batch regions differ from this request's original configuration",
            ));
        }
        self.batch_regions = Some(regions);
        Ok(self)
    }
    pub fn evaluation(&self) -> usize {
        self.continuation.step_index()
    }
    pub fn is_complete(&self) -> bool {
        self.continuation.is_complete()
    }
    pub fn remaining_evaluations(&self) -> usize {
        self.plan.steps().len().saturating_sub(self.evaluation())
    }
    pub fn video(&self) -> &ManagedHostTensor {
        self.continuation.video()
    }
    pub fn audio(&self) -> &ManagedHostTensor {
        self.continuation.audio()
    }
    /// Inputs for the same retained denoising plan at its current solver frontier.
    pub fn denoising_inputs(&self) -> Result<Vec<ManagedHostTensor>, VNextError> {
        if self.is_complete() {
            return Err(invalid("solver already complete"));
        }
        let mut inputs = self.config.backbone.managed_role_tables(
            &self.tables,
            &self.schedule,
            &[self.evaluation()],
            &self.host,
        )?;
        for (sample, name) in [
            (self.video(), FastH3DenoisingFamily::VIDEO_INPUT_ID),
            (self.audio(), FastH3DenoisingFamily::AUDIO_INPUT_ID),
        ] {
            inputs.push(ManagedHostTensor::new(
                id(name)?,
                sample.tensor().clone(),
                sample.bytes().clone(),
            )?);
        }
        inputs.push(self.text.clone());
        inputs.extend(self.rotary.iter().cloned());
        Ok(inputs)
    }
    /// Admit the completed backbone result for this interval's original PDD
    /// output graph. Completion and request binding remain the execution
    /// owner's responsibility; labels do not authenticate cached outputs.
    pub fn final_output_inputs(
        &self,
        config: &FastH3FinalOutputConfig,
        output: &TensorExecutionOutput,
    ) -> Result<Vec<ManagedHostTensor>, VNextError> {
        self.final_output_rows(config, output, 1, 0)
    }
    fn final_output_rows(
        &self,
        config: &FastH3FinalOutputConfig,
        output: &TensorExecutionOutput,
        batches: u64,
        slot: usize,
    ) -> Result<Vec<ManagedHostTensor>, VNextError> {
        self.validate_final_output_config(config)?;
        if config.input_residency() != FastH3PredictionInput::Host
            || output.value_id().as_str() != FastH3DenoisingFamily::OUTPUT_ID
            || output.tensor().dimensions()
                != [
                    self.config
                        .backbone
                        .packed_rows()
                        .checked_mul(batches)
                        .ok_or_else(|| invalid("batch output rows overflow"))?,
                    self.config.backbone.hidden_size(),
                ]
            || output.tensor().element_type() != ElementType::Bf16
            || slot as u64 >= batches
        {
            return Err(invalid("output stage or backbone result differs"));
        }
        Ok(vec![
            view_output_rows(
                &self.host,
                output,
                id(FastH3FinalOutputFamily::INPUT_ID)?,
                slot as u64 * self.config.backbone.packed_rows(),
                self.config.backbone.packed_rows(),
            )?,
            self.output_table
                .modulation_input(config.normalization(), &self.host)?,
        ])
    }

    fn final_output_state_inputs(
        &self,
        config: &FastH3FinalOutputConfig,
    ) -> Result<Vec<ManagedHostTensor>, VNextError> {
        self.validate_final_output_config(config)?;
        if config.input_residency() != FastH3PredictionInput::RequestState {
            return Err(invalid("prediction does not declare device request state"));
        }
        Ok(vec![self
            .output_table
            .modulation_input(config.normalization(), &self.host)?])
    }

    fn validate_final_output_config(
        &self,
        config: &FastH3FinalOutputConfig,
    ) -> Result<(), VNextError> {
        config
            .projection(FastH3OutputModality::Video)
            .validate_schedule(&self.plan)?;
        config
            .projection(FastH3OutputModality::Audio)
            .validate_schedule(&self.plan)?;
        if self.is_complete()
            || config.projection(FastH3OutputModality::Video).evaluation() != self.evaluation()
            || config.projection(FastH3OutputModality::Audio).evaluation() != self.evaluation()
            || config.normalization().rows() != self.config.backbone.packed_rows()
            || config.normalization().hidden_size() != self.config.backbone.hidden_size()
        {
            return Err(invalid(
                "output stage, evaluation or backbone result differs",
            ));
        }
        Ok(())
    }
    /// Both target streams are prepared before either is committed. Frozen
    /// reference prefixes and the original per-modality sigma grids are kept.
    pub fn advance(&mut self, outputs: &[TensorExecutionOutput]) -> anyhow::Result<()> {
        self.prepare_advance(outputs)?.commit();
        Ok(())
    }
    /// Does not publish a frontier. A batch prepares every member before any
    /// commit; dropping a guard rolls back only the uncommitted successor.
    pub fn prepare_advance(
        &mut self,
        outputs: &[TensorExecutionOutput],
    ) -> anyhow::Result<PreparedFastH3Advance<'_>> {
        if outputs.len() != 2 {
            return Err(invalid("PDD requires video and audio predictions").into());
        }
        let get = |name| {
            outputs
                .iter()
                .find(|o| o.value_id().as_str() == name)
                .ok_or_else(|| invalid("missing PDD output identity"))
        };
        let video = get(FastH3FinalOutputFamily::VIDEO_OUTPUT_ID)?;
        let audio = get(FastH3FinalOutputFamily::AUDIO_OUTPUT_ID)?;
        let request = self.request().clone();
        let prediction = FastH3PredictionView::new(
            &request,
            self.evaluation(),
            FastH3VelocityInput::new(video.tensor(), video.bytes()),
            FastH3VelocityInput::new(audio.tensor(), audio.bytes()),
        );
        self.continuation.prepare_advance(prediction)
    }
}
fn view_output_rows(
    host: &HostMemoryAccount,
    output: &TensorExecutionOutput,
    id: ProgramValueId,
    start: u64,
    rows: u64,
) -> Result<ManagedHostTensor, VNextError> {
    let dimensions = output.tensor().dimensions();
    if dimensions.len() != 2
        || rows == 0
        || start
            .checked_add(rows)
            .is_none_or(|end| end > dimensions[0])
        || !host.owns(output.bytes())
        || output.bytes().class() != HostMemoryClass::Output
        || output.tensor().layout() != &ResolvedTensorLayout::Contiguous
        || output.tensor().minimum_storage_bytes()? != output.bytes().len() as u64
    {
        return Err(invalid(
            "completed output row partition or ownership differs",
        ));
    }
    let spec = ResolvedTensorSpec::new(
        vec![rows, dimensions[1]],
        output.tensor().element_type(),
        ResolvedTensorLayout::Contiguous,
    )?;
    let stride = output.bytes().len() / usize::try_from(dimensions[0]).map_err(invalid)?;
    let begin = usize::try_from(start)
        .map_err(invalid)?
        .checked_mul(stride)
        .ok_or_else(|| invalid("output offset overflow"))?;
    let bytes = usize::try_from(spec.minimum_storage_bytes()?).map_err(invalid)?;
    let retained = output
        .bytes()
        .slice(begin..begin + bytes)
        .map_err(invalid)?;
    ManagedHostTensor::from_retained_output(id, spec, retained)
}
/// A completed Host output crosses a deliberate component boundary through
/// an admitted InputPreprocess copy; its original allocation is not relabelled.
pub(super) fn copy_output(
    host: &HostMemoryAccount,
    output: &TensorExecutionOutput,
    id: ProgramValueId,
) -> Result<ManagedHostTensor, VNextError> {
    if !host.owns(output.bytes())
        || output.bytes().class() != HostMemoryClass::Output
        || output.tensor().layout() != &ResolvedTensorLayout::Contiguous
        || output.tensor().minimum_storage_bytes()? != output.bytes().len() as u64
    {
        return Err(invalid("completed output has foreign ownership or storage"));
    }
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, output.bytes().len())
        .map_err(invalid)?;
    bytes.as_mut().copy_from_slice(output.bytes().bytes());
    ManagedHostTensor::new(id, output.tensor().clone(), bytes.freeze())
}
