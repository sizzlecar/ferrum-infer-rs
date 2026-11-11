use super::*;
use ferrum_execution::{component::TensorComponentSession, TensorExecutionError};
use std::fmt;
mod actor;
use actor::TextExecution;

/// Separate transaction identities for the two existing reusable sessions.
pub struct PH3ImageTextConditioningRun {
    pub encoder_run: RunId,
    pub encoder_request: RequestIdentity,
    pub projection_run: RunId,
    pub projection_request: RequestIdentity,
}

/// Runtime failures retain their original retirement/ownership object. A
/// caller can inspect/recover it; reducing it to a diagnostic String would
/// discard that interface. No partial conditioning is returned on any error.
#[must_use = "execution failures may retain retirement/resource ownership"]
pub enum PH3ImageTextConditioningError<R: DeviceRuntime> {
    Preparation(anyhow::Error),
    Encoder(TensorExecutionError<R>),
    Projection(TensorExecutionError<R>),
}
impl<R: DeviceRuntime> fmt::Debug for PH3ImageTextConditioningError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preparation(e) => f.debug_tuple("Preparation").field(e).finish(),
            Self::Encoder(e) => f.debug_tuple("Encoder").field(e).finish(),
            Self::Projection(e) => f.debug_tuple("Projection").field(e).finish(),
        }
    }
}

pub struct PH3ImageTextConditioningOutput {
    conditioning: ManagedHostTensor,
    valid_lengths: Vec<u32>,
}
impl PH3ImageTextConditioningOutput {
    pub fn conditioning(&self) -> &ManagedHostTensor {
        &self.conditioning
    }
    /// Per-sample valid prefix, including its sink token. Right-padding rows
    /// are storage only and must not become DiT text rows or valid attention.
    pub fn valid_lengths(&self) -> &[u32] {
        &self.valid_lengths
    }
    pub fn into_parts(self) -> (ManagedHostTensor, Vec<u32>) {
        (self.conditioning, self.valid_lengths)
    }
}

/// Borrows already initialized sessions. The caller supplies their existing
/// runtime, account, static weight/cache policy, and initialization lifecycle.
/// Binding/requests never construct a runtime, reload large weights, or create
/// an independent Host account. Drop releases only these borrows.
pub struct PH3ImageTextConditioning<'a, R: DeviceRuntime> {
    config: &'a PH3ImageTextConditioningConfig,
    connector: PH3ImageTextConnectorConfig,
    checkpoint: &'a PH3ImageTextConnectorCheckpoint<'a>,
    encoder: &'a mut TensorComponentSession<R>,
    projection: &'a mut TensorComponentSession<R>,
    host: HostMemoryAccount,
    precision: PH3ImageConditioningPrecision,
}
impl<'a, R: DeviceRuntime> PH3ImageTextConditioning<'a, R> {
    pub fn bind(
        config: &'a PH3ImageTextConditioningConfig,
        limits: PH3ImageTextConditioningLimits,
        checkpoint: &'a PH3ImageTextConnectorCheckpoint<'a>,
        encoder: &'a mut TensorComponentSession<R>,
        projection: &'a mut TensorComponentSession<R>,
        precision: PH3ImageConditioningPrecision,
    ) -> Result<Self> {
        let family = PH3ImageTextConditioningFamily::new(limits)?;
        let profiles = family.numerical_profiles(config)?;
        let profile =
            NumericalProfileId::new(PH3ImageTextConditioningFamily::cuda_profile_id(config))
                .map_err(anyhow::Error::msg)?;
        let program = family.semantic_program(config, profiles.resolve(&profile)?)?;
        let encoder_plan = encoder.executable().execution_plan().payload();
        ensure!(
            encoder_plan.family_id() == family.family_id()
                && encoder_plan.program_fingerprint() == program.fingerprint()?,
            "encoder session differs from the composed token-to-five-taps program"
        );
        let connector = config.connector_config()?;
        let projection_config = config.projection_config()?;
        ensure!(
            checkpoint.layout() == PH3ImageTextConnectorCheckpointLayout::ComfyMergedTextEncoder
                && checkpoint.projection_config()? == projection_config,
            "conditioning requires this exact merged Comfy connector checkpoint"
        );
        let projection_family =
            PH3ImageTextConnectorProjectionFamily::new(PH3ImageTextConnectorProjectionLimits {
                max_projection_bytes: projection_config.projection_bytes()?,
            })?;
        let projection_profiles = projection_family.numerical_profiles(&projection_config)?;
        let projection_profile =
            NumericalProfileId::new(PH3ImageTextConnectorProjectionFamily::CUDA_PROFILE_ID)
                .map_err(anyhow::Error::msg)?;
        let expected = projection_family.semantic_program(
            &projection_config,
            projection_profiles.resolve(&projection_profile)?,
        )?;
        let projection_plan = projection.executable().execution_plan().payload();
        ensure!(
            projection_plan.family_id() == projection_family.family_id()
                && projection_plan.program_fingerprint() == expected.fingerprint()?
                && projection_plan.device_id() == encoder_plan.device_id(),
            "projection session differs from the Stored F32 connector program/device"
        );
        let rows = config.encoder().text().layout().rows();
        let ports = projection_plan
            .tensor_ports()
            .context("projection lacks tensor ports")?;
        ensure!(
            ports.inputs().len() == 1
                && ports.outputs().len() == 1
                && ports.inputs()[0].value_id().as_str()
                    == PH3ImageTextConnectorProjectionFamily::INPUT_ID
                && ports.inputs()[0].tensor().element_type() == ElementType::F32
                && ports.inputs()[0].tensor().dimensions() == [rows, connector.input_dim()]
                && ports.outputs()[0].tensor().element_type() == ElementType::F32
                && ports.outputs()[0].tensor().dimensions() == [rows, connector.out_channels()],
            "projection ports differ from this batch/sequence connector shape"
        );
        ensure!(
            encoder
                .host_account()
                .same_parent(projection.host_account()),
            "conditioning sessions must share the caller's existing Host parent"
        );
        let host = encoder.host_account().clone();
        Ok(Self {
            config,
            connector,
            checkpoint,
            encoder,
            projection,
            host,
            precision,
        })
    }
    pub fn execute(
        &mut self,
        prompt: &PH3ImageTextPromptBatch,
        run: PH3ImageTextConditioningRun,
    ) -> std::result::Result<PH3ImageTextConditioningOutput, PH3ImageTextConditioningError<R>> {
        self.execute_with(prompt, run, TextExecution::Direct)
    }
    fn execute_with(
        &mut self,
        prompt: &PH3ImageTextPromptBatch,
        run: PH3ImageTextConditioningRun,
        mut execution: TextExecution<'_, R>,
    ) -> std::result::Result<PH3ImageTextConditioningOutput, PH3ImageTextConditioningError<R>> {
        let prepare = || -> Result<_> {
            ensure!(
                run.encoder_run != run.projection_run,
                "encoder and projection need distinct run identities"
            );
            prompt.managed_inputs(self.config, &self.host)
        };
        let inputs = prepare().map_err(PH3ImageTextConditioningError::Preparation)?;
        let outputs = execution
            .execute(
                &mut *self.encoder,
                run.encoder_run,
                run.encoder_request,
                inputs,
            )
            .map_err(PH3ImageTextConditioningError::Encoder)?;
        let standardize = || -> Result<_> {
            let ports = self
                .encoder
                .executable()
                .execution_plan()
                .payload()
                .tensor_ports()
                .context("encoder lacks compiled tensor ports")?;
            let states = self
                .config
                .encoder()
                .text()
                .hidden_state_indices()
                .iter()
                .map(|&index| {
                    let id = Qwen3VLTextComfyFamily::hidden_state_output_id(index)?;
                    let port = ports
                        .outputs()
                        .iter()
                        .find(|p| p.value_id() == &id)
                        .context("missing declared tap")?;
                    let output = outputs
                        .iter()
                        .find(|o| o.value_id() == &id)
                        .context("missing actual tap")?;
                    ensure!(
                        self.host.owns(output.bytes()),
                        "tap readback belongs to another Host parent"
                    );
                    PH3ImageF32HiddenState::from_output(u32::try_from(index)?, port, output)
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                outputs.len() == 5,
                "encoder must read back exactly five taps"
            );
            self.adapter()?
                .managed_standardized_f32_input(&self.host, &states, self.checkpoint)
        };
        let normalized = standardize().map_err(PH3ImageTextConditioningError::Preparation)?;
        // The normalized allocation owns its bytes; original five readbacks
        // are no longer needed during projection upload/execution/readback.
        drop(outputs);
        let mut projected = execution
            .execute(
                &mut *self.projection,
                run.projection_run,
                run.projection_request,
                vec![normalized],
            )
            .map_err(PH3ImageTextConditioningError::Projection)?;
        let finish = || -> Result<_> {
            ensure!(
                projected.len() == 1,
                "connector projection must have exactly one output"
            );
            ensure!(
                self.host.owns(projected[0].bytes()),
                "projection readback belongs to another Host parent"
            );
            self.adapter()?.managed_conditioning(
                &self.host,
                &projected[0],
                self.checkpoint,
                self.precision,
            )
        };
        let conditioning = finish().map_err(PH3ImageTextConditioningError::Preparation)?;
        projected.clear();
        Ok(PH3ImageTextConditioningOutput {
            conditioning,
            valid_lengths: prompt.valid_lengths().to_vec(),
        })
    }
    fn adapter(&self) -> Result<PH3ImageTextConnectorManagedAdapter<'_>> {
        Ok(PH3ImageTextConnectorManagedAdapter::new(
            &self.connector,
            self.config.encoder().text().layout().batch(),
            self.config.encoder().text().layout().sequence_length() as usize,
        )?
        .with_output_affine(super::super::PH3ImageTextConnectorAffineArithmetic::FusedAddcmulF32))
    }
}
