//! Bounded Host adapters around the actual F32 connector projection component.
//! All large buffers enter the caller's existing InputPreprocess account;
//! checkpoint tensors and encoder/projection outputs remain borrowed from
//! their original immutable owners. No runtime or resource account is created.

use anyhow::{ensure, Context, Result};
use ferrum_execution::TensorExecutionOutput;
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::{
    CompiledTensorPort, ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostTensor,
    ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
};
use half::bf16;

use super::text_connector::PH3ImageTextConnectorAffineArithmetic;
use super::text_connector_projection::PH3ImageTextConnectorCheckpointLayout;
use super::{
    PH3ImageTextConnectorConfig, PH3ImageTextConnectorProjectionConfig,
    PH3ImageTextConnectorProjectionFamily as Projection, PH3ImageTextConnectorProjectionLimits,
    PH3ImageTextConnectorShape,
};

/// Final boundary only: normalization, matrix multiplication and output affine
/// arithmetic remain FP32 in either mode. The official conditioning adapter
/// converts the completed connector result to BF16 before the DiT entrance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PH3ImageConditioningPrecision {
    F32,
    Bf16,
}
impl PH3ImageConditioningPrecision {
    fn dtype(self) -> ElementType {
        match self {
            Self::F32 => ElementType::F32,
            Self::Bf16 => ElementType::Bf16,
        }
    }
    fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// Declared connector checkpoint schema, with exact F32 dtype/shape and
/// finite parameter validation. Every returned payload borrows the archive;
/// its lifetime prevents the source mmap/provenance from disappearing while
/// adapters or component initialization still use these views.
pub struct PH3ImageTextConnectorCheckpoint<'a> {
    config: &'a PH3ImageTextConnectorConfig,
    archive: &'a SafetensorsArchive,
    layout: PH3ImageTextConnectorCheckpointLayout,
    input_mean: &'a [u8],
    input_std: &'a [u8],
    output_mean: &'a [u8],
    output_std: &'a [u8],
    sink: &'a [u8],
}
impl<'a> PH3ImageTextConnectorCheckpoint<'a> {
    pub fn new(
        config: &'a PH3ImageTextConnectorConfig,
        archive: &'a SafetensorsArchive,
        limits: PH3ImageTextConnectorProjectionLimits,
    ) -> Result<Self> {
        Self::new_with_layout(
            config,
            archive,
            limits,
            PH3ImageTextConnectorCheckpointLayout::default(),
        )
    }
    pub fn new_with_layout(
        config: &'a PH3ImageTextConnectorConfig,
        archive: &'a SafetensorsArchive,
        limits: PH3ImageTextConnectorProjectionLimits,
        layout: PH3ImageTextConnectorCheckpointLayout,
    ) -> Result<Self> {
        let projection_config = PH3ImageTextConnectorProjectionConfig::from_connector(config)?;
        ensure!(
            limits.max_projection_bytes > 0
                && projection_config.projection_bytes()? <= limits.max_projection_bytes,
            "connector projection exceeds declared checkpoint capacity"
        );
        let input = [config.input_dim()];
        let output = [config.out_channels()];
        let names = layout.names();
        // Validate even projection parameters used only by subsequently
        // replaced sink rows. No copied/dequantized weight buffer is made.
        f32_tensor(
            archive,
            names.projection,
            &[config.out_channels(), config.input_dim()],
        )?;
        let checkpoint = Self {
            config,
            archive,
            layout,
            input_mean: f32_tensor(archive, names.input_mean, &input)?,
            input_std: f32_tensor(archive, names.input_std, &input)?,
            output_mean: f32_tensor(archive, names.output_mean, &output)?,
            output_std: f32_tensor(archive, names.output_std, &output)?,
            sink: f32_tensor(archive, names.sink, &output)?,
        };
        ensure!(
            checkpoint
                .input_std
                .chunks_exact(4)
                .chain(checkpoint.output_std.chunks_exact(4))
                .all(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()) > 0.0),
            "connector checkpoint standard deviations must be positive"
        );
        Ok(checkpoint)
    }
    pub fn archive(&self) -> &'a SafetensorsArchive {
        self.archive
    }
    pub fn layout(&self) -> PH3ImageTextConnectorCheckpointLayout {
        self.layout
    }
    pub fn projection_config(&self) -> Result<PH3ImageTextConnectorProjectionConfig> {
        Ok(
            PH3ImageTextConnectorProjectionConfig::from_connector(self.config)?
                .with_checkpoint_layout(self.layout),
        )
    }
}

/// Borrowed BF16 hidden-state tensor in exact contiguous `[B*S,H]` or
/// `[B,S,H]` storage. No intermediate F32 Vec is created.
#[derive(Clone, Copy)]
pub struct PH3ImageBf16HiddenState<'a> {
    index: u32,
    spec: &'a ResolvedTensorSpec,
    bytes: &'a [u8],
}
impl<'a> PH3ImageBf16HiddenState<'a> {
    pub fn new(index: u32, spec: &'a ResolvedTensorSpec, bytes: &'a [u8]) -> Self {
        Self { index, spec, bytes }
    }
    pub fn from_output(index: u32, output: &'a TensorExecutionOutput) -> Result<Self> {
        let expected = crate::qwen3_vl_text::Qwen3VLTextEncoderFamily::hidden_state_output_id(
            u64::from(index),
        )?;
        ensure!(
            output.value_id() == &expected,
            "connector hidden-state index differs from declared encoder output port"
        );
        validate_hidden_storage(output.tensor(), output.bytes().bytes(), ElementType::Bf16)?;
        Ok(Self::new(index, output.tensor(), output.bytes().bytes()))
    }
}

/// Native F32 encoder activation, borrowed without a BF16 round trip. The
/// caller retains the original readback and its full physical Host charge.
#[derive(Clone, Copy)]
pub struct PH3ImageF32HiddenState<'a> {
    index: u32,
    spec: &'a ResolvedTensorSpec,
    bytes: &'a [u8],
}
impl<'a> PH3ImageF32HiddenState<'a> {
    pub fn new(index: u32, spec: &'a ResolvedTensorSpec, bytes: &'a [u8]) -> Self {
        Self { index, spec, bytes }
    }
    /// Bind a tap to its compiler-declared output, including its exact shape.
    /// F32 encoder families may declare their own logical ports; neither an
    /// unverified name nor an assumed BF16 encoder port substitutes for this.
    pub fn from_output(
        index: u32,
        declared: &CompiledTensorPort,
        output: &'a TensorExecutionOutput,
    ) -> Result<Self> {
        ensure!(
            output.value_id() == declared.value_id() && output.tensor() == declared.tensor(),
            "connector F32 hidden state differs from its declared output port or shape"
        );
        validate_hidden_storage(output.tensor(), output.bytes().bytes(), ElementType::F32)?;
        Ok(Self::new(index, output.tensor(), output.bytes().bytes()))
    }
}

trait HiddenStateView {
    const DTYPE: ElementType;
    const ELEMENT_BYTES: usize;
    fn index(&self) -> u32;
    fn spec(&self) -> &ResolvedTensorSpec;
    fn bytes(&self) -> &[u8];
    fn value(&self, index: usize) -> f32;
}
impl HiddenStateView for PH3ImageBf16HiddenState<'_> {
    const DTYPE: ElementType = ElementType::Bf16;
    const ELEMENT_BYTES: usize = 2;
    fn index(&self) -> u32 {
        self.index
    }
    fn spec(&self) -> &ResolvedTensorSpec {
        self.spec
    }
    fn bytes(&self) -> &[u8] {
        self.bytes
    }
    fn value(&self, index: usize) -> f32 {
        bf16::from_bits(u16::from_le_bytes(
            self.bytes[index * 2..index * 2 + 2].try_into().unwrap(),
        ))
        .to_f32()
    }
}
impl HiddenStateView for PH3ImageF32HiddenState<'_> {
    const DTYPE: ElementType = ElementType::F32;
    const ELEMENT_BYTES: usize = 4;
    fn index(&self) -> u32 {
        self.index
    }
    fn spec(&self) -> &ResolvedTensorSpec {
        self.spec
    }
    fn bytes(&self) -> &[u8] {
        self.bytes
    }
    fn value(&self, index: usize) -> f32 {
        read_f32(self.bytes, index)
    }
}

pub struct PH3ImageTextConnectorManagedAdapter<'a> {
    config: &'a PH3ImageTextConnectorConfig,
    shape: PH3ImageTextConnectorShape,
    output_affine: PH3ImageTextConnectorAffineArithmetic,
}
impl<'a> PH3ImageTextConnectorManagedAdapter<'a> {
    pub const CONDITIONING_ID: &'static str = "value.p_h3_image.text_connector.conditioning";
    pub fn new(
        config: &'a PH3ImageTextConnectorConfig,
        batch: usize,
        tokens: usize,
    ) -> Result<Self> {
        Ok(Self {
            config,
            shape: PH3ImageTextConnectorShape::new(config, batch, tokens)?,
            output_affine: PH3ImageTextConnectorAffineArithmetic::default(),
        })
    }
    /// Diffusers' separate F32 multiply/add remains the default. Select
    /// FusedAddcmulF32 explicitly for Comfy's output destandardization; this
    /// choice is independent of the final F32/BF16 conditioning boundary.
    pub fn with_output_affine(mut self, arithmetic: PH3ImageTextConnectorAffineArithmetic) -> Self {
        self.output_affine = arithmetic;
        self
    }
    pub fn output_affine(&self) -> PH3ImageTextConnectorAffineArithmetic {
        self.output_affine
    }
    pub fn shape(&self) -> PH3ImageTextConnectorShape {
        self.shape
    }

    /// Check all original hidden values and parameters, then zero each sink
    /// row before projection. Its discarded intermediate arithmetic must not
    /// make an otherwise valid final connector fail. Other rows logically
    /// concatenate taps and standardize directly into charged F32 storage.
    pub fn managed_standardized_input(
        &self,
        host: &HostMemoryAccount,
        states: &[PH3ImageBf16HiddenState<'_>],
        checkpoint: &PH3ImageTextConnectorCheckpoint<'_>,
    ) -> Result<ManagedHostTensor> {
        self.standardized_input(host, states, checkpoint)
    }

    /// F32Native route: concatenate and standardize the original F32 tap
    /// values directly. BF16Promote callers keep using the existing method.
    pub fn managed_standardized_f32_input(
        &self,
        host: &HostMemoryAccount,
        states: &[PH3ImageF32HiddenState<'_>],
        checkpoint: &PH3ImageTextConnectorCheckpoint<'_>,
    ) -> Result<ManagedHostTensor> {
        self.standardized_input(host, states, checkpoint)
    }

    fn standardized_input<S: HiddenStateView>(
        &self,
        host: &HostMemoryAccount,
        states: &[S],
        checkpoint: &PH3ImageTextConnectorCheckpoint<'_>,
    ) -> Result<ManagedHostTensor> {
        self.check_checkpoint(checkpoint)?;
        self.validate_states(states)?;
        let rows = self.rows();
        for row in 0..rows {
            if row % self.shape.tokens() == 0 {
                continue;
            }
            for column in 0..self.shape.input_dim() {
                ensure!(
                    self.normalized(states, checkpoint, row, column).is_finite(),
                    "connector standardized input is not finite"
                );
            }
        }
        let elements = rows
            .checked_mul(self.shape.input_dim())
            .context("connector standardized element count overflows")?;
        let bytes = elements
            .checked_mul(4)
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .context("connector standardized bytes exceed address space")?;
        let mut allocation = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        for (index, destination) in allocation.as_mut().chunks_exact_mut(4).enumerate() {
            let row = index / self.shape.input_dim();
            let value = if row % self.shape.tokens() == 0 {
                0.0
            } else {
                self.normalized(states, checkpoint, row, index % self.shape.input_dim())
            };
            destination.copy_from_slice(&value.to_le_bytes());
        }
        let config = checkpoint.projection_config()?;
        let spec = config.input_spec(u64::try_from(rows)?)?;
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(Projection::INPUT_ID)?,
            ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout)?,
            allocation.freeze(),
        )?)
    }

    pub fn managed_conditioning(
        &self,
        host: &HostMemoryAccount,
        projection: &TensorExecutionOutput,
        checkpoint: &PH3ImageTextConnectorCheckpoint<'_>,
        precision: PH3ImageConditioningPrecision,
    ) -> Result<ManagedHostTensor> {
        ensure!(
            projection.value_id().as_str() == Projection::OUTPUT_ID,
            "connector projection output port differs"
        );
        self.managed_conditioning_from_f32(
            host,
            projection.tensor(),
            projection.bytes().bytes(),
            checkpoint,
            precision,
        )
    }

    /// Raw immutable view variant for mathematical/reference callers. The
    /// actual component entry above additionally checks its declared output
    /// port. Source owners retain their original charges throughout this call.
    pub fn managed_conditioning_from_f32(
        &self,
        host: &HostMemoryAccount,
        projection_spec: &ResolvedTensorSpec,
        projection_bytes: &[u8],
        checkpoint: &PH3ImageTextConnectorCheckpoint<'_>,
        precision: PH3ImageConditioningPrecision,
    ) -> Result<ManagedHostTensor> {
        self.check_checkpoint(checkpoint)?;
        let rows = self.rows();
        ensure!(
            projection_spec.element_type() == ElementType::F32
                && projection_spec.layout() == &ResolvedTensorLayout::Contiguous
                && projection_spec.dimensions() == [rows as u64, self.shape.output_dim() as u64]
                && projection_bytes.len() == self.shape.output_elements() * 4,
            "connector projection must be exact contiguous F32 [B*S,O]"
        );
        for index in 0..self.shape.output_elements() {
            let value = self.finished(checkpoint, projection_bytes, index);
            ensure!(
                value.is_finite()
                    && (precision != PH3ImageConditioningPrecision::Bf16
                        || bf16::from_f32(value).is_finite()),
                "connector final conditioning is not finite in its declared precision"
            );
        }
        let bytes = self
            .shape
            .output_elements()
            .checked_mul(precision.bytes())
            .context("connector conditioning byte count overflows")?;
        let mut allocation = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        for (index, destination) in allocation
            .as_mut()
            .chunks_exact_mut(precision.bytes())
            .enumerate()
        {
            let value = self.finished(checkpoint, projection_bytes, index);
            match precision {
                PH3ImageConditioningPrecision::F32 => {
                    destination.copy_from_slice(&value.to_le_bytes())
                }
                PH3ImageConditioningPrecision::Bf16 => {
                    destination.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
                }
            }
        }
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(Self::CONDITIONING_ID)?,
            ResolvedTensorSpec::new(
                vec![rows as u64, self.shape.output_dim() as u64],
                precision.dtype(),
                ResolvedTensorLayout::Contiguous,
            )?,
            allocation.freeze(),
        )?)
    }

    fn check_checkpoint(&self, checkpoint: &PH3ImageTextConnectorCheckpoint<'_>) -> Result<()> {
        ensure!(
            checkpoint.config == self.config,
            "connector checkpoint and adapter configurations differ"
        );
        Ok(())
    }
    fn rows(&self) -> usize {
        self.shape.batch() * self.shape.tokens()
    }
    fn validate_states<S: HiddenStateView>(&self, states: &[S]) -> Result<()> {
        ensure!(
            states.len() == self.shape.taps(),
            "connector selected hidden-state count differs"
        );
        for (state, &index) in states.iter().zip(self.config.hidden_state_indices()) {
            let matrix = [self.rows() as u64, self.shape.hidden_size() as u64];
            let batched = [
                self.shape.batch() as u64,
                self.shape.tokens() as u64,
                self.shape.hidden_size() as u64,
            ];
            ensure!(
                state.index() == index
                    && state.spec().element_type() == S::DTYPE
                    && state.spec().layout() == &ResolvedTensorLayout::Contiguous
                    && (state.spec().dimensions() == matrix
                        || state.spec().dimensions() == batched)
                    && state.bytes().len() == self.shape.state_elements() * S::ELEMENT_BYTES,
                "connector hidden-state index/dtype/shape differs"
            );
            ensure!(
                (0..self.shape.state_elements()).all(|element| state.value(element).is_finite()),
                "connector original hidden states must be finite, including sink rows"
            );
        }
        Ok(())
    }
    fn normalized<S: HiddenStateView>(
        &self,
        states: &[S],
        checkpoint: &PH3ImageTextConnectorCheckpoint<'_>,
        row: usize,
        column: usize,
    ) -> f32 {
        let tap = column / self.shape.hidden_size();
        let hidden = column % self.shape.hidden_size();
        let value = states[tap].value(row * self.shape.hidden_size() + hidden);
        let centered = value - read_f32(checkpoint.input_mean, column);
        centered / read_f32(checkpoint.input_std, column)
    }
    fn finished(
        &self,
        checkpoint: &PH3ImageTextConnectorCheckpoint<'_>,
        projection: &[u8],
        index: usize,
    ) -> f32 {
        let row = index / self.shape.output_dim();
        let channel = index % self.shape.output_dim();
        if row % self.shape.tokens() == 0 {
            read_f32(checkpoint.sink, channel)
        } else {
            self.output_affine.apply(
                read_f32(projection, index),
                read_f32(checkpoint.output_std, channel),
                read_f32(checkpoint.output_mean, channel),
            )
        }
    }
}

fn validate_hidden_storage(
    spec: &ResolvedTensorSpec,
    bytes: &[u8],
    dtype: ElementType,
) -> Result<()> {
    ensure!(
        spec.element_type() == dtype
            && spec.layout() == &ResolvedTensorLayout::Contiguous
            && matches!(spec.dimensions().len(), 2 | 3)
            && !spec.dimensions().contains(&0)
            && usize::try_from(spec.minimum_storage_bytes()?)? == bytes.len(),
        "connector hidden state must have its declared contiguous dtype and 2D/3D shape"
    );
    Ok(())
}

fn f32_tensor<'a>(archive: &'a SafetensorsArchive, name: &str, shape: &[u64]) -> Result<&'a [u8]> {
    let tensor = archive.tensor(name)?;
    ensure!(
        tensor.element_type() == Some(ElementType::F32) && tensor.shape() == shape,
        "connector checkpoint {name} must have exact F32 shape"
    );
    let bytes = tensor.bytes();
    ensure!(
        bytes
            .chunks_exact(4)
            .all(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()).is_finite()),
        "connector checkpoint {name} parameters must be finite"
    );
    Ok(bytes)
}
fn read_f32(bytes: &[u8], index: usize) -> f32 {
    f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
}

#[cfg(test)]
mod tests;
