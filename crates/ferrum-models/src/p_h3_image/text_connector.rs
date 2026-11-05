//! Borrowed FP32 mathematical reference for `PH3ImageTextConnector` from
//! Diffusers `c55ec27943462eeeb65c177d6ff7d8272d21d7a7`.
//!
//! This is a scalar CPU oracle for future device implementations, not a model
//! pipeline, weight loader, GPU operation, or runtime allocation authority.
//! Projection accumulates left to right in FP32 with separate multiply/add;
//! GPU GEMM reduction order may differ and requires measured numerical checks.

use anyhow::{ensure, Context, Result};

use super::PH3ImageTextConnectorConfig;

/// Output destandardization only. The fused profile models Comfy's F32
/// `torch.addcmul(mean, projected, std)` expression; it is a reference arithmetic
/// choice, not a claim of measured Torch binary or GEMM reduction parity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PH3ImageTextConnectorAffineArithmetic {
    #[default]
    SeparateMulAddF32,
    FusedAddcmulF32,
}
impl PH3ImageTextConnectorAffineArithmetic {
    pub const fn numerical_profile_id(self) -> &'static str {
        match self {
            Self::SeparateMulAddF32 => "p_h3_image.text_connector.affine.separate_mul_add.f32.v1",
            Self::FusedAddcmulF32 => "p_h3_image.text_connector.affine.fused_addcmul.f32.v1",
        }
    }
    pub(super) fn apply(self, projected: f32, std: f32, mean: f32) -> f32 {
        match self {
            Self::SeparateMulAddF32 => {
                let scaled = projected * std;
                scaled + mean
            }
            Self::FusedAddcmulF32 => projected.mul_add(std, mean),
        }
    }
}

/// Checked contiguous `[B,S,H]` hidden states and `[B,S,O]` conditioning shape.
/// The projection is the original bias-free PyTorch `[O, taps * H]` weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PH3ImageTextConnectorShape {
    batch: usize,
    tokens: usize,
    hidden_size: usize,
    taps: usize,
    input_dim: usize,
    output_dim: usize,
    rows: usize,
    state_elements: usize,
    projection_elements: usize,
    output_elements: usize,
}

impl PH3ImageTextConnectorShape {
    pub fn new(config: &PH3ImageTextConnectorConfig, batch: usize, tokens: usize) -> Result<Self> {
        ensure!(
            batch > 0 && tokens > 0,
            "text connector batch and token count must be positive"
        );
        let hidden_size = usize::try_from(config.text_hidden_size())?;
        let taps = config.hidden_state_indices().len();
        let input_dim = usize::try_from(config.input_dim())?;
        let output_dim = usize::try_from(config.out_channels())?;
        let rows = batch
            .checked_mul(tokens)
            .context("text connector row count overflows")?;
        let state_elements = checked_storage(&[rows, hidden_size])?;
        let projection_elements = checked_storage(&[output_dim, input_dim])?;
        let output_elements = checked_storage(&[rows, output_dim])?;
        // The statistics/sink slices obey the same byte/address-space limit,
        // including shapes whose matrix extents already failed above.
        checked_storage(&[input_dim])?;
        checked_storage(&[output_dim])?;
        Ok(Self {
            batch,
            tokens,
            hidden_size,
            taps,
            input_dim,
            output_dim,
            rows,
            state_elements,
            projection_elements,
            output_elements,
        })
    }

    pub fn batch(self) -> usize {
        self.batch
    }
    pub fn tokens(self) -> usize {
        self.tokens
    }
    pub fn hidden_size(self) -> usize {
        self.hidden_size
    }
    pub fn taps(self) -> usize {
        self.taps
    }
    pub fn input_dim(self) -> usize {
        self.input_dim
    }
    pub fn output_dim(self) -> usize {
        self.output_dim
    }
    pub fn state_elements(self) -> usize {
        self.state_elements
    }
    pub fn projection_elements(self) -> usize {
        self.projection_elements
    }
    pub fn output_elements(self) -> usize {
        self.output_elements
    }
}

/// One selected encoder state in contiguous `[B,S,H]` order. Entries supplied
/// to the reference must follow the configuration's tap order exactly; the
/// explicit index prevents silently treating differently ordered states alike.
#[derive(Clone, Copy, Debug)]
pub struct PH3ImageHiddenState<'a> {
    pub index: u32,
    pub values: &'a [f32],
}

/// Original FP32 checkpoint tensor views. Means/stds broadcast over `[B,S]`;
/// projection rows are output channels. There is no projection bias. This
/// view borrows existing backing and neither allocates nor owns a resource
/// account. Values are validated on each reference call before output changes.
#[derive(Clone, Copy, Debug)]
pub struct PH3ImageTextConnectorWeights<'a> {
    pub input_mean: &'a [f32],
    pub input_std: &'a [f32],
    pub projection: &'a [f32],
    pub output_mean: &'a [f32],
    pub output_std: &'a [f32],
    pub sink_embedding: &'a [f32],
}

impl PH3ImageTextConnectorWeights<'_> {
    pub fn validate(self, shape: PH3ImageTextConnectorShape) -> Result<()> {
        for (name, values, expected) in [
            ("input_mean", self.input_mean, shape.input_dim),
            ("input_std", self.input_std, shape.input_dim),
            ("projection", self.projection, shape.projection_elements),
            ("output_mean", self.output_mean, shape.output_dim),
            ("output_std", self.output_std, shape.output_dim),
            ("sink_embedding", self.sink_embedding, shape.output_dim),
        ] {
            ensure!(
                values.len() == expected,
                "text connector {name} needs {expected} elements, got {}",
                values.len()
            );
            ensure!(
                values.iter().all(|value| value.is_finite()),
                "text connector {name} must be finite"
            );
        }
        ensure!(
            self.input_std
                .iter()
                .chain(self.output_std)
                .all(|value| *value > 0.0),
            "text connector standard deviations must be strictly positive"
        );
        Ok(())
    }
}

/// Checked shape plus borrowed configuration for the FP32 CPU oracle. Each
/// call has independent input/output storage and no retained inference state.
pub struct PH3ImageTextConnectorReference<'a> {
    config: &'a PH3ImageTextConnectorConfig,
    shape: PH3ImageTextConnectorShape,
    output_affine: PH3ImageTextConnectorAffineArithmetic,
}

impl<'a> PH3ImageTextConnectorReference<'a> {
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

    pub fn with_output_affine(mut self, arithmetic: PH3ImageTextConnectorAffineArithmetic) -> Self {
        self.output_affine = arithmetic;
        self
    }

    pub fn shape(&self) -> PH3ImageTextConnectorShape {
        self.shape
    }

    /// Concatenate the declared taps logically along H, compute
    /// `((states - input_mean) / input_std) @ projection.T`, then apply the
    /// selected output affine profile. All arithmetic and outputs stay FP32.
    /// Every sequence's token zero is replaced with the shared sink embedding.
    ///
    /// No concatenation/scratch allocation is made. A first pass validates all
    /// surviving projected outputs, then a second writes caller-owned storage;
    /// any error leaves its previous bits unchanged. Inputs, checkpoint values
    /// and stds are checked even for sink rows; projected sink values themselves
    /// need not be representable because the output replaces them entirely.
    pub fn write_f32(
        &self,
        states: &[PH3ImageHiddenState<'_>],
        weights: PH3ImageTextConnectorWeights<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        let shape = self.shape;
        ensure!(
            output.len() == shape.output_elements,
            "text connector output length differs from checked shape"
        );
        ensure!(
            states.len() == shape.taps,
            "text connector state count differs from declared taps"
        );
        weights.validate(shape)?;
        for (state, expected) in states.iter().zip(self.config.hidden_state_indices()) {
            ensure!(
                state.index == *expected,
                "text connector hidden-state order differs from declared taps"
            );
            ensure!(
                state.values.len() == shape.state_elements,
                "text connector hidden-state length differs from checked shape"
            );
            ensure!(
                state.values.iter().all(|value| value.is_finite()),
                "text connector hidden states must be finite"
            );
        }
        for row in 0..shape.rows {
            if row % shape.tokens == 0 {
                continue;
            }
            for channel in 0..shape.output_dim {
                ensure!(
                    self.project(states, weights, row, channel).is_finite(),
                    "text connector projected output is not finite at row {row}, channel {channel}"
                );
            }
        }
        for row in 0..shape.rows {
            let target = &mut output[row * shape.output_dim..(row + 1) * shape.output_dim];
            if row % shape.tokens == 0 {
                target.copy_from_slice(weights.sink_embedding);
            } else {
                for (channel, value) in target.iter_mut().enumerate() {
                    *value = self.project(states, weights, row, channel);
                }
            }
        }
        Ok(())
    }

    fn project(
        &self,
        states: &[PH3ImageHiddenState<'_>],
        weights: PH3ImageTextConnectorWeights<'_>,
        row: usize,
        channel: usize,
    ) -> f32 {
        let shape = self.shape;
        let mut sum = 0.0_f32;
        for (tap, state) in states.iter().enumerate() {
            for hidden in 0..shape.hidden_size {
                let column = tap * shape.hidden_size + hidden;
                let centered =
                    state.values[row * shape.hidden_size + hidden] - weights.input_mean[column];
                let normalized = centered / weights.input_std[column];
                let product = normalized * weights.projection[channel * shape.input_dim + column];
                sum += product;
            }
        }
        self.output_affine.apply(
            sum,
            weights.output_std[channel],
            weights.output_mean[channel],
        )
    }
}

fn checked_storage(dimensions: &[usize]) -> Result<usize> {
    let elements = dimensions.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .context("text connector element count overflows")
    })?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .context("text connector FP32 byte count overflows")?;
    ensure!(
        bytes <= isize::MAX as usize,
        "text connector FP32 storage exceeds address space"
    );
    Ok(elements)
}

#[cfg(test)]
mod tests;
