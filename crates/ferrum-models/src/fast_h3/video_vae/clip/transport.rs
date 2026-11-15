use super::*;
use ferrum_diffusion::SpatiotemporalLatentShape;
use ferrum_execution::TensorExecutionOutput;
use half::bf16;

/// Raw BF16 decoder clip in NCTHW order. Temporal/spatial overlap blending and
/// final RGB range conversion are separate operations, not implicit here.
#[derive(Debug)]
pub struct FastH3VideoRawClip {
    pub(in crate::fast_h3::video_vae) shape: [usize; 5],
    pub(in crate::fast_h3::video_vae) bytes: ManagedHostBytes,
}
impl FastH3VideoRawClip {
    pub fn ncthw_shape(&self) -> [usize; 5] {
        self.shape
    }
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
}
impl FastH3VideoClipConfig {
    /// F32 coordinates/angles with libm trigonometry, then BF16. This declares
    /// the host recipe; it does not assert Torch CUDA trigonometry bit parity.
    pub const ROTARY_PROFILE_ID: &'static str = "fast_h3.video_clip.grid_f32.libm-0.2.15.bf16.v1";

    fn input_shape(&self) -> Result<SpatiotemporalLatentShape, VNextError> {
        let [b, t, h, w] = self.grid.map(|n| usize::try_from(n).map_err(invalid));
        SpatiotemporalLatentShape::new(
            [b?, self.architecture.latent_channels(), t?, h?, w?],
            [1, 1, 1],
        )
        .map_err(invalid)
    }
    fn output_shape(&self) -> Result<SpatiotemporalLatentShape, VNextError> {
        let [b, t, h, w] = self.grid.map(|n| usize::try_from(n).map_err(invalid));
        let pt = self
            .architecture
            .temporal_compression_ratio()
            .map_err(invalid)?;
        let ps = self
            .architecture
            .spatial_compression_ratio()
            .map_err(invalid)?;
        SpatiotemporalLatentShape::new(
            [
                b?,
                3,
                t?.checked_mul(pt)
                    .ok_or_else(|| invalid("pixel time overflow"))?,
                h?.checked_mul(ps)
                    .ok_or_else(|| invalid("pixel height overflow"))?,
                w?.checked_mul(ps)
                    .ok_or_else(|| invalid("pixel width overflow"))?,
            ],
            [pt, ps, ps],
        )
        .map_err(invalid)
    }
    /// The source is already in decoder latent units (normalization undone).
    /// No dtype conversion, hidden scale or framewise decode is performed.
    pub fn managed_input_from_decoder_ncthw(
        &self,
        input: &[f32],
        host: &HostMemoryAccount,
        limits: FastH3VideoClipLimits,
    ) -> Result<ManagedHostTensor, VNextError> {
        self.validate(limits)?;
        let shape = self.input_shape()?;
        if input.len() != shape.element_count() || input.iter().any(|x| !x.is_finite()) {
            return Err(invalid(
                "decoder latent input length or finite domain differs",
            ));
        }
        let bytes = shape.byte_len::<f32>().map_err(invalid)?;
        if bytes as u64 > limits.maximum_activation_bytes {
            return Err(invalid("latent preprocessing exceeds capacity"));
        }
        let mut result = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)
            .map_err(invalid)?;
        shape.for_each_packed_index(|p, d| {
            result.as_mut()[4 * p..4 * p + 4].copy_from_slice(&input[d].to_le_bytes())
        });
        ManagedHostTensor::new(
            ProgramValueId::new(FastH3VideoClipFamily::INPUT_ID)?,
            ResolvedTensorSpec::new(
                vec![
                    self.patch_rows()?,
                    self.architecture.latent_channels() as u64,
                ],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?,
            result.freeze(),
        )
    }
    pub fn managed_bf16_frequencies(
        &self,
        host: &HostMemoryAccount,
        limits: FastH3VideoClipLimits,
    ) -> Result<(ManagedHostTensor, ManagedHostTensor), VNextError> {
        self.validate(limits)?;
        let rows = usize::try_from(self.rows()?).map_err(invalid)?;
        let width = self.architecture.rotary_dimension();
        let bytes = rows
            .checked_mul(width)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("rotary extent overflow"))?;
        if bytes as u64 > limits.maximum_activation_bytes {
            return Err(invalid("rotary table exceeds capacity"));
        }
        let mut cosine = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)
            .map_err(invalid)?;
        let mut sine = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)
            .map_err(invalid)?;
        let sequence = usize::try_from(self.sequence_rows()?).map_err(invalid)?;
        let patches = usize::try_from(self.patch_rows_per_clip()?).map_err(invalid)?;
        let [_, time, height, wide] = self.grid.map(|n| n as usize);
        let theta = self.architecture.rotary_theta() as f32;
        if !theta.is_finite() || theta <= 0. {
            return Err(invalid("rotary theta is not finite positive F32"));
        }
        for frequency in 0..width / 6 {
            // torch.arange(0,1,6/rot) is a float step multiplied by index.
            let exponent = frequency as f32 * (6.0f32 / width as f32);
            let inverse = 1f32 / libm::powf(theta, exponent);
            if !inverse.is_finite() || inverse <= 0. {
                return Err(invalid("invalid inverse rotary frequency"));
            }
            for row in 0..rows {
                let token = row % sequence;
                let coordinates = if token < patches {
                    let ijk = [
                        token / (height * wide),
                        (token / wide) % height,
                        token % wide,
                    ];
                    let sizes = [time, height, wide];
                    std::array::from_fn(|axis| {
                        2f32 * ((ijk[axis] as f32 + 0.5f32) / sizes[axis] as f32) - 1f32
                    })
                } else {
                    [0f32; 3]
                };
                for (axis, coordinate) in coordinates.into_iter().enumerate() {
                    // Source computes (2*pi*position) before frequency product.
                    let angle = (2f32 * std::f32::consts::PI * coordinate) * inverse;
                    if !angle.is_finite() {
                        return Err(invalid("nonfinite rotary angle"));
                    }
                    let c = bf16::from_f32(libm::cosf(angle)).to_bits().to_le_bytes();
                    let s = bf16::from_f32(libm::sinf(angle)).to_bits().to_le_bytes();
                    let column = axis * (width / 6) + frequency;
                    for j in [column, column + width / 2] {
                        let offset = 2 * (row * width + j);
                        cosine.as_mut()[offset..offset + 2].copy_from_slice(&c);
                        sine.as_mut()[offset..offset + 2].copy_from_slice(&s);
                    }
                }
            }
        }
        let spec = ResolvedTensorSpec::new(
            vec![rows as u64, width as u64],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        Ok((
            ManagedHostTensor::new(
                ProgramValueId::new(FastH3VideoClipFamily::COSINE_ID)?,
                spec.clone(),
                cosine.freeze(),
            )?,
            ManagedHostTensor::new(
                ProgramValueId::new(FastH3VideoClipFamily::SINE_ID)?,
                spec,
                sine.freeze(),
            )?,
        ))
    }
    /// Consume a completed output view only after exact identity and geometry
    /// checks. The new Output-class permutation is admitted before allocation;
    /// the source completion and its original owner are never relabelled.
    pub fn unpack_execution(
        &self,
        output: &TensorExecutionOutput,
        host: &HostMemoryAccount,
        limits: FastH3VideoClipLimits,
    ) -> Result<FastH3VideoRawClip, VNextError> {
        self.unpack_execution_batch_range(
            output,
            0..usize::try_from(self.grid[0]).map_err(invalid)?,
            host,
            limits,
        )
    }
    /// Select complete leading decoder items from one original Core readback.
    /// Each unpack owns independent Host backing, so retaining a raw spatial
    /// neighbour never pins the larger readback group through a hidden view.
    pub fn unpack_execution_batch_range(
        &self,
        output: &TensorExecutionOutput,
        batch: std::ops::Range<usize>,
        host: &HostMemoryAccount,
        limits: FastH3VideoClipLimits,
    ) -> Result<FastH3VideoRawClip, VNextError> {
        self.validate(limits)?;
        let expected = ResolvedTensorSpec::new(
            vec![
                self.patch_rows()?,
                self.architecture.output_patch_width().map_err(invalid)? as u64,
            ],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        if output.value_id().as_str() != FastH3VideoClipFamily::OUTPUT_ID
            || output.tensor() != &expected
            || output.bytes().class() != HostMemoryClass::Output
        {
            return Err(invalid(
                "clip output identity, tensor or owner class differs",
            ));
        }
        self.unpack_bytes_batch_range(output.bytes().bytes(), batch, host, limits)
    }
    pub(super) fn unpack_bytes(
        &self,
        input: &[u8],
        host: &HostMemoryAccount,
        limits: FastH3VideoClipLimits,
    ) -> Result<FastH3VideoRawClip, VNextError> {
        self.unpack_bytes_batch_range(
            input,
            0..usize::try_from(self.grid[0]).map_err(invalid)?,
            host,
            limits,
        )
    }
    pub(in crate::fast_h3::video_vae) fn unpack_bytes_batch_range(
        &self,
        input: &[u8],
        batch: std::ops::Range<usize>,
        host: &HostMemoryAccount,
        limits: FastH3VideoClipLimits,
    ) -> Result<FastH3VideoRawClip, VNextError> {
        let shape = self.output_shape()?;
        let bytes = shape.byte_len::<u16>().map_err(invalid)?;
        if input.len() != bytes
            || bytes as u64 > limits.maximum_activation_bytes
            || batch.start >= batch.end
            || batch.end > shape.ncthw_dims()[0]
        {
            return Err(invalid(
                "raw clip batch range, source extent or capacity differs",
            ));
        }
        let stride = bytes / shape.ncthw_dims()[0];
        let input = &input[batch.start * stride..batch.end * stride];
        if input
            .chunks_exact(2)
            .any(|b| !bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite())
        {
            return Err(invalid(
                "raw clip output extent, finite domain or capacity differs",
            ));
        }
        let mut dimensions = shape.ncthw_dims();
        dimensions[0] = batch.len();
        let shape = SpatiotemporalLatentShape::new(
            dimensions,
            [
                self.architecture
                    .temporal_compression_ratio()
                    .map_err(invalid)?,
                self.architecture
                    .spatial_compression_ratio()
                    .map_err(invalid)?,
                self.architecture
                    .spatial_compression_ratio()
                    .map_err(invalid)?,
            ],
        )
        .map_err(invalid)?;
        let mut output = host
            .allocate_zeroed(HostMemoryClass::Output, input.len())
            .map_err(invalid)?;
        let destination = output.as_mut();
        shape.for_each_packed_span(|p, d| {
            destination[2 * d.start..2 * d.end].copy_from_slice(&input[2 * p.start..2 * p.end])
        });
        Ok(FastH3VideoRawClip {
            shape: shape.ncthw_dims(),
            bytes: output.freeze(),
        })
    }
}
