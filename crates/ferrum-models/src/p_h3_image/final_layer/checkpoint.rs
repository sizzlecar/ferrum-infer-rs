use super::*;
use anyhow::{ensure, Context, Result};
use ferrum_quantization::SafetensorsArchive;
use half::bf16;
pub struct PH3ImageFinalLayerCheckpoint<'a> {
    config: &'a PH3ImageFinalLayerConfig,
    source: &'a SafetensorsArchive,
    curve: PH3ImageCurveTimeEmbedding,
}
impl<'a> PH3ImageFinalLayerCheckpoint<'a> {
    pub fn new(
        config: &'a PH3ImageFinalLayerConfig,
        source: &'a SafetensorsArchive,
        limits: PH3ImageFinalLayerLimits,
    ) -> Result<Self> {
        PH3ImageFinalLayerFamily::new(limits)?.check(config)?;
        for (name, shape, dtype) in weights::entries(config) {
            let t = source.tensor(name)?;
            ensure!(t.shape()==shape&&t.element_type()==Some(dtype),"final checkpoint {name} shape/dtype differs; stacked PDD head banks are unsupported");
            let finite = match dtype {
                ElementType::Bf16 => t.bytes().chunks_exact(2).all(|b| {
                    bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()
                }),
                ElementType::F32 => t
                    .bytes()
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                _ => unreachable!(),
            };
            ensure!(finite, "final checkpoint {name} contains nonfinite values");
        }
        let table = source.tensor("adaln_t_table")?;
        ensure!(
            table.element_type() == Some(ElementType::F32)
                && table.shape() == [config.grid, config.basis],
            "final curve table shape/dtype differs"
        );
        ensure!(
            table
                .bytes()
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "curve table must be finite"
        );
        let curve = PH3ImageCurveTimeEmbedding::new(
            usize::try_from(config.grid)?,
            usize::try_from(config.basis)?,
        )?;
        Ok(Self {
            config,
            source,
            curve,
        })
    }
    pub fn source(&self) -> &'a SafetensorsArchive {
        self.source
    }
    pub fn config(&self) -> &'a PH3ImageFinalLayerConfig {
        self.config
    }
    /// Only the small completed basis input allocates, under the caller's
    /// existing Host account. Projection/norm/modulation/heads run on device.
    pub fn managed_basis(
        &self,
        host: &HostMemoryAccount,
        times: &[f32],
        arithmetic: PH3ImageCurveLerpArithmetic,
    ) -> Result<ManagedHostTensor> {
        ensure!(
            times.len() == usize::try_from(self.config.basis_rows)?
                && cfg!(target_endian = "little"),
            "basis row count or host byte order differs"
        );
        let table = self.source.tensor("adaln_t_table")?;
        let table: &[f32] = bytemuck::try_cast_slice(table.bytes())
            .map_err(|e| anyhow::anyhow!("curve alignment: {e}"))?;
        let bytes = self
            .curve
            .output_elements(times.len())?
            .checked_mul(4)
            .context("basis bytes overflow")?;
        let mut allocation = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        let out: &mut [f32] = bytemuck::try_cast_slice_mut(allocation.as_mut())
            .map_err(|e| anyhow::anyhow!("managed basis alignment: {e}"))?;
        self.curve.write_f32(table, times, arithmetic, out)?;
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(PH3ImageFinalLayerFamily::BASIS_ID)?,
            ResolvedTensorSpec::new(
                vec![self.config.basis_rows, self.config.basis],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?,
            allocation.freeze(),
        )?)
    }
}
impl WeightComponentSource for PH3ImageFinalLayerCheckpoint<'_> {
    fn component<'source>(
        &'source self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.source.component(c)
    }
}
