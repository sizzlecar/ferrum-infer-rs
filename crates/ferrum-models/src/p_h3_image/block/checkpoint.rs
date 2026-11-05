//! Borrowed original checkpoint provenance and a charged curve-basis adapter.
use super::*;
use anyhow::{ensure, Context, Result};
use ferrum_quantization::comfy_convrot::{ComfyCanonicalWeightSource, ComfyQuantizedMatrix};
use half::bf16;
pub struct PH3ImageCurveBlockCheckpoint<'a> {
    config: &'a PH3ImageCurveBlockConfig,
    source: &'a ComfyCanonicalWeightSource,
    curve: PH3ImageCurveTimeEmbedding,
}
impl<'a> PH3ImageCurveBlockCheckpoint<'a> {
    pub fn new(
        config: &'a PH3ImageCurveBlockConfig,
        source: &'a ComfyCanonicalWeightSource,
        limits: PH3ImageCurveBlockLimits,
    ) -> Result<Self> {
        PH3ImageCurveBlockFamily::new(limits)?.check(config)?;
        let archive = source.archive();
        for (name, shape, dtype) in weights::entries(config) {
            let tensor = archive.tensor(&weights::external(config, &name))?;
            ensure!(
                tensor.shape() == shape,
                "block checkpoint {name} shape differs"
            );
            if name.ends_with(".weight_s_rel") {
                ensure!(
                    tensor.dtype() == safetensors::Dtype::F8_E4M3,
                    "W4 relative scale must retain original E4M3FN dtype"
                );
            } else {
                ensure!(
                    tensor.element_type() == Some(dtype),
                    "block checkpoint {name} dtype differs"
                );
            }
            match dtype {
                ElementType::Bf16 => {
                    ensure!(
                        tensor.bytes().chunks_exact(2).all(|b| bf16::from_bits(
                            u16::from_le_bytes(b.try_into().unwrap())
                        )
                        .is_finite()),
                        "block BF16 parameter must be finite"
                    )
                }
                ElementType::F32 => ensure!(
                    tensor
                        .bytes()
                        .chunks_exact(4)
                        .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                    "block F32 parameter must be finite"
                ),
                _ => {}
            }
        }
        for (name, expected) in config.projection_storage.modules() {
            let matrix =
                ComfyQuantizedMatrix::from_archive(archive, &weights::external(config, name))?;
            ensure!(
                PH3ImageProjectionStorage::from_metadata(matrix.metadata())? == expected,
                "block projection differs from its declared canonical storage"
            );
        }
        let table = archive.tensor("adaln_t_table")?;
        ensure!(
            table.element_type() == Some(ElementType::F32)
                && table.shape() == [config.grid, config.basis],
            "curve table header and block metadata differ"
        );
        let curve = PH3ImageCurveTimeEmbedding::new(
            usize::try_from(config.grid)?,
            usize::try_from(config.basis)?,
        )?;
        ensure!(
            table
                .bytes()
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "curve table must be finite"
        );
        Ok(Self {
            config,
            source,
            curve,
        })
    }
    pub fn source(&self) -> &'a ComfyCanonicalWeightSource {
        self.source
    }
    pub fn config(&self) -> &'a PH3ImageCurveBlockConfig {
        self.config
    }
    /// Inputs and table stay borrowed from their existing immutable owners.
    /// Only the completed small F32 basis matrix allocates, under the caller's
    /// existing InputPreprocess budget; no uncharged concatenation Vec exists.
    pub fn managed_basis(
        &self,
        host: &HostMemoryAccount,
        times: &[f32],
        arithmetic: super::super::PH3ImageCurveLerpArithmetic,
    ) -> Result<ManagedHostTensor> {
        ensure!(
            times.len() == usize::try_from(self.config.basis_rows)?,
            "basis time row count differs from block plan"
        );
        ensure!(
            cfg!(target_endian = "little"),
            "checkpoint F32 view requires a little-endian host"
        );
        let table = self.source.archive().tensor("adaln_t_table")?;
        let table: &[f32] = bytemuck::try_cast_slice(table.bytes())
            .map_err(|e| anyhow::anyhow!("curve table view alignment: {e}"))?;
        let bytes = self
            .curve
            .output_elements(times.len())?
            .checked_mul(4)
            .context("basis bytes overflow")?;
        let mut allocation = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        let output: &mut [f32] = bytemuck::try_cast_slice_mut(allocation.as_mut())
            .map_err(|e| anyhow::anyhow!("managed basis view alignment: {e}"))?;
        self.curve.write_f32(table, times, arithmetic, output)?;
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(PH3ImageCurveBlockFamily::BASIS_ID)?,
            ResolvedTensorSpec::new(
                vec![self.config.basis_rows, self.config.basis],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?,
            allocation.freeze(),
        )?)
    }
}
