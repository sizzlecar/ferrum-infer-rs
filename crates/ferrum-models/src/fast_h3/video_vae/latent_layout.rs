//! Explicit storage transport between DiT spatial patches and VAE NCTHW.
use anyhow::{ensure, Result};
use ferrum_diffusion::SpatiotemporalLatentShape;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBytes};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FastH3VideoLatentLayout {
    #[default]
    Ncthw,
    PackedSpatial2x2,
}
impl FastH3VideoLatentLayout {
    /// Preserve the borrowed source owner; a packed input gets a separately
    /// admitted dense buffer. No normalization or scalar rounding occurs here.
    pub fn managed_ncthw(
        self,
        dense_shape: [usize; 5],
        source: &ManagedHostBytes,
        host: &HostMemoryAccount,
        maximum_bytes: usize,
    ) -> Result<ManagedHostBytes> {
        let patch = match self {
            Self::Ncthw => [1, 1, 1],
            Self::PackedSpatial2x2 => [1, 2, 2],
        };
        let shape = SpatiotemporalLatentShape::new(dense_shape, patch)?;
        let bytes = shape.byte_len::<f32>()?;
        ensure!(
            source.len() == bytes && bytes <= maximum_bytes,
            "video latent storage differs from declared layout or exceeds capacity"
        );
        ensure!(
            source
                .bytes()
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "video latent input contains nonfinite F32"
        );
        if self == Self::Ncthw {
            return Ok(source.clone());
        }
        let mut dense = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        shape.for_each_packed_index(|packed, index| {
            dense.as_mut()[index * 4..index * 4 + 4]
                .copy_from_slice(&source.bytes()[packed * 4..packed * 4 + 4]);
        });
        Ok(dense.freeze())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
    #[test]
    fn original_dit_spatial_patches_become_decoder_ncthw_without_relabeling_output_owner() {
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                4096,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 4096,
                    transfer_staging_bytes: 4096,
                    output_bytes: 4096,
                    weight_backing_bytes: 4096,
                },
            )
            .unwrap(),
        );
        // N=1,C=2,T=2,H=2,W=2, original packed order is (time,channel,dh,dw).
        let values = [
            0f32, 1., 2., 3., 8., 9., 10., 11., 4., 5., 6., 7., 12., 13., 14., 15.,
        ];
        let mut bytes = host.allocate_zeroed(HostMemoryClass::Output, 64).unwrap();
        for (slot, x) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
            slot.copy_from_slice(&x.to_le_bytes());
        }
        let source = bytes.freeze();
        let dense = FastH3VideoLatentLayout::PackedSpatial2x2
            .managed_ncthw([1, 2, 2, 2, 2], &source, &host, 64)
            .unwrap();
        assert_eq!(source.class(), HostMemoryClass::Output);
        assert_eq!(
            dense
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(),
            (0..16).map(|i| i as f32).collect::<Vec<_>>()
        );
        let before = host.usage().unwrap().committed_bytes();
        let alias = FastH3VideoLatentLayout::Ncthw
            .managed_ncthw([1, 2, 2, 2, 2], &dense, &host, 64)
            .unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
        assert_eq!(alias.bytes(), dense.bytes());
        assert!(FastH3VideoLatentLayout::PackedSpatial2x2
            .managed_ncthw([1, 2, 2, 2, 2], &source, &host, 63)
            .is_err());
        assert!(FastH3VideoLatentLayout::PackedSpatial2x2
            .managed_ncthw([1, 2, 2, 1, 4], &source, &host, 64)
            .is_err());
        drop((source, dense, alias));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
