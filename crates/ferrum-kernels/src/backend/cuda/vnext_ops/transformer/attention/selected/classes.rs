//! Fixed entry/block identities from the actual launch selector's static part.
use super::*;

pub(super) fn kernel_class(
    entry: &str,
    block: (u32, u32, u32),
) -> Option<SelectedAlgorithmClassV1> {
    const PREFIX: &[u8] = b"gdn.native-kernel-geometry.v1";
    let mut layout = [0u8; PREFIX.len() + 12];
    layout[..PREFIX.len()].copy_from_slice(PREFIX);
    for (slot, dimension) in layout[PREFIX.len()..]
        .chunks_exact_mut(4)
        .zip([block.0, block.1, block.2])
    {
        slot.copy_from_slice(&dimension.to_le_bytes());
    }
    class(entry, &layout)
}

struct Entry {
    name: &'static str,
    block: (u32, u32, u32),
    class: SelectedAlgorithmClassV1,
}

pub(in crate::backend::cuda::vnext_ops::transformer::attention) struct PreparedKernelClasses {
    entries: Vec<Entry>,
}

impl PreparedKernelClasses {
    pub(in crate::backend::cuda::vnext_ops::transformer::attention) fn new(
        shape: AttentionShape,
        precision: AttentionPrecision,
    ) -> Option<Self> {
        let cuda = shape.cuda_shape().ok()?;
        let prepare = match shape.decay_parameterization {
            GatedDeltaDecayParameterization::LogRate => PREPARE_FUNCTION,
            GatedDeltaDecayParameterization::NegativeRate => PREPARE_NEGATIVE_RATE_FUNCTION,
        };
        let flat = (THREADS_PER_BLOCK, 1, 1);
        let entries = [
            (
                precision.norm(),
                launch_geometry::rms_block(cuda.hidden_size),
            ),
            (prepare, flat),
            (CONV_STATE_COMMIT_FUNCTION, flat),
            (QK_NORM_FUNCTION, launch_geometry::qk_block(cuda)),
            (
                launch_geometry::delta_entry(cuda),
                launch_geometry::delta_block(cuda),
            ),
            (GATED_NORM_FUNCTION, launch_geometry::gated_block(cuda)),
            (F32_TO_F16_FUNCTION, flat),
            (precision.residual(), flat),
        ]
        .into_iter()
        .map(|(name, block)| {
            Some(Entry {
                name,
                block,
                class: kernel_class(name, block)?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
        Some(Self { entries })
    }

    pub(super) fn get(
        &self,
        name: &str,
        block: (u32, u32, u32),
    ) -> Option<SelectedAlgorithmClassV1> {
        self.entries
            .iter()
            .find(|entry| entry.name == name && entry.block == block)
            .map(|entry| entry.class)
    }
}
