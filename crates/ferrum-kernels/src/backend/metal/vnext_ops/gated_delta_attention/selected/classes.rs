//! Immutable algorithm identities for one bound GDN layout.
use super::*;

fn layout_fields(p: &GatedDeltaParams) -> [u32; 14] {
    [
        p.hidden_size,
        p.key_heads,
        p.value_heads,
        p.key_dim,
        p.value_dim,
        p.qkv_features,
        p.value_features,
        p.qkvz_features,
        p.ba_features,
        p.conv_kernel,
        p.epsilon.to_bits(),
        p.scale.to_bits(),
        p.decay_parameterization,
        p.value_head_mapping,
    ]
}

fn identity(
    entry: &'static str,
    fields: &[u32; 14],
    threads: u64,
) -> Option<SelectedAlgorithmClassV1> {
    static NUMERIC: OnceLock<[u8; 32]> = OnceLock::new();
    let numeric = *NUMERIC.get_or_init(|| {
        let mut h = Sha256::new();
        h.update(b"metal.gdn.default-compile-options.recurrent.v1");
        h.update(SHADER_SOURCE.as_bytes());
        h.update(include_str!("../selected.rs").as_bytes());
        h.update(include_str!("observation.rs").as_bytes());
        h.finalize().into()
    });
    let mut layout = Sha256::new();
    layout.update(b"gdn.params.f16conv.f32state.v1");
    for n in fields {
        layout.update(n.to_le_bytes());
    }
    layout.update(threads.to_le_bytes());
    SelectedAlgorithmClassV1::new(entry, 1, numeric, layout.finalize().into()).ok()
}

struct Entry {
    name: &'static str,
    threads: u64,
    class: SelectedAlgorithmClassV1,
}

/// The table belongs to the immutable plan, with no request rows, addresses,
/// or current work. Actual launch selection still chooses the entry/threads.
pub(in crate::backend::metal::vnext_ops::gated_delta_attention) struct PreparedKernelClasses {
    fields: [u32; 14],
    entries: Vec<Entry>,
}
impl PreparedKernelClasses {
    pub(in crate::backend::metal::vnext_ops::gated_delta_attention) fn new(
        params: &GatedDeltaParams,
    ) -> Option<Self> {
        let fields = layout_fields(params);
        let entries = [
            (PREPARE_CONV_KERNEL, THREADS_PER_GROUP),
            (COLLECT_CONV_STATE_KERNEL, THREADS_PER_GROUP),
            (COPY_F16_KERNEL, THREADS_PER_GROUP),
            (PREPARE_GATES_KERNEL, THREADS_PER_GROUP),
            (QK_NORM_KERNEL, THREADS_PER_GROUP),
            (GATED_NORM_KERNEL, THREADS_PER_GROUP),
            (DELTA_KERNEL, THREADS_PER_GROUP),
            (SIMD_DELTA_KERNEL, SIMD_DELTA_THREADS),
        ]
        .into_iter()
        .map(|(name, threads)| {
            Some(Entry {
                name,
                threads,
                class: identity(name, &fields, threads)?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
        Some(Self { fields, entries })
    }
    pub(super) fn matches(&self, params: &GatedDeltaParams) -> bool {
        self.fields == layout_fields(params)
    }
    pub(super) fn get(&self, name: &str, threads: u64) -> Option<SelectedAlgorithmClassV1> {
        self.entries
            .iter()
            .find(|entry| entry.name == name && entry.threads == threads)
            .map(|entry| entry.class)
    }
}

pub(super) fn fresh(
    entry: &'static str,
    params: &GatedDeltaParams,
    threads: u64,
) -> Option<SelectedAlgorithmClassV1> {
    identity(entry, &layout_fields(params), threads)
}
