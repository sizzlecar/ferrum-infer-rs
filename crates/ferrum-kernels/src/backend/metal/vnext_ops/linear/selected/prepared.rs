//! Fixed linear ABI classes for a bound provider. The real query still selects
//! its current PSO and dynamic grid; this table supplies only the same immutable
//! class digest that `class` would otherwise rebuild for that selection.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
struct MatrixAbi {
    input: u32,
    output: u32,
    stride: u32,
    column: u32,
    activation: ElementType,
    format: LinearPhysicalFormat,
}
impl MatrixAbi {
    fn from_launch(launch: LinearLaunch) -> Option<Self> {
        if launch.transform.is_some() {
            return None;
        }
        Some(Self {
            input: launch.params.in_features,
            output: launch.params.out_features,
            stride: launch.params.output_stride,
            column: launch.params.output_column_offset,
            activation: launch.activation_type,
            format: launch.format,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ClassKey {
    entry: &'static str,
    threads: [u32; 3],
    shared_bytes: u64,
    staged: bool,
}
impl ClassKey {
    const fn new(entry: &'static str, threads: [u32; 3], shared_bytes: u64, staged: bool) -> Self {
        Self {
            entry,
            threads,
            shared_bytes,
            staged,
        }
    }
    fn geometry(self) -> Grid {
        // These placeholders are not a launch. Only block dimensions and shared
        // memory enter `class`; groups/padded work are computed by the query.
        Grid {
            groups: [1, 1, 1],
            threads: self.threads,
            threadgroup_memory: Some(self.shared_bytes),
            padded_outputs: 1,
        }
    }
}

const GEMV: [u32; 3] = [32, 2, 1];
const TILED: [u32; 3] = [128, 1, 1];
// Complete existing mapped plain/staged evidence vocabulary. This is not an
// algorithm selector: entry() must first identify the actual selected PSO.
const CLASSES: &[ClassKey] = &[
    ClassKey::new(LINEAR_DENSE_KERNEL, GEMV, 0, false),
    ClassKey::new(LINEAR_DENSE_F32_KERNEL, GEMV, 0, false),
    ClassKey::new(LINEAR_Q8_0_KERNEL, GEMV, 0, false),
    ClassKey::new(LINEAR_Q8_0_F32_KERNEL, GEMV, 0, false),
    ClassKey::new(
        crate::backend::metal::q4_k_gemv_v2::F16_BATCHED_KERNEL_NAME,
        GEMV,
        0,
        false,
    ),
    ClassKey::new(
        crate::backend::metal::q4_k_gemv_v2::F32_BATCHED_KERNEL_NAME,
        GEMV,
        0,
        false,
    ),
    ClassKey::new(
        crate::backend::metal::q5_k_gemv::F16_BATCHED_KERNEL_NAME,
        GEMV,
        0,
        false,
    ),
    ClassKey::new(
        crate::backend::metal::q6_k_gemv::F16_BATCHED_KERNEL_NAME,
        GEMV,
        0,
        false,
    ),
    ClassKey::new(
        crate::backend::metal::q6_k_gemv::F32_BATCHED_KERNEL_NAME,
        GEMV,
        0,
        false,
    ),
    ClassKey::new("gemm_f16a_q4kw_tiled", TILED, 8192, false),
    ClassKey::new("gemm_f16a_q5kw_tiled", TILED, 8192, false),
    ClassKey::new("gemm_f16a_q6kw_tiled", TILED, 8192, false),
    ClassKey::new("gemm_f16a_q8_0w_tiled", TILED, 8192, false),
    ClassKey::new("gemm_f16a_q4kw_m8", TILED, 8192, false),
    ClassKey::new("gemm_f16a_q5kw_m8", TILED, 8192, false),
    ClassKey::new("gemm_f16a_q6kw_m8", TILED, 8192, false),
    ClassKey::new(
        LINEAR_DENSE_NARROW_KERNEL,
        [NARROW_DENSE_THREADS as u32, 1, 1],
        0,
        false,
    ),
    ClassKey::new("q4_shared_b2", GEMV, 0, false),
    ClassKey::new("q4_shared_b3", GEMV, 0, false),
    ClassKey::new("q4_shared_b4", GEMV, 0, false),
    ClassKey::new("q5_shared_b2", GEMV, 0, false),
    ClassKey::new("q5_shared_b3", GEMV, 0, false),
    ClassKey::new("q5_shared_b4", GEMV, 0, false),
    ClassKey::new("q6_shared_b2", GEMV, 0, false),
    ClassKey::new("q6_shared_b3", GEMV, 0, false),
    ClassKey::new("q6_shared_b4", GEMV, 0, false),
    ClassKey::new("q6_shared_f32_b2", GEMV, 0, false),
    ClassKey::new("q6_shared_f32_b3", GEMV, 0, false),
    ClassKey::new("q6_shared_f32_b4", GEMV, 0, false),
    ClassKey::new("stage_q4k_f16", TILED, 0, true),
    ClassKey::new("stage_q5k_f16", TILED, 0, true),
    ClassKey::new("stage_q6k_f16", TILED, 0, true),
    ClassKey::new("gemm_f16a_f16w_tiled", TILED, 8192, true),
];

pub(in crate::backend::metal::vnext_ops) struct PreparedLinearClasses {
    abi: MatrixAbi,
    entries: Vec<(ClassKey, SelectedAlgorithmClassV1)>,
}
impl PreparedLinearClasses {
    pub(in crate::backend::metal::vnext_ops) fn new(launch: LinearLaunch) -> Option<Self> {
        let abi = MatrixAbi::from_launch(launch)?;
        let mut entries = Vec::new();
        entries.try_reserve_exact(CLASSES.len()).ok()?;
        for &key in CLASSES {
            entries.push((key, class(key.entry, launch, key.geometry(), key.staged)?));
        }
        entries.sort_unstable_by_key(|(key, _)| *key);
        Some(Self { abi, entries })
    }
    pub(super) fn get(
        &self,
        entry: &'static str,
        launch: LinearLaunch,
        geometry: Grid,
        staged: bool,
    ) -> Option<SelectedAlgorithmClassV1> {
        if MatrixAbi::from_launch(launch)? != self.abi {
            return None;
        }
        let key = ClassKey::new(
            entry,
            geometry.threads,
            geometry.threadgroup_memory.unwrap_or(0),
            staged,
        );
        let index = self
            .entries
            .binary_search_by_key(&key, |(key, _)| *key)
            .ok()?;
        Some(self.entries[index].1)
    }
}

/// Class vocabulary for one plain SwiGLU node. Matrix extents/partition ABI
/// stay bound to that node; current row counts and selected PSOs stay dynamic.
pub(in crate::backend::metal::vnext_ops::linear) struct PreparedSwiGluClasses {
    pub(super) gate: Vec<PreparedLinearClasses>,
    pub(super) down: PreparedLinearClasses,
    activation_abi: MatrixAbi,
    activation_key: ClassKey,
    activation: SelectedAlgorithmClassV1,
}
impl PreparedSwiGluClasses {
    pub(in crate::backend::metal::vnext_ops::linear) fn new(
        gate: &[LinearLaunch],
        down: LinearLaunch,
        gate_up_stride: u64,
    ) -> Option<Self> {
        if gate.is_empty() {
            return None;
        }
        let gate = gate
            .iter()
            .map(|&launch| PreparedLinearClasses::new(launch))
            .collect::<Option<Vec<_>>>()?;
        let mut activation_abi = down;
        activation_abi.params.output_stride = u32::try_from(gate_up_stride).ok()?;
        activation_abi.params.output_column_offset = 0;
        let activation_key = ClassKey::new(
            SWIGLU_KERNEL,
            [u32::try_from(THREADS_PER_GROUP).ok()?, 1, 1],
            0,
            false,
        );
        Some(Self {
            gate,
            down: PreparedLinearClasses::new(down)?,
            activation_abi: MatrixAbi::from_launch(activation_abi)?,
            activation_key,
            activation: class(
                SWIGLU_KERNEL,
                activation_abi,
                activation_key.geometry(),
                false,
            )?,
        })
    }

    pub(super) fn activation(
        &self,
        launch: LinearLaunch,
        geometry: Grid,
    ) -> Option<SelectedAlgorithmClassV1> {
        let key = ClassKey::new(
            SWIGLU_KERNEL,
            geometry.threads,
            geometry.threadgroup_memory.unwrap_or(0),
            false,
        );
        (MatrixAbi::from_launch(launch)? == self.activation_abi && key == self.activation_key)
            .then_some(self.activation)
    }
}

#[cfg(test)]
mod tests;
