//! Explicit G32 ABIs. Numerical selection is retained by the plan;
//! scalar and row-tile-8 kernels consume the same typed activation layout.
use super::{weights, CudaNativeBlockKernels};
use crate::backend::cuda::vnext_runtime::CudaDeviceRuntimeError;
use crate::gguf_blocks::GgufBlockFormat;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    ElementType, PreparedProjection, PreparedProjectionNumerics, Q8ActSwiGluProfile, WeightEncoding,
};
use std::sync::Arc;

// Physical scratch order is a backend implementation detail, not a new
// numerical profile. Never select pack/scalar/tiled symbols independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QwordOrder {
    WordMajor,
    #[cfg(test)]
    GroupMajor,
}

#[derive(Clone, Copy)]
struct KernelSymbols {
    pack: &'static str,
    scalar: &'static str,
    tiled: &'static str,
}

impl KernelSymbols {
    fn entries(self) -> [&'static str; 3] {
        [self.pack, self.scalar, self.tiled]
    }
}

impl QwordOrder {
    const SELECTED: Self = Self::WordMajor;

    fn symbols(self) -> KernelSymbols {
        self.symbols_for(GgufBlockFormat::Iq4Xs)
    }

    fn symbols_for(self, format: GgufBlockFormat) -> KernelSymbols {
        match (self, format) {
            (Self::WordMajor, GgufBlockFormat::Iq4Xs) => KernelSymbols {
                pack: "vnext_gguf_q8_f32scale_pack_word_major_f16_prototype",
                scalar: "vnext_gguf_iq4xs_q8_f32scale_dp4a_group32_word_major_f16_prototype",
                tiled: "vnext_gguf_iq4xs_q8_f32scale_dp4a_group32_word_major_tiled_f16_prototype",
            },
            #[cfg(test)]
            (Self::GroupMajor, GgufBlockFormat::Iq4Xs) => KernelSymbols {
                pack: "vnext_gguf_q8_f32scale_pack_f16_prototype",
                scalar: "vnext_gguf_iq4xs_q8_f32scale_dp4a_group32_f16_prototype",
                tiled: "vnext_gguf_iq4xs_q8_f32scale_dp4a_group32_tiled_f16_prototype",
            },
            (Self::WordMajor, GgufBlockFormat::Q4K) => KernelSymbols {
                pack: "vnext_gguf_q8_f32scale_pack_word_major_f16_prototype",
                scalar: "vnext_gguf_q4k_q8_f32scale_dp4a_group32_word_major_f16_prototype",
                tiled: "vnext_gguf_q4k_q8_f32scale_dp4a_group32_word_major_tiled_f16_prototype",
            },
            (Self::WordMajor, GgufBlockFormat::Q5K) => KernelSymbols {
                pack: "vnext_gguf_q8_f32scale_pack_word_major_f16_prototype",
                scalar: "vnext_gguf_q5k_q8_f32scale_dp4a_group32_word_major_f16_prototype",
                tiled: "vnext_gguf_q5k_q8_f32scale_dp4a_group32_word_major_tiled_f16_prototype",
            },
            _ => unreachable!("only declared Q8act formats have a packed ABI"),
        }
    }
}

fn profile_formats(profile: Q8ActSwiGluProfile) -> &'static [GgufBlockFormat] {
    match profile {
        Q8ActSwiGluProfile::Iq4Xs => &[GgufBlockFormat::Iq4Xs],
        Q8ActSwiGluProfile::Q4KQ5KIq4Xs => &[
            GgufBlockFormat::Q4K,
            GgufBlockFormat::Q5K,
            GgufBlockFormat::Iq4Xs,
        ],
    }
}

pub(in crate::backend::cuda::vnext_ops) fn compiled(ptx: &str) -> bool {
    compiled_for_profile(ptx, Q8ActSwiGluProfile::Iq4Xs)
}

pub(in crate::backend::cuda::vnext_ops) fn compiled_for_profile(
    ptx: &str,
    profile: Q8ActSwiGluProfile,
) -> bool {
    profile_formats(profile).iter().all(|&format| {
        QwordOrder::SELECTED
            .symbols_for(format)
            .entries()
            .iter()
            .all(|name| {
                ptx.lines().any(|line| {
                    let code = line.split("//").next().unwrap_or_default();
                    code.contains(&format!(".entry {name}("))
                })
            })
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::backend::cuda::vnext_ops) struct PackLayout {
    pub scales_bytes: u64,
    pub total_bytes: u64,
}

impl PackLayout {
    pub fn new(rows: u64, k: u64) -> Result<Self, String> {
        if rows == 0 || k == 0 || k % 256 != 0 {
            return Err("Q8act pack requires positive rows and complete K256 blocks".into());
        }
        let values = rows.checked_mul(k).ok_or("Q8act packed extent overflows")?;
        let scales_bytes = values / 8;
        let total_bytes = values
            .checked_add(scales_bytes)
            .ok_or("Q8act workspace extent overflows")?;
        Ok(Self {
            scales_bytes,
            total_bytes,
        })
    }
}

// Only the pack owner creates this view. Equal byte extents do not make the
// group-major and word-major layouts interchangeable; rows/K are also part
// of the physical order. The owner scopes reuse to consumers of the same input
// on the same stream; callers cannot create, clone or retain this view.
pub(in crate::backend::cuda::vnext_ops) struct PackedActivation {
    order: QwordOrder,
    rows: u32,
    k: u32,
    scales: u64,
    words: u64,
    input: u64,
    stream: usize,
}

impl PackedActivation {
    fn new(
        order: QwordOrder,
        rows: u32,
        k: u32,
        workspace: u64,
        workspace_bytes: u64,
    ) -> Result<Self, String> {
        let extent = PackLayout::new(u64::from(rows), u64::from(k))?;
        if workspace == 0 || workspace % 4 != 0 || workspace_bytes < extent.total_bytes {
            return Err("Q8act pack exceeds admitted workspace".into());
        }
        let words = workspace
            .checked_add(extent.scales_bytes)
            .ok_or("Q8act qword pointer overflows")?;
        workspace
            .checked_add(extent.total_bytes)
            .ok_or("Q8act workspace end overflows")?;
        Ok(Self {
            order,
            rows,
            k,
            scales: workspace,
            words,
            input: 0,
            stream: 0,
        })
    }

    fn validate_for(&self, order: QwordOrder, rows: u32, k: u32) -> Result<(), String> {
        if self.order != order || self.rows != rows || self.k != k {
            return Err("Q8act packed input order or shape differs from the kernel ABI".into());
        }
        Ok(())
    }
}

pub(in crate::backend::cuda::vnext_ops) fn workspace_per_token(
    plan: &PreparedProjectionNumerics,
) -> Result<u64, String> {
    plan.projections()
        .iter()
        .filter(|p| p.has_staged_leaf())
        .try_fold(0, |maximum, p| {
            Ok(maximum.max(PackLayout::new(1, p.input_features())?.total_bytes))
        })
}

#[derive(Clone)]
pub(in crate::backend::cuda::vnext_ops) struct Q8ActKernels {
    profile: Q8ActSwiGluProfile,
    order: QwordOrder,
    pack: CudaFunction,
    pairs: Vec<(GgufBlockFormat, KernelPair)>,
}

#[derive(Clone)]
struct KernelPair {
    scalar: CudaFunction,
    tiled: CudaFunction,
}

impl Q8ActKernels {
    /// Attention has its own numerical operation identity. It uses this exact
    /// physical format bundle; the retained attention contract owns routing.
    pub fn load_attention(context: &Arc<CudaContext>) -> Result<Self, CudaDeviceRuntimeError> {
        Self::load_for_profile(context, Q8ActSwiGluProfile::Q4KQ5KIq4Xs)
    }
    pub fn load(context: &Arc<CudaContext>) -> Result<Self, CudaDeviceRuntimeError> {
        Self::load_for_profile(context, Q8ActSwiGluProfile::Iq4Xs)
    }

    pub fn load_for_profile(
        context: &Arc<CudaContext>,
        profile: Q8ActSwiGluProfile,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if !compiled_for_profile(crate::ptx::VNEXT_GGUF, profile) {
            return Err(CudaDeviceRuntimeError::contract(
                "required Q8act G32 kernel exports are missing",
            ));
        }
        let module = context
            .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("Q8act module load", error))?;
        let load = |name| {
            module
                .load_function(name)
                .map_err(|error| CudaDeviceRuntimeError::driver("Q8act function load", error))
        };
        let order = QwordOrder::SELECTED;
        let symbols = order.symbols();
        Ok(Self {
            profile,
            order,
            pack: load(symbols.pack)?,
            pairs: profile_formats(profile)
                .iter()
                .map(|&format| {
                    let symbols = order.symbols_for(format);
                    Ok((
                        format,
                        KernelPair {
                            scalar: load(symbols.scalar)?,
                            tiled: load(symbols.tiled)?,
                        },
                    ))
                })
                .collect::<Result<_, CudaDeviceRuntimeError>>()?,
        })
    }

    pub fn profile(&self) -> Q8ActSwiGluProfile {
        self.profile
    }

    /// Validate the retained native metadata against the plan's static leaf
    /// decisions; no fallback is chosen in response to a launch/load failure.
    pub fn validate_parts(
        projection: &PreparedProjection,
        parts: &[weights::MatrixPart],
    ) -> Result<(), String> {
        Self::validate_parts_for_profile(Q8ActSwiGluProfile::Iq4Xs, projection, parts)
    }

    pub fn validate_parts_for_profile(
        profile: Q8ActSwiGluProfile,
        projection: &PreparedProjection,
        parts: &[weights::MatrixPart],
    ) -> Result<(), String> {
        if parts.len() != projection.leaves().len() {
            return Err("Q8act plan and matrix inventories differ".into());
        }
        for (part, leaf) in parts.iter().zip(projection.leaves()) {
            let format_matches = match (part.format, leaf.encoding()) {
                (
                    weights::MatrixFormat::DenseF16,
                    WeightEncoding::Dense {
                        element_type: ElementType::F16,
                    },
                ) => true,
                (weights::MatrixFormat::Block(actual), WeightEncoding::BlockQuantized(spec)) => {
                    GgufBlockFormat::from_spec(spec)? == actual
                }
                _ => false,
            };
            if !format_matches
                || &part.component_id != leaf.component_id()
                || u64::from(part.columns) != projection.input_features()
                || u64::from(part.rows) != leaf.output_features()
                || u64::from(part.output_offset) != leaf.output_offset()
                || part.transform.is_some() != leaf.has_weight_transform()
                || (leaf.is_staged()
                    && (!matches!(part.format, weights::MatrixFormat::Block(format)
                            if profile_formats(profile).contains(&format))
                        || part.transform.is_some()))
            {
                return Err("Q8act retained matrix differs from the prepared leaf decision".into());
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        strict: &CudaNativeBlockKernels,
        stream: &CudaStream,
        projection: &PreparedProjection,
        parts: &[weights::MatrixPart],
        weights: &[u64],
        input: u64,
        output: u64,
        rows: u32,
        output_stride: u32,
        workspace: u64,
        workspace_bytes: u64,
        transform_scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        self.validate_launch(
            projection,
            parts,
            weights,
            input,
            output,
            rows,
            output_stride,
        )?;
        let k = u32::try_from(projection.input_features())
            .map_err(|_| CudaDeviceRuntimeError::contract("Q8act K exceeds u32"))?;
        self.with_packed_input(
            stream,
            projection.has_staged_leaf(),
            input,
            rows,
            k,
            workspace,
            workspace_bytes,
            |packed| {
                self.launch_validated_with_packed(
                    strict,
                    stream,
                    projection,
                    parts,
                    weights,
                    input,
                    output,
                    rows,
                    output_stride,
                    packed,
                    transform_scratch,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_launch(
        &self,
        projection: &PreparedProjection,
        parts: &[weights::MatrixPart],
        weights: &[u64],
        input: u64,
        output: u64,
        rows: u32,
        output_stride: u32,
    ) -> Result<(), CudaDeviceRuntimeError> {
        Self::validate_parts_for_profile(self.profile, projection, parts)
            .map_err(CudaDeviceRuntimeError::contract)?;
        if rows == 0
            || rows > u16::MAX as u32
            || weights.len() != weights::region_count(parts)
            || input == 0
            || output == 0
            || input % 2 != 0
            || output % 2 != 0
        {
            return Err(CudaDeviceRuntimeError::contract(
                "invalid Q8act pointers, rows or matrix inventory",
            ));
        }
        if parts.iter().any(|part| {
            part.output_offset
                .checked_add(part.rows)
                .is_none_or(|end| end > output_stride)
        }) {
            return Err(CudaDeviceRuntimeError::contract(
                "Q8act output part exceeds stride",
            ));
        }
        Ok(())
    }

    /// Reuse is scoped to one pack and one ordered stream. In particular this
    /// is not an activation cache across waves or distinct normalization inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn with_packed_input<R>(
        &self,
        stream: &CudaStream,
        required: bool,
        input: u64,
        rows: u32,
        k: u32,
        workspace: u64,
        workspace_bytes: u64,
        consume: impl FnOnce(Option<&PackedActivation>) -> Result<R, CudaDeviceRuntimeError>,
    ) -> Result<R, CudaDeviceRuntimeError> {
        let packed = if required {
            if input == 0 || input % 2 != 0 || rows == 0 || rows > u16::MAX as u32 {
                return Err(CudaDeviceRuntimeError::contract(
                    "invalid Q8act pack input or rows",
                ));
            }
            let mut packed = PackedActivation::new(self.order, rows, k, workspace, workspace_bytes)
                .map_err(CudaDeviceRuntimeError::contract)?;
            packed.input = input;
            packed.stream = stream.cu_stream() as usize;
            let groups = u64::from(rows) * u64::from(k / 32);
            let blocks = u32::try_from(groups.div_ceil(4))
                .map_err(|_| CudaDeviceRuntimeError::contract("Q8act pack grid overflows"))?;
            let mut launch = stream.launch_builder(&self.pack);
            launch
                .arg(&input)
                .arg(&packed.scales)
                .arg(&packed.words)
                .arg(&rows)
                .arg(&k);
            // SAFETY: complete typed input and admitted output spans, K256 and
            // positive rows; pack uses one warp per K32 group and guards tails.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| CudaDeviceRuntimeError::driver("Q8act activation pack", error))?;
            Some(packed)
        } else {
            None
        };
        consume(packed.as_ref())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_with_packed(
        &self,
        strict: &CudaNativeBlockKernels,
        stream: &CudaStream,
        projection: &PreparedProjection,
        parts: &[weights::MatrixPart],
        weights: &[u64],
        input: u64,
        output: u64,
        rows: u32,
        output_stride: u32,
        packed: Option<&PackedActivation>,
        transform_scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        self.validate_launch(
            projection,
            parts,
            weights,
            input,
            output,
            rows,
            output_stride,
        )?;
        self.launch_validated_with_packed(
            strict,
            stream,
            projection,
            parts,
            weights,
            input,
            output,
            rows,
            output_stride,
            packed,
            transform_scratch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_validated_with_packed(
        &self,
        strict: &CudaNativeBlockKernels,
        stream: &CudaStream,
        projection: &PreparedProjection,
        parts: &[weights::MatrixPart],
        weights: &[u64],
        input: u64,
        output: u64,
        rows: u32,
        output_stride: u32,
        packed: Option<&PackedActivation>,
        transform_scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let k = u32::try_from(projection.input_features())
            .map_err(|_| CudaDeviceRuntimeError::contract("Q8act K exceeds u32"))?;
        for (index, (part, leaf)) in parts.iter().zip(projection.leaves()).enumerate() {
            if leaf.is_staged() {
                let packed = packed.ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("eligible Q8act leaf has no packed input")
                })?;
                packed
                    .validate_for(self.order, rows, k)
                    .map_err(CudaDeviceRuntimeError::contract)?;
                if packed.input != input || packed.stream != stream.cu_stream() as usize {
                    return Err(CudaDeviceRuntimeError::contract(
                        "Q8act packed input identity or stream differs",
                    ));
                }
                let pair = self
                    .pairs
                    .iter()
                    .find_map(|(format, pair)| {
                        (part.format == weights::MatrixFormat::Block(*format)).then_some(pair)
                    })
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "eligible Q8act leaf is missing its format-specific implementation",
                        )
                    })?;
                let (function, tile) = if rows == 1 {
                    (&pair.scalar, 1)
                } else {
                    (&pair.tiled, 8)
                };
                let mut launch = stream.launch_builder(function);
                launch
                    .arg(&packed.scales)
                    .arg(&packed.words)
                    .arg(&weights[index])
                    .arg(&output)
                    .arg(&rows)
                    .arg(&k)
                    .arg(&part.rows)
                    .arg(&output_stride)
                    .arg(&part.output_offset);
                // SAFETY: retained byte-safe native block rows and packed input,
                // validated part offset/stride; T8 guards partial rows/columns.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (part.rows.div_ceil(4), rows.div_ceil(tile), 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| CudaDeviceRuntimeError::driver("Q8act G32 projection", error))?;
            } else {
                strict.transformed_linear(
                    stream,
                    input,
                    weights[index],
                    output,
                    part,
                    rows,
                    output_stride,
                    ElementType::F16,
                    part.signs_region.map_or(0, |i| weights[i]),
                    transform_scratch,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8act_pack_layout_checks_extent_and_preserves_typed_alignment() {
        for rows in [1, 3, 8, 9, 1025] {
            for k in [256, 5120, 17408] {
                let layout = PackLayout::new(rows, k).unwrap();
                assert_eq!(layout.scales_bytes, rows * (k / 32) * 4);
                assert_eq!(layout.total_bytes - layout.scales_bytes, rows * k);
                assert_eq!(layout.scales_bytes % 4, 0);
                assert_eq!(
                    layout.total_bytes,
                    PackLayout::new(1, k).unwrap().total_bytes * rows
                );
            }
        }
        for (rows, k) in [
            (0, 256),
            (1, 0),
            (1, 255),
            (1, 257),
            (u64::MAX, 256),
            (u64::MAX / 256, 256),
        ] {
            assert!(PackLayout::new(rows, k).is_err());
        }
    }

    #[test]
    fn q8act_capability_requires_all_executable_exports() {
        let entry = |name| format!(".visible .entry {name}(\n) {{ ret; }}\n");
        let selected = QwordOrder::SELECTED.symbols().entries();
        let legacy = QwordOrder::GroupMajor.symbols().entries();
        assert_eq!(QwordOrder::SELECTED, QwordOrder::WordMajor);
        let ptx = selected.map(entry).concat();
        assert!(compiled(&ptx));
        // An old complete ABI or any partially switched pack/scalar/T8 bundle
        // must not advertise the selected physical layout's capability.
        for mask in 0..7 {
            let mixed = (0..3)
                .map(|i| {
                    entry(if mask & (1 << i) == 0 {
                        legacy[i]
                    } else {
                        selected[i]
                    })
                })
                .collect::<String>();
            assert!(!compiled(&mixed), "mixed export triplet mask {mask}");
        }
        for missing in selected {
            let partial = selected
                .into_iter()
                .filter(|name| *name != missing)
                .map(entry)
                .collect::<String>();
            assert!(!compiled(&format!("{partial}// .entry {missing}(\n")));
        }
        assert!(!compiled(""));
    }

    #[test]
    fn q8act_three_format_exports_are_isolated_from_the_legacy_profile() {
        let entry = |name| format!(".visible .entry {name}(\n) {{ ret; }}\n");
        let old = Q8ActSwiGluProfile::Iq4Xs;
        let new = Q8ActSwiGluProfile::Q4KQ5KIq4Xs;
        let exports = profile_formats(new)
            .iter()
            .flat_map(|&format| QwordOrder::SELECTED.symbols_for(format).entries())
            .collect::<std::collections::BTreeSet<_>>();
        let full = exports.iter().map(|name| entry(*name)).collect::<String>();
        assert!(compiled_for_profile(&full, old));
        assert!(compiled_for_profile(&full, new));
        for missing in &exports {
            let partial = exports
                .iter()
                .filter(|name| *name != missing)
                .map(|name| entry(*name))
                .collect::<String>();
            assert!(!compiled_for_profile(&partial, new), "missing {missing}");
            let required_by_old = QwordOrder::SELECTED.symbols().entries().contains(missing);
            assert_eq!(
                compiled_for_profile(&partial, old),
                !required_by_old,
                "independent profile qualification for missing {missing}"
            );
        }
        let legacy = QwordOrder::SELECTED.symbols().entries().map(entry).concat();
        assert!(compiled_for_profile(&legacy, old));
        assert!(!compiled_for_profile(&legacy, new));
    }

    #[test]
    fn q8act_packed_view_rejects_order_shape_and_workspace_mismatch() {
        for (rows, k) in [(1, 256), (3, 5120), (8, 17408), (9, 256)] {
            let extent = PackLayout::new(u64::from(rows), u64::from(k)).unwrap();
            for order in [QwordOrder::WordMajor, QwordOrder::GroupMajor] {
                let packed =
                    PackedActivation::new(order, rows, k, 4096, extent.total_bytes).unwrap();
                assert_eq!(packed.scales, 4096);
                assert_eq!(packed.words, 4096 + extent.scales_bytes);
                assert!(packed.validate_for(order, rows, k).is_ok());
                let other = match order {
                    QwordOrder::WordMajor => QwordOrder::GroupMajor,
                    QwordOrder::GroupMajor => QwordOrder::WordMajor,
                };
                assert!(packed.validate_for(other, rows, k).is_err());
                assert!(packed.validate_for(order, rows + 1, k).is_err());
                assert!(packed.validate_for(order, rows, k + 256).is_err());
                assert!(PackedActivation::new(order, rows, k, 0, extent.total_bytes).is_err());
                assert!(PackedActivation::new(order, rows, k, 4098, extent.total_bytes).is_err());
                assert!(
                    PackedActivation::new(order, rows, k, 4096, extent.total_bytes - 1).is_err()
                );
            }
        }
        // Equal total bytes are insufficient: the row/word/group indexing is
        // different after reshaping and must be repacked, even for one order.
        let extent = PackLayout::new(8, 5120).unwrap();
        assert_eq!(
            extent.total_bytes,
            PackLayout::new(1, 40960).unwrap().total_bytes
        );
        let packed =
            PackedActivation::new(QwordOrder::SELECTED, 8, 5120, 4096, extent.total_bytes).unwrap();
        assert!(packed.validate_for(QwordOrder::SELECTED, 1, 40960).is_err());
        assert!(PackedActivation::new(QwordOrder::SELECTED, 1, 256, u64::MAX - 127, 288).is_err());
    }
}
