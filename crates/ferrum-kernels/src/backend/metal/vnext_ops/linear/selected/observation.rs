//! CPU-only frozen choices from the actual immutable Metal selector.
use super::*;
use ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown;
use ferrum_interfaces::vnext::{DeviceObservationTemplate, FrozenObservationInput};

#[derive(Clone, Copy)]
enum Stage {
    Plain {
        entry: &'static str,
        kind: LinearDispatchKind,
        launch: LinearLaunch,
    },
    Staged(LinearLaunch),
    Activation {
        launch: SwiGluLaunch,
        abi: LinearLaunch,
    },
}
pub(in crate::backend::metal::vnext_ops) struct FrozenLinear {
    stages: Vec<Stage>,
}
impl FrozenLinear {
    pub(in crate::backend::metal::vnext_ops) fn variable_upper(
        dispatches: u64,
        chains: usize,
    ) -> Option<usize> {
        usize::try_from(dispatches)
            .ok()?
            .checked_mul(4)?
            .checked_add(chains.checked_mul(8)?)?
            .checked_mul(std::mem::size_of::<Stage>())
    }
    pub(in crate::backend::metal::vnext_ops) fn empty() -> Self {
        Self { stages: Vec::new() }
    }
    fn push(&mut self, stage: Stage) -> Option<()> {
        if self.stages.len() >= ferrum_interfaces::execution_cost::MAX_COST_COMMANDS {
            return None;
        }
        self.stages.try_reserve(1).ok()?;
        self.stages.push(stage);
        Some(())
    }
    fn plain(&mut self, p: &MetalLinearPipelines, launch: LinearLaunch) -> Option<()> {
        let (actual, kind) =
            p.plain_linear_dispatch(launch.format, launch.activation_type, launch.params);
        // Record the actual catalog entry and dispatch kind. Geometry, work,
        // hashes and aggregation are computed only by append on the worker.
        self.push(Stage::Plain {
            entry: entry(p, actual)?,
            kind,
            launch,
        })
    }
    pub(in crate::backend::metal::vnext_ops) fn projection(
        &mut self,
        p: &MetalLinearPipelines,
        launch: LinearLaunch,
        policy: Option<staged_prefill::StagingPolicy>,
    ) -> Option<()> {
        if launch.transform.is_some() {
            return None;
        }
        if policy.is_some_and(|policy| staged_prefill::selected_for(launch, policy)) {
            self.push(Stage::Staged(launch))
        } else if let Some(parts) = launch.plain_plan.grouped_parts(launch) {
            parts.into_iter().try_for_each(|part| self.plain(p, part))
        } else if let Some(parts) = launch.plain_plan.parts(launch) {
            parts.into_iter().try_for_each(|part| self.plain(p, part))
        } else {
            self.plain(p, launch)
        }
    }
    pub(in crate::backend::metal::vnext_ops) fn occurrences(&self) -> Option<u64> {
        self.stages.iter().try_fold(0u64, |n, s| {
            n.checked_add(if matches!(s, Stage::Staged(_)) { 2 } else { 1 })
        })
    }
    pub(in crate::backend::metal::vnext_ops) fn variable_bytes(&self) -> Option<usize> {
        self.stages
            .capacity()
            .checked_mul(std::mem::size_of::<Stage>())
    }
    pub(in crate::backend::metal::vnext_ops) fn append(
        &self,
        b: &mut SelectedCommandCostBuilderV1,
        scratch: u64,
    ) -> Option<()> {
        for stage in &self.stages {
            match *stage {
                Stage::Plain {
                    entry,
                    kind,
                    launch,
                } => {
                    let geometry = grid(launch.params, kind)?;
                    append_plain(b, entry, launch, geometry, false, scratch)?;
                }
                Stage::Staged(launch) => append_staged(b, launch, scratch)?,
                Stage::Activation { launch, abi } => append_activation(b, launch, abi, scratch)?,
            }
        }
        Some(())
    }
}
fn append_plain(
    b: &mut SelectedCommandCostBuilderV1,
    entry: &'static str,
    launch: LinearLaunch,
    geometry: Grid,
    staged: bool,
    scratch: u64,
) -> Option<()> {
    b.kernel(
        class(entry, launch, geometry, staged)?,
        KernelNumericWorkV1 {
            logical_units: u64::from(launch.params.rows)
                .checked_mul(u64::from(launch.params.out_features))?,
            padded_units: geometry.padded_outputs,
            inner_units_per_logical_unit: u64::from(launch.params.in_features),
            grid: geometry.groups,
            scratch_bytes: scratch,
            staged_weight_bytes: 0,
        },
    )
    .ok()
}
fn append_staged(
    b: &mut SelectedCommandCostBuilderV1,
    launch: LinearLaunch,
    scratch: u64,
) -> Option<()> {
    let elements =
        u64::from(launch.params.in_features).checked_mul(u64::from(launch.params.out_features))?;
    let blocks = elements.checked_div(256)?;
    let entry = match launch.format {
        LinearPhysicalFormat::Q4K => "stage_q4k_f16",
        LinearPhysicalFormat::Q5K => "stage_q5k_f16",
        LinearPhysicalFormat::Q6K => "stage_q6k_f16",
        _ => return None,
    };
    let geometry = Grid {
        groups: [u32::try_from(blocks.div_ceil(8)).ok()?, 1, 1],
        threads: [128, 1, 1],
        threadgroup_memory: Some(0),
        padded_outputs: blocks.div_ceil(8).checked_mul(8)?.checked_mul(256)?,
    };
    b.kernel(
        class(entry, launch, geometry, true)?,
        KernelNumericWorkV1 {
            logical_units: elements,
            padded_units: geometry.padded_outputs,
            inner_units_per_logical_unit: 1,
            grid: geometry.groups,
            scratch_bytes: scratch,
            staged_weight_bytes: elements.checked_mul(2)?,
        },
    )
    .ok()?;
    append_plain(
        b,
        "gemm_f16a_f16w_tiled",
        launch,
        grid(launch.params, LinearDispatchKind::TiledGemm)?,
        true,
        scratch,
    )
}
fn append_activation(
    b: &mut SelectedCommandCostBuilderV1,
    activation: SwiGluLaunch,
    mut abi: LinearLaunch,
    scratch: u64,
) -> Option<()> {
    let units = u64::from(activation.params.rows)
        .checked_mul(u64::from(activation.params.intermediate_size))?;
    let geometry = Grid {
        groups: [u32::try_from(units.div_ceil(THREADS_PER_GROUP)).ok()?, 1, 1],
        threads: [u32::try_from(THREADS_PER_GROUP).ok()?, 1, 1],
        threadgroup_memory: None,
        padded_outputs: units
            .div_ceil(THREADS_PER_GROUP)
            .checked_mul(THREADS_PER_GROUP)?,
    };
    abi.params.output_stride = activation.params.gate_up_stride;
    abi.params.output_column_offset = 0;
    b.kernel(
        class(SWIGLU_KERNEL, abi, geometry, false)?,
        KernelNumericWorkV1 {
            logical_units: units,
            padded_units: geometry.padded_outputs,
            inner_units_per_logical_unit: 1,
            grid: geometry.groups,
            scratch_bytes: scratch,
            staged_weight_bytes: 0,
        },
    )
    .ok()
}
struct Template {
    chain: FrozenLinear,
    tokens: u64,
    scratch: u64,
    capture: ferrum_types::SloStructuredCostCapture,
}
pub(in crate::backend::metal::vnext_ops) fn payload_upper(dispatches: u64) -> Option<usize> {
    std::mem::size_of::<Template>()
        .checked_add(2 * std::mem::size_of::<usize>())?
        .checked_add(FrozenLinear::variable_upper(dispatches, 1)?)
}
impl DeviceObservationTemplate for Template {
    fn command_count(&self) -> usize {
        1
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.chain.variable_bytes()?)?
            .checked_add(2 * std::mem::size_of::<usize>())
    }
    fn projection_retained_bytes_upper_bound(&self) -> Option<usize> {
        SelectedCommandCostEvidenceV1::maximum_working_payload_bytes(self.chain.occurrences()?)?
            .checked_add(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())
    }
    fn project(
        &self,
        input: &FrozenObservationInput,
    ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown> {
        if input.tokens() != self.tokens
            || !input.participant_ranges().is_empty()
            || !input.source_ranges().is_empty()
        {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        let mut b =
            crate::backend::metal::vnext_runtime::selected_cost_builder(self.capture, self.tokens);
        self.chain
            .append(&mut b, self.scratch)
            .ok_or(StatisticalEvidenceUnknown::Overflow)?;
        Ok(vec![Some(b.finish()?)])
    }
}
pub(in crate::backend::metal::vnext_ops) fn dense(
    p: &MetalLinearPipelines,
    launches: &[LinearLaunch],
    tokens: u64,
) -> Option<Arc<dyn DeviceObservationTemplate>> {
    if launches.is_empty() {
        return None;
    }
    let mut chain = FrozenLinear::empty();
    for &launch in launches {
        chain.projection(p, launch, None)?;
    }
    Some(Arc::new(Template {
        chain,
        tokens,
        scratch: 0,
        capture: p.structured_capture(),
    }))
}
pub(in crate::backend::metal::vnext_ops) fn swiglu(
    p: &MetalLinearPipelines,
    gate: &[LinearLaunch],
    down: LinearLaunch,
    activation: SwiGluLaunch,
    policy: Option<staged_prefill::StagingPolicy>,
    tokens: u64,
    scratch: u64,
) -> Option<Arc<dyn DeviceObservationTemplate>> {
    let mut chain = FrozenLinear::empty();
    for &launch in gate {
        chain.projection(p, launch, policy)?;
    }
    chain.push(Stage::Activation {
        launch: activation,
        abi: down,
    })?;
    chain.projection(p, down, policy)?;
    Some(Arc::new(Template {
        chain,
        tokens,
        scratch,
        capture: p.structured_capture(),
    }))
}
