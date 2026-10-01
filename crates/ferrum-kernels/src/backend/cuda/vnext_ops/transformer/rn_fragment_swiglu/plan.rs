//! One checked selector and launch description for eager, future and replay.
use super::*;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1,
};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Per-projection implementation; the packet ABI and numerical stages are shared.
/// Q4/Q5 retain the original entry and its 4 KiB static shared-memory footprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FragmentKernel {
    GlobalV1,
    Q6PacketPrefetchV1,
}
impl FragmentKernel {
    pub(super) fn for_format(format: RnF16FragmentSourceFormatV1) -> Self {
        match format {
            RnF16FragmentSourceFormatV1::Q4K | RnF16FragmentSourceFormatV1::Q5K => Self::GlobalV1,
            RnF16FragmentSourceFormatV1::Q6K => Self::Q6PacketPrefetchV1,
        }
    }
    pub(super) fn entry(self) -> &'static str {
        match self {
            Self::GlobalV1 => ENTRY,
            Self::Q6PacketPrefetchV1 => Q6_PREFETCH_ENTRY,
        }
    }
    pub(super) fn replay_tag(self) -> u32 {
        match self {
            Self::GlobalV1 => 0,
            Self::Q6PacketPrefetchV1 => 1,
        }
    }
    fn layout_domain(self) -> &'static [u8] {
        match self {
            Self::GlobalV1 => b"cuda.rn-f16-fragment.N16.M8.K32.warp8.f32-reduce.v1",
            Self::Q6PacketPrefetchV1 => {
                b"cuda.rn-f16-fragment.N16.M8.K32.warp8.q6-packet-prefetch.v1"
            }
        }
    }
}
pub(super) fn fragment_function<'a>(
    plan: RnF16FragmentPlanV1,
    global: &'a CudaFunction,
    q6_prefetch: &'a CudaFunction,
) -> &'a CudaFunction {
    match FragmentKernel::for_format(plan.source_format()) {
        FragmentKernel::GlobalV1 => global,
        FragmentKernel::Q6PacketPrefetchV1 => q6_prefetch,
    }
}

/// Bound-plan metadata only: neither current rows nor resource authority is
/// retained. The typed weight plans have already checked all physical byte spans.
#[derive(Clone, Copy)]
struct PreparedWeights {
    gate: RnF16FragmentPlanV1,
    down: RnF16FragmentPlanV1,
    algorithms: Option<[SelectedAlgorithmClassV1; 2]>,
}
impl PreparedWeights {
    fn new(gate: RnF16FragmentPlanV1, down: RnF16FragmentPlanV1) -> Self {
        Self {
            gate,
            down,
            algorithms: None,
        }
    }

    fn prepare_algorithms(mut self) -> Result<Self, String> {
        self.algorithms = Some([
            fragment_algorithm(self.gate)?,
            fragment_algorithm(self.down)?,
        ]);
        Ok(self)
    }

    #[cfg(test)]
    fn from_axes(
        hidden: u64,
        intermediate: u64,
        gate_format: RnF16FragmentSourceFormatV1,
        down_format: RnF16FragmentSourceFormatV1,
    ) -> Result<Self, String> {
        let hidden = checked_i32(hidden, "RN fragment hidden")?;
        let intermediate = checked_i32(intermediate, "RN fragment intermediate")?;
        let width = intermediate
            .checked_mul(2)
            .ok_or("RN fragment gate/up width overflows")?;
        Ok(Self::new(
            checked_weight_plan(gate_format, width as u64, hidden as u64)?,
            checked_weight_plan(down_format, hidden as u64, intermediate as u64)?,
        ))
    }
}

#[cfg(test)]
fn checked_weight_plan(
    format: RnF16FragmentSourceFormatV1,
    n: u64,
    k: u64,
) -> Result<RnF16FragmentPlanV1, String> {
    count_preparation(0);
    RnF16FragmentPlanV1::new(format, n, k).map_err(|e| e.to_string())
}

#[cfg(test)]
std::thread_local! {
    // Actual local weight-plan builds, algorithm-class builds, dynamic instances.
    static PREPARATION_COUNTS: std::cell::Cell<[usize; 3]> =
        const { std::cell::Cell::new([0; 3]) };
}
#[cfg(test)]
fn count_preparation(index: usize) {
    PREPARATION_COUNTS.with(|counts| {
        let mut current = counts.get();
        current[index] += 1;
        counts.set(current);
    });
}

/// Immutable logical axes and independently validated physical weight layouts.
/// Current token extents and the fragment/library choice are never retained.
pub(super) struct PreparedShape {
    hidden: u64,
    intermediate: u64,
    weights: PreparedWeights,
}

impl PreparedShape {
    pub(super) fn from_values(
        values: &[ResolvedValueBinding],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, String> {
        let mut prepared = Self::from_bindings(values, attributes)?;
        prepared.weights = prepared.weights.prepare_algorithms()?;
        Ok(prepared)
    }

    fn from_bindings(
        values: &[ResolvedValueBinding],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, String> {
        let hidden = unsigned_attribute(attributes, "hidden_size")?;
        let intermediate = unsigned_attribute(attributes, "intermediate_size")?;
        let gate = binding(values, ResolvedValueRole::Input, 1)?;
        let down = binding(values, ResolvedValueRole::Input, 2)?;
        validate_dense_swiglu(
            binding(values, ResolvedValueRole::Input, 0)?,
            gate,
            down,
            binding(values, ResolvedValueRole::Output, 0)?,
            hidden,
            intermediate,
        )?;
        Ok(Self {
            hidden,
            intermediate,
            weights: PreparedWeights::new(weights::validate(gate)?, weights::validate(down)?),
        })
    }

    pub(super) fn for_tokens(&self, tokens: u64) -> Result<Shape, String> {
        Shape::with_weights(tokens, self.hidden, self.intermediate, self.weights)
    }
}

#[derive(Clone, Copy)]
pub(in crate::backend::cuda::vnext_ops::transformer) struct Shape {
    pub(super) tokens: u64,
    pub(super) rows: i32,
    pub(super) hidden: i32,
    pub(super) intermediate: i32,
    pub(super) activation_elements: u64,
    pub(super) gate_up_bytes: u64,
    pub(super) scratch_bytes: u64,
    pub(super) gate_weight: RnF16FragmentPlanV1,
    pub(super) down_weight: RnF16FragmentPlanV1,
    fragment_algorithms: Option<[SelectedAlgorithmClassV1; 2]>,
    pub(super) gate: GemmF16ApiPlan,
    pub(super) down: GemmF16ApiPlan,
}
impl Shape {
    #[cfg(test)]
    pub(super) fn new(
        tokens: u64,
        hidden: u64,
        intermediate: u64,
        gate_format: RnF16FragmentSourceFormatV1,
        down_format: RnF16FragmentSourceFormatV1,
    ) -> Result<Self, String> {
        let weights = PreparedWeights::from_axes(hidden, intermediate, gate_format, down_format)?;
        Self::with_weights(tokens, hidden, intermediate, weights)
    }

    fn with_weights(
        tokens: u64,
        hidden: u64,
        intermediate: u64,
        weights: PreparedWeights,
    ) -> Result<Self, String> {
        #[cfg(test)]
        count_preparation(2);
        let activation_elements = tokens
            .checked_mul(intermediate)
            .ok_or("RN fragment activation count overflows")?;
        let gate_up_bytes = activation_elements
            .checked_mul(4)
            .ok_or("RN fragment gate/up bytes overflow")?;
        let scratch_bytes = activation_elements
            .checked_mul(6)
            .ok_or("RN fragment scratch overflows")?;
        let rows = checked_i32(tokens, "RN fragment physical M")?;
        let hidden = checked_i32(hidden, "RN fragment hidden")?;
        let intermediate = checked_i32(intermediate, "RN fragment intermediate")?;
        let width = intermediate
            .checked_mul(2)
            .ok_or("RN fragment gate/up width overflows")?;
        silu_mul_launch_config(activation_elements).map_err(|e| e.to_string())?;
        // Typed plans cannot contain unchecked spans. Rechecking their complete
        // logical axes is sufficient; reconstructing those same plans is not.
        if weights.gate.n() != width as u64
            || weights.gate.k() != hidden as u64
            || weights.down.n() != hidden as u64
            || weights.down.k() != intermediate as u64
        {
            return Err("RN fragment physical weights differ from logical FFN axes".into());
        }
        Ok(Self {
            tokens,
            rows,
            hidden,
            intermediate,
            activation_elements,
            gate_up_bytes,
            scratch_bytes,
            gate_weight: weights.gate,
            down_weight: weights.down,
            fragment_algorithms: weights.algorithms,
            gate: GemmF16ApiPlan::new(rows, width, hidden).map_err(|e| e.to_string())?,
            down: GemmF16ApiPlan::new(rows, hidden, intermediate).map_err(|e| e.to_string())?,
        })
    }
    pub(super) fn from_values(
        values: &[ResolvedValueBinding],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
        tokens: u64,
    ) -> Result<Self, String> {
        // Direct encoding preserves lazy evidence construction when capture is disabled.
        PreparedShape::from_bindings(values, attributes)?.for_tokens(tokens)
    }
    pub(super) fn fragment(self) -> bool {
        (1..=8).contains(&self.tokens)
    }
    pub(in crate::backend::cuda::vnext_ops::transformer) fn project(
        self,
        tokens: u64,
        identity: Option<CublasHandleApiIdentity>,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        if tokens != self.tokens {
            return None;
        }
        self.selected(SloStructuredCostCapture::HostSettledV1, identity)
    }
    pub(super) fn selected(
        self,
        capture: SloStructuredCostCapture,
        identity: Option<CublasHandleApiIdentity>,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        let mut builder =
            crate::backend::cuda::vnext_runtime::selected_cost::builder(capture, self.tokens)?;
        if self.fragment() {
            append_fragment(
                &mut builder,
                self.tokens,
                self.gate_weight,
                self.fragment_algorithms.map(|algorithms| algorithms[0]),
                self.scratch_bytes,
            )?;
        } else {
            self.gate.append_selected(&mut builder, identity?).ok()?;
        }
        native_swiglu::append_silu(
            &mut builder,
            self.rows as u32,
            self.intermediate as u32,
            self.scratch_bytes,
        )?;
        if self.fragment() {
            append_fragment(
                &mut builder,
                self.tokens,
                self.down_weight,
                self.fragment_algorithms.map(|algorithms| algorithms[1]),
                self.scratch_bytes,
            )?;
        } else {
            self.down.append_selected(&mut builder, identity?).ok()?;
        }
        builder.finish().ok()
    }
}

pub(super) fn format_code(format: RnF16FragmentSourceFormatV1) -> u32 {
    match format {
        RnF16FragmentSourceFormatV1::Q4K => 12,
        RnF16FragmentSourceFormatV1::Q5K => 13,
        RnF16FragmentSourceFormatV1::Q6K => 14,
    }
}

#[cfg(test)]
mod prepared_tests;

pub(super) fn launch_config(rows: u32, plan: RnF16FragmentPlanV1) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((plan.n() as u32).div_ceil(16), rows.div_ceil(8), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn fragment_algorithm(plan: RnF16FragmentPlanV1) -> Result<SelectedAlgorithmClassV1, String> {
    #[cfg(test)]
    count_preparation(1);
    static PTX: OnceLock<[u8; 32]> = OnceLock::new();
    let kernel = FragmentKernel::for_format(plan.source_format());
    let mut layout = Sha256::new();
    layout.update(kernel.layout_domain());
    layout.update(plan.packing_abi().to_le_bytes());
    layout.update(format_code(plan.source_format()).to_le_bytes());
    SelectedAlgorithmClassV1::new(
        kernel.entry(),
        1,
        *PTX.get_or_init(|| Sha256::digest(crate::ptx::VNEXT_GGUF.as_bytes()).into()),
        layout.finalize().into(),
    )
    .map_err(|_| "RN fragment algorithm identity is invalid".into())
}

fn append_fragment(
    builder: &mut SelectedCommandCostBuilderV1,
    rows: u64,
    plan: RnF16FragmentPlanV1,
    algorithm: Option<SelectedAlgorithmClassV1>,
    scratch: u64,
) -> Option<()> {
    if !(1..=8).contains(&rows) {
        return None;
    }
    let config = launch_config(rows as u32, plan);
    let fixed = [
        rows,
        plan.k(),
        plan.n(),
        plan.n(),
        0,
        u64::from(format_code(plan.source_format())),
        u64::from(plan.packing_abi()),
        plan.packed_bytes(),
    ];
    builder
        .kernel_with_replay_geometry(
            algorithm.or_else(|| fragment_algorithm(plan).ok())?,
            KernelNumericWorkV1 {
                logical_units: rows.checked_mul(plan.n())?,
                padded_units: u64::from(config.grid_dim.0).checked_mul(16 * 8)?,
                inner_units_per_logical_unit: plan.k(),
                grid: [config.grid_dim.0, config.grid_dim.1, 1],
                scratch_bytes: scratch,
                staged_weight_bytes: 0,
            },
            KernelReplayGeometryV1 {
                block: [256, 1, 1],
                dynamic_shared_bytes: 0,
                fixed_parameters: &fixed,
            },
        )
        .ok()
}

pub(super) fn launch(
    stream: &CudaStream,
    function: &CudaFunction,
    shape: Shape,
    plan: RnF16FragmentPlanV1,
    input: u64,
    packed: u64,
    output: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    if !shape.fragment() {
        return Err(CudaDeviceRuntimeError::contract(
            "RN fragment kernel requires whole physical M=1..8",
        ));
    }
    let (rows, k, n, stride, offset, format, abi, bytes) = (
        shape.rows as u32,
        plan.k() as u32,
        plan.n() as u32,
        plan.n() as u32,
        0u32,
        format_code(plan.source_format()),
        plan.packing_abi(),
        plan.packed_bytes(),
    );
    let mut launch = stream.launch_builder(function);
    launch
        .arg(&input)
        .arg(&packed)
        .arg(&bytes)
        .arg(&output)
        .arg(&rows)
        .arg(&k)
        .arg(&n)
        .arg(&stride)
        .arg(&offset)
        .arg(&format)
        .arg(&abi);
    // SAFETY: prepare retains exact validated weights, packed input/output and
    // admitted scratch. The checked plan bounds every launch dimension and span.
    unsafe { launch.launch(launch_config(rows, plan)) }
        .map_err(|e| CudaDeviceRuntimeError::driver("RN fragment MMA launch", e))?;
    Ok(())
}
