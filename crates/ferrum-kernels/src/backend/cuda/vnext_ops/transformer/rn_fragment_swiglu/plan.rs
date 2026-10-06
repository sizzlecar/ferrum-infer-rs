//! Checked whole-invocation selector and launch description for eager and replay.
use super::*;
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
    pub(super) fn replay_tag(self) -> u32 {
        match self {
            Self::GlobalV1 => 0,
            Self::Q6PacketPrefetchV1 => 1,
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
}
impl PreparedWeights {
    fn new(gate: RnF16FragmentPlanV1, down: RnF16FragmentPlanV1) -> Self {
        Self { gate, down }
    }
}

/// Immutable logical axes and independently validated physical weight layouts.
/// Current token extents and the fragment/library choice are never retained.
pub(super) struct PreparedShape {
    hidden: u64,
    intermediate: u64,
    weights: PreparedWeights,
}

impl PreparedShape {
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
}
impl Shape {
    fn with_weights(
        tokens: u64,
        hidden: u64,
        intermediate: u64,
        weights: PreparedWeights,
    ) -> Result<Self, String> {
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
        checked_i32(activation_elements, "RN fragment SiLU element count")?;
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
        })
    }
    pub(super) fn from_values(
        values: &[ResolvedValueBinding],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
        tokens: u64,
    ) -> Result<Self, String> {
        PreparedShape::from_bindings(values, attributes)?.for_tokens(tokens)
    }
    pub(super) fn fragment(self) -> bool {
        (1..=8).contains(&self.tokens)
    }
}

pub(super) fn format_code(format: RnF16FragmentSourceFormatV1) -> u32 {
    match format {
        RnF16FragmentSourceFormatV1::Q4K => 12,
        RnF16FragmentSourceFormatV1::Q5K => 13,
        RnF16FragmentSourceFormatV1::Q6K => 14,
    }
}

pub(super) fn launch_config(rows: u32, plan: RnF16FragmentPlanV1) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((plan.n() as u32).div_ceil(16), rows.div_ceil(8), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
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
