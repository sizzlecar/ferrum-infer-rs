//! One checked selector and launch description for eager, future and replay.
use super::*;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1,
};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

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
    pub(super) gate: GemmF16ApiPlan,
    pub(super) down: GemmF16ApiPlan,
}
impl Shape {
    pub(super) fn new(
        tokens: u64,
        hidden: u64,
        intermediate: u64,
        gate_format: RnF16FragmentSourceFormatV1,
        down_format: RnF16FragmentSourceFormatV1,
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
        silu_mul_launch_config(activation_elements).map_err(|e| e.to_string())?;
        Ok(Self {
            tokens,
            rows,
            hidden,
            intermediate,
            activation_elements,
            gate_up_bytes,
            scratch_bytes,
            gate_weight: RnF16FragmentPlanV1::new(gate_format, width as u64, hidden as u64)
                .map_err(|e| e.to_string())?,
            down_weight: RnF16FragmentPlanV1::new(down_format, hidden as u64, intermediate as u64)
                .map_err(|e| e.to_string())?,
            gate: GemmF16ApiPlan::new(rows, width, hidden).map_err(|e| e.to_string())?,
            down: GemmF16ApiPlan::new(rows, hidden, intermediate).map_err(|e| e.to_string())?,
        })
    }
    pub(super) fn from_values(
        values: &[ResolvedValueBinding],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
        tokens: u64,
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
        let gate_weight = weights::validate(gate)?;
        let down_weight = weights::validate(down)?;
        let shape = Self::new(
            tokens,
            hidden,
            intermediate,
            gate_weight.source_format(),
            down_weight.source_format(),
        )?;
        if shape.gate_weight != gate_weight || shape.down_weight != down_weight {
            return Err("RN fragment physical weights differ from logical FFN axes".into());
        }
        Ok(shape)
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
pub(super) fn launch_config(rows: u32, plan: RnF16FragmentPlanV1) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((plan.n() as u32).div_ceil(16), rows.div_ceil(8), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn append_fragment(
    builder: &mut SelectedCommandCostBuilderV1,
    rows: u64,
    plan: RnF16FragmentPlanV1,
    scratch: u64,
) -> Option<()> {
    if !(1..=8).contains(&rows) {
        return None;
    }
    static PTX: OnceLock<[u8; 32]> = OnceLock::new();
    let config = launch_config(rows as u32, plan);
    let mut layout = Sha256::new();
    layout.update(b"cuda.rn-f16-fragment.N16.M8.K32.warp8.f32-reduce.v1");
    layout.update(plan.packing_abi().to_le_bytes());
    layout.update(format_code(plan.source_format()).to_le_bytes());
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
            SelectedAlgorithmClassV1::new(
                ENTRY,
                1,
                *PTX.get_or_init(|| Sha256::digest(crate::ptx::VNEXT_GGUF.as_bytes()).into()),
                layout.finalize().into(),
            )
            .ok()?,
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
