//! Invocation-scoped CUDA workspace geometry for routed/shared SwiGLU MoE.

const ALIGNMENT_BYTES: u64 = 16;
const F16_BYTES: u64 = 2;
const I32_BYTES: u64 = 4;
const F32_BYTES: u64 = 4;
const MARLIN_MIN_THREAD_N: u64 = 64;
const MARLIN_MAX_THREAD_N: u64 = 256;
const MARLIN_BLOCKS_PER_SM_BOUND: u64 = 4;
const REGION_COUNT: u64 = 15;

pub(super) const MOE_BLOCK_SIZE: u64 = 16;
pub(super) const MAX_ROUTER_EXPERTS: u64 = 256;
pub(super) const MAX_ROUTER_TOP_K: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WorkspaceRegion {
    offset_bytes: u64,
    length_bytes: u64,
}

impl WorkspaceRegion {
    pub(super) const fn offset_bytes(self) -> u64 {
        self.offset_bytes
    }

    pub(super) const fn length_bytes(self) -> u64 {
        self.length_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MoeWorkspaceLayout {
    pub(super) router_logits: WorkspaceRegion,
    pub(super) route_ids: WorkspaceRegion,
    pub(super) route_weights: WorkspaceRegion,
    pub(super) sorted_token_ids: WorkspaceRegion,
    pub(super) expert_block_ids: WorkspaceRegion,
    pub(super) total_tokens_post_pad: WorkspaceRegion,
    pub(super) marlin_workspace: WorkspaceRegion,
    pub(super) marlin_c_tmp: WorkspaceRegion,
    pub(super) routed_gate_up: WorkspaceRegion,
    pub(super) routed_activation: WorkspaceRegion,
    pub(super) routed_down_slots: WorkspaceRegion,
    pub(super) shared_gate: WorkspaceRegion,
    pub(super) shared_gate_up: WorkspaceRegion,
    pub(super) shared_activation: WorkspaceRegion,
    pub(super) shared_output: WorkspaceRegion,
    pub(super) total_bytes: u64,
    pub(super) pair_count: u64,
    pub(super) sorted_capacity: u64,
    pub(super) block_capacity: u64,
}

impl MoeWorkspaceLayout {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        tokens: u64,
        expert_count: u64,
        experts_per_token: u64,
        hidden_size: u64,
        routed_intermediate_size: u64,
        shared_intermediate_size: u64,
        multiprocessor_count: u64,
    ) -> Result<Self, String> {
        validate_shape(
            tokens,
            expert_count,
            experts_per_token,
            hidden_size,
            routed_intermediate_size,
            shared_intermediate_size,
            multiprocessor_count,
        )?;
        let pair_count = checked_mul(tokens, experts_per_token, "MoE pair count")?;
        let expert_padding = checked_mul(expert_count, MOE_BLOCK_SIZE, "MoE expert padding")?;
        let sorted_capacity = checked_add(pair_count, expert_padding, "MoE sorted capacity")?;
        let pair_blocks = div_ceil(pair_count, MOE_BLOCK_SIZE)?;
        let block_capacity = checked_add(expert_count, pair_blocks, "MoE block capacity")?;
        let maximum_n =
            checked_mul(routed_intermediate_size, 2, "MoE gate/up width")?.max(hidden_size);
        let maximum_n_tiles = maximum_n / MARLIN_MIN_THREAD_N;
        let marlin_workspace_elements =
            checked_mul(maximum_n_tiles, block_capacity, "Marlin-MoE lock workspace")?;
        let marlin_c_tmp_elements = multiprocessor_count
            .checked_mul(MARLIN_BLOCKS_PER_SM_BOUND)
            .and_then(|value| value.checked_mul(MOE_BLOCK_SIZE))
            .and_then(|value| value.checked_mul(MARLIN_MAX_THREAD_N))
            .ok_or_else(|| "Marlin-MoE FP32 reduction workspace overflows u64".to_owned())?;

        let mut cursor = WorkspaceCursor::default();
        let router_logits = cursor.allocate(elements_bytes(
            checked_mul(tokens, expert_count, "MoE router logits")?,
            F16_BYTES,
            "MoE router logits",
        )?)?;
        let route_ids = cursor.allocate(elements_bytes(pair_count, I32_BYTES, "MoE route ids")?)?;
        let route_weights =
            cursor.allocate(elements_bytes(pair_count, F32_BYTES, "MoE route weights")?)?;
        let sorted_token_ids = cursor.allocate(elements_bytes(
            sorted_capacity,
            I32_BYTES,
            "MoE sorted token ids",
        )?)?;
        let expert_block_ids = cursor.allocate(elements_bytes(
            block_capacity,
            I32_BYTES,
            "MoE expert block ids",
        )?)?;
        let total_tokens_post_pad = cursor.allocate(I32_BYTES)?;
        let marlin_workspace = cursor.allocate(elements_bytes(
            marlin_workspace_elements,
            I32_BYTES,
            "Marlin-MoE locks",
        )?)?;
        let marlin_c_tmp = cursor.allocate(elements_bytes(
            marlin_c_tmp_elements,
            F32_BYTES,
            "Marlin-MoE FP32 reduction",
        )?)?;
        let routed_gate_up = cursor.allocate(elements_bytes(
            checked_mul(
                checked_mul(pair_count, routed_intermediate_size, "MoE routed gate/up")?,
                2,
                "MoE routed gate/up",
            )?,
            F16_BYTES,
            "MoE routed gate/up",
        )?)?;
        let routed_activation = cursor.allocate(elements_bytes(
            checked_mul(
                pair_count,
                routed_intermediate_size,
                "MoE routed activation",
            )?,
            F16_BYTES,
            "MoE routed activation",
        )?)?;
        let routed_down_slots = cursor.allocate(elements_bytes(
            checked_mul(pair_count, hidden_size, "MoE routed down slots")?,
            F16_BYTES,
            "MoE routed down slots",
        )?)?;
        let shared_gate = cursor.allocate(elements_bytes(tokens, F16_BYTES, "MoE shared gate")?)?;
        let shared_gate_up = cursor.allocate(elements_bytes(
            checked_mul(
                checked_mul(tokens, shared_intermediate_size, "MoE shared gate/up")?,
                2,
                "MoE shared gate/up",
            )?,
            F16_BYTES,
            "MoE shared gate/up",
        )?)?;
        let shared_activation = cursor.allocate(elements_bytes(
            checked_mul(tokens, shared_intermediate_size, "MoE shared activation")?,
            F16_BYTES,
            "MoE shared activation",
        )?)?;
        let shared_output = cursor.allocate(elements_bytes(
            checked_mul(tokens, hidden_size, "MoE shared output")?,
            F16_BYTES,
            "MoE shared output",
        )?)?;
        let total_bytes = align_up(cursor.offset, ALIGNMENT_BYTES)?;

        let (fixed_bytes, bytes_per_token) = workspace_formula_terms(
            expert_count,
            experts_per_token,
            hidden_size,
            routed_intermediate_size,
            shared_intermediate_size,
            multiprocessor_count,
        )?;
        let admitted = fixed_bytes
            .checked_add(checked_mul(
                bytes_per_token,
                tokens,
                "MoE admitted workspace",
            )?)
            .ok_or_else(|| "MoE admitted workspace overflows u64".to_owned())?;
        if total_bytes > admitted {
            return Err(format!(
                "MoE workspace layout {total_bytes} exceeds affine estimate {admitted}"
            ));
        }

        Ok(Self {
            router_logits,
            route_ids,
            route_weights,
            sorted_token_ids,
            expert_block_ids,
            total_tokens_post_pad,
            marlin_workspace,
            marlin_c_tmp,
            routed_gate_up,
            routed_activation,
            routed_down_slots,
            shared_gate,
            shared_gate_up,
            shared_activation,
            shared_output,
            total_bytes,
            pair_count,
            sorted_capacity,
            block_capacity,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn workspace_formula_terms(
    expert_count: u64,
    experts_per_token: u64,
    hidden_size: u64,
    routed_intermediate_size: u64,
    shared_intermediate_size: u64,
    multiprocessor_count: u64,
) -> Result<(u64, u64), String> {
    validate_shape(
        1,
        expert_count,
        experts_per_token,
        hidden_size,
        routed_intermediate_size,
        shared_intermediate_size,
        multiprocessor_count,
    )?;
    let blocks_per_token = div_ceil(experts_per_token, MOE_BLOCK_SIZE)?;
    let maximum_n = checked_mul(routed_intermediate_size, 2, "MoE gate/up width")?.max(hidden_size);
    let maximum_n_tiles = maximum_n / MARLIN_MIN_THREAD_N;

    let fixed_terms = [
        checked_mul(
            checked_mul(expert_count, MOE_BLOCK_SIZE, "MoE sorted fixed elements")?,
            I32_BYTES,
            "MoE sorted fixed bytes",
        )?,
        checked_mul(expert_count, I32_BYTES, "MoE block fixed bytes")?,
        I32_BYTES,
        checked_mul(
            checked_mul(
                expert_count,
                maximum_n_tiles,
                "Marlin-MoE fixed lock elements",
            )?,
            I32_BYTES,
            "Marlin-MoE fixed lock bytes",
        )?,
        multiprocessor_count
            .checked_mul(MARLIN_BLOCKS_PER_SM_BOUND)
            .and_then(|value| value.checked_mul(MOE_BLOCK_SIZE))
            .and_then(|value| value.checked_mul(MARLIN_MAX_THREAD_N))
            .and_then(|value| value.checked_mul(F32_BYTES))
            .ok_or_else(|| "Marlin-MoE fixed FP32 workspace overflows u64".to_owned())?,
    ];
    let mut fixed_bytes = checked_sum(fixed_terms, "MoE fixed workspace")?;
    fixed_bytes = checked_add(
        fixed_bytes,
        checked_mul(REGION_COUNT + 1, ALIGNMENT_BYTES, "MoE alignment reserve")?,
        "MoE fixed workspace alignment",
    )?;

    let routed_gate_up_bytes = experts_per_token
        .checked_mul(routed_intermediate_size)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(F16_BYTES))
        .ok_or_else(|| "MoE routed gate/up bytes per token overflow u64".to_owned())?;
    let routed_activation_bytes = experts_per_token
        .checked_mul(routed_intermediate_size)
        .and_then(|value| value.checked_mul(F16_BYTES))
        .ok_or_else(|| "MoE routed activation bytes per token overflow u64".to_owned())?;
    let routed_down_bytes = experts_per_token
        .checked_mul(hidden_size)
        .and_then(|value| value.checked_mul(F16_BYTES))
        .ok_or_else(|| "MoE routed down bytes per token overflow u64".to_owned())?;
    let shared_gate_up_bytes = shared_intermediate_size
        .checked_mul(2)
        .and_then(|value| value.checked_mul(F16_BYTES))
        .ok_or_else(|| "MoE shared gate/up bytes per token overflow u64".to_owned())?;
    let shared_activation_bytes = checked_mul(
        shared_intermediate_size,
        F16_BYTES,
        "MoE shared activation bytes per token",
    )?;
    let shared_output_bytes =
        checked_mul(hidden_size, F16_BYTES, "MoE shared output bytes per token")?;
    let per_token_terms = [
        checked_mul(expert_count, F16_BYTES, "MoE router bytes per token")?,
        checked_mul(experts_per_token, I32_BYTES, "MoE route ids per token")?,
        checked_mul(experts_per_token, F32_BYTES, "MoE route weights per token")?,
        checked_mul(experts_per_token, I32_BYTES, "MoE sorted ids per token")?,
        checked_mul(blocks_per_token, I32_BYTES, "MoE blocks per token")?,
        blocks_per_token
            .checked_mul(maximum_n_tiles)
            .and_then(|value| value.checked_mul(I32_BYTES))
            .ok_or_else(|| "Marlin-MoE lock bytes per token overflow u64".to_owned())?,
        routed_gate_up_bytes,
        routed_activation_bytes,
        routed_down_bytes,
        F16_BYTES,
        shared_gate_up_bytes,
        shared_activation_bytes,
        shared_output_bytes,
    ];
    let bytes_per_token = checked_sum(per_token_terms, "MoE bytes per token")?;
    Ok((fixed_bytes, bytes_per_token))
}

#[allow(clippy::too_many_arguments)]
fn validate_shape(
    tokens: u64,
    expert_count: u64,
    experts_per_token: u64,
    hidden_size: u64,
    routed_intermediate_size: u64,
    shared_intermediate_size: u64,
    multiprocessor_count: u64,
) -> Result<(), String> {
    if tokens == 0
        || expert_count == 0
        || expert_count > MAX_ROUTER_EXPERTS
        || experts_per_token == 0
        || experts_per_token > expert_count
        || experts_per_token > MAX_ROUTER_TOP_K
        || hidden_size == 0
        || routed_intermediate_size == 0
        || shared_intermediate_size == 0
        || multiprocessor_count == 0
    {
        return Err("CUDA MoE workspace received unsupported zero or routing geometry".to_owned());
    }
    let gate_up_width = checked_mul(routed_intermediate_size, 2, "MoE gate/up width")?;
    if !hidden_size.is_multiple_of(MARLIN_MIN_THREAD_N)
        || !routed_intermediate_size.is_multiple_of(MARLIN_MIN_THREAD_N)
        || !gate_up_width.is_multiple_of(MARLIN_MIN_THREAD_N)
    {
        return Err(format!(
            "CUDA Marlin-MoE H={hidden_size}, R={routed_intermediate_size} must be divisible by {MARLIN_MIN_THREAD_N}"
        ));
    }
    Ok(())
}

#[derive(Debug, Default)]
struct WorkspaceCursor {
    offset: u64,
}

impl WorkspaceCursor {
    fn allocate(&mut self, length_bytes: u64) -> Result<WorkspaceRegion, String> {
        if length_bytes == 0 {
            return Err("CUDA MoE workspace region cannot be empty".to_owned());
        }
        let offset_bytes = align_up(self.offset, ALIGNMENT_BYTES)?;
        self.offset = offset_bytes
            .checked_add(length_bytes)
            .ok_or_else(|| "CUDA MoE workspace cursor overflows u64".to_owned())?;
        Ok(WorkspaceRegion {
            offset_bytes,
            length_bytes,
        })
    }
}

fn elements_bytes(elements: u64, bytes_per_element: u64, label: &str) -> Result<u64, String> {
    checked_mul(elements, bytes_per_element, label)
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{label} overflows u64"))
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{label} overflows u64"))
}

fn checked_sum<const N: usize>(values: [u64; N], label: &str) -> Result<u64, String> {
    values.into_iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| format!("{label} overflows u64"))
    })
}

fn div_ceil(value: u64, divisor: u64) -> Result<u64, String> {
    value
        .checked_add(divisor - 1)
        .map(|value| value / divisor)
        .ok_or_else(|| "CUDA MoE ceil division overflows u64".to_owned())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| "CUDA MoE workspace alignment overflows u64".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: u64 = 2048;
    const E: u64 = 256;
    const K: u64 = 8;
    const R: u64 = 512;
    const S: u64 = 512;
    const SMS: u64 = 128;

    #[test]
    fn qwen35_workspace_is_affine_bounded_for_decode_and_prefill() {
        let (fixed, per_token) = workspace_formula_terms(E, K, H, R, S, SMS).unwrap();
        for tokens in [1, 2, 7, 32, 127, 512, 2048] {
            let layout = MoeWorkspaceLayout::new(tokens, E, K, H, R, S, SMS).unwrap();
            let admitted = fixed + per_token * tokens;
            assert!(layout.total_bytes <= admitted);
            assert_eq!(layout.pair_count, tokens * K);
            assert_eq!(layout.sorted_capacity, tokens * K + E * MOE_BLOCK_SIZE);
            assert_eq!(
                layout.block_capacity,
                E + (tokens * K).div_ceil(MOE_BLOCK_SIZE)
            );
        }
    }

    #[test]
    fn every_region_is_aligned_nonempty_and_nonoverlapping() {
        let layout = MoeWorkspaceLayout::new(32, E, K, H, R, S, SMS).unwrap();
        let regions = [
            layout.router_logits,
            layout.route_ids,
            layout.route_weights,
            layout.sorted_token_ids,
            layout.expert_block_ids,
            layout.total_tokens_post_pad,
            layout.marlin_workspace,
            layout.marlin_c_tmp,
            layout.routed_gate_up,
            layout.routed_activation,
            layout.routed_down_slots,
            layout.shared_gate,
            layout.shared_gate_up,
            layout.shared_activation,
            layout.shared_output,
        ];
        for region in regions {
            assert_eq!(region.offset_bytes() % ALIGNMENT_BYTES, 0);
            assert!(region.length_bytes() > 0);
        }
        for pair in regions.windows(2) {
            assert!(pair[0].offset_bytes() + pair[0].length_bytes() <= pair[1].offset_bytes());
        }
        let last = regions.last().unwrap();
        assert!(last.offset_bytes() + last.length_bytes() <= layout.total_bytes);
    }

    #[test]
    fn fp32_reduce_capacity_matches_the_kernel_sm_bound() {
        let layout = MoeWorkspaceLayout::new(1, E, K, H, R, S, SMS).unwrap();
        assert_eq!(
            layout.marlin_c_tmp.length_bytes(),
            SMS * MARLIN_BLOCKS_PER_SM_BOUND * MOE_BLOCK_SIZE * MARLIN_MAX_THREAD_N * F32_BYTES
        );
    }

    #[test]
    fn rejects_shapes_outside_the_current_kernel_contract() {
        assert!(MoeWorkspaceLayout::new(1, 257, K, H, R, S, SMS).is_err());
        assert!(MoeWorkspaceLayout::new(1, E, 33, H, R, S, SMS).is_err());
        assert!(MoeWorkspaceLayout::new(1, E, K, H + 1, R, S, SMS).is_err());
        assert!(MoeWorkspaceLayout::new(0, E, K, H, R, S, SMS).is_err());
    }
}
