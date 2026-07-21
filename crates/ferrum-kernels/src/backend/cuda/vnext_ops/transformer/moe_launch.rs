//! Raw CUDA launches owned by the routed/shared SwiGLU MoE provider.

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use super::super::super::vnext_runtime::{CudaDeviceRuntime, CudaDeviceRuntimeError};
use super::moe_workspace::{WorkspaceRegion, MOE_BLOCK_SIZE};

pub(super) const ROUTER_FUNCTION_NAME: &str = "moe_router_topk_softmax_f16";
pub(super) const ALIGN_FUNCTION_NAME: &str = "moe_align_block_size_pair_ids_f32";
pub(super) const WEIGHTED_SUM_FUNCTION_NAME: &str = "weighted_sum_batched_f16";
pub(super) const TOKEN_GATE_ADD_FUNCTION_NAME: &str = "apply_token_gate_and_add_inplace_f16";
pub(super) const SILU_MUL_FUNCTION_NAME: &str = "fused_silu_mul_interleaved_f16";

#[derive(Clone)]
pub(super) struct MoeCudaKernels {
    router: CudaFunction,
    align: CudaFunction,
    weighted_sum: CudaFunction,
    token_gate_add: CudaFunction,
    silu_mul: CudaFunction,
}

impl MoeCudaKernels {
    pub(super) fn load(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let router_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::MOE_ROUTER.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("MoE router module load", error))?;
        let align_module = runtime
            .context()
            .load_module(Ptx::from_src(
                crate::ptx::MOE_ALIGN_BLOCK_SIZE_PAIR_IDS.to_owned(),
            ))
            .map_err(|error| CudaDeviceRuntimeError::driver("MoE align module load", error))?;
        let combine_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::MOE_COMBINE.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("MoE combine module load", error))?;
        let silu_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("MoE SiLU module load", error))?;
        Ok(Self {
            router: load_function(&router_module, ROUTER_FUNCTION_NAME, "MoE router")?,
            align: load_function(&align_module, ALIGN_FUNCTION_NAME, "MoE align")?,
            weighted_sum: load_function(
                &combine_module,
                WEIGHTED_SUM_FUNCTION_NAME,
                "MoE weighted sum",
            )?,
            token_gate_add: load_function(
                &combine_module,
                TOKEN_GATE_ADD_FUNCTION_NAME,
                "MoE shared gate merge",
            )?,
            silu_mul: load_function(&silu_module, SILU_MUL_FUNCTION_NAME, "MoE SiLU")?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_router(
        &self,
        stream: &CudaStream,
        logits: u64,
        route_ids: u64,
        route_weights: u64,
        tokens: i32,
        expert_count: i32,
        experts_per_token: i32,
        normalize_topk: bool,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let normalize_topk = i32::from(normalize_topk);
        let shared_mem_bytes = u32::try_from(expert_count)
            .ok()
            .and_then(|experts| experts.checked_mul(4))
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("MoE router shared memory overflows")
            })?;
        let grid_x = u32::try_from(tokens)
            .map_err(|_| CudaDeviceRuntimeError::contract("MoE router token count exceeds u32"))?;
        let mut builder = stream.launch_builder(&self.router);
        builder.arg(&logits);
        builder.arg(&route_ids);
        builder.arg(&route_weights);
        builder.arg(&tokens);
        builder.arg(&expert_count);
        builder.arg(&experts_per_token);
        builder.arg(&normalize_topk);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid_x, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("MoE router launch", error))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_align(
        &self,
        stream: &CudaStream,
        route_ids: u64,
        sorted_token_ids: u64,
        expert_block_ids: u64,
        total_tokens_post_pad: u64,
        pair_count: i32,
        expert_count: i32,
        sorted_capacity: i32,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let block_size = MOE_BLOCK_SIZE as i32;
        let mut builder = stream.launch_builder(&self.align);
        builder.arg(&route_ids);
        builder.arg(&sorted_token_ids);
        builder.arg(&expert_block_ids);
        builder.arg(&total_tokens_post_pad);
        builder.arg(&pair_count);
        builder.arg(&expert_count);
        builder.arg(&block_size);
        builder.arg(&sorted_capacity);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("MoE align launch", error))
    }

    pub(super) fn launch_silu(
        &self,
        stream: &CudaStream,
        gate_up: u64,
        output: u64,
        intermediate_size: i32,
        activation_elements: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        super::launch_silu_mul(
            stream,
            &self.silu_mul,
            gate_up,
            output,
            intermediate_size,
            activation_elements,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_weighted_sum(
        &self,
        stream: &CudaStream,
        slots: u64,
        weights: u64,
        output: u64,
        tokens: i32,
        experts_per_token: i32,
        hidden_size: i32,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let hidden_u32 = u32::try_from(hidden_size)
            .map_err(|_| CudaDeviceRuntimeError::contract("MoE hidden size exceeds u32"))?;
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| CudaDeviceRuntimeError::contract("MoE token count exceeds u32"))?;
        let mut builder = stream.launch_builder(&self.weighted_sum);
        builder.arg(&slots);
        builder.arg(&weights);
        builder.arg(&output);
        builder.arg(&tokens);
        builder.arg(&experts_per_token);
        builder.arg(&hidden_size);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (hidden_u32.div_ceil(256), tokens_u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("MoE weighted sum launch", error))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_token_gate_add(
        &self,
        stream: &CudaStream,
        output: u64,
        shared_values: u64,
        shared_gate: u64,
        tokens: i32,
        hidden_size: i32,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let elements = i64::from(tokens)
            .checked_mul(i64::from(hidden_size))
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| CudaDeviceRuntimeError::contract("MoE shared merge size overflows"))?;
        let grid_x = u32::try_from(elements.div_ceil(256))
            .map_err(|_| CudaDeviceRuntimeError::contract("MoE shared merge grid exceeds u32"))?;
        let mut builder = stream.launch_builder(&self.token_gate_add);
        builder.arg(&output);
        builder.arg(&shared_values);
        builder.arg(&shared_gate);
        builder.arg(&tokens);
        builder.arg(&hidden_size);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid_x, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("MoE shared gate merge launch", error))
    }
}

pub(super) fn region_pointer(
    scratch_base: u64,
    region: WorkspaceRegion,
) -> Result<u64, CudaDeviceRuntimeError> {
    scratch_base
        .checked_add(region.offset_bytes())
        .ok_or_else(|| CudaDeviceRuntimeError::contract("MoE scratch pointer overflows"))
}

pub(super) fn zero_region(
    stream: &CudaStream,
    base: u64,
    region: WorkspaceRegion,
) -> Result<(), CudaDeviceRuntimeError> {
    let pointer = region_pointer(base, region)?;
    let length = usize::try_from(region.length_bytes())
        .map_err(|_| CudaDeviceRuntimeError::contract("MoE zero length exceeds usize"))?;
    unsafe { cudarc::driver::result::memset_d8_async(pointer, 0, length, stream.cu_stream()) }
        .map_err(|error| CudaDeviceRuntimeError::driver("MoE workspace zero", error))
}

fn load_function(
    module: &cudarc::driver::CudaModule,
    name: &'static str,
    operation: &'static str,
) -> Result<CudaFunction, CudaDeviceRuntimeError> {
    module
        .load_function(name)
        .map_err(|error| CudaDeviceRuntimeError::driver(operation, error))
}
