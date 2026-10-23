//! One fixed native module, retained until all providers and commands retire.

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use cudarc::driver::CudaFunction;
use cudarc::nvrtc::Ptx;

use super::*;
use crate::backend::cuda::vnext_runtime::CudaTensorProviderRetention;

pub(super) struct RetainedFunctions {
    // Every CudaFunction retains the same native module/context. Field order
    // releases these owners before returning the one module capacity permit.
    functions: [CudaFunction; 7],
    norm_functions: [CudaFunction; 3],
    rope_function: CudaFunction,
    attention_function: CudaFunction,
    fused_attention_functions: [CudaFunction; 3],
    transport_functions: [CudaFunction; 2],
    half_split_rope_function: CudaFunction,
    gqa_attention_function: CudaFunction,
    spatial_functions: [CudaFunction; 3],
    extended_spatial_functions: [CudaFunction; 3],
    spatial_matmul_functions: [CudaFunction; 3],
    philox_function: CudaFunction,
    primitive_function: CudaFunction,
    _provider: CudaTensorProviderRetention,
}

impl RetainedFunctions {
    pub(super) fn load(runtime: &CudaDeviceRuntime) -> Result<Arc<Self>, CudaDeviceRuntimeError> {
        let provider = runtime.reserve_tensor_provider()?;
        let loaded = catch_unwind(AssertUnwindSafe(|| {
            let module = runtime
                .context()
                .load_module(Ptx::from_src(crate::ptx::VNEXT_BF16_SILU.to_owned()))
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("BF16 pointwise module load", error)
                })?;
            let load = |kind: Kind| {
                module.load_function(kind.function()).map_err(|error| {
                    CudaDeviceRuntimeError::driver("BF16 pointwise function load", error)
                })
            };
            let pointwise = [
                load(Kind::Silu)?,
                load(Kind::SiluMul)?,
                load(Kind::Tanh)?,
                load(Kind::Modulation)?,
                load(Kind::GatedResidual)?,
                load(Kind::GeluTanh)?,
                load(Kind::PlainResidualAdd)?,
            ];
            let load_norm = |kind: norm::Kind| {
                module.load_function(kind.function()).map_err(|error| {
                    CudaDeviceRuntimeError::driver("BF16 normalization function load", error)
                })
            };
            let normalization = [
                load_norm(norm::Kind::Layer)?,
                load_norm(norm::Kind::HeadRms)?,
                load_norm(norm::Kind::ZeroCenteredRms)?,
            ];
            let rope_function = module.load_function(rope::FUNCTION).map_err(|error| {
                CudaDeviceRuntimeError::driver("BF16 adjacent RoPE function load", error)
            })?;
            let attention_function =
                module.load_function(attention::FUNCTION).map_err(|error| {
                    CudaDeviceRuntimeError::driver("BF16 structured attention function load", error)
                })?;
            let fused_attention_functions = [
                "ferrum_vnext_structured_attention_range_bf16",
                "ferrum_vnext_attention_pack_kv_bf16",
                "ferrum_vnext_attention_zero_range_bf16",
            ]
            .map(|name| module.load_function(name));
            let [range, pack, zero] = fused_attention_functions;
            let fused_attention_functions = [range, pack, zero].map(|function| {
                function.map_err(|error| {
                    CudaDeviceRuntimeError::driver(
                        "BF16 fused attention support function load",
                        error,
                    )
                })
            });
            let [range, pack, zero] = fused_attention_functions;
            let fused_attention_functions = [range?, pack?, zero?];
            let load_transport = |kind: transport::Kind| {
                module.load_function(kind.function()).map_err(|error| {
                    CudaDeviceRuntimeError::driver("BF16 transport function load", error)
                })
            };
            let transport = [
                load_transport(transport::Kind::RowGather)?,
                load_transport(transport::Kind::BatchedRowConcat)?,
            ];
            let half_split_rope_function = module
                .load_function(half_split_rope::FUNCTION)
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("BF16 half-split RoPE function load", error)
                })?;
            let gqa_attention_function = module.load_function(gqa::FUNCTION).map_err(|error| {
                CudaDeviceRuntimeError::driver("BF16 structured GQA attention function load", error)
            })?;
            let spatial = [
                "ferrum_vnext_nchw_pad_bf16",
                "ferrum_vnext_nchw_bias_bf16",
                "ferrum_vnext_channel_l2_norm_bf16",
            ]
            .map(|name| module.load_function(name));
            let [pad, bias, channel_norm] = spatial;
            let spatial_functions = [pad, bias, channel_norm]
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("BF16 spatial function load", error)
                })?;
            let spatial_functions: [CudaFunction; 3] = spatial_functions
                .try_into()
                .unwrap_or_else(|_| unreachable!());
            // Inspect the function held by this admitted module. This is the
            // driver's loaded implementation, not a separately loaded probe.
            // Dynamic shared memory remains a per-launch provider requirement.
            tracing::debug!(
                target: "ferrum_kernels::cuda::resources",
                kernel = "ferrum_vnext_channel_l2_norm_bf16",
                registers = ?spatial_functions[2].num_regs(),
                local_bytes = ?spatial_functions[2].local_size_bytes(),
                static_shared_bytes = ?spatial_functions[2].shared_size_bytes(),
                ptx_version = ?spatial_functions[2].ptx_version(),
                binary_version = ?spatial_functions[2].binary_version(),
                "managed CUDA function resources"
            );
            let extended_spatial_functions = [
                module.load_function(spatial_extended::Kind::Attention.function()),
                module.load_function(spatial_extended::Kind::Gather.function()),
                module.load_function(spatial_extended::Kind::GroupMean.function()),
            ];
            let [attention, gather, group_mean] = extended_spatial_functions;
            let extended_spatial_functions = [
                attention.map_err(|e| {
                    CudaDeviceRuntimeError::driver("BF16 spatial attention function load", e)
                })?,
                gather.map_err(|e| {
                    CudaDeviceRuntimeError::driver("BF16 spatial gather function load", e)
                })?,
                group_mean.map_err(|e| {
                    CudaDeviceRuntimeError::driver("BF16 spatial group mean function load", e)
                })?,
            ];
            let load_spatial_matmul = |name| {
                module.load_function(name).map_err(|error| {
                    CudaDeviceRuntimeError::driver(
                        "BF16 spatial matrix attention support function load",
                        error,
                    )
                })
            };
            let spatial_matmul_functions = [
                load_spatial_matmul("ferrum_vnext_spatial_values_f32")?,
                load_spatial_matmul("ferrum_vnext_spatial_softmax_f32")?,
                load_spatial_matmul("ferrum_vnext_spatial_output_bf16")?,
            ];
            let philox_function = module.load_function(philox::FUNCTION).map_err(|e| {
                CudaDeviceRuntimeError::driver("BF16 Philox normal function load", e)
            })?;
            let primitive_function = module
                .load_function(primitives::FUNCTION)
                .map_err(|e| CudaDeviceRuntimeError::driver("tensor primitive function load", e))?;
            Ok::<_, CudaDeviceRuntimeError>((
                pointwise,
                normalization,
                rope_function,
                attention_function,
                fused_attention_functions,
                transport,
                half_split_rope_function,
                gqa_attention_function,
                spatial_functions,
                extended_spatial_functions,
                spatial_matmul_functions,
                philox_function,
                primitive_function,
            ))
        }));
        let (
            functions,
            norm_functions,
            rope_function,
            attention_function,
            fused_attention_functions,
            transport_functions,
            half_split_rope_function,
            gqa_attention_function,
            spatial_functions,
            extended_spatial_functions,
            spatial_matmul_functions,
            philox_function,
            primitive_function,
        ) = match loaded {
            Ok(Ok(functions)) => functions,
            Ok(Err(error)) => {
                // A partially failed load has not certified native cleanup.
                std::mem::forget(provider);
                return Err(error);
            }
            Err(payload) => {
                std::mem::forget(provider);
                resume_unwind(payload);
            }
        };
        Ok(Arc::new(Self {
            functions,
            norm_functions,
            rope_function,
            attention_function,
            fused_attention_functions,
            transport_functions,
            half_split_rope_function,
            gqa_attention_function,
            spatial_functions,
            extended_spatial_functions,
            spatial_matmul_functions,
            philox_function,
            primitive_function,
            _provider: provider,
        }))
    }

    pub(super) fn function(&self, kind: Kind) -> &CudaFunction {
        &self.functions[kind.index()]
    }

    pub(super) fn primitive_function(&self) -> &CudaFunction {
        &self.primitive_function
    }

    pub(super) fn philox_function(&self) -> &CudaFunction {
        &self.philox_function
    }

    pub(super) fn extended_spatial_function(&self, kind: spatial_extended::Kind) -> &CudaFunction {
        &self.extended_spatial_functions[kind.index()]
    }

    pub(super) fn spatial_function(&self, index: usize) -> &CudaFunction {
        &self.spatial_functions[index]
    }

    pub(super) fn spatial_matmul_function(&self, index: usize) -> &CudaFunction {
        &self.spatial_matmul_functions[index]
    }

    pub(super) fn norm_function(&self, kind: norm::Kind) -> &CudaFunction {
        &self.norm_functions[kind.index()]
    }

    pub(super) fn rope_function(&self) -> &CudaFunction {
        &self.rope_function
    }

    pub(super) fn attention_function(&self) -> &CudaFunction {
        &self.attention_function
    }

    pub(super) fn fused_attention_function(&self, index: usize) -> &CudaFunction {
        &self.fused_attention_functions[index]
    }

    pub(super) fn half_split_rope_function(&self) -> &CudaFunction {
        &self.half_split_rope_function
    }

    pub(super) fn gqa_attention_function(&self) -> &CudaFunction {
        &self.gqa_attention_function
    }

    pub(super) fn transport_function(&self, kind: transport::Kind) -> &CudaFunction {
        &self.transport_functions[kind.index()]
    }
}
