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
    fused_head_rms_partial_rope_function: CudaFunction,
    row_rms_addcmul_functions: [CudaFunction; 2],
    gqa_attention_function: CudaFunction,
    gqa_attention_f32_function: CudaFunction,
    spatial_functions: [CudaFunction; 3],
    extended_spatial_functions: [CudaFunction; 3],
    spatial_matmul_functions: [CudaFunction; 3],
    philox_function: CudaFunction,
    philox_f32_function: CudaFunction,
    primitive_function: CudaFunction,
    block_decode_function: CudaFunction,
    comfy_convrot_functions: [CudaFunction; 9],
    cast_f32_to_bf16_function: CudaFunction,
    pointwise_f32_function: CudaFunction,
    gelu_f32_function: CudaFunction,
    reshape_copy_f32_function: CudaFunction,
    row_scatter_f32_function: CudaFunction,
    channelwise_f32_function: CudaFunction,
    row_selected_modulation_function: CudaFunction,
    row_selected_bf16_functions: [CudaFunction; 2],
    batched_suffix_append_f32_function: CudaFunction,
    layer_norm_f32_function: CudaFunction,
    cnn_f32_functions: [CudaFunction; 4],
    norm_rope_f32_functions: [CudaFunction; 3],
    runtime_instance: u64,
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
            let row_rms_addcmul_functions =
                row_rms_addcmul::FUNCTIONS.map(|name| module.load_function(name));
            let [row_rms, addcmul] = row_rms_addcmul_functions;
            let row_rms_addcmul_functions = [
                row_rms.map_err(|e| {
                    CudaDeviceRuntimeError::driver("single-round RowRMS function load", e)
                })?,
                addcmul.map_err(|e| {
                    CudaDeviceRuntimeError::driver("single-round addcmul function load", e)
                })?,
            ];
            let gqa_attention_function = module.load_function(gqa::FUNCTION).map_err(|error| {
                CudaDeviceRuntimeError::driver("BF16 structured GQA attention function load", error)
            })?;
            let gqa_attention_f32_function =
                module.load_function(gqa_f32::FUNCTION).map_err(|error| {
                    CudaDeviceRuntimeError::driver(
                        "F32 structured GQA attention function load",
                        error,
                    )
                })?;
            let fused_head_rms_partial_rope_function = module
                .load_function(fused_head_rms_partial_rope::FUNCTION)
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver(
                        "fused head RMS/partial RoPE function load",
                        error,
                    )
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
            let philox_f32_function = module.load_function(philox_f32::FUNCTION).map_err(|e| {
                CudaDeviceRuntimeError::driver("F32 Philox normal function load", e)
            })?;
            let primitive_function = module
                .load_function(primitives::FUNCTION)
                .map_err(|e| CudaDeviceRuntimeError::driver("tensor primitive function load", e))?;
            let block_decode_function = module
                .load_function("ferrum_vnext_block_decode_bf16")
                .map_err(|e| {
                    CudaDeviceRuntimeError::driver("native block BF16 decode function load", e)
                })?;
            let comfy_convrot_functions = [
                "ferrum_comfy_convrot_bf16_f32",
                "ferrum_comfy_quantize_activation_bf16_i8",
                "ferrum_comfy_decode_codebook_w4_i8",
                "ferrum_comfy_int8_epilogue_bf16",
                "ferrum_comfy_swiglu_convrot_bf16_f32",
                "ferrum_comfy_convrot_f32_f32",
                "ferrum_comfy_quantize_activation_f32_i8",
                "ferrum_comfy_int8_epilogue_f32",
                "ferrum_comfy_convrot_int8_embedding_f32",
            ]
            .map(|name| module.load_function(name));
            let [rotate, quantize, decode, epilogue, swiglu_rotate, f32_rotate, f32_quantize, f32_epilogue, embedding] =
                comfy_convrot_functions;
            let comfy_convrot_functions = [
                rotate,
                quantize,
                decode,
                epilogue,
                swiglu_rotate,
                f32_rotate,
                f32_quantize,
                f32_epilogue,
                embedding,
            ]
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| CudaDeviceRuntimeError::driver("ConvRot W4A8 function load", error))?;
            let comfy_convrot_functions: [CudaFunction; 9] =
                comfy_convrot_functions.try_into().map_err(|_| {
                    CudaDeviceRuntimeError::contract(
                        "ConvRot function list differs from its module ABI",
                    )
                })?;
            let cast_f32_to_bf16_function = module
                .load_function("ferrum_vnext_cast_f32_to_bf16")
                .map_err(|e| {
                CudaDeviceRuntimeError::driver("F32 to BF16 cast function load", e)
            })?;
            let [head, rope, unit_partial] =
                norm_rope_f32::FUNCTIONS.map(|name| module.load_function(name));
            let channelwise_f32_function = module
                .load_function(channelwise_f32::FUNCTION)
                .map_err(|e| CudaDeviceRuntimeError::driver("F32 channelwise function load", e))?;
            let row_selected_modulation_function = module
                .load_function(row_selected_modulation::FUNCTION)
                .map_err(|e| {
                    CudaDeviceRuntimeError::driver("row-selected modulation function load", e)
                })?;
            let pointwise_f32_function = module
                .load_function("ferrum_vnext_pointwise_f32")
                .map_err(|e| CudaDeviceRuntimeError::driver("F32 pointwise function load", e))?;
            let [modulation, addcmul] =
                row_selected_bf16::FUNCTIONS.map(|name| module.load_function(name));
            let row_selected_bf16_functions = [
                modulation.map_err(|e| {
                    CudaDeviceRuntimeError::driver("compact BF16 modulation function load", e)
                })?,
                addcmul.map_err(|e| {
                    CudaDeviceRuntimeError::driver("compact BF16 addcmul function load", e)
                })?,
            ];
            let gelu_f32_function = module
                .load_function(gelu_f32::FUNCTION)
                .map_err(|e| CudaDeviceRuntimeError::driver("F32 GELU function load", e))?;
            let reshape_copy_f32_function = module
                .load_function(reshape_copy_f32::FUNCTION)
                .map_err(|e| CudaDeviceRuntimeError::driver("F32 reshape function load", e))?;
            let row_scatter_f32_function = module
                .load_function(row_scatter_f32::FUNCTION)
                .map_err(|e| CudaDeviceRuntimeError::driver("F32 row scatter function load", e))?;
            let batched_suffix_append_f32_function = module
                .load_function(batched_suffix_append_f32::FUNCTION)
                .map_err(|e| {
                    CudaDeviceRuntimeError::driver("F32 suffix append function load", e)
                })?;
            let layer_norm_f32_function = module
                .load_function(layer_norm_f32::FUNCTION)
                .map_err(|e| CudaDeviceRuntimeError::driver("F32 layer norm function load", e))?;
            let cnn_f32_functions = cnn_f32::FUNCTIONS.map(|name| module.load_function(name));
            let [pad, bias, norm, residual] = cnn_f32_functions;
            let cnn_f32_functions = [pad, bias, norm, residual]
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| CudaDeviceRuntimeError::driver("F32 CNN function load", e))?;
            let cnn_f32_functions: [CudaFunction; 4] = cnn_f32_functions
                .try_into()
                .unwrap_or_else(|_| unreachable!());
            let norm_rope_f32_functions = [
                head.map_err(|e| CudaDeviceRuntimeError::driver("F32 head RMS function load", e))?,
                rope.map_err(|e| {
                    CudaDeviceRuntimeError::driver("F32 half-split RoPE function load", e)
                })?,
                unit_partial.map_err(|e| {
                    CudaDeviceRuntimeError::driver(
                        "F32 unit head RMS/partial RoPE function load",
                        e,
                    )
                })?,
            ];
            Ok::<_, CudaDeviceRuntimeError>((
                pointwise,
                normalization,
                rope_function,
                attention_function,
                fused_attention_functions,
                transport,
                half_split_rope_function,
                fused_head_rms_partial_rope_function,
                row_rms_addcmul_functions,
                gqa_attention_function,
                gqa_attention_f32_function,
                spatial_functions,
                extended_spatial_functions,
                spatial_matmul_functions,
                philox_function,
                philox_f32_function,
                primitive_function,
                block_decode_function,
                comfy_convrot_functions,
                cast_f32_to_bf16_function,
                pointwise_f32_function,
                gelu_f32_function,
                reshape_copy_f32_function,
                row_scatter_f32_function,
                channelwise_f32_function,
                row_selected_modulation_function,
                row_selected_bf16_functions,
                batched_suffix_append_f32_function,
                layer_norm_f32_function,
                cnn_f32_functions,
                norm_rope_f32_functions,
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
            fused_head_rms_partial_rope_function,
            row_rms_addcmul_functions,
            gqa_attention_function,
            gqa_attention_f32_function,
            spatial_functions,
            extended_spatial_functions,
            spatial_matmul_functions,
            philox_function,
            philox_f32_function,
            primitive_function,
            block_decode_function,
            comfy_convrot_functions,
            cast_f32_to_bf16_function,
            pointwise_f32_function,
            gelu_f32_function,
            reshape_copy_f32_function,
            row_scatter_f32_function,
            channelwise_f32_function,
            row_selected_modulation_function,
            row_selected_bf16_functions,
            batched_suffix_append_f32_function,
            layer_norm_f32_function,
            cnn_f32_functions,
            norm_rope_f32_functions,
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
            fused_head_rms_partial_rope_function,
            row_rms_addcmul_functions,
            gqa_attention_function,
            gqa_attention_f32_function,
            spatial_functions,
            extended_spatial_functions,
            spatial_matmul_functions,
            philox_function,
            philox_f32_function,
            primitive_function,
            block_decode_function,
            comfy_convrot_functions,
            cast_f32_to_bf16_function,
            pointwise_f32_function,
            gelu_f32_function,
            reshape_copy_f32_function,
            row_scatter_f32_function,
            channelwise_f32_function,
            row_selected_modulation_function,
            row_selected_bf16_functions,
            batched_suffix_append_f32_function,
            layer_norm_f32_function,
            cnn_f32_functions,
            norm_rope_f32_functions,
            runtime_instance: runtime.instance_identity(),
            _provider: provider,
        }))
    }

    pub(super) const fn runtime_instance(&self) -> u64 {
        self.runtime_instance
    }

    pub(super) fn function(&self, kind: Kind) -> &CudaFunction {
        &self.functions[kind.index()]
    }

    pub(super) fn primitive_function(&self) -> &CudaFunction {
        &self.primitive_function
    }
    pub(super) fn block_decode_function(&self) -> &CudaFunction {
        &self.block_decode_function
    }

    pub(super) fn comfy_convrot_function(&self, index: usize) -> &CudaFunction {
        &self.comfy_convrot_functions[index]
    }
    pub(super) fn cast_f32_to_bf16_function(&self) -> &CudaFunction {
        &self.cast_f32_to_bf16_function
    }
    pub(super) fn pointwise_f32_function(&self) -> &CudaFunction {
        &self.pointwise_f32_function
    }
    pub(super) fn gelu_f32_function(&self) -> &CudaFunction {
        &self.gelu_f32_function
    }
    pub(super) fn reshape_copy_f32_function(&self) -> &CudaFunction {
        &self.reshape_copy_f32_function
    }
    pub(super) fn row_scatter_f32_function(&self) -> &CudaFunction {
        &self.row_scatter_f32_function
    }
    pub(super) fn channelwise_f32_function(&self) -> &CudaFunction {
        &self.channelwise_f32_function
    }
    pub(super) fn row_selected_modulation_function(&self) -> &CudaFunction {
        &self.row_selected_modulation_function
    }
    pub(super) fn row_selected_bf16_function(
        &self,
        kind: ferrum_runtime::RowSelectedBf16Kind,
    ) -> &CudaFunction {
        &self.row_selected_bf16_functions[row_selected_bf16::index(kind)]
    }
    pub(super) fn batched_suffix_append_f32_function(&self) -> &CudaFunction {
        &self.batched_suffix_append_f32_function
    }
    pub(super) fn layer_norm_f32_function(&self) -> &CudaFunction {
        &self.layer_norm_f32_function
    }
    pub(super) fn cnn_f32_function(&self, index: usize) -> &CudaFunction {
        &self.cnn_f32_functions[index]
    }
    pub(super) fn norm_rope_f32_function(&self, index: usize) -> &CudaFunction {
        &self.norm_rope_f32_functions[index]
    }

    pub(super) fn philox_f32_function(&self) -> &CudaFunction {
        &self.philox_f32_function
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
    pub(super) fn fused_head_rms_partial_rope_function(&self) -> &CudaFunction {
        &self.fused_head_rms_partial_rope_function
    }
    pub(super) fn row_rms_addcmul_function(&self, index: usize) -> &CudaFunction {
        &self.row_rms_addcmul_functions[index]
    }

    pub(super) fn gqa_attention_function(&self) -> &CudaFunction {
        &self.gqa_attention_function
    }

    pub(super) fn gqa_attention_f32_function(&self) -> &CudaFunction {
        &self.gqa_attention_f32_function
    }

    pub(super) fn transport_function(&self, kind: transport::Kind) -> &CudaFunction {
        &self.transport_functions[kind.index()]
    }
}
