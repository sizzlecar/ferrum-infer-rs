//! Fixed tensor providers. Native ownership is admitted once per runtime.
use super::super::vnext_runtime::{MetalDeviceCommand, MetalTensorProviderRetention};
use super::*;
use ferrum_runtime::{
    DeviceCommandLogicalWork, EncodedDeviceOperation, NodeWorkContract, OperationResourceEstimator,
    ReusableExecutionTopologyRequest, TensorInvocationWork, TensorOperationInvocation,
};
use metal::{CompileOptions, ComputePipelineState, MTLSize};

mod attention;
mod fp8;
mod linear;
mod norm;
mod operation;
mod philox;
mod pointwise;
mod primitives;
mod rope;
mod scatter;
mod spatial;
#[cfg(test)]
mod tests;
mod transport;
pub use operation::MetalTensorOperation;

type TensorParticipant<'a> = OperationInvocation<'a, MetalDeviceBuffer, TensorInvocationWork<'a>>;

pub const METAL_TENSOR_NUMERICAL_IDENTITY: &str = concat!(
    "metal-std-f32;fast-math=false;fp-contract=off;bf16-integer-rne;",
    "silu_mul=round-silu-then-mul;modulation=round-one-plus-scale-then-mul;",
    "gated-residual=round-gate-times-update-then-add;nan-class-only;",
    "gemm=MPP-BF16xBF16-F32-m64n64-interior-m32n64-remainder-dynamicK-relaxed-false-int-range;",
    "gemm-tails-and-fallback=simdgroup-f32-m32n64k32;disjoint-bottom-then-right;",
    "gemm=at-most-four-dispatches;OI;BF16-final-boundary;",
    "native-block=decode-f32-then-BF16;admitted-invocation-scratch;",
    "norm=zero-padded-adjacent-FP32-tree;separate-sqrt-div;device-bounded-threadgroup;",
    "head-rms=BF16-normalized-before-gamma;zero-centered=FP32-one-plus-gamma;",
    "adjacent-rope=F32-frequency-explicit-FMA;half-split=BF16-staged-products;",
    "attention-fallback=MPP-BF16xBF16-F32-QK-head128-full-visible-tiles-or-FP32-simdgroup-QK;",
    "attention-head128-primary=SIMD-FP32-register-P2-O16-BQ32-BK16;ushort-exact-BF16-staging;",
    "register-softmax=coordinate-calibrated-xor1-xor8-or-dynamic-four-peer-or-full-shuffle-FP32-reduction;exp;",
    "register-finite-PV=guarded-explicit-FP32-output-rescale-then-inplace-MMA-C;",
    "register-guards=sticky-V-above-2p93-O-nonfinite-or-above-2p124-P-alpha-outside-0-2;fallback=zero-PV-MMA-separate-add;",
    "register-nonfinite-V=scalar-PV-then-separate-rescale-add;",
    "register-key-runs=generic-or-validated-all-batch-all-valid;causal-segment-tail-ieee-order-unchanged;",
    "register-attention=shared15120;previous-MPP-SIMD-tile16x32-fallback;cross-path-unchanged;",
    "fallback-Q-fragments=generic-head128-cache;MPP-fallback-staged32;head128-output-cache-unchanged;",
    "fallback-PV=MPP-FP32xFP32-FP32-finite-threadgroup-tiles-or-FP32-simdgroup;16-query-32-key-online-softmax;",
    "FP32-tile-SIMD-exp-reduction;separate-online-rescale;BF16-final-only;",
    "GQA=indexed-KV-no-replication;empty-row=positive-zero;",
    "cross-attention=independent-Q-KV-strides;key-only-mask;shared-online-QK-PV;",
    "masked-score-discard;nonfinite-visible-score=whole-head-NaN;",
    "nonfinite-V=visible-only-scalar-PV;zero-global-scratch;device-bounded-on-chip;",
    "transport=BF16-bitcopy;bounded-row-runs;per-batch-concat;zero-arithmetic;",
    "row-scatter=ordered-disjoint-runs;F32-add-BF16-final;untouched-bitcopy;",
    "conv=F32-simdgroup-implicit-im2col;explicit-pad-admitted-before-GEMM;BF16-dot-before-bias;",
    "channel-L2=padded-pairwise-F32;sqrt-clamp-div;BF16-div-scale-gamma-plus-zero;",
    "spatial-attention=16x32-online-F32;SIMD-exp-reduction;NCHW-QKV;admitted-F32-accumulator;",
    "spatial-upsample=bitcopy-explicit-channel-map;",
    "spatial-group-mean=last-real-frame;serial-F32-first-element;F32-div;BF16-final;",
    "affine=F32-dot-plus-bias;patch=BF16-dot-before-bias;affine-norm=F32-scale-bias;",
    "vision-rotary=F32-tables-products-add;gelu-erf=SunPro-F32-piecewise;",
    "segmented-attention=isolated-noncausal;reshape=exact-bitcopy;",
    "noise=Philox4x32-10;flat-normal-ordinal;open-U23-midpoint;Box-Muller-F32;",
    "FP8=E4M3FN-RNE;row-scale=max(amax/448,1/229376);nonfinite-row=NaN;",
    "FP8-GEMM=packed-on-chip-decode;F32-dot-times-row-scale-times-weight-scale;",
    "independent-Metal-profile;no-CUDA-or-Reference-bitwise-claim"
);

pub fn metal_tensor_runtime_config(
    device_id: DeviceId,
) -> Result<MetalDeviceRuntimeConfig, VNextError> {
    Ok(MetalDeviceRuntimeConfig {
        device_id,
        runtime_implementation_fingerprint: fingerprint(),
        capabilities: MetalTensorOperation::ALL
            .into_iter()
            .map(|op| CapabilityId::new(op.capability()))
            .collect::<Result<_, _>>()?,
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )?]),
    })
}

fn fingerprint() -> String {
    implementation_fingerprint(&[
        include_bytes!("mod.rs"),
        include_bytes!("operation.rs"),
        include_bytes!("fp8.rs"),
        include_bytes!("fp8/storage.rs"),
        include_bytes!("fp8.metal"),
        include_bytes!("philox.rs"),
        include_bytes!("philox.metal"),
        include_bytes!("pointwise.rs"),
        include_bytes!("primitives.rs"),
        include_bytes!("primitives.metal"),
        include_bytes!("erf.metal"),
        include_bytes!("bf16.metal"),
        include_bytes!("pointwise.metal"),
        include_bytes!("linear.rs"),
        include_bytes!("linear.metal"),
        linear::mpp::FINGERPRINT_SOURCE.as_bytes(),
        linear::mpp::NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("block_decode.metal"),
        include_bytes!("norm.rs"),
        include_bytes!("norm.metal"),
        include_bytes!("rope.rs"),
        include_bytes!("rope.metal"),
        include_bytes!("attention.rs"),
        include_bytes!("attention.metal"),
        attention::mpp::FINGERPRINT_SOURCE.as_bytes(),
        attention::mpp::NUMERICAL_IDENTITY.as_bytes(),
        attention::register128::FINGERPRINT_SOURCE.as_bytes(),
        attention::register128::NUMERICAL_IDENTITY.as_bytes(),
        attention::cross::FINGERPRINT_SOURCE.as_bytes(),
        attention::cross::PROFILE.as_bytes(),
        include_bytes!("transport.rs"),
        include_bytes!("transport.metal"),
        include_bytes!("scatter.rs"),
        include_bytes!("scatter.metal"),
        include_bytes!("spatial.rs"),
        spatial::FINGERPRINT_SOURCE.as_bytes(),
        native_blocks::FINGERPRINT_SOURCE.as_bytes(),
        METAL_TENSOR_NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("../mod.rs"),
    ])
}

struct Pipelines {
    pointwise: Option<ComputePipelineState>,
    linear: Option<ComputePipelineState>,
    linear_mpp: Option<ComputePipelineState>,
    linear_mpp_large: Option<ComputePipelineState>,
    block_decode: Option<ComputePipelineState>,
    fp8_quantize: Option<ComputePipelineState>,
    fp8_linear: Option<ComputePipelineState>,
    norm: Option<ComputePipelineState>,
    norm_shared_limit: u64,
    adjacent_rope: Option<ComputePipelineState>,
    half_split_rope: Option<ComputePipelineState>,
    attention: Option<ComputePipelineState>,
    attention_head128: Option<ComputePipelineState>,
    attention_mpp_head128: Option<ComputePipelineState>,
    attention_register128: Option<ComputePipelineState>,
    attention_register128_all_valid: Option<ComputePipelineState>,
    attention_shared_limit: u64,
    cross_attention: Option<attention::cross::CrossPipelines>,
    transport: Option<ComputePipelineState>,
    scatter: Option<ComputePipelineState>,
    primitives: Option<ComputePipelineState>,
    philox: Option<ComputePipelineState>,
    spatial: spatial::Pipelines,
    // Last field: native pipeline states release before their budget owner.
    _retention: MetalTensorProviderRetention,
}

pub fn metal_tensor_providers(
    runtime: &MetalDeviceRuntime,
    selected: &[MetalTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<MetalDeviceRuntime>>>, MetalDeviceRuntimeError> {
    if selected.is_empty() || selected.iter().collect::<BTreeSet<_>>().len() != selected.len() {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal tensor provider selection is empty or repeated",
        ));
    }
    // Validate every contract and capability before native creation.
    let descriptors = selected
        .iter()
        .map(|&kind| {
            let contract = kind.contract().map_err(contract_error)?;
            let weights: &[&str] = match kind {
                MetalTensorOperation::DenseLinear
                | MetalTensorOperation::AffineLinear
                | MetalTensorOperation::PatchProjection
                | MetalTensorOperation::AffineLayerNorm
                | MetalTensorOperation::HeadRmsNorm
                | MetalTensorOperation::ZeroCenteredRmsNorm
                | MetalTensorOperation::RowGather
                | MetalTensorOperation::RowScatterAdd
                | MetalTensorOperation::Conv2d
                | MetalTensorOperation::ChannelL2Norm
                | MetalTensorOperation::ChannelL2NormImage
                | MetalTensorOperation::SpatialUpsampleGather
                | MetalTensorOperation::SpatialGroupMean
                | MetalTensorOperation::BatchedRowConcat => {
                    &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID]
                }
                MetalTensorOperation::BlockLinear => &[GGUF_NATIVE_BLOCK_FORMAT_ID],
                MetalTensorOperation::Fp8Linear => &[DENSE_SAFETENSORS_FORMAT_ID],
                _ => &[],
            };
            let quantization: Vec<_> = if kind == MetalTensorOperation::BlockLinear {
                linear::FORMATS
                    .iter()
                    .map(|format| format.format_id())
                    .collect()
            } else if kind == MetalTensorOperation::Fp8Linear {
                vec![ferrum_runtime::FP8_E4M3FN_WHOLE_TENSOR_F32_FORMAT_ID]
            } else {
                Vec::new()
            };
            provider_descriptor(
                runtime,
                &contract,
                &format!("provider.metal.{}.bf16", kind.name()),
                kind.capability(),
                &format!("resource-estimator.metal.{}.bf16", kind.name()),
                contiguous_bindings(kind.inputs()),
                weights,
                &quantization,
                implementation_fingerprint(&[fingerprint().as_bytes(), kind.profile().as_bytes()]),
            )
            .map(|d| d.with_fixed_tensor_work(true))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let retention = runtime.retain_tensor_provider()?;
    let has_pointwise = selected.iter().any(|op| op.is_pointwise());
    let has_linear = selected.iter().any(|op| op.is_linear());
    let has_blocks = selected.contains(&MetalTensorOperation::BlockLinear);
    let has_fp8 = selected.contains(&MetalTensorOperation::Fp8Linear);
    let has_norm = selected.iter().any(|op| op.is_norm());
    let has_rope = selected.iter().any(|op| op.is_rope());
    let has_attention = selected.iter().any(|op| op.is_attention());
    let has_transport = selected.iter().any(|op| op.is_transport());
    let has_scatter = selected.contains(&MetalTensorOperation::RowScatterAdd);
    let has_primitives = selected.iter().any(|op| op.is_element_primitive());
    let has_philox = selected.contains(&MetalTensorOperation::PhiloxNormal);
    let mut source = String::from(include_str!("bf16.metal"));
    if has_blocks {
        source.push_str(&native_blocks::decoder_source());
    }
    if has_pointwise {
        source.push_str(include_str!("pointwise.metal"));
    }
    if has_linear {
        source.push_str(include_str!("linear.metal"));
    }
    if has_blocks {
        source.push_str(include_str!("block_decode.metal"));
    }
    if has_norm {
        source.push_str(include_str!("norm.metal"));
    }
    if has_rope {
        source.push_str(include_str!("rope.metal"));
    }
    if has_attention {
        source.push_str(include_str!("attention.metal"));
    }
    if has_transport {
        source.push_str(include_str!("transport.metal"));
    }
    if has_scatter {
        source.push_str(include_str!("scatter.metal"));
    }
    if has_primitives {
        source.push_str(include_str!("erf.metal"));
        source.push_str(include_str!("primitives.metal"));
    }
    if has_philox {
        source.push_str(include_str!("philox.metal"));
    }
    if has_fp8 {
        source.push_str(include_str!("fp8.metal"));
    }
    spatial::append_source(selected, &mut source);
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = runtime
        .device()
        .new_library_with_source(&source, &options)
        .map_err(MetalDeviceRuntimeError::contract)?;
    let pipeline = |selected: bool, name: &str| -> Result<_, MetalDeviceRuntimeError> {
        if !selected {
            return Ok(None);
        }
        let function = library
            .get_function(name, Some(metal::FunctionConstantValues::new()))
            .map_err(MetalDeviceRuntimeError::contract)?;
        runtime
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map(Some)
            .map_err(MetalDeviceRuntimeError::contract)
    };
    let pointwise = pipeline(has_pointwise, "ferrum_tensor_pointwise_bf16")?;
    let linear = pipeline(has_linear, "ferrum_tensor_linear_bf16")?;
    // MPP belongs to the same admitted provider bundle, but uses a separate
    // MSL 4 library. Older systems never compile its headers; actual compiler
    // errors on an eligible device remain errors rather than silent fallback.
    let [linear_mpp, linear_mpp_large] = if has_linear {
        let [small, large] =
            linear::mpp::compile(runtime.device()).map_err(MetalDeviceRuntimeError::contract)?;
        [
            (linear::mpp::TILE_ROWS, small),
            (linear::mpp::LARGE_TILE_ROWS, large),
        ]
        .map(|(tile_rows, availability)| match availability {
            linear::mpp::MppPipelineAvailability::Unsupported { reason } => {
                tracing::info!(tile_rows, %reason, "Metal MPP BF16 GEMM tile unavailable");
                None
            }
            linear::mpp::MppPipelineAvailability::Ready(pipeline) => {
                tracing::info!(
                    tile_rows,
                    static_threadgroup_bytes = pipeline.static_threadgroup_memory_length(),
                    "Metal MPP BF16 GEMM enabled for supported full-tile shapes"
                );
                Some(pipeline)
            }
        })
    } else {
        [None, None]
    };
    let block_decode = pipeline(has_blocks, "ferrum_tensor_block_decode_bf16")?;
    let fp8_quantize = pipeline(has_fp8, "ferrum_tensor_fp8_quantize")?;
    let fp8_linear = pipeline(has_fp8, "ferrum_tensor_linear_fp8_bf16")?;
    if let Some(p) = &fp8_quantize {
        if p.max_total_threads_per_threadgroup() < 256
            || p.static_threadgroup_memory_length()
                > runtime.device().max_threadgroup_memory_length()
        {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal FP8 row reduction requires 256 threads and declared on-chip memory",
            ));
        }
    }
    let norm = pipeline(has_norm, "ferrum_tensor_norm_bf16")?;
    let norm_shared_limit = match &norm {
        Some(p) => runtime
            .device()
            .max_threadgroup_memory_length()
            .checked_sub(p.static_threadgroup_memory_length())
            .ok_or_else(|| {
                MetalDeviceRuntimeError::contract("Metal normalization exceeds threadgroup memory")
            })?,
        None => 0,
    };
    let adjacent_rope = pipeline(
        selected.contains(&MetalTensorOperation::AdjacentRope),
        "ferrum_tensor_adjacent_rope_bf16",
    )?;
    let half_split_rope = pipeline(
        selected.contains(&MetalTensorOperation::HalfSplitRope),
        "ferrum_tensor_half_split_rope_bf16",
    )?;
    let attention = pipeline(has_attention, "ferrum_tensor_attention_bf16")?;
    let attention_head128 = if has_attention {
        attention::head128_pipeline(runtime, &library)?
    } else {
        None
    };
    let [attention_register128, attention_register128_all_valid] = if has_attention {
        attention::register128::compile_variants(runtime)?
    } else {
        [None, None]
    };
    let attention_mpp_head128 = if has_attention {
        match attention::mpp::compile(runtime.device())
            .map_err(MetalDeviceRuntimeError::contract)?
        {
            linear::mpp::MppPipelineAvailability::Unsupported { reason } => {
                tracing::info!(%reason, "Metal MPP attention unavailable; using SIMD-group fallback");
                None
            }
            linear::mpp::MppPipelineAvailability::Ready(pipeline) => {
                tracing::info!(
                    static_threadgroup_bytes = pipeline.static_threadgroup_memory_length(),
                    "Metal MPP Head128 attention enabled: complete visible BF16 QK and finite FP32 PV tiles"
                );
                Some(pipeline)
            }
        }
    } else {
        None
    };
    let transport = pipeline(has_transport, "ferrum_tensor_transport_bf16")?;
    let scatter = pipeline(has_scatter, "ferrum_tensor_row_scatter_add_bf16")?;
    let primitives = pipeline(has_primitives, "ferrum_tensor_primitive_bf16")?;
    let philox = pipeline(has_philox, "ferrum_tensor_philox_normal_bf16")?;
    let attention_shared_limit = match &attention {
        Some(p) => {
            if p.thread_execution_width() != 32 || p.max_total_threads_per_threadgroup() < 128 {
                return Err(MetalDeviceRuntimeError::contract(
                    "Metal attention needs four 32-lane SIMD groups",
                ));
            }
            runtime
                .device()
                .max_threadgroup_memory_length()
                .checked_sub(p.static_threadgroup_memory_length())
                .ok_or_else(|| {
                    MetalDeviceRuntimeError::contract("Metal attention exceeds threadgroup memory")
                })?
        }
        None => 0,
    };
    for pipeline in [&linear, &fp8_linear].into_iter().flatten() {
        if pipeline.thread_execution_width() != 32
            || pipeline.max_total_threads_per_threadgroup() < 128
            || pipeline.static_threadgroup_memory_length()
                > runtime.device().max_threadgroup_memory_length()
        {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal BF16 GEMM needs four 32-lane SIMD groups and its declared on-chip storage",
            ));
        }
    }
    let pipelines = Arc::new(Pipelines {
        spatial: spatial::pipelines(runtime, selected, &library)?,
        cross_attention: selected
            .contains(&MetalTensorOperation::CrossAttention)
            .then(|| attention::cross::pipelines(runtime))
            .transpose()?,
        pointwise,
        linear,
        linear_mpp,
        linear_mpp_large,
        block_decode,
        fp8_quantize,
        fp8_linear,
        norm,
        norm_shared_limit,
        adjacent_rope,
        half_split_rope,
        attention,
        attention_head128,
        attention_mpp_head128,
        attention_register128,
        attention_register128_all_valid,
        attention_shared_limit,
        transport,
        scatter,
        primitives,
        philox,
        _retention: retention,
    });
    Ok(selected
        .iter()
        .copied()
        .zip(descriptors)
        .map(
            |(kind, descriptor)| -> Box<dyn OperationProvider<MetalDeviceRuntime>> {
                if kind.is_linear() {
                    return Box::new(linear::LinearProvider {
                        kind,
                        descriptor,
                        pipelines: Arc::clone(&pipelines),
                    });
                }
                Box::new(TensorProvider {
                    kind,
                    descriptor,
                    pipelines: Arc::clone(&pipelines),
                })
            },
        )
        .collect())
}

struct TensorProvider {
    kind: MetalTensorOperation,
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<Pipelines>,
}

impl OperationResourceEstimator for TensorProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if self.kind == MetalTensorOperation::Fp8Linear {
            return fp8::estimate(&self.descriptor, &request);
        }
        let estimate =
            estimate_without_workspace(&self.descriptor, &request, self.kind.operation())?;
        if self.kind == MetalTensorOperation::PhiloxNormal {
            philox::geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        } else if self.kind.is_norm() {
            norm::geometry(
                self.kind,
                request.values(),
                request.attributes(),
                self.pipelines.norm_shared_limit,
            )
            .map_err(invalid_plan)?;
        } else if self.kind.is_rope() {
            rope::geometry(self.kind, request.values(), request.attributes())
                .map_err(invalid_plan)?;
        } else if self.kind == MetalTensorOperation::CrossAttention {
            attention::cross::geometry(
                request.values(),
                request.attributes(),
                self.pipelines
                    .cross_attention
                    .as_ref()
                    .expect("selected cross attention pipelines")
                    .shared_limit,
            )
            .map_err(invalid_plan)?;
        } else if self.kind.is_attention() {
            attention::geometry(
                self.kind,
                request.values(),
                request.attributes(),
                self.pipelines.attention_shared_limit,
            )
            .map_err(invalid_plan)?;
        } else if self.kind.is_transport() {
            transport::geometry(self.kind, request.values(), request.attributes())
                .map_err(invalid_plan)?;
        } else if self.kind == MetalTensorOperation::RowScatterAdd {
            scatter::geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        } else if self.kind.is_element_primitive() {
            primitives::geometry(self.kind, request.values(), request.attributes())
                .map_err(invalid_plan)?;
        } else if self.kind.is_spatial() {
            let bytes = spatial::scratch_bytes(
                self.kind,
                request.values(),
                request.attributes(),
                &self.pipelines.spatial,
            )
            .map_err(invalid_plan)?;
            return spatial::estimate_workspace(&self.descriptor, &request, estimate, bytes);
        } else {
            pointwise::geometry(self.kind, request.values(), request.attributes())
                .map_err(invalid_plan)?;
        }
        Ok(estimate)
    }
}

impl OperationProvider<MetalDeviceRuntime> for TensorProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "metal.tensor",
            "Metal tensor operation requires fixed tensor work".into(),
        ))
    }

    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<MetalDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        self.encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|reason| provider_failure(identity, "metal.tensor", reason))
    }
}

impl TensorProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<MetalDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("Metal operation requires one fixed tensor participant".into());
        };
        if invocation.operation().id.as_str() != self.kind.operation()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("Metal operation received a different operation or work contract".into());
        }
        if self.kind == MetalTensorOperation::Fp8Linear {
            return fp8::encode(participant, Arc::clone(&self.pipelines));
        }
        if self.kind == MetalTensorOperation::PhiloxNormal {
            return philox::encode(participant, Arc::clone(&self.pipelines));
        }
        if self.kind.is_norm() {
            return norm::encode(self.kind, participant, Arc::clone(&self.pipelines));
        }
        if self.kind.is_rope() {
            return rope::encode(self.kind, participant, Arc::clone(&self.pipelines));
        }
        if self.kind == MetalTensorOperation::CrossAttention {
            return attention::cross::encode(participant, Arc::clone(&self.pipelines));
        }
        if self.kind.is_attention() {
            return attention::encode(self.kind, participant, Arc::clone(&self.pipelines));
        }
        if self.kind.is_transport() {
            return transport::encode(self.kind, participant, Arc::clone(&self.pipelines));
        }
        if self.kind == MetalTensorOperation::RowScatterAdd {
            return scatter::encode(participant, Arc::clone(&self.pipelines));
        }
        if self.kind.is_element_primitive() {
            return primitives::encode(self.kind, participant, Arc::clone(&self.pipelines));
        }
        if self.kind.is_spatial() {
            return spatial::encode(self.kind, participant, Arc::clone(&self.pipelines));
        }
        let shape =
            pointwise::geometry(self.kind, participant.bindings(), participant.attributes())?;
        let mut regions = Vec::with_capacity(self.kind.inputs() as usize + 1);
        for ordinal in 0..self.kind.inputs() {
            let region = contiguous_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
                ElementType::Bf16,
            )?;
            let expected = if self.kind.broadcasts() && ordinal == 1 {
                shape.parameter_bytes
            } else {
                shape.bytes
            };
            validate_region(&region, expected)?;
            regions.push(region);
        }
        let output = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::Bf16,
        )?;
        validate_region(&output, shape.bytes)?;
        for input in &regions {
            if input
                .overlaps_shared_region(&output)
                .map_err(|e| e.to_string())?
            {
                return Err("Metal pointwise output overlaps a live input".into());
            }
        }
        regions.push(output);
        self.command(shape, regions)
            .map_err(|error| error.to_string())
    }

    fn command(
        &self,
        shape: pointwise::Shape,
        regions: Vec<MetalBufferRegion>,
    ) -> Result<MetalDeviceCommand, MetalDeviceRuntimeError> {
        let pipelines = Arc::clone(&self.pipelines);
        let kind = self.kind;
        let params = Params {
            elements: shape.elements,
            width: shape.width,
            kind: kind.pointwise_shader_kind(),
            row_broadcast: u32::from(shape.row_broadcast),
        };
        let work = DeviceCommandLogicalWork::tensor(shape.elements, shape.operations)
            .map_err(contract_error)?;
        MetalDeviceCommand::tensor_operation(
            "metal.pointwise.bf16",
            regions,
            work,
            move |encoder, regions| {
                let compute = encoder.compute_encoder();
                let pipeline = pipelines
                    .pointwise
                    .as_ref()
                    .expect("selected pointwise pipeline");
                compute.set_compute_pipeline_state(pipeline);
                for (slot, index) in [
                    0,
                    if kind.inputs() > 1 { 1 } else { 0 },
                    if kind.inputs() > 2 { 2 } else { 0 },
                    kind.inputs() as usize,
                ]
                .into_iter()
                .enumerate()
                {
                    let region = &regions[index];
                    compute.set_buffer(slot as u64, Some(region.buffer()), region.offset_bytes());
                }
                compute.set_bytes(
                    4,
                    std::mem::size_of::<Params>() as u64,
                    (&params as *const Params).cast(),
                );
                let threads = pipeline.max_total_threads_per_threadgroup().min(256);
                compute.dispatch_threads(
                    MTLSize::new(shape.elements, 1, 1),
                    MTLSize::new(threads, 1, 1),
                );
                Ok(())
            },
        )
    }
}

fn validate_region(region: &MetalBufferRegion, bytes: u64) -> Result<(), String> {
    if region.length_bytes() != bytes || !region.offset_bytes().is_multiple_of(2) {
        return Err("Metal pointwise requires an exact aligned BF16 region".into());
    }
    Ok(())
}

fn tensor_port_region(
    participant: &TensorParticipant<'_>,
    role: ResolvedValueRole,
    ordinal: u32,
    element: ElementType,
    bytes: u64,
) -> Result<MetalBufferRegion, String> {
    let region = contiguous_region(
        participant,
        binding(participant.bindings(), role, ordinal)?,
        element,
    )?;
    if region.length_bytes() != bytes || !region.offset_bytes().is_multiple_of(element.size_bytes())
    {
        return Err("Metal tensor port requires an exact typed aligned range".into());
    }
    Ok(region)
}

fn set_region(encoder: &metal::ComputeCommandEncoderRef, index: u64, region: &MetalBufferRegion) {
    encoder.set_buffer(index, Some(region.buffer()), region.offset_bytes());
}
fn set_params<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, params: &T) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        (params as *const T).cast(),
    );
}
fn disjoint(read: &MetalBufferRegion, write: &MetalBufferRegion) -> Result<(), String> {
    if read
        .overlaps_shared_region(write)
        .map_err(|e| e.to_string())?
    {
        return Err("Metal tensor live input/output/scratch ranges overlap".into());
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    elements: u64,
    width: u64,
    kind: u32,
    row_broadcast: u32,
}
