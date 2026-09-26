//! Family numerical declarations derived from typed semantics before a program
//! or backend exists. Source encodings constrain qualification; an explicit
//! profile controls the graph and never changes source tensor bytes.

use super::*;
use ferrum_interfaces::vnext::{
    CheckpointInputDependency, KvStateStorage, KvStorageFormat, NumericalOperationContract,
    NumericalProfileId, StateCheckpointCapability, StateCheckpointContents,
    StateCheckpointContract,
};

pub const F16_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f16";
pub const F32_MASTER_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f32-master";
pub const F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.gguf-f16-projections";
pub const F32_MASTER_GGUF_F16_RN_FRAGMENT_M1_TO8_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.gguf-f16-projections.ffn-rn-fragment-m1to8";
pub const F32_MASTER_GGUF_F16_ATTENTION_RESIDUAL2_FFN_M2TO8_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.gguf-f16-attention.q8-residual2-ffn-m2to8";
pub const F32_MASTER_F16_HEAD_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f32-master.f16-head";
pub const F32_MASTER_Q8_SWIGLU_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f32-master.q8-swiglu";
pub const F32_MASTER_Q8_SWIGLU_INPUT_SUM_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.q8-swiglu-input-sum";
pub const F32_MASTER_Q8_GATE_UP_STREAM_MMQ_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.q8-gate-up-stream-mmq";
pub const F32_MASTER_Q8_SWIGLU_GDN_PROJECTIONS_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.q8-swiglu-gdn-projections";
pub const F16_INT8_KV_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f16.int8-kv";
pub const F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f32-master.int8-kv";

pub(super) fn profiles(
    family_id: &ModelFamilyId,
    config: &Qwen35FamilyConfig,
) -> Result<FamilyNumericalProfiles, VNextError> {
    let text = Qwen35FamilyProvider::text_config(config)?;
    let (states, kv_storage) = states(&text, config.max_position_embeddings, KvStorageFormat::F16)?;
    let f16 = profile(family_id, &text, &states, &kv_storage, false, false)?;
    let f32 = profile(family_id, &text, &states, &kv_storage, true, false)?;
    let gguf_f16_projections = gguf_f16_projections_eligible(config, &text)
        .then(|| gguf_f16_projections_profile(&f32))
        .transpose()?;
    let rn_fragment = gguf_rn_f16_fragment_eligible(config, &text)
        .then(|| gguf_rn_f16_fragment_profile(&f32))
        .transpose()?;
    let hybrid = gguf_f16_attention_residual2_ffn_m2to8_eligible(config, &text)
        .then(|| gguf_f16_attention_residual2_ffn_m2to8_profile(&f32))
        .transpose()?;
    let f16_head = f16_head_eligible(config, &text)
        .then(|| f16_head_profile(&f32))
        .transpose()?;
    let q8_swiglu = q8_swiglu_eligible(config, &text)
        .then(|| q8_swiglu_profile(&f32))
        .transpose()?;
    let q8_input_sum = q8_swiglu_eligible(config, &text)
        .then(|| q8_input_sum_profile(&f32))
        .transpose()?;
    let q8_gate_up_stream_mmq = q8_gate_up_stream_mmq_eligible(config, &text)
        .then(|| q8_gate_up_stream_mmq_profile(&f32))
        .transpose()?;
    let q8_gdn_projections = q8_swiglu
        .as_ref()
        .filter(|_| q8_gdn_projections_eligible(config, &text))
        .map(q8_gdn_projections_profile)
        .transpose()?;
    // Qualification covers both physical encoding and recurrent parameter ABI.
    // In particular, an unquantized negative-rate source must not silently
    // acquire the as-yet unqualified F16 behavior merely because it has no blocks.
    let native = config.weights.iter().all(|weight| {
        matches!(
            weight.source_encoding,
            FamilyWeightSourceEncoding::Dense { .. }
                | FamilyWeightSourceEncoding::BlockQuantized(_)
        )
    });
    let portable = config.weights.iter().all(|weight| {
        !matches!(
            weight.source_encoding,
            FamilyWeightSourceEncoding::BlockQuantized(_)
        )
    });
    let mut automatic = match config.recurrent_weight_abi {
        RecurrentWeightAbi::LogRateGrouped if portable => vec![f16.id.clone()],
        RecurrentWeightAbi::NegativeRateInterleaved if native => vec![f32.id.clone()],
        _ => Vec::new(),
    };
    let mut profiles = vec![f16, f32];
    // Activation-rotation execution is qualified separately from KV encoding.
    // Until their combination has been validated, offer only F16 KV profiles
    // for source-declared Hadamard weights on both run and serve paths.
    if !kv_storage.is_empty() && config.gguf_hadamard.is_none() {
        let (states, kv_storage) = states_for_int8(&text, config.max_position_embeddings)?;
        let f16 = profile(family_id, &text, &states, &kv_storage, false, true)?;
        let f32 = profile(family_id, &text, &states, &kv_storage, true, true)?;
        match config.recurrent_weight_abi {
            RecurrentWeightAbi::LogRateGrouped if portable => automatic.push(f16.id.clone()),
            RecurrentWeightAbi::NegativeRateInterleaved if native => automatic.push(f32.id.clone()),
            _ => {}
        }
        profiles.extend([f16, f32]);
    }
    // This arithmetic policy is opt-in. In particular, keep both existing Auto
    // preference lists unchanged when suitable K-block leaves are present.
    profiles.extend(q8_swiglu);
    profiles.extend(q8_gdn_projections);
    profiles.extend(f16_head);
    profiles.extend(q8_input_sum);
    profiles.extend(q8_gate_up_stream_mmq);
    profiles.extend(gguf_f16_projections);
    profiles.extend(hybrid);
    profiles.extend(rn_fragment);
    FamilyNumericalProfiles::new(family_id, ContractVersion::new(1, 8), profiles, automatic)
}

/// This is a declared source/ABI combination, not full-model quality approval.
/// The materializer independently validates physical formats and all consumers.
pub(super) fn gguf_f16_projections_eligible(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> bool {
    config.weight_format == FamilyWeightFormat::GgufNative
        && text.moe.is_none()
        && config.gguf_hadamard.is_none()
        && config.recurrent_weight_abi == RecurrentWeightAbi::NegativeRateInterleaved
        && config.weights.iter().all(|weight| {
            let Some(layer) = weight.layer_index else { return true; };
            let consumed = match weight.role.as_str() {
                "mlp_gate" | "mlp_up" | "mlp_down" => true,
                "linear_attn_qkv" | "linear_attn_z" | "linear_attn_b" | "linear_attn_a" | "linear_attn_out" =>
                    text.layer_types.get(layer as usize) == Some(&Qwen35LayerType::LinearAttention),
                "self_attn_q" | "self_attn_k" | "self_attn_v" | "self_attn_o" =>
                    text.layer_types.get(layer as usize) == Some(&Qwen35LayerType::FullAttention),
                _ => false,
            };
            if !consumed { return true; }
            // Family roles are validated typed program inputs; this never
            // infers conversion authority from an external tensor name.
            match &weight.source_encoding {
                FamilyWeightSourceEncoding::Dense { element_type } => *element_type == ElementType::F16,
                FamilyWeightSourceEncoding::BlockQuantized(spec) => {
                    let block = match (spec.format_id.as_str(),spec.logical_values_per_block,spec.bytes_per_block) {
                        ("quantization.gguf.q4-k",256,144) | ("quantization.gguf.q5-k",256,176)
                            | ("quantization.gguf.q6-k",256,210) => 256,
                        ("quantization.gguf.q8-0",32,34) => 32,
                        _ => return false,
                    };
                    matches!(weight.dimensions.as_slice(), [n,k] if *n > 0 && *k > 0 && *k % block == 0)
                }
                _ => false,
            }
        })
}

/// Source/role/shape qualification; no model name or current batch controls
/// capability. Each whole gate/up packet has one source encoding.
pub(super) fn gguf_rn_f16_fragment_eligible(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> bool {
    if !gguf_f16_projections_eligible(config, text) {
        return false;
    }
    let classify =
        |weight: &FamilyWeight| -> Option<ferrum_interfaces::vnext::RnF16FragmentSourceFormatV1> {
            use ferrum_interfaces::vnext::{RnF16FragmentPlanV1, RnF16FragmentSourceFormatV1};
            let FamilyWeightSourceEncoding::BlockQuantized(spec) = &weight.source_encoding else {
                return None;
            };
            let format = match (
                spec.format_id.as_str(),
                spec.logical_values_per_block,
                spec.bytes_per_block,
            ) {
                ("quantization.gguf.q4-k", 256, 144) => RnF16FragmentSourceFormatV1::Q4K,
                ("quantization.gguf.q5-k", 256, 176) => RnF16FragmentSourceFormatV1::Q5K,
                ("quantization.gguf.q6-k", 256, 210) => RnF16FragmentSourceFormatV1::Q6K,
                _ => return None,
            };
            RnF16FragmentPlanV1::from_dimensions(format, &weight.dimensions).ok()?;
            Some(format)
        };
    (0..text.num_hidden_layers).all(|layer| {
        let find = |role| {
            config
                .weights
                .iter()
                .find(|w| w.layer_index == Some(layer as u32) && w.role == role)
        };
        let (Some(gate), Some(up), Some(down)) =
            (find("mlp_gate"), find("mlp_up"), find("mlp_down"))
        else {
            return false;
        };
        classify(gate).is_some_and(|format| {
            if classify(up) != Some(format) || gate.dimensions != up.dimensions {
                return false;
            }
            let mut whole = gate.dimensions.clone();
            let Some(n) = whole.first_mut() else {
                return false;
            };
            let Some(joined) = n.checked_mul(2) else {
                return false;
            };
            *n = joined;
            ferrum_interfaces::vnext::RnF16FragmentPlanV1::from_dimensions(format, &whole).is_ok()
        }) && classify(down).is_some()
    })
}

fn gguf_rn_f16_fragment_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = gguf_f16_projections_profile(master)?;
    profile.id =
        NumericalProfileId::new(F32_MASTER_GGUF_F16_RN_FRAGMENT_M1_TO8_NUMERICAL_PROFILE_ID)
            .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    let ffn = profile
        .operations
        .iter_mut()
        .find(|op| op.operation_id.as_str() == DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID)
        .ok_or_else(|| {
            invalid_config("numerical_profile.operations", "RN FFN contract is missing")
        })?;
    ffn.operation_id = operation_id(DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID)?;
    profile
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    Ok(profile)
}

/// One explicit combination, not permission to combine independently qualified
/// policies. At least one real Q4 gate/up pair can exercise residual2; other
/// leaves retain the operation's declared strict fallback. Attention alone is
/// rounded by the typed materializer. Whole M, never leaf M, selects FFN math.
pub(super) fn gguf_f16_attention_residual2_ffn_m2to8_eligible(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> bool {
    gguf_f16_projections_eligible(config, text) && q8_gate_up_stream_mmq_eligible(config, text)
}

fn gguf_f16_attention_residual2_ffn_m2to8_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = gguf_f16_projections_profile(master)?;
    profile.id = NumericalProfileId::new(
        F32_MASTER_GGUF_F16_ATTENTION_RESIDUAL2_FFN_M2TO8_NUMERICAL_PROFILE_ID,
    )
    .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    let ffn = profile
        .operations
        .iter_mut()
        .find(|operation| {
            operation.operation_id.as_str() == DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID
        })
        .ok_or_else(|| {
            invalid_config("numerical_profile.operations", "RN FFN contract is missing")
        })?;
    ffn.operation_id = operation_id(DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_OPERATION_ID)?;
    // This describes the eligible integer products; the contract separately
    // fixes residual reconstruction and the strict compressed fallback.
    ffn.multiplication_type = Some(ElementType::I8);
    ffn.accumulation_type = Some(ElementType::F32);
    profile
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    Ok(profile)
}

fn gguf_f16_projections_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = master.clone();
    profile.id = NumericalProfileId::new(F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    for operation in &mut profile.operations {
        let replacement = match operation.operation_id.as_str() {
            DENSE_SWIGLU_OPERATION_ID => DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID,
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID => {
                GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID
            }
            CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID => {
                CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID
            }
            _ => continue,
        };
        operation.operation_id = operation_id(replacement)?;
        operation.version = ContractVersion::new(1, 0);
        // This summarizes the projections, not the F32 recurrent/attention core.
        operation.multiplication_type = Some(ElementType::F16);
        operation.accumulation_type = Some(ElementType::F32);
    }
    profile
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    Ok(profile)
}

pub(super) fn f16_head_eligible(config: &Qwen35FamilyConfig, text: &Qwen35TextConfig) -> bool {
    config.gguf_hadamard.is_none()
        && config.recurrent_weight_abi == RecurrentWeightAbi::NegativeRateInterleaved
        && config.weights.iter().all(|weight| {
            matches!(
                weight.source_encoding,
                FamilyWeightSourceEncoding::Dense { .. }
                    | FamilyWeightSourceEncoding::BlockQuantized(_)
            )
        })
        && output_projection_weight(config).is_ok_and(|weight| {
            let FamilyWeightSourceEncoding::BlockQuantized(spec) = &weight.source_encoding else {
                return false;
            };
            spec.format_id.as_str() == "quantization.gguf.q6-k"
                && spec.logical_values_per_block == 256
                && spec.bytes_per_block == 210
                && matches!(weight.dimensions.as_slice(), [rows, columns]
                    if *rows == config.vocab_size && *rows > 0
                        && *columns == text.hidden_size as u64
                        && *columns > 0 && *columns % 256 == 0)
        })
}

fn f16_head_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = master.clone();
    profile.id = NumericalProfileId::new(F32_MASTER_F16_HEAD_NUMERICAL_PROFILE_ID)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    let head = profile
        .operations
        .iter_mut()
        .find(|operation| {
            operation.operation_id.as_str() == LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID
        })
        .ok_or_else(|| {
            invalid_config(
                "numerical_profile.operations",
                "F32 output head contract is missing",
            )
        })?;
    head.operation_id = operation_id(LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID)?;
    head.version = ContractVersion::new(1, 0);
    head.multiplication_type = Some(ElementType::F16);
    head.accumulation_type = Some(ElementType::F32);
    profile
        .operations
        .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
    Ok(profile)
}

pub(super) fn q8_swiglu_eligible(config: &Qwen35FamilyConfig, text: &Qwen35TextConfig) -> bool {
    text.moe.is_none()
        && config.gguf_hadamard.is_none()
        && config.recurrent_weight_abi == RecurrentWeightAbi::NegativeRateInterleaved
        && config.weights.iter().all(|weight| {
            matches!(
                weight.source_encoding,
                FamilyWeightSourceEncoding::Dense { .. }
                    | FamilyWeightSourceEncoding::BlockQuantized(_)
            )
        })
        && config.weights.iter().any(|weight| {
            if weight.layer_index.is_none()
                || !matches!(weight.role.as_str(), "mlp_gate" | "mlp_up" | "mlp_down")
            {
                return false;
            }
            // Qualify physical leaves, not a checkpoint/container name. Other
            // native leaves keep their original arithmetic, including a dense
            // projection whose input width is not a multiple of 256.
            q8_k_leaf(weight)
        })
}

pub(super) fn q8_gate_up_stream_mmq_eligible(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> bool {
    // Family eligibility establishes a possible real pair. Actual packed row
    // count and physical leaf/layout/transform checks belong to the provider.
    q8_swiglu_eligible(config, text)
        && config.weights.iter().any(|gate| {
            let Some(layer) = gate.layer_index else {
                return false;
            };
            if gate.role != "mlp_gate" || !q4_k_leaf(gate) {
                return false;
            }
            config.weights.iter().any(|up| {
                up.layer_index == Some(layer)
                    && up.role == "mlp_up"
                    && up.dimensions == gate.dimensions
                    && q4_k_leaf(up)
            })
        })
}

fn q4_k_leaf(weight: &FamilyWeight) -> bool {
    matches!(&weight.source_encoding, FamilyWeightSourceEncoding::BlockQuantized(spec)
        if spec.format_id.as_str() == "quantization.gguf.q4-k"
            && spec.logical_values_per_block == 256 && spec.bytes_per_block == 144)
        && matches!(weight.dimensions.as_slice(), [n, k] if *n > 0 && *k > 0 && *k % 256 == 0)
}

fn q8_k_leaf(weight: &FamilyWeight) -> bool {
    let FamilyWeightSourceEncoding::BlockQuantized(spec) = &weight.source_encoding else {
        return false;
    };
    matches!(
        (
            spec.format_id.as_str(),
            spec.logical_values_per_block,
            spec.bytes_per_block
        ),
        ("quantization.gguf.q4-k", 256, 144)
            | ("quantization.gguf.q5-k", 256, 176)
            | ("quantization.gguf.q6-k", 256, 210)
    ) && matches!(weight.dimensions.as_slice(), [rows, columns] if *rows > 0 && *columns > 0 && *columns % 256 == 0)
}

pub(super) fn q8_gdn_projections_eligible(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> bool {
    q8_swiglu_eligible(config, text)
        && config.weights.iter().any(|weight| {
            // Input ordinal 2 is the logical stack of these four physical
            // leaves; ordinal 7 is the output projection. A matching role on
            // an absent/full-attention layer does not qualify a GDN operation.
            weight.layer_index.is_some_and(|layer| {
                text.layer_types.get(layer as usize) == Some(&Qwen35LayerType::LinearAttention)
            }) && matches!(
                weight.role.as_str(),
                "linear_attn_qkv"
                    | "linear_attn_z"
                    | "linear_attn_b"
                    | "linear_attn_a"
                    | "linear_attn_out"
            ) && q8_k_leaf(weight)
        })
}

fn q8_swiglu_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = master.clone();
    profile.id = NumericalProfileId::new(F32_MASTER_Q8_SWIGLU_NUMERICAL_PROFILE_ID)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    let dense = profile
        .operations
        .iter_mut()
        .find(|operation| operation.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID)
        .ok_or_else(|| {
            invalid_config(
                "numerical_profile.operations",
                "dense FFN contract is missing",
            )
        })?;
    dense.operation_id = operation_id(DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID)?;
    dense.version = ContractVersion::new(1, 0);
    dense.multiplication_type = Some(ElementType::I8);
    dense.accumulation_type = Some(ElementType::F32);
    profile
        .operations
        .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
    Ok(profile)
}

fn q8_input_sum_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    // Start with the existing FFN-only policy; neither GDN projections nor the
    // output head inherit this new arithmetic. Old profile bytes stay intact.
    let mut profile = q8_swiglu_profile(master)?;
    profile.id = NumericalProfileId::new(F32_MASTER_Q8_SWIGLU_INPUT_SUM_NUMERICAL_PROFILE_ID)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    let dense = profile
        .operations
        .iter_mut()
        .find(|operation| operation.operation_id.as_str() == DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID)
        .ok_or_else(|| {
            invalid_config("numerical_profile.operations", "Q8 FFN contract is missing")
        })?;
    dense.operation_id = operation_id(DENSE_SWIGLU_Q8_F32SCALE_INPUT_SUM_OPERATION_ID)?;
    profile
        .operations
        .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
    Ok(profile)
}

fn q8_gate_up_stream_mmq_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = master.clone();
    profile.id = NumericalProfileId::new(F32_MASTER_Q8_GATE_UP_STREAM_MMQ_NUMERICAL_PROFILE_ID)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    let dense = profile
        .operations
        .iter_mut()
        .find(|o| o.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID)
        .ok_or_else(|| {
            invalid_config(
                "numerical_profile.operations",
                "dense FFN contract is missing",
            )
        })?;
    dense.operation_id = operation_id(DENSE_SWIGLU_Q8_GATE_UP_STREAM_MMQ_OPERATION_ID)?;
    // This marks the operation's eligible integer projections, not the strict
    // down projection or a promise that every invocation uses the integer path.
    dense.multiplication_type = Some(ElementType::I8);
    dense.accumulation_type = Some(ElementType::F32);
    profile
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    Ok(profile)
}

fn q8_gdn_projections_profile(
    swiglu: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = swiglu.clone();
    profile.id = NumericalProfileId::new(F32_MASTER_Q8_SWIGLU_GDN_PROJECTIONS_NUMERICAL_PROFILE_ID)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    let gdn = profile
        .operations
        .iter_mut()
        .find(|operation| {
            operation.operation_id.as_str()
                == GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID
        })
        .ok_or_else(|| invalid_config("numerical_profile.operations", "GDN contract is missing"))?;
    gdn.operation_id =
        operation_id(GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q8_PROJECTIONS_OPERATION_ID)?;
    gdn.version = ContractVersion::new(1, 0);
    gdn.multiplication_type = Some(ElementType::I8);
    gdn.accumulation_type = Some(ElementType::F32);
    profile
        .operations
        .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
    Ok(profile)
}

fn profile(
    family_id: &ModelFamilyId,
    text: &Qwen35TextConfig,
    states: &[StateSpec],
    kv_storage: &[KvStateStorage],
    master: bool,
    int8_kv: bool,
) -> Result<NumericalExecutionProfile, VNextError> {
    let (name, selections, activation) = if master {
        (
            if int8_kv {
                F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID
            } else {
                F32_MASTER_NUMERICAL_PROFILE_ID
            },
            if int8_kv {
                Qwen35OperationProfile::F32_MASTER_INT8_KV
            } else {
                Qwen35OperationProfile::F32_MASTER
            },
            ElementType::F32,
        )
    } else {
        (
            if int8_kv {
                F16_INT8_KV_NUMERICAL_PROFILE_ID
            } else {
                F16_NUMERICAL_PROFILE_ID
            },
            if int8_kv {
                Qwen35OperationProfile::F16_INT8_KV
            } else {
                Qwen35OperationProfile::F16
            },
            ElementType::F16,
        )
    };
    let primary_activation = value_id("value.hidden.embedding")?;
    let mut boundaries = BTreeMap::from([
        (primary_activation.clone(), activation),
        (value_id("value.output.final_hidden")?, activation),
        (value_id("value.output.logits")?, activation),
        (value_id("value.output.greedy_token")?, ElementType::U32),
    ]);
    let mut operations = BTreeMap::new();
    let mut declare =
        |selection: OperationSelection, multiply, accumulate| -> Result<(), VNextError> {
            let operation_id = operation_id(selection.id)?;
            operations.insert(
                operation_id.clone(),
                NumericalOperationContract {
                    operation_id,
                    version: selection.version,
                    multiplication_type: multiply,
                    accumulation_type: accumulate,
                },
            );
            Ok(())
        };
    declare(selections.token_embedding, None, None)?;
    declare(
        selections.final_norm,
        Some(ElementType::F32),
        Some(ElementType::F32),
    )?;
    declare(
        selections.logits,
        Some(ElementType::F32),
        Some(ElementType::F32),
    )?;
    // Penalty arithmetic is part of this operation; the selected token itself
    // is U32. It must not be reported as a floating-point logit boundary.
    declare(selections.argmax, Some(ElementType::F32), None)?;
    for (index, layer) in text.layer_types.iter().enumerate() {
        for (role, dtype) in [
            ("attention", activation),
            ("post_attention_norm", ElementType::F16),
            ("mlp", ElementType::F16),
            ("output", activation),
        ] {
            boundaries.insert(value_id(format!("value.layer.{index}.{role}"))?, dtype);
        }
        let attention = match layer {
            Qwen35LayerType::LinearAttention => selections.linear_attention,
            Qwen35LayerType::FullAttention => selections.causal_attention,
        };
        declare(attention, Some(ElementType::F32), Some(ElementType::F32))?;
        declare(
            selections.post_attention_norm,
            Some(ElementType::F32),
            Some(ElementType::F32),
        )?;
        let (feed_forward, residual) = if text.moe.is_some() {
            (
                OperationSelection::new(ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID, 1, 0),
                selections.moe_residual,
            )
        } else {
            (selections.dense_feed_forward, selections.dense_residual)
        };
        declare(feed_forward, Some(ElementType::F32), Some(ElementType::F32))?;
        declare(residual, None, Some(ElementType::F32))?;
    }
    Ok(NumericalExecutionProfile {
        id: NumericalProfileId::new(name)
            .map_err(|reason| invalid_config("numerical_profile.id", reason))?,
        version: ContractVersion::new(1, 0),
        family_id: family_id.clone(),
        primary_activation,
        boundaries,
        states: states.to_vec(),
        kv_storage: kv_storage.to_vec(),
        operations: operations.into_values().collect(),
    })
}

fn states_for_int8(
    text: &Qwen35TextConfig,
    maximum_tokens: u64,
) -> Result<(Vec<StateSpec>, Vec<KvStateStorage>), VNextError> {
    states(
        text,
        maximum_tokens,
        KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
    )
}

fn states(
    text: &Qwen35TextConfig,
    maximum_tokens: u64,
    storage: KvStorageFormat,
) -> Result<(Vec<StateSpec>, Vec<KvStateStorage>), VNextError> {
    let mut states = Vec::new();
    let mut kv_storage = Vec::new();
    for (index, layer) in text.layer_types.iter().enumerate() {
        match layer {
            Qwen35LayerType::LinearAttention => {
                for (role, dimensions, dtype) in [
                    (
                        "conv",
                        text.recurrent_conv_state_shape()
                            .map_err(|reason| invalid_config("states.conv", reason))?
                            .to_vec(),
                        ElementType::F16,
                    ),
                    (
                        "delta",
                        text.recurrent_delta_state_shape()
                            .map_err(|reason| invalid_config("states.delta", reason))?
                            .to_vec(),
                        data_type_to_element_type(text.mamba_ssm_dtype)
                            .map_err(|reason| invalid_config("states.delta.dtype", reason))?,
                    ),
                ] {
                    states.push(StateSpec {
                        id: state_id(format!("state.layer.{index}.{role}"))?,
                        value_id: value_id(format!("value.state.layer.{index}.{role}"))?,
                        tensor: tensor_spec(
                            dimensions.into_iter().map(|extent| extent as u64).collect(),
                            dtype,
                        ),
                        lifetime: StateLifetime::Sequence,
                        capacity_demand: StateCapacityDemand::FixedPerScope,
                        initialization: StateInitialization::Zero,
                        // The convolution window and recurrent accumulator
                        // together contain the complete state at this boundary.
                        checkpoint: StateCheckpointCapability::CompletedBoundary(
                            StateCheckpointContract::new(
                                StateCheckpointContents::BoundaryValue,
                                CheckpointInputDependency::ExactTokenPrefix,
                            ),
                        ),
                    });
                }
            }
            Qwen35LayerType::FullAttention => {
                let checkpoint =
                    StateCheckpointCapability::CompletedBoundary(StateCheckpointContract::new(
                        StateCheckpointContents::PrefixPositions,
                        CheckpointInputDependency::ExactTokenPrefix,
                    ));
                if storage == KvStorageFormat::F16 {
                    let mut state = crate::vnext::numerical::kv_state(
                        index as u64,
                        text.num_key_value_heads as u64,
                        text.head_dim as u64,
                        maximum_tokens,
                    )?;
                    state.checkpoint = checkpoint;
                    kv_storage.push(KvStateStorage::F16 {
                        state: state.id.clone(),
                    });
                    states.push(state);
                } else {
                    let (quantized, declaration) = crate::vnext::numerical::int8_kv_states(
                        index as u64,
                        text.num_key_value_heads as u64,
                        text.head_dim as u64,
                        maximum_tokens,
                        checkpoint,
                    )?;
                    states.extend(quantized);
                    kv_storage.push(declaration);
                }
            }
        }
    }
    Ok((states, kv_storage))
}
