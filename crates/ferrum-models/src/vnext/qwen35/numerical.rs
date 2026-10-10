//! Family numerical declarations derived from typed semantics before a program
//! or backend exists. Source encodings only constrain Auto qualification; an
//! explicit profile controls the graph and never changes source tensor bytes.

use super::*;
use ferrum_interfaces::vnext::{
    CheckpointInputDependency, DeclaredProjectionArithmetic, KvStateStorage, KvStorageFormat,
    NumericalOperationContract, NumericalProfileId, ProjectionRole, Q8ActAttentionProfile,
    Q8ActSwiGluProfile, StateCheckpointCapability, StateCheckpointContents,
    StateCheckpointContract,
};

pub const F32_MASTER_FFN_ATTENTION_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f32-master.ffn-attention-q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-prefill";
pub const F32_MASTER_FFN_ATTENTION_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f32-master.ffn-attention-q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-prefill";
pub const F32_MASTER_FFN_ATTENTION_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f32-master.ffn-attention-q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-all-rows";

pub const F16_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f16";
pub const F32_MASTER_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f32-master";
pub const F32_MASTER_GGUF_Q8_HEAD_V1_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.gguf-q8-head-v1";
pub const F32_MASTER_Q6_HEAD_ATTENTION_M8_MMQ_GEOMETRY_V1_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.gguf-q6-head.attention-m8-mmq-geometry-v1";
pub const F32_MASTER_FFN_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.ffn-iq4xs-q8act-g32";
pub const F32_MASTER_FFN_Q4K_Q5K_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.ffn-q4k-q5k-iq4xs-q8act-g32";
pub const F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.ffn-attention-q4k-q5k-iq4xs-q8act-g32";
pub const F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.ffn-attention-q4k-q5k-iq4xs-upstream-marker-v2";
pub const F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_NUMERICAL_PROFILE_ID:
    &str = "qwen3_5.f32-master.ffn-attention-q4k-q5k-iq4xs-upstream-marker-v2-prefill";
pub const F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_NUMERICAL_PROFILE_ID:
    &str = "qwen3_5.f32-master.ffn-attention-q4k-q5k-iq4xs-g32-mmq-prefill-marker-v1";
const F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.gguf-f16-projections";
pub const F32_MASTER_GGUF_F16_RN_FRAGMENT_M1_TO8_NUMERICAL_PROFILE_ID: &str =
    "qwen3_5.f32-master.gguf-f16-projections.ffn-rn-fragment-m1to8";
pub const F16_INT8_KV_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f16.int8-kv";
pub const F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f32-master.int8-kv";

mod attention_geometry;
mod q6_head;
#[cfg(test)]
mod q8act_attention_tests;
#[cfg(test)]
mod q8act_tests;
mod upstream_marker_v2;
pub(super) use q6_head::eligible as q6_head_eligible;
pub(super) use upstream_marker_v2::eligible as upstream_marker_v2_eligible;
pub(super) use upstream_marker_v2::extra_eligible as upstream_extra_marker_v2_eligible;

pub(super) fn profiles(
    family_id: &ModelFamilyId,
    config: &Qwen35FamilyConfig,
) -> Result<FamilyNumericalProfiles, VNextError> {
    let text = Qwen35FamilyProvider::text_config(config)?;
    let (states, kv_storage) = states(&text, config.max_position_embeddings, KvStorageFormat::F16)?;
    let f16 = profile(family_id, &text, &states, &kv_storage, false, false)?;
    let f32 = profile(family_id, &text, &states, &kv_storage, true, false)?;
    let rn_fragment = gguf_rn_f16_fragment_eligible(config, &text)
        .then(|| gguf_rn_f16_fragment_profile(&f32))
        .transpose()?;
    let q8act = [Q8ActSwiGluProfile::Iq4Xs, Q8ActSwiGluProfile::Q4KQ5KIq4Xs]
        .into_iter()
        .filter(|&profile| q8act_ffn_eligible_for(config, &text, profile))
        .map(|profile| q8act_ffn_profile(&f32, profile))
        .collect::<Result<Vec<_>, _>>()?;
    let q8act_attention = q8act_attention_eligible(config, &text)
        .then(|| q8act_attention_profile(&f32))
        .transpose()?;
    let upstream_marker = upstream_marker_v2_eligible(config, &text)
        .then(|| upstream_marker_v2::profile(&f32))
        .transpose()?;
    let upstream_prefill = upstream_marker_v2_eligible(config, &text)
        .then(|| upstream_marker_v2::profile_for(&f32, true))
        .transpose()?;
    let hybrid = upstream_marker_v2_eligible(config, &text)
        .then(|| upstream_marker_v2::hybrid_profile(&f32))
        .transpose()?;
    let upstream_extra = upstream_extra_marker_v2_eligible(config, &text)
        .then(|| upstream_marker_v2::extra_profile(&f32))
        .transpose()?;
    let upstream_extra_prefill = upstream_extra_marker_v2_eligible(config, &text)
        .then(|| upstream_marker_v2::extra_large_prefill_profile(&f32))
        .transpose()?;
    let upstream_extra_all_rows = upstream_extra_marker_v2_eligible(config, &text)
        .then(|| upstream_marker_v2::extra_all_rows_profile(&f32))
        .transpose()?;
    let q6_head = q6_head_eligible(config, &text)
        .then(|| q6_head::profile(&f32))
        .transpose()?;
    let attention_geometry = q6_head_eligible(config, &text)
        .then(|| attention_geometry::profile(&f32))
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
    // Explicit opt-in: Auto retains its existing numerical policy.
    profiles.extend(rn_fragment);
    profiles.extend(q8act);
    profiles.extend(q8act_attention);
    profiles.extend(upstream_marker);
    profiles.extend(upstream_prefill);
    profiles.extend(hybrid);
    profiles.extend(upstream_extra);
    profiles.extend(upstream_extra_prefill);
    profiles.extend(upstream_extra_all_rows);
    profiles.extend(q6_head);
    profiles.extend(attention_geometry);
    FamilyNumericalProfiles::new(family_id, ContractVersion::new(1, 2), profiles, automatic)
}

pub(super) fn q8act_attention_eligible(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> bool {
    config.weight_format == FamilyWeightFormat::GgufNative
        && text.moe.is_none()
        && config.recurrent_weight_abi == RecurrentWeightAbi::NegativeRateInterleaved
        && config.gguf_hadamard.is_none()
        && config.weights.iter().any(|weight| {
            if !weight
                .layer_index
                .is_some_and(|layer| (layer as usize) < text.num_hidden_layers)
            {
                return false;
            }
            let (kind, role) = match weight.role.as_str() {
                "linear_attn_qkv" | "linear_attn_z" | "linear_attn_a" | "linear_attn_b" => (
                    Q8ActAttentionProfile::GatedDelta,
                    ProjectionRole::GatedDeltaInput,
                ),
                "linear_attn_out" => (
                    Q8ActAttentionProfile::GatedDelta,
                    ProjectionRole::GatedDeltaOutput,
                ),
                "self_attn_q" => (Q8ActAttentionProfile::Causal, ProjectionRole::CausalQuery),
                "self_attn_k" => (Q8ActAttentionProfile::Causal, ProjectionRole::CausalKey),
                "self_attn_v" => (Q8ActAttentionProfile::Causal, ProjectionRole::CausalValue),
                "self_attn_o" => (Q8ActAttentionProfile::Causal, ProjectionRole::CausalOutput),
                _ => return false,
            };
            let layer = weight.layer_index.unwrap() as usize;
            let expected = match kind {
                Q8ActAttentionProfile::GatedDelta => Qwen35LayerType::LinearAttention,
                Q8ActAttentionProfile::Causal => Qwen35LayerType::FullAttention,
            };
            if text.layer_types.get(layer) != Some(&expected) {
                return false;
            }
            let FamilyWeightSourceEncoding::BlockQuantized(spec) = &weight.source_encoding else {
                return false;
            };
            let [n, k] = weight.dimensions.as_slice() else {
                return false;
            };
            matches!(
                kind.arithmetic()
                    .declared_projection_arithmetic(role, Some(spec), *k, *n, false),
                Ok(DeclaredProjectionArithmetic::Staged(_))
            )
        })
}

fn q8act_attention_profile(
    master: &NumericalExecutionProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    // The established FFN contract is reused verbatim, not broadened in place.
    let mut profile = q8act_ffn_profile(master, Q8ActSwiGluProfile::Q4KQ5KIq4Xs)?;
    profile.id = NumericalProfileId::new(
        F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID,
    )
    .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    for (base, kind) in [
        (
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
            Q8ActAttentionProfile::GatedDelta,
        ),
        (
            CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
            Q8ActAttentionProfile::Causal,
        ),
    ] {
        if let Some(operation) = profile
            .operations
            .iter_mut()
            .find(|op| op.operation_id.as_str() == base)
        {
            operation.operation_id = operation_id(kind.operation_id())?;
            operation.version = ContractVersion::new(1, 0);
            operation.multiplication_type = None;
            operation.accumulation_type = None;
            operation.composite_arithmetic = Some(kind.arithmetic());
        }
    }
    profile
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    profile.validate()?;
    Ok(profile)
}

pub(super) fn q8act_profile_kind(id: &str) -> Option<Q8ActSwiGluProfile> {
    match id {
        F32_MASTER_FFN_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID => Some(Q8ActSwiGluProfile::Iq4Xs),
        F32_MASTER_FFN_Q4K_Q5K_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID => {
            Some(Q8ActSwiGluProfile::Q4KQ5KIq4Xs)
        }
        _ => None,
    }
}

pub(super) fn q8act_ffn_eligible_for(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
    profile: Q8ActSwiGluProfile,
) -> bool {
    let arithmetic = profile.arithmetic();
    config.weight_format == FamilyWeightFormat::GgufNative
        && text.moe.is_none()
        && config.recurrent_weight_abi == RecurrentWeightAbi::NegativeRateInterleaved
        // Family registration admits only the unrotated source ABI.
        // The provider still declares strict fallback for transformed leaves.
        && config.gguf_hadamard.is_none()
        && config.weights.iter().any(|weight| {
            if !weight.layer_index.is_some_and(|layer| (layer as usize) < text.num_hidden_layers) {
                return false;
            }
            let role = match weight.role.as_str() {
                "mlp_gate" | "mlp_up" => ProjectionRole::SwiGluGateUp,
                "mlp_down" => ProjectionRole::SwiGluDown,
                _ => return false,
            };
            let FamilyWeightSourceEncoding::BlockQuantized(spec) = &weight.source_encoding else {
                return false;
            };
            let [n, k] = weight.dimensions.as_slice() else { return false; };
            matches!(arithmetic.declared_projection_arithmetic(role, Some(spec), *k, *n, false),
                Ok(DeclaredProjectionArithmetic::Staged(_)))
        })
}

fn q8act_ffn_profile(
    master: &NumericalExecutionProfile,
    kind: Q8ActSwiGluProfile,
) -> Result<NumericalExecutionProfile, VNextError> {
    let mut profile = master.clone();
    let profile_id = match kind {
        Q8ActSwiGluProfile::Iq4Xs => F32_MASTER_FFN_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID,
        Q8ActSwiGluProfile::Q4KQ5KIq4Xs => {
            F32_MASTER_FFN_Q4K_Q5K_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID
        }
    };
    profile.id = NumericalProfileId::new(profile_id)
        .map_err(|reason| invalid_config("numerical_profile.id", reason))?;
    let ffn = profile
        .operations
        .iter_mut()
        .find(|op| op.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID)
        .ok_or_else(|| {
            invalid_config(
                "numerical_profile.operations",
                "Q8act FFN contract is missing",
            )
        })?;
    ffn.operation_id = operation_id(kind.operation_id())?;
    ffn.version = ContractVersion::new(1, 0);
    ffn.multiplication_type = None;
    ffn.accumulation_type = None;
    ffn.composite_arithmetic = Some(kind.arithmetic());
    profile
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    profile.validate()?;
    Ok(profile)
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
                    staged_arithmetic: None,
                    composite_arithmetic: None,
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
