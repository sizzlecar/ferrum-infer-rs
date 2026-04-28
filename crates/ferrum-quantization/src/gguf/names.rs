//! GGUF ↔ ferrum tensor-name translation.
//!
//! Ferrum models address weights using HuggingFace-style names
//! (`model.layers.0.self_attn.q_proj.weight`). GGUF files use llama.cpp's
//! shorthand (`blk.0.attn_q.weight`). This module is the single source of
//! truth for that mapping; both `GgufLoader` and any future tooling go
//! through `ferrum_to_gguf`.
//!
//! Scope (Phase 1C): dense Llama-family models — Qwen3, Qwen2.x, Llama-3.x,
//! Mistral, TinyLlama. MoE-specific names (`ffn_gate_inp`, `ffn_*_exps`)
//! land in Phase 2 alongside the MoE runtime.

/// Translate a ferrum tensor name to its GGUF equivalent.
///
/// Returns `None` for names that have no GGUF counterpart (yet) or aren't
/// recognised — caller treats this as "tensor not found".
///
/// Accepts both bare stems (`"lm_head"`, `"model.layers.0.self_attn.o_proj"`)
/// and fully-qualified names (`"...weight"`, `"...bias"`). The `.weight` /
/// `.bias` suffix passes through unchanged.
pub fn ferrum_to_gguf(name: &str) -> Option<String> {
    // Top-level tensors first — they don't fit the layer pattern.
    if let Some(out) = map_top_level(name) {
        return Some(out);
    }

    // Layer-scoped: must be "model.layers.{idx}.<rest>"
    let rest = name.strip_prefix("model.layers.")?;
    let (idx_str, after_idx) = rest.split_once('.')?;
    let idx: usize = idx_str.parse().ok()?;
    let mapped = map_layer_scoped(after_idx)?;
    Some(format!("blk.{idx}.{mapped}"))
}

fn map_top_level(name: &str) -> Option<String> {
    let mapped = match name {
        "model.embed_tokens" => "token_embd",
        "model.embed_tokens.weight" => "token_embd.weight",
        "model.norm" => "output_norm",
        "model.norm.weight" => "output_norm.weight",
        "lm_head" => "output",
        "lm_head.weight" => "output.weight",
        _ => return None,
    };
    Some(mapped.to_string())
}

fn map_layer_scoped(rest: &str) -> Option<String> {
    // Peel off the .weight / .bias suffix, map the stem, then re-attach.
    let (stem, suffix) = if let Some(s) = rest.strip_suffix(".weight") {
        (s, ".weight")
    } else if let Some(s) = rest.strip_suffix(".bias") {
        (s, ".bias")
    } else {
        (rest, "")
    };

    let mapped_stem = match stem {
        // RMSNorms
        "input_layernorm" => "attn_norm",
        "post_attention_layernorm" => "ffn_norm",
        // Attention projections
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        // Qwen3 QK-norm — only present on that family
        "self_attn.q_norm" => "attn_q_norm",
        "self_attn.k_norm" => "attn_k_norm",
        // MLP projections
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        _ => return None,
    };

    Some(format!("{mapped_stem}{suffix}"))
}

/// The three sub-tensor names that fuse into `qkv_proj`, in the order the
/// model expects them stacked along axis 0 (rows = output neurons).
pub fn qkv_split_parts(layer_prefix: &str) -> [String; 3] {
    [
        format!("{layer_prefix}self_attn.q_proj"),
        format!("{layer_prefix}self_attn.k_proj"),
        format!("{layer_prefix}self_attn.v_proj"),
    ]
}

/// The two sub-tensor names that fuse into `gate_up_proj`, stacked along
/// axis 0 (gate first, then up).
pub fn gate_up_split_parts(layer_prefix: &str) -> [String; 2] {
    [
        format!("{layer_prefix}mlp.gate_proj"),
        format!("{layer_prefix}mlp.up_proj"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_top_level_tensors() {
        assert_eq!(
            ferrum_to_gguf("model.embed_tokens.weight"),
            Some("token_embd.weight".into())
        );
        assert_eq!(
            ferrum_to_gguf("model.embed_tokens"),
            Some("token_embd".into())
        );
        assert_eq!(
            ferrum_to_gguf("model.norm.weight"),
            Some("output_norm.weight".into())
        );
        assert_eq!(ferrum_to_gguf("lm_head"), Some("output".into()));
        assert_eq!(
            ferrum_to_gguf("lm_head.weight"),
            Some("output.weight".into())
        );
    }

    #[test]
    fn maps_layer_attention_weights() {
        assert_eq!(
            ferrum_to_gguf("model.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".into())
        );
        assert_eq!(
            ferrum_to_gguf("model.layers.27.self_attn.k_proj.weight"),
            Some("blk.27.attn_k.weight".into())
        );
        assert_eq!(
            ferrum_to_gguf("model.layers.5.self_attn.v_proj.weight"),
            Some("blk.5.attn_v.weight".into())
        );
        assert_eq!(
            ferrum_to_gguf("model.layers.0.self_attn.o_proj.weight"),
            Some("blk.0.attn_output.weight".into())
        );
        // bare stem (load_linear-style)
        assert_eq!(
            ferrum_to_gguf("model.layers.0.self_attn.o_proj"),
            Some("blk.0.attn_output".into())
        );
    }

    #[test]
    fn maps_qwen3_qk_norm() {
        assert_eq!(
            ferrum_to_gguf("model.layers.0.self_attn.q_norm.weight"),
            Some("blk.0.attn_q_norm.weight".into())
        );
        assert_eq!(
            ferrum_to_gguf("model.layers.0.self_attn.k_norm.weight"),
            Some("blk.0.attn_k_norm.weight".into())
        );
    }

    #[test]
    fn maps_attention_bias() {
        assert_eq!(
            ferrum_to_gguf("model.layers.0.self_attn.q_proj.bias"),
            Some("blk.0.attn_q.bias".into())
        );
    }

    #[test]
    fn maps_layer_norms() {
        assert_eq!(
            ferrum_to_gguf("model.layers.0.input_layernorm.weight"),
            Some("blk.0.attn_norm.weight".into())
        );
        assert_eq!(
            ferrum_to_gguf("model.layers.0.post_attention_layernorm.weight"),
            Some("blk.0.ffn_norm.weight".into())
        );
    }

    #[test]
    fn maps_mlp_projections() {
        assert_eq!(
            ferrum_to_gguf("model.layers.0.mlp.gate_proj.weight"),
            Some("blk.0.ffn_gate.weight".into())
        );
        assert_eq!(
            ferrum_to_gguf("model.layers.0.mlp.up_proj.weight"),
            Some("blk.0.ffn_up.weight".into())
        );
        assert_eq!(
            ferrum_to_gguf("model.layers.0.mlp.down_proj.weight"),
            Some("blk.0.ffn_down.weight".into())
        );
    }

    #[test]
    fn rejects_unknown_names() {
        assert_eq!(ferrum_to_gguf("totally_made_up"), None);
        assert_eq!(ferrum_to_gguf("model.layers.0.unknown_part.weight"), None);
        assert_eq!(
            ferrum_to_gguf("model.layers.bad_idx.input_layernorm.weight"),
            None
        );
    }

    #[test]
    fn split_parts_helpers() {
        assert_eq!(
            qkv_split_parts("model.layers.3."),
            [
                "model.layers.3.self_attn.q_proj".to_string(),
                "model.layers.3.self_attn.k_proj".into(),
                "model.layers.3.self_attn.v_proj".into(),
            ]
        );
        assert_eq!(
            gate_up_split_parts("model.layers.3."),
            [
                "model.layers.3.mlp.gate_proj".to_string(),
                "model.layers.3.mlp.up_proj".into(),
            ]
        );
    }
}
