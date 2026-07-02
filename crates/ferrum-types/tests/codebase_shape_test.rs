use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

struct ShapeLimit {
    path: &'static str,
    max_lines: usize,
}

const SHAPE_LIMITS: &[ShapeLimit] = &[
    ShapeLimit {
        path: "crates/ferrum-engine/src/continuous_engine.rs",
        max_lines: 2347,
    },
    ShapeLimit {
        path: "crates/ferrum-models/src/models/qwen3_moe.rs",
        max_lines: 150,
    },
    ShapeLimit {
        path: "crates/ferrum-kernels/src/backend/traits.rs",
        max_lines: 2322,
    },
    ShapeLimit {
        path: "crates/ferrum-models/src/models/qwen3_moe_forward_unified.rs",
        max_lines: 445,
    },
];

const HOT_PATH_ROOTS: &[&str] = &[
    "crates/ferrum-models/src/models",
    "crates/ferrum-models/src/moe",
    "crates/ferrum-kernels/src/backend",
];

const MAX_HOT_PATH_PARAMS: usize = 15;

const LEGACY_LONG_SIGNATURES: &[&str] = &[
    "crates/ferrum-kernels/src/backend/cpu.rs::<root>::fn validate_linear_attention_decode_prepare_shape (18)",
    "crates/ferrum-kernels/src/backend/cpu.rs::<root>::fn validate_linear_attention_prepare_shape (17)",
    "crates/ferrum-kernels/src/backend/cpu.rs::impl CpuBackend::method linear_attention_decode_prepare_batch_f32 (21)",
    "crates/ferrum-kernels/src/backend/cpu.rs::impl CpuBackend::method linear_attention_decode_prepare_f32 (20)",
    "crates/ferrum-kernels/src/backend/cpu.rs::impl CpuBackend::method linear_attention_prepare_f32 (19)",
    "crates/ferrum-kernels/src/backend/cpu.rs::impl CpuBackend::method linear_attention_prepare_varlen_f32 (24)",
    "crates/ferrum-kernels/src/backend/cpu.rs::impl CpuBackend::method linear_attention_prepare_varlen_packed_qkvz_ba_f32 (24)",
    "crates/ferrum-kernels/src/backend/cpu.rs::impl CpuBackend::method qk_norm_rope_partial (16)",
    "crates/ferrum-kernels/src/backend/cpu.rs::impl CpuBackend::method recurrent_gated_delta_rule_batch_f32 (16)",
    "crates/ferrum-kernels/src/backend/cpu.rs::impl CpuBackend::method recurrent_gated_delta_rule_f32 (16)",
    "crates/ferrum-kernels/src/backend/cpu.rs::impl CpuBackend::method recurrent_gated_delta_rule_varlen_f32 (18)",
    "crates/ferrum-kernels/src/backend/cuda/cuda_decode.rs::impl CudaDecodeRunner::method launch_paged_varlen_attention (19)",
    "crates/ferrum-kernels/src/backend/cuda/fa2_ffi.rs::<root>::fn call_paged_varlen_fn (19)",
    "crates/ferrum-kernels/src/backend/cuda/fa2_ffi.rs::<root>::fn paged_varlen_attention_fa2_ffi (18)",
    "crates/ferrum-kernels/src/backend/cuda/gated_delta_rule.rs::<root>::fn recurrent_gated_delta_rule_batch_f32 (16)",
    "crates/ferrum-kernels/src/backend/cuda/gated_delta_rule.rs::<root>::fn recurrent_gated_delta_rule_batch_indexed_f32 (17)",
    "crates/ferrum-kernels/src/backend/cuda/gated_delta_rule.rs::<root>::fn recurrent_gated_delta_rule_f32 (16)",
    "crates/ferrum-kernels/src/backend/cuda/gated_delta_rule.rs::<root>::fn recurrent_gated_delta_rule_varlen_f32 (18)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn linear_attention_decode_prepare_batch_f32 (21)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn linear_attention_decode_prepare_batch_indexed_f32 (22)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32 (22)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn linear_attention_decode_prepare_f32 (20)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn linear_attention_prepare_f32 (19)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn linear_attention_prepare_varlen_f32 (24)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn linear_attention_prepare_varlen_packed_qkvz_ba_f32 (24)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn validate_decode_prepare_batch_indexed_packed_shape (20)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn validate_decode_prepare_batch_indexed_shape (20)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn validate_decode_prepare_batch_shape (19)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn validate_decode_prepare_shape (18)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn validate_prepare_shape (17)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn validate_prepare_varlen_packed_shape (22)",
    "crates/ferrum-kernels/src/backend/cuda/linear_attention.rs::<root>::fn validate_prepare_varlen_shape (22)",
    "crates/ferrum-kernels/src/backend/cuda/marlin.rs::<root>::fn marlin_gemm_moe_vllm (16)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method linear_attention_decode_prepare_batch_f32 (21)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method linear_attention_decode_prepare_batch_indexed_f32 (22)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32 (22)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method linear_attention_decode_prepare_f32 (20)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method linear_attention_prepare_f32 (19)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method linear_attention_prepare_varlen_f32 (24)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method linear_attention_prepare_varlen_packed_qkvz_ba_f32 (24)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method qk_norm_rope_partial (16)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method recurrent_gated_delta_rule_batch_f32 (16)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method recurrent_gated_delta_rule_batch_indexed_f32 (17)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method recurrent_gated_delta_rule_f32 (16)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method recurrent_gated_delta_rule_varlen_f32 (18)",
    "crates/ferrum-kernels/src/backend/cuda/mod.rs::impl CudaBackend::method split_qkv_norm_rope (16)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::<root>::fn paged_varlen_split_k_dispatch (17)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::impl CudaBackend::method paged_varlen_attention (17)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::impl CudaBackend::method paged_varlen_attention_fa2_ffi (18)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::impl CudaBackend::method paged_varlen_attention_vllm (16)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::impl CudaBackend::method paged_varlen_attention_vllm_tiled_q4 (17)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::impl CudaBackend::method qwen35_split_qkv_norm_rope_into_paged_cache_varlen (28)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::impl CudaBackend::method qwen35_split_qkv_norm_rope_into_paged_cache_varlen_vllm (28)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::impl CudaBackend::method split_qkv_norm_rope_into_paged_cache (22)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::impl CudaBackend::method split_qkv_norm_rope_into_paged_cache_varlen (21)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::impl CudaBackend::method split_qkv_norm_rope_into_paged_cache_varlen_vllm (21)",
    "crates/ferrum-kernels/src/backend/cuda/paged.rs::impl CudaBackend::method split_qkv_norm_rope_into_paged_cache_vllm (22)",
    "crates/ferrum-kernels/src/backend/cuda/vllm_marlin.rs::<root>::fn launch_marlin_mm_f16_u4b8 (23)",
    "crates/ferrum-kernels/src/backend/kv_layer.rs::impl KvFp16::method contig_write (20)",
    "crates/ferrum-kernels/src/backend/kv_layer.rs::impl KvFp16::method paged_write (19)",
    "crates/ferrum-kernels/src/backend/kv_layer.rs::impl KvInt8::method paged_write (19)",
    "crates/ferrum-kernels/src/backend/kv_layer.rs::trait KvLayer::trait_fn contig_write (20)",
    "crates/ferrum-kernels/src/backend/kv_layer.rs::trait KvLayer::trait_fn paged_write (19)",
    "crates/ferrum-kernels/src/backend/metal/mod.rs::impl MetalBackend::method split_qkv_norm_rope (16)",
    "crates/ferrum-kernels/src/backend/metal/mod.rs::impl MetalBackend::method split_qkv_norm_rope_into_cache (18)",
    "crates/ferrum-kernels/src/backend/metal/paged.rs::impl MetalBackend::method split_qkv_norm_rope_into_paged_cache (22)",
    "crates/ferrum-kernels/src/backend/metal/q4_k_moe_id_gemm.rs::<root>::fn dispatch_gemm_q4k_moe_id_indirect_on_encoder (16)",
    "crates/ferrum-kernels/src/backend/metal/q4_k_moe_id_gemm.rs::<root>::fn dispatch_gemm_q4k_moe_id_inner (16)",
    "crates/ferrum-kernels/src/backend/metal/q6_k_moe_id_gemm.rs::<root>::fn dispatch_gemm_q6k_moe_id_indirect_on_encoder (16)",
    "crates/ferrum-kernels/src/backend/metal/q6_k_moe_id_gemm.rs::<root>::fn dispatch_gemm_q6k_moe_id_inner (16)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn linear_attention_decode_prepare_batch_f32 (21)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn linear_attention_decode_prepare_batch_indexed_f32 (22)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32 (22)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn linear_attention_decode_prepare_f32 (20)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn linear_attention_prepare_f32 (19)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn linear_attention_prepare_varlen_f32 (24)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn linear_attention_prepare_varlen_packed_qkvz_ba_f32 (24)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn qk_norm_rope_partial (16)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn recurrent_gated_delta_rule_batch_f32 (16)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn recurrent_gated_delta_rule_batch_indexed_f32 (17)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn recurrent_gated_delta_rule_f32 (16)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn recurrent_gated_delta_rule_varlen_f32 (18)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn split_qkv_norm_rope (16)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait Backend::trait_fn split_qkv_norm_rope_into_cache (18)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait BackendPagedKv::trait_fn paged_varlen_attention (17)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait BackendPagedKv::trait_fn paged_varlen_attention_fa2_ffi (18)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait BackendPagedKv::trait_fn paged_varlen_attention_vllm (16)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait BackendPagedKv::trait_fn paged_varlen_attention_vllm_tiled_q4 (17)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait BackendPagedKv::trait_fn qwen35_split_qkv_norm_rope_into_paged_cache_varlen (28)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait BackendPagedKv::trait_fn qwen35_split_qkv_norm_rope_into_paged_cache_varlen_vllm (28)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait BackendPagedKv::trait_fn split_qkv_norm_rope_into_paged_cache (22)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait BackendPagedKv::trait_fn split_qkv_norm_rope_into_paged_cache_varlen (21)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait BackendPagedKv::trait_fn split_qkv_norm_rope_into_paged_cache_varlen_vllm (21)",
    "crates/ferrum-kernels/src/backend/traits.rs::trait BackendPagedKv::trait_fn split_qkv_norm_rope_into_paged_cache_vllm (22)",
    "crates/ferrum-models/src/models/llama_family_forward_batched.rs::impl LlamaFamilyModel::method unified_forward_layer (24)",
    "crates/ferrum-models/src/models/qwen35.rs::<root>::fn qwen35_dense_linear_attention_layer_cpu (18)",
    "crates/ferrum-models/src/models/qwen35.rs::<root>::fn qwen35_full_attention_prefill_batch_layer_backend (18)",
    "crates/ferrum-models/src/models/qwen35.rs::<root>::fn qwen35_linear_attention_prefill_varlen_compact_core_backend (16)",
    "crates/ferrum-models/src/models/qwen35.rs::<root>::fn qwen35_linear_attention_prefill_varlen_core_backend (16)",
    "crates/ferrum-models/src/models/qwen35.rs::<root>::fn qwen35_sparse_moe_full_attention_layer_cpu (18)",
    "crates/ferrum-models/src/models/qwen35.rs::<root>::fn qwen35_sparse_moe_linear_attention_layer_cpu (22)",
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("ferrum-types must live under crates/")
        .to_path_buf()
}

fn line_count(path: &Path) -> usize {
    let file = File::open(path).unwrap_or_else(|err| {
        panic!("failed to open {}: {err}", path.display());
    });
    BufReader::new(file).lines().count()
}

fn rust_files_under(root: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).unwrap_or_else(|err| {
        panic!("failed to read {}: {err}", root.display());
    }) {
        let entry = entry.expect("read_dir entry");
        let path = entry.path();
        if path.is_dir() {
            rust_files_under(&path, out);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn typed_param_count(signature: &syn::Signature) -> usize {
    signature
        .inputs
        .iter()
        .filter(|input| matches!(input, syn::FnArg::Typed(_)))
        .count()
}

fn collect_long_signatures_from_items(
    rel_path: &str,
    module_path: &mut Vec<String>,
    items: &[syn::Item],
    out: &mut Vec<String>,
) {
    for item in items {
        match item {
            syn::Item::Fn(item_fn) => {
                collect_signature(rel_path, module_path, "fn", &item_fn.sig, out);
            }
            syn::Item::Impl(item_impl) => {
                let impl_name = quote_type_name(&item_impl.self_ty);
                module_path.push(format!("impl {impl_name}"));
                for impl_item in &item_impl.items {
                    if let syn::ImplItem::Fn(method) = impl_item {
                        collect_signature(rel_path, module_path, "method", &method.sig, out);
                    }
                }
                module_path.pop();
            }
            syn::Item::Trait(item_trait) => {
                module_path.push(format!("trait {}", item_trait.ident));
                for trait_item in &item_trait.items {
                    if let syn::TraitItem::Fn(method) = trait_item {
                        collect_signature(rel_path, module_path, "trait_fn", &method.sig, out);
                    }
                }
                module_path.pop();
            }
            syn::Item::Mod(item_mod) => {
                if let Some((_, nested_items)) = &item_mod.content {
                    module_path.push(format!("mod {}", item_mod.ident));
                    collect_long_signatures_from_items(rel_path, module_path, nested_items, out);
                    module_path.pop();
                }
            }
            syn::Item::ForeignMod(_) => {
                // FFI declarations are low-level ABI boundaries, not Rust
                // product/hot-path API shape.
            }
            _ => {}
        }
    }
}

fn collect_signature(
    rel_path: &str,
    module_path: &[String],
    kind: &str,
    signature: &syn::Signature,
    out: &mut Vec<String>,
) {
    let param_count = typed_param_count(signature);
    if param_count <= MAX_HOT_PATH_PARAMS {
        return;
    }
    let scope = if module_path.is_empty() {
        "<root>".to_string()
    } else {
        module_path.join("::")
    };
    out.push(format!(
        "{rel_path}::{scope}::{kind} {} ({param_count})",
        signature.ident
    ));
}

fn quote_type_name(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(path) => path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
            .unwrap_or_else(|| "<type>".to_string()),
        syn::Type::Reference(reference) => quote_type_name(&reference.elem),
        _ => "<type>".to_string(),
    }
}

#[test]
fn milestone_h_core_surface_files_stay_below_locked_line_limits() {
    let root = repo_root();
    let mut failures = Vec::new();

    for limit in SHAPE_LIMITS {
        let full_path = root.join(limit.path);
        let actual = line_count(&full_path);
        println!("{}: {actual}/{}", limit.path, limit.max_lines);
        if actual > limit.max_lines {
            failures.push(format!(
                "{} has {actual} lines; limit is {}",
                limit.path, limit.max_lines
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Milestone H codebase-shape limits failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn milestone_h_model_backend_hot_paths_do_not_add_new_long_signatures() {
    let root = repo_root();
    let mut rust_files = Vec::new();
    for hot_root in HOT_PATH_ROOTS {
        rust_files_under(&root.join(hot_root), &mut rust_files);
    }
    rust_files.sort();

    let mut long_signatures = Vec::new();
    for file in rust_files {
        let rel_path = relative_path(&root, &file);
        let source = std::fs::read_to_string(&file).unwrap_or_else(|err| {
            panic!("failed to read {}: {err}", file.display());
        });
        let parsed = syn::parse_file(&source).unwrap_or_else(|err| {
            panic!("failed to parse {}: {err}", file.display());
        });
        collect_long_signatures_from_items(
            &rel_path,
            &mut Vec::new(),
            &parsed.items,
            &mut long_signatures,
        );
    }
    long_signatures.sort();

    let legacy: std::collections::BTreeSet<_> = LEGACY_LONG_SIGNATURES.iter().copied().collect();
    let unexpected: Vec<_> = long_signatures
        .iter()
        .filter(|signature| !legacy.contains(signature.as_str()))
        .cloned()
        .collect();
    let stale: Vec<_> = LEGACY_LONG_SIGNATURES
        .iter()
        .filter(|signature| !long_signatures.iter().any(|found| found == *signature))
        .copied()
        .collect();

    assert!(
        unexpected.is_empty() && stale.is_empty(),
        "Milestone H long-signature baseline changed\nunexpected:\n{}\n\nstale:\n{}",
        unexpected.join("\n"),
        stale.join("\n")
    );
}
