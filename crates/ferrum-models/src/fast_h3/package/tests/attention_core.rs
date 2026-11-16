use super::*;
use crate::fast_h3::{vsa::*, *};
use ferrum_runtime::*;
use std::sync::Arc;
#[cfg(all(feature = "cuda-cutlass-nvfp4", feature = "cuda-sage-vsa-fp4"))]
mod cuda;
#[cfg(all(feature = "cuda-cutlass-nvfp4", feature = "cuda-sage-vsa-fp4"))]
pub(super) mod cuda_factory;
mod dyadic;
type F = FastH3AttentionCoreFamily;
fn id(s: &str) -> ProgramValueId {
    ProgramValueId::new(s).unwrap()
}
pub(super) fn capacities() -> FastH3AttentionCoreLimits {
    let transport = FastH3VsaRowTransportLimits {
        maximum_rows: 16384,
        maximum_columns: 256,
        maximum_batches: 2,
        maximum_map_bytes: 65536,
        maximum_metadata_bytes: 8192,
    };
    FastH3AttentionCoreLimits {
        prefix: FastH3PrefixAttentionLimits {
            center: FastH3KeyCenterLimits {
                maximum_batches: 2,
                maximum_rows: 16384,
                maximum_columns: 256,
                maximum_tensor_bytes: 16 << 20,
                maximum_metadata_bytes: 4096,
            },
            transport,
            maximum_plan_nodes: 32,
            maximum_stored_bytes: 1 << 20,
            maximum_metadata_bytes: 1 << 20,
        },
        quantize: FastH3VsaQkvQuantizeLimits {
            maximum_batches: 2,
            maximum_heads: 2,
            maximum_rows: 16384,
            maximum_input_bytes: 64 << 20,
            maximum_output_bytes: 32 << 20,
            maximum_metadata_bytes: 4096,
        },
        pooled: FastH3PooledAttentionLimits {
            maximum_batches: 2,
            maximum_heads: 2,
            maximum_tiles: 128,
            maximum_head_dim: 128,
            maximum_score_bytes: 1 << 20,
            maximum_metadata_bytes: 4096,
        },
        pooling: FastH3VsaPoolLimits {
            maximum_batches: 2,
            maximum_tiles: 128,
            maximum_columns: 256,
            maximum_count_bytes: 512,
            maximum_metadata_bytes: 8192,
        },
        transport,
        head_pack: FastH3HeadAxisPackLimits {
            maximum_batches: 2,
            maximum_rows: 16384,
            maximum_columns: 256,
            maximum_input_bytes: 32 << 20,
            maximum_metadata_bytes: 4096,
        },
        projection: FastH3AttentionOutputLimits {
            maximum_rows: 32768,
            maximum_metadata_bytes: 16384,
            maximum_stored_bytes: 1 << 20,
        },
        maximum_region_bytes: 16384,
        maximum_plan_nodes: 128,
        maximum_stored_bytes: 1 << 20,
        maximum_metadata_bytes: 1 << 20,
    }
}
pub(super) fn layout() -> FastH3VsaLayout {
    let c = FastH3Checkpoint::parse(
        include_bytes!("../../fixtures/inference.json"),
        include_bytes!("../../fixtures/transformer.json"),
        include_bytes!("../../fixtures/video_scheduler.json"),
        include_bytes!("../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap();
    FastH3VsaLayout::ref2va(
        &c,
        3,
        &[FastH3Reference::Video {
            latent_frames: 5,
            latent_height: 10,
            latent_width: 18,
            audio_latents_per_channel: Some(2),
        }],
        &FastH3VideoGeometry::new(64, 64, 5).unwrap(),
        FastH3VsaLayoutLimits {
            maximum_packed_rows: 10000,
            maximum_tiles: 128,
            maximum_metadata_bytes: 1 << 20,
        },
    )
    .unwrap()
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: bytes,
                transfer_staging_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
fn package() -> (tempfile::TempDir, Arc<FastH3TransformerPackage>) {
    let c = config_with_attention(128, 2, 128, 128);
    let (dir, p) = Fixture::with_attention(&c).open_with_config(
        &c,
        FastH3WeightLimits {
            maximum_source_bytes: 8 << 20,
            ..limits()
        },
    );
    (dir, Arc::new(p.unwrap()))
}
#[test]
fn attention_core_shares_scores_and_centered_keys_with_real_stored_output_projection() {
    let (_dir, package) = package();
    let layout = layout();
    let account = host(1 << 20);
    for arithmetic in [
        FastH3PrefixAttentionArithmetic::ReferenceSequential,
        FastH3PrefixAttentionArithmetic::VendorFused,
    ] {
        let (c, source, regions) = FastH3AttentionCoreSource::prepare(
            package.clone(),
            0,
            &layout,
            2,
            FastH3VsaSelection::Trained,
            arithmetic,
            &account,
            capacities(),
        )
        .unwrap();
        let prepared = TypedFamilyRegistration::new(F::new(capacities()).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&c).unwrap(),
                &NumericalProfileId::new(c.profile_id()).unwrap(),
            )
            .unwrap();
        let graph = prepared.program();
        let options = c.compile_options(capacities()).unwrap();
        let nodes = &graph.blocks()[0].nodes;
        let only = |name: &str| {
            let found: Vec<_> = nodes
                .iter()
                .filter(|n| n.operation_id.as_str() == name)
                .collect();
            assert_eq!(found.len(), 1, "{name}");
            found[0]
        };
        assert_eq!(graph.inputs().len(), 7);
        assert_eq!(graph.outputs(), &[id(F::OUTPUT_ID)]);
        assert_eq!(
            options.tensor_specs()[&id(F::OUTPUT_ID)].dimensions,
            [2 * layout.packed().sequence_rows() as u64, 128]
        );
        let scores = only(SCALE_RANK4_F32_OPERATION_ID);
        let sparse = only(SPARSE_FP4_ATTENTION_REGION_TOPK_OPERATION_ID);
        let softmax = only(ROW_SOFTMAX_RANK4_F32_OPERATION_ID);
        assert_eq!(sparse.inputs[6], scores.outputs[0]);
        assert_eq!(softmax.inputs[0], scores.outputs[0]);
        assert_eq!(
            sparse.attributes[&AttributeId::new("pattern_sha256").unwrap()],
            SemanticValue::Text(regions.digest().into())
        );
        let center = only(ROW_CENTER_BF16_OPERATION_ID);
        let quant = only(SPARSE_FP4_QKV_QUANTIZE_BF16_OPERATION_ID);
        assert_eq!(center.inputs[0], id(F::KEY_ID));
        assert_eq!(quant.inputs[1], center.outputs[0]);
        assert_eq!(quant.inputs[2], id(F::VALUE_ID));
        let fp8_k = nodes
            .iter()
            .find(|n| {
                n.operation_id.as_str() == INDEXED_HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID
                    && n.inputs[0] == center.outputs[0]
            })
            .unwrap();
        assert_ne!(quant.inputs[1], fp8_k.outputs[0]);
        let valid_values = nodes
            .iter()
            .find(|n| {
                n.operation_id.as_str() == ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID
                    && n.inputs[0] == id(F::VALUE_ID)
            })
            .unwrap();
        assert_eq!(
            options.tensor_specs()[&fp8_k.outputs[0]],
            options.tensor_specs()[&valid_values.outputs[0]]
        );
        let fine_pack = only(PACK_HEAD_AXIS_BF16_OPERATION_ID);
        assert_eq!(fine_pack.inputs[0], sparse.outputs[0]);
        let fma = only(ADDCMUL_FMA_BF16_OPERATION_ID);
        let gate_flat = nodes
            .iter()
            .find(|n| n.outputs.contains(&fma.inputs[2]))
            .unwrap();
        assert_eq!(
            gate_flat.operation_id.as_str(),
            MERGE_LEADING_BATCH_BF16_VIEW_OPERATION_ID
        );
        assert_eq!(gate_flat.inputs, [id(F::GATE_ID)]);
        let output_quant = only(NVFP4_QUANTIZE_BF16_OPERATION_ID);
        let output_gemm = only(NVFP4_MATMUL_BF16_OPERATION_ID);
        assert_eq!(output_quant.inputs[0], fma.outputs[0]);
        assert_eq!(output_gemm.inputs[0], output_quant.outputs[0]);
        let ModelWeights::Stored(schema) = prepared.weights() else {
            panic!("Stored")
        };
        // K and V must use the same physical valid-row map, even when graph
        // fragments give it separate logical names. Video FP4 keeps full K.
        let row_map = |value: &ProgramValueId| {
            let weight = graph
                .weights()
                .iter()
                .find(|w| &w.value_id == value)
                .unwrap();
            ResolvedWeightBinding::from_schema(schema, &weight.weight_id).unwrap()
        };
        assert_eq!(
            row_map(&fp8_k.inputs[1]).components(),
            row_map(&valid_values.inputs[1]).components()
        );
        super::ffn::validate_graph_bindings(graph, schema, &options);
        let mut retained = vec![];
        for component in &schema.components {
            let payload = source.component(component).unwrap();
            assert_eq!(
                payload.bytes().len() as u64,
                component.physical_bytes().unwrap()
            );
            retained.push(payload.retained_host_memory().unwrap().clone());
            if component.role == WeightComponentRole::PackedValues {
                assert!(component
                    .external_names
                    .iter()
                    .all(|n| n.contains("attn.to_out::")));
                assert_eq!(
                    component.encoding,
                    WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4())
                );
            }
            let mut forged = component.clone();
            forged.external_names = vec!["another source".into()];
            assert!(source.component(&forged).is_err());
        }
        let map_bytes = 4 * (3 * layout.packed().sequence_rows() + layout.padded_rows());
        let region_bytes = 12 * layout.regions().len() + 8 * layout.tile_count();
        assert_eq!(
            account.usage().unwrap().allocated_bytes(),
            map_bytes + region_bytes + 8
        );
        drop(source);
        assert_eq!(
            account.usage().unwrap().allocated_bytes(),
            map_bytes + region_bytes + 8
        );
        drop(retained);
        drop(regions);
        assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    }
}
#[test]
fn attention_core_rejects_geometry_and_capacity_before_leaking_partial_owners() {
    let (_dir, package) = package();
    let layout = layout();
    let l = capacities();
    let account = host(1 << 20);
    let (c, source, regions) = FastH3AttentionCoreSource::prepare(
        package.clone(),
        0,
        &layout,
        2,
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::VendorFused,
        &account,
        l,
    )
    .unwrap();
    let family = F::new(l).unwrap();
    let raw = serde_json::to_value(&c).unwrap();
    for (path, replacement) in [
        (vec!["quantize", "query_rows"], serde_json::json!(128)),
        (vec!["pooled_pack", "kind"], serde_json::json!("bf16")),
        (vec!["projection", "block"], serde_json::json!(1)),
        (vec!["region_digest"], serde_json::json!("untrusted")),
    ] {
        let mut forged = raw.clone();
        let mut slot = &mut forged;
        for field in path {
            slot = &mut slot[field];
        }
        *slot = replacement;
        assert!(family.parse_config(&forged).is_err());
    }
    for bound in [
        FastH3AttentionCoreLimits {
            maximum_plan_nodes: 1,
            ..l
        },
        FastH3AttentionCoreLimits {
            maximum_stored_bytes: 1,
            ..l
        },
        FastH3AttentionCoreLimits {
            maximum_metadata_bytes: 1,
            ..l
        },
    ] {
        assert!(c.compile_options(bound).is_err());
    }
    let exact = account.usage().unwrap().allocated_bytes();
    drop(source);
    drop(regions);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    let insufficient = host(exact - 1);
    assert!(FastH3AttentionCoreSource::prepare(
        package,
        0,
        &layout,
        2,
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::VendorFused,
        &insufficient,
        l
    )
    .is_err());
    assert_eq!(insufficient.usage().unwrap().allocated_bytes(), 0);
}
