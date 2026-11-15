use super::*;
use crate::fast_h3::{vsa::*, *};
use ferrum_execution::component::{CompiledTensorComponent, TensorComponentPolicy};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::*;
use std::sync::Arc;
type F = FastH3AttentionInputFamily;
#[cfg(feature = "cuda-cutlass-nvfp4")]
mod cuda;
mod dyadic;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
pub(super) fn capacities() -> FastH3AttentionInputLimits {
    FastH3AttentionInputLimits {
        projection: FastH3AttentionProjectionLimits {
            maximum_rows: 32768,
            maximum_metadata_bytes: 32768,
            maximum_stored_bytes: 1 << 20,
        },
        qk: FastH3QkNormRopeLimits {
            maximum_rows: 32768,
            maximum_metadata_bytes: 8192,
            maximum_stored_bytes: 4096,
        },
        transport: FastH3VsaRowTransportLimits {
            maximum_rows: 16384,
            maximum_columns: 1024,
            maximum_batches: 2,
            maximum_map_bytes: 65536,
            maximum_metadata_bytes: 8192,
        },
        pooling: FastH3VsaPoolLimits {
            maximum_batches: 2,
            maximum_tiles: 128,
            maximum_columns: 1024,
            maximum_count_bytes: 512,
            maximum_metadata_bytes: 8192,
        },
        maximum_plan_nodes: 22,
        maximum_stored_bytes: 1 << 20,
        maximum_metadata_bytes: 1 << 20,
    }
}
fn layout() -> FastH3VsaLayout {
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
    super::input_projection::host(bytes)
}

#[test]
fn same_graph_tiles_qkv_but_keeps_gate_packed_and_shares_actual_maps_counts_and_unit() {
    let (_dir, package) = Fixture::new().open(limits());
    let package = Arc::new(package.unwrap());
    let layout = layout();
    let account = host(1 << 20);
    let (c, source) =
        FastH3AttentionInputSource::prepare(package, 0, &layout, 2, &account, capacities())
            .unwrap();
    let expected_backing = 4 * (layout.padded_rows() + layout.tile_count()) + 20;
    assert_eq!(account.usage().unwrap().allocated_bytes(), expected_backing);
    let prepared = TypedFamilyRegistration::new(F::new(capacities()).unwrap())
        .prepare_with_profile(&serde_json::to_value(&c).unwrap(), &id(F::PROFILE_ID))
        .unwrap();
    let graph = prepared.program();
    let nodes = &graph.blocks()[0].nodes;
    let options = c.compile_options(capacities()).unwrap();
    assert_eq!(nodes.len(), 22);
    assert_eq!(graph.inputs().len(), 3);
    assert_eq!(graph.outputs().len(), 7);
    let by_op = |name: &str| {
        nodes
            .iter()
            .filter(|n| n.operation_id.as_str() == name)
            .collect::<Vec<_>>()
    };
    let gather = by_op(ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID);
    assert_eq!(gather.len(), 3);
    assert!(gather.iter().all(|n| n.inputs[1] == gather[0].inputs[1]));
    let quant = by_op(NVFP4_QUANTIZE_BF16_OPERATION_ID);
    assert_eq!(quant.len(), 2);
    assert_eq!(quant[0].inputs[1], quant[1].inputs[1]);
    let expected_input = |n: &ProgramNode| {
        let merge = nodes
            .iter()
            .find(|p| p.outputs.contains(&n.inputs[0]))
            .unwrap();
        assert_eq!(
            merge.operation_id.as_str(),
            MERGE_LEADING_BATCH_BF16_VIEW_OPERATION_ID
        );
        merge.inputs[0].clone()
    };
    assert_eq!(expected_input(quant[0]), gather[0].outputs[0]);
    assert_eq!(expected_input(quant[1]).as_str(), F::HIDDEN_ID);
    let gemm = by_op(NVFP4_MATMUL_BF16_OPERATION_ID);
    assert_eq!(gemm.len(), 4);
    for n in &gemm[..3] {
        assert_eq!(n.inputs[0], quant[0].outputs[0]);
    }
    assert_eq!(gemm[3].inputs[0], quant[1].outputs[0]);
    let pool = by_op(MASKED_TILE_MEAN_BF16_F32_OPERATION_ID);
    assert_eq!(pool.len(), 3);
    assert!(pool.iter().all(|n| n.inputs[1] == pool[0].inputs[1]));
    for name in [F::QUERY_ID, F::KEY_ID, F::VALUE_ID] {
        assert_eq!(
            options.tensor_specs()[&id(name)].dimensions,
            [2, layout.padded_rows() as u64, 48]
        );
    }
    assert_eq!(
        options.tensor_specs()[&id(F::GATE_ID)].dimensions,
        [2, layout.packed().sequence_rows() as u64, 48]
    );
    for name in [F::POOLED_QUERY_ID, F::POOLED_KEY_ID, F::POOLED_VALUE_ID] {
        let t = &options.tensor_specs()[&id(name)];
        assert_eq!(t.dimensions, [2, 3, layout.tile_count() as u64, 16]);
        assert_eq!(t.element_type, ElementType::F32);
    }
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("Stored")
    };
    assert_eq!(schema.components.len(), 17);
    assert_eq!(schema.tensors.len(), 17);
    assert_eq!(graph.weights().len(), 17);
    super::ffn::validate_graph_bindings(graph, schema, &options);
    let mut retained = None;
    for component in &schema.components {
        let payload = source.component(component).unwrap();
        assert_eq!(
            payload.bytes().len() as u64,
            component.physical_bytes().unwrap()
        );
        assert!(payload.retained_host_memory().is_some());
        if component.id.as_str().contains("unit_f32") {
            assert_eq!(payload.bytes(), 1f32.to_le_bytes());
            retained = payload.retained_host_memory().cloned();
        }
        let mut changed = component.clone();
        changed.external_names = vec!["unrelated".into()];
        assert!(source.component(&changed).is_err());
    }
    // Reference has no NVFP4 backend. Preparing the real composition is useful
    // metadata evidence; inventing a CPU provider here would hide that gap.
    let reference = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.attention_input"),
        64 << 20,
    )
    .unwrap();
    let d = reference.runtime().descriptor();
    let result = CompiledTensorComponent::compile(
        prepared,
        reference.registry(),
        reference.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        reference.weight_materializers(),
        &options,
    );
    let error = match result {
        Ok(_) => panic!("NVFP4 execution must require its provider"),
        Err(e) => e,
    };
    assert!(error.to_string().contains("nvfp4"), "{error}");
    drop(source);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 4);
    drop(retained);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn attention_input_rejects_mixed_routing_groups_capacity_and_source_identity_changes() {
    let (_dir, package) = Fixture::new().open(limits());
    let package = Arc::new(package.unwrap());
    let layout = layout();
    let needed = 4 * (layout.padded_rows() + layout.tile_count()) + 20;
    let small = host(needed - 1);
    assert!(FastH3AttentionInputSource::prepare(
        package.clone(),
        0,
        &layout,
        2,
        &small,
        capacities()
    )
    .is_err());
    assert_eq!(small.usage().unwrap().committed_bytes(), 0);
    let account = host(needed);
    let (c, source) =
        FastH3AttentionInputSource::prepare(package.clone(), 0, &layout, 2, &account, capacities())
            .unwrap();
    let family = F::new(capacities()).unwrap();
    for (field, child, value) in [
        ("gate", "group", serde_json::json!("qkv")),
        (
            "gate",
            "rows",
            serde_json::json!(c.batches() * c.tiled_rows()),
        ),
        (
            "qk",
            "rows",
            serde_json::json!(c.batches() * c.packed_rows()),
        ),
        ("tile_rotary", "columns", serde_json::json!(1)),
        ("pooling", "heads", serde_json::json!(1)),
    ] {
        let mut raw = serde_json::to_value(&c).unwrap();
        raw[field][child] = value;
        assert!(family.parse_config(&raw).is_err(), "{field}.{child}");
    }
    let mut raw = serde_json::to_value(&c).unwrap();
    raw["gate"]["projections"][0]["alpha"]["stored_bits"] = serde_json::json!(0.5f32.to_bits());
    let changed = family.parse_config(&raw).unwrap();
    let ModelWeights::Stored(schema) = family.weights(&changed).unwrap() else {
        panic!("Stored")
    };
    assert!(schema
        .components
        .iter()
        .any(|v| source.component(v).is_err()));
    for kind in 0..3 {
        let mut l = capacities();
        match kind {
            0 => l.maximum_plan_nodes = 21,
            1 => l.maximum_stored_bytes = 4,
            _ => l.maximum_metadata_bytes = 4,
        }
        assert!(c.compile_options(l).is_err());
    }
    drop(source);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    let mut l = capacities();
    l.maximum_plan_nodes = 21;
    assert!(FastH3AttentionInputSource::prepare(package, 0, &layout, 2, &account, l).is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
