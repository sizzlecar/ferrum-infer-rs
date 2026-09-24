use super::*;
use ferrum_interfaces::vnext::{
    BlockQuantizationSpec, ContractVersion, HadamardApplication, HadamardSigns,
    HadamardTransformSpec, PhysicalWeightComponentBinding, PhysicalWeightPadding,
    WeightComponentRole, WeightComponentSpec, WeightEncoding, WeightSchema, WeightTensorSpec,
};
use std::num::{NonZeroU32, NonZeroU64};

fn id<T>(text: &str) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(text.to_owned()).unwrap()
}

fn schema(output: u64, input: u64, q6: bool) -> WeightSchema {
    WeightSchema {
        format_id: id("weight-format.gguf.native-block"),
        layout_id: id("weight-layout.causal-cost-test"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("component.weight"),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["weight".into()],
            dimensions: vec![output, input / 256],
            encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: id(if q6 {
                    "quantization.gguf.q6-k"
                } else {
                    "quantization.gguf.q4-k"
                }),
                logical_values_per_block: 256,
                bytes_per_block: if q6 { 210 } else { 144 },
            }),
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.projection"),
            dimensions: vec![output, input],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::BlockQuantized {
                blocks: PhysicalWeightComponentBinding::exact_contiguous(id("component.weight")),
                block_axis: 1,
                block_padding: PhysicalWeightPadding::Exact,
            },
            required: true,
        }],
    }
}

fn resolved(schema: &WeightSchema) -> ResolvedWeightBinding {
    ResolvedWeightBinding::from_schema(schema, &id("weight.projection")).unwrap()
}

fn shape(grouped: bool) -> CausalAttentionShape {
    let (head_dim, query_heads) = if grouped { (256, 8) } else { (128, 4) };
    CausalAttentionShape {
        hidden_size: 256,
        query_heads,
        key_value_heads: 2,
        head_dim,
        query_features: query_heads * head_dim,
        query_projection_features: 2 * query_heads * head_dim,
        kv_features: 2 * head_dim,
        rope_dim: 32,
        maximum_context_tokens: 4096,
        epsilon: 1e-6,
        rope_theta: 10000.0,
        rope_interleaved: false,
        output_gate: true,
    }
}

fn caps(kv_type: ElementType) -> Capabilities {
    Capabilities {
        kv_type,
        maximum_attention_simdgroups: 16,
        maximum_threadgroup_memory_length: 32768,
        supports_gqa_tiled_prefill: true,
        batched_grouped: [true; 2],
    }
}

fn row(offset: u64, count: u64) -> OperationCostWorkRow {
    OperationCostWorkRow {
        offset,
        count: NonZeroU64::new(count).unwrap(),
        full_input_tokens: NonZeroU64::new(offset + count).unwrap(),
    }
}

fn weights(shape: CausalAttentionShape) -> [PreparedLinearPart; 4] {
    [
        (shape.query_projection_features, shape.hidden_size, false),
        (shape.kv_features, shape.hidden_size, false),
        (shape.kv_features, shape.hidden_size, true),
        (shape.hidden_size, shape.query_features, true),
    ]
    .map(|(output, input, q6)| {
        leaf_weight(&resolved(&schema(output, input, q6)), output, input)
            .unwrap()
            .unwrap()
    })
}

fn route(
    shape: CausalAttentionShape,
    rows: &[OperationCostWorkRow],
    packed: bool,
    caps: Capabilities,
) -> Option<[OperationCostCommand; 2]> {
    project(
        shape,
        ElementType::F32,
        rows,
        rows.iter().map(|row| row.count.get()).sum(),
        packed,
        weights(shape),
        caps,
    )
    .unwrap()
}

#[test]
fn numerical_params_keep_payload_scale_page_boundaries_and_context() {
    let shape = shape(false);
    for (kv_type, at128, at129, page_elements) in [
        (ElementType::F16, 2, 3, 32768),
        (ElementType::I8, 1, 2, 65536),
    ] {
        let first = row_params(shape, 1, 127, 128, caps(kv_type)).unwrap();
        let next = row_params(shape, 1, 128, 129, caps(kv_type)).unwrap();
        assert_eq!(first.page_count, at128);
        assert_eq!(next.page_count, at129);
        assert_eq!(next.page_elements, page_elements);
        assert_eq!(next.position_start, 128);
        assert_eq!(next.tokens, 1);
        assert_eq!(next.attention_simdgroups, 16);
    }
    assert_eq!(
        shape.physical_scale_bytes(4096).unwrap(),
        VNEXT_KV_PAGE_BYTES
    );
    assert_eq!(
        shape.physical_scale_bytes(4097).unwrap(),
        2 * VNEXT_KV_PAGE_BYTES
    );
    assert_eq!(
        row_params(shape, 1, 0, 1, caps(ElementType::F16))
            .unwrap()
            .attention_simdgroups,
        1
    );
    for (tokens, offset, full) in [
        (0, 0, 1),
        (2, 3, 4),
        (1, 4096, 4097),
        (1, u64::MAX, u64::MAX),
    ] {
        assert!(row_params(shape, tokens, offset, full, caps(ElementType::F16)).is_err());
    }
}

#[test]
fn packed_q4_split_and_q6_leaf_match_the_shared_physical_accounting() {
    let shape = shape(false);
    let rows = [row(0, 16), row(0, 17)];
    let [host, compute] = route(shape, &rows, true, caps(ElementType::F16)).unwrap();
    assert!(host.host_only());
    assert_eq!(host.phase(), DeviceCommandPhase::DynamicBinding);
    assert_eq!(host.participant_count(), 2);
    assert_eq!(compute.batching(), DeviceBatchingForm::Packed);
    assert_eq!(compute.token_count(), 33);
    // Q4 query has a 32-row prefix + one-row tail. Other three projections
    // each retain one dispatch; two local prepare/attention pairs remain.
    assert_eq!(compute.compute_dispatch_count(), 11);
    assert_eq!(compute.transfer_command_count(), 0);
    let [_, separate] = route(shape, &rows, false, caps(ElementType::F16)).unwrap();
    assert_eq!(separate.batching(), DeviceBatchingForm::ParticipantLoop);
    assert_eq!(separate.compute_dispatch_count(), 16);
}

#[test]
fn possible_batched_grouped_decode_requires_live_page_nonalias_evidence() {
    let shape = shape(true);
    let rows = [row(511, 1), row(383, 1)];
    assert!(route(shape, &rows, true, caps(ElementType::F16)).is_none());
    let [_, scalar] = route(shape, &rows[..1], false, caps(ElementType::F16)).unwrap();
    assert_eq!(scalar.compute_dispatch_count(), 9);
    let unsupported = Capabilities {
        batched_grouped: [false; 2],
        ..caps(ElementType::F16)
    };
    assert_eq!(
        route(shape, &rows, true, unsupported).unwrap()[1].compute_dispatch_count(),
        12
    );
    assert!(route(
        shape,
        &[row(511, 1), row(0, 2)],
        true,
        caps(ElementType::F16)
    )
    .is_some());
    // I8 readers do not use grouped decode or its physical-alias-dependent merge.
    let [_, int8] = route(shape, &rows, true, caps(ElementType::I8)).unwrap();
    assert_eq!(int8.compute_dispatch_count(), 10);
    assert_eq!(
        int8.transfer_command_count(),
        2,
        "one actual error-flag reset per row"
    );
}

#[test]
fn actual_pipeline_memory_capability_selects_supported_prefill_without_guessing() {
    let shape = shape(true);
    let params = row_params(shape, 8, 0, 8, caps(ElementType::I8)).unwrap();
    assert_eq!(
        caps(ElementType::I8).dispatch_plan(&params).kind,
        AttentionDispatchKind::GqaTiledPrefill
    );
    let no_gqa = Capabilities {
        supports_gqa_tiled_prefill: false,
        ..caps(ElementType::I8)
    };
    assert_eq!(
        no_gqa.dispatch_plan(&params).kind,
        AttentionDispatchKind::TiledPrefill
    );
    let low_memory = Capabilities {
        maximum_threadgroup_memory_length: 4096,
        ..caps(ElementType::I8)
    };
    assert_eq!(
        low_memory.dispatch_plan(&params).kind,
        AttentionDispatchKind::General
    );
    let f16_params = row_params(shape, 2, 512, 514, caps(ElementType::F16)).unwrap();
    assert_eq!(
        caps(ElementType::F16).dispatch_plan(&f16_params).kind,
        AttentionDispatchKind::GqaTiledPrefill
    );
    assert_eq!(
        Capabilities {
            maximum_threadgroup_memory_length: 4096,
            ..caps(ElementType::F16)
        }
        .dispatch_plan(&f16_params)
        .kind,
        AttentionDispatchKind::General
    );
}

#[test]
fn grouped_batch_command_chunks_and_kv_precision_have_exact_counts() {
    let count = GROUPED_BATCH_ROWS + 1;
    let [host, compute] = commands(
        ElementType::F16,
        ElementType::F16,
        count,
        count as u64,
        true,
        count as u64,
        true,
        0,
    )
    .unwrap();
    assert!(host.host_only());
    assert_eq!(compute.compute_dispatch_count(), count as u64 + 6 + 4);
    for (hidden, kv, label, resets) in [
        (
            ElementType::F16,
            ElementType::F16,
            "vnext_causal_paged_attention",
            0,
        ),
        (
            ElementType::F32,
            ElementType::F16,
            "vnext_causal_paged_attention_f32_master",
            0,
        ),
        (
            ElementType::F16,
            ElementType::I8,
            "vnext_causal_paged_attention_int8_kv",
            1,
        ),
        (
            ElementType::F32,
            ElementType::I8,
            "vnext_causal_paged_attention_f32_master_int8_kv",
            1,
        ),
    ] {
        let [_, compute] = commands(hidden, kv, 1, 4, false, 0, false, 0).unwrap();
        assert_eq!(compute.native_operation(), label);
        assert_eq!(compute.transfer_command_count(), resets);
        assert_eq!(compute.compute_dispatch_count(), 8);
    }
    assert!(commands(ElementType::F16, ElementType::F16, 2, 2, true, 1, true, 0).is_err());
    assert!(commands(ElementType::F16, ElementType::I8, 2, 2, true, 2, false, 0).is_err());
    assert!(commands(
        ElementType::F16,
        ElementType::F16,
        1,
        1,
        false,
        0,
        false,
        u64::MAX
    )
    .is_err());
}

#[test]
fn plain_abi_and_projection_scratch_checks_reject_mismatch_and_transform() {
    let plain = schema(1024, 256, false);
    assert!(leaf_weight(&resolved(&plain), 1023, 256).is_err());
    assert!(leaf_weight(&resolved(&plain), 1024, 255).is_err());
    let mut transformed = plain;
    transformed.tensors[0].physical_layout = PhysicalWeightLayout::Hadamard {
        values: Box::new(transformed.tensors[0].physical_layout.clone()),
        transform: HadamardTransformSpec {
            block_size: NonZeroU32::new(256).unwrap(),
            signs: HadamardSigns::Identity,
            application: HadamardApplication::BeforeMatmul {
                input_permutation: None,
            },
        },
    };
    assert!(leaf_weight(&resolved(&transformed), 1024, 256)
        .unwrap()
        .is_none());
    let shape = shape(false);
    let weights = weights(shape);
    let layout = ScratchLayout::new_with_storage(shape, 2, 2, ElementType::F16).unwrap();
    assert!(projected_launches(shape, weights, layout, 2, 1).is_err());
    assert!(projected_launches(shape, weights, layout, u64::MAX, 2).is_err());
    assert!(projected_launches(
        shape,
        weights,
        layout,
        0,
        u64::from(u32::MAX) / shape.hidden_size + 1
    )
    .is_err());
    let overflowing = CausalAttentionShape {
        maximum_context_tokens: u64::MAX,
        ..shape
    };
    assert!(overflowing.validate_page_count(ElementType::F16).is_err());
}
