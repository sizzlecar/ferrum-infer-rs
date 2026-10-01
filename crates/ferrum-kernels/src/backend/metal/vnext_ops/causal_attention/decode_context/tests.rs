use super::*;

fn shape(head_dim: u64, query_heads_per_kv: u64, maximum: u64) -> CausalAttentionShape {
    let key_value_heads = 2;
    let query_heads = key_value_heads * query_heads_per_kv;
    CausalAttentionShape {
        hidden_size: 256,
        query_heads,
        key_value_heads,
        head_dim,
        query_features: query_heads * head_dim,
        query_projection_features: 2 * query_heads * head_dim,
        kv_features: key_value_heads * head_dim,
        rope_dim: 32,
        maximum_context_tokens: maximum,
        epsilon: 1e-6,
        rope_theta: 10000.0,
        rope_interleaved: false,
        output_gate: true,
    }
}

fn caps(kv_type: ElementType, simdgroups: u32) -> cost_route::Capabilities {
    cost_route::Capabilities {
        kv_type,
        maximum_attention_simdgroups: simdgroups,
        maximum_threadgroup_memory_length: 32768,
        supports_gqa_tiled_prefill: true,
        batched_grouped: [true; 2],
    }
}

#[test]
fn metal_decode_context_boundaries_match_every_original_selector_class_transition() {
    for kv_type in [ElementType::F16, ElementType::I8] {
        for head in [64, 128, 256] {
            for ratio in [1, 2, 4, 6, 8] {
                for simdgroups in [4, 16] {
                    let shape = shape(head, ratio, 4096);
                    let caps = caps(kv_type, simdgroups);
                    let declared = coverage(shape, caps);
                    assert!(!declared.is_complete());
                    // Exhaustive small-context scan of the actual dispatch
                    // plan, independent of the declaration's candidate cuts.
                    let mut expected = Vec::new();
                    let mut previous = caps.dispatch_plan(&params(shape, caps, 1).unwrap());
                    for sequence in 2..=shape.maximum_context_tokens {
                        let next = caps.dispatch_plan(&params(shape, caps, sequence).unwrap());
                        let kind = if (
                            previous.kind,
                            previous.threads_per_threadgroup,
                            previous.threadgroup_memory_bytes,
                        ) != (
                            next.kind,
                            next.threads_per_threadgroup,
                            next.threadgroup_memory_bytes,
                        ) {
                            Some(DecodeContextBoundaryKind::KernelFamily)
                        } else {
                            None
                        };
                        if let Some(kind) = kind {
                            expected.push(DecodeContextBoundary {
                                first_sequence_tokens: NonZeroU64::new(sequence).unwrap(),
                                kind,
                            });
                        }
                        previous = next;
                    }
                    assert_eq!(
                        declared.known_boundaries(),
                        expected,
                        "kv={kv_type:?}, head={head}, ratio={ratio}, SIMDgroups={simdgroups}"
                    );
                }
            }
        }
    }
}

#[test]
fn metal_decode_context_boundaries_clip_to_compiled_capacity_without_claiming_completeness() {
    let caps = caps(ElementType::F16, 16);
    let full = coverage(shape(256, 4, 4096), caps);
    assert!(!full.known_boundaries().is_empty());
    for boundary in full.known_boundaries() {
        let at = boundary.first_sequence_tokens.get();
        let before = coverage(shape(256, 4, at - 1), caps);
        let after = coverage(shape(256, 4, at), caps);
        assert_eq!(before.maximum_sequence_tokens().unwrap().get(), at - 1);
        assert!(!before.known_boundaries().contains(boundary));
        assert_eq!(after.known_boundaries().last(), Some(boundary));
        assert!(!after.is_complete());
    }
    assert!(coverage(shape(256, 4, u64::MAX), caps)
        .known_boundaries()
        .is_empty());
    assert!(coverage(shape(256, 4, 0), caps)
        .maximum_sequence_tokens()
        .is_none());
    for invalid in [0, u32::MAX] {
        assert!(
            coverage(shape(256, 4, 4096), self::caps(ElementType::F16, invalid))
                .known_boundaries()
                .is_empty()
        );
    }
}

#[test]
fn metal_decode_context_int8_and_non_grouped_shapes_do_not_borrow_f16_grouped_boundary() {
    let grouped = coverage(shape(256, 4, 4096), caps(ElementType::F16, 16));
    assert!(!grouped.known_boundaries().is_empty());
    assert!(grouped
        .known_boundaries()
        .iter()
        .all(|b| b.kind == DecodeContextBoundaryKind::KernelFamily));
    for (shape, caps) in [
        (shape(256, 4, 4096), caps(ElementType::I8, 16)),
        (shape(128, 2, 4096), caps(ElementType::F16, 16)),
    ] {
        let coverage = coverage(shape, caps);
        // Direct decode still specializes its launch threads and threadgroup
        // memory for the original SIMDgroup ramp. Only the F16 grouped
        // kernel-family transition is absent from these installed selectors.
        let expected: Vec<_> = (2..=u64::from(caps.maximum_attention_simdgroups))
            .map(|sequence| DecodeContextBoundary {
                first_sequence_tokens: NonZeroU64::new(sequence).unwrap(),
                kind: DecodeContextBoundaryKind::KernelFamily,
            })
            .collect();
        assert_eq!(coverage.known_boundaries(), expected);
        assert!(!coverage.known_boundaries().iter().any(|boundary| {
            boundary.first_sequence_tokens.get() == GROUPED_DECODE_MINIMUM_CONTEXT
        }));
        assert!(!coverage.is_complete());
        for sequence in [1, 70, 255, 256, 257, 4096] {
            assert_eq!(
                caps.dispatch_plan(&params(shape, caps, sequence).unwrap())
                    .kind,
                AttentionDispatchKind::DirectDecode
            );
        }
    }
}
