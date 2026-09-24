use super::*;

fn params(tokens: u32, position: u32) -> CausalAttentionParams {
    CausalAttentionParams {
        page_elements: 32768,
        page_count: 8,
        position_start: position,
        tokens,
        query_heads: 16,
        key_value_heads: 4,
        head_dim: 256,
        rope_dim: 64,
        query_projection_stride: 8192,
        query_head_stride: 512,
        kv_projection_stride: 1024,
        output_gate: 1,
        rope_interleaved: 1,
        attention_simdgroups: 16,
        epsilon: 1e-6,
        rope_theta: 10_000_000.0,
    }
}
#[test]
fn causal_statistical_pairs_include_the_retained_prefix_and_each_new_query() {
    assert_eq!(pairs(&params(1, 618)), Some(619 * 16));
    assert_eq!(pairs(&params(3, 618)), Some((619 + 620 + 621) * 16));
    assert_eq!(pairs(&params(3, 0)), Some(6 * 16));
    for kind in [
        AttentionDispatchKind::General,
        AttentionDispatchKind::DirectDecode,
        AttentionDispatchKind::GroupedDecode,
        AttentionDispatchKind::TiledPrefill,
        AttentionDispatchKind::GqaTiledPrefill,
    ] {
        assert!(rectangular(&params(3, 618), kind).unwrap() >= pairs(&params(3, 618)).unwrap());
    }
    let mut overflow = params(u32::MAX, u32::MAX);
    overflow.query_heads = u32::MAX;
    assert_eq!(pairs(&overflow), None);
}
#[test]
fn causal_selected_real_psos_bind_specializations_grouped_reduce_and_context_work() {
    let device = Device::system_default().expect("actual Metal causal pipeline catalog");
    let a = MetalCausalAttentionPipelines::new(&device).unwrap();
    let p = params(1, 618);
    let plan = a.dispatch_plan(&p);
    assert_eq!(plan.kind, AttentionDispatchKind::GroupedDecode);
    let (entry, special) = selected_entry(&a, &p, plan.kind, false).unwrap();
    assert_eq!(entry, GROUPED_DECODE_PARTIAL_ATTENTION_KERNEL);
    let actual = a.attention_pipeline(&p, plan.kind);
    assert_eq!(
        special,
        if std::ptr::eq(actual, &a.grouped_decode_partial_attention) {
            0
        } else {
            256
        }
    );
    let mut b = SelectedCommandCostBuilderV1::new(1);
    attention(&mut b, &a, &p, 4096).unwrap();
    let e = b.finish().unwrap();
    e.validate_command(1, 2, 0).unwrap();
    let mut longer = p;
    longer.position_start = 1088;
    let mut b = SelectedCommandCostBuilderV1::new(1);
    attention(&mut b, &a, &longer, 4096).unwrap();
    let next = b.finish().unwrap();
    assert_eq!(e.family_signature(), next.family_signature());
    assert!(next.work().inner_work_units > e.work().inner_work_units);
    assert_eq!(next.work().peak_scratch_bytes, 4096);
    for v in [params(1, 0), params(8, 0), params(128, 618)] {
        let mut b = SelectedCommandCostBuilderV1::new(u64::from(v.tokens));
        let plan = a.dispatch_plan(&v);
        attention(&mut b, &a, &v, 4096).unwrap();
        b.finish()
            .unwrap()
            .validate_command(
                u64::from(v.tokens),
                if plan.kind == AttentionDispatchKind::GroupedDecode {
                    2
                } else {
                    1
                },
                0,
            )
            .unwrap();
    }
}

#[test]
fn causal_batched_real_pso_metadata_keeps_heterogeneous_partition_work() {
    let device = Device::system_default().expect("actual Metal grouped catalog");
    let a = MetalCausalAttentionPipelines::new(&device).unwrap();
    let rows = [params(1, 618), params(1, 1088)];
    if a.specialization(&rows[0])
        .and_then(|p| p.batched_grouped.as_ref())
        .is_none()
    {
        let mut b = SelectedCommandCostBuilderV1::new(2);
        assert!(batched(&mut b, &a, rows.iter(), 4096).is_none());
        return;
    }
    let mut b = SelectedCommandCostBuilderV1::new(2);
    batched(&mut b, &a, rows.iter(), 4096).unwrap();
    let evidence = b.finish().unwrap();
    evidence.validate_command(2, 2, 0).unwrap();
    let reduce = rows
        .iter()
        .map(|p| u64::from(p.query_heads) * u64::from(p.head_dim) * grouped_decode_partitions(p))
        .sum::<u64>();
    let expected = rows
        .iter()
        .map(|p| pairs(p).unwrap() * u64::from(p.head_dim))
        .sum::<u64>()
        + reduce;
    assert_eq!(evidence.work().inner_work_units, expected);
    let mut b = SelectedCommandCostBuilderV1::new(2);
    let invalid = [rows[0], params(2, 1088)];
    assert!(batched(&mut b, &a, invalid.iter(), 4096).is_none());
}
