use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn query_maps_split_at_checked_prefix_without_compacting_padding_or_changing_old_wire() {
    let layout = super::super::tests::layout();
    let rows = layout.padded_rows();
    let prefix = layout.prefix_tiles() * 128;
    let x = (0..2 * rows * 3)
        .map(|i| (i as u16).wrapping_mul(149))
        .collect::<Vec<_>>();
    for (kind, range) in [
        (FastH3VsaRowTransport::PrefixQueryRows, 0..prefix),
        (FastH3VsaRowTransport::VideoQueryRows, prefix..rows),
    ] {
        let actual = execute(&layout, kind, &x);
        let expected = (0..2)
            .flat_map(|b| {
                x[b * rows * 3 + range.start * 3..b * rows * 3 + range.end * 3]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        let h = host(1 << 20);
        let (c, s) =
            FastH3VsaRowTransportSource::prepare(&layout, kind, 2, 3, &h, limits()).unwrap();
        assert_eq!(c.output_rows(), range.len() as u64);
        let mut wire = serde_json::to_value(&c).unwrap();
        wire["query_range"]["start"] = 1.into();
        assert!(serde_json::from_value::<FastH3VsaRowTransportConfig>(wire)
            .unwrap()
            .compile_options(limits())
            .is_err());
        drop(s);
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
    for kind in [
        FastH3VsaRowTransport::Tile,
        FastH3VsaRowTransport::Untile,
        FastH3VsaRowTransport::ExpandTileValues,
        FastH3VsaRowTransport::ValidTileRows,
    ] {
        let h = host(1 << 20);
        let (c, _s) =
            FastH3VsaRowTransportSource::prepare(&layout, kind, 2, 3, &h, limits()).unwrap();
        let wire = serde_json::to_value(c).unwrap();
        assert!(wire.get("query_range").is_none());
    }
}

type F = FastH3VsaRowTransportFamily;
fn limits() -> FastH3VsaRowTransportLimits {
    FastH3VsaRowTransportLimits {
        maximum_rows: 4096,
        maximum_columns: 128,
        maximum_batches: 2,
        maximum_map_bytes: 16384,
        maximum_metadata_bytes: 4096,
    }
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
struct Counted {
    source: FastH3VsaRowTransportSource,
    reads: AtomicUsize,
}
impl WeightComponentSource for Counted {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.source.component(c)
    }
}
fn execute(layout: &FastH3VsaLayout, kind: FastH3VsaRowTransport, input: &[u16]) -> Vec<u16> {
    let account = host(1 << 20);
    let (config, source) =
        FastH3VsaRowTransportSource::prepare(layout, kind, 2, 3, &account, limits()).unwrap();
    assert_eq!(input.len(), (2 * config.source_rows() * 3) as usize);
    let source = Counted {
        source,
        reads: AtomicUsize::new(0),
    };
    let prepared = TypedFamilyRegistration::new(F::new(limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::new(F::PROFILE_ID).unwrap(),
        )
        .unwrap();
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("Stored I32 map")
    };
    assert_eq!(schema.components.len(), 1);
    assert_eq!(
        schema.components[0].encoding,
        WeightEncoding::Dense {
            element_type: ElementType::I32
        }
    );
    let mut wrong = schema.components[0].clone();
    wrong.dimensions[0] += 1;
    assert!(source.source.component(&wrong).is_err());
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.vsa_rows").unwrap(),
        1 << 20,
    )
    .unwrap();
    let d = composition.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        &config.compile_options(limits()).unwrap(),
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            account.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.vsa_rows.init").unwrap(),
                request_id: id("request.vsa_rows.init").unwrap(),
                transaction_id: id("transaction.vsa_rows.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(1 << 20, 4).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    // The device retains the initialized map; no further source lookup is possible.
    drop(source);
    let mut first = None;
    let mut retained = None;
    for _ in 0..2 {
        let mut bytes = account
            .allocate_zeroed(HostMemoryClass::InputPreprocess, input.len() * 2)
            .unwrap();
        for (chunk, v) in bytes.as_mut().chunks_exact_mut(2).zip(input) {
            chunk.copy_from_slice(&v.to_le_bytes());
        }
        let tensor = ManagedHostTensor::new(
            id(F::INPUT_ID).unwrap(),
            ResolvedTensorSpec::new(
                vec![2, config.source_rows(), 3],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            bytes.freeze(),
        )
        .unwrap();
        let outputs = session
            .execute(
                id("run.vsa_rows.execute").unwrap(),
                id("request.vsa_rows.execute").unwrap(),
                vec![tensor],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].tensor().dimensions(),
            [2, config.output_rows(), 3]
        );
        let values: Vec<_> = outputs[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        if let Some(previous) = &first {
            assert_eq!(previous, &values);
        } else {
            first = Some(values);
        }
        retained = Some(outputs);
    }
    assert!(matches!(
        session.close().unwrap_or_else(|e| {
            let reason = format!("{:?}", e.failure());
            std::mem::forget(e);
            panic!("close retained: {reason}")
        }),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    assert!(account.owns(retained.as_ref().unwrap()[0].bytes()));
    drop(retained);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    first.unwrap()
}

#[test]
fn stored_maps_tile_untile_and_expand_mixed_reference_rows_without_changing_any_bits() {
    let layout = super::super::tests::layout();
    // Independent source geometry: text3/audio4/image2/audio16 and a 5x5x9
    // reference video, split into 4x4x8 tiles; the target video occupies tile12.
    let mut tile_to_packed = vec![None; 13 * 128];
    for (tile, start, len) in [
        (0, 0, 3),
        (1, 3, 4),
        (2, 232, 2),
        (3, 234, 16),
        (12, 250, 8),
    ] {
        for i in 0..len {
            tile_to_packed[tile * 128 + i] = Some(start + i);
        }
    }
    let mut used = [0; 8];
    for t in 0..5 {
        for y in 0..5 {
            for x in 0..9 {
                let tile = (t / 4) * 4 + (y / 4) * 2 + x / 8;
                tile_to_packed[(tile + 4) * 128 + used[tile]] = Some(7 + t * 45 + y * 9 + x);
                used[tile] += 1;
            }
        }
    }
    let mut packed_to_tile = vec![0; 258];
    for (slot, p) in tile_to_packed.iter().enumerate() {
        if let Some(p) = p {
            packed_to_tile[*p] = slot;
        }
    }
    let special = [
        0x8000, 0x0000, 0x7f81, 0xff81, 0x7fc1, 0xffff, 0x0001, 0x7f80,
    ];
    let input: Vec<u16> = (0..2 * 258 * 3)
        .map(|i| {
            if i < special.len() {
                special[i]
            } else {
                (i as u16).wrapping_mul(139)
            }
        })
        .collect();
    let tiled = execute(&layout, FastH3VsaRowTransport::Tile, &input);
    for batch in 0..2 {
        for (slot, p) in tile_to_packed.iter().enumerate() {
            for c in 0..3 {
                assert_eq!(
                    tiled[(batch * 1664 + slot) * 3 + c],
                    p.map_or(0, |p| input[(batch * 258 + p) * 3 + c])
                );
            }
        }
    }
    assert_eq!(
        execute(&layout, FastH3VsaRowTransport::Untile, &tiled),
        input
    );
    let pooled: Vec<u16> = (0..2 * 13 * 3).map(|i| 0x8000 + i as u16).collect();
    let expanded = execute(&layout, FastH3VsaRowTransport::ExpandTileValues, &pooled);
    for batch in 0..2 {
        for p in 0..258 {
            for c in 0..3 {
                assert_eq!(
                    expanded[(batch * 258 + p) * 3 + c],
                    pooled[(batch * 13 + packed_to_tile[p] / 128) * 3 + c]
                );
            }
        }
    }
}

#[test]
fn map_preparation_rejects_real_capacity_and_changed_layout_before_retaining_host_bytes() {
    let layout = super::super::tests::layout();
    let account = host(1 << 20);
    for mutation in 0..5 {
        let mut limits = limits();
        match mutation {
            0 => limits.maximum_rows = layout.padded_rows() as u64 - 1,
            1 => limits.maximum_map_bytes = layout.padded_rows() as u64 * 4 - 1,
            2 => limits.maximum_batches = 1,
            3 => limits.maximum_columns = 2,
            4 => limits.maximum_metadata_bytes = 1,
            _ => unreachable!(),
        }
        assert!(FastH3VsaRowTransportSource::prepare(
            &layout,
            FastH3VsaRowTransport::Tile,
            2,
            3,
            &account,
            limits
        )
        .is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
    let tiny = host(layout.padded_rows() * 4 - 1);
    assert!(FastH3VsaRowTransportSource::prepare(
        &layout,
        FastH3VsaRowTransport::Tile,
        2,
        3,
        &tiny,
        limits()
    )
    .is_err());
    assert_eq!(tiny.usage().unwrap().committed_bytes(), 0);
    let mut forged = layout.clone();
    forged.tile_to_packed[0] = 1;
    assert!(FastH3VsaRowTransportSource::prepare(
        &forged,
        FastH3VsaRowTransport::Tile,
        2,
        3,
        &account,
        limits()
    )
    .is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
