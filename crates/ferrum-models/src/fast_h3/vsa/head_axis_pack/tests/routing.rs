use super::*;
use crate::fast_h3::{vsa::*, *};
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;
fn prepare<F: ModelFamilyProvider>(
    family: F,
    config: &F::Config,
    profile: &str,
) -> PreparedModelFamily {
    TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(profile).unwrap(),
        )
        .unwrap()
}
fn execute_once(
    composition: &ReferenceVNextComposition,
    prepared: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    source: ComponentWeightSource<'_>,
    account: &HostMemoryAccount,
    input: ManagedHostTensor,
) -> Vec<ferrum_execution::TensorExecutionOutput> {
    let runtime = composition.runtime();
    let d = runtime.descriptor();
    let compiled = CompiledTensorComponent::<ReferenceDeviceRuntime>::compile(
        prepared,
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        &options,
    )
    .unwrap();
    let mut s = compiled
        .initialize(
            runtime.clone(),
            account.clone(),
            source,
            ComponentInitializationConfig {
                run_id: id("run.head_route.init").unwrap(),
                request_id: id("request.head_route.init").unwrap(),
                transaction_id: id("transaction.head_route.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("head route initialization retained")
        });
    let output = s
        .execute(
            id("run.head_route.execute").unwrap(),
            id("request.head_route.execute").unwrap(),
            vec![input],
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("head route execution retained")
        });
    assert!(matches!(
        s.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("head route close retained")
        }),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    output
}
#[test]
fn live_head_major_outputs_feed_checked_fine_untile_and_pooled_expand_after_close() {
    let checkpoint = FastH3Checkpoint::parse(
        include_bytes!("../../../fixtures/inference.json"),
        include_bytes!("../../../fixtures/transformer.json"),
        include_bytes!("../../../fixtures/video_scheduler.json"),
        include_bytes!("../../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap();
    let layout = FastH3VsaLayout::ref2va(
        &checkpoint,
        3,
        &[FastH3Reference::Image {
            latent_height: 2,
            latent_width: 4,
        }],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        FastH3VsaLayoutLimits {
            maximum_packed_rows: 4096,
            maximum_tiles: 128,
            maximum_metadata_bytes: 1 << 20,
        },
    )
    .unwrap();
    let pool = FastH3VsaPoolConfig::from_layout(
        &layout,
        2,
        checkpoint.transformer(),
        FastH3VsaPoolLimits {
            maximum_batches: 2,
            maximum_tiles: 128,
            maximum_columns: 8192,
            maximum_count_bytes: 512,
            maximum_metadata_bytes: 4096,
        },
    )
    .unwrap();
    let mut head_limits = limits();
    head_limits.maximum_input_bytes = 64 << 20;
    let map_limits = FastH3VsaRowTransportLimits {
        maximum_rows: 4096,
        maximum_columns: 8192,
        maximum_batches: 2,
        maximum_map_bytes: 1 << 20,
        maximum_metadata_bytes: 4096,
    };
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.head_route").unwrap(),
        128 << 20,
    )
    .unwrap();
    let account = HostMemoryAccount::new(
        HostMemoryLimits::new(
            128 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 128 << 20,
                transfer_staging_bytes: 128 << 20,
                output_bytes: 128 << 20,
                weight_backing_bytes: 128 << 20,
            },
        )
        .unwrap(),
    );
    for (head, map_kind) in [
        (
            FastH3HeadAxisPackConfig::from_fine_layout(
                &layout,
                2,
                checkpoint.transformer(),
                head_limits,
            )
            .unwrap(),
            FastH3VsaRowTransport::Untile,
        ),
        (
            FastH3HeadAxisPackConfig::from_pool(&pool, head_limits).unwrap(),
            FastH3VsaRowTransport::ExpandTileValues,
        ),
    ] {
        let (map, source) = FastH3VsaRowTransportSource::prepare(
            &layout,
            map_kind,
            2,
            head.output_shape().unwrap()[2],
            &account,
            map_limits,
        )
        .unwrap();
        assert_eq!(
            head.output_shape().unwrap(),
            vec![map.batches(), map.source_rows(), map.columns()]
        );
        let bytes = input(&head, 1);
        let expected_heads = oracle(&head, &bytes);
        let packed = execute_once(
            &composition,
            prepare(
                FastH3HeadAxisPackFamily::new(head_limits).unwrap(),
                &head,
                head.profile_id(),
            ),
            head.compile_options(head_limits).unwrap(),
            ComponentWeightSource::None,
            &account,
            managed(
                &account,
                FastH3HeadAxisPackFamily::INPUT_ID,
                head.input_shape(),
                head.kind.input_type(),
                &bytes,
            ),
        );
        // This test crosses two closed components via actual host readback.
        // Output credit cannot masquerade as InputPreprocess credit: admit a
        // separate input copy. Same-graph composition needs neither readback.
        let before = account.usage().unwrap().allocated_bytes();
        let routed_input = managed(
            &account,
            FastH3VsaRowTransportFamily::INPUT_ID,
            head.output_shape().unwrap(),
            ElementType::Bf16,
            packed[0].bytes().bytes(),
        );
        assert_eq!(
            account.usage().unwrap().allocated_bytes() - before,
            packed[0].bytes().bytes().len()
        );
        let output = execute_once(
            &composition,
            prepare(
                FastH3VsaRowTransportFamily::new(map_limits).unwrap(),
                &map,
                FastH3VsaRowTransportFamily::PROFILE_ID,
            ),
            map.compile_options(map_limits).unwrap(),
            ComponentWeightSource::Stored(&source),
            &account,
            routed_input,
        );
        let columns = map.columns() as usize;
        let mut expected = Vec::new();
        for b in 0..2 {
            for &tile_row in layout.packed_to_tile_rows() {
                let row = if map_kind == FastH3VsaRowTransport::Untile {
                    tile_row as usize
                } else {
                    tile_row as usize / FAST_H3_VSA_TILE_ROWS
                };
                let start = (b * map.source_rows() as usize + row) * columns;
                expected.extend_from_slice(&expected_heads[start..start + columns]);
            }
        }
        for (raw, wanted) in output[0].bytes().bytes().chunks_exact(2).zip(expected) {
            let actual = u16::from_le_bytes(raw.try_into().unwrap());
            if head.kind == HeadAxisPackKind::F32ToBf16 && bf16::from_bits(wanted).is_nan() {
                assert!(bf16::from_bits(actual).is_nan());
            } else {
                assert_eq!(actual, wanted);
            }
        }
        drop(output);
        drop(packed);
        drop(source);
        assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    }
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}
