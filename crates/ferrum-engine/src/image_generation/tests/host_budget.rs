use super::*;
use crate::image_generation::host_budget::{
    decoder_output_peak, ports, staging_bytes, sum_ports, ExecutionHostBudget,
};
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;
use ferrum_models::qwen_image21::IMAGE_PROJECTION_OUTPUT_ID;

fn stage_grant(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    stage: &PreparedStage<ReferenceDeviceRuntime>,
    cold: bool,
) -> Result<HostMemoryScope, HostMemoryError> {
    let ports = ports(stage).unwrap();
    let budget = ExecutionHostBudget {
        input: sum_ports(&pipeline.host, ports.inputs().iter()).unwrap(),
        output: sum_ports(&pipeline.host, ports.outputs().iter()).unwrap(),
        staging: if cold {
            staging_bytes(&pipeline.host, stage, pipeline.config.static_initialization).unwrap()
        } else {
            0
        },
        weights: 0,
    };
    pipeline.host.reserve_scope(&budget.claims())
}

#[test]
fn scoped_stage_survives_competing_consumers_warm_reuse_and_shape_change() {
    // Reference lacks the production Philox provider. Use its real nonzero
    // projection to exercise stage initialization, execution and residency.
    let (mut pipeline, _directory) = residency::numeric_pipeline();
    pipeline.host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1312,
            HostMemoryClassLimits {
                input_preprocess_bytes: 256,
                output_bytes: 32,
                transfer_staging_bytes: 1024,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut receipts = Vec::new();
    let mut outputs = Vec::new();
    for job in 0..2 {
        let stage = residency::prepared(&pipeline, 1);
        let grant = stage_grant(&pipeline, &stage, job == 0).unwrap();
        pipeline.execution_host = Some(grant.account());
        if job == 1 {
            // A previous consumer owns 16 bytes; this grant owns the other 16.
            assert!(matches!(
                pipeline.host.reserve(HostMemoryClass::Output, 1),
                Err(HostMemoryError::Capacity { .. })
            ));
        }
        let input = residency::input(pipeline.execution_host(), 1, job as usize * 8);
        let active = pipeline
            .start_stage(stage, job, &mut receipts)
            .unwrap_or_else(|_| panic!("start scoped stage"));
        assert_eq!(active.receipt.session_reused, job == 1);
        let (active, output) = pipeline
            .execute_stage(active, job, vec![input])
            .unwrap_or_else(|_| panic!("execute scoped stage"));
        let output = only_output(output, IMAGE_PROJECTION_OUTPUT_ID).unwrap();
        assert_eq!(
            u16::from_le_bytes(output.bytes().bytes()[..2].try_into().unwrap()),
            half::bf16::from_f32((job * 8 + 1) as f32).to_bits()
        );
        pipeline
            .finish_stage(active, &mut receipts)
            .unwrap_or_else(|_| panic!("finish scoped stage"));
        pipeline.execution_host = None;
        drop(grant);
        outputs.push(output);
    }
    let next = residency::prepared(&pipeline, 2);
    let before = pipeline.runtime.snapshot();
    let rejected = stage_grant(&pipeline, &next, false).unwrap_err();
    assert!(matches!(rejected, HostMemoryError::Capacity { .. }));
    assert_eq!(pipeline.runtime.snapshot().submissions, before.submissions);
    assert!(!pipeline.is_stopped());
    // Closing grants did not release a slow consumer's real bytes. Releasing
    // both consumers now admits a larger shape without recreating the pipeline.
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 32);
    drop(outputs);
    let grant = stage_grant(&pipeline, &next, false).unwrap();
    pipeline.execution_host = Some(grant.account());
    let input = residency::input(pipeline.execution_host(), 2, 0);
    let active = pipeline
        .start_stage(next, 2, &mut receipts)
        .unwrap_or_else(|_| panic!("start changed shape"));
    assert!(active.receipt.weights_reused);
    let (active, outputs) = pipeline
        .execute_stage(active, 2, vec![input])
        .unwrap_or_else(|_| panic!("execute changed shape"));
    pipeline
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|_| panic!("finish scoped stage"));
    pipeline.execution_host = None;
    drop(grant);
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 32);
    drop(outputs);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn decoder_peak_counts_physical_backings_and_retained_batch_slices() {
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap();
    let single = fixture::decoder().output_tensor().unwrap();
    let logical = usize::try_from(single.minimum_storage_bytes().unwrap()).unwrap();
    let physical = host.backing_bytes(logical).unwrap();
    let batch = 3;
    let single_peak =
        decoder_output_peak(physical, batch, ImageDecoderBatchPolicy::SingleImage).unwrap();
    let full_physical = host.backing_bytes(logical * batch).unwrap();
    assert!(single_peak > full_physical);
    assert_eq!(
        decoder_output_peak(full_physical, batch, ImageDecoderBatchPolicy::FullBatch).unwrap(),
        full_physical
    );
    let grant = host
        .reserve_scope(&[(HostMemoryClass::Output, single_peak)])
        .unwrap();
    let view = grant.account();
    let outputs: Vec<_> = (0..batch)
        .map(|_| {
            view.allocate_zeroed(HostMemoryClass::Output, logical)
                .unwrap()
                .freeze()
        })
        .collect();
    assert!(matches!(
        view.reserve(HostMemoryClass::Output, 1),
        Err(HostMemoryError::ScopeCapacity { .. })
    ));
    drop(grant);
    assert_eq!(host.usage().unwrap().committed_bytes(), single_peak);
    drop(outputs);
    let grant = host
        .reserve_scope(&[(HostMemoryClass::Output, full_physical)])
        .unwrap();
    let all = grant
        .account()
        .allocate_zeroed(HostMemoryClass::Output, logical * batch)
        .unwrap()
        .freeze();
    let slice = all.slice(logical..logical * 2).unwrap();
    drop((all, grant));
    assert_eq!(host.usage().unwrap().committed_bytes(), full_physical);
    drop(slice);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
