//! A joint producer publishes disjoint leading-axis views without Host output.
use super::*;
use ferrum_execution::TensorRequestInitialization;

fn joint_input(host: &HostMemoryAccount) -> ManagedHostTensor {
    let values = [1., -2., 3., 4., -5., 6., 7., -8.];
    let mut owner = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
        .unwrap();
    for (chunk, value) in owner.as_mut().chunks_exact_mut(2).zip(values) {
        chunk.copy_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
    }
    ManagedHostTensor::new(
        id(X),
        ResolvedTensorSpec::new(
            vec![4, 2],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        owner.freeze(),
    )
    .unwrap()
}

fn selection(output: &str, start: u64, length: u64) -> Vec<TensorRequestStateMapping> {
    vec![TensorRequestStateMapping {
        source_output: id(output),
        destination_state: id(CACHE),
        selection: TensorOutputSelection::LeadingAxis { start, length },
    }]
}

#[test]
fn joint_device_output_initializes_independent_requests_without_host_readback() {
    let c = ReferenceVNextComposition::create_with_capacity(id("device.request.partition"), 16384)
        .unwrap();
    let host = host();
    let mut producer = component_with_rows(&c, &host, true, 4);
    let consumer = component(&c, &host, false);
    let second_consumer = component(&c, &host, false);
    let mut actor = TensorSerialActor::new(c.runtime().clone()).unwrap();
    let mut first = consumer
        .begin_request(
            id("run.partition.first"),
            id("request.partition.first"),
            vec![input(&host, [0.; 4])],
        )
        .unwrap();
    let mut second = second_consumer
        .begin_request(
            id("run.partition.second"),
            id("request.partition.second"),
            vec![input(&host, [0.; 4])],
        )
        .unwrap();
    let first_mapping = selection(CACHE, 0, 2);
    let second_mapping = selection(Y, 2, 2);
    let before = c.runtime().snapshot();
    let outputs = producer
        .execute_initializing_requests_on_actor(
            &mut actor,
            id("run.partition.producer"),
            id("request.partition.producer"),
            vec![joint_input(&host)],
            &mut [
                TensorRequestInitialization {
                    request: &mut first,
                    mappings: &first_mapping,
                },
                TensorRequestInitialization {
                    request: &mut second,
                    mappings: &second_mapping,
                },
            ],
        )
        .unwrap();
    assert!(outputs.is_empty());
    assert_eq!(c.runtime().snapshot().readback_bytes, before.readback_bytes);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert!(matches!(
        producer.try_release_idle_dynamic_backing().unwrap(),
        IdleDynamicReleaseOutcome::Released(_)
    ));
    // Producer backing can retire while both separately charged destinations
    // remain live. Closing one request cannot invalidate its sibling's state.
    close(producer);
    let output = first
        .execute_on_actor(&mut actor, vec![input(&host, [10., 20., 30., 40.])])
        .unwrap();
    assert_eq!(output[0].bytes().bytes(), bytes(&[12., 16., 36., 48.]));
    drop(output);
    first.close().unwrap();
    let output = second
        .execute_on_actor(&mut actor, vec![input(&host, [1.; 4])])
        .unwrap();
    assert_eq!(output[0].bytes().bytes(), bytes(&[-14., 19., 22., -23.]));
    drop(output);
    second.close().unwrap();
    close(consumer);
    close(second_consumer);
    drop(actor);
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn invalid_later_selection_submits_nothing_and_preserves_both_request_owners() {
    for (label, start, length) in [
        ("empty", 0, 0),
        ("tail", 3, 2),
        ("overflow", u64::MAX, 2),
        ("shape", 0, 1),
    ] {
        let c = ReferenceVNextComposition::create_with_capacity(
            id(format!("device.request.partition.{label}")),
            16384,
        )
        .unwrap();
        let host = host();
        let mut producer = component_with_rows(&c, &host, true, 4);
        let consumer = component(&c, &host, false);
        let second_consumer = component(&c, &host, false);
        let mut actor = TensorSerialActor::new(c.runtime().clone()).unwrap();
        let mut first = consumer
            .begin_request(
                id("run.invalid.first"),
                id("request.invalid.first"),
                vec![input(&host, [0.; 4])],
            )
            .unwrap();
        let mut second = second_consumer
            .begin_request(
                id("run.invalid.second"),
                id("request.invalid.second"),
                vec![input(&host, [0.; 4])],
            )
            .unwrap();
        let valid = selection(CACHE, 0, 2);
        let invalid = selection(Y, start, length);
        let before = c.runtime().snapshot();
        let failure = producer
            .execute_initializing_requests_on_actor(
                &mut actor,
                id("run.invalid.producer"),
                id("request.invalid.producer"),
                vec![joint_input(&host)],
                &mut [
                    TensorRequestInitialization {
                        request: &mut first,
                        mappings: &valid,
                    },
                    TensorRequestInitialization {
                        request: &mut second,
                        mappings: &invalid,
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(
            c.runtime().snapshot().submissions,
            before.submissions,
            "{label}"
        );
        assert_eq!(
            c.runtime().snapshot().readback_bytes,
            before.readback_bytes,
            "{label}"
        );
        drop(failure);
        assert!(first.execute(vec![input(&host, [0.; 4])]).is_err());
        assert!(second.execute(vec![input(&host, [0.; 4])]).is_err());
        first.close().unwrap();
        second.close().unwrap();
        close(producer);
        close(consumer);
        close(second_consumer);
        drop(actor);
        assert_eq!(c.runtime().snapshot().live_allocations, 0, "{label}");
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0, "{label}");
    }
}
