use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
use serde::ser::SerializeStruct;

fn host(output_bytes: usize, alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            output_bytes.max(1),
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}
fn new(
    host: &HostMemoryAccount,
    maximum_bytes: usize,
) -> (ImageContinuousTraceRecorder, ImageContinuousTraceHandle) {
    ImageContinuousTraceRecorder::new(host, ImageContinuousTraceLimits { maximum_bytes }).unwrap()
}
fn records(
    handle: &ImageContinuousTraceHandle,
) -> (Vec<serde_json::Value>, ImageContinuousTraceStatus) {
    let mut bytes = Vec::new();
    let status = handle.write_jsonl(&mut bytes).unwrap();
    let records = bytes
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    (records, status)
}
fn event_len(event: &impl Serialize) -> usize {
    serde_json::to_vec(&Envelope {
        schema_version: SCHEMA_VERSION,
        session_uuid: Uuid::nil(),
        sequence: 1,
        monotonic_ns: 0,
        event_ns: 0,
        event,
    })
    .unwrap()
    .len()
        + 1
}

#[test]
fn limits_and_original_physical_output_boundary_are_checked_before_allocation() {
    let empty = host(0, 16);
    assert!(ImageContinuousTraceRecorder::new(
        &empty,
        ImageContinuousTraceLimits { maximum_bytes: 0 }
    )
    .is_err());
    assert!(ImageContinuousTraceRecorder::new(
        &empty,
        ImageContinuousTraceLimits { maximum_bytes: 1 }
    )
    .is_err());
    assert_eq!(empty.usage().unwrap().committed_bytes(), 0);
    assert!(ImageContinuousTraceLimits {
        maximum_bytes: usize::MAX
    }
    .validate()
    .is_err());
    assert!(
        serde_json::from_str::<ImageContinuousTraceLimits>(r#"{"maximum_bytes":4,"other":1}"#)
            .is_err()
    );

    let logical = 257;
    let query = host(4096, 64);
    let physical = query.backing_bytes(logical).unwrap();
    let short = host(physical - 1, 64);
    assert!(ImageContinuousTraceRecorder::new(
        &short,
        ImageContinuousTraceLimits {
            maximum_bytes: logical
        }
    )
    .is_err());
    assert_eq!(short.usage().unwrap().committed_bytes(), 0);
    let exact = host(physical, 64);
    let (recorder, handle) = new(&exact, logical);
    assert_eq!(exact.usage().unwrap().allocated_bytes(), physical);
    assert_eq!(handle.status().backing_bytes, physical);
    assert_eq!(handle.status().maximum_bytes, logical);
    assert!(exact.allocate_zeroed(HostMemoryClass::Output, 1).is_err());
    drop(recorder);
    drop(handle);
    assert_eq!(exact.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn exact_complete_record_fits_and_full_append_keeps_the_original_prefix() {
    let event = serde_json::json!({"request":"request.image.h3.7","step":2});
    let capacity = event_len(&event);
    let host = host(capacity, 1);
    let (recorder, handle) = new(&host, capacity);
    assert_eq!(
        recorder.0.append(&mut recorder.0.lock(), &event, 0, 0),
        Some(1)
    );
    assert_eq!(handle.status().used_bytes, capacity);
    recorder.record(&serde_json::json!({"later":true}));
    recorder.0.append(&mut recorder.0.lock(), &event, 0, 0);
    recorder.seal();
    let (lines, status) = records(&handle);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["event"], event);
    assert_eq!(lines[0]["sequence"], 1);
    assert_eq!(lines[0]["session_uuid"], status.session_uuid.to_string());
    assert_eq!(lines[1]["footer"]["recorded"], 1);
    assert_eq!(lines[1]["footer"]["dropped"], 2);
    assert!(status.sealed && status.truncated);
    assert_eq!(host.usage().unwrap().allocated_bytes(), capacity);
}

#[test]
fn missing_final_newline_rolls_back_the_entire_first_record() {
    let event = serde_json::json!({"cohort":4,"members":[1,2]});
    let capacity = event_len(&event) - 1;
    let host = host(capacity, 1);
    let (recorder, handle) = new(&host, capacity);
    assert_eq!(
        recorder.0.append(&mut recorder.0.lock(), &event, 0, 0),
        None
    );
    let (lines, status) = records(&handle);
    assert_eq!(lines.len(), 1);
    assert_eq!(
        (status.recorded, status.dropped, status.used_bytes),
        (0, 1, 0)
    );
    assert!(!status.sealed && status.truncated);
    assert!(recorder.0.lock().buffer.as_ref().iter().all(|&b| b == 0));
}

#[test]
fn typed_boundaries_use_the_one_session_clock_without_clamping_concurrent_events() {
    use ferrum_types::ImageRequestCorrelation;
    let alignment = 64;
    let host = host(8192 + alignment - 1, alignment);
    let (recorder, handle) = new(&host, 8192);
    let a = ImageRequestCorrelation::new(None);
    let b = ImageRequestCorrelation::new(None);
    // Two producers can arrive at the serializer in an order different from
    // the order of their original boundary Instants. Neither timestamp is fake.
    assert_eq!(
        handle.record_at(
            recorder.0.epoch,
            &ImageContinuousTraceEvent::RequestQueued { correlation: a }
        ),
        Some(1)
    );
    let boundary = Instant::now();
    assert_eq!(
        handle.record_at(
            boundary,
            &ImageContinuousTraceEvent::RequestQueued { correlation: b }
        ),
        Some(2)
    );
    assert_eq!(
        handle.record_at(
            recorder.0.epoch,
            &ImageContinuousTraceEvent::RequestDequeued {
                request_id: a.request_id
            }
        ),
        Some(3)
    );
    recorder.seal();
    let (lines, status) = records(&handle);
    assert_eq!(status.schema_version, 2);
    assert_eq!(handle.session_uuid(), status.session_uuid);
    assert_eq!(lines[0]["event_ns"], 0);
    assert_eq!(lines[2]["event_ns"], 0);
    assert!(lines[1]["event_ns"].as_u64().unwrap() > 0);
    for pair in lines[..3].windows(2) {
        assert!(pair[0]["monotonic_ns"].as_u64() <= pair[1]["monotonic_ns"].as_u64());
    }
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        status.backing_bytes
    );
    drop(recorder);
    drop(handle);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn invalid_clock_boundary_or_regression_keeps_the_original_complete_prefix() {
    let host = host(8192, 1);
    let (recorder, handle) = new(&host, 8192);
    let event = serde_json::json!({"kind":"test"});
    assert_eq!(
        recorder.0.append(&mut recorder.0.lock(), &event, 100, 90),
        Some(1)
    );
    let before = handle.status();
    assert_eq!(
        recorder.0.append(&mut recorder.0.lock(), &event, 99, 90),
        None
    );
    let (_, after) = records(&handle);
    assert_eq!(before.used_bytes, after.used_bytes);
    assert_eq!(
        (after.recorded, after.dropped, after.truncated),
        (1, 1, true)
    );

    let host = self::host(8192, 1);
    let (recorder, handle) = new(&host, 8192);
    let future = Instant::now()
        .checked_add(std::time::Duration::from_secs(60))
        .unwrap();
    assert_eq!(
        handle.record_at(
            future,
            &ImageContinuousTraceEvent::InitializationStarted {
                request_id: Uuid::new_v4()
            }
        ),
        None
    );
    assert_eq!((handle.status().recorded, handle.status().dropped), (0, 1));
    assert!(handle.status().truncated);
    assert!(recorder.0.lock().buffer.as_ref().iter().all(|&b| b == 0));
}

struct EncodingFailure;
impl Serialize for EncodingFailure {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serializer.serialize_struct("EncodingFailure", 2)?;
        value.serialize_field("written_prefix", &123)?;
        Err(serde::ser::Error::custom("deliberate encoding failure"))
    }
}
#[test]
fn serialization_failure_retains_committed_bytes_and_does_not_poison_or_resume() {
    let host = host(4096, 16);
    let (recorder, handle) = new(&host, 2048);
    let original = serde_json::json!({"step":0});
    recorder.record(&original);
    let before = handle.status();
    recorder.record(&EncodingFailure);
    recorder.record(&serde_json::json!({"step":1}));
    let (lines, after) = records(&handle);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["event"], original);
    assert_eq!(after.used_bytes, before.used_bytes);
    assert_eq!((after.recorded, after.dropped), (1, 2));
    assert!(after.truncated);
    assert_eq!(host.usage().unwrap().allocated_bytes(), after.backing_bytes);
}

#[test]
fn unwinding_serializer_cannot_poison_the_collector_or_replace_committed_work() {
    struct Panics;
    impl Serialize for Panics {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut value = serializer.serialize_struct("Panics", 1)?;
            value.serialize_field("partial", &true)?;
            panic!("deliberate observation serializer unwind")
        }
    }
    let host = host(4096, 16);
    let (recorder, handle) = new(&host, 2048);
    recorder.record(&serde_json::json!({"committed":true}));
    let before = handle.status();
    recorder.record(&Panics);
    assert!(!recorder.0.state.is_poisoned());
    recorder.seal();
    let (lines, after) = records(&handle);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["event"]["committed"], true);
    assert_eq!(after.used_bytes, before.used_bytes);
    assert_eq!((after.recorded, after.dropped), (1, 1));
    assert!(after.sealed && after.truncated);
}

#[test]
fn unsealed_export_is_a_snapshot_and_seal_does_not_allow_later_writes() {
    let host = host(4096, 16);
    let (recorder, handle) = new(&host, 2048);
    recorder.record(&serde_json::json!({"step":0}));
    let (lines, first) = records(&handle);
    assert!(!first.sealed && !first.truncated);
    assert_eq!(lines[1]["footer"]["sealed"], false);
    recorder.record(&serde_json::json!({"step":1}));
    recorder.seal();
    let (_, sealed) = records(&handle);
    assert!(sealed.sealed && !sealed.truncated);
    assert_eq!(sealed.recorded, 2);
    recorder.record(&serde_json::json!({"step":2}));
    let (lines, after) = records(&handle);
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[1]["sequence"], 2);
    assert_eq!(after.used_bytes, sealed.used_bytes);
    assert!(after.sealed && after.truncated);
    assert_eq!(after.dropped, 1);
}

#[test]
fn export_io_failure_is_retryable_without_mutating_the_collector() {
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed sink"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let host = host(4096, 16);
    let (recorder, handle) = new(&host, 2048);
    recorder.record(&serde_json::json!({"step":0}));
    recorder.seal();
    let before = handle.status();
    assert!(handle.write_jsonl(&mut Broken).is_err());
    assert_eq!(handle.status(), before);
    let (lines, after) = records(&handle);
    assert_eq!(lines.len(), 2);
    assert_eq!(before, after);
}

#[test]
fn only_the_last_handle_or_recorder_refunds_the_one_original_backing() {
    let host = host(4096, 64);
    let (recorder, handle) = new(&host, 2048);
    let another = handle.clone();
    let physical = handle.status().backing_bytes;
    recorder.record(&serde_json::json!({"members":["a","b"]}));
    drop(recorder);
    assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
    drop(handle);
    assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
    assert_eq!(records(&another).1.recorded, 1);
    drop(another);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
