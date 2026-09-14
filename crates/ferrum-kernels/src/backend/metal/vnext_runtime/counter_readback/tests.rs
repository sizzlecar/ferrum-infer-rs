use super::*;
use std::cell::Cell;

#[test]
fn valid_gpu_samples_never_resolve_cpu_or_copy() {
    let gpu = [100, 110, 130, 180];
    let result =
        resolve_missing_samples(&gpu, 100, 200, || panic!("GPU data is complete")).unwrap();
    assert!(matches!(result, Cow::Borrowed(_)));
    assert_eq!(result.as_ref(), gpu);
}

#[test]
fn missing_samples_use_one_exact_cpu_page_preserving_known_values() {
    let calls = Cell::new(0);
    let gpu = [0, 110, 130, 0];
    let result = resolve_missing_samples(&gpu, 100, 200, || {
        calls.set(calls.get() + 1);
        Ok(vec![100, 110, 130, 180])
    })
    .unwrap();
    assert!(matches!(result, Cow::Owned(_)));
    assert_eq!(result.as_ref(), [100, 110, 130, 180]);
    assert_eq!(calls.get(), 1);
    assert_eq!(
        gpu,
        [0, 110, 130, 0],
        "source readback must not be overwritten"
    );
}

#[test]
fn gpu_errors_and_nonzero_bad_values_cannot_be_laundered_through_cpu() {
    for gpu in [
        vec![METAL_COUNTER_ERROR_VALUE, 0],
        vec![0, METAL_COUNTER_ERROR_VALUE],
        vec![99, 0],
        vec![0, 201],
        vec![150, 140, 0, 0],
        vec![150, 150, 0, 0],
        vec![],
        vec![0],
    ] {
        assert!(
            resolve_missing_samples(&gpu, 100, 200, || panic!("invalid GPU source must fail"))
                .is_err()
        );
    }
}

#[test]
fn cpu_nil_zero_errors_wrong_length_and_nonzero_mismatch_remain_unavailable() {
    assert_eq!(
        resolve_missing_samples(&[0, 0], 100, 200, || Err("nil_nsdata")),
        Err("nil_nsdata")
    );
    for cpu in [
        vec![],
        vec![110],
        vec![110, 120, 130],
        vec![0, 120],
        vec![110, 0],
        vec![METAL_COUNTER_ERROR_VALUE, 120],
        vec![110, METAL_COUNTER_ERROR_VALUE],
        vec![99, 120],
        vec![110, 201],
        vec![120, 110],
        vec![110, 110],
    ] {
        assert!(resolve_missing_samples(&[0, 0], 100, 200, || Ok(cpu)).is_err());
    }
    assert_eq!(
        resolve_missing_samples(&[110, 0], 100, 200, || Ok(vec![111, 120])),
        Err("cpu_gpu_sample_mismatch")
    );
    assert_eq!(
        resolve_missing_samples(&[0, 120], 100, 200, || Ok(vec![110, 121])),
        Err("cpu_gpu_sample_mismatch")
    );
}

#[test]
fn cpu_byte_decode_requires_exact_length_and_handles_unaligned_storage() {
    let values = [u64::MAX - 4, u64::MAX - 2];
    let mut storage = vec![0xab];
    storage.extend(values.into_iter().flat_map(u64::to_ne_bytes));
    assert_eq!(decode_cpu_bytes(&storage[1..], 2).unwrap(), values);
    assert!(decode_cpu_bytes(&storage[1..16], 2).is_err());
    assert!(decode_cpu_bytes(&storage, 2).is_err());
    let readback = resolve_missing_samples(&[0, 0], u64::MAX - 8, u64::MAX - 1, || {
        decode_cpu_bytes(&storage[1..], 2)
    })
    .unwrap();
    assert_eq!(
        readback.as_ref(),
        values,
        "timestamps must retain all 64 bits"
    );
}

#[test]
fn cpu_byte_bounds_reject_empty_excess_and_overflowing_ranges() {
    for count in [0, METAL_COUNTER_SAMPLES_PER_PAGE + 1, u64::MAX] {
        assert!(expected_bytes(count).is_err());
    }
    let count = METAL_COUNTER_SAMPLES_PER_PAGE;
    let bytes = vec![0; expected_bytes(count).unwrap()];
    assert_eq!(
        decode_cpu_bytes(&bytes, count).unwrap().len(),
        count as usize
    );
    assert!(decode_cpu_bytes(&bytes[..bytes.len() - 1], count).is_err());
}

#[test]
fn fallback_observation_counts_only_accepted_pages() {
    let stats = CounterReadbackStats::new(1, "test device");
    let page = resolve_missing_samples(&[0, 0], 100, 200, || Ok(vec![110, 120])).unwrap();
    if matches!(page, Cow::Owned(_)) {
        stats.record_cpu_page(page.len() as u64);
    }
    assert_eq!(stats.pages.load(Ordering::Relaxed), 1);
    assert_eq!(stats.samples.load(Ordering::Relaxed), 2);
    assert!(resolve_missing_samples(&[0, 0], 100, 200, || Err("nil_nsdata")).is_err());
    assert_eq!(stats.pages.load(Ordering::Relaxed), 1);
}
