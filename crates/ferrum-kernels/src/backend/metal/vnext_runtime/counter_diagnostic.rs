//! Failure-only test output. All values come from the failing recording or
//! resolution attempt; this diagnostic never samples a new timestamp anchor.

use super::*;

fn checked_cpu_resolve(
    page: &MetalCounterPage,
    mapping: &MetalCounterIntervalMapping,
) -> serde_json::Value {
    let Some(expected_bytes) = page
        .used_samples
        .checked_mul(std::mem::size_of::<u64>() as u64)
    else {
        return serde_json::json!({ "status": "byte_count_overflow" });
    };
    let sample_count = page._sample_buffer.sample_count();
    if page.used_samples == 0
        || page.used_samples > sample_count
        || mapping.start_sample_index >= page.used_samples
        || mapping.end_sample_index >= page.used_samples
        || expected_bytes > isize::MAX as u64
    {
        return serde_json::json!({
            "status": "invalid_sample_range",
            "sample_count": sample_count,
            "used_samples": page.used_samples,
        });
    }
    metal::objc::rc::autoreleasepool(|| {
        // MTLCounters.h declares nullable NSData*. metal-rs 0.31's wrapper
        // hides a nil result behind an initially zero-filled Vec. Keep the
        // autoreleased NSData alive locally and copy values before pool drain.
        let data: *mut Object = unsafe {
            msg_send![&*page._sample_buffer,
                resolveCounterRange: NSRange::new(0, page.used_samples)
            ]
        };
        if data.is_null() {
            return serde_json::json!({
                "status": "nil_nsdata",
                "expected_bytes": expected_bytes,
            });
        }
        let actual_bytes: metal::NSUInteger = unsafe { msg_send![data, length] };
        if actual_bytes as u64 != expected_bytes {
            return serde_json::json!({
                "status": "unexpected_nsdata_length",
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
            });
        }
        let bytes: *const std::ffi::c_void = unsafe { msg_send![data, bytes] };
        if bytes.is_null() {
            return serde_json::json!({
                "status": "null_nsdata_bytes",
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
            });
        }
        // The timestamp counter set resolves to MTLCounterResultTimestamp,
        // whose SDK layout is one uint64_t. Range and byte length are checked
        // above; NSData's byte pointer need not promise Rust u64 alignment.
        let read = |index: u64| unsafe {
            bytes
                .cast::<u8>()
                .add((index * std::mem::size_of::<u64>() as u64) as usize)
                .cast::<u64>()
                .read_unaligned()
        };
        serde_json::json!({
            "status": "resolved",
            "expected_bytes": expected_bytes,
            "actual_bytes": actual_bytes,
            "raw_start": read(mapping.start_sample_index),
            "raw_end": read(mapping.end_sample_index),
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn report(
    cause: &str,
    anchor_start: (u64, u64),
    anchor_end: Option<(u64, u64)>,
    command: Option<u32>,
    page: Option<usize>,
    sample_indices: Option<(u64, u64)>,
    raw_start: Option<u64>,
    raw_end: Option<u64>,
) {
    eprintln!(
        "ferrum_metal_counter_failure {}",
        serde_json::json!({
            "cause": cause,
            "cpu_anchor_start": anchor_start.0,
            "gpu_anchor_start": anchor_start.1,
            "cpu_anchor_end": anchor_end.map(|anchor| anchor.0),
            "gpu_anchor_end": anchor_end.map(|anchor| anchor.1),
            "command_index": command,
            "page_index": page,
            "start_sample_index": sample_indices.map(|indices| indices.0),
            "end_sample_index": sample_indices.map(|indices| indices.1),
            "raw_start": raw_start,
            "raw_end": raw_end,
        })
    );
}

impl MetalCounterCapture {
    pub(super) fn diagnose(
        &self,
        cause: &str,
        anchor_end: Option<(u64, u64)>,
        mapping: Option<&MetalCounterIntervalMapping>,
        page_samples: &[&[u64]],
    ) {
        let samples = mapping.and_then(|mapping| page_samples.get(mapping.page_index));
        let raw = |index: Option<u64>| {
            index
                .and_then(|index| usize::try_from(index).ok())
                .and_then(|index| samples.and_then(|samples| samples.get(index)))
                .copied()
        };
        report(
            cause,
            (self.cpu_anchor_start, self.gpu_anchor_start),
            anchor_end,
            mapping.map(|mapping| mapping.command_index),
            mapping.map(|mapping| mapping.page_index),
            mapping.map(|mapping| (mapping.start_sample_index, mapping.end_sample_index)),
            raw(mapping.map(|mapping| mapping.start_sample_index)),
            raw(mapping.map(|mapping| mapping.end_sample_index)),
        );
        // measurement() calls resolve() only after this exact command buffer
        // is Completed. This compares the same recorded indices, without
        // submitting work, sampling another anchor, or choosing new timings.
        if let Some(mapping) = mapping {
            let cpu = self
                .pages
                .get(mapping.page_index)
                .map(|page| checked_cpu_resolve(page, mapping));
            eprintln!(
                "ferrum_metal_counter_cpu_gpu_comparison {}",
                serde_json::json!({
                    "cause": cause,
                    "device_name": self.device.name(),
                    "cpu_anchor_start": self.cpu_anchor_start,
                    "gpu_anchor_start": self.gpu_anchor_start,
                    "cpu_anchor_end": anchor_end.map(|anchor| anchor.0),
                    "gpu_anchor_end": anchor_end.map(|anchor| anchor.1),
                    "command_index": mapping.command_index,
                    "page_index": mapping.page_index,
                    "start_sample_index": mapping.start_sample_index,
                    "end_sample_index": mapping.end_sample_index,
                    "gpu_resolved_start": raw(Some(mapping.start_sample_index)),
                    "gpu_resolved_end": raw(Some(mapping.end_sample_index)),
                    "cpu_resolve": cpu,
                })
            );
        }
    }
}
