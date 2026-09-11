//! Resolve missing GPU readback from the same completed Shared counter page.
//!
//! This does not resample timestamps or change their clock conversion. The
//! caller is the completed-fence OnceLock path; a page is CPU-resolved at most
//! once there. Apple's Shared-buffer resolveCounterRange: API is synchronous:
//! https://developer.apple.com/documentation/metal/converting-a-gpus-counter-data-into-a-readable-format

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    MetalCounterPage, NSRange, Object, METAL_COUNTER_ERROR_VALUE, METAL_COUNTER_SAMPLES_PER_PAGE,
};
use metal::objc::{msg_send, sel, sel_impl};

pub(super) struct CounterReadbackStats {
    runtime_instance: u64,
    device_name: String,
    pages: AtomicU64,
    samples: AtomicU64,
}

impl CounterReadbackStats {
    pub(super) fn new(runtime_instance: u64, device_name: &str) -> Self {
        Self {
            runtime_instance,
            device_name: device_name.to_owned(),
            pages: AtomicU64::new(0),
            samples: AtomicU64::new(0),
        }
    }

    pub(super) fn record_cpu_page(&self, sample_count: u64) {
        self.samples.fetch_add(sample_count, Ordering::Relaxed);
        if self.pages.fetch_add(1, Ordering::Relaxed) == 0 {
            tracing::info!(
                runtime_instance = self.runtime_instance,
                device = %self.device_name,
                counter_readback = "shared_cpu_resolve",
                reason = "gpu_readback_contains_zero",
                "Metal profiling resolved missing readback from the original completed counter samples"
            );
        }
    }
}

impl Drop for CounterReadbackStats {
    fn drop(&mut self) {
        let pages = self.pages.load(Ordering::Relaxed);
        if pages > 0 {
            tracing::info!(
                runtime_instance = self.runtime_instance,
                device = %self.device_name,
                counter_readback = "shared_cpu_resolve",
                recovered_pages = pages,
                recovered_samples = self.samples.load(Ordering::Relaxed),
                "Metal profiling counter readback fallback totals"
            );
        }
    }
}

fn validate_samples(
    samples: &[u64],
    start: u64,
    end: u64,
    permit_zero: bool,
) -> Result<(), &'static str> {
    if samples.is_empty() || !samples.len().is_multiple_of(2) {
        return Err("invalid_sample_count");
    }
    for pair in samples.chunks_exact(2) {
        for &sample in pair {
            if sample == METAL_COUNTER_ERROR_VALUE {
                return Err("sample_error_value");
            }
            if sample == 0 && permit_zero {
                continue;
            }
            if sample == 0 || sample < start || sample > end {
                return Err("sample_outside_original_gpu_anchors");
            }
        }
        if pair[0] != 0 && pair[1] != 0 && pair[1] <= pair[0] {
            return Err("end_not_after_start");
        }
    }
    Ok(())
}

pub(super) fn resolve_missing_samples<'a>(
    gpu: &'a [u64],
    anchor_start: u64,
    anchor_end: u64,
    resolve_cpu: impl FnOnce() -> Result<Vec<u64>, &'static str>,
) -> Result<Cow<'a, [u64]>, &'static str> {
    // Real GPU error sentinels, nonzero invalid values and reversed intervals
    // are errors, not a reason to try another source until one looks valid.
    validate_samples(gpu, anchor_start, anchor_end, true)?;
    if !gpu.contains(&0) {
        return Ok(Cow::Borrowed(gpu));
    }
    let cpu = resolve_cpu()?;
    if cpu.len() != gpu.len() {
        return Err("unexpected_cpu_sample_count");
    }
    validate_samples(&cpu, anchor_start, anchor_end, false)?;
    if gpu
        .iter()
        .zip(&cpu)
        .any(|(&gpu, &cpu)| gpu != 0 && gpu != cpu)
    {
        return Err("cpu_gpu_sample_mismatch");
    }
    Ok(Cow::Owned(cpu))
}

fn expected_bytes(sample_count: u64) -> Result<usize, &'static str> {
    if sample_count == 0 || sample_count > METAL_COUNTER_SAMPLES_PER_PAGE {
        return Err("invalid_cpu_sample_range");
    }
    sample_count
        .checked_mul(std::mem::size_of::<u64>() as u64)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or("cpu_resolve_byte_count_overflow")
}

fn decode_cpu_bytes(bytes: &[u8], sample_count: u64) -> Result<Vec<u64>, &'static str> {
    if bytes.len() != expected_bytes(sample_count)? {
        return Err("unexpected_nsdata_length");
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|sample| u64::from_ne_bytes(sample.try_into().expect("complete timestamp bytes")))
        .collect())
}

pub(super) fn resolve_cpu_page(page: &MetalCounterPage) -> Result<Vec<u64>, &'static str> {
    let expected = expected_bytes(page.used_samples)?;
    if page.used_samples > page._sample_buffer.sample_count() {
        return Err("cpu_sample_range_exceeds_buffer");
    }
    metal::objc::rc::autoreleasepool(|| {
        // The metal-rs wrapper creates a zero Vec even for a nil NSData. Keep
        // the actual nullable result and exact length visible instead.
        let data: *mut Object = unsafe {
            msg_send![&*page._sample_buffer, resolveCounterRange: NSRange::new(0, page.used_samples)]
        };
        if data.is_null() {
            return Err("nil_nsdata");
        }
        let length: metal::NSUInteger = unsafe { msg_send![data, length] };
        if usize::try_from(length).ok() != Some(expected) {
            return Err("unexpected_nsdata_length");
        }
        let bytes: *const std::ffi::c_void = unsafe { msg_send![data, bytes] };
        if bytes.is_null() {
            return Err("null_nsdata_bytes");
        }
        // NSData remains alive until pool drain; parsing copies the exact
        // timestamps and makes no u64 alignment assumption about its bytes.
        let bytes = unsafe { std::slice::from_raw_parts(bytes.cast::<u8>(), expected) };
        decode_cpu_bytes(bytes, page.used_samples)
    })
}

#[cfg(test)]
mod tests;
