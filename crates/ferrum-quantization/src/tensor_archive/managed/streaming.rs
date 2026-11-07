//! Bounded normalization of metadata-validated immutable BF16 archive ranges.
use super::*;
use std::num::NonZeroUsize;

/// Explicit CPU parallelism within one existing, charged conversion slot.
/// This changes neither conversion arithmetic nor the number of payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16F16ConversionPolicy {
    maximum_workers: NonZeroUsize,
}

impl Bf16F16ConversionPolicy {
    pub const MAXIMUM_WORKERS: usize = 16;
    /// Small continuous source parts stay serial. Each worker handles at
    /// least approximately this many bytes; composite parts are not coalesced.
    pub const MINIMUM_BYTES_PER_WORKER: usize = 1024 * 1024;

    pub fn new(maximum_workers: NonZeroUsize) -> std::result::Result<Self, VNextError> {
        if maximum_workers.get() > Self::MAXIMUM_WORKERS {
            return Err(invalid(
                "conversion worker count exceeds the bounded maximum",
            ));
        }
        Ok(Self { maximum_workers })
    }

    pub const fn maximum_workers(self) -> NonZeroUsize {
        self.maximum_workers
    }

    fn workers_for(self, bytes: usize) -> usize {
        self.maximum_workers
            .get()
            .min(bytes / Self::MINIMUM_BYTES_PER_WORKER)
            .max(1)
    }
}

impl Default for Bf16F16ConversionPolicy {
    fn default() -> Self {
        Self {
            maximum_workers: NonZeroUsize::MIN,
        }
    }
}

/// Complete source metadata bound to the original archive. Preparing this
/// descriptor reads no tensor values and allocates no converted payload.
/// Execution-plan identity and placement remain the caller's responsibility.
pub struct PreparedBf16F16Component<'archive> {
    _archive: &'archive dyn TensorArchive,
    component: WeightComponentSpec,
    source_files: Vec<String>,
    parts: Vec<RetainedHostMemoryRegion>,
    full_bytes: usize,
}

/// One continuously charged Host allocation. Every published range retains
/// this entire allocation, including padding. Native users must release all
/// aliases after their terminal fence before a subsequent conversion can write.
pub struct ManagedDenseArchiveStreamingSlot<'host> {
    host: &'host HostMemoryAccount,
    backing: ReusableBacking,
}

impl<'archive> ManagedDenseArchiveSource<'archive> {
    /// Validate a plain BF16 -> dense F16 single tensor or ordered composite.
    /// `None` means another declared representation is required; malformed
    /// metadata is an error. This does not arm or allocate an installation.
    pub fn prepare_bf16_f16_stream(
        &self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<Option<PreparedBf16F16Component<'archive>>, VNextError> {
        if component.encoding
            != (WeightEncoding::Dense {
                element_type: ElementType::F16,
            })
        {
            return Ok(None);
        }
        if component.external_names.is_empty()
            || component.dimensions.is_empty()
            || component.dimensions.contains(&0)
        {
            return Err(invalid(
                "component requires named sources and a positive shape",
            ));
        }
        let full_bytes = usize::try_from(component.physical_bytes()?)
            .ok()
            .filter(|&bytes| bytes > 0 && bytes <= isize::MAX as usize)
            .ok_or_else(|| invalid("component exceeds address space"))?;
        let tensors = component
            .external_names
            .iter()
            .map(|name| {
                let tensor = self
                    .archive
                    .tensor(name)
                    .map_err(|error| invalid(error.to_string()))?;
                if tensor.external_name() != name || !valid_source_file(tensor.source_file()) {
                    return Err(invalid(
                        "archive source identity differs from the component",
                    ));
                }
                Ok(tensor)
            })
            .collect::<std::result::Result<Vec<_>, VNextError>>()?;
        let shape_matches = if let [tensor] = tensors.as_slice() {
            tensor.shape() == component.dimensions
        } else {
            let stacked = component.dimensions.first().copied()
                == u64::try_from(tensors.len()).ok()
                && tensors
                    .iter()
                    .all(|tensor| tensor.shape() == &component.dimensions[1..]);
            let concatenated = tensors.iter().all(|tensor| {
                tensor.shape().len() == component.dimensions.len()
                    && tensor.shape()[1..] == component.dimensions[1..]
            }) && tensors.iter().try_fold(0u64, |rows, tensor| {
                rows.checked_add(*tensor.shape().first()?)
            }) == component.dimensions.first().copied();
            stacked || concatenated
        };
        if !shape_matches {
            return Err(invalid(
                "source shapes differ from the exact component/stack/concatenation",
            ));
        }
        if tensors.iter().any(|tensor| {
            tensor.encoding()
                != &(WeightEncoding::Dense {
                    element_type: ElementType::Bf16,
                })
        }) {
            return Ok(None);
        }
        let mut total = 0usize;
        for tensor in &tensors {
            let elements = tensor.shape().iter().try_fold(1usize, |n, &dimension| {
                usize::try_from(dimension)
                    .ok()
                    .and_then(|d| n.checked_mul(d))
            });
            if elements.and_then(|n| n.checked_mul(2)) != Some(tensor.bytes().len()) {
                return Err(invalid("source extent differs from its validated shape"));
            }
            total = total
                .checked_add(tensor.bytes().len())
                .ok_or_else(|| invalid("combined source extent overflows"))?;
        }
        if total != full_bytes {
            return Err(invalid(
                "combined source extent differs from the full component",
            ));
        }
        Ok(Some(PreparedBf16F16Component {
            _archive: self.archive,
            component: component.clone(),
            source_files: tensors.iter().map(|t| t.source_file().to_owned()).collect(),
            parts: tensors
                .iter()
                .map(|t| t.retained_host_memory().clone())
                .collect(),
            full_bytes,
        }))
    }

    /// Allocate only after the caller validates its original admitted Host
    /// scope and computes physical capacity with `HostMemoryAccount::backing_bytes`.
    /// The logical capacity must hold complete F16 elements. No second slot or
    /// unaccounted fallback is allocated if this reservation is rejected.
    pub fn bounded_streaming_slot(
        &self,
        logical_bytes: NonZeroUsize,
    ) -> std::result::Result<ManagedDenseArchiveStreamingSlot<'archive>, VNextError> {
        if !logical_bytes.get().is_multiple_of(2) {
            return Err(invalid("stream slot requires complete F16 elements"));
        }
        Ok(ManagedDenseArchiveStreamingSlot {
            host: self.host,
            backing: ReusableBacking::new(self.host, logical_bytes)?,
        })
    }
}

impl ManagedDenseArchiveStreamingSlot<'_> {
    pub fn logical_bytes(&self) -> usize {
        self.backing.logical_bytes()
    }

    pub fn backing_bytes(&self) -> usize {
        self.backing.physical_bytes()
    }
}

impl PreparedBf16F16Component<'_> {
    pub fn component(&self) -> &WeightComponentSpec {
        &self.component
    }

    pub fn source_files(&self) -> &[String] {
        &self.source_files
    }

    pub fn full_bytes(&self) -> usize {
        self.full_bytes
    }

    /// Normalize exactly this element-aligned component range using the same
    /// converter as the full source. This is not a complete component payload.
    /// Core must separately enforce ordered full coverage and native custody.
    pub fn materialize_range(
        &self,
        slot: &ManagedDenseArchiveStreamingSlot<'_>,
        offset_bytes: usize,
        length_bytes: usize,
    ) -> std::result::Result<ferrum_runtime::ManagedHostBytes, VNextError> {
        self.materialize_range_with_policy(
            slot,
            offset_bytes,
            length_bytes,
            Bf16F16ConversionPolicy::default(),
        )
    }

    /// Convert disjoint, element-aligned portions of each sufficiently large
    /// contiguous source part. Every started worker is joined before returning
    /// success or failure; only complete success freezes the original slot.
    /// No second payload, global thread pool, or upload overlap is introduced.
    pub fn materialize_range_with_policy(
        &self,
        slot: &ManagedDenseArchiveStreamingSlot<'_>,
        offset_bytes: usize,
        length_bytes: usize,
        policy: Bf16F16ConversionPolicy,
    ) -> std::result::Result<ferrum_runtime::ManagedHostBytes, VNextError> {
        let end = offset_bytes
            .checked_add(length_bytes)
            .filter(|&end| end <= self.full_bytes)
            .ok_or_else(|| invalid("stream range exceeds the component"))?;
        if length_bytes == 0
            || !offset_bytes.is_multiple_of(2)
            || !length_bytes.is_multiple_of(2)
            || length_bytes > slot.logical_bytes()
        {
            return Err(invalid(
                "stream range requires complete elements within the slot",
            ));
        }
        let mut destination = BackingLease::allocate(slot.host, length_bytes, Some(&slot.backing))
            .map_err(invalid)?;
        let mut part_start = 0usize;
        let mut written = 0usize;
        for part in &self.parts {
            let part_end = part_start + part.length_bytes(); // total checked during preparation
            let start = offset_bytes.max(part_start);
            let stop = end.min(part_end);
            if start < stop {
                let bytes = stop - start;
                convert_part_with(
                    &part.bytes()[start - part_start..stop - part_start],
                    &mut destination.as_mut()[written..written + bytes],
                    policy,
                    |_, input, output| bf16_to_f16::convert(input, output),
                    |_| Ok(()),
                )?;
                written += bytes;
            }
            part_start = part_end;
            if part_start >= end {
                break;
            }
        }
        if written != length_bytes {
            return Err(invalid(
                "prepared source parts did not cover the complete range",
            ));
        }
        destination.freeze()
    }
}

// The narrow callbacks let lifecycle tests inject a converter panic or a
// failed launch while using real scoped workers and the original Host slot.
// Production always uses the unchanged converter and an infallible prelaunch.
pub(super) fn convert_part_with(
    input: &[u8],
    output: &mut [u8],
    policy: Bf16F16ConversionPolicy,
    convert: impl Fn(usize, &[u8], &mut [u8]) -> std::result::Result<(), &'static str> + Sync,
    mut before_spawn: impl FnMut(usize) -> std::io::Result<()>,
) -> std::result::Result<(), VNextError> {
    if input.len() != output.len() || !input.len().is_multiple_of(2) {
        return Err(invalid(
            "parallel conversion requires equal whole-element extents",
        ));
    }
    let workers = policy.workers_for(input.len());
    if workers == 1 {
        return std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            convert(0, input, output).map_err(invalid)
        }))
        .unwrap_or_else(|_| Err(invalid("serial conversion panicked")));
    }
    let chunk_bytes = (input.len() / 2).div_ceil(workers) * 2;
    // Even a panic in launch/bookkeeping waits for the scope's workers before
    // becoming an ordinary error. It cannot unwind through the slot's mutex
    // guard and poison an otherwise recoverable CPU conversion allocation.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            let mut error = None;
            for (worker, (source, destination)) in input
                .chunks(chunk_bytes)
                .zip(output.chunks_mut(chunk_bytes))
                .enumerate()
            {
                let convert = &convert;
                let launched = before_spawn(worker).and_then(|()| {
                    std::thread::Builder::new()
                        .name(format!("bf16-f16-range-{worker}"))
                        .spawn_scoped(scope, move || convert(worker, source, destination))
                });
                match launched {
                    Ok(handle) => handles.push((worker, handle)),
                    Err(cause) => {
                        error = Some(invalid(format!(
                            "conversion worker {worker} could not start: {cause}"
                        )));
                        break;
                    }
                }
            }
            // Do not use early-return `?`: all successful launches must join
            // even if another worker failed conversion or panicked.
            for (worker, handle) in handles {
                let failure = match handle.join() {
                    Ok(Ok(())) => None,
                    Ok(Err(cause)) => Some(invalid(format!("conversion worker {worker}: {cause}"))),
                    Err(_) => Some(invalid(format!("conversion worker {worker} panicked"))),
                };
                if error.is_none() {
                    error = failure;
                }
            }
            error.map_or(Ok(()), Err)
        })
    }));
    result.unwrap_or_else(|_| Err(invalid("conversion scope panicked after worker retirement")))
}

fn valid_source_file(file: &str) -> bool {
    !file.is_empty()
        && !file.starts_with('/')
        && !file.contains('\\')
        && file.split('/').all(|part| !matches!(part, "" | "." | ".."))
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("managed archive stream: {}", reason.into()),
    }
}
