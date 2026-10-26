use ferrum_types::{FerrumError, Result};
use memmap2::Mmap;
use serde::{Deserialize, Deserializer, Serialize};
#[cfg(any(unix, test))]
use std::ops::Range;

/// Optional OS advice, not a memory reservation or a guarantee of residency.
/// Rejected/unsupported advice is observable and leaves normal reads intact.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SafetensorsReadAheadPolicy {
    #[default]
    Off,
    WillNeed {
        /// Tensors larger than this bound are skipped in full, not prefetched
        /// through an unbounded sequence of smaller advice requests.
        #[serde(deserialize_with = "deserialize_maximum_range_bytes")]
        maximum_range_bytes: u64,
    },
}

fn validate_maximum_range_bytes(bytes: u64) -> Result<()> {
    if bytes == 0 || bytes > isize::MAX as u64 {
        return Err(FerrumError::config(
            "safetensors read-ahead maximum_range_bytes must be positive and addressable",
        ));
    }
    Ok(())
}

fn deserialize_maximum_range_bytes<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<u64, D::Error> {
    let bytes = u64::deserialize(deserializer)?;
    validate_maximum_range_bytes(bytes).map_err(serde::de::Error::custom)?;
    Ok(bytes)
}

impl SafetensorsReadAheadPolicy {
    pub fn validate(self) -> Result<()> {
        match self {
            Self::Off => Ok(()),
            Self::WillNeed {
                maximum_range_bytes,
            } => validate_maximum_range_bytes(maximum_range_bytes),
        }
    }

    pub(super) fn apply(self, mmap: &Mmap, start: usize, end: usize, tensor: &str) {
        let Self::WillNeed {
            maximum_range_bytes,
        } = self
        else {
            return;
        };
        #[cfg(unix)]
        {
            let begin = std::time::Instant::now();
            // SAFETY: sysconf has no pointer arguments or ownership effects.
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            let range = usize::try_from(page_size).ok().and_then(|page_size| {
                full_page_range(start, end, mmap.len(), page_size, maximum_range_bytes).ok()
            });
            match range {
                Some(Some(range)) => {
                    match mmap.advise_range(memmap2::Advice::WillNeed, range.start, range.len()) {
                        Ok(()) => tracing::debug!(
                            tensor,
                            status = "requested",
                            offset_bytes = range.start,
                            length_bytes = range.len(),
                            wall_us = begin.elapsed().as_micros(),
                            "safetensors read-ahead advice"
                        ),
                        Err(error) => tracing::warn!(tensor, status = "rejected", %error,
                        offset_bytes = range.start, length_bytes = range.len(),
                        wall_us = begin.elapsed().as_micros(), "safetensors read-ahead advice"),
                    }
                }
                Some(None) => tracing::debug!(
                    tensor,
                    status = "skipped",
                    maximum_range_bytes,
                    payload_bytes = end.saturating_sub(start),
                    "safetensors read-ahead: above cap or no complete interior page"
                ),
                None => tracing::warn!(
                    tensor,
                    status = "unavailable",
                    page_size,
                    "safetensors read-ahead: invalid page size or unaddressable range"
                ),
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (mmap, start, end, maximum_range_bytes);
            tracing::warn!(
                tensor,
                status = "unsupported",
                "safetensors read-ahead requires Unix"
            );
        }
    }
}

/// Align inward: memmap2 may otherwise round an unaligned start down and advise
/// bytes belonging to the header or a neighboring tensor. File mappings begin
/// on a page boundary. No partial page is included, even at EOF.
#[cfg(any(unix, test))]
fn full_page_range(
    start: usize,
    end: usize,
    mapping_bytes: usize,
    page_size: usize,
    maximum_range_bytes: u64,
) -> Result<Option<Range<usize>>> {
    validate_maximum_range_bytes(maximum_range_bytes)?;
    if page_size == 0 || start > end || end > mapping_bytes || mapping_bytes > isize::MAX as usize {
        return Err(FerrumError::config("invalid safetensors read-ahead range"));
    }
    if (end - start) as u64 > maximum_range_bytes {
        return Ok(None);
    }
    let padding = (page_size - start % page_size) % page_size;
    let aligned_start = start
        .checked_add(padding)
        .ok_or_else(|| FerrumError::config("safetensors read-ahead alignment overflows"))?;
    let aligned_end = end - end % page_size;
    Ok((aligned_start < aligned_end).then_some(aligned_start..aligned_end))
}

#[cfg(test)]
mod tests;
