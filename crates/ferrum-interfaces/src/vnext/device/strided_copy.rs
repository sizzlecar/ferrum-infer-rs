//! Checked rectangles of initialized bytes. Pitch gaps are never copied.
use super::{BufferDescriptor, VNextError};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StridedCopyRegion {
    source_offset_bytes: u64,
    destination_offset_bytes: u64,
    width_bytes: u64,
    height: u64,
    source_pitch_bytes: u64,
    destination_pitch_bytes: u64,
}

impl StridedCopyRegion {
    pub fn new(
        source_offset_bytes: u64,
        destination_offset_bytes: u64,
        width_bytes: u64,
        height: u64,
        source_pitch_bytes: u64,
        destination_pitch_bytes: u64,
    ) -> Result<Self, VNextError> {
        let region = Self {
            source_offset_bytes,
            destination_offset_bytes,
            width_bytes,
            height,
            source_pitch_bytes,
            destination_pitch_bytes,
        };
        if width_bytes == 0
            || height == 0
            || source_pitch_bytes < width_bytes
            || destination_pitch_bytes < width_bytes
        {
            return Err(invalid(
                "strided copy has empty rows or overlapping row pitches",
            ));
        }
        region.source_end_bytes()?;
        region.destination_end_bytes()?;
        region.length_bytes()?;
        Ok(region)
    }
    pub const fn source_offset_bytes(self) -> u64 {
        self.source_offset_bytes
    }
    pub const fn destination_offset_bytes(self) -> u64 {
        self.destination_offset_bytes
    }
    pub const fn width_bytes(self) -> u64 {
        self.width_bytes
    }
    pub const fn height(self) -> u64 {
        self.height
    }
    pub const fn source_pitch_bytes(self) -> u64 {
        self.source_pitch_bytes
    }
    pub const fn destination_pitch_bytes(self) -> u64 {
        self.destination_pitch_bytes
    }
    pub fn source_extent_bytes(self) -> Result<u64, VNextError> {
        self.extent(self.source_pitch_bytes)
    }
    pub fn destination_extent_bytes(self) -> Result<u64, VNextError> {
        self.extent(self.destination_pitch_bytes)
    }
    fn extent(self, pitch: u64) -> Result<u64, VNextError> {
        (self.height - 1)
            .checked_mul(pitch)
            .and_then(|n| n.checked_add(self.width_bytes))
            .ok_or_else(|| invalid("strided copy physical extent overflows u64"))
    }
    pub fn source_end_bytes(self) -> Result<u64, VNextError> {
        self.source_offset_bytes
            .checked_add(self.source_extent_bytes()?)
            .ok_or_else(|| invalid("strided copy source end overflows u64"))
    }
    pub fn destination_end_bytes(self) -> Result<u64, VNextError> {
        self.destination_offset_bytes
            .checked_add(self.destination_extent_bytes()?)
            .ok_or_else(|| invalid("strided copy destination end overflows u64"))
    }
    pub fn length_bytes(self) -> Result<u64, VNextError> {
        self.width_bytes
            .checked_mul(self.height)
            .ok_or_else(|| invalid("strided copy initialized bytes overflow u64"))
    }
    pub fn reversed(self) -> Self {
        Self {
            source_offset_bytes: self.destination_offset_bytes,
            destination_offset_bytes: self.source_offset_bytes,
            source_pitch_bytes: self.destination_pitch_bytes,
            destination_pitch_bytes: self.source_pitch_bytes,
            ..self
        }
    }
    pub fn validate_bounds(
        self,
        source: &BufferDescriptor,
        destination: &BufferDescriptor,
    ) -> Result<(), VNextError> {
        if self.source_end_bytes()? > source.size_bytes
            || self.destination_end_bytes()? > destination.size_bytes
        {
            return Err(invalid("strided copy exceeds a physical buffer boundary"));
        }
        Ok(())
    }
}

fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.to_owned(),
    }
}

pub(crate) struct StridedCopyFragment {
    pub source_segment: usize,
    pub destination_segment: usize,
    /// Offsets are relative to each translated segment's start.
    pub region: StridedCopyRegion,
}

pub(crate) enum StridedCopySplitError<E> {
    LimitExceeded,
    Contract(VNextError),
    Poll(E),
}

/// Shared by native encoding and numeric forecasting. Complete rows within a
/// segment pair become one rectangle; a row crossing an endpoint is split only
/// at that physical boundary. The same widths, heights and pitches are retained.
pub(crate) fn split_strided_copy<E>(
    region: StridedCopyRegion,
    source_lengths: &[u64],
    destination_lengths: &[u64],
    maximum_fragments: usize,
    mut poll: impl FnMut() -> Result<(), E>,
) -> Result<Vec<StridedCopyFragment>, StridedCopySplitError<E>> {
    use StridedCopySplitError::Contract;
    let coverage = |lengths: &[u64], required: u64| -> Result<(), VNextError> {
        if lengths.is_empty()
            || lengths.contains(&0)
            || lengths.iter().try_fold(0_u64, |n, x| n.checked_add(*x)) != Some(required)
        {
            return Err(invalid(
                "strided copy translated segments have invalid coverage",
            ));
        }
        Ok(())
    };
    coverage(
        source_lengths,
        region.source_extent_bytes().map_err(Contract)?,
    )
    .map_err(Contract)?;
    coverage(
        destination_lengths,
        region.destination_extent_bytes().map_err(Contract)?,
    )
    .map_err(Contract)?;
    if region.source_offset_bytes != 0 || region.destination_offset_bytes != 0 {
        return Err(Contract(invalid(
            "strided copy fragment input must use translated origins",
        )));
    }
    let (mut i, mut j, mut source_start, mut destination_start) = (0, 0, 0_u64, 0_u64);
    let (mut row, mut column) = (0_u64, 0_u64);
    let mut result = Vec::new();
    while row < region.height {
        poll().map_err(StridedCopySplitError::Poll)?;
        let a = row * region.source_pitch_bytes + column;
        let b = row * region.destination_pitch_bytes + column;
        while a >= source_start + source_lengths[i] {
            source_start += source_lengths[i];
            i += 1;
        }
        while b >= destination_start + destination_lengths[j] {
            destination_start += destination_lengths[j];
            j += 1;
        }
        let source_remaining = source_start + source_lengths[i] - a;
        let destination_remaining = destination_start + destination_lengths[j] - b;
        let width = (region.width_bytes - column)
            .min(source_remaining)
            .min(destination_remaining);
        let height = if column == 0 && width == region.width_bytes {
            (region.height - row)
                .min((source_remaining - width) / region.source_pitch_bytes + 1)
                .min((destination_remaining - width) / region.destination_pitch_bytes + 1)
        } else {
            1
        };
        if result.len() >= maximum_fragments {
            return Err(StridedCopySplitError::LimitExceeded);
        }
        result.push(StridedCopyFragment {
            source_segment: i,
            destination_segment: j,
            region: StridedCopyRegion::new(
                a - source_start,
                b - destination_start,
                width,
                height,
                region.source_pitch_bytes,
                region.destination_pitch_bytes,
            )
            .map_err(Contract)?,
        });
        if column == 0 && width == region.width_bytes {
            row += height;
        } else {
            column += width;
            if column == region.width_bytes {
                column = 0;
                row += 1;
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
