//! Exact scratch for delta, metadata writes and the ABI's unused LSE span.
//! Upstream noncausal epilogue disables LSE writes; never publish its bytes.
use super::{DeviceSpan, Error, Plan};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceLayout {
    ranges: [(u64, u64); 5],
    bytes: u64,
}
impl WorkspaceLayout {
    pub(crate) fn new(plan: &Plan) -> Result<Self, Error> {
        plan.validate()?;
        let s = plan.bytes;
        let mut ranges = [(0, 0); 5];
        let mut end = 0u64;
        for (range, bytes) in
            ranges
                .iter_mut()
                .zip([s.zero_delta, s.indices, s.counts, s.half_valid, s.lse])
        {
            let start = end.checked_add(15).map(|n| n & !15).ok_or(Error::Extent)?;
            *range = (start, bytes);
            end = start.checked_add(bytes).ok_or(Error::Extent)?;
        }
        let bytes = end
            .checked_add(15)
            .map(|n| n & !15)
            .filter(|&n| n <= isize::MAX as u64)
            .ok_or(Error::Extent)?;
        Ok(Self { ranges, bytes })
    }
    pub(crate) const fn bytes(self) -> u64 {
        self.bytes
    }
    pub(crate) fn spans(self, span: DeviceSpan) -> Result<[DeviceSpan; 5], Error> {
        if span.address == 0
            || span.address % 16 != 0
            || span.bytes != self.bytes
            || span.address.checked_add(span.bytes).is_none()
        {
            return Err(Error::Span);
        }
        Ok(self.ranges.map(|(offset, bytes)| DeviceSpan {
            address: span.address + offset,
            bytes,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Device, Request};
    use super::*;
    #[test]
    fn sage_workspace_charges_padding_and_separates_every_native_write() {
        // Structural fixture only; compiled native Plan supplies actual shared memory.
        let request = Request::new(
            1,
            1,
            1,
            3,
            0.125,
            Device {
                ordinal: 0,
                architecture: 1200,
                multiprocessors: 4,
                maximum_dynamic_shared_bytes: 131072,
            },
        )
        .unwrap();
        let plan = Plan {
            request,
            abi: 1,
            size: std::mem::size_of::<Plan>() as u32,
            kernel_id: 1,
            shared_bytes: 4096,
            threads: 384,
            grid_blocks: 4,
            upload_count: 3,
            memset_count: 1,
            bytes: request.buffer_sizes().unwrap(),
        };
        let layout = WorkspaceLayout::new(&plan).unwrap();
        assert_eq!(layout.bytes(), 2112); // 1536 delta + 12 indices + 4 count + 24 halves + 512 unused LSE, aligned sections
        let base = DeviceSpan {
            address: 0x10000,
            bytes: layout.bytes(),
        };
        let spans = layout.spans(base).unwrap();
        assert_eq!(spans.map(|s| s.bytes), [1536, 12, 4, 24, 512]);
        for pair in spans.windows(2) {
            assert!(pair[0].address + pair[0].bytes <= pair[1].address);
        }
        for bad in [
            DeviceSpan {
                bytes: base.bytes - 1,
                ..base
            },
            DeviceSpan {
                address: base.address + 1,
                ..base
            },
            DeviceSpan {
                address: u64::MAX - 15,
                ..base
            },
        ] {
            assert!(layout.spans(bad).is_err());
        }
    }
}
