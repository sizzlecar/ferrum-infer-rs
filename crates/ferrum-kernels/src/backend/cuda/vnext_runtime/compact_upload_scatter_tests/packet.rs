//! Transport-fixture packet validation, not Core resource authorization.
//!
//! Each descriptor is five little-endian u64s: source offset from the packet
//! start, destination offset from the arena start, row bytes, rows, destination
//! pitch. All descriptors precede the tightly packed payloads, without padding.

use super::{CudaProgramBindingTransfer, Result};

const DESCRIPTOR_BYTES: usize = 5 * std::mem::size_of::<u64>();

#[derive(Debug)]
pub(super) struct Packet {
    pub(super) bytes: Vec<u8>,
    pub(super) descriptor_count: u32,
    pub(super) live_bytes: usize,
}

impl Packet {
    pub(super) fn validate(
        transfers: &[CudaProgramBindingTransfer],
        arena_bytes: usize,
    ) -> Result<()> {
        if transfers.is_empty() || arena_bytes == 0 {
            return Err("scatter packet transfers or arena are empty".into());
        }
        u32::try_from(transfers.len()).map_err(|_| "scatter descriptor count exceeds u32")?;
        let descriptor_bytes = transfers
            .len()
            .checked_mul(DESCRIPTOR_BYTES)
            .ok_or("scatter descriptor bytes overflow")?;
        let arena_bytes =
            u64::try_from(arena_bytes).map_err(|_| "scatter arena bytes exceed u64")?;
        let mut live_bytes = 0_usize;
        let mut destination_rows = Vec::new();
        for transfer in transfers {
            if transfer.row_bytes == 0 || transfer.row_count == 0 {
                return Err("scatter transfer has zero row width or count".into());
            }
            let payload_bytes = transfer
                .row_bytes
                .checked_mul(transfer.row_count)
                .ok_or("scatter packed payload bytes overflow")?;
            if payload_bytes != transfer.payload.len() {
                return Err("scatter packed payload length differs from its rows".into());
            }
            let row_bytes =
                u64::try_from(transfer.row_bytes).map_err(|_| "scatter row bytes exceed u64")?;
            let rows =
                u64::try_from(transfer.row_count).map_err(|_| "scatter row count exceeds u64")?;
            let last_end = (rows - 1)
                .checked_mul(transfer.destination_stride_bytes)
                .and_then(|offset| transfer.destination_offset_bytes.checked_add(offset))
                .and_then(|offset| offset.checked_add(row_bytes))
                .ok_or("scatter destination extent overflows")?;
            if last_end > arena_bytes {
                return Err("scatter destination exceeds arena".into());
            }
            live_bytes = live_bytes
                .checked_add(payload_bytes)
                .ok_or("scatter live payload bytes overflow")?;
            destination_rows
                .try_reserve_exact(transfer.row_count)
                .map_err(|_| "scatter destination row capacity unavailable")?;
            for row in 0..rows {
                let start = row
                    .checked_mul(transfer.destination_stride_bytes)
                    .and_then(|offset| transfer.destination_offset_bytes.checked_add(offset))
                    .ok_or("scatter destination row offset overflows")?;
                let end = start
                    .checked_add(row_bytes)
                    .ok_or("scatter destination row end overflows")?;
                destination_rows.push((start, end));
            }
        }
        // Bounding envelopes may overlap: only actual written row intervals
        // conflict. This also rejects overlapping rows within one descriptor.
        destination_rows.sort_unstable();
        if destination_rows
            .windows(2)
            .any(|pair| pair[0].1 > pair[1].0)
        {
            return Err("scatter destination rows overlap".into());
        }
        let packet_bytes = descriptor_bytes
            .checked_add(live_bytes)
            .ok_or("scatter packet byte count overflows")?;
        u64::try_from(packet_bytes).map_err(|_| "scatter packet bytes exceed u64")?;
        Ok(())
    }

    pub(super) fn prepare(
        transfers: &[CudaProgramBindingTransfer],
        arena_bytes: usize,
    ) -> Result<Self> {
        // The direct arm calls the same validation, without constructing a
        // packet. Packing below is additional work measured by the scatter arm.
        Self::validate(transfers, arena_bytes)?;
        let descriptor_count =
            u32::try_from(transfers.len()).map_err(|_| "scatter descriptor count exceeds u32")?;
        let descriptor_bytes = transfers
            .len()
            .checked_mul(DESCRIPTOR_BYTES)
            .ok_or("scatter descriptor bytes overflow")?;
        let live_bytes = transfers.iter().try_fold(0_usize, |total, transfer| {
            total
                .checked_add(transfer.payload.len())
                .ok_or("scatter live payload bytes overflow")
        })?;
        let packet_bytes = descriptor_bytes
            .checked_add(live_bytes)
            .ok_or("scatter packet byte count overflows")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(packet_bytes)
            .map_err(|_| "scatter packet capacity unavailable")?;
        let mut source_offset = descriptor_bytes;
        for transfer in transfers {
            let fields = [
                u64::try_from(source_offset).map_err(|_| "scatter source offset exceeds u64")?,
                transfer.destination_offset_bytes,
                u64::try_from(transfer.row_bytes).map_err(|_| "scatter row bytes exceed u64")?,
                u64::try_from(transfer.row_count).map_err(|_| "scatter row count exceeds u64")?,
                transfer.destination_stride_bytes,
            ];
            for field in fields {
                bytes.extend_from_slice(&field.to_le_bytes());
            }
            source_offset = source_offset
                .checked_add(transfer.payload.len())
                .ok_or("scatter source payload end overflows")?;
        }
        for transfer in transfers {
            bytes.extend_from_slice(&transfer.payload);
        }
        Ok(Self {
            bytes,
            descriptor_count,
            live_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transfer(
        offset: u64,
        pitch: u64,
        width: usize,
        rows: usize,
        payload: &[u8],
    ) -> CudaProgramBindingTransfer {
        CudaProgramBindingTransfer {
            destination_offset_bytes: offset,
            destination_stride_bytes: pitch,
            row_bytes: width,
            row_count: rows,
            payload: payload.into(),
        }
    }

    fn descriptor(packet: &Packet, index: usize) -> [u64; 5] {
        let start = index * 40;
        std::array::from_fn(|field| {
            let offset = start + field * 8;
            u64::from_le_bytes(packet.bytes[offset..offset + 8].try_into().unwrap())
        })
    }

    #[test]
    fn compact_packet_encodes_2d_rows_and_absolute_payload_offsets() {
        let transfers = [
            transfer(4, 8, 3, 2, &[1, 2, 3, 4, 5, 6]),
            transfer(20, 0, 2, 1, &[7, 8]),
        ];
        Packet::validate(&transfers, 22).unwrap();
        let packet = Packet::prepare(&transfers, 22).unwrap();
        assert_eq!(packet.descriptor_count, 2);
        assert_eq!(packet.live_bytes, 8);
        assert_eq!(descriptor(&packet, 0), [80, 4, 3, 2, 8]);
        assert_eq!(descriptor(&packet, 1), [86, 20, 2, 1, 0]);
        assert_eq!(&packet.bytes[80..], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(packet.bytes.len(), 88);
    }

    #[test]
    fn compact_packet_accepts_interleaved_destination_holes() {
        let packet = Packet::prepare(
            &[transfer(4, 8, 4, 2, &[2; 8]), transfer(0, 8, 4, 2, &[1; 8])],
            16,
        )
        .unwrap();
        assert_eq!(descriptor(&packet, 0), [80, 4, 4, 2, 8]);
        assert_eq!(descriptor(&packet, 1), [88, 0, 4, 2, 8]);
        assert_eq!(&packet.bytes[80..88], &[2; 8]);
        assert_eq!(&packet.bytes[88..], &[1; 8]);
    }

    #[test]
    fn compact_packet_rejects_cross_transfer_and_self_overlap() {
        let transfers = [transfer(0, 8, 4, 2, &[1; 8]), transfer(7, 0, 2, 1, &[2; 2])];
        assert!(Packet::validate(&transfers, 16).is_err());
        assert!(Packet::prepare(&transfers, 16).is_err());
        assert!(Packet::prepare(&[transfer(0, 3, 4, 2, &[1; 8])], 16).is_err());
        assert!(Packet::prepare(&[transfer(0, 0, 1, 2, &[1; 2])], 16).is_err());
    }

    #[test]
    fn compact_packet_rejects_payload_and_destination_overflow() {
        assert!(Packet::prepare(&[transfer(0, 1, usize::MAX, 2, &[])], 16).is_err());
        assert!(Packet::prepare(&[transfer(u64::MAX, 0, 1, 1, &[1])], 16).is_err());
        assert!(Packet::prepare(&[transfer(1, u64::MAX, 1, 2, &[1; 2])], 16).is_err());
        assert!(Packet::prepare(&[transfer(0, u64::MAX, 1, 3, &[1; 3])], 16).is_err());
    }

    #[test]
    fn compact_packet_rejects_short_or_long_payload_and_out_of_bounds_rows() {
        assert!(Packet::prepare(&[transfer(0, 8, 4, 2, &[1; 7])], 16).is_err());
        assert!(Packet::prepare(&[transfer(0, 8, 4, 2, &[1; 9])], 16).is_err());
        assert!(Packet::prepare(&[transfer(0, 8, 4, 2, &[1; 8])], 11).is_err());
        assert!(Packet::prepare(&[transfer(16, 0, 1, 1, &[1])], 16).is_err());
    }

    #[test]
    fn compact_packet_rejects_empty_input_arena_or_rows() {
        assert!(Packet::prepare(&[], 16).is_err());
        assert!(Packet::prepare(&[transfer(0, 0, 1, 1, &[1])], 0).is_err());
        assert!(Packet::prepare(&[transfer(0, 0, 0, 1, &[])], 16).is_err());
        assert!(Packet::prepare(&[transfer(0, 0, 1, 0, &[])], 16).is_err());
    }
}
