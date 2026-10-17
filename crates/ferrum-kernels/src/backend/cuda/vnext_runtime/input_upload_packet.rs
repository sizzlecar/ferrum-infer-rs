//! Fresh byte-exact input transport after Core has authorized every upload.
//! This only changes the physical submission, never logical resource authority.

use super::{
    CudaDeviceCommand, CudaDeviceRuntime, CudaDeviceRuntimeError, CudaWorkDeclaration,
    DeviceCommandPhase, HOST_UPLOAD_NATIVE_OPERATION_ID,
};

type Result<T> = std::result::Result<T, CudaDeviceRuntimeError>;
type Entry = (DeviceCommandPhase, Option<u32>, CudaDeviceCommand);
const HEADER_BYTES: usize = 24;
// A small eager packet amortizes the extra launch. Bulk transfers keep their
// direct-copy implementation; this path is intended for per-request scalars.
const MIN_UPLOADS: usize = 8;
const MAX_UPLOAD_BYTES: usize = 256;

pub(super) fn coalesce(runtime: &CudaDeviceRuntime, entries: Vec<Entry>) -> Result<Vec<Entry>> {
    if entries
        .iter()
        .any(|(_, _, command)| command.runtime_instance != runtime.runtime_instance)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA input batch contains work from another runtime instance",
        ));
    }
    let mut output = Vec::with_capacity(entries.len());
    let mut entries = entries.into_iter().peekable();
    while let Some(entry) = entries.next() {
        if !eligible(&entry) {
            output.push(entry);
            continue;
        }
        let mut run = vec![entry];
        while entries.peek().is_some_and(eligible) {
            run.push(entries.next().expect("peeked CUDA input upload"));
        }
        if run.len() >= MIN_UPLOADS {
            if let Some(command) = compact(runtime, &run)? {
                output.push((DeviceCommandPhase::DynamicBinding, None, command));
                continue;
            }
        }
        // No destination write has occurred. Overlap, scratch limits, host
        // capacity, and allocation OOM preserve every original ordered copy.
        output.extend(run);
    }
    Ok(output)
}

fn eligible((phase, node_index, command): &Entry) -> bool {
    if *phase != DeviceCommandPhase::DynamicBinding
        || node_index.is_some()
        || command.operation != HOST_UPLOAD_NATIVE_OPERATION_ID.as_str()
        || command.participant_start != 0
        || command.participant_count != 0
        || command.token_count != 0
        || command.compute_dispatch_count != 0
        || command.transfer_command_count != 1
        || command.replay_key.is_some()
        || command.reusable_address_scope.is_some()
        || command.program_binding_patch.is_some()
        || command.binding_program.is_some()
        || command.reusable_execution.is_some()
        || !command.fence_dependencies.is_empty()
        || !command.binding_owner_tables.is_empty()
        || !command.completion_checks.is_empty()
    {
        return false;
    }
    let Some(executable) = &command.executable else {
        return false;
    };
    let Some(layout) = executable.host_upload_layout else {
        return false;
    };
    let ([region], [bytes]) = (&executable.regions[..], &executable.host_storage[..]) else {
        return false;
    };
    executable.work_declaration == CudaWorkDeclaration::Static
        && !bytes.is_empty()
        && bytes.len() <= MAX_UPLOAD_BYTES
        && layout.validate_bytes(bytes.len()).is_ok()
        && layout.element_type() == region.element_type
        && region.runtime_instance == command.runtime_instance
        && region.length_bytes == bytes.len() as u64
}

fn compact(runtime: &CudaDeviceRuntime, run: &[Entry]) -> Result<Option<CudaDeviceCommand>> {
    let writes = run
        .iter()
        .map(|(_, _, command)| {
            let executable = command.executable.as_ref().expect("typed input executable");
            Write {
                address: executable.regions[0].device_ptr,
                bytes: &executable.host_storage[0],
            }
        })
        .collect::<Vec<_>>();
    let Some(packet) = Packet::prepare(&writes, runtime.input_upload_scatter.packet_limit())?
    else {
        return Ok(None);
    };
    let Some(owner) = runtime.input_upload_scatter.prepare_addressed(
        &runtime.allocation_stream,
        packet.bytes.len(),
        packet.count,
    )?
    else {
        return Ok(None);
    };
    let regions = run
        .iter()
        .map(|(_, _, command)| {
            command
                .executable
                .as_ref()
                .expect("typed input executable")
                .regions[0]
                .clone()
        })
        .collect();
    let mut command = CudaDeviceCommand::transfer(
        runtime.runtime_instance,
        "vnext_addressed_input_upload_v1",
        regions,
        vec![packet.bytes.into_boxed_slice()],
        Box::new(move |stream, _, _, storage| {
            owner.upload(stream, &storage[0])?;
            owner.scatter_addressed(stream)
        }),
    );
    // All addresses and source bytes are fresh and belong only to this command.
    // Existing submit quarantines this command after *any* enqueue error,
    // including a failed scatter after a successful H2D. Never retry the copies.
    command.compute_dispatch_count = 1;
    Ok(Some(command))
}

struct Write<'a> {
    address: u64,
    bytes: &'a [u8],
}

struct Packet {
    bytes: Vec<u8>,
    count: u32,
}

impl Packet {
    fn prepare(writes: &[Write<'_>], limit: usize) -> Result<Option<Self>> {
        let invalid = CudaDeviceRuntimeError::contract;
        let count = u32::try_from(writes.len())
            .map_err(|_| invalid("input packet descriptor count exceeds u32"))?;
        let header = writes
            .len()
            .checked_mul(HEADER_BYTES)
            .ok_or_else(|| invalid("input packet headers overflow"))?;
        let total = writes.iter().try_fold(header, |total, write| {
            total
                .checked_add(write.bytes.len())
                .ok_or_else(|| invalid("input packet payload overflows"))
        })?;
        if writes.is_empty() || total > limit {
            return Ok(None);
        }
        let mut ranges = Vec::new();
        let mut bytes = Vec::new();
        if ranges.try_reserve_exact(writes.len()).is_err()
            || bytes.try_reserve_exact(total).is_err()
        {
            return Ok(None);
        }
        for write in writes {
            let length = u64::try_from(write.bytes.len())
                .map_err(|_| invalid("input packet payload exceeds u64"))?;
            let end = write
                .address
                .checked_add(length)
                .ok_or_else(|| invalid("input packet destination address overflows"))?;
            if write.address == 0 || length == 0 {
                return Err(invalid("input packet destination is empty"));
            }
            ranges.push((write.address, end));
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Ok(None);
        }
        let mut source = header;
        for write in writes {
            let source_u64 = u64::try_from(source)
                .map_err(|_| invalid("input packet source offset exceeds u64"))?;
            for field in [source_u64, write.address, write.bytes.len() as u64] {
                bytes.extend_from_slice(&field.to_le_bytes());
            }
            source += write.bytes.len();
        }
        for write in writes {
            bytes.extend_from_slice(write.bytes);
        }
        Ok(Some(Self { bytes, count }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_packet_preserves_independent_destinations_and_payload_order() {
        let writes = [
            Write {
                address: 0x90_000,
                bytes: &[1, 2, 3, 4],
            },
            Write {
                address: 0x10_008,
                bytes: &[5, 6, 7, 8, 9, 10, 11, 12],
            },
            Write {
                address: 0x10_004,
                bytes: &[13, 14, 15, 16],
            },
        ];
        let packet = Packet::prepare(&writes, 1024).unwrap().unwrap();
        assert_eq!(packet.count, 3);
        let mut source = writes.len() * HEADER_BYTES;
        for (index, write) in writes.iter().enumerate() {
            let descriptor = &packet.bytes[index * HEADER_BYTES..][..HEADER_BYTES];
            let fields = descriptor
                .chunks_exact(8)
                .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                fields,
                [source as u64, write.address, write.bytes.len() as u64]
            );
            assert_eq!(
                &packet.bytes[source..source + write.bytes.len()],
                write.bytes
            );
            source += write.bytes.len();
        }
        assert_eq!(source, packet.bytes.len());
    }

    #[test]
    fn input_packet_overlap_and_capacity_preserve_original_copy_route() {
        let writes = [
            Write {
                address: 8,
                bytes: &[1; 8],
            },
            Write {
                address: 12,
                bytes: &[2; 4],
            },
        ];
        assert!(Packet::prepare(&writes, 1024).unwrap().is_none());
        let writes = [
            Write {
                address: 8,
                bytes: &[1; 8],
            },
            Write {
                address: 16,
                bytes: &[2; 4],
            },
        ];
        assert!(Packet::prepare(&writes, 59).unwrap().is_none());
        assert!(Packet::prepare(&writes, 60).unwrap().is_some());
        let writes = [Write {
            address: u64::MAX - 3,
            bytes: &[1; 4],
        }];
        assert!(Packet::prepare(&writes, 1024).is_err());
        assert!(Packet::prepare(
            &[Write {
                address: 1,
                bytes: &[]
            }],
            1024
        )
        .is_err());
    }
}
