use super::*;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            32768,
            HostMemoryClassLimits {
                input_preprocess_bytes: 32768,
                transfer_staging_bytes: 32768,
                output_bytes: 32768,
                weight_backing_bytes: 32768,
            },
        )
        .unwrap(),
        4096,
    )
    .unwrap()
}
fn payload(host: &HostMemoryAccount, values: &[i32]) -> ManagedHostBytes {
    let mut p = host
        .allocate_zeroed(HostMemoryClass::TransferStaging, values.len() * 4)
        .unwrap();
    for (dst, n) in p.as_mut().chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&n.to_le_bytes());
    }
    p.freeze()
}
#[test]
fn sage_region_owner_binds_budget_and_retains_original_padded_allocations() {
    let host = host();
    let shape = ListShape::new(2, 3, 4, 5).unwrap();
    let data = payload(&host, &[1, 5, 2]);
    let alias = data.clone();
    let address = data.bytes().as_ptr();
    let owner = CheckedVsaRegions::new(shape, data, payload(&host, &[64; 10])).unwrap();
    assert_eq!(owner.payloads()[0].bytes().as_ptr(), address);
    assert_eq!(owner.selected_tiles(), 2 * 3 * 4 * 3);
    let charged = host.usage().unwrap().allocated_bytes();
    assert_eq!(charged, 2 * (2 * 4096 - 1));
    let changed =
        CheckedVsaRegions::new(shape, payload(&host, &[1, 5, 3]), payload(&host, &[64; 10]))
            .unwrap();
    assert_ne!(owner.digest(), changed.digest());
    let alias = alias.try_thaw().unwrap_err();
    drop(changed);
    drop(owner);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 2 * 4096 - 1);
    assert_eq!(alias.try_thaw().unwrap().as_ref().as_ptr(), address);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn sage_region_workspace_accounts_for_every_write_and_rejects_short_admission() {
    let request = Request::new(
        2,
        3,
        4,
        5,
        0.125,
        super::super::Device {
            ordinal: 0,
            architecture: 1200,
            multiprocessors: 4,
            maximum_dynamic_shared_bytes: 131072,
        },
    )
    .unwrap();
    let base = super::super::Plan {
        request,
        abi: 1,
        size: 192,
        kernel_id: 1,
        shared_bytes: 4096,
        threads: 384,
        grid_blocks: 4,
        upload_count: 3,
        memset_count: 1,
        bytes: request.buffer_sizes().unwrap(),
    };
    let r = RegionRequestV1 {
        attention: request,
        region_count: 2,
        selection_profile: REGION_SELECTION_PROFILE,
    };
    let plan = RegionPlanV1 {
        attention: base,
        region_count: 2,
        selection_profile: REGION_SELECTION_PROFILE,
        selection_threads: 128,
        selection_grid_blocks: 24,
        upload_count: 2,
        memset_count: 1,
        scores_bytes: r.score_bytes().unwrap(),
        regions_bytes: r.region_bytes(),
    };
    let layout = RegionWorkspaceLayout::new(&plan).unwrap();
    let span = DeviceSpan {
        address: 0x10000,
        bytes: layout.bytes(),
    };
    let (attention, regions) = layout.spans(span).unwrap();
    for write in attention {
        assert!(write.address + write.bytes <= regions.address);
    }
    assert_eq!(regions.bytes, 24);
    assert!(regions.address + regions.bytes <= span.address + span.bytes);
    assert!(layout
        .spans(DeviceSpan {
            bytes: span.bytes - 1,
            ..span
        })
        .is_err());
    let mut forged = plan;
    forged.scores_bytes -= 4;
    assert!(RegionWorkspaceLayout::new(&forged).is_err());
    forged = plan;
    forged.upload_count = 3;
    assert!(RegionWorkspaceLayout::new(&forged).is_err());
}
