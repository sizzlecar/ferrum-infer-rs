use super::*;
use crate::{HostMemoryClassLimits, HostMemoryLimits};

fn account() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            12,
            HostMemoryClassLimits {
                input_preprocess_bytes: 8,
                transfer_staging_bytes: 8,
                output_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

#[test]
fn static_staging_capacity_is_admitted_before_payload_allocation() {
    let host = account();
    let input = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
        .unwrap();
    assert!(matches!(
        stage_upload(&host, &[1, 2, 3, 4, 5, 6, 7, 8]),
        Err(HostMemoryError::Capacity { .. })
    ));
    let usage = host.usage().unwrap();
    assert_eq!(usage.allocated_bytes(), 8);
    assert_eq!(
        usage.class_committed_bytes(HostMemoryClass::TransferStaging),
        0
    );
    drop(input);
    let first = host.reserve(HostMemoryClass::TransferStaging, 8).unwrap();
    assert_eq!(host.usage().unwrap().reserved_bytes(), 8);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    let bytes = copy_staged_upload(first, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    assert_eq!(bytes.bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(host.usage().unwrap().reserved_bytes(), 0);
    drop(bytes);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn upload_owner_keeps_the_exact_staging_charge_until_its_last_reference() {
    let host = account();
    let bytes = stage_upload(&host, &[0x80, 0x3f, 0x00, 0xc0, 0, 0, 0, 0]).unwrap();
    assert!(host.owns(&bytes));
    assert_eq!(bytes.class(), HostMemoryClass::TransferStaging);
    let retained = bytes.clone();
    let subrange = bytes.slice(2..4).unwrap();
    drop(bytes);
    assert!(matches!(
        stage_upload(&host, &[9]),
        Err(HostMemoryError::Capacity { .. })
    ));
    drop(retained);
    assert_eq!(subrange.bytes(), &[0x00, 0xc0]);
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::TransferStaging),
        8
    );
    drop(subrange);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(stage_upload(&host, &[9; 8]).is_ok());
}

#[test]
fn incorrect_staging_reservation_rolls_back_without_a_payload() {
    let host = account();
    let wrong_class = host.reserve(HostMemoryClass::InputPreprocess, 4).unwrap();
    assert!(matches!(
        copy_staged_upload(wrong_class, &[1; 4]),
        Err(HostMemoryError::InvalidReservation)
    ));
    let wrong_length = host.reserve(HostMemoryClass::TransferStaging, 8).unwrap();
    assert!(matches!(
        copy_staged_upload(wrong_length, &[1; 4]),
        Err(HostMemoryError::InvalidReservation)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}
