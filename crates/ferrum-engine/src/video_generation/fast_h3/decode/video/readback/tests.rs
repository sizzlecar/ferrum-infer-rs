use super::*;

fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                weight_backing_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
const OVERLAP: FastH3VideoPostprocessing = FastH3VideoPostprocessing::OverlapIfCapacityAllows;

#[test]
fn optional_decode_credit_never_revokes_earlier_joint_admission() {
    let parent = host(512);
    let first = parent
        .reserve_scope(&[(HostMemoryClass::Output, 128)])
        .unwrap();
    let second = parent
        .reserve_scope(&[(HostMemoryClass::Output, 128)])
        .unwrap();
    let joint = parent
        .reserve_scope(&[(HostMemoryClass::Output, 192)])
        .unwrap();
    // A completed decode may simply stay inline while another phase owns credit.
    assert!(try_admit(OVERLAP, &parent, 64).unwrap().is_none());
    assert_eq!(parent.usage().unwrap().committed_bytes(), 448);
    let preserved = first
        .account()
        .allocate_zeroed(HostMemoryClass::Output, 128)
        .unwrap();
    drop(preserved);
    joint.close().unwrap();
    let grant = try_admit(OVERLAP, &parent, 64).unwrap().unwrap();
    let readback = grant.scope.account();
    let current = readback
        .allocate_zeroed(HostMemoryClass::Output, 64)
        .unwrap();
    let next = readback
        .allocate_zeroed(HostMemoryClass::Output, 64)
        .unwrap();
    assert!(readback
        .allocate_zeroed(HostMemoryClass::Output, 1)
        .is_err());
    drop(grant);
    // Closing a grant refunds only unused credit, not retained CPU readbacks.
    assert_eq!(parent.usage().unwrap().committed_bytes(), 384);
    drop((current, next, first, second));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn hard_capacity_falls_back_but_invalid_authority_and_overflow_stay_errors() {
    let parent = host(127);
    assert!(try_admit(OVERLAP, &parent, 64).unwrap().is_none());
    assert!(try_admit(FastH3VideoPostprocessing::Inline, &parent, 64)
        .unwrap()
        .is_none());
    assert!(matches!(
        try_admit(OVERLAP, &parent, usize::MAX),
        Err(HostMemoryError::ArithmeticOverflow)
    ));
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::Output, 64)])
        .unwrap();
    assert!(matches!(
        try_admit(OVERLAP, &scope.account(), 1),
        Err(HostMemoryError::NestedScope)
    ));
    drop(scope);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}
