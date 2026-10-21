use super::*;

fn waiting(admission: u64, through_tokens: usize) -> PersistentCapacityMember {
    PersistentCapacityMember {
        request_id: RequestId::new(),
        lifecycle: RequestLifecycleId {
            producer: 1,
            admission,
        },
        state: PersistentCapacityState::Waiting,
        through_tokens,
    }
}

fn query(members: &[PersistentCapacityMember]) -> PersistentCapacityQuery<'_> {
    PersistentCapacityQuery {
        members,
        max_live_sequences: 2,
        max_wave_sequences: 2,
        inspection_budget: Default::default(),
    }
}

#[test]
fn persistent_capacity_contract_binds_exact_lifetimes_ceilings_and_incarnations() {
    let issuer = crate::token_execution::ExecutorSequenceIssuer::new();
    let mut first = waiting(1, 64);
    first.state = PersistentCapacityState::Resident {
        sequence: issuer.issue_sequence().unwrap(),
    };
    let members = [first, waiting(2, 128)];
    let snapshot = PersistentCohortCapacitySnapshot::from_verified_query(&query(&members)).unwrap();
    assert_eq!(snapshot.members(), members);
    assert_eq!(snapshot.max_live_sequences(), 2);
    assert_eq!(snapshot.max_wave_sequences(), 2);
    assert_eq!(
        snapshot.retained_bytes(),
        query(&members).retained_bytes().unwrap()
    );
    let mut different = members.clone();
    different[0].state = PersistentCapacityState::Resident {
        sequence: issuer.issue_sequence().unwrap(),
    };
    assert_ne!(snapshot.members(), different);
    different = members.clone();
    different[1].through_tokens += 1;
    assert_ne!(snapshot.members(), different);
}

#[test]
fn persistent_capacity_contract_rejects_duplicate_identity_and_invalid_bounds() {
    let members = [waiting(1, 64), waiting(2, 64)];
    let mut duplicate = members.clone();
    duplicate[1].request_id = duplicate[0].request_id.clone();
    assert!(query(&duplicate).validate().is_err());
    duplicate = members.clone();
    duplicate[1].lifecycle = duplicate[0].lifecycle;
    assert!(query(&duplicate).validate().is_err());
    let identity = crate::token_execution::ExecutorSequenceIssuer::new()
        .issue_sequence()
        .unwrap();
    for member in &mut duplicate {
        member.state = PersistentCapacityState::Resident {
            sequence: identity.clone(),
        };
    }
    duplicate[1].lifecycle.admission = 2;
    assert!(query(&duplicate).validate().is_err());
    let mut invalid = query(&members);
    invalid.max_wave_sequences = 3;
    assert!(invalid.validate().is_err());
    invalid = query(&members);
    invalid.inspection_budget.max_extents = 0;
    assert_eq!(
        invalid.validate(),
        Err(PersistentCapacityUnknown::InspectionBudget)
    );
    assert!(query(&[waiting(1, 0)]).validate().is_err());
    assert!(query(&[]).validate().is_err());
}

#[test]
fn persistent_capacity_contract_bounds_members_and_checked_token_sum() {
    let members = (1..=MAX_PERSISTENT_CAPACITY_MEMBERS + 1)
        .map(|i| waiting(i as u64, 1))
        .collect::<Vec<_>>();
    assert!(query(&members).validate().is_err());
    let maximum = &members[..MAX_PERSISTENT_CAPACITY_MEMBERS];
    assert!(query(maximum).validate().is_ok());
    assert!(query(maximum).retained_bytes().unwrap() <= MAX_PERSISTENT_CAPACITY_RETAINED_BYTES);
    if usize::BITS == 64 {
        assert_eq!(
            query(&[waiting(1, usize::MAX), waiting(2, 1)]).validate(),
            Err(PersistentCapacityUnknown::Arithmetic)
        );
    }
}
