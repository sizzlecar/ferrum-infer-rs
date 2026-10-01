use super::super::tests::{assert_unsubmitted_clean, fixture, limits, requests};
use super::*;
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use ferrum_types::TokenId;

async fn tokenizer() -> HuggingFaceTokenizer {
    let vocab = (0..64)
        .map(|i| {
            let surface = match i {
                0 => "<unk>".to_owned(),
                10 => "a".to_owned(),
                11 => "Ã".to_owned(),
                12 => "©".to_owned(),
                _ => format!("v{i}"),
            };
            (surface, i)
        })
        .collect();
    let mut tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    HuggingFaceTokenizer::from_source_bytes(
        tokenizer.to_string(false).unwrap().as_bytes(),
        None,
        None,
    )
    .await
    .unwrap()
}

fn constraint(
    tokenizer: &dyn Tokenizer,
    request_id: RequestId,
    ids: &[u32],
) -> GeometryPrefixConstraint {
    let token_ids: Vec<_> = ids.iter().copied().map(TokenId::new).collect();
    let mut scratch = vec![0; tokenizer.bounded_token_bytes_bound().unwrap().get()];
    let token_bytes = token_ids
        .iter()
        .map(|&id| {
            let written = tokenizer
                .token_bytes_bounded_into(id, &mut scratch)
                .unwrap()
                .unwrap();
            scratch[..written].to_vec()
        })
        .collect();
    GeometryPrefixConstraint {
        request_id,
        release_generated: ids.len() as u32,
        slot: StructuredPrefixSlotV5 {
            tokenizer_policy_sha256: tokenizer.host_output_policy_identity().unwrap(),
            token_ids,
            token_bytes,
        },
    }
}

#[tokio::test]
async fn geometry_prefix_checks_bytes_and_preserves_each_history_frontier() {
    let tokenizer = tokenizer().await;
    let budget = limits();
    let mut declared = constraint(&tokenizer, RequestId::new(), &[10, 11, 12]);
    let checked = ValidatedPrefix::validate(
        &tokenizer,
        4,
        &declared,
        &budget,
        budget.maximum_retained_bytes,
    )
    .unwrap();
    assert_eq!(checked.pending, [false, false, true, false]);
    assert_eq!(checked.unique, [0, 1, 2, 3]);
    declared.slot.token_bytes[1] = b"a".to_vec();
    assert!(matches!(
        ValidatedPrefix::validate(
            &tokenizer,
            4,
            &declared,
            &budget,
            budget.maximum_retained_bytes
        ),
        Err(GeometryProjectionUnknown::InvalidPrefix)
    ));
    let repeated = constraint(&tokenizer, RequestId::new(), &[10, 10]);
    let checked = ValidatedPrefix::validate(
        &tokenizer,
        4,
        &repeated,
        &budget,
        budget.maximum_retained_bytes,
    )
    .unwrap();
    assert_eq!(checked.unique, [0, 1, 1]);
    assert!(matches!(
        ValidatedPrefix::validate(&tokenizer, 4, &repeated, &budget, 1),
        Err(GeometryProjectionUnknown::Capacity)
    ));
}

#[tokio::test]
async fn geometry_prefix_keeps_clean_peers_clean_and_releases_future_uncertainty() {
    let (mut session, executor) = fixture(2).await;
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
    let probes = requests(&session, 2);
    let prefixes = [
        constraint(
            session.engine.inner.tokenizer.as_ref(),
            probes[0].request.id.clone(),
            &[10, 10],
        ),
        constraint(
            session.engine.inner.tokenizer.as_ref(),
            probes[1].request.id.clone(),
            &[10, 11],
        ),
    ];
    let points = [
        GeometryProjectionPoint {
            rows: 2,
            sequence_tokens: 3,
        },
        GeometryProjectionPoint {
            rows: 2,
            sequence_tokens: 4,
        },
    ];
    let report = session
        .project_geometry_with_prefixes(probes, &points, limits(), &prefixes)
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 2);
    for outcome in &report.outcomes {
        assert_eq!(outcome.unknown, None);
        assert_eq!(outcome.branches.len(), 1);
        assert_eq!(
            outcome.branches[0].host_branch,
            GeometryHostBranch::FullLogits
        );
        assert_eq!(
            outcome.branches[0].family,
            outcome.branches[0]
                .query
                .input()
                .numerical_family_key()
                .unwrap()
        );
    }
    let first = &report.outcomes[0];
    assert_eq!(
        first.prefix_condition,
        Some(GeometryPrefixCondition {
            release_generated: 2,
            first_ordinary_wave: true
        })
    );
    assert_eq!(
        first.branches[0]
            .query
            .input()
            .physical_host_rows()
            .iter()
            .filter(|r| r.pending_decoded_utf8)
            .count(),
        1
    );
    let later = &report.outcomes[1];
    assert_eq!(
        later.prefix_condition,
        Some(GeometryPrefixCondition {
            release_generated: 2,
            first_ordinary_wave: false
        })
    );
    assert_eq!(
        later.branches[0]
            .query
            .input()
            .physical_host_rows()
            .iter()
            .filter(|r| r.pending_decoded_utf8)
            .count(),
        2
    );
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_prefix_rejects_unbound_slots_and_cleans_admitted_roots() {
    let (mut session, executor) = fixture(1).await;
    let probes = requests(&session, 1);
    let prefixes = [constraint(
        session.engine.inner.tokenizer.as_ref(),
        RequestId::new(),
        &[10],
    )];
    let report = session
        .project_geometry_with_prefixes(
            probes,
            &[GeometryProjectionPoint {
                rows: 1,
                sequence_tokens: 2,
            }],
            limits(),
            &prefixes,
        )
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 1);
    assert_eq!(report.projection_attempts, 0);
    assert_eq!(
        report.outcomes[0].unknown,
        Some(GeometryProjectionUnknown::InvalidPrefix)
    );
    assert!(report.outcomes[0].branches.is_empty());
    assert_eq!(report.outcomes[0].prefix_condition, None);
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_prefix_uses_the_reports_remaining_retained_capacity() {
    let (mut session, executor) = fixture(1).await;
    let probes = requests(&session, 1);
    let prefixes = [constraint(
        session.engine.inner.tokenizer.as_ref(),
        probes[0].request.id.clone(),
        &[10],
    )];
    let mut budget = limits();
    budget.maximum_retained_bytes = std::mem::size_of::<GeometryProjectionReport>()
        + std::mem::size_of::<GeometryProjectionOutcome>()
        + std::mem::size_of::<GeometryInputReport>()
        + std::mem::size_of::<GeometryInputOutcome>()
        + std::mem::size_of::<GeometryInputTarget>()
        + super::super::PrefillStates::overhead_bytes(
            1,
            budget.maximum_route_states,
            super::super::PrefillReuse::Share,
        )
        .unwrap();
    let report = session
        .project_geometry_with_prefixes(
            probes,
            &[GeometryProjectionPoint {
                rows: 1,
                sequence_tokens: 2,
            }],
            budget,
            &prefixes,
        )
        .await
        .unwrap();
    assert_eq!(report.projection_attempts, 0);
    assert_eq!(
        report.outcomes[0].unknown,
        Some(GeometryProjectionUnknown::Capacity)
    );
    assert!(report.outcomes[0].branches.is_empty());
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}
