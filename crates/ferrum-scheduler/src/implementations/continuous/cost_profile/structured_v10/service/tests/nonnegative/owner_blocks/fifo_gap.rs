use super::*;
use ferrum_interfaces::execution_cost::ActualWaveGraphState;
fn advancing(
    ticket: u64,
    call: u64,
    generation: u64,
    generated: u64,
    kv: u32,
) -> StructuredServiceRecordV7 {
    let (p, _, _) = old::prepared_route_policy_bounds(
        "retained-across-blocks",
        generation,
        generated,
        None,
        ActualWaveGraphState::Disabled,
        None,
        kv,
        5,
    );
    let stages = old::stages(&old::header(), &p, call, 1_000);
    StructuredServiceRecordV7::Completed {
        wave: StructuredServiceWaveV7::from_diagnostic(
            ticket,
            call * 2_000,
            call * 3,
            serde_json::to_value(stages).unwrap(),
            None,
        )
        .unwrap(),
    }
}
fn discovery_with_retained_request() -> StructuredServiceCollectorV7 {
    discovery_with_header(block_header())
}
fn discovery_with_header(header: StructuredServiceHeaderV7) -> StructuredServiceCollectorV7 {
    let mut c =
        StructuredServiceCollectorV7::new(header, CostProfileLoadLimits::default()).unwrap();
    c.open_block(1_999, 0).unwrap();
    for ticket in 1..8 {
        c.push(&block_wave(ticket)).unwrap();
    }
    c.push(&advancing(8, 8, 2, 1, 64)).unwrap();
    c.close_block(paired(17_101)).unwrap();
    c
}

fn first_offer_header() -> StructuredServiceHeaderV7 {
    use crate::implementations::continuous::cost_model::structured_v2::OwnerOpeningFrontierPolicyV1;
    let mut h = block_header();
    h.declaration.schedule.opening_frontier = Some(OwnerOpeningFrontierPolicyV1::FirstOfferFifoV1);
    StructuredServiceHeaderV7::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap()
}

#[test]
fn source7_first_offer_fifo_policy_is_explicit_and_signature_bound() {
    let original = block_header();
    assert!(!serde_json::to_string(&original.declaration.schedule)
        .unwrap()
        .contains("opening_frontier"));
    let declared = first_offer_header();
    assert_ne!(original.declaration_sha256, declared.declaration_sha256);
    assert_ne!(original.protocol, declared.protocol);
    let mut tampered = original;
    tampered.declaration.schedule.opening_frontier = declared.declaration.schedule.opening_frontier;
    assert!(StructuredServiceCollectorV7::new(tampered, CostProfileLoadLimits::default()).is_err());
}

#[test]
fn source7_first_offer_fifo_gap_recaptures_only_original_block_opening_frontier() {
    // BlockOpen occurs before call 9 is enqueued. That genuinely unoffered
    // call has FIFO 27; the first original offer arrives with FIFO 30.
    let mut c = discovery_with_header(first_offer_header());
    c.open_block(19_999, 24).unwrap();
    c.push(&advancing(9, 10, 4, 3, 66)).unwrap();
    assert_eq!(c.offered(), 9);
    assert!(
        c.push(&advancing(10, 11, 6, 3, 68)).is_err(),
        "a later FIFO gap must not excuse an in-block generation skip"
    );

    let mut contiguous = discovery_with_header(first_offer_header());
    contiguous.open_block(17_101, 24).unwrap();
    let mut skipped = advancing(9, 9, 4, 3, 66);
    let StructuredServiceRecordV7::Completed { wave } = &mut skipped else {
        unreachable!()
    };
    wave.fifo = 25;
    assert!(
        contiguous.push(&skipped).is_err(),
        "no gap preserves the old request frontier"
    );
    let mut missing_ticket = discovery_with_header(first_offer_header());
    missing_ticket.open_block(19_999, 24).unwrap();
    assert!(
        missing_ticket.push(&advancing(10, 10, 4, 3, 66)).is_err(),
        "FIFO never replaces the original complete ticket population"
    );
}
#[test]
fn source7_explicit_processed_fifo_gap_allows_current_frontier_but_no_gap_or_inblock_skip_fails() {
    // One real canonical decode occurred in the explicitly unoffered interval.
    // It is deliberately not fabricated as an original source ticket.
    let unoffered = advancing(9, 9, 3, 2, 65);
    let StructuredServiceRecordV7::Completed { wave: unoffered } = unoffered else {
        unreachable!()
    };
    assert_eq!(unoffered.issued_at_ns, 18_000);
    let mut without_gap = discovery_with_retained_request();
    without_gap.open_block(19_999, 24).unwrap();
    assert!(without_gap.push(&advancing(9, 10, 4, 3, 66)).is_err());
    let mut with_gap = discovery_with_retained_request();
    with_gap.open_block(19_999, 27).unwrap();
    with_gap.push(&advancing(9, 10, 4, 3, 66)).unwrap();
    assert_eq!(with_gap.offered(), 9);
    // A gap inside the active block is not an opening-frontier declaration.
    assert!(with_gap.push(&advancing(10, 11, 6, 3, 68)).is_err());
}
#[test]
fn source7_fifo_gap_never_renews_original_epoch_deadline() {
    let h = block_header();
    let deadline = h.opening.monotonic_ns + h.declaration.maximum_window_ns;
    let mut c = discovery_with_retained_request();
    assert!(c.open_block(deadline + 1, 27).is_err());
    assert_eq!(c.offered(), 8);
}
