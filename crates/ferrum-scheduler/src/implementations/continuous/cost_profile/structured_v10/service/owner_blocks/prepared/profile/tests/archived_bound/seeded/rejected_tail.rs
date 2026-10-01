//! A pinned failed journal can contain the original rejected attempt before its
//! failure marker. Diagnose its accepted prefix without laundering that attempt
//! into a valid source or changing the original clock contract.
use super::*;

pub(super) fn verify_and_close_candidate(
    header: &StructuredServiceHeaderV7,
    original: &StructuredServiceCollectorV7,
    candidate: &mut StructuredServiceCollectorV7,
    rejected: &StructuredServiceRecordV7,
    error: CostProfileError,
    lines: &mut impl Iterator<Item = std::io::Result<Vec<u8>>>,
    prefix_bytes: u64,
    mut prefix: Sha256,
) -> Value {
    assert!(
        matches!(&error, CostProfileError::Metadata(reason)
        if *reason == "source7 observation clock differs"),
        "{error:?}"
    );
    let StructuredServiceRecordV7::Completed { wave } = rejected else {
        panic!("only an original completed attempt may have this negative tail");
    };
    let deadline = header
        .opening
        .monotonic_ns
        .checked_add(header.declaration.maximum_window_ns)
        .unwrap();
    let finalized = wave.host_stages.finalized_at_ns.unwrap();
    assert!(wave.issued_at_ns <= deadline && finalized > deadline);
    let audit = original.audit();
    assert_eq!(wave.ticket, audit.offered + 1);
    assert!(wave.fifo > audit.last_fifo);
    assert_eq!(candidate.audit().offered, audit.offered);
    assert_eq!(candidate.audit().last_fifo, audit.last_fifo);
    // The diagnostic file also retains the rejected raw attempt. The original
    // collector's accepted-prefix receipt does not count it; keep both facts.
    assert_eq!(
        original.source_receipt(),
        (prefix_bytes, <[u8; 32]>::from(prefix.clone().finalize()))
    );

    let failed_bytes = lines
        .next()
        .expect("original Failed immediately follows")
        .unwrap();
    let failed: StructuredServiceRecordV7 = serde_json::from_slice(&failed_bytes).unwrap();
    let StructuredServiceRecordV7::Failed {
        ticket,
        fifo,
        at_ns,
        reason,
        source_prefix_bytes,
        source_prefix_sha256,
    } = &failed
    else {
        panic!("rejected attempt lacks its original Failed");
    };
    assert_eq!((*ticket, *fifo), (wave.ticket, wave.fifo));
    assert_eq!(
        (*source_prefix_bytes, *source_prefix_sha256),
        (prefix_bytes, <[u8; 32]>::from(prefix.clone().finalize()))
    );
    assert!(*at_ns >= finalized && !reason.is_empty());
    prefix.update(&failed_bytes);
    prefix.update(b"\n");

    let footer_bytes = lines
        .next()
        .expect("original Footer immediately follows")
        .unwrap();
    let footer: StructuredServiceRecordV7 = serde_json::from_slice(&footer_bytes).unwrap();
    let StructuredServiceRecordV7::Footer {
        closing,
        offered,
        accepted_fifo_cutoff,
        incomplete_block,
        source_prefix_bytes,
        source_prefix_sha256,
    } = footer
    else {
        panic!("negative source lacks original Footer");
    };
    assert_eq!(
        (source_prefix_bytes, source_prefix_sha256),
        (
            prefix_bytes + failed_bytes.len() as u64 + 1,
            <[u8; 32]>::from(prefix.finalize())
        )
    );
    assert_eq!(
        (offered, accepted_fifo_cutoff),
        (audit.offered, audit.last_fifo)
    );
    assert!(incomplete_block && closing.monotonic_ns >= *at_ns);
    assert!(
        lines.next().is_none(),
        "no records may follow the original negative tail"
    );

    // No failed observation enters the candidate's sample set. Keep its original
    // failed position and incomplete tail; no extra close or checkpoint occurs.
    candidate
        .fail(*ticket, *fifo, *at_ns, reason.clone())
        .unwrap();
    let stopped = candidate.stop(closing).unwrap();
    assert!(matches!(
        stopped,
        StructuredServiceRecordV7::Footer {
            incomplete_block: true,
            ..
        }
    ));
    json!({"original_error":format!("{error:?}"), "call_id":wave.host_stages.call_id,
        "ticket":wave.ticket,"fifo":wave.fifo,"issued_at_ns":wave.issued_at_ns,
        "finalized_at_ns":finalized,"original_deadline_ns":deadline,
        "accepted_offered":audit.offered,"accepted_fifo_cutoff":audit.last_fifo,
        "interpretation":"whole original bytes verified; only accepted prefix replayed; original rejected attempt and incomplete failure preserved; not a valid full-source replay"})
}
