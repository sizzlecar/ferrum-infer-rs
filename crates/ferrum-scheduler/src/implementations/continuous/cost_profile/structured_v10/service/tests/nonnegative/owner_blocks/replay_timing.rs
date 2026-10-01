//! Diagnostic replay of an external original journal. No fixture or timing
//! threshold is a CI gate; every original protocol transition is still checked.
use super::*;
use std::io::{BufRead, BufReader};
use std::time::Instant;

#[test]
#[ignore = "requires FERRUM_TEST_SOURCE7_REPLAY_JOURNAL; original-record timing diagnostic"]
fn source7_original_journal_replay_timing() {
    let path = std::env::var_os("FERRUM_TEST_SOURCE7_REPLAY_JOURNAL")
        .expect("supply an original source7 journal outside the repository");
    let mut reader = BufReader::new(std::fs::File::open(path).unwrap());
    let mut line = Vec::new();
    reader.read_until(b'\n', &mut line).unwrap();
    let header: StructuredServiceHeaderV7 = serde_json::from_slice(&line).unwrap();
    assert_eq!(record_bytes_v7(&header).unwrap(), line);
    let maximum_file_bytes = usize::try_from(header.maximum_file_bytes).unwrap();
    let mut original_hash = Sha256::new();
    original_hash.update(&line);
    let mut original_bytes = line.len();
    // Reproduce the automatic producer's streaming byte contract. Its original
    // journal cap is independent of the offline profile loader's file cap.
    let source_budget = std::num::NonZeroU64::new(header.maximum_file_bytes).unwrap();
    let mut collector = StructuredServiceCollectorV7::new_streaming(
        header,
        CostProfileLoadLimits::default(),
        source_budget,
    )
    .unwrap();
    let mut block_offers = 0usize;
    let mut block_parse_ns = 0u128;
    let mut block_push_ns = 0u128;
    let mut maximum_offer_push_ns = 0u128;
    let mut records = 0usize;
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line).unwrap() == 0 {
            break;
        }
        assert!(line.len() <= 8 * 1024 * 1024, "original record line limit");
        original_bytes = original_bytes.checked_add(line.len()).unwrap();
        assert!(original_bytes <= maximum_file_bytes);
        original_hash.update(&line);
        let started = Instant::now();
        let record: StructuredServiceRecordV7 = serde_json::from_slice(&line).unwrap();
        assert_eq!(record_bytes_v7(&record).unwrap(), line);
        let parse_ns = started.elapsed().as_nanos();
        let started = Instant::now();
        collector.push(&record).unwrap();
        let push_ns = started.elapsed().as_nanos();
        records += 1;
        match record {
            StructuredServiceRecordV7::BlockOpen { .. } => {
                block_offers = 0;
                block_parse_ns = 0;
                block_push_ns = 0;
                maximum_offer_push_ns = 0;
            }
            StructuredServiceRecordV7::Completed { .. }
            | StructuredServiceRecordV7::OutsideDeclaredRoute { .. }
            | StructuredServiceRecordV7::NotSubmitted { .. } => {
                block_offers += 1;
                block_parse_ns += parse_ns;
                block_push_ns += push_ns;
                maximum_offer_push_ns = maximum_offer_push_ns.max(push_ns);
            }
            StructuredServiceRecordV7::BlockClose { block, .. } => {
                eprintln!(
                    "source7_replay_timing block={block} offers={block_offers} parse_ns={block_parse_ns} offers_push_ns={block_push_ns} max_offer_push_ns={maximum_offer_push_ns} close_push_ns={push_ns}"
                );
            }
            _ => {}
        }
    }
    let expected_hash: [u8; 32] = original_hash.finalize().into();
    assert_eq!(
        collector.source_receipt(),
        (original_bytes as u64, expected_hash),
        "every original record must survive strict independent replay"
    );
    assert!(records > 0);
    eprintln!("source7_replay_complete records={records} bytes={original_bytes}");
}
