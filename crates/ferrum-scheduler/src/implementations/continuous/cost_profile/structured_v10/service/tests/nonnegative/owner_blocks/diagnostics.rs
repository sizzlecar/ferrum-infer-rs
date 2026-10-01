//! Original canonical waves reach a real Qualification failure. The cold
//! diagnostics must survive sample clearing without changing source bytes.
use super::*;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id, Record},
    Event, Metadata, Subscriber,
};

#[derive(Clone)]
struct Events {
    enabled: bool,
    rows: Arc<Mutex<Vec<BTreeMap<String, String>>>>,
}
impl Subscriber for Events {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.enabled && metadata.target() == "ferrum_scheduler::structured_owner_diagnostics"
    }
    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, event: &Event<'_>) {
        #[derive(Default)]
        struct Fields(BTreeMap<String, String>);
        impl Visit for Fields {
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                self.0.insert(field.name().into(), format!("{value:?}"));
            }
            fn record_str(&mut self, field: &Field, value: &str) {
                self.0.insert(field.name().into(), value.into());
            }
        }
        let mut fields = Fields::default();
        event.record(&mut fields);
        self.rows.lock().unwrap().push(fields.0);
    }
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

fn failed_qualification(enabled: bool) -> (Vec<u8>, Vec<BTreeMap<String, String>>) {
    let events = Events {
        enabled,
        rows: Arc::default(),
    };
    let rows = Arc::clone(&events.rows);
    let bytes = tracing::subscriber::with_default(events, || {
        let mut original = block_header();
        original.declaration.settings.static_margin_ns = 0;
        original.declaration.nonnegative_envelope.as_mut().unwrap().planning_estimator =
            crate::implementations::continuous::cost_model::structured_v2::NonNegativePlanningEstimatorV1::FittedResidualV1;
        let header = StructuredServiceHeaderV7::new(
            original.capture_identity,
            original.generation,
            original.fingerprint,
            original.producer,
            original.opening,
            original.declaration,
            original.maximum_file_bytes,
        )
        .unwrap();
        let mut bytes = record_bytes_v7(&header).unwrap();
        let mut collector =
            StructuredServiceCollectorV7::new(header, CostProfileLoadLimits::default()).unwrap();
        for block in 1..=3 {
            complete_block(&mut collector, &mut bytes, block);
        }
        append(&mut bytes, &collector.open_block(49_999, 72).unwrap());
        for ticket in 25..=32 {
            let (prepared, _, _) = old::prepared(&format!("physical-{ticket}"), 2, 1);
            let stages = old::stages(&old::header(), &prepared, ticket, 1_500);
            let record = StructuredServiceRecordV7::Completed {
                wave: StructuredServiceWaveV7::from_diagnostic(
                    ticket,
                    ticket * 2_000,
                    ticket * 3,
                    serde_json::to_value(stages).unwrap(),
                    None,
                )
                .unwrap(),
            };
            collector.push(&record).unwrap();
            append(&mut bytes, &record);
        }
        let record = collector.close_block(paired(65_901)).unwrap();
        let StructuredServiceRecordV7::BlockClose { freezes, .. } = &record else {
            unreachable!()
        };
        assert_eq!(freezes.len(), 1);
        assert_eq!(freezes[0].close.phase, StructuredPhaseV2::Qualification);
        assert_eq!(freezes[0].close.member_count, 8);
        assert_eq!(
            freezes[0].failure.as_deref(),
            Some("QualificationUnderestimate")
        );
        // The original lifecycle consumed the samples and lost its active
        // phase. The event must use the immutable freeze, not this final audit.
        let audit = collector.audit();
        assert_eq!(audit.owners[0].phase, None);
        assert_eq!(audit.owners[0].eligible, 0);
        assert_eq!(collector.qualified_children(), 0);
        append(&mut bytes, &record);
        assert_eq!(
            collector.source_receipt(),
            (bytes.len() as u64, Sha256::digest(&bytes).into())
        );
        bytes
    });
    let captured = rows.lock().unwrap().clone();
    (bytes, captured)
}

#[test]
fn original_failure_event_retains_consumed_phase_and_members_without_changing_source() {
    let (without, hidden) = failed_qualification(false);
    assert!(hidden.is_empty());
    let (with, events) = failed_qualification(true);
    assert_eq!(
        with, without,
        "diagnostics cannot alter canonical records or qualification"
    );
    let failure = events
        .iter()
        .find(|event| {
            event.get("event").map(String::as_str) == Some("structured_owner_phase_failure_v1")
        })
        .unwrap();
    assert_eq!(failure["phase"], "Qualification");
    assert_eq!(failure["member_count"], "8");
    assert_eq!(failure["reason"], "QualificationUnderestimate");
    assert_eq!(failure["first_offered"], "25");
    assert_eq!(failure["last_offered"], "32");
    assert_eq!(failure["input_context_truncated"], "false");
    assert!(failure["input_context"].contains("wall_min_ns: 1500"));
    assert!(failure["input_context"].contains("wall_max_ns: 1500"));
    let summary = events
        .iter()
        .rev()
        .find(|event| {
            event.get("event").map(String::as_str) == Some("structured_owner_block_summary_v1")
        })
        .unwrap();
    assert_eq!(summary["all_owners_failed"], "true");
    assert_eq!(summary["failed"], "1");
    assert_eq!(summary["qualified"], "0");
}
