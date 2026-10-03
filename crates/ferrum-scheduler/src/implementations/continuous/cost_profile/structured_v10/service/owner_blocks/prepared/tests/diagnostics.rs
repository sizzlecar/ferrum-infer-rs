//! Failure diagnostics observe a real source8 replay without changing its result.
use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::prepared;
use ferrum_interfaces::execution_cost::ActualWaveGraphState;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id, Record},
    Event, Metadata, Subscriber,
};

type Fields = BTreeMap<String, String>;
#[derive(Clone)]
struct Events {
    enabled: bool,
    rows: Arc<Mutex<Vec<Fields>>>,
}
impl Subscriber for Events {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.enabled
            && metadata.target() == "ferrum_scheduler::structured_owner_diagnostics"
            && *metadata.level() == tracing::Level::WARN
    }
    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, event: &Event<'_>) {
        #[derive(Default)]
        struct Values(Fields);
        impl Visit for Values {
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                self.0.insert(field.name().into(), format!("{value:?}"));
            }
            fn record_str(&mut self, field: &Field, value: &str) {
                self.0.insert(field.name().into(), value.into());
            }
        }
        let mut values = Values::default();
        event.record(&mut values);
        self.rows.lock().unwrap().push(values.0);
    }
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

#[test]
fn source8_replay_rejection_reports_typed_site_and_original_position_without_changing_error() {
    let (bytes, _, _) = numerical_family::collected();
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let original: StructuredPreparedOwnerBlockHeaderV8 =
        serde_json::from_slice(lines.next().unwrap()).unwrap();
    let mut prefix = Vec::new();
    let failed = loop {
        let record: StructuredPreparedOwnerBlockRecordV8 =
            serde_json::from_slice(lines.next().unwrap()).unwrap();
        if matches!(
            record,
            StructuredPreparedOwnerBlockRecordV8::Population(
                StructuredServiceRecordV7::Completed { .. }
            )
        ) {
            break record;
        }
        prefix.push(record);
    };
    let StructuredPreparedOwnerBlockRecordV8::Population(StructuredServiceRecordV7::Completed {
        wave,
    }) = &failed
    else {
        unreachable!()
    };
    let record_bytes = record_bytes_v7(&failed).unwrap();
    let mut changed = original.clone();
    let envelope = changed
        .declaration
        .population
        .nonnegative_envelope
        .as_mut()
        .unwrap();
    // A different original canonical roster creates a valid fixed universe.
    // This controlled failure exercises the diagnostic site, not a claim about
    // which numerical error occurred in a separate hardware capture.
    let (p, offered, _, _) = old::prepared_batch_algorithms_and_future(
        &["other-declaration"],
        2,
        1,
        None,
        ActualWaveGraphState::Disabled,
        Some([44; 32]),
        64,
        &[4],
        None,
        &["fixture.diagnostic.other"],
    );
    let input =
        prepared::project_service_actual_with_domain(&p, &offered, &envelope.workload_domain)
            .unwrap();
    envelope.algorithm_universe =
        Some(DeclaredAlgorithmUniverseV1::from_inputs([&input], 4096).unwrap());
    let changed = StructuredPreparedOwnerBlockHeaderV8::new(
        changed.capture_identity,
        changed.generation,
        changed.fingerprint,
        changed.producer,
        changed.opening,
        changed.declaration,
        changed.maximum_file_bytes,
    )
    .unwrap();
    let run = |header, enabled, reject| {
        let events = Events {
            enabled,
            rows: Arc::default(),
        };
        let rows = Arc::clone(&events.rows);
        let result = tracing::subscriber::with_default(events, || {
            let mut c = StructuredPreparedOwnerBlockCollectorV8::new(
                header,
                CostProfileLoadLimits::default(),
            )
            .unwrap();
            for record in &prefix {
                c.push(record).unwrap();
            }
            assert!(
                rows.lock().unwrap().is_empty(),
                "successful prefix emits no diagnostics"
            );
            let receipt = c.source_receipt();
            let result = c.push(&failed);
            if reject {
                assert!(matches!(
                    result,
                    Err(CostProfileError::Metadata(
                        "invalid structured numerical replay"
                    ))
                ));
                assert!(c.audit().poisoned);
                assert_eq!(
                    c.source_receipt(),
                    receipt,
                    "failed record stays outside the journal"
                );
            } else {
                result.unwrap();
                assert!(!c.audit().poisoned);
            }
            (record_bytes_v7(&c.audit()).unwrap(), c.source_receipt())
        });
        let captured = rows.lock().unwrap().clone();
        (result, captured)
    };
    let (_, successful) = run(original, true, false);
    assert!(
        successful.is_empty(),
        "successful physical replay emits no diagnostics"
    );
    let (without, hidden) = run(changed.clone(), false, true);
    let (with, events) = run(changed, true, true);
    assert!(hidden.is_empty());
    assert_eq!(
        with, without,
        "logging cannot change replay accounting or poisoning"
    );
    assert_eq!(record_bytes_v7(&failed).unwrap(), record_bytes);
    let numeric = events
        .iter()
        .find(|e| {
            e.get("event").map(String::as_str) == Some("structured_numerical_replay_rejected_v1")
        })
        .unwrap();
    assert_eq!(numeric["site"], "PhysicalEnvelopeProjection");
    assert_eq!(numeric["reason"], "WrongDomain");
    let rejection = events
        .iter()
        .find(|e| {
            e.get("event").map(String::as_str) == Some("structured_source8_record_rejected_v1")
        })
        .unwrap();
    assert_eq!(rejection["record_kind"], "completed");
    assert_eq!(rejection["active_phase_index"], "Some(0)");
    assert_eq!(rejection["active_cohort"], "Some(0)");
    assert_eq!(rejection["ticket"], format!("Some({})", wave.ticket));
    assert_eq!(rejection["fifo"], format!("Some({})", wave.fifo));
    assert_eq!(
        rejection["call"],
        format!("Some({})", wave.host_stages.call_id)
    );
    assert_eq!(rejection["universe_policy"], "None");
    assert_eq!(rejection["declared_universe"], "true");
    assert_eq!(rejection["discovered_universe"], "false");
    assert!(events
        .iter()
        .all(|e| !e.contains_key("record") && !e.contains_key("input")));
}
