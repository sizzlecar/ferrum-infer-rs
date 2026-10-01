//! Capture the original failed prediction, without changing the numerical gate.
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

#[test]
fn structured_v2_qualification_failure_records_first_original_frozen_bound_without_changing_result()
{
    let mut heldout = observations(StructuredPhaseV2::Qualification);
    let baseline = calibrated_errors(-100, 100, 200)
        .unwrap()
        .qualify(&heldout, 490)
        .unwrap()
        .predict_query(
            &fp(),
            &StructuredQueryV2::exact(heldout[3].input.clone()),
            500,
        )
        .unwrap();
    assert_eq!(baseline.planning_ns, 1320);
    assert_eq!(baseline.learned_span_margin_ns, 200);
    heldout[3].wall_ns = 1321;
    heldout[7].wall_ns = 9000;
    for enabled in [false, true] {
        let events = Events {
            enabled,
            rows: Arc::default(),
        };
        let rows = events.rows.clone();
        let result = tracing::subscriber::with_default(events, || {
            calibrated_errors(-100, 100, 200)
                .unwrap()
                .qualify(&heldout, 490)
        });
        assert!(matches!(
            result,
            Err(StructuredUnknown::QualificationUnderestimate)
        ));
        let rows = rows.lock().unwrap();
        if !enabled {
            assert!(rows.is_empty());
            continue;
        }
        assert_eq!(
            rows.len(),
            1,
            "the original first failure ends qualification"
        );
        let row = &rows[0];
        assert_eq!(row["event"], "structured_qualification_underestimate_v1");
        assert_eq!(row["call_id"], heldout[3].call_id.to_string());
        assert_eq!(
            row["member_ordinal"],
            heldout[3].membership.member_ordinal.to_string()
        );
        assert_eq!(
            row["offered_ordinal"],
            heldout[3].membership.offered_ordinal.to_string()
        );
        assert_eq!(row["accepted_fifo"], heldout[3].ordinal.to_string());
        for (field, expected) in [
            ("wall_ns", 1321),
            ("fitted_upper_ns", baseline.fitted_upper_ns),
            ("fit_error_floor_ns", baseline.fit_error_floor_ns),
            ("residual_ns", baseline.residual_ns),
            ("effective_residual_ns", baseline.effective_residual_ns),
            ("static_margin_ns", 20),
            ("learned_span_margin_ns", 200),
            ("planning_ns", 1320),
            ("excess_ns", 1),
        ] {
            assert_eq!(row[field], expected.to_string(), "{field}");
        }
    }
}
