//! Original source7 streaming replay, including incomplete/failed tails.
//! This diagnostic never appends a close, checkpoint, sample or qualification.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::*;
use std::collections::BTreeMap;

pub(super) fn spec() -> AuditSpec {
    serde_json::from_slice(
        &std::fs::read(std::env::var_os("FERRUM_ARCHIVED_BOUND_SPEC").expect("audit spec"))
            .unwrap(),
    )
    .unwrap()
}

pub(super) fn raw_history(wave: &StructuredServiceWaveV7) -> (u64, u64) {
    let (prepared, _) =
        physical::original_prepared(&wave.host_stages, wave.independent.as_ref()).unwrap();
    let rows = &prepared.exact.numeric_features.as_ref().unwrap().rows;
    (
        rows.iter().map(|row| row.sampling_history_tokens).sum(),
        rows.iter()
            .map(|row| row.sampling_history_tokens)
            .max()
            .unwrap(),
    )
}

pub(super) fn sample_ranges(
    samples: &[StructuredNumericObservationV2],
    histories: &BTreeMap<u64, (u64, u64)>,
) -> Value {
    let mut minima = vec![
        u64::MAX;
        samples
            .first()
            .map_or(0, |s| s.input.regression_axes().len())
    ];
    let mut maxima = vec![0; minima.len()];
    for sample in samples {
        for ((minimum, maximum), &x) in minima
            .iter_mut()
            .zip(&mut maxima)
            .zip(sample.input.regression_axes())
        {
            assert!(x >= 0. && x.fract() == 0.);
            *minimum = (*minimum).min(x as u64);
            *maximum = (*maximum).max(x as u64);
        }
    }
    json!({
        "members": samples.len(),
        "first_call": samples.first().map(|s|s.call_id),
        "last_call": samples.last().map(|s|s.call_id),
        "first_observed_at": samples.first().map(|s|s.observed_at_ns),
        "last_observed_at": samples.last().map(|s|s.observed_at_ns),
        "history_sum_min": samples.iter().filter_map(|s|histories.get(&s.call_id)).map(|v|v.0).min(),
        "history_sum_max": samples.iter().filter_map(|s|histories.get(&s.call_id)).map(|v|v.0).max(),
        "history_row_max": samples.iter().filter_map(|s|histories.get(&s.call_id)).map(|v|v.1).max(),
        "axis_minima": minima, "axis_maxima": maxima,
    })
}

fn target_gap(facts: Option<&OwnerInputTargetV1>, target: Option<&OwnerInputTargetV1>) -> Value {
    let (Some(facts), Some(target)) = (facts, target) else {
        return json!({"facts_present":facts.is_some(),"target_present":target.is_some()});
    };
    let facts = serde_json::to_value(facts).unwrap();
    let target = serde_json::to_value(target).unwrap();
    let seen = facts["positive"].as_array().unwrap();
    let wanted = target["positive"].as_array().unwrap();
    assert_eq!(seen.len(), wanted.len());
    let missing: Vec<_> = seen
        .iter()
        .zip(wanted)
        .enumerate()
        .filter_map(|(i, (s, t))| (t.as_bool().unwrap() && !s.as_bool().unwrap()).then_some(i))
        .collect();
    let branches: Vec<_> = facts["branches"]
        .as_array()
        .unwrap()
        .iter()
        .zip(target["branches"].as_array().unwrap())
        .enumerate()
        .filter_map(|(i, (s, t))| {
            let (s, t) = (s.as_u64().unwrap(), t.as_u64().unwrap());
            (s & t != t).then_some(json!({"index":i,"seen":s,"required":t}))
        })
        .collect();
    json!({"missing_positive_axes":missing,"missing_branches":branches,
        "branch_order":["pending","length","mask_upload","repetition"]})
}

#[test]
#[ignore = "requires original source7 via FERRUM_ARCHIVED_BOUND_SPEC"]
fn archived_source7_original_blocks_and_qualification_progress() {
    let spec = spec();
    let limits = CostProfileLoadLimits::default();
    let mut lines = BufReader::new(std::fs::File::open(&spec.actual_source).unwrap()).split(b'\n');
    let mut first = lines.next().unwrap().unwrap();
    first.push(b'\n');
    let h: StructuredServiceHeaderV7 = serde_json::from_slice(&first).unwrap();
    assert_eq!(record_bytes_v7(&h).unwrap(), first);
    let mut digest = Sha256::new();
    digest.update(&first);
    let mut collector = StructuredServiceCollectorV7::new_streaming(
        h.clone(),
        limits,
        std::num::NonZeroU64::new(h.maximum_file_bytes).unwrap(),
    )
    .unwrap();
    let mut histories = BTreeMap::new();
    let mut targets = BTreeMap::<u64, OwnerInputTargetV1>::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut checkpoints = 0usize;
    let mut last_closing = None;
    let mut original_failed = false;
    for (index, line) in lines.enumerate() {
        let mut line = line.unwrap();
        assert!(line.len() < 8 * 1024 * 1024);
        line.push(b'\n');
        digest.update(&line);
        let record: StructuredServiceRecordV7 = serde_json::from_slice(&line).unwrap();
        assert_eq!(
            record_bytes_v7(&record).unwrap(),
            line,
            "canonical record {}",
            index + 2
        );
        let value: Value = serde_json::from_slice(&line).unwrap();
        *counts
            .entry(value["kind"].as_str().unwrap().to_owned())
            .or_default() += 1;
        if let StructuredServiceRecordV7::Completed { wave } = &record {
            histories.insert(wave.host_stages.call_id, raw_history(wave));
        }
        if let StructuredServiceRecordV7::BlockClose {
            block,
            closing,
            freezes,
            ..
        } = &record
        {
            last_closing = Some(closing.clone());
            let before = collector.audit();
            for owner in &collector.owners {
                let audit = before
                    .owners
                    .iter()
                    .find(|a| a.owner_attempt_id == owner.contract.owner_attempt_id)
                    .unwrap();
                let facts = (!owner.samples.is_empty())
                    .then(|| OwnerInputTargetV1::from_samples(&owner.samples).unwrap());
                let target = targets
                    .get(&owner.contract.owner_attempt_id)
                    .or(owner.contract.input_target.as_ref());
                eprintln!(
                    "ORIGINAL_SOURCE7_OWNER {}",
                    json!({
                        "record_line":index+2,"block":block,"owner":audit,
                        "ranges":sample_ranges(&owner.samples,&histories),
                        "input_target_gap":target_gap(facts.as_ref(),target),
                        "original_closing_ns":closing.monotonic_ns,
                    })
                );
                if freezes.iter().any(|f| {
                    f.owner_attempt_id == owner.contract.owner_attempt_id
                        && f.close.phase == StructuredPhaseV2::Fit
                        && f.failure.is_none()
                }) {
                    targets.insert(owner.contract.owner_attempt_id, facts.unwrap());
                }
            }
            for freeze in freezes {
                eprintln!(
                    "ORIGINAL_SOURCE7_FREEZE {}",
                    json!({
                        "block":block,"owner_attempt_id":freeze.owner_attempt_id,
                        "close":freeze.close,"failure":freeze.failure,
                        "original_fit_certificate":freeze.nonnegative_fit_certificate,
                        "parameters":freeze.parameters_sha256,
                    })
                );
            }
        }
        if let StructuredServiceRecordV7::Checkpoint { closing, .. } = &record {
            checkpoints += 1;
            last_closing = Some(closing.clone());
        }
        original_failed |= matches!(&record, StructuredServiceRecordV7::Failed { .. });
        collector
            .push(&record)
            .unwrap_or_else(|error| panic!("original record {}: {error:?}", index + 2));
        if matches!(&record, StructuredServiceRecordV7::BlockClose { .. }) {
            eprintln!(
                "ORIGINAL_SOURCE7_BLOCK {}",
                serde_json::to_string(&collector.audit()).unwrap()
            );
        }
    }
    let actual_hash: [u8; 32] = digest.finalize().into();
    assert_eq!(actual_hash, spec.actual_source_sha256);
    assert_eq!(collector.source_receipt().1, actual_hash);
    let audit = collector.audit();
    assert!(
        audit.closed,
        "original source must actually contain its footer"
    );
    assert_eq!(audit.poisoned, original_failed);
    eprintln!(
        "ORIGINAL_SOURCE7_FINAL {}",
        json!({"original_source_sha256":actual_hash,
        "record_counts":counts,"checkpoint_count":checkpoints,
        "qualified_children":collector.qualified_children(),"audit":audit,
        "last_original_closing":last_closing,
        "note":"Only original records replayed; no fabricated complete tail or checkpoint"})
    );
}
