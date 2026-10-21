use super::*;
use crate::{
    learning::TrainingHistoryPolicy,
    model::{TrainingSample, WorkFeatures},
};
use ferrum_execution::{execution_attempt::*, token_execution::ExecutionRange};
use ferrum_scheduler::cost_prediction::CostWork;

fn config() -> LearningConfig {
    LearningConfig {
        training_history: TrainingHistoryPolicy::WidthAndExtrema {
            samples_per_width: 2,
        },
        model_ttl_ms: 100,
        ..Default::default()
    }
}

fn observation(sequence: u64, width: usize, context: usize, at: Instant) -> Observation {
    let work = CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(context - 1, 1).unwrap(),
        context_tokens: context,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    };
    Observation {
        sample: TrainingSample {
            features: WorkFeatures::extract(&vec![work; width], 256).unwrap(),
            service_seconds: 0.01,
        },
        sequence,
        completed_at: at,
    }
}

#[test]
fn extrema_retains_original_facts_with_bounded_storage_and_no_label_selection() {
    let config = config();
    let at = Instant::now();
    let mut selections = Vec::new();
    for label in [0.01, 100.0] {
        let mut history = ExtremaHistory::new(&config);
        let capacity = history.samples.capacity();
        for sequence in 1..=200 {
            let mut sample = observation(
                sequence,
                ((sequence * 7) % 16 + 1) as usize,
                ((sequence * 13) % 80 + 1) as usize,
                at,
            );
            sample.sample.service_seconds = label * sequence as f64;
            history.promote(&config, sample, at);
            assert!(history.samples.len() <= MAX_EXTREMA);
            assert_eq!(history.samples.capacity(), capacity);
            assert!(history
                .samples
                .iter()
                .all(|record| record.completed_at == at));
        }
        selections.push(
            history
                .samples
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(selections[0], selections[1]);
    let recent = ExtremaHistory::new(&LearningConfig::default());
    assert_eq!(recent.samples.capacity(), 0);
}

#[test]
fn extrema_ties_need_real_new_evidence_and_expire_on_original_clock() {
    let config = config();
    let start = Instant::now();
    let mut history = ExtremaHistory::new(&config);
    history.promote(&config, observation(1, 1, 80, start), start);
    let later = start + Duration::from_millis(10);
    history.promote(&config, observation(2, 1, 80, later), later);
    assert_eq!(history.samples.len(), 1);
    assert_eq!(history.samples[0].sequence, 2);
    assert_eq!(history.samples[0].completed_at, later);
    // Replaying a sequence or a future completion cannot renew its clock.
    history.promote(
        &config,
        observation(2, 1, 90, later),
        later + Duration::from_millis(1),
    );
    history.promote(
        &config,
        observation(3, 1, 90, later + Duration::from_millis(2)),
        later,
    );
    assert_eq!(history.samples.len(), 1);
    assert_eq!(history.samples[0].sample.features.values()[8], 80.0);
    let expired = later + Duration::from_millis(config.model_ttl_ms);
    history.promote(&config, observation(4, 1, 16, expired), expired);
    assert_eq!(history.samples.len(), 1);
    assert_eq!(history.samples[0].sequence, 4);
    assert_eq!(history.samples[0].sample.features.values()[8], 16.0);
}

#[test]
fn expired_dominant_extreme_does_not_invent_a_discarded_runner_up() {
    let config = config();
    let start = Instant::now();
    let mut history = ExtremaHistory::new(&config);
    for (sequence, elapsed, context) in [(1, 0, 100), (2, 10, 80), (3, 20, 16)] {
        let at = start + Duration::from_millis(elapsed);
        history.promote(&config, observation(sequence, 1, context, at), at);
    }
    assert!(!history
        .samples
        .iter()
        .any(|record| record.sample.features.values()[8] == 80.0));
    let at = start + Duration::from_millis(100);
    history.promote(&config, observation(4, 1, 16, at), at);
    // Context80 would still be within its own TTL, but the bounded reservoir
    // does not contain it. Expiry cannot reconstruct evidence it discarded.
    assert!(history
        .samples
        .iter()
        .all(|record| record.sample.features.values()[8] == 16.0));
}
