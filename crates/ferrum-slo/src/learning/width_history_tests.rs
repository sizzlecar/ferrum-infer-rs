use super::*;
use crate::model::{TrainingSample, WorkFeatures};
use ferrum_execution::{execution_attempt::*, token_execution::ExecutionRange};
use ferrum_scheduler::cost_prediction::CostWork;

fn config() -> LearningConfig {
    LearningConfig {
        training_history: TrainingHistoryPolicy::WidthStratified {
            samples_per_width: 2,
        },
        training_samples: 4,
        validation_samples: 2,
        calibration_samples: 2,
        min_calibration_samples: 2,
        window_samples: 8,
        ..Default::default()
    }
}

fn observation(sequence: u64, batch: usize, at: Instant, label: f64) -> Observation {
    let work = CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(3, 1).unwrap(),
        context_tokens: 4,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    };
    Observation {
        sample: TrainingSample {
            features: WorkFeatures::extract(&vec![work; batch], 256).unwrap(),
            service_seconds: label,
        },
        sequence,
        completed_at: at,
    }
}

fn feed(
    config: &LearningConfig,
    labels: impl Fn(u64) -> f64,
    at: Instant,
) -> (VecDeque<Observation>, WidthHistory) {
    let mut window = VecDeque::new();
    let mut history = WidthHistory::new(config);
    for sequence in 1..=16 {
        let observed_at = at + Duration::from_millis(sequence);
        let item = observation(
            sequence,
            if sequence <= 2 { 1 } else { 4 },
            observed_at,
            labels(sequence),
        );
        if window.len() == config.window_samples {
            window.pop_front();
        }
        window.push_back(item);
        history.observe(config, item, observed_at);
    }
    (window, history)
}

#[test]
fn width_history_keeps_only_latest_bounded_actual_representatives() {
    let config = config();
    let at = Instant::now();
    let mut history = WidthHistory::new(&config);
    for sequence in 1..=32 {
        history.observe(
            &config,
            observation(sequence, ((sequence - 1) % 4 + 1) as usize, at, 0.01),
            at,
        );
    }
    assert_eq!(history.samples.len(), 8);
    assert_eq!(
        history
            .samples
            .iter()
            .map(|item| item.sequence)
            .collect::<Vec<_>>(),
        (25..=32).collect::<Vec<_>>()
    );
    assert!(history.samples.iter().all(|item| item.completed_at == at));
    history.observe(&config, observation(33, 1, at, f64::NAN), at);
    assert_eq!(history.samples.back().unwrap().sequence, 32);
    let recent = LearningConfig {
        training_history: TrainingHistoryPolicy::Recent,
        ..config
    };
    let mut history = WidthHistory::new(&recent);
    history.observe(&recent, observation(1, 1, at, 0.01), at);
    assert_eq!(history.samples.capacity(), 0);
}

#[test]
fn width_history_selection_preserves_suffix_and_deduplicates_training_without_labels() {
    let config = config();
    let at = Instant::now();
    let mut selections = Vec::new();
    let labelings: [fn(u64) -> f64; 2] = [|_| 0.01, |sequence| sequence as f64 * 900.0];
    for labels in labelings {
        let (window, history) = feed(&config, labels, at);
        let selected =
            fit_observations(&config, &window, &history, at + Duration::from_millis(16)).unwrap();
        let sequences: Vec<_> = selected.iter().map(|item| item.sequence).collect();
        assert_eq!(&sequences[..4], &[1, 2, 11, 12]);
        assert_eq!(&sequences[4..], &[13, 14, 15, 16]);
        assert!(sequences.windows(2).all(|pair| pair[0] < pair[1]));
        selections.push(sequences);
    }
    assert_eq!(selections[0], selections[1]);
    let (window, history) = feed(&config, |_| 0.01, at);
    let recent = LearningConfig {
        training_history: TrainingHistoryPolicy::Recent,
        ..config
    };
    let selected = fit_observations(&recent, &window, &history, at).unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|item| item.sequence)
            .collect::<Vec<_>>(),
        (9..=16).collect::<Vec<_>>()
    );
}

#[test]
fn width_history_original_age_expires_without_outcome_or_clock_renewal() {
    let config = LearningConfig {
        model_ttl_ms: 15,
        ..config()
    };
    let at = Instant::now();
    let (window, history) = feed(&config, |_| 0.01, at);
    let selected =
        fit_observations(&config, &window, &history, at + Duration::from_millis(17)).unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|item| item.sequence)
            .collect::<Vec<_>>(),
        (9..=16).collect::<Vec<_>>()
    );
    assert!(fit_observations(&config, &window, &history, at + Duration::from_millis(30)).is_err());
}

#[test]
fn width_history_config_accounts_for_retention_and_temporary_selection() {
    let recent = LearningConfig::default();
    let mut stratified = LearningConfig {
        training_history: TrainingHistoryPolicy::WidthStratified {
            samples_per_width: 2,
        },
        ..recent.clone()
    };
    recent.validate().unwrap();
    stratified.validate().unwrap();
    assert_eq!(
        stratified.width_history_capacity(),
        Some(2 * stratified.collection.max_participants)
    );
    assert!(stratified.width_training_scratch_bytes().unwrap() > 0);
    // Find the accepted Recent memory boundary, then require the extra retained
    // representatives to be charged instead of silently fitting that budget.
    let mut low = 0;
    let mut high = recent.max_retained_bytes;
    while low < high {
        let middle = low + (high - low) / 2;
        if (LearningConfig {
            max_retained_bytes: middle,
            ..recent.clone()
        })
        .validate()
        .is_ok()
        {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    stratified.max_retained_bytes = low;
    assert!(stratified.validate().is_err());
    for samples_per_width in [0, 3, usize::MAX] {
        assert!(LearningConfig {
            training_history: TrainingHistoryPolicy::WidthStratified { samples_per_width },
            ..recent.clone()
        }
        .validate()
        .is_err());
    }
}

fn contextual_observation(sequence: u64, batch: usize, context: usize, at: Instant) -> Observation {
    let mut item = observation(sequence, batch, at, 0.01);
    let work = CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(context - 1, 1).unwrap(),
        context_tokens: context,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    };
    item.sample.features = WorkFeatures::extract(&vec![work; batch], 256).unwrap();
    item
}

fn append(
    config: &LearningConfig,
    history: &mut WidthHistory,
    window: &mut VecDeque<Observation>,
    item: Observation,
) {
    if window.len() == config.window_samples {
        window.pop_front();
    }
    window.push_back(item);
    history.observe(config, item, item.completed_at);
    history.promote_prefix(config, window, item.completed_at);
}

#[test]
fn extrema_promotion_preserves_causal_prefix_before_new_suffix_extremes() {
    let config = LearningConfig {
        training_history: TrainingHistoryPolicy::WidthAndExtrema {
            samples_per_width: 2,
        },
        ..config()
    };
    let at = Instant::now();
    let mut history = WidthHistory::new(&config);
    let mut window = VecDeque::new();
    for sequence in 1..=8 {
        append(
            &config,
            &mut history,
            &mut window,
            contextual_observation(sequence, 1, if sequence <= 4 { 80 } else { 160 }, at),
        );
    }
    assert!(history
        .extrema
        .samples()
        .iter()
        .all(|record| record.sequence <= 4 && record.sample.features.values()[8] == 80.0));
    let selected = fit_observations(&config, &window, &history, at).unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>(),
        (1..=8).collect::<Vec<_>>()
    );
    append(
        &config,
        &mut history,
        &mut window,
        contextual_observation(9, 1, 160, at),
    );
    assert!(history
        .extrema
        .samples()
        .iter()
        .any(|record| record.sequence == 5 && record.sample.features.values()[8] == 160.0));
    assert!(history
        .extrema
        .samples()
        .iter()
        .all(|record| record.sequence <= 5));
}

#[test]
fn distinct_extrema_cannot_silently_exceed_the_training_budget() {
    let config = LearningConfig {
        training_history: TrainingHistoryPolicy::WidthAndExtrema {
            samples_per_width: 2,
        },
        training_samples: 2,
        validation_samples: 1,
        calibration_samples: 1,
        min_calibration_samples: 1,
        ..config()
    };
    let at = Instant::now();
    let mut history = WidthHistory::new(&config);
    let mut window = VecDeque::new();
    for (index, (width, context)) in [(1, 100), (4, 30), (2, 20), (3, 25), (3, 25)]
        .into_iter()
        .enumerate()
    {
        append(
            &config,
            &mut history,
            &mut window,
            contextual_observation(index as u64 + 1, width, context, at),
        );
    }
    assert!(matches!(
        fit_observations(&config, &window, &history, at),
        Err(ModelError::ResourceLimit(_))
    ));
}

#[test]
fn extrema_configuration_charges_retention_and_selection_without_raising_limits() {
    let base = LearningConfig {
        training_history: TrainingHistoryPolicy::WidthStratified {
            samples_per_width: 2,
        },
        ..Default::default()
    };
    let mut extrema = LearningConfig {
        training_history: TrainingHistoryPolicy::WidthAndExtrema {
            samples_per_width: 2,
        },
        ..base.clone()
    };
    base.validate().unwrap();
    extrema.validate().unwrap();
    assert!(extrema.width_training_scratch_bytes() > base.width_training_scratch_bytes());
    let mut low = 0;
    let mut high = base.max_retained_bytes;
    while low < high {
        let middle = low + (high - low) / 2;
        if (LearningConfig {
            max_retained_bytes: middle,
            ..base.clone()
        })
        .validate()
        .is_ok()
        {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    extrema.max_retained_bytes = low;
    assert!(extrema.validate().is_err());
    extrema.max_retained_bytes = low
        + base.max_domains
            * super::super::extrema_history::MAX_EXTREMA
            * std::mem::size_of::<Observation>();
    extrema.validate().unwrap();
    extrema.max_retained_bytes -= 1;
    assert!(extrema.validate().is_err());
}
