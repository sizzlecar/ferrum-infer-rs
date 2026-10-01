//! The actual model registry, mask ledger and Metal lane share completion and
//! forecast evidence without allocating or submitting a model wave.
use super::*;
use ferrum_interfaces::model_executor::{
    ExecutorCompletionPlanningCapture, ExecutorPlanningCapture,
};
use ferrum_interfaces::vnext::ResourcePlanningBudget;

struct Budget(bool);
impl ResourcePlanningBudget for Budget {
    fn has_budget(&mut self) -> bool {
        self.0
    }
}
struct Observer {
    completion: Budget,
    forecast: Budget,
    ready: usize,
    finished: usize,
}
impl ExecutorPlanningCapture for Observer {
    fn resource_budget(&mut self) -> &mut dyn ResourcePlanningBudget {
        &mut self.completion
    }
    fn completion_ready(&mut self, resources: &ResourcePlanningView) -> bool {
        assert_eq!(self.ready, 0);
        assert_eq!(resources.participants().len(), 3);
        assert_eq!(resources.limits().maximum_projected_waves, 1);
        self.ready += 1;
        true
    }
    fn forecast_budget(&mut self) -> &mut dyn ResourcePlanningBudget {
        &mut self.forecast
    }
    fn forecast_finished(&mut self) {
        assert_eq!(self.ready, 1);
        assert_eq!(self.finished, 0);
        self.finished += 1;
    }
}

#[tokio::test]
async fn actual_model_completion_capture_preserves_resources_when_optional_work_fails() {
    let fixture = Fixture::grouped().await;
    let inputs: Vec<_> = (0..3)
        .map(|_| {
            PlanRuntimePrefillInput::new(
                RequestId::new(),
                vec![TokenId::new(1); 4],
                512,
                PrefillChunk::new(0, 4, 4).unwrap(),
            )
            .unwrap()
        })
        .collect();
    for input in &inputs {
        fixture.admit(input);
    }
    fixture.warm(&inputs, &[]);
    let before = fixture.target_resource_evidence(&inputs, &[]);
    let submissions = fixture.submissions();
    let masks = mask_counts(&fixture);
    let requests: Vec<_> = inputs
        .iter()
        .map(|input| ExecutorResourcePlanningRequest {
            request_id: &input.request_id,
            cache_id: None,
        })
        .collect();
    let completion_limits = ResourcePlanningLimits {
        maximum_projected_waves: 1,
        ..Default::default()
    };
    let forecast_limits = ResourcePlanningLimits {
        maximum_projected_waves: 3,
        ..completion_limits
    };
    for optional_available in [true, false] {
        let mut observer = Observer {
            completion: Budget(true),
            forecast: Budget(optional_available),
            ready: 0,
            finished: 0,
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        let ExecutorCompletionPlanningCapture {
            resources,
            forecast,
        } = loop {
            match fixture.executor.execution_completion_planning_capture(
                &requests,
                completion_limits,
                forecast_limits,
                &mut observer,
            ) {
                ResourcePlanningAvailability::Known(value) => break value,
                ResourcePlanningAvailability::Unknown(
                    ResourcePlanningUnknown::ReadUnavailable(_),
                ) if Instant::now() < deadline => {
                    assert_eq!(observer.ready, 0);
                    std::thread::yield_now();
                }
                ResourcePlanningAvailability::Unknown(reason) => {
                    panic!("actual model completion capture: {reason:?}")
                }
            }
        };
        assert_eq!(observer.ready, 1);
        assert_eq!(observer.finished, 1);
        assert_eq!(resources.participants(), before.as_slice());
        match forecast {
            Some(ExecutionCostRouteAvailability::Known(view)) if optional_available => {
                assert_eq!(view.resource_view().limits(), forecast_limits);
                assert_eq!(
                    view.resource_view().participants(),
                    resources.participants()
                );
            }
            Some(ExecutionCostRouteAvailability::Unknown(
                ExecutionCostRouteUnknown::BudgetExhausted,
            )) if !optional_available => {}
            other => panic!("actual model optional forecast: {other:?}"),
        }
        assert_eq!(fixture.submissions(), submissions);
        assert_eq!(mask_counts(&fixture), masks);
        assert_eq!(fixture.target_resource_evidence(&inputs, &[]), before);
    }
    for input in &inputs {
        assert!(fixture.executor.cancel_prefill_admission(&input.request_id));
    }
}
