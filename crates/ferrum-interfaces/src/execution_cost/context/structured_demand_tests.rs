use super::*;
use ferrum_types::{SloStructuredActualCapturePolicy as P, SloStructuredCostCapture as C};

#[test]
fn actual_sample_demand_is_explicit_and_never_creates_backend_capability() {
    for call in [false, true] {
        for artifact in [false, true] {
            let legacy = StructuredCostSampleDemand::for_call(P::LegacyEveryWave, call, artifact);
            assert_eq!(legacy, StructuredCostSampleDemand::RuntimePolicy);
            assert!(legacy.enabled(C::HostSettledV1));
            assert!(!legacy.enabled(C::Disabled));
            let demand = StructuredCostSampleDemand::for_call(P::ConsumerDrivenV1, call, artifact);
            assert_eq!(demand.enabled(C::HostSettledV1), call || artifact);
            assert!(!demand.enabled(C::Disabled));
        }
    }
}
