use super::*;
use ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1;
use ferrum_types::SloStructuredCostCapture;

#[test]
fn cuda_selected_embedding_dense_chunks_and_fixed_abi_are_complete() {
    let plan = dense_plan(32, 257, 3).unwrap();
    assert_eq!(plan.parameters, [3, 257, 32]);
    assert_eq!(plan.config.grid_dim, (2, 3, 1));
    let work = selected_dense(
        [(32, 257, 65536)],
        65536,
        SloStructuredCostCapture::HostSettledV1,
    )
    .unwrap();
    work.validate_command(65536, 2, 0).unwrap();
    work.algorithm_work()
        .unwrap()
        .unwrap()
        .validate_command(&work)
        .unwrap();
    let original =
        selected_dense([(32, 257, 3)], 3, SloStructuredCostCapture::HostSettledV1).unwrap();
    let template = SelectedReplayAlgorithmTemplateV1::from_selected(&original, 3, 1, 0).unwrap();
    let changed =
        selected_dense([(33, 257, 3)], 3, SloStructuredCostCapture::HostSettledV1).unwrap();
    assert!(
        template.validate_binding(&changed).is_err(),
        "same launch grid cannot change captured vocabulary bound"
    );
    assert!(dense_plan(32, 0, 1).is_err());
    assert!(dense_plan(32, i32::MAX as u64 + 1, 1).is_err());
    assert!(dense_plan(32, 256, 65536).is_err());
}

#[test]
fn cuda_selected_embedding_dense_disabled_is_lazy_and_partial_population_is_unknown() {
    let lazy = std::iter::from_fn(|| -> Option<(u64, u64, u64)> {
        panic!("Off enumerated embedding rows")
    });
    assert!(selected_dense(lazy, 1, SloStructuredCostCapture::Disabled).is_none());
    for leaves in [vec![], vec![(32, 256, 0)], vec![(32, 256, 2)]] {
        assert!(selected_dense(leaves, 1, SloStructuredCostCapture::HostSettledV1).is_none());
    }
}
