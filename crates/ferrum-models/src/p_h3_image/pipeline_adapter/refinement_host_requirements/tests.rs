use super::*;
use crate::p_h3_image::dit_forward::tests::{shared_fixture, transformer};
use crate::p_h3_image::PH3ImageDenoiserAttentionPolicy;

fn request(adapter: &PH3ImageDenoiserAdapter, name: &str, rows: usize) -> PH3ImageDenoiserRequest {
    let tags = vec![PH3ImageModality::Text; rows];
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let layout = PH3ImageRequestLayout::new(adapter.transformer(), &tags, target, &[]).unwrap();
    adapter
        .request(
            RequestIdentity::new(name).unwrap(),
            &tags,
            target,
            None,
            &vec![0; layout.rows()],
        )
        .unwrap()
}
#[test]
fn public_query_tracks_stable_ids_and_cache_shapes_through_reorder_and_removal() {
    query_tracks_stable_ids_and_cache_shapes(PH3ImageDenoiserAttentionPolicy::Sequential);
}

#[test]
fn public_query_tracks_stable_ids_and_cache_shapes_through_reorder_and_removal_with_cudnn_sdpa() {
    query_tracks_stable_ids_and_cache_shapes(
        PH3ImageDenoiserAttentionPolicy::CudnnSdpaExperimentalV1,
    );
}

fn query_tracks_stable_ids_and_cache_shapes(policy: PH3ImageDenoiserAttentionPolicy) {
    let files = shared_fixture::files(&shared_fixture::config());
    let mut adapter = PH3ImageDenoiserAdapter::new(
        transformer(),
        Arc::new(files.source),
        PH3ImageDenoiserLimits {
            max_metadata_bytes: 64 << 20,
            max_plan_nodes: 4096,
            max_stored_bytes: 1 << 30,
        },
    )
    .unwrap();
    adapter.set_attention_policy(policy).unwrap();
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        4096,
    )
    .unwrap();
    let prepare = |phase, members: &[(&str, usize)]| {
        adapter
            .prepare(
                phase,
                1,
                members
                    .iter()
                    .map(|&(name, rows)| request(&adapter, name, rows))
                    .collect(),
            )
            .unwrap()
    };
    let prepared = prepare(
        PH3ImageDenoiserPhase::RefineFreshText,
        &[("request.host.a", 2), ("request.host.b", 5)],
    );
    let report = prepared.refinement_host_requirements(&host).unwrap();
    assert_eq!(
        report.work_input.dimensions(),
        &[7, adapter.transformer().text_dim()]
    );
    assert_eq!(
        report.work_output.dimensions(),
        &[7, adapter.transformer().hidden_size()]
    );
    for (entry, (name, rows)) in report
        .persistent_cache_by_request
        .iter()
        .zip([("request.host.a", 2), ("request.host.b", 5)])
    {
        assert_eq!(entry.request.as_str(), name);
        assert_eq!(
            entry.tensor.dimensions(),
            &[rows, adapter.transformer().hidden_size()]
        );
        assert_eq!(
            entry.input_bytes,
            host.backing_bytes(entry.tensor.minimum_storage_bytes().unwrap() as usize)
                .unwrap()
        );
    }
    assert_eq!(
        report
            .work_scope_claims()
            .iter()
            .map(|(_, bytes)| bytes)
            .sum::<usize>(),
        report.work_scope_bytes
    );
    let reordered = prepare(
        PH3ImageDenoiserPhase::RefineFreshText,
        &[("request.host.b", 5), ("request.host.a", 2)],
    )
    .refinement_host_requirements(&host)
    .unwrap();
    assert_eq!(
        reordered.persistent_cache_by_request,
        vec![
            report.persistent_cache_by_request[1].clone(),
            report.persistent_cache_by_request[0].clone()
        ]
    );
    let removed = prepare(
        PH3ImageDenoiserPhase::RefineFreshText,
        &[("request.host.b", 5)],
    )
    .refinement_host_requirements(&host)
    .unwrap();
    assert_eq!(
        removed.persistent_cache_by_request,
        vec![report.persistent_cache_by_request[1].clone()]
    );
    for phase in [
        PH3ImageDenoiserPhase::InstallFullWeights,
        PH3ImageDenoiserPhase::DenoiseRefinedText,
    ] {
        assert!(prepare(phase, &[("request.host.b", 5)])
            .refinement_host_requirements(&host)
            .is_err());
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
