//! Scaled original-format parameters verify the public preparation boundary.
//! No CUDA execution or released-model image-quality claim follows.
use super::*;
#[path = "tests/family_preparation.rs"]
mod family_preparation;
use crate::p_h3_image::dit_forward::tests::{shared_fixture, transformer};
fn limits() -> PH3ImageDenoiserLimits {
    PH3ImageDenoiserLimits {
        max_metadata_bytes: 64 << 20,
        max_plan_nodes: 4096,
        max_stored_bytes: 1 << 30,
    }
}
fn request(adapter: &PH3ImageDenoiserAdapter, name: &str) -> PH3ImageDenoiserRequest {
    let tags = [PH3ImageModality::Text; 3];
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let reference = Some(PH3ImageReferenceImage::new(2, 4).unwrap());
    let layout =
        PH3ImageRequestLayout::new(adapter.transformer(), &tags, target, reference.as_slice())
            .unwrap();
    let mut times = vec![0; layout.rows()];
    times[layout.reference_rows()].fill(2);
    adapter
        .request(
            RequestIdentity::new(name).unwrap(),
            &tags,
            target,
            reference,
            &times,
        )
        .unwrap()
}
#[test]
fn public_adapter_prepares_real_mixed_source_for_install_refine_and_steady() {
    let files = shared_fixture::files(&shared_fixture::config());
    let source = Arc::new(files.source);
    let adapter = PH3ImageDenoiserAdapter::new(transformer(), source.clone(), limits()).unwrap();
    let mut fingerprints = Vec::new();
    for phase in [
        PH3ImageDenoiserPhase::InstallFullWeights,
        PH3ImageDenoiserPhase::RefineFreshText,
        PH3ImageDenoiserPhase::DenoiseRefinedText,
    ] {
        let prepared = adapter
            .prepare(
                phase,
                3,
                vec![
                    request(&adapter, "request.public.a"),
                    request(&adapter, "request.public.b"),
                ],
            )
            .unwrap();
        assert_eq!(prepared.phase(), phase);
        assert_eq!(
            prepared
                .request_ids()
                .iter()
                .map(RequestIdentity::as_str)
                .collect::<Vec<_>>(),
            ["request.public.a", "request.public.b"]
        );
        assert!(std::ptr::eq(prepared.source(), source.as_ref()));
        fingerprints.push(
            prepared
                .family()
                .stored_weight_schema()
                .unwrap()
                .fingerprint()
                .unwrap(),
        );
        assert_eq!(
            adapter.stored_bytes(),
            prepared
                .family()
                .stored_weight_schema()
                .unwrap()
                .components
                .iter()
                .map(|c| c.physical_bytes().unwrap())
                .sum::<u64>()
        );
        let program = prepared.family().program();
        let expected = program
            .inputs()
            .iter()
            .chain(
                program
                    .blocks()
                    .iter()
                    .flat_map(|b| &b.nodes)
                    .flat_map(|n| &n.outputs),
            )
            .collect::<BTreeSet<_>>();
        assert_eq!(
            expected,
            prepared.compile_options().tensor_specs().keys().collect()
        );
        match phase {
            PH3ImageDenoiserPhase::InstallFullWeights => {
                assert_eq!(program.outputs().len(), 3);
                assert!(prepared.velocity_layout().is_err());
            }
            PH3ImageDenoiserPhase::RefineFreshText => {
                assert_eq!(program.inputs().len(), 1);
                assert_eq!(program.outputs().len(), 1);
                assert!(prepared.velocity_layout().is_err());
            }
            PH3ImageDenoiserPhase::DenoiseRefinedText => {
                assert_eq!(program.outputs().len(), 2);
                prepared.velocity_layout().unwrap();
            }
        }
    }
    assert!(fingerprints.windows(2).all(|p| p[0] == p[1]));
    assert!(adapter
        .prepare(
            PH3ImageDenoiserPhase::DenoiseRefinedText,
            3,
            vec![
                request(&adapter, "request.duplicate"),
                request(&adapter, "request.duplicate")
            ]
        )
        .is_err());
}
#[test]
fn public_adapter_rejects_wrong_architecture_and_actual_source_capacity() {
    let files = shared_fixture::files(&shared_fixture::config());
    let source = Arc::new(files.source);
    let wrong = crate::p_h3_image::test_fixture::tiny_transformer(16, 2);
    assert!(PH3ImageDenoiserAdapter::new(wrong, source.clone(), limits()).is_err());
    assert!(PH3ImageDenoiserAdapter::new(
        transformer(),
        source,
        PH3ImageDenoiserLimits {
            max_stored_bytes: 1,
            ..limits()
        }
    )
    .is_err());
}

#[test]
fn request_row_runs_borrow_original_generation_and_edit_routing_in_every_phase() {
    let files = shared_fixture::files(&shared_fixture::config());
    let adapter =
        PH3ImageDenoiserAdapter::new(transformer(), Arc::new(files.source), limits()).unwrap();
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let reference = PH3ImageReferenceImage::new(2, 4).unwrap();
    for phase in [
        PH3ImageDenoiserPhase::InstallFullWeights,
        PH3ImageDenoiserPhase::RefineFreshText,
        PH3ImageDenoiserPhase::DenoiseRefinedText,
    ] {
        let mut requests = Vec::new();
        let mut expected = Vec::new();
        for (name, tags, reference, image_time, audio_time, reference_time) in [
            (
                "request.runs.generate",
                vec![PH3ImageModality::Text; 3],
                None,
                0,
                1,
                0,
            ),
            (
                "request.runs.edit",
                vec![
                    PH3ImageModality::Text,
                    PH3ImageModality::Image,
                    PH3ImageModality::Text,
                ],
                Some(reference),
                2,
                3,
                4,
            ),
        ] {
            let layout = PH3ImageRequestLayout::new(
                adapter.transformer(),
                &tags,
                target,
                reference.as_slice(),
            )
            .unwrap();
            let mut times = vec![image_time; layout.rows()];
            times[layout.reference_rows()].fill(reference_time);
            times[layout.audio_rows()].fill(audio_time);
            let modalities = (0..layout.rows())
                .map(|row| layout.modality(row).unwrap())
                .collect::<Vec<_>>();
            requests.push(
                adapter
                    .request(
                        RequestIdentity::new(name).unwrap(),
                        &tags,
                        target,
                        reference,
                        &times,
                    )
                    .unwrap(),
            );
            expected.push((times, modalities));
        }
        let prepared = adapter.prepare(phase, 5, requests).unwrap();
        let mut observed = prepared.request_row_runs();
        let mut originals = prepared.config.request_row_runs();
        assert_eq!(observed.len(), prepared.request_ids().len());
        for (index, (times, modalities)) in expected.into_iter().enumerate() {
            assert_eq!(observed.len(), 2 - index);
            let runs = observed.next().unwrap();
            let original = originals.next().unwrap();
            assert!(
                std::ptr::eq(runs, original),
                "the public adapter must borrow its config's exact slice"
            );
            assert_eq!(
                runs.iter().map(|run| run.length()).sum::<u64>(),
                times.len() as u64
            );
            let expand_times = runs
                .iter()
                .flat_map(|run| std::iter::repeat_n(run.time_index(), run.length() as usize));
            assert!(expand_times.eq(times));
            let expand_modalities = runs
                .iter()
                .flat_map(|run| std::iter::repeat_n(run.modality(), run.length() as usize));
            assert!(expand_modalities.eq(modalities));
        }
        assert_eq!(observed.len(), 0);
        assert!(observed.next().is_none());
        assert!(originals.next().is_none());
    }
}
