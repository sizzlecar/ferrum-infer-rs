//! Sample-specific CUDA qualification of the production step helper. A member
//! joins only at a completed fence; fresh text uses RefineOnly, and all old
//! samples/caches/schedules survive membership changes with their own scope.
//! The caller supplies these fence transitions; run/serve worker cancellation,
//! full TE/VAE image quality, throughput, SLOs and global VRAM are separate
//! evidence. No output image or successful service response is synthesized.
use super::super::*;
use super::cuda_fixture::*;
use crate::image_generation::{
    p_h3::editing::ReferenceState, ImageCudaAttentionPolicy, ImageWeightOwnerPin,
};
use ferrum_models::p_h3_image::{PH3ImageDenoiserPhase, PH3ImageRefinementDestination};
use serde_json::json;

struct Lane {
    geometry: RequestGeometry,
    image: ManagedHostTensor,
    audio: ManagedHostTensor,
    sampling: PH3ImageManagedSampling,
    cache: PH3ImageRefinedTextCache,
    reference: Option<ReferenceState>,
    _scope: HostMemoryScope,
    pin: ImageWeightOwnerPin<Runtime>,
}
impl Lane {
    fn new(f: &mut Fixture, name: &str, seed: usize, with_reference: bool, sigmas: &[f32]) -> Self {
        let geometry = f.geometry(name, with_reference);
        let batch = f.batch(&[&geometry]);
        let compiled = f.compile(
            batch
                .prepare(&f.adapter, PH3ImageDenoiserPhase::RefineFreshText)
                .unwrap(),
        );
        let host = f.execution.host_account();
        let requirements = compiled.recipe.refinement_host_requirements(host).unwrap();
        let backing = |n| host.backing_bytes(n).unwrap();
        let image_bytes = backing(geometry.image_sample.minimum_storage_bytes().unwrap() as usize);
        let audio_bytes = backing(geometry.audio_sample.minimum_storage_bytes().unwrap() as usize);
        let reference_bytes = if with_reference {
            2 * backing(16 * 4)
        } else {
            0
        };
        let scope = host
            .reserve_scope(&[(
                HostMemoryClass::InputPreprocess,
                requirements.persistent_cache_bytes
                    + 2 * (image_bytes + audio_bytes)
                    + reference_bytes,
            )])
            .unwrap();
        let persistent = scope.account();
        let image = tensor(
            &persistent,
            &format!("value.ph3.step.{name}.image"),
            geometry.image_sample.clone(),
            (0..32).map(|i| ((i * 7 + seed * 3) % 19) as f32 / 32. - 0.25),
        );
        let audio = tensor(
            &persistent,
            &format!("value.ph3.step.{name}.audio"),
            geometry.audio_sample.clone(),
            (0..8).map(|i| ((i * 3 + seed * 7) % 13) as f32 / 32. - 0.125),
        );
        let reference = with_reference.then(|| ReferenceState {
            latent: tensor(
                &persistent,
                "value.ph3.step.reference",
                ResolvedTensorSpec::new(
                    vec![1, 2, 1, 2, 4],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                (0..16).map(|i| ((i * 5 + seed * 11) % 17) as f32 / 32. - 0.25),
            ),
            patch_noise: tensor(
                &persistent,
                "value.ph3.step.reference_noise",
                ResolvedTensorSpec::new(
                    vec![2, 8],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                (0..16).map(|i| ((i * 13 + seed * 19) % 23) as f32 / 32. - 0.375),
            ),
            augmentation: 0.999,
        });
        let pin = f
            .execution
            .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
            .unwrap();
        let work = f.execution.execution_host().clone();
        let connector = tensor(
            &work,
            "value.ph3.step.connector",
            requirements.borrowed_conditioning.clone(),
            (0..f.assets.text_rows() * 4).map(|i| ((i * 17 + seed * 3) % 41) as f32 / 32. - 0.625),
        );
        let mut active = f
            .execution
            .start_stage(compiled.stage, 100 + seed as u64, &mut f.receipts)
            .unwrap_or_else(|e| panic!("fresh text attach: {e}"));
        let mut consumer = compiled
            .recipe
            .bind_refinement(&mut active.session, pin.weights(), work)
            .unwrap();
        let cache = consumer
            .execute_on_actor(
                f.execution.serial_actor.as_mut().unwrap(),
                id(format!("run.ph3.step.refine.{name}")),
                geometry.identity.clone(),
                &connector,
                &[PH3ImageRefinementDestination::new(
                    &geometry.identity,
                    &scope,
                )],
            )
            .unwrap_or_else(|e| panic!("real fresh-text refinement: {e:?}"));
        drop(consumer);
        assert_eq!(cache.len(), 1);
        let cache = cache.into_iter().next().unwrap();
        assert_eq!(cache.key().request_identity(), &geometry.identity);
        assert!(cache.key().matches_snapshot(pin.weights().evidence()));
        // This bookkeeping follows the production refinement caller only after
        // execute_on_actor returned real caches. It is not a synthetic runtime
        // receipt or the test's evidence that a GPU forward completed.
        active.receipt.completed_executions += 1;
        if let Some(p) = active.session.take_execution_profile() {
            active.receipt.device_profiles.push(p);
        }
        f.execution
            .finish_stage(active, &mut f.receipts)
            .unwrap_or_else(|e| panic!("refine retire: {e}"));
        drop(connector);
        let sampling = PH3ImageManagedSampling::new_for_request(
            &schedule(sigmas),
            persistent,
            geometry.image_sample.clone(),
            geometry.audio_sample.clone(),
            PH3ImageVelocityPrecision::Bf16,
            geometry.identity.clone(),
        )
        .unwrap();
        assert_eq!(
            sampling.step_index(),
            0,
            "fresh text cannot advance the solver"
        );
        Self {
            geometry,
            image,
            audio,
            sampling,
            cache,
            reference,
            _scope: scope,
            pin,
        }
    }
    fn samples(&self) -> [Vec<u8>; 2] {
        [
            self.image.bytes().bytes().to_vec(),
            self.audio.bytes().bytes().to_vec(),
        ]
    }
}
fn try_forward(
    f: &mut Fixture,
    lanes: &mut [&mut Lane],
    job: u64,
) -> std::result::Result<(), ImageGenerationFailure<Runtime>> {
    let mut geometry = f.batch(&lanes.iter().map(|l| &l.geometry).collect::<Vec<_>>());
    let steps = lanes
        .iter()
        .map(|l| (&l.geometry.identity, l.sampling.current_step().unwrap()))
        .collect::<Vec<_>>();
    let times = geometry.times(&steps, 0.999).unwrap();
    let compiled = f.compile(
        geometry
            .prepare(&f.adapter, PH3ImageDenoiserPhase::DenoiseRefinedText)
            .unwrap(),
    );
    let positions = std::mem::take(&mut geometry.positions);
    drop(geometry);
    drop(steps);
    let weights = lanes[0].pin.weights().clone();
    assert!(lanes
        .iter()
        .all(|l| l.pin.weights().evidence() == weights.evidence()));
    let host = f.execution.execution_host().clone();
    let active = f
        .execution
        .start_stage(compiled.stage.clone(), job, &mut f.receipts)?;
    assert!(
        active.receipt.weights_reused,
        "cohort replacement must attach the original snapshot"
    );
    let mut borrowed = lanes
        .iter_mut()
        .map(|l| step::StepLane {
            identity: &l.geometry.identity,
            image: &mut l.image,
            audio: &mut l.audio,
            sampling: &mut l.sampling,
            cache: &l.cache,
            reference: l.reference.as_ref().map(ReferenceState::view),
        })
        .collect::<Vec<_>>();
    let recorded_before = f.trace.status().recorded;
    let (active, advanced) = step::execute(
        &mut f.execution,
        &compiled,
        active,
        job,
        &host,
        &weights,
        &times,
        &positions,
        &mut borrowed,
        None,
    )?;
    drop(borrowed);
    let observed = trace_events(&f.trace, false);
    let added = &observed[usize::try_from(recorded_before).unwrap()..];
    let starts = added
        .iter()
        .enumerate()
        .filter(|(_, event)| event["kind"] == "packed_step_started")
        .collect::<Vec<_>>();
    let commits = added
        .iter()
        .filter(|event| event["kind"] == "packed_step_committed")
        .collect::<Vec<_>>();
    assert_eq!(
        starts.len(),
        1,
        "one original actor attempt begins this completed step"
    );
    assert_eq!(
        commits.len(),
        1,
        "only the fenced all-lane solver commit publishes completion"
    );
    let event = commits[0];
    assert_eq!(
        event["start_sequence"],
        recorded_before + starts[0].0 as u64 + 1,
        "completion must pair with the actual recorded actor boundary"
    );
    assert_eq!(starts[0].1["cohort_job"], event["cohort_job"]);
    assert_eq!(starts[0].1["plan_hash"], event["plan_hash"]);
    assert_eq!(
        starts[0].1["program_fingerprint"],
        event["program_fingerprint"]
    );
    let started_members = starts[0].1["members"].as_array().unwrap();
    let committed_members = event["members"].as_array().unwrap();
    assert_eq!(started_members.len(), committed_members.len());
    for (started, committed) in started_members.iter().zip(committed_members) {
        assert_eq!(started["request"], committed["request"]);
        assert_eq!(started["step_index"], committed["step_index"]);
    }
    assert_eq!(event["kind"], "packed_step_committed");
    assert_eq!(event["cohort_job"], job);
    assert_eq!(event["plan_hash"], compiled.stage.plan().plan_hash);
    assert_eq!(
        event["program_fingerprint"],
        compiled
            .stage
            .executable()
            .execution_plan()
            .payload()
            .program_fingerprint()
    );
    let observed_times = event["time_basis"].as_array().unwrap();
    assert_eq!(observed_times.len(), times.len());
    for (observed, actual) in observed_times.iter().zip(&times) {
        assert_eq!(
            (observed.as_f64().unwrap() as f32).to_bits(),
            actual.to_bits()
        );
    }
    assert_eq!(event["packed_request_count"], lanes.len());
    let members = event["members"].as_array().unwrap();
    assert_eq!(members.len(), lanes.len());
    let mut row_start = 0;
    for (((member, lane), step), runs) in members
        .iter()
        .zip(lanes.iter())
        .zip(&advanced)
        .zip(compiled.recipe.request_row_runs())
    {
        assert_eq!(member["request"], json!(lane.geometry.identity));
        assert_eq!(member["step_index"], step.index());
        for (field, actual) in [
            ("sigma", step.sigma()),
            ("next_sigma", step.next_sigma()),
            ("scheduler_timestep", step.timestep()),
        ] {
            assert_eq!(
                (member[field].as_f64().unwrap() as f32).to_bits(),
                actual.to_bits()
            );
        }
        assert_eq!(member["row_runs"], json!(runs));
        assert_eq!(member["packed_row_start"], row_start);
        row_start += runs.iter().map(|run| run.length()).sum::<u64>();
        assert_eq!(member["packed_row_end"], row_start);
    }
    for (lane, step) in lanes.iter().zip(advanced) {
        assert_eq!(lane.sampling.step_index(), step.index() + 1);
        for sample in [&lane.image, &lane.audio] {
            assert!(host.owns(sample.bytes()));
            assert!(sample
                .bytes()
                .bytes()
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
        }
    }
    f.execution.finish_stage(active, &mut f.receipts)?;
    Ok(())
}
fn forward(f: &mut Fixture, lanes: &mut [&mut Lane], job: u64) {
    try_forward(f, lanes, job).unwrap_or_else(|e| panic!("actual production step/fence: {e}"));
}
fn independent(
    f: &mut Fixture,
    name: &str,
    seed: usize,
    reference: bool,
    sigmas: &[f32],
) -> Vec<[Vec<u8>; 2]> {
    let mut lane = Lane::new(f, name, seed, reference, sigmas);
    let mut history = Vec::new();
    while !lane.sampling.is_complete() {
        let index = lane.sampling.step_index();
        forward(f, &mut [&mut lane], 1000 + seed as u64 * 16 + index as u64);
        history.push(lane.samples());
    }
    history
}
fn matches_b1(lane: &Lane, history: &[[Vec<u8>; 2]]) {
    assert_eq!(
        lane.samples(),
        history[lane.sampling.step_index() - 1],
        "{}: selected tiny F32 image/audio trajectory differs from independent B1 at step {}",
        lane.geometry.identity,
        lane.sampling.step_index()
    );
}

// Fixture-local, declared before execution: absolute F32 latent error <=2^-8
// and RMSE <=2^-10 in EACH image/audio stream. This allows vendor BF16 SDPA
// rounding without granting Torch parity, a global model bound or image quality.
// Oracle separation below must exceed eight times BOTH bounds. Do not widen
// these values based on a failed run: investigate the numerical boundary first.
const SDPA_MAX_ABSOLUTE_ERROR: f64 = 1. / 256.;
const SDPA_MAX_RMSE: f64 = 1. / 1024.;
fn latent_error(actual: &[u8], expected: &[u8]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty() && actual.len() % 4 == 0);
    let mut maximum = 0f64;
    let mut squares = 0f64;
    for (actual, expected) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
        let a = f32::from_le_bytes(actual.try_into().unwrap());
        let b = f32::from_le_bytes(expected.try_into().unwrap());
        assert!(a.is_finite() && b.is_finite());
        let error = (f64::from(a) - f64::from(b)).abs();
        maximum = maximum.max(error);
        squares += error * error;
    }
    (maximum, (squares / (actual.len() / 4) as f64).sqrt())
}
fn within_sdpa_bound(error: (f64, f64)) -> bool {
    error.0 <= SDPA_MAX_ABSOLUTE_ERROR && error.1 <= SDPA_MAX_RMSE
}
fn matches_policy(f: &Fixture, lane: &Lane, history: &[[Vec<u8>; 2]]) {
    if f.attention_policy == ImageCudaAttentionPolicy::Sequential {
        matches_b1(lane, history); // Preserve the original exact-byte contract.
        return;
    }
    let actual = lane.samples();
    let expected = &history[lane.sampling.step_index() - 1];
    for (stream, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = latent_error(actual, expected);
        assert!(within_sdpa_bound(error),
            "{} step {} stream {stream}: independent SDPA B1 differs: max_abs={}, rmse={}; declared bounds {SDPA_MAX_ABSOLUTE_ERROR}/{SDPA_MAX_RMSE}",
            lane.geometry.identity, lane.sampling.step_index(), error.0, error.1);
    }
}
fn separated_oracles(a: &[Vec<u8>; 2], b: &[Vec<u8>; 2]) {
    for (stream, (a, b)) in a.iter().zip(b).enumerate() {
        let error = latent_error(a, b);
        assert!(error.0 > 8. * SDPA_MAX_ABSOLUTE_ERROR && error.1 > 8. * SDPA_MAX_RMSE,
            "stream {stream}: independent request oracles are not separated enough to reject contamination: max_abs={}, rmse={}", error.0, error.1);
        assert!(
            !within_sdpa_bound(error),
            "a whole-lane swap must fail the exact comparison used below"
        );
    }
}

#[test]
#[ignore = "requires configured CUDA host; qualifies DiT+RefineOnly+Euler only, not full image generation"]
fn cuda_dit_refine_euler_mixed_progress_matches_b1_when_lanes_join_finish_cancel_and_replace() {
    let _gpu = gpu_guard();
    mixed_progress_trajectory(ImageCudaAttentionPolicy::Sequential);
}

#[test]
#[ignore = "requires actual CUDA/cuDNN and declared native libraries; vendor latent trajectory/RefineOnly isolation, not full service or image quality"]
fn cuda_dit_explicit_sdpa_latent_trajectory_isolates_mixed_progress_reorder_cancel_and_refill() {
    let _gpu = gpu_guard();
    mixed_progress_trajectory(ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1);
}

fn mixed_progress_trajectory(policy: ImageCudaAttentionPolicy) {
    let mut f = Fixture::with_attention_policy(policy);
    let initial = f.geometry("a", false);
    f.install(&initial);
    let a_grid = [1., 0.875, 0.5, 0.25, 0.];
    let b_grid = [1., 0.625, 0.];
    let c_grid = [1., 0.75, 0.5, 0.];
    let d_grid = [1., 0.625, 0.];
    let e_grid = [1., 0.];
    // Sequential retains its original exact-byte expectation. Explicit SDPA
    // uses the predeclared per-stream latent bounds above. Neither is a general
    // cross-M cuBLAS/Torch error claim or SLO.
    // Independent continuations use their own identities in the trace. Their
    // same seed-derived inputs and schedules remain the numerical B1 oracle.
    let a_expected = independent(&mut f, "oracle_a", 1, false, &a_grid);
    let b_expected = independent(&mut f, "oracle_b", 2, true, &b_grid);
    let c_expected = independent(&mut f, "oracle_c", 3, true, &c_grid);
    let d_expected = independent(&mut f, "oracle_d", 4, false, &d_grid);
    let e_expected = independent(&mut f, "oracle_e", 5, true, &e_grid);
    if policy == ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 {
        // Establish non-degeneracy at the SAME progress positions later used
        // by every real mixed cohort, before running those packed forwards.
        for (a, b) in [
            (&a_expected[1], &b_expected[0]),
            (&b_expected[1], &a_expected[2]),
            (&c_expected[0], &a_expected[3]),
            (&e_expected[0], &d_expected[1]),
        ] {
            separated_oracles(a, b);
        }
        // Same old lane/cursor with a second, strongly changed neighbor. The
        // actual second request is independently refined and has its own B1
        // trajectory; no zero/fake conditioning or copied model output is used.
        let changed_expected = independent(&mut f, "oracle_changed_neighbor", 9, true, &b_grid);
        separated_oracles(&a_expected[1], &changed_expected[0]);
        let mut old = Lane::new(&mut f, "changed_neighbor_survivor", 1, false, &a_grid);
        forward(&mut f, &mut [&mut old], 2100);
        matches_policy(&f, &old, &a_expected);
        let mut changed = Lane::new(&mut f, "changed_neighbor", 9, true, &b_grid);
        forward(&mut f, &mut [&mut changed, &mut old], 2101);
        matches_policy(&f, &old, &a_expected);
        matches_policy(&f, &changed, &changed_expected);
        drop((old, changed)); // Actual completed fence, no extra solver step.
    }
    let mut a = Lane::new(&mut f, "a", 1, false, &a_grid);
    forward(&mut f, &mut [&mut a], 2000);
    matches_policy(&f, &a, &a_expected);
    let a_before_join = a.samples();
    let mut b = Lane::new(&mut f, "b", 2, true, &b_grid);
    assert_eq!(
        a.samples(),
        a_before_join,
        "fresh join must not run DiT for old members"
    );
    forward(&mut f, &mut [&mut a, &mut b], 2001);
    matches_policy(&f, &a, &a_expected);
    matches_policy(&f, &b, &b_expected);
    assert_ne!(
        a.sampling.current_step().unwrap().timestep(),
        b.sampling.current_step().unwrap().timestep()
    );
    forward(&mut f, &mut [&mut b, &mut a], 2002);
    matches_policy(&f, &a, &a_expected);
    matches_policy(&f, &b, &b_expected);
    assert!(b.sampling.is_complete());
    drop(b); // completed member retires before its replacement is admitted
    let mut c = Lane::new(&mut f, "c", 3, true, &c_grid);
    forward(&mut f, &mut [&mut c, &mut a], 2003);
    matches_policy(&f, &a, &a_expected);
    matches_policy(&f, &c, &c_expected);
    assert!(a.sampling.is_complete());
    drop(a);
    assert!(!c.sampling.is_complete());
    let cancelled_index = c.sampling.step_index();
    let cancelled_cache = c.cache.key().clone();
    drop(c); // cancellation at a completed fence, without another model step
    let mut d = Lane::new(&mut f, "d", 4, false, &d_grid);
    assert_ne!(
        d.cache.key().request_identity(),
        cancelled_cache.request_identity()
    );
    forward(&mut f, &mut [&mut d], 2004);
    matches_policy(&f, &d, &d_expected);
    let mut e = Lane::new(&mut f, "e", 5, true, &e_grid);
    forward(&mut f, &mut [&mut e, &mut d], 2005);
    matches_policy(&f, &d, &d_expected);
    matches_policy(&f, &e, &e_expected);
    assert!(d.sampling.is_complete() && e.sampling.is_complete());
    assert_eq!(
        cancelled_index, 1,
        "cancelled member was advanced only at its one completed step"
    );
    drop(d);
    drop(e);
    if policy == ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 {
        assert!(
            f.receipts.iter().any(|receipt| receipt.execution.plan.stage
                == crate::image_generation::ImageStage::TextRefiner
                && receipt.execution.completed_executions > 0
                && receipt.execution.plan.profile.as_str()
                    == "p_h3_image.dit_forward.cuda.refine_only.comfy_mixed.cudnn_sdpa.v1"),
            "the actual fresh-text cache must come from an executed RefineOnly vendor phase"
        );
    }
    let (weak, events) = f.close();
    assert!(weak.upgrade().is_none());
    let cohort_steps = events
        .iter()
        .filter(|event| {
            event["kind"] == "packed_step_committed"
                && event["cohort_job"]
                    .as_u64()
                    .is_some_and(|job| (2000..=2005).contains(&job))
        })
        .collect::<Vec<_>>();
    assert_eq!(cohort_steps.len(), 6);
    let expected: &[&[(&str, usize)]] = &[
        &[("a", 0)],
        &[("a", 1), ("b", 0)],
        &[("b", 1), ("a", 2)],
        &[("c", 0), ("a", 3)],
        &[("d", 0)],
        &[("e", 0), ("d", 1)],
    ];
    for ((event, ordered), job) in cohort_steps.iter().zip(expected).zip(2000u64..) {
        assert_eq!(event["cohort_job"], job);
        let members = event["members"].as_array().unwrap();
        assert_eq!(members.len(), ordered.len());
        for (member, (name, step)) in members.iter().zip(*ordered) {
            assert_eq!(member["request"], format!("request.ph3.step.{name}"));
            assert_eq!(member["step_index"], *step);
        }
    }
    assert_ne!(
        cohort_steps[1]["members"][0]["step_index"],
        cohort_steps[1]["members"][1]["step_index"]
    );
    assert_ne!(
        cohort_steps[1]["members"][0]["scheduler_timestep"],
        cohort_steps[1]["members"][1]["scheduler_timestep"]
    );
    assert!(
        cohort_steps[4..].iter().all(|event| event["members"]
            .as_array()
            .unwrap()
            .iter()
            .all(|member| member["request"] != "request.ph3.step.c")),
        "the retired cancelled lane cannot reappear in a later real packed forward"
    );
    assert!(
        events
            .iter()
            .filter(|event| matches!(
                event["kind"].as_str(),
                Some("packed_step_started" | "packed_step_committed")
            ) && event["cohort_job"]
                .as_u64()
                .is_some_and(|job| (2004..=2005).contains(&job)))
            .all(|event| event["members"]
                .as_array()
                .unwrap()
                .iter()
                .all(|member| member["request"] != "request.ph3.step.c")),
        "a cancelled lane cannot enter even an uncommitted later actor attempt"
    );
    // This low-level caller supplies the fence transition and has no product
    // request owner. Only actual step starts/commits are recorded; no synthetic exit,
    // publication or full-asset header is manufactured for that cancellation.
}

#[test]
#[ignore = "requires configured CUDA host; qualifies DiT+RefineOnly+Euler atomic successor admission only"]
fn cuda_dit_refine_euler_late_successor_refusal_preserves_samples_and_cursors_before_retry() {
    let _gpu = gpu_guard();
    let mut f = Fixture::new();
    let initial = f.geometry("late_a", false);
    f.install(&initial);
    let grid = [1., 0.5, 0.];
    let mut a = Lane::new(&mut f, "late_a", 6, false, &grid);
    let mut b = Lane::new(&mut f, "late_b", 7, true, &grid);
    let before = [a.samples(), b.samples()];
    let parent = f.execution.host_account().clone();
    // These original exact refund baselines include the independently charged
    // trace Output allocation. The filler below still occupies only the two
    // original logical successor Input shapes; its capacity math is unchanged.
    let reserved_before = parent.usage().unwrap().committed_bytes();
    let allocated_before = parent.usage().unwrap().allocated_bytes();
    // Exactly the second lane's two currently unused successor backings are
    // occupied, leaving the first lane's existing persistent grant untouched.
    let persistent = b._scope.account();
    // Allocate the two logical shapes independently, exactly as sampling does.
    // Passing an already padded physical sum as one logical length would charge
    // alignment twice and reject this fixture before the actual model forward.
    let filler = [
        persistent
            .allocate_zeroed(HostMemoryClass::InputPreprocess, b.image.bytes().len())
            .unwrap()
            .freeze(),
        persistent
            .allocate_zeroed(HostMemoryClass::InputPreprocess, b.audio.bytes().len())
            .unwrap()
            .freeze(),
    ];
    assert_eq!(
        filler
            .iter()
            .map(ManagedHostBytes::backing_bytes)
            .sum::<usize>(),
        parent.backing_bytes(b.image.bytes().len()).unwrap()
            + parent.backing_bytes(b.audio.bytes().len()).unwrap()
    );
    let failure = match try_forward(&mut f, &mut [&mut a, &mut b], 3000) {
        Err(e) => e,
        Ok(()) => panic!("the last lane lacks its successor grant"),
    };
    assert_eq!([a.samples(), b.samples()], before);
    assert_eq!((a.sampling.step_index(), b.sampling.step_index()), (0, 0));
    let active = match failure {
        ImageGenerationFailure::Processing {
            active: Some(active),
            ..
        } => active,
        other => {
            panic!("the real completed forward must retain its original active owner: {other}")
        }
    };
    assert_eq!(
        active.receipt.completed_executions, 1,
        "the model forward actually completed before successor admission failed"
    );
    f.execution
        .finish_stage(active, &mut f.receipts)
        .unwrap_or_else(|e| panic!("failed-step owner retirement: {e}"));
    drop(filler);
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        reserved_before,
        "failed execution cannot retain another grant"
    );
    assert_eq!(
        parent.usage().unwrap().allocated_bytes(),
        allocated_before,
        "first-lane pending successors and all failed readbacks must refund before retry"
    );
    forward(&mut f, &mut [&mut a, &mut b], 3001);
    assert_eq!((a.sampling.step_index(), b.sampling.step_index()), (1, 1));
    drop(a);
    drop(b);
    let (_, events) = f.close();
    assert!(
        events
            .iter()
            .all(|event| event["kind"] != "packed_step_committed" || event["cohort_job"] != 3000),
        "a completed forward with a refused successor cannot record a commit"
    );
    assert!(
        events
            .iter()
            .any(|event| event["kind"] == "packed_step_started" && event["cohort_job"] == 3000),
        "the refused successor retains its genuine actor attempt without inventing a commit"
    );
    let retried = events
        .iter()
        .filter(|event| event["kind"] == "packed_step_committed" && event["cohort_job"] == 3001)
        .collect::<Vec<_>>();
    assert_eq!(retried.len(), 1);
    assert_eq!(retried[0]["members"].as_array().unwrap().len(), 2);
    assert_eq!(
        retried[0]["members"][0]["request"],
        "request.ph3.step.late_a"
    );
    assert_eq!(
        retried[0]["members"][1]["request"],
        "request.ph3.step.late_b"
    );
    assert_eq!(retried[0]["members"][0]["step_index"], 0);
    assert_eq!(retried[0]["members"][1]["step_index"], 0);
}
