//! Actual stored DiT + managed Euler integration across changing cohorts.
//!
//! This test composes real managed forwards, current latent gathers, per-row
//! time conditioning and atomic Euler publication. Cohort decisions occur at
//! completed boundaries in the test driver; ImageService is not involved.
//! It therefore establishes mathematical/ownership equivalence, not service
//! scheduler admission, HTTP cancellation or warm-plan reuse qualification.
use super::*;
use ferrum_diffusion::{Bf16EulerProfile, FlowMatchEulerSchedule, ManagedBf16Euler};
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
    TensorComponentPolicy,
};

const SAMPLE_BYTES: usize = IMAGE * H * 2;

#[derive(Clone, Copy)]
struct RequestPlan {
    name: &'static str,
    fixture_row: usize,
    sigmas: &'static [f32],
    latent_offset: f32,
}

struct Request {
    plan: RequestPlan,
    solver: ManagedBf16Euler,
    sample: ManagedHostTensor,
    // Test-only snapshots outside the runtime ledger, used for full-trajectory
    // comparisons. Runtime samples/readbacks remain managed throughout.
    history: Vec<Vec<u16>>,
}

impl Request {
    fn new(fixture: &Fixture, host: &HostMemoryAccount, plan: RequestPlan) -> Self {
        let spec = sample_spec();
        let mut owner = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, SAMPLE_BYTES)
            .unwrap();
        let source = &fixture.input[&id(F::LATENTS_ID)]
            [plan.fixture_row * SAMPLE_BYTES..(plan.fixture_row + 1) * SAMPLE_BYTES];
        for (destination, value) in owner
            .as_mut()
            .chunks_exact_mut(2)
            .zip(source.chunks_exact(2))
        {
            let x = bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32();
            destination.copy_from_slice(&bits(x + plan.latent_offset).to_le_bytes());
        }
        let sample = ManagedHostTensor::new(
            id(format!("value.trajectory.{}", plan.name)),
            spec.clone(),
            owner.freeze(),
        )
        .unwrap();
        let solver = ManagedBf16Euler::new(
            FlowMatchEulerSchedule::new(plan.sigmas, 0.0, 1000.0).unwrap(),
            host.clone(),
            spec,
            Bf16EulerProfile::CudaScalarBf16StagedRneV1,
        )
        .unwrap();
        let history = vec![snapshot(sample.bytes())];
        Self {
            plan,
            solver,
            sample,
            history,
        }
    }
}

fn sample_spec() -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![IMAGE as u64, H as u64],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}

fn snapshot(owner: &ManagedHostBytes) -> Vec<u16> {
    owner
        .bytes()
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect()
}

#[derive(Debug)]
struct CompletedWave {
    requests: Vec<&'static str>,
    steps: Vec<usize>,
    latent_rows: usize,
}

struct Harness<'a> {
    fixture: &'a Fixture,
    composition: ReferenceVNextComposition,
    host: HostMemoryAccount,
    waves: Vec<CompletedWave>,
}

impl<'a> Harness<'a> {
    fn new(fixture: &'a Fixture) -> Self {
        Self {
            fixture,
            composition: ReferenceVNextComposition::create(id("device.reference.trajectory"))
                .unwrap(),
            host: HostMemoryAccount::new(
                HostMemoryLimits::new(
                    12288,
                    HostMemoryClassLimits {
                        input_preprocess_bytes: 4096,
                        output_bytes: 4096,
                        transfer_staging_bytes: 4096,
                        weight_backing_bytes: 0,
                    },
                )
                .unwrap(),
            ),
            waves: Vec::new(),
        }
    }

    fn request(&self, plan: RequestPlan) -> Request {
        Request::new(self.fixture, &self.host, plan)
    }

    fn advance_wave(&mut self, rows: &mut [&mut Request]) {
        assert!(!rows.is_empty());
        let lengths = [2, 1];
        let config = QwenImage21TransformerConfig::new(
            self.fixture.config.transformer().clone(),
            QwenImage21T2iLayout::new(
                TEXT as u32,
                2,
                2,
                rows.iter()
                    .map(|request| lengths[request.plan.fixture_row])
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap();
        let step_indices = rows
            .iter()
            .map(|request| request.solver.step_index())
            .collect::<Vec<_>>();
        let timesteps = rows
            .iter()
            .map(|request| request.solver.current_step().unwrap().timestep().as_f32())
            .collect::<Vec<_>>();
        let family = TypedFamilyRegistration::new(F::new().unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&config).unwrap(),
                &profile_id(F::REFERENCE_PROFILE_ID).unwrap(),
            )
            .unwrap();
        let descriptor = self.composition.runtime().descriptor();
        let compiled = CompiledTensorComponent::compile(
            family,
            self.composition.registry(),
            self.composition.catalog(),
            &TensorComponentPolicy::new(
                descriptor.total_memory_bytes,
                descriptor
                    .dynamic_storage_profiles
                    .iter()
                    .copied()
                    .collect(),
            )
            .unwrap(),
            self.composition.weight_materializers(),
            &config.compile_options().unwrap(),
        )
        .unwrap();
        let wave = self.waves.len();
        let mut session = compiled
            .initialize(
                self.composition.runtime().clone(),
                self.host.clone(),
                ComponentWeightSource::Stored(&self.fixture.full.source),
                ComponentInitializationConfig {
                    request_id: id(format!("request.trajectory.init.{wave}")),
                    run_id: id(format!("run.trajectory.init.{wave}")),
                    transaction_id: id(format!("transaction.trajectory.init.{wave}")),
                    static_policy: StaticInitializationPolicy::new(4096, 64).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap_or_else(|e| panic!("initialization failed: {e:?}"));
        let ports = compiled
            .executable()
            .execution_plan()
            .payload()
            .tensor_ports()
            .unwrap();
        let mut inputs = Vec::new();
        for port in ports.inputs() {
            if port.value_id().as_str() == F::TIME_FEATURES_ID {
                let time = config
                    .managed_time_features_per_sample(&self.host, &timesteps)
                    .unwrap();
                assert_eq!(time.tensor(), port.tensor());
                inputs.push(time);
                continue;
            }
            let mut owner = self
                .host
                .allocate_zeroed(
                    HostMemoryClass::InputPreprocess,
                    port.tensor().minimum_storage_bytes().unwrap() as usize,
                )
                .unwrap();
            match port.value_id().as_str() {
                F::LATENTS_ID => {
                    assert_eq!(owner.len(), rows.len() * SAMPLE_BYTES);
                    for (destination, request) in owner
                        .as_mut()
                        .chunks_exact_mut(SAMPLE_BYTES)
                        .zip(rows.iter())
                    {
                        destination.copy_from_slice(request.sample.bytes().bytes());
                    }
                }
                F::TEXT_ID => {
                    let stride = TEXT * H * 2;
                    assert_eq!(owner.len(), rows.len() * stride);
                    let source = &self.fixture.input[&id(F::TEXT_ID)];
                    for (destination, request) in
                        owner.as_mut().chunks_exact_mut(stride).zip(rows.iter())
                    {
                        let offset = request.plan.fixture_row * stride;
                        destination.copy_from_slice(&source[offset..offset + stride]);
                    }
                }
                F::FREQUENCIES_ID => owner
                    .as_mut()
                    .copy_from_slice(&self.fixture.input[&id(F::FREQUENCIES_ID)]),
                other => panic!("unexpected DiT input {other}"),
            }
            inputs.push(
                ManagedHostTensor::new(
                    port.value_id().clone(),
                    port.tensor().clone(),
                    owner.freeze(),
                )
                .unwrap(),
            );
        }
        let outputs = session
            .execute(
                id(format!("run.trajectory.{wave}")),
                id(format!("request.trajectory.{wave}")),
                inputs,
            )
            .unwrap_or_else(|e| panic!("DiT forward failed: {e:?}"));
        let output = outputs
            .iter()
            .find(|output| output.value_id().as_str() == F::VELOCITY_ID)
            .unwrap();
        assert_eq!(
            output.tensor().dimensions(),
            [(rows.len() * IMAGE) as u64, H as u64]
        );
        assert_eq!(output.bytes().len(), rows.len() * SAMPLE_BYTES);
        let velocities = (0..rows.len())
            .map(|index| {
                let slice = output
                    .bytes()
                    .slice(index * SAMPLE_BYTES..(index + 1) * SAMPLE_BYTES)
                    .unwrap();
                assert_eq!(slice.class(), HostMemoryClass::Output);
                assert!(self.host.owns(&slice));
                slice
            })
            .collect::<Vec<_>>();
        if velocities.len() > 1 {
            assert!(!velocities[0].same_region(&velocities[1]));
        }
        let pending = rows
            .iter_mut()
            .zip(&velocities)
            .map(|(request, velocity)| {
                let Request { solver, sample, .. } = &mut **request;
                solver
                    .prepare_advance(sample, &sample_spec(), velocity)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        // No request publishes progress until the complete real readback has
        // admitted and calculated every request's new sample successfully.
        let updated = pending
            .into_iter()
            .map(|pending| pending.commit())
            .collect::<Vec<_>>();
        for ((request, (step, sample)), expected_step) in
            rows.iter_mut().zip(updated).zip(&step_indices)
        {
            assert_eq!(step.index(), *expected_step);
            assert_eq!(sample.value_id(), request.sample.value_id());
            request.sample = sample;
            request.history.push(snapshot(request.sample.bytes()));
        }
        self.waves.push(CompletedWave {
            requests: rows.iter().map(|request| request.plan.name).collect(),
            steps: step_indices,
            latent_rows: output.tensor().dimensions()[0] as usize,
        });
        drop((velocities, outputs));
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("close failed: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(self.composition.runtime().snapshot().live_allocations, 0);
        assert_eq!(
            self.host
                .usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::Output),
            0
        );
    }
}

fn standalone(fixture: &Fixture, plan: RequestPlan) -> Vec<Vec<u16>> {
    let mut harness = Harness::new(fixture);
    let mut request = harness.request(plan);
    while !request.solver.is_complete() {
        harness.advance_wave(&mut [&mut request]);
    }
    let history = request.history.clone();
    drop(request);
    assert_eq!(harness.host.usage().unwrap().committed_bytes(), 0);
    history
}

#[test]
fn late_join_retirement_reordering_and_boundary_cancel_match_standalone_trajectories() {
    let fixture = Fixture::new();
    let plans = [
        RequestPlan {
            name: "a",
            fixture_row: 0,
            sigmas: &[1., 0.5],
            latent_offset: 0.,
        },
        RequestPlan {
            name: "b",
            fixture_row: 1,
            sigmas: &[1., 0.75, 0.25],
            latent_offset: 0.,
        },
        RequestPlan {
            name: "c",
            fixture_row: 0,
            sigmas: &[1., 0.75, 0.5, 0.25],
            latent_offset: 0.5,
        },
        RequestPlan {
            name: "d",
            fixture_row: 1,
            sigmas: &[1., 0.5, 0.25],
            latent_offset: -0.5,
        },
    ];
    let expected = plans.map(|plan| standalone(&fixture, plan));
    assert!(
        expected
            .iter()
            .all(|trajectory| trajectory.windows(2).any(|pair| pair[0] != pair[1])),
        "nonzero model velocities must change actual latent values, not only progress counters"
    );
    let mut harness = Harness::new(&fixture);
    let mut a = harness.request(plans[0]);
    harness.advance_wave(&mut [&mut a]);
    let mut b = harness.request(plans[1]);
    harness.advance_wave(&mut [&mut b, &mut a]);
    assert_eq!(harness.waves.last().unwrap().steps, [0, 1]);
    assert_eq!(harness.waves.last().unwrap().latent_rows, 2 * IMAGE);
    assert!(a.solver.is_complete());
    assert_eq!(a.history, expected[0]);
    let before = harness.host.usage().unwrap().committed_bytes();
    drop(a);
    assert_eq!(
        harness.host.usage().unwrap().committed_bytes(),
        before - SAMPLE_BYTES
    );

    let mut c = harness.request(plans[2]);
    harness.advance_wave(&mut [&mut c, &mut b]);
    harness.advance_wave(&mut [&mut b, &mut c]);
    assert!(b.solver.is_complete());
    assert_eq!(b.history, expected[1]);
    drop(b);

    let mut d = harness.request(plans[3]);
    harness.advance_wave(&mut [&mut d, &mut c]);
    assert!(!d.solver.is_complete());
    assert_eq!(d.history, expected[3][..d.history.len()]);
    let cancelled_backing = d.sample.bytes().clone();
    let cancelled_snapshot = snapshot(&cancelled_backing);
    let before = harness.host.usage().unwrap().committed_bytes();
    // Model the cancellation decision at the completed boundary, without
    // implying that the unchanged HTTP service now calls a step-cancel API.
    drop(d);
    assert_eq!(harness.host.usage().unwrap().committed_bytes(), before);
    harness.advance_wave(&mut [&mut c]);
    assert_eq!(harness.waves.last().unwrap().requests, ["c"]);
    assert_eq!(harness.waves.last().unwrap().latent_rows, IMAGE);
    assert_eq!(snapshot(&cancelled_backing), cancelled_snapshot);
    assert!(c.solver.is_complete());
    assert_eq!(c.history, expected[2]);
    drop((cancelled_backing, c));
    assert_eq!(harness.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(harness.host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(harness.composition.runtime().snapshot().live_allocations, 0);
}
