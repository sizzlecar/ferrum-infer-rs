//! Successful component lifecycle through the real synchronous Reference backend.
//! The existing image-projection family is reduced to one tiny BF16 matrix;
//! these exact arithmetic checks make no image-model or GPU parity claim.

use std::collections::{BTreeMap, HashMap};

use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentInitializationReceipt,
    ComponentWeightSource, TensorComponentPolicy, TensorComponentSession,
};
use ferrum_execution::{TensorExecutionOutput, TensorExecutorConfig};
use ferrum_kernels::backend::reference::{ReferenceDeviceRuntime, ReferenceVNextComposition};
use ferrum_models::qwen_image21::{
    QwenImage21Config, QwenImage21ImageProjectionFamily, StructureLimits,
    IMAGE_PROJECTION_INPUT_ID, IMAGE_PROJECTION_OUTPUT_ID, IMAGE_PROJECTION_PROFILE_ID,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use half::bf16;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde_json::json;

#[path = "vnext_component_session/request_state.rs"]
mod request_state;
#[path = "vnext_component_session/shared_weights.rs"]
mod shared_weights;

const DEVICE_BYTES: u64 = 4096;
const INPUT_BYTES: usize = 12;
const OUTPUT_BYTES: usize = 24;
const WEIGHT_BYTES: usize = 36;
const INPUT: [f32; 6] = [1., 2., 3., -1., 0., 2.];
const WEIGHTS: [f32; 18] = [
    2., -1., 1., 1., 3., -2., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.,
];
// X * W^T, independently calculated for the two rows. All sums are exact.
const EXPECTED: [f32; 12] = [3., 1., 1., 2., 3., 6., 0., -5., -1., 0., 2., 1.];

fn id<T: TryFrom<String, Error = VNextError>>(value: impl Into<String>) -> T {
    T::try_from(value.into()).unwrap()
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

struct Fixture {
    composition: ReferenceVNextComposition,
    family: PreparedModelFamily,
    host: HostMemoryAccount,
    capacity: DeviceCapacityOwner,
    source: SafetensorsArchive,
    _directory: tempfile::TempDir,
}

impl Fixture {
    fn new(device: &str) -> Self {
        let config = QwenImage21Config::from_hf_json(
            &serde_json::to_vec(&json!({
                "_class_name":"QwenImage21Transformer2DModel",
                "num_layers":1,"num_attention_heads":1,"attention_head_dim":6,
                "context_in_dim":6,"in_channels":3,"out_channels":3,
                "patch_size":1,"mlp_ratio":2,"axes_dims_rope":[2,2,2],
                "causal_condition":true,"eps":0.000001
            }))
            .unwrap(),
            StructureLimits::default(),
        )
        .unwrap();
        let family = TypedFamilyRegistration::new(QwenImage21ImageProjectionFamily::new().unwrap())
            .prepare_with_profile(
                &serde_json::to_value(config).unwrap(),
                &NumericalProfileId::new(IMAGE_PROJECTION_PROFILE_ID).unwrap(),
            )
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let weights = bytes(&WEIGHTS);
        serialize_to_file(
            vec![(
                "img_in.weight",
                TensorView::new(Dtype::BF16, vec![6, 3], &weights).unwrap(),
            )],
            &None::<HashMap<String, String>>,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let source = SafetensorsArchive::open(directory.path()).unwrap();
        let composition =
            ReferenceVNextComposition::create_with_capacity(id(device), DEVICE_BYTES).unwrap();
        // Keep the existing process account alive through plan replacements.
        // This is an accounting probe, not a second allocation ledger.
        let capacity =
            DeviceCapacityOwner::open(composition.runtime().descriptor(), DEVICE_BYTES).unwrap();
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                INPUT_BYTES + 2 * OUTPUT_BYTES + WEIGHT_BYTES,
                HostMemoryClassLimits {
                    input_preprocess_bytes: INPUT_BYTES,
                    output_bytes: 2 * OUTPUT_BYTES,
                    transfer_staging_bytes: WEIGHT_BYTES,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        );
        Self {
            composition,
            family,
            host,
            capacity,
            source,
            _directory: directory,
        }
    }

    fn initialize(&self, suffix: &str) -> TensorComponentSession<ReferenceDeviceRuntime> {
        let component = self.compile_rows(2, &self.family);
        let runtime = self.composition.runtime();
        let session = component
            .initialize(
                runtime.clone(),
                self.host.clone(),
                ComponentWeightSource::Stored(&self.source),
                Self::init_config(suffix),
            )
            .unwrap();
        let ComponentInitializationReceipt::Initialized { receipt } =
            session.initialization_receipt()
        else {
            panic!("real Stored weights require initialization")
        };
        assert_eq!(receipt.uploaded_bytes(), WEIGHT_BYTES as u64);
        assert_eq!(receipt.initialized_resource_count(), 1);
        assert!(runtime.snapshot().live_allocations > 0);
        assert!(self
            .capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
            .is_err());
        assert_eq!(self.host.usage().unwrap().committed_bytes(), 0);
        session
    }

    fn init_config(suffix: &str) -> ComponentInitializationConfig {
        ComponentInitializationConfig {
            request_id: id(format!("request.component.init.{suffix}")),
            run_id: id(format!("run.component.init.{suffix}")),
            transaction_id: id(format!("transaction.component.init.{suffix}")),
            static_policy: StaticInitializationPolicy::new(WEIGHT_BYTES as u64, 8).unwrap(),
            executor: TensorExecutorConfig::default(),
        }
    }

    fn compile_rows(
        &self,
        rows: u64,
        family: &PreparedModelFamily,
    ) -> CompiledTensorComponent<ReferenceDeviceRuntime> {
        let runtime = self.composition.runtime();
        let policy = TensorComponentPolicy::new(
            DEVICE_BYTES,
            runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap();
        // Deliberately do not retain the output here: the component compiler
        // must retain every output declared by the existing typed family.
        let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id(IMAGE_PROJECTION_INPUT_ID),
            ProgramTensorSpec {
                dimensions: vec![rows, family.program().weights()[0].tensor.dimensions[1]],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))
        .unwrap();
        CompiledTensorComponent::compile(
            family.clone(),
            self.composition.registry(),
            self.composition.catalog(),
            &policy,
            self.composition.weight_materializers(),
            &options,
        )
        .unwrap()
    }

    fn input(&self, factor: f32) -> ManagedHostTensor {
        let mut buffer = self
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, INPUT_BYTES)
            .unwrap();
        for (destination, value) in buffer.as_mut().chunks_exact_mut(2).zip(INPUT) {
            destination.copy_from_slice(&bf16::from_f32(value * factor).to_bits().to_le_bytes());
        }
        ManagedHostTensor::new(
            id(IMAGE_PROJECTION_INPUT_ID),
            ResolvedTensorSpec::new(
                vec![2, 3],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            buffer.freeze(),
        )
        .unwrap()
    }

    fn assert_device_reusable(&self) {
        assert_eq!(self.composition.runtime().snapshot().live_allocations, 0);
        // A full-capacity claim succeeds only after the old plan's real static
        // and dynamic grants are returned. No buffer is allocated by this probe.
        drop(
            self.capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
                .unwrap(),
        );
    }
}

fn execute(
    session: &mut TensorComponentSession<ReferenceDeviceRuntime>,
    fixture: &Fixture,
    suffix: &str,
    factor: f32,
) -> Vec<TensorExecutionOutput> {
    let outputs = session
        .execute(
            id(format!("run.component.execute.{suffix}")),
            id(format!("request.component.execute.{suffix}")),
            vec![fixture.input(factor)],
        )
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].value_id().as_str(), IMAGE_PROJECTION_OUTPUT_ID);
    assert_eq!(outputs[0].tensor().dimensions(), &[2, 6]);
    let expected = EXPECTED.map(|x| x * factor);
    assert_eq!(outputs[0].bytes().bytes(), bytes(&expected));
    assert!(fixture.host.owns(outputs[0].bytes()));
    assert_eq!(outputs[0].bytes().class(), HostMemoryClass::Output);
    outputs
}

#[test]
fn stored_component_reuses_static_weights_and_outputs_outlive_explicit_close() {
    let fixture = Fixture::new("device.component.explicit-close");
    let mut session = fixture.initialize("explicit");
    let after_init = fixture.composition.runtime().snapshot();
    let first = execute(&mut session, &fixture, "first", 1.);
    let second = execute(&mut session, &fixture, "second", 2.);
    let after_execute = fixture.composition.runtime().snapshot();
    assert_eq!(
        after_execute.dense_linear_launches - after_init.dense_linear_launches,
        2
    );
    // Repeated execution uploads only its inputs, not the initialized matrix.
    assert_eq!(
        after_execute.uploaded_bytes - after_init.uploaded_bytes,
        2 * INPUT_BYTES as u64
    );
    assert_eq!(
        fixture.host.usage().unwrap().allocated_bytes(),
        2 * OUTPUT_BYTES
    );
    let alias = first[0].bytes().slice(2..6).unwrap();
    match session.close() {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            assert_eq!(receipt.released_static_resources(), 1)
        }
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("close retained {strong_count} owners")
        }
        Err(error) => panic!("close failed: {:?}", error.failure()),
    }
    fixture.assert_device_reusable();
    assert_eq!(first[0].bytes().bytes(), bytes(&EXPECTED));
    assert_eq!(second[0].bytes().bytes(), bytes(&EXPECTED.map(|x| x * 2.)));
    drop((first, second));
    assert_eq!(alias.bytes(), bytes(&[1., 1.]));
    assert_eq!(
        fixture.host.usage().unwrap().committed_bytes(),
        OUTPUT_BYTES
    );
    drop(alias);
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(fixture.host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn dropping_a_completed_session_releases_owners_for_another_session() {
    let fixture = Fixture::new("device.component.drop-reuse");
    let mut first = fixture.initialize("before-drop");
    drop(execute(&mut first, &fixture, "before-drop", 1.));
    assert!(fixture.composition.runtime().snapshot().live_allocations > 0);
    drop(first); // Exercise the library Drop path, not explicit close.
    fixture.assert_device_reusable();
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);

    // The same physical runtime, capacity owner, Host parent and providers can
    // admit new static resources and another execution lane after abandonment.
    let mut second = fixture.initialize("after-drop");
    drop(execute(&mut second, &fixture, "after-drop", 2.));
    drop(second);
    fixture.assert_device_reusable();
    assert_eq!(
        fixture
            .composition
            .runtime()
            .snapshot()
            .dense_linear_launches,
        2
    );
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(fixture.host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn stored_component_trims_dynamic_backing_and_regrows_without_reuploading_weights() {
    let fixture = Fixture::new("device.component.idle-trim");
    let mut session = fixture.initialize("trim");
    let static_bytes = session
        .executable()
        .execution_plan()
        .payload()
        .memory()
        .static_bytes();
    let initialized_uploads = fixture.composition.runtime().snapshot().uploaded_bytes;
    let first = execute(&mut session, &fixture, "before-trim", 1.);
    let before = fixture.composition.runtime().snapshot();
    let IdleDynamicReleaseOutcome::Released(receipt) =
        session.try_release_idle_dynamic_backing().unwrap()
    else {
        panic!("completed execution must be idle");
    };
    assert!(receipt.reclaimed_bytes() > 0);
    assert!(receipt
        .pools()
        .iter()
        .all(|pool| pool.published_capacity_bytes() == 0));
    let trimmed = fixture.composition.runtime().snapshot();
    assert_eq!(
        trimmed.live_allocations, 1,
        "only the actual matrix remains resident"
    );
    assert!(before.live_allocations > trimmed.live_allocations);
    // Probe the original process account: all non-static capacity is reusable.
    // This reserves bytes without allocating a second device buffer.
    drop(
        fixture
            .capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                DEVICE_BYTES - static_bytes,
            )
            .unwrap(),
    );
    assert_eq!(first[0].bytes().bytes(), bytes(&EXPECTED));
    assert_eq!(
        fixture.host.usage().unwrap().committed_bytes(),
        OUTPUT_BYTES
    );
    let IdleDynamicReleaseOutcome::Released(empty) =
        session.try_release_idle_dynamic_backing().unwrap()
    else {
        panic!("already trimmed session remains open and idle");
    };
    assert_eq!(empty.reclaimed_bytes(), 0);
    let second = execute(&mut session, &fixture, "after-trim", 2.);
    let regrown = fixture.composition.runtime().snapshot();
    assert!(regrown.allocations > trimmed.allocations);
    assert_eq!(
        regrown.uploaded_bytes - initialized_uploads,
        2 * INPUT_BYTES as u64
    );
    assert_eq!(regrown.dense_linear_launches, 2);
    match session
        .close()
        .unwrap_or_else(|failure| panic!("close failed: {:?}", failure.failure()))
    {
        PlanRuntimeCloseOutcome::Closed(receipt) => {
            assert_eq!(receipt.released_static_resources(), 1)
        }
        PlanRuntimeCloseOutcome::Referenced { .. } => panic!("trim must not retain a root alias"),
    }
    fixture.assert_device_reusable();
    assert_eq!(second[0].bytes().bytes(), bytes(&EXPECTED.map(|x| x * 2.)));
    drop((first, second));
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
}
