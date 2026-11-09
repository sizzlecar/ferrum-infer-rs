#![cfg(feature = "cuda")]
//! Fresh Core sessions must admit backing and overwrite managed inputs on a native cache hit.
//! This family is weightless: its matrix port is an input, not a stored-weight source.

#[path = "support/tensor_numeric_component.rs"]
mod fixture;

use ferrum_execution::{
    component::{
        ComponentInitializationConfig, ComponentInitializationReceipt, ComponentWeightSource,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_dense_f32_provider},
    vnext_runtime::{
        CudaDeviceRuntime, CudaNativeAllocationReuse, CudaTensorRuntimeConfig,
        CudaTensorRuntimeOptions,
    },
};
use ferrum_runtime::*;
use fixture::{id, Config, Kind, INPUT, WEIGHT};
use std::{
    num::{NonZeroU64, NonZeroUsize},
    sync::{Arc, Mutex},
};
use tracing::{field::Visit, span, Event, Metadata, Subscriber};

const CEILING: u64 = 1 << 30;
const BOOTSTRAP: u64 = 256 << 20;
const CACHE_BYTES: u64 = 4 << 20;

// A global subscriber observes allocation events on the real driver thread.
// It retains only the typed cache-hit field, never native owners or log text.
#[derive(Clone, Default)]
struct AllocationEvents(Arc<Mutex<Vec<bool>>>);

impl Subscriber for AllocationEvents {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }
    fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
    fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
    fn enter(&self, _: &span::Id) {}
    fn exit(&self, _: &span::Id) {}
    fn event(&self, event: &Event<'_>) {
        #[derive(Default)]
        struct Reused(Option<bool>);
        impl Visit for Reused {
            fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
                if field.name() == "native_allocation_reused" {
                    self.0 = Some(value);
                }
            }
            fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
        }
        let mut reused = Reused::default();
        event.record(&mut reused);
        if let Some(value) = reused.0 {
            let mut events = self.0.lock().unwrap();
            assert!(events.len() < 128, "bounded fixture allocation observation");
            events.push(value);
        }
    }
}

// A rejected reservation is an actual shared-account boundary observation.
// It neither allocates native memory nor changes the existing claims.
fn claimed_bytes(owner: &DeviceCapacityOwner) -> u64 {
    match owner.reserve(DeviceOpaqueMemoryKind::Bootstrap, CEILING) {
        Err(VNextError::DeviceCapacityUnavailable(pressure)) => {
            assert_eq!(pressure.process_usable_bytes(), CEILING);
            pressure.process_claimed_bytes()
        }
        other => panic!(
            "expected the existing runtime claims to reject a full-ceiling reservation: {other:?}"
        ),
    }
}

#[test]
#[ignore = "requires exclusive CUDA; fresh Core permits and full managed-input overwrite across native backing reuse"]
fn native_backing_reuse_preserves_fresh_core_sessions_and_capacity() {
    let events = AllocationEvents::default();
    tracing::subscriber::set_global_default(events.clone()).unwrap();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager_with_options(
            CudaTensorRuntimeConfig {
                device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
                usable_memory_bytes: CEILING,
                bootstrap_memory_bytes: BOOTSTRAP,
                stream_memory_bytes: 16 << 20,
                maximum_streams: 2,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: 64,
                opaque_memory_profile: "unqualified-native-reuse-core-fixture".into(),
            },
            CudaTensorRuntimeOptions {
                native_allocation_reuse: CudaNativeAllocationReuse::Bounded {
                    maximum_idle_bytes: NonZeroU64::new(CACHE_BYTES).unwrap(),
                    maximum_entry_bytes: NonZeroU64::new(CACHE_BYTES).unwrap(),
                    maximum_entries: NonZeroUsize::new(16).unwrap(),
                },
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let owner = DeviceCapacityOwner::open(runtime.descriptor(), CEILING).unwrap();
    assert_eq!(claimed_bytes(&owner), BOOTSTRAP + CACHE_BYTES);
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(dense_linear_f32_contract().unwrap())],
        vec![cuda_vnext_dense_f32_provider(&runtime).unwrap()],
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.native_reuse.fixture"),
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(descriptor.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let config = Config {
        rows: 2,
        inputs: 3,
        outputs: 2,
        kind: Kind::F32Dense,
    };
    let component = config.compile(runtime.as_ref(), &registry, &catalog, &materializers);
    let host = fixture::host();
    let cases = [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 0.0, 2.0, 0.0, 1.0, 3.0],
            [7.0, 11.0, 16.0, 23.0],
        ),
        (
            [-3.0, 2.0, -1.0, 6.0, -5.0, 4.0],
            [2.0, -1.0, 3.0, -2.0, 4.0, 1.0],
            [-11.0, 13.0, 29.0, -28.0],
        ),
    ];
    for (round, (input, matrix, expected)) in cases.into_iter().enumerate() {
        let start = events.0.lock().unwrap().len();
        let mut session = component
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id(format!("run.native_reuse.init.{round}")),
                    request_id: id(format!("request.native_reuse.init.{round}")),
                    transaction_id: id(format!("transaction.native_reuse.init.{round}")),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        assert!(matches!(
            session.initialization_receipt(),
            ComponentInitializationReceipt::NoStatic
        ));
        let managed = |name, values: &[f32]| {
            config.managed(
                &host,
                name,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        let output = session
            .execute(
                id(format!("run.native_reuse.execute.{round}")),
                id(format!("request.native_reuse.execute.{round}")),
                vec![managed(INPUT, &input), managed(WEIGHT, &matrix)],
            )
            .unwrap();
        assert_eq!(output.len(), 1);
        let actual: Vec<_> = output[0]
            .bytes()
            .bytes()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected);
        drop(output);
        let IdleDynamicBackingOutcome::Retained(retained) =
            session.try_retain_idle_dynamic_backing(u64::MAX).unwrap()
        else {
            panic!("completed Core session must retain its admitted dynamic backing");
        };
        assert!(retained.retained_bytes() > 0);
        let live_claims = claimed_bytes(&owner);
        let IdleDynamicReleaseOutcome::Released(released) =
            session.try_release_idle_dynamic_backing().unwrap()
        else {
            panic!("completed Core session must retire its own dynamic claims");
        };
        assert_eq!(released.reclaimed_bytes(), retained.retained_bytes());
        assert_eq!(
            live_claims - claimed_bytes(&owner),
            retained.retained_bytes(),
            "a cache hit still consumes the complete fresh Core capacity claim"
        );
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|error| panic!("{:?}", error.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(
            claimed_bytes(&owner),
            BOOTSTRAP + CACHE_BYTES,
            "retired native backing remains charged to the fixed cache owner"
        );
        let observations = events.0.lock().unwrap();
        let current = &observations[start..];
        assert!(
            !current.is_empty(),
            "must observe actual driver allocation events"
        );
        if round == 1 {
            assert!(
                current.iter().any(|reused| *reused),
                "second fresh session must actually hit native backing cache"
            );
        }
    }
    drop((component, catalog, registry, materializers, runtime));
    let refunded = owner
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, CEILING)
        .unwrap();
    assert_eq!(
        refunded.bytes(),
        CEILING,
        "all native/cache owners retire before their capacity is refunded"
    );
}
