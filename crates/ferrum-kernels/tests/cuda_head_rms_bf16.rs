#![cfg(feature = "cuda")]
//! Predeclared qualification: exact BF16 bits against an independent RN32
//! adjacent-carry/two-round oracle, NaN class only, and <=2 BF16 ULP against
//! the F64 mathematical oracle on bounded finite fixtures. No Torch parity.
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
#[path = "support/head_rms_bf16_oracle.rs"]
mod oracle;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use fixture::*;
use std::sync::Arc;

#[test]
#[ignore = "requires configured CUDA host; managed HeadRMS with explicit carry reduction and two BF16 boundaries"]
fn managed_head_rms_wide_and_legacy_shapes_preserve_declared_rounding_and_ownership() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 32,
            opaque_memory_profile: "unqualified-head-rms-carry-fixture-v1".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(head_rms_norm_bf16_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::HeadRmsNorm]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.head-rms-carry"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    for width in [8193, 16384] {
        assert!(Config {
            rows: 2,
            inputs: width,
            outputs: 2 * width,
            kind: Kind::HeadRmsBf16
        }
        .try_compile(runtime.as_ref(), &registry, &catalog, &materializers)
        .is_err());
    }
    for width in oracle::WIDTHS {
        let c = Config {
            rows: 2,
            inputs: width as u64,
            outputs: 2 * width as u64,
            kind: Kind::HeadRmsBf16,
        };
        let compiled = c.compile(runtime.as_ref(), &registry, &catalog, &materializers);
        let account = host();
        let mut session = compiled
            .initialize(
                runtime.clone(),
                account.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id(format!("run.head-rms.init.{width}")),
                    request_id: id(format!("request.head-rms.init.{width}")),
                    transaction_id: id(format!("transaction.head-rms.init.{width}")),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut retained = None;
        let mut max_math_ulp = 0;
        for trial in 0..3 {
            let (x, gamma) = oracle::fixture(width, trial);
            let expected = oracle::reference(&x, &gamma, false);
            let mathematical = (trial < 2).then(|| oracle::mathematical(&x, &gamma));
            for repeat in 0..2 {
                let output = session
                    .execute(
                        id(format!("run.head-rms.{width}.{trial}.{repeat}")),
                        id(format!("request.head-rms.{width}.{trial}.{repeat}")),
                        vec![
                            c.managed(&account, INPUT, &oracle::bytes(&x)),
                            c.managed(&account, WEIGHT, &oracle::bytes(&gamma)),
                        ],
                    )
                    .unwrap();
                let actual: Vec<_> = output[0]
                    .bytes()
                    .bytes()
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                oracle::check_bits(&actual, &expected);
                if let Some(math) = &mathematical {
                    max_math_ulp = max_math_ulp.max(oracle::mathematical_error(&actual, math));
                }
                retained = Some(output);
            }
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|f| panic!("HeadRMS close: {:?}", f.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert!(
            account.usage().unwrap().allocated_bytes() > 0,
            "last output retains its original Host owner after close"
        );
        drop(retained);
        assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
        println!("HeadRMS width={width}: bitwise staged oracle, maximum mathematical BF16 ULP={max_math_ulp}, all owners retired");
    }
}
