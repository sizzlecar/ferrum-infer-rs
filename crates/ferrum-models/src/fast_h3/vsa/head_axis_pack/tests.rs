use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use half::bf16;
use std::sync::Arc;
mod routing;

#[cfg(feature = "cuda")]
mod cuda;
pub(super) fn limits() -> FastH3HeadAxisPackLimits {
    FastH3HeadAxisPackLimits {
        maximum_batches: 2,
        maximum_rows: 1024,
        maximum_columns: 8192,
        maximum_input_bytes: 8 << 20,
        maximum_metadata_bytes: 4096,
    }
}
pub(super) fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            16 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                transfer_staging_bytes: 16 << 20,
                output_bytes: 16 << 20,
                weight_backing_bytes: 16 << 20,
            },
        )
        .unwrap(),
    )
}
pub(super) fn managed(
    account: &HostMemoryAccount,
    name: &str,
    shape: Vec<u64>,
    dtype: ElementType,
    bytes: &[u8],
) -> ManagedHostTensor {
    let mut b = account
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
        .unwrap();
    b.as_mut().copy_from_slice(bytes);
    ManagedHostTensor::new(
        id(name).unwrap(),
        ResolvedTensorSpec::new(shape, dtype, ResolvedTensorLayout::Contiguous).unwrap(),
        b.freeze(),
    )
    .unwrap()
}
fn input(c: &FastH3HeadAxisPackConfig, trial: usize) -> Vec<u8> {
    let elements = (c.batches * c.heads * c.rows * c.head_dim) as usize;
    let special = [
        0x00000000u32,
        0x80000000,
        0x00008000,
        0x00008001,
        0x00018000,
        0x007fffff,
        0x3f808000,
        0x3f818000,
        0xbf808000,
        0x7f7f7fff,
        0x7f7f8000,
        0xff7f8000,
        0x7f800001,
        0xff800001,
        0x7fc00000,
        0xff800000,
    ];
    match c.kind {
        HeadAxisPackKind::Bf16 => (0..elements)
            .flat_map(|i| ((i + trial) as u16).to_le_bytes())
            .collect(),
        HeadAxisPackKind::F32ToBf16 => (0..elements)
            .flat_map(|i| {
                let bits = if i % 19 == 0 {
                    special[(i / 19 + trial) % special.len()]
                } else {
                    (((i + trial) as u16 as u32) << 16)
                        | [0u32, 0x7fff, 0x8000, 0x8001, 0xffff][i % 5]
                };
                bits.to_le_bytes()
            })
            .collect(),
    }
}
fn oracle(c: &FastH3HeadAxisPackConfig, bytes: &[u8]) -> Vec<u16> {
    let mut y = Vec::new();
    for b in 0..c.batches {
        for row in 0..c.rows {
            for h in 0..c.heads {
                for d in 0..c.head_dim {
                    let index = (((b * c.heads + h) * c.rows + row) * c.head_dim + d) as usize;
                    y.push(match c.kind {
                        HeadAxisPackKind::Bf16 => {
                            u16::from_le_bytes(bytes[2 * index..2 * index + 2].try_into().unwrap())
                        }
                        HeadAxisPackKind::F32ToBf16 => bf16::from_f32(f32::from_le_bytes(
                            bytes[4 * index..4 * index + 4].try_into().unwrap(),
                        ))
                        .to_bits(),
                    });
                }
            }
        }
    }
    y
}
pub(super) fn qualify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    for kind in HeadAxisPackKind::ALL {
        let c = FastH3HeadAxisPackConfig::new(kind, 2, 3, 257, 43, limits()).unwrap();
        let family = TypedFamilyRegistration::new(FastH3HeadAxisPackFamily::new(limits()).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&c).unwrap(),
                &NumericalProfileId::new(c.profile_id()).unwrap(),
            )
            .unwrap();
        let d = runtime.descriptor();
        let compiled = Arc::new(
            CompiledTensorComponent::compile(
                family,
                registry,
                catalog,
                &TensorComponentPolicy::new(
                    d.total_memory_bytes,
                    d.dynamic_storage_profiles.iter().copied().collect(),
                )
                .unwrap(),
                materializers,
                &c.compile_options(limits()).unwrap(),
            )
            .unwrap(),
        );
        let account = host();
        let phase = (compiled.clone(), runtime.clone());
        let mut s = compiled
            .initialize_retained(
                runtime.clone(),
                account.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id("run.head_axis.init").unwrap(),
                    request_id: id("request.head_axis.init").unwrap(),
                    transaction_id: id("transaction.head_axis.init").unwrap(),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
                phase,
            )
            .unwrap_or_else(|e| {
                std::mem::forget(e);
                panic!("head-axis initialization owner retained")
            });
        let mut retained = None;
        for trial in 0..2 {
            let bytes = input(&c, trial);
            let expected = oracle(&c, &bytes);
            let inputs = vec![managed(
                &account,
                FastH3HeadAxisPackFamily::INPUT_ID,
                c.input_shape(),
                kind.input_type(),
                &bytes,
            )];
            let outputs = match s.session_mut().execute(
                id("run.head_axis.execute").unwrap(),
                id("request.head_axis.execute").unwrap(),
                inputs,
            ) {
                Ok(v) => v,
                Err(e) => {
                    std::mem::forget((s, e));
                    panic!("head-axis execution owners retained")
                }
            };
            assert_eq!(outputs[0].tensor().dimensions(), c.output_shape().unwrap());
            for (b, wanted) in outputs[0].bytes().bytes().chunks_exact(2).zip(expected) {
                let actual = u16::from_le_bytes(b.try_into().unwrap());
                if kind == HeadAxisPackKind::F32ToBf16 && bf16::from_bits(wanted).is_nan() {
                    assert!(bf16::from_bits(actual).is_nan());
                } else {
                    assert_eq!(actual, wanted);
                }
            }
            retained = Some(outputs);
        }
        let (_, receipt) = s.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("head-axis close owner retained")
        });
        drop(receipt);
        assert_eq!(
            account.usage().unwrap().allocated_bytes(),
            (c.batches * c.heads * c.rows * c.head_dim * 2) as usize
        );
        drop(retained);
        assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    }
}
#[test]
fn head_axis_managed_reference_preserves_all_bf16_encodings_and_narrows_once_with_independent_rne()
{
    let c = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.head_axis").unwrap(),
        16 << 20,
    )
    .unwrap();
    qualify(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
#[test]
fn head_axis_family_rejects_wrong_numeric_profile_and_real_input_capacity() {
    let mut cap = limits();
    cap.maximum_input_bytes = 2 * 3 * 5 * 7 * 4 - 1;
    assert!(FastH3HeadAxisPackConfig::new(HeadAxisPackKind::F32ToBf16, 2, 3, 5, 7, cap).is_err());
    let c = FastH3HeadAxisPackConfig::new(HeadAxisPackKind::Bf16, 2, 3, 5, 7, limits()).unwrap();
    let f = TypedFamilyRegistration::new(FastH3HeadAxisPackFamily::new(limits()).unwrap());
    assert!(f
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(FastH3HeadAxisPackFamily::F32_PROFILE_ID).unwrap()
        )
        .is_err());
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["heads"] = serde_json::json!(u64::MAX);
    assert!(f
        .prepare_with_profile(&wire, &NumericalProfileId::new(c.profile_id()).unwrap())
        .is_err());
}
