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

#[cfg(feature = "cuda")]
mod cuda;
pub(super) fn limits() -> FastH3KeyCenterLimits {
    FastH3KeyCenterLimits {
        maximum_batches: 2,
        maximum_rows: 1024,
        maximum_columns: 8192,
        maximum_tensor_bytes: 8 << 20,
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
fn input(c: &FastH3KeyCenterConfig, trial: usize) -> Vec<u16> {
    (0..c.batches * c.padded_rows * c.columns)
        .map(|i| {
            let batch = i / (c.padded_rows * c.columns);
            let row = (i / c.columns) % c.padded_rows;
            let column = i % c.columns;
            bf16::from_f32(if row + 3 >= c.padded_rows {
                0.0
            } else {
                (batch * 3 + column + trial as u64) as f32 / 8. + (row % 17) as f32 / 16.
            })
            .to_bits()
        })
        .collect()
}
fn oracle(c: &FastH3KeyCenterConfig, x: &[u16]) -> (Vec<u16>, Vec<u16>) {
    let mut mean = Vec::new();
    for b in 0..c.batches {
        for col in 0..c.columns {
            let sum = (0..c.padded_rows)
                .map(|r| {
                    f64::from(
                        bf16::from_bits(x[((b * c.padded_rows + r) * c.columns + col) as usize])
                            .to_f32(),
                    )
                })
                .sum::<f64>();
            mean.push(bf16::from_f32((sum / c.padded_rows as f64) as f32).to_bits());
        }
    }
    let y = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let b = i as u64 / (c.padded_rows * c.columns);
            let col = i as u64 % c.columns;
            bf16::from_f32(
                bf16::from_bits(v).to_f32()
                    - bf16::from_bits(mean[(b * c.columns + col) as usize]).to_f32(),
            )
            .to_bits()
        })
        .collect();
    (y, mean)
}
#[derive(Clone, Copy)]
enum Case {
    Dyadic(u64, u64, u64),
    MeanBoundary,
    ReductionOrder,
}
impl Case {
    fn config(self) -> FastH3KeyCenterConfig {
        let (batches, padded_rows, columns) = match self {
            Self::Dyadic(b, s, c) => (b, s, c),
            Self::MeanBoundary => (1, 2, 3),
            Self::ReductionOrder => (1, 9, 3),
        };
        FastH3KeyCenterConfig {
            batches,
            padded_rows,
            columns,
        }
    }
    fn values(self, c: &FastH3KeyCenterConfig, trial: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        match self {
            Self::Dyadic(..) => {
                let x = input(c, trial);
                let (y, m) = oracle(c, &x);
                (x, y, m)
            }
            Self::MeanBoundary => (
                vec![0x3f80, 1, 0x8000, 0x3f81, 1, 0x8000],
                vec![0, 0, 0x8000, 0x3c00, 0, 0x8000],
                vec![0x3f80, 1, 0],
            ),
            Self::ReductionOrder => {
                // The declared lane/tree order loses both unit additions.
                // A sequential reduction would instead keep them and change
                // the mean, so this case detects an accidental reassociation.
                let column = [16777216f32, -16777216., 1., 0., 0., 0., 0., 0., 1.];
                let x = column
                    .iter()
                    .flat_map(|&v| [bf16::from_f32(v).to_bits(), 1, 0])
                    .collect();
                let y = column
                    .iter()
                    .flat_map(|&v| [bf16::from_f32(v).to_bits(), 0, 0])
                    .collect();
                (x, y, vec![0, 1, 0])
            }
        }
    }
}
pub(super) fn qualify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    for case in [
        Case::Dyadic(2, 17, 35),
        Case::Dyadic(1, 129, 130),
        Case::MeanBoundary,
        Case::ReductionOrder,
    ] {
        let c = case.config();
        let family = TypedFamilyRegistration::new(FastH3KeyCenterFamily::new(limits()).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&c).unwrap(),
                &NumericalProfileId::new(FastH3KeyCenterFamily::PROFILE_ID).unwrap(),
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
                    run_id: id("run.key_center.init").unwrap(),
                    request_id: id("request.key_center.init").unwrap(),
                    transaction_id: id("transaction.key_center.init").unwrap(),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
                phase,
            )
            .unwrap_or_else(|e| {
                std::mem::forget(e);
                panic!("key center initialization owner retained")
            });
        let mut retained = None;
        for trial in 0..2 {
            let (values, expected, expected_mean) = case.values(&c, trial);
            let bytes = values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let inputs = vec![managed(
                &account,
                FastH3KeyCenterFamily::INPUT_ID,
                c.input_shape(),
                ElementType::Bf16,
                &bytes,
            )];
            let outputs = match s.session_mut().execute(
                id("run.key_center.execute").unwrap(),
                id("request.key_center.execute").unwrap(),
                inputs,
            ) {
                Ok(v) => v,
                Err(e) => {
                    std::mem::forget((s, e));
                    panic!("key center execution owners retained")
                }
            };
            for (name, shape, wanted) in [
                (FastH3KeyCenterFamily::OUTPUT_ID, c.input_shape(), expected),
                (
                    FastH3KeyCenterFamily::MEAN_ID,
                    c.mean_shape(),
                    expected_mean,
                ),
            ] {
                let output = outputs
                    .iter()
                    .find(|v| v.value_id().as_str() == name)
                    .unwrap();
                assert_eq!(output.tensor().dimensions(), shape);
                let actual = output
                    .bytes()
                    .bytes()
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(actual, wanted);
            }
            retained = Some(outputs);
        }
        let (_, receipt) = s.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("key center close owner retained")
        });
        drop(receipt);
        assert_eq!(
            account.usage().unwrap().allocated_bytes(),
            (c.batches * (c.padded_rows + 1) * c.columns * 2) as usize
        );
        drop(retained);
        assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    }
}
#[test]
fn key_center_managed_reference_uses_independent_f64_mean_and_retires_both_outputs() {
    let c = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.key_center").unwrap(),
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
fn key_center_family_rejects_wrong_profile_and_actual_capacity() {
    let c = FastH3KeyCenterConfig {
        batches: 2,
        padded_rows: 17,
        columns: 35,
    };
    let mut cap = limits();
    cap.maximum_tensor_bytes = 2 * 17 * 35 * 2 - 1;
    assert!(c.compile_options(cap).is_err());
    let f = TypedFamilyRegistration::new(FastH3KeyCenterFamily::new(limits()).unwrap());
    assert!(f
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new("unrelated.profile").unwrap()
        )
        .is_err());
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["padded_rows"] = serde_json::json!(u64::MAX);
    assert!(f
        .prepare_with_profile(
            &wire,
            &NumericalProfileId::new(FastH3KeyCenterFamily::PROFILE_ID).unwrap()
        )
        .is_err());
}
#[test]
fn key_center_layout_uses_padded_sequence_instead_of_valid_rows() {
    use crate::fast_h3::{vsa::*, *};
    let cp = FastH3Checkpoint::parse(
        include_bytes!("../../fixtures/inference.json"),
        include_bytes!("../../fixtures/transformer.json"),
        include_bytes!("../../fixtures/video_scheduler.json"),
        include_bytes!("../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap();
    let layout = FastH3VsaLayout::ref2va(
        &cp,
        3,
        &[FastH3Reference::Image {
            latent_height: 2,
            latent_width: 4,
        }],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        FastH3VsaLayoutLimits {
            maximum_packed_rows: 4096,
            maximum_tiles: 128,
            maximum_metadata_bytes: 1 << 20,
        },
    )
    .unwrap();
    let mut cap = limits();
    cap.maximum_tensor_bytes = 64 << 20;
    let c = FastH3KeyCenterConfig::from_layout(&layout, 2, cp.transformer(), cap).unwrap();
    assert_eq!(c.input_shape()[1], layout.padded_rows() as u64);
    assert!(
        layout
            .valid_rows_per_tile()
            .iter()
            .map(|&n| u64::from(n))
            .sum::<u64>()
            < c.input_shape()[1]
    );
}
