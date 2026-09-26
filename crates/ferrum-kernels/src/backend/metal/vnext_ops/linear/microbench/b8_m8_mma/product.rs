//! Three arms through real LinearLaunch/plain_plan/dispatch_linear.
//! Old product is reconstructed by PSO substitution, not a second old ELF.
use super::super::b8_two_b4::{ffn, gdn};
use super::*;
use ferrum_interfaces::vnext::{BufferRequest, BufferUsage, DeviceId, ResourceId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProductArm {
    OldProduct,
    CurrentProduct,
    ForcedSingleM8,
    SingleM32Oracle,
}

impl ProductArm {
    fn single(self) -> bool {
        matches!(self, Self::ForcedSingleM8 | Self::SingleM32Oracle)
    }
    fn old(self) -> bool {
        matches!(self, Self::OldProduct | Self::SingleM32Oracle)
    }
}

struct ProductPipelines {
    old: MetalLinearPipelines,
    current: MetalLinearPipelines,
}

impl ProductPipelines {
    fn new(device: &Device) -> Self {
        let mut old = MetalLinearPipelines::new(device).unwrap();
        // Only unsplit rows=8 reaches these slots. M8 and M32 both have
        // grid.x=1 at this width, with the same 128 threads and 8192-byte grant.
        // Existing rows=4 split calls keep the actual shared-weight PSOs.
        old.k_quant_gemm.q4_k_m8 = old.k_quant_gemm.q4_k.clone();
        old.k_quant_gemm.q5_k_m8 = old.k_quant_gemm.q5_k.clone();
        old.k_quant_gemm.q6_k_m8 = old.k_quant_gemm.q6_k.clone();
        Self {
            old,
            current: MetalLinearPipelines::new(device).unwrap(),
        }
    }
    fn get(&self, arm: ProductArm) -> &MetalLinearPipelines {
        if arm.old() {
            &self.old
        } else {
            &self.current
        }
    }
}

/// Replace a fixture-owned shared buffer with a real retained runtime region.
/// Alignment one requests no extra leading allocation offset; the existing
/// Metal shared buffer still has its real device alignment. Prefix/tail guards
/// are copied unchanged and stay visible to the original fixture readers.
fn retain(
    runtime: &MetalDeviceRuntime,
    buffer: &mut Buffer,
    dtype: ElementType,
) -> MetalBufferRegion {
    let bytes = buffer.length();
    let region = runtime
        .allocate_test_region(
            &BufferRequest::new(
                ResourceId::new("m8.product.fixture").unwrap(),
                bytes,
                1,
                BufferUsage::Transfer,
                dtype,
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(region.offset_bytes(), 0);
    assert_eq!(region.length_bytes(), bytes);
    // SAFETY: independent owned shared allocations, no command is in flight.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().cast::<u8>(),
            region.buffer().contents().cast::<u8>(),
            bytes as usize,
        );
    }
    *buffer = region.buffer().to_owned();
    region
}

fn retain_projection(
    runtime: &MetalDeviceRuntime,
    projection: &mut Projection,
) -> MetalBufferRegion {
    let parent = retain(runtime, &mut projection.weight, ElementType::U8);
    parent
        .test_subregion(WEIGHT_PREFIX as u64..(WEIGHT_PREFIX + projection.bytes.len()) as u64)
        .unwrap()
}

fn launch(
    projection: &Projection,
    weight: usize,
    input: usize,
    output: usize,
    stride: usize,
    column: usize,
) -> LinearLaunch {
    let format = match projection.shape.format {
        GgufBlockFormat::Q8_0 => LinearPhysicalFormat::Q8_0,
        value => physical(value),
    };
    linear_launch(
        PreparedLinearPart {
            region: weight,
            transform: None,
            format,
            output_offset: column.try_into().unwrap(),
            out_features: projection.shape.output,
        },
        input,
        output,
        ROWS as u64,
        u64::from(projection.shape.input),
        stride as u64,
        (PREFIX * 2) as u64,
        (PREFIX * 2) as u64,
    )
    .unwrap()
}

enum Fixture {
    Ffn(ffn::Case),
    Gdn(gdn::Case),
}

struct Prepared {
    fixture: Fixture,
    regions: Vec<MetalBufferRegion>,
    launches: Vec<LinearLaunch>,
    widths: Vec<usize>,
}

impl Prepared {
    fn ffn(
        runtime: &MetalDeviceRuntime,
        device: &Device,
        hidden: usize,
        intermediate: usize,
        format: GgufBlockFormat,
    ) -> Self {
        let mut case = ffn::Case::new(device, hidden, intermediate, format);
        let regions = vec![
            retain(runtime, &mut case.input.buffer, ElementType::F16),
            retain_projection(runtime, &mut case.gate),
            retain_projection(runtime, &mut case.up),
            retain_projection(runtime, &mut case.down),
            retain(runtime, &mut case.output[1].packed.buffer, ElementType::F16),
            retain(
                runtime,
                &mut case.output[1].activated.buffer,
                ElementType::F16,
            ),
            retain(runtime, &mut case.output[1].down.buffer, ElementType::F16),
        ];
        let launches = vec![
            launch(&case.gate, 1, 0, 4, intermediate * 2, 0),
            launch(&case.up, 2, 0, 4, intermediate * 2, intermediate),
            launch(&case.down, 3, 5, 6, hidden + 9, 4),
        ];
        validate_launch_regions(&regions, &launches).unwrap();
        Self {
            fixture: Fixture::Ffn(case),
            regions,
            launches,
            widths: vec![intermediate, intermediate, intermediate, hidden],
        }
    }

    fn gdn(
        runtime: &MetalDeviceRuntime,
        device: &Device,
        input: usize,
        widths: &[usize],
        formats: &[GgufBlockFormat],
    ) -> Self {
        let mut case = gdn::Case::new(device, input, widths, formats);
        let mut regions = vec![
            retain(runtime, &mut case.input.buffer, ElementType::F16),
            retain(runtime, &mut case.output[1].buffer, ElementType::F16),
        ];
        for projection in &mut case.projections {
            regions.push(retain_projection(runtime, projection));
        }
        let mut column = 0;
        let launches = case
            .projections
            .iter()
            .enumerate()
            .map(|(i, projection)| {
                let result = launch(projection, i + 2, 0, 1, case.width, column);
                column += projection.shape.output as usize;
                result
            })
            .collect::<Vec<_>>();
        validate_launch_regions(&regions, &launches).unwrap();
        Self {
            fixture: Fixture::Gdn(case),
            regions,
            launches,
            widths: widths.to_vec(),
        }
    }

    fn arm_launch(&self, index: usize, arm: ProductArm) -> LinearLaunch {
        let mut result = self.launches[index];
        if arm.single() {
            result.plain_plan = PlainLinearPlan::Single;
        }
        result
    }

    fn dispatches(&self, arm: ProductArm) -> usize {
        (0..self.launches.len())
            .map(|i| self.arm_launch(i, arm).dispatch_count() as usize)
            .sum::<usize>()
            + usize::from(matches!(&self.fixture, Fixture::Ffn(_)))
    }

    fn encode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        pipelines: &MetalLinearPipelines,
        arm: ProductArm,
    ) {
        for i in 0..self.launches.len() {
            if i == 2 {
                if let Fixture::Ffn(case) = &self.fixture {
                    case.encode_activation(encoder, pipelines, Arm::M8Mma);
                }
            }
            dispatch_linear(pipelines, encoder, &self.regions, self.arm_launch(i, arm));
        }
    }

    fn reset(&self) {
        match &self.fixture {
            Fixture::Ffn(case) => case.reset(Arm::M8Mma),
            Fixture::Gdn(case) => case.reset(Arm::M8Mma),
        }
    }

    fn snapshot(&self) -> Vec<Vec<f16>> {
        match &self.fixture {
            Fixture::Ffn(case) => {
                let s = case.snapshot(Arm::M8Mma);
                vec![s.gate, s.up, s.activated, s.down]
            }
            Fixture::Gdn(case) => case.snapshot(Arm::M8Mma),
        }
    }

    fn immutable(&self) {
        match &self.fixture {
            Fixture::Ffn(case) => case.immutable(),
            Fixture::Gdn(case) => case.immutable(),
        }
    }

    fn addresses(&self) -> Vec<usize> {
        self.launches
            .iter()
            .map(|l| self.regions[l.weight_region].buffer().contents() as usize)
            .collect()
    }

    fn raw_reference(&self) -> Vec<Vec<f16>> {
        match &self.fixture {
            Fixture::Ffn(case) => {
                let s = case.reference(false);
                vec![s.gate, s.up, s.activated, s.down]
            }
            Fixture::Gdn(case) => case.reference(false),
        }
    }

    fn diagnostic(&self, actual: &[Vec<f16>], expected: &[Vec<f16>]) -> serde_json::Value {
        match &self.fixture {
            Fixture::Ffn(case) => {
                let snapshot = |rows: &[Vec<f16>]| ffn::Snapshot {
                    gate: rows[0].clone(),
                    up: rows[1].clone(),
                    activated: rows[2].clone(),
                    down: rows[3].clone(),
                };
                ffn::qualify(&snapshot(actual), &snapshot(expected), case)
            }
            Fixture::Gdn(_) => serde_json::json!({
                "leaves":actual.iter().zip(expected).zip(&self.widths)
                    .map(|((a,e),&width)|metrics(a,e,width)).collect::<Vec<_>>()
            }),
        }
    }

    fn routing(&self, arm: ProductArm) -> serde_json::Value {
        serde_json::json!({
            "arm":format!("{arm:?}"), "physical_dispatches_per_workset":self.dispatches(arm),
            "launches":(0..self.launches.len()).map(|i|{
                let launch=self.arm_launch(i,arm);
                serde_json::json!({"rows":launch.params.rows,"input":launch.params.in_features,
                    "output":launch.params.out_features,"output_stride":launch.params.output_stride,
                    "column":launch.params.output_column_offset,"format":format!("{:?}",launch.format),
                    "plain_plan":format!("{:?}",launch.plain_plan),"dispatches":launch.dispatch_count()})
            }).collect::<Vec<_>>()
        })
    }
}

fn equal(actual: &[Vec<f16>], expected: &[Vec<f16>]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert_finite_bits(a, e, "product stage");
    }
}

fn changed(actual: &[Vec<f16>], expected: &[Vec<f16>]) -> Vec<usize> {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| {
            assert_eq!(a.len(), e.len());
            a.iter()
                .zip(e)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count()
        })
        .collect()
}

fn measure(
    queue: &CommandQueueRef,
    pipelines: &ProductPipelines,
    cases: &[Prepared],
    arm: ProductArm,
) -> serde_json::Value {
    let count = cases[0].dispatches(arm);
    assert!(cases.iter().all(|case| case.dispatches(arm) == count));
    let mut measurement = run(queue, Arm::M8Mma, cases.len(), count, |encoder| {
        for case in cases {
            case.encode(encoder, pipelines.get(arm), arm);
        }
    });
    measurement["arm"] = serde_json::json!(format!("{arm:?}"));
    measurement
}

fn experiment(label: &str, timing: bool, build: impl Fn(&MetalDeviceRuntime, &Device) -> Prepared) {
    let device = Device::system_default().expect("product route screen requires Metal");
    let config = crate::backend::metal::vnext_ops::metal_vnext_runtime_config(
        DeviceId::new("metal.m8.product-screen").unwrap(),
    )
    .unwrap();
    let runtime = MetalDeviceRuntime::new(config).unwrap();
    let queue = device.new_command_queue();
    let pipelines = ProductPipelines::new(&device);
    let cases = (0..if timing { 4 } else { 1 })
        .map(|_| build(&runtime, &device))
        .collect::<Vec<_>>();
    let arms = [
        ProductArm::OldProduct,
        ProductArm::CurrentProduct,
        ProductArm::ForcedSingleM8,
    ];
    println!(
        "{}",
        serde_json::json!({"kind":"b8_m8_product_configuration","scope":label,
        "device":device.name(),"rows":ROWS,"routing":arms.map(|arm|cases[0].routing(arm)),
        "independent_weight_addresses":cases.iter().map(Prepared::addresses).collect::<Vec<_>>(),
        "baseline":"real unchanged plain_plan, test-only original M32 PSO substitution",
        "candidate":"real unchanged plain_plan and current product PSOs",
        "third_arm":"test-only Single M8; not production or quality-qualified",
        "shader_sha256":format!("{:x}",Sha256::digest(k_quant_gemm::SHADER_SOURCE.as_bytes())),
        "warmup_rounds":WARMUP_ROUNDS,"measured_rounds":MEASURED_ROUNDS,
        "release_approved":false,"absolute_numerical_quality_claimed":false})
    );
    let raw = cases[0].raw_reference();
    let mut snapshots: Vec<Vec<Vec<Vec<f16>>>> = Vec::new();
    for arm in [
        ProductArm::OldProduct,
        ProductArm::CurrentProduct,
        ProductArm::ForcedSingleM8,
        ProductArm::SingleM32Oracle,
    ] {
        for case in &cases {
            case.reset();
        }
        measure(&queue, &pipelines, &cases, arm);
        let observed = cases.iter().map(Prepared::snapshot).collect::<Vec<_>>();
        for (i, case) in cases.iter().enumerate() {
            case.immutable();
            equal(&observed[i], &observed[i]);
            if arm == ProductArm::CurrentProduct {
                equal(&observed[i], &snapshots[0][i]);
            }
            if arm == ProductArm::SingleM32Oracle {
                equal(&observed[i], &snapshots[2][i]);
            }
            println!(
                "{}",
                serde_json::json!({"kind":"b8_m8_product_qualification","scope":label,
                "arm":format!("{arm:?}"),"workset":i,
                "raw_coefficient_f64_catalog":case.diagnostic(&observed[i],&raw),
                "different_elements_from_original_product_by_stage":if snapshots.is_empty(){vec![0;observed[i].len()]}else{changed(&observed[i],&snapshots[0][i])},
                "original_product_difference_metrics_by_stage":if snapshots.is_empty(){serde_json::Value::Null}else{
                    serde_json::json!(observed[i].iter().zip(&snapshots[0][i]).zip(&case.widths)
                        .map(|((a,e),&width)|metrics(a,e,width)).collect::<Vec<_>>())},
                "bitwise_contract":match arm {ProductArm::CurrentProduct=>"equal_original_product",ProductArm::SingleM32Oracle=>"equal_forced_single_m8",_=>"recorded_finite_reference"},
                "release_approved":false})
            );
        }
        snapshots.push(observed);
    }
    if timing {
        for count in [1, 4] {
            for round in 0..WARMUP_ROUNDS + MEASURED_ROUNDS {
                let order = if round % 2 == 0 {
                    arms
                } else {
                    [arms[2], arms[1], arms[0]]
                };
                for arm in order {
                    for case in &cases[..count] {
                        case.reset();
                    }
                    let measurement = measure(&queue, &pipelines, &cases[..count], arm);
                    let index = match arm {
                        ProductArm::OldProduct => 0,
                        ProductArm::CurrentProduct => 1,
                        ProductArm::ForcedSingleM8 => 2,
                        _ => unreachable!(),
                    };
                    for (i, case) in cases[..count].iter().enumerate() {
                        equal(&case.snapshot(), &snapshots[index][i]);
                    }
                    println!(
                        "{}",
                        serde_json::json!({"kind":"b8_m8_product_timing","scope":label,
                        "round":round,"warmup":round<WARMUP_ROUNDS,"measurement":measurement,
                        "all_stage_elements_and_guards_rechecked":true,"release_approved":false})
                    );
                }
            }
        }
    }
    for case in &cases {
        case.immutable();
    }
}

#[test]
fn b8_m8_product_keeps_real_wide_split_and_unsplit_routes() {
    experiment("ffn_q4_real_split", false, |r, d| {
        let case = Prepared::ffn(r, d, 1024, 1024, GgufBlockFormat::Q4K);
        assert_eq!(case.dispatches(ProductArm::OldProduct), 7);
        assert_eq!(case.dispatches(ProductArm::CurrentProduct), 7);
        assert_eq!(case.dispatches(ProductArm::ForcedSingleM8), 4);
        case
    });
    experiment("gdn_real_split_and_unsplit", false, |r, d| {
        let case = Prepared::gdn(
            r,
            d,
            1024,
            &[1088, 1024, 32, 32],
            &[
                GgufBlockFormat::Q5K,
                GgufBlockFormat::Q4K,
                GgufBlockFormat::Q8_0,
                GgufBlockFormat::Q8_0,
            ],
        );
        assert_eq!(
            case.launches
                .iter()
                .map(|launch| launch.dispatch_count())
                .collect::<Vec<_>>(),
            vec![1, 2, 1, 1]
        );
        assert_eq!(case.dispatches(ProductArm::OldProduct), 5);
        assert_eq!(case.dispatches(ProductArm::CurrentProduct), 5);
        assert_eq!(case.dispatches(ProductArm::ForcedSingleM8), 4);
        case
    });
}

#[test]
#[ignore = "exclusive Metal GPU: actual product plans, original/new/forced single full FFN"]
fn b8_m8_product_complete_ffn_microbench() {
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        experiment(format.format_id(), true, |r, d| {
            Prepared::ffn(r, d, 4096, 12288, format)
        });
    }
}

#[test]
#[ignore = "exclusive Metal GPU: actual product GDN projection plans; no recurrence"]
fn b8_m8_product_gdn_projections_microbench() {
    experiment("gdn_qkvzba_four_projections", true, |r, d| {
        Prepared::gdn(
            r,
            d,
            4096,
            &[8192, 4096, 32, 32],
            &[
                GgufBlockFormat::Q5K,
                GgufBlockFormat::Q4K,
                GgufBlockFormat::Q8_0,
                GgufBlockFormat::Q8_0,
            ],
        )
    });
    experiment("gdn_output_projection_only", true, |r, d| {
        Prepared::gdn(r, d, 4096, &[4096], &[GgufBlockFormat::Q5K])
    });
}
