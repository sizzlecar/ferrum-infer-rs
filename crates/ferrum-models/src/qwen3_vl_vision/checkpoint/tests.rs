use super::*;
use crate::qwen3_vl_vision::{comfy::test_fixture, Qwen3VLVisionArchitecture, Qwen3VLVisionGrid};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use half::bf16;
use std::sync::atomic::{AtomicUsize, Ordering};
mod video;

fn id<T: TryFrom<String>>(s: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
struct Count<'a> {
    inner: Qwen3VLVisionCheckpointSource<'a>,
    reads: AtomicUsize,
}
impl WeightComponentSource for Count<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.component(c)
    }
}
#[test]
fn original_bf16_checkpoint_executes_main_and_three_deepstack_outputs() {
    let mut raw = test_fixture::metadata();
    raw["vision_config"]
        .as_object_mut()
        .unwrap()
        .remove("dtype");
    let a = Qwen3VLVisionArchitecture::from_hf_value(&raw).unwrap();
    let c = Qwen3VLVisionConfig::new(a, vec![Qwen3VLVisionGrid::new(1, 2, 2).unwrap()]).unwrap();
    let mut original = test_fixture::Original::new(&test_fixture::config(2, 2));
    for (name, values) in &mut original.values {
        values.fill(if name.contains("norm") && name.ends_with("weight") {
            1.
        } else {
            0.
        });
    }
    let prefixes = [
        "model.visual.merger",
        "model.visual.deepstack_merger_list.0",
        "model.visual.deepstack_merger_list.1",
        "model.visual.deepstack_merger_list.2",
    ];
    for (stage, prefix) in prefixes.iter().enumerate() {
        for (i, x) in original
            .values
            .get_mut(&format!("{prefix}.linear_fc2.bias"))
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *x = (stage as f32 + 1.) / 8. + (i % 4) as f32 / 32.;
        }
    }
    let (_dir, archive) = original.archive();
    let host = HostMemoryAccount::new(
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
    );
    let p = PreparedQwen3VLVisionCheckpoint::prepare(
        c,
        Arc::new(archive),
        &NumericalProfileId::new(Qwen3VLVisionFamily::REFERENCE_PROFILE_ID).unwrap(),
        Qwen3VLVisionCheckpointLimits {
            maximum_metadata_bytes: 2 << 20,
            maximum_plan_nodes: 256,
            maximum_source_bytes: 4 << 20,
        },
    )
    .unwrap();
    let source = Count {
        inner: p.managed_source(&host),
        reads: AtomicUsize::new(0),
    };
    let components = p.family().stored_weight_schema().unwrap().components.len();
    let mut wrong = p.family().stored_weight_schema().unwrap().components[0].clone();
    wrong.encoding = WeightEncoding::Dense {
        element_type: ElementType::F32,
    };
    assert!(source.inner.component(&wrong).is_err());
    let composition =
        ReferenceVNextComposition::create(id("device.reference.vision_original")).unwrap();
    let d = composition.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        p.family().clone(),
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        p.compile_options(),
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.vision.init"),
                request_id: id("request.vision.init"),
                transaction_id: id("transaction.vision.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    assert_eq!(source.reads.load(Ordering::Relaxed), components);
    let mut inputs = p.managed_position_inputs(&host).unwrap().to_vec();
    let columns = p.config().architecture().patch_columns().unwrap();
    let mut patches = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, (4 * columns * 2) as usize)
        .unwrap();
    for b in patches.as_mut().chunks_exact_mut(2) {
        b.copy_from_slice(&bf16::from_f32(0.5).to_bits().to_le_bytes());
    }
    inputs.push(
        ManagedHostTensor::new(
            id(Qwen3VLVisionFamily::PATCHES_ID),
            ResolvedTensorSpec::new(
                vec![4, columns],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            patches.freeze(),
        )
        .unwrap(),
    );
    drop(source);
    drop(p);
    let outputs = session
        .execute(
            id("run.vision.forward"),
            id("request.vision.forward"),
            inputs,
        )
        .unwrap();
    assert_eq!(outputs.len(), 4);
    let ids = [
        Qwen3VLVisionFamily::OUTPUT_ID,
        Qwen3VLVisionFamily::DEEPSTACK_IDS[0],
        Qwen3VLVisionFamily::DEEPSTACK_IDS[1],
        Qwen3VLVisionFamily::DEEPSTACK_IDS[2],
    ];
    for (stage, name) in ids.into_iter().enumerate() {
        let o = outputs
            .iter()
            .find(|o| o.value_id().as_str() == name)
            .unwrap();
        assert_eq!(o.tensor().dimensions(), [1, 256]);
        for (i, bits) in o.bytes().bytes().chunks_exact(2).enumerate() {
            let expected =
                bf16::from_f32((stage as f32 + 1.) / 8. + (i % 4) as f32 / 32.).to_bits();
            assert_eq!(u16::from_le_bytes(bits.try_into().unwrap()), expected);
        }
    }
    assert!(session.close().is_ok());
    assert_eq!(outputs[0].bytes().bytes().len(), 512);
    drop(outputs);
    drop(compiled);
    drop(composition);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
