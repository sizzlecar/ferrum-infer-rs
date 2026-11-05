pub(in crate::p_h3_image_vae) use super::test_fixture::{
    architecture, archive, f32_bytes, limits, originals,
};
use super::*;
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use half::bf16;
type Family = PH3ImageVaeDecoderFamily;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn config(batch: u64, layers: u64) -> PH3ImageVaeDecoderExecutionConfig {
    PH3ImageVaeDecoderExecutionConfig::new(
        architecture(layers),
        batch,
        1,
        2,
        PH3ImageVaeDecoderBlockExecutionPreset::F32A8FullAttention,
    )
    .unwrap()
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            128 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                transfer_staging_bytes: 32 << 20,
                output_bytes: 48 << 20,
                weight_backing_bytes: 32 << 20,
            },
        )
        .unwrap(),
    )
}
fn floats(source: &ComfyCanonicalWeightSource, name: &str) -> Vec<f32> {
    let t = source.archive().tensor(name).unwrap();
    if t.element_type() == Some(ElementType::Bf16) {
        t.bytes()
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect()
    } else {
        t.bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}
fn affine(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let columns = weight.len() / bias.len();
    x.chunks_exact(columns)
        .flat_map(|row| {
            bias.iter().enumerate().map(move |(r, b)| {
                let mut sum = 0.0f32;
                for (i, x) in row.iter().enumerate() {
                    sum += *x * weight[r * columns + i];
                }
                sum + *b
            })
        })
        .collect()
}
#[test]
fn complete_declared_stack_reuses_block_math_and_has_one_shared_stored_inventory() {
    for layers in [2, 36] {
        let c = config(2, layers);
        let f = Family::new(limits()).unwrap();
        let built = compose::build(&c, limits()).unwrap();
        let schema = built.schema;
        let graph = built.program;
        let options = built.options;
        assert_eq!(graph.inputs().len(), 3);
        assert_eq!(graph.outputs(), [id::<ProgramValueId>(Family::OUTPUT_ID)]);
        assert_eq!(
            schema
                .components
                .iter()
                .filter(|p| p.physical_element_type() == ElementType::I8)
                .count() as u64,
            4 * layers
        );
        for l in 0..layers {
            assert!(schema.components.iter().any(|p| p.external_names
                == [format!("decoder.transformer_blocks.{l}.attn.to_qkv.weight")]));
        }
        for node in graph.blocks().iter().flat_map(|b| &b.nodes) {
            for output in &node.outputs {
                assert!(options.tensor_specs().contains_key(output));
            }
        }
        assert!(options
            .tensor_specs()
            .values()
            .all(|t| t.element_type == ElementType::F32));
        assert_eq!(
            options.tensor_specs()[&id::<ProgramValueId>(Family::OUTPUT_ID)].dimensions,
            [14, 12]
        );
        let bytes = schema
            .components
            .iter()
            .map(|p| p.physical_bytes().unwrap())
            .sum::<u64>();
        let mut low = limits();
        low.max_stored_bytes = bytes - 1;
        assert!(Family::new(low).unwrap().weights(&c).is_err());
        let mut exact = limits();
        exact.max_stored_bytes = bytes;
        assert!(Family::new(exact).unwrap().weights(&c).is_ok());
        let mut small = limits();
        small.max_metadata_bytes = serde_json::to_vec(&c).unwrap().len();
        assert!(Family::new(small).unwrap().weights(&c).is_err());
        let _ = f;
    }
}
#[test]
fn original_bf16_outer_views_and_small_vectors_have_real_host_backing_and_raw_w8_stays_compressed()
{
    let c = config(2, 2);
    let (_d, source) = archive(&originals(&c));
    let host = host();
    let checkpoint =
        PH3ImageVaeDecoderCheckpoint::new(&c, &source, host.clone(), limits()).unwrap();
    let ModelWeights::Stored(schema) = Family::new(limits()).unwrap().weights(&c).unwrap() else {
        unreachable!()
    };
    for component in &schema.components {
        let t = source
            .archive()
            .tensor(&component.external_names[0])
            .unwrap();
        let payload = checkpoint.component(component).unwrap();
        if t.element_type() == Some(ElementType::Bf16) {
            assert!(host.usage().unwrap().committed_bytes() > 0);
            assert!(payload.retained_host_memory().is_some());
            assert_eq!(payload.dimensions(), component.dimensions);
            let wanted = f32_bytes(floats(&source, &component.external_names[0]));
            assert_eq!(payload.bytes(), wanted);
        } else {
            assert_eq!(payload.bytes().as_ptr(), t.bytes().as_ptr());
            assert!(payload.retained_host_memory().is_some());
        }
        drop(payload);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    let bad_host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            16,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 16,
            },
        )
        .unwrap(),
    );
    let limited =
        PH3ImageVaeDecoderCheckpoint::new(&c, &source, bad_host.clone(), limits()).unwrap();
    let conv = schema
        .components
        .iter()
        .find(|p| p.external_names == ["post_quant_conv.weight"])
        .unwrap();
    assert!(limited.component(conv).is_err());
    assert_eq!(bad_host.usage().unwrap().committed_bytes(), 0);
}
mod canonical_affine;
#[cfg(feature = "cuda")]
pub(in crate::p_h3_image_vae) mod cuda;
