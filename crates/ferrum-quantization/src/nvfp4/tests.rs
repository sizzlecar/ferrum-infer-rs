use std::collections::BTreeMap;

use ferrum_runtime::{
    ElementType, QuantizationFormatId, QuantizationGrouping, QuantizationPacking, QuantizationSpec,
    WeightComponentRole, WeightComponentSource, WeightComponentSpec, WeightEncoding, WeightId,
};
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

use super::*;
use crate::SafetensorsArchive;

mod scale_validation;

#[test]
fn codes_and_scale_extremes_match_the_format() {
    let positive = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for (i, value) in positive.into_iter().enumerate() {
        assert_eq!(decode_e2m1(i as u8).unwrap(), value);
        assert_eq!(
            decode_e2m1(i as u8 + 8).unwrap().to_bits(),
            (-value).to_bits()
        );
    }
    assert!(decode_e2m1(16).is_err());
    assert_eq!(decode_e4m3_scale(0).unwrap(), 0.0);
    assert_eq!(decode_e4m3_scale(1).unwrap(), 1.0 / 512.0);
    assert_eq!(decode_e4m3_scale(7).unwrap(), 7.0 / 512.0);
    assert_eq!(decode_e4m3_scale(8).unwrap(), 1.0 / 64.0);
    assert_eq!(decode_e4m3_scale(0x38).unwrap(), 1.0);
    assert_eq!(decode_e4m3_scale(0x7e).unwrap(), 448.0);
    for invalid in [0x7f, 0x80, 0xb8, 0xff] {
        assert!(decode_e4m3_scale(invalid).is_err());
    }
}

#[test]
fn tiled_scale_coordinates_cross_both_tile_boundaries() {
    let g = Nvfp4Geometry::new(129, 80).unwrap();
    assert_eq!(g.scale_shape(), [256, 8]);
    assert_eq!(g.scale_bytes(), 2048);
    for ((row, group), offset) in [
        ((0, 0), 0),
        ((32, 0), 4),
        ((1, 0), 16),
        ((127, 3), 511),
        ((0, 4), 512),
        ((128, 0), 1024),
        ((128, 4), 1536),
    ] {
        assert_eq!(g.scale_index(row, group).unwrap(), offset);
    }
    let mut tiled = vec![0; g.scale_bytes()];
    for (offset, value) in [
        (0, 1),
        (4, 2),
        (16, 3),
        (511, 4),
        (512, 5),
        (1024, 6),
        (1536, 7),
    ] {
        tiled[offset] = value;
    }
    let mut rows = vec![0; 129 * 5];
    g.unswizzle_scales_into(&tiled, &mut rows).unwrap();
    for (index, value) in [
        (0, 1),
        (32 * 5, 2),
        (5, 3),
        (127 * 5 + 3, 4),
        (4, 5),
        (128 * 5, 6),
        (128 * 5 + 4, 7),
    ] {
        assert_eq!(rows[index], value);
    }
    assert!(g.scale_index(129, 0).is_err());
    assert!(g.scale_index(0, 5).is_err());
    let mut untouched = [99; 3];
    assert!(g.unswizzle_scales_into(&tiled, &mut untouched).is_err());
    assert_eq!(untouched, [99; 3]);
}

#[test]
fn geometry_rejects_invalid_or_unaddressable_extents() {
    for (rows, columns) in [
        (0, 16),
        (1, 0),
        (1, 15),
        (usize::MAX, 16),
        (1, usize::MAX - 15),
        (isize::MAX as usize, 16),
    ] {
        assert!(
            Nvfp4Geometry::new(rows, columns).is_err(),
            "{rows} {columns}"
        );
    }
}

struct Fixture {
    names: Nvfp4MatrixNames,
    geometry: Nvfp4Geometry,
    tensors: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
}
impl Fixture {
    fn new(names: Nvfp4MatrixNames, geometry: Nvfp4Geometry) -> Self {
        let mut tensors = BTreeMap::from([
            (
                names.packed.clone(),
                (
                    Dtype::U8,
                    geometry.packed_shape().to_vec(),
                    vec![0x21; geometry.packed_bytes()],
                ),
            ),
            (
                names.scales.clone(),
                (
                    Dtype::U8,
                    geometry.scale_shape().to_vec(),
                    vec![0x40; geometry.scale_bytes()],
                ),
            ),
            (
                names.global_scale.clone(),
                (Dtype::F32, vec![1], 4.0f32.to_le_bytes().to_vec()),
            ),
        ]);
        if let Some(name) = &names.output_multiplier {
            tensors.insert(
                name.clone(),
                (Dtype::F32, vec![], 0.3f32.to_le_bytes().to_vec()),
            );
            tensors.insert(
                names.global_scale.clone(),
                (
                    Dtype::BF16,
                    vec![],
                    half::bf16::from_f32(3.328125)
                        .to_bits()
                        .to_le_bytes()
                        .to_vec(),
                ),
            );
        }
        if let Some(name) = &names.input_global_scale {
            tensors.insert(
                name.clone(),
                (Dtype::F32, vec![], 8.0f32.to_le_bytes().to_vec()),
            );
        }
        Self {
            names,
            geometry,
            tensors,
        }
    }
    fn open(&self) -> (tempfile::TempDir, SafetensorsArchive) {
        let directory = tempfile::tempdir().unwrap();
        let views = self.tensors.iter().map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        });
        serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
        let archive = SafetensorsArchive::open(directory.path()).unwrap();
        (directory, archive)
    }
    fn spec(&self) -> Nvfp4MatrixSpec {
        Nvfp4MatrixSpec {
            names: self.names.clone(),
            geometry: self.geometry,
        }
    }
    fn source(
        &self,
    ) -> (
        tempfile::TempDir,
        ferrum_types::Result<Nvfp4SafetensorsSource>,
    ) {
        let (directory, archive) = self.open();
        (
            directory,
            Nvfp4SafetensorsSource::new(archive, vec![self.spec()], limits()),
        )
    }
}
fn limits() -> Nvfp4SourceLimits {
    Nvfp4SourceLimits {
        maximum_matrices: 4,
        maximum_source_bytes: 64 * 1024,
    }
}
fn fixture() -> Fixture {
    Fixture::new(
        Nvfp4MatrixNames::serialized_encoder("layer.q"),
        Nvfp4Geometry::new(2, 32).unwrap(),
    )
}

#[test]
fn dequantization_preserves_nibble_order_sign_and_zero_blocks() {
    let mut fixture = fixture();
    fixture.tensors.get_mut(&fixture.names.packed).unwrap().2[0] = 0xe7;
    fixture.tensors.get_mut(&fixture.names.scales).unwrap().2[1] = 0;
    let (_directory, source) = fixture.source();
    let source = source.unwrap();
    let matrix = source.matrix(&fixture.names.packed).unwrap();
    assert_eq!(matrix.output_multiplier(), 0.25);
    assert_eq!(matrix.input_global_scale(), None);
    assert_eq!(matrix.value(0, 0).unwrap(), 3.0);
    assert_eq!(matrix.value(0, 1).unwrap(), -2.0);
    assert_eq!(matrix.value(0, 2).unwrap(), 0.25);
    assert_eq!(matrix.value(0, 3).unwrap(), 0.5);
    assert_eq!(matrix.value(0, 16).unwrap(), 0.0);
    assert_eq!(matrix.value(1, 0).unwrap(), 0.25);
    let mut output = vec![0.0; matrix.geometry().element_count()];
    matrix.dequantize_f32_into(&mut output).unwrap();
    assert_eq!(&output[..4], &[3.0, -2.0, 0.25, 0.5]);
    let mut untouched = [77.0; 2];
    assert!(matrix.dequantize_f32_into(&mut untouched).is_err());
    assert_eq!(untouched, [77.0; 2]);
}

#[test]
fn dit_preserves_serialized_alpha_independent_of_bf16_global_scale() {
    let fixture = Fixture::new(
        Nvfp4MatrixNames::fastvideo_dit("block.ff", true),
        Nvfp4Geometry::new(2, 32).unwrap(),
    );
    let (_directory, source) = fixture.source();
    let source = source.unwrap();
    let matrix = source.matrix(&fixture.names.packed).unwrap();
    assert_eq!(matrix.output_multiplier(), 0.3);
    assert_ne!(matrix.output_multiplier(), 1.0f32 / 3.328125);
    assert_eq!(matrix.input_global_scale(), Some(8.0));
    assert_eq!(matrix.value(0, 1).unwrap(), 0.6);
}

#[test]
fn decoder_accepts_raw_fp8_scale_storage_and_ignores_padding_values() {
    let mut fixture = Fixture::new(
        Nvfp4MatrixNames::packed_decoder("decoder.layers.0.q"),
        Nvfp4Geometry::new(1, 16).unwrap(),
    );
    let scales = fixture.tensors.get_mut(&fixture.names.scales).unwrap();
    scales.0 = Dtype::F8_E4M3;
    scales.1 = vec![512];
    scales.2.fill(0xff); // padding is not a logical block scale
    scales.2[0] = 0x40;
    let (_directory, source) = fixture.source();
    let source = source.unwrap();
    assert_eq!(
        source
            .matrix(&fixture.names.packed)
            .unwrap()
            .value(0, 0)
            .unwrap(),
        0.25
    );
}

#[test]
fn malformed_scales_shapes_and_missing_members_fail_before_execution() {
    for code in [0x7f, 0x80, 0xff] {
        let mut fixture = fixture();
        fixture.tensors.get_mut(&fixture.names.scales).unwrap().2[0] = code;
        assert!(fixture.source().1.is_err());
    }
    let mut fixture = fixture();
    fixture.tensors.get_mut(&fixture.names.packed).unwrap().0 = Dtype::I8;
    assert!(fixture.source().1.is_err());
    fixture.tensors.get_mut(&fixture.names.packed).unwrap().0 = Dtype::U8;
    fixture.tensors.get_mut(&fixture.names.packed).unwrap().1 = vec![4, 8];
    assert!(fixture.source().1.is_err());
    fixture.tensors.remove(&fixture.names.global_scale);
    assert!(fixture.source().1.is_err());
}

#[test]
fn scalar_and_capacity_validation_are_explicit() {
    for value in [0.0, -1.0, f32::INFINITY, f32::NAN, f32::from_bits(1)] {
        let mut fixture = fixture();
        fixture
            .tensors
            .get_mut(&fixture.names.global_scale)
            .unwrap()
            .2 = value.to_le_bytes().to_vec();
        assert!(fixture.source().1.is_err());
    }
    let fixture = fixture();
    let (_dir, archive) = fixture.open();
    assert!(Nvfp4SafetensorsSource::new(
        archive,
        vec![fixture.spec()],
        Nvfp4SourceLimits {
            maximum_matrices: 0,
            ..limits()
        }
    )
    .is_err());
    let (_dir, archive) = fixture.open();
    assert!(Nvfp4SafetensorsSource::new(
        archive,
        vec![fixture.spec()],
        Nvfp4SourceLimits {
            maximum_source_bytes: 1,
            ..limits()
        }
    )
    .is_err());
    let (_dir, archive) = fixture.open();
    assert!(
        Nvfp4SafetensorsSource::new(archive, vec![fixture.spec(), fixture.spec()], limits())
            .is_err()
    );
}

#[test]
fn dequantization_overflow_does_not_partially_modify_output() {
    let mut fixture = fixture();
    fixture
        .tensors
        .get_mut(&fixture.names.global_scale)
        .unwrap()
        .2 = 1.0e-36f32.to_le_bytes().to_vec();
    fixture
        .tensors
        .get_mut(&fixture.names.scales)
        .unwrap()
        .2
        .fill(0x7e);
    // The first element is zero, the second is six; checking only during
    // output writes would modify the first element before detecting overflow.
    fixture.tensors.get_mut(&fixture.names.packed).unwrap().2[0] = 0x70;
    let (_directory, source) = fixture.source();
    let source = source.unwrap();
    let matrix = source.matrix(&fixture.names.packed).unwrap();
    let mut output = vec![77.0; matrix.geometry().element_count()];
    assert!(matrix.dequantize_f32_into(&mut output).is_err());
    assert!(output.iter().all(|value| *value == 77.0));
}

fn packed_component(fixture: &Fixture) -> WeightComponentSpec {
    WeightComponentSpec {
        id: WeightId::new("component.nvfp4.packed").unwrap(),
        role: WeightComponentRole::PackedValues,
        external_names: vec![fixture.names.packed.clone()],
        dimensions: fixture.geometry.packed_shape().map(|n| n as u64).to_vec(),
        encoding: WeightEncoding::Quantized(QuantizationSpec {
            format_id: QuantizationFormatId::new(NVFP4_E2M1_E4M3_128X4_FORMAT_ID).unwrap(),
            bits_per_weight: 4,
            grouping: QuantizationGrouping::fixed(16),
            packing: QuantizationPacking::Interleaved,
            scale_type: ElementType::U8,
            zero_point_type: None,
        }),
        required: true,
    }
}

#[test]
fn source_payload_borrows_the_exact_mmap_and_rejects_mxfp4_contract() {
    let fixture = fixture();
    let (_directory, archive) = fixture.open();
    let archive = std::sync::Arc::new(archive);
    let source = Nvfp4SafetensorsSource::from_shared_archive(
        archive.clone(),
        vec![fixture.spec()],
        limits(),
    )
    .unwrap();
    let other = Nvfp4SafetensorsSource::from_shared_archive(
        archive.clone(),
        vec![fixture.spec()],
        limits(),
    )
    .unwrap();
    drop(archive);
    let mut component = packed_component(&fixture);
    let payload = source.component(&component).unwrap();
    assert_eq!(
        payload.bytes().as_ptr(),
        source
            .archive()
            .tensor(&fixture.names.packed)
            .unwrap()
            .bytes()
            .as_ptr()
    );
    assert!(payload.retained_host_memory().is_some());
    let retained = payload.retained_host_memory().unwrap().clone();
    assert_eq!(
        payload.bytes().as_ptr(),
        other.component(&component).unwrap().bytes().as_ptr()
    );
    let expected = payload.bytes().to_vec();
    assert_eq!(payload.element_type(), ElementType::U8);
    let WeightEncoding::Quantized(q) = &mut component.encoding else {
        unreachable!()
    };
    q.grouping = QuantizationGrouping::fixed(32);
    assert!(source.component(&component).is_err());
    drop(payload);
    drop((source, other));
    assert_eq!(retained.bytes(), expected);
}
