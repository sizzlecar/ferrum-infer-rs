//! Exercise the production preparation boundary without a CUDA device/runtime.
#[path = "../src/backend/cuda/vnext_ops/native_blocks/upstream_linear/preparation.rs"]
mod preparation;

use ferrum_interfaces::vnext::*;
use preparation::ProjectionPreparation::{BindingsOnly, Full};
use std::borrow::Cow;

fn binding(ordinal: u32, quantized: bool) -> ResolvedValueBinding {
    let component = WeightId::new(format!("component.{ordinal}")).unwrap();
    let encoding = if quantized {
        WeightEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: "quantization.gguf.iq4-xs".to_owned().try_into().unwrap(),
            logical_values_per_block: 256,
            bytes_per_block: 136,
        })
    } else {
        WeightEncoding::Dense {
            element_type: ElementType::F16,
        }
    };
    let component_spec = WeightComponentSpec {
        id: component.clone(),
        role: if quantized {
            WeightComponentRole::PackedValues
        } else {
            WeightComponentRole::Values
        },
        external_names: vec![component.to_string()],
        dimensions: vec![17, if quantized { 1 } else { 256 }],
        encoding,
        required: true,
    };
    let storage = ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
        Some(component.clone()),
        format!("resource.{ordinal}").try_into().unwrap(),
        0,
        component_spec.physical_bytes().unwrap(),
        component_spec.physical_element_type(),
    )
    .unwrap()])
    .unwrap();
    let layout = if quantized {
        PhysicalWeightLayout::BlockQuantized {
            blocks: PhysicalWeightComponentBinding::exact_contiguous(component),
            block_axis: 1,
            block_padding: PhysicalWeightPadding::Exact,
        }
    } else {
        PhysicalWeightLayout::Dense {
            component_id: component,
        }
    };
    let weight_id = WeightId::new(format!("weight.{ordinal}")).unwrap();
    let schema = WeightSchema {
        format_id: "weight-format.fixture".to_owned().try_into().unwrap(),
        layout_id: "layout.fixture".to_owned().try_into().unwrap(),
        version: ContractVersion::new(1, 0),
        components: vec![component_spec],
        tensors: vec![WeightTensorSpec {
            id: weight_id.clone(),
            dimensions: vec![17, 256],
            logical_element_type: ElementType::F16,
            physical_layout: layout,
            required: true,
        }],
    };
    let weight = ResolvedWeightBinding::from_schema(&schema, &weight_id).unwrap();
    ResolvedValueBinding::new(
        format!("value.{ordinal}").try_into().unwrap(),
        ResolvedValueRole::Input,
        ordinal,
        ResolvedTensorSpec::new(
            vec![17, 256],
            ElementType::F16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        Some(weight),
        storage,
    )
    .unwrap()
}

fn prepared(profile: UpstreamMarkerV2Profile, quantized: bool) -> PreparedProjectionNumerics {
    let values: Vec<_> = (2..=5).map(|i| binding(i, quantized && i == 2)).collect();
    let policy = profile.arithmetic();
    let prepared = PreparedProjectionNumerics::prepare(&policy, &values).unwrap();
    prepared.validate_bindings(&policy, &values).unwrap();
    prepared
}

fn facts(rows: u32, quantized: bool) -> UpstreamProjectionWaveFacts {
    UpstreamProjectionWaveFacts {
        role: ProjectionRole::CausalQuery,
        component_id: WeightId::new("component.2").unwrap(),
        local_rows: rows,
        layout: UpstreamProjectionLayout::Columns,
        input_stride: 256,
        output_stride: 20,
        input_byte_offset: 6,
        output_byte_offset: 10,
        weight_byte_offset: 12,
        input_available_bytes: u64::from(rows) * 256 * 2,
        output_available_bytes: u64::from(rows) * 20 * 2,
        weight_available_bytes: 17 * if quantized { 136 } else { 256 * 2 },
        retained_zero_padded_weight_rows: 17,
    }
}

#[test]
fn binding_strict_proof_matches_full_but_cannot_be_used_as_a_compute_key() {
    let numerics = prepared(UpstreamMarkerV2Profile::CausalPrefill, false);
    for rows in [1, 3, 8, 33, 2049] {
        let facts = facts(rows, false);
        let (full, full_key) = Full.prepare_wave(&numerics, &facts, None).unwrap();
        let (bindings, omitted) = BindingsOnly.prepare_wave(&numerics, &facts, None).unwrap();
        assert_eq!(bindings, full);
        assert!(matches!(
            bindings.route(),
            PreparedUpstreamProjectionRoute::StrictBase { .. }
        ));
        assert_eq!(full_key.required().unwrap(), full.fingerprint().unwrap());
        assert!(
            omitted.required().is_err(),
            "Full must never substitute an empty replay key"
        );
    }
}

#[test]
fn binding_preparation_retains_live_shape_range_and_native_plan_failures() {
    let strict = prepared(UpstreamMarkerV2Profile::CausalPrefill, false);
    let base = facts(3, false);
    let mut invalid = Vec::new();
    let mut f = base.clone();
    f.local_rows = 0;
    invalid.push(f);
    let mut f = base.clone();
    f.input_available_bytes = 1;
    invalid.push(f);
    let mut f = base.clone();
    f.output_available_bytes = 1;
    invalid.push(f);
    let mut f = base.clone();
    f.input_byte_offset = 7;
    invalid.push(f);
    let mut f = base.clone();
    f.input_stride = u64::MAX;
    invalid.push(f);
    let mut f = base.clone();
    f.output_byte_offset = u64::MAX - 1;
    invalid.push(f);
    let mut f = base.clone();
    f.component_id = WeightId::new("other.component").unwrap();
    invalid.push(f);
    let mut f = base.clone();
    f.role = ProjectionRole::SwiGluDown;
    invalid.push(f);
    for f in invalid {
        let full = Full
            .prepare_wave(&strict, &f, None)
            .err()
            .expect("invalid full proof");
        assert_eq!(
            BindingsOnly.prepare_wave(&strict, &f, None).err(),
            Some(full)
        );
    }
    let selected = prepared(UpstreamMarkerV2Profile::CausalPrefill, true);
    let f = facts(8, true);
    for purpose in [Full, BindingsOnly] {
        assert!(
            purpose.prepare_wave(&selected, &f, None).is_err(),
            "eligible leaf cannot silently become strict"
        );
    }
}

#[test]
fn native_and_g32_routes_preserve_the_exact_full_replay_digest() {
    let numerics = prepared(UpstreamMarkerV2Profile::CausalPrefill, true);
    let f = facts(33, true);
    let native = UpstreamNativePlanFacts {
        implementation_fingerprint: "fixture.native.marker-v2".into(),
        device_architecture: 1200,
        multiprocessors: 128,
        maximum_dynamic_shared_bytes: 65536,
        geometry: UpstreamNativeGeometry::Mmq {
            padded_inputs: 512,
            row_tile: 32,
            column_tile: 128,
            threads: 256,
            shared_bytes: 32768,
            packed_guard_blocks: 32,
            blocks: 128,
            fixup: true,
        },
    };
    for purpose in [Full, BindingsOnly] {
        let (wave, key) = purpose.prepare_wave(&numerics, &f, Some(&native)).unwrap();
        assert!(matches!(
            wave.route(),
            PreparedUpstreamProjectionRoute::Selected { .. }
        ));
        assert_eq!(key.required().unwrap(), wave.fingerprint().unwrap());
    }
    let g32 = prepared(UpstreamMarkerV2Profile::CausalG32MmqPrefill, true);
    for rows in [1, 3, 7, 32] {
        let mut f = facts(rows, true);
        f.weight_byte_offset = 5; // Byte-safe G32 still accepts an odd physical view.
        let (full, key) = Full.prepare_wave(&g32, &f, None).unwrap();
        let (bindings, other) = BindingsOnly.prepare_wave(&g32, &f, None).unwrap();
        assert!(matches!(
            full.route(),
            PreparedUpstreamProjectionRoute::G32 { .. }
        ));
        assert_eq!(full, bindings);
        assert_eq!(key.required().unwrap(), other.required().unwrap());
        f.weight_available_bytes -= 1;
        assert!(BindingsOnly.prepare_wave(&g32, &f, None).is_err());
    }
}

#[test]
fn binding_metadata_borrows_exact_immutable_plan_and_compute_metadata_owns_it() {
    let numerics = prepared(UpstreamMarkerV2Profile::CausalPrefill, false);
    let wire = serde_json::to_vec(&numerics).unwrap();
    let borrowed = BindingsOnly.numerics(&numerics);
    assert!(matches!(&borrowed, Cow::Borrowed(v) if std::ptr::eq(*v, &numerics)));
    let owned = Full.numerics(&numerics);
    assert!(matches!(&owned, Cow::Owned(_)));
    assert_eq!(serde_json::to_vec(&borrowed).unwrap(), wire);
    assert_eq!(serde_json::to_vec(&owned).unwrap(), wire);
    let escaped = owned.into_owned();
    drop(borrowed);
    drop(numerics);
    assert_eq!(serde_json::to_vec(&escaped).unwrap(), wire);
}
