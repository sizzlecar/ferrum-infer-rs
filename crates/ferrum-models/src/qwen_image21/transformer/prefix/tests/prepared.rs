//! CPU artifact compatibility across genuine profile/phase graph changes.
//! This exercises no device provider and makes no CUDA/Metal execution claim.
use super::*;
use ferrum_quantization::{Bf16WholeTensorFp8Materializer, PreparedFp8Archive, SafetensorsArchive};
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

fn prepare(config: &QwenImage21TransformerConfig, profile: &str) -> PreparedModelFamily {
    TypedFamilyRegistration::new(F::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &profile_id(profile).unwrap(),
        )
        .unwrap()
}

#[test]
fn prepared_fp8_manifest_preserves_fused_and_prefix_weight_contracts() {
    // Existing fixture includes D128 (required by the CUDA fused profile),
    // two batches, reference-image prefix and padding in the second batch.
    let editing = config();
    // Export uses the production command's plain WholeStage profile and a
    // generation layout; edit/prefix layouts must not change its weight ABI.
    let generation = QwenImage21TransformerConfig::new(
        editing.transformer().clone(),
        QwenImage21T2iLayout::new(1, 2, 2, vec![1]).unwrap(),
    )
    .unwrap();
    let exported_family = prepare(&generation, F::METAL_FP8_PROFILE_ID);
    let groups = F::fp8_weight_groups(&generation).unwrap();
    let schema = exported_family.stored_weight_schema().unwrap();
    let directory = tempfile::tempdir().unwrap();
    // Small deterministic BF16 data, including signs and signed zero. The
    // converter handles real bytes; no fabricated manifest can satisfy this test.
    let rows: Vec<Vec<u8>> = schema
        .components
        .iter()
        .map(|component| {
            assert_eq!(component.physical_element_type(), ElementType::Bf16);
            (0..component.physical_bytes().unwrap() / 2)
                .flat_map(|i| [0_u16, 0x8000, 0x3f80, 0xbf80, 0x4380][i as usize % 5].to_le_bytes())
                .collect()
        })
        .collect();
    let tensors = schema
        .components
        .iter()
        .zip(&rows)
        .map(|(component, bytes)| {
            assert_eq!(component.external_names.len(), 1);
            (
                component.external_names[0].as_str(),
                TensorView::new(
                    Dtype::BF16,
                    component
                        .dimensions
                        .iter()
                        .map(|&d| usize::try_from(d).unwrap())
                        .collect(),
                    bytes,
                )
                .unwrap(),
            )
        });
    serialize_to_file(
        tensors,
        &None,
        &directory.path().join("original.safetensors"),
    )
    .unwrap();
    let source =
        SafetensorsArchive::open_single_file(directory.path(), "original.safetensors").unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            8 * 1024 * 1024,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 8 * 1024 * 1024,
            },
        )
        .unwrap(),
    );
    let materializer = Bf16WholeTensorFp8Materializer::new(host.clone(), groups.clone()).unwrap();
    let receipt = materializer
        .export_prepared(
            &exported_family,
            &source,
            &directory.path().join("prepared"),
        )
        .unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let archive =
        PreparedFp8Archive::open(&receipt.manifest_path, &receipt.manifest_sha256).unwrap();
    let expected = archive.manifest();
    let mut selected_descriptor = None;
    for profile in [
        F::CUDA_FP8_PROFILE_ID,
        F::CUDA_FP8_FUSED_PROFILE_ID,
        F::METAL_FP8_PROFILE_ID,
        F::METAL_FP8_FUSED_PROFILE_ID,
    ] {
        for phase in [
            QwenImage21TransformerPhase::Full,
            QwenImage21TransformerPhase::PrefixPrefill,
            QwenImage21TransformerPhase::PrefixDecode,
        ] {
            let config = editing.clone().with_execution_phase(phase).unwrap();
            let family = prepare(&config, profile);
            let groups = F::fp8_weight_groups(&config).unwrap();
            archive.validate_recipe(&family, &groups).unwrap();
            assert_eq!(
                family.stored_weight_schema().unwrap(),
                &expected.source_schema
            );
            let bound = Bf16WholeTensorFp8Materializer::new(host.clone(), groups)
                .unwrap()
                .with_prepared_archive(&family, archive.clone())
                .unwrap();
            // This public method re-derives the execution schema, rejects an
            // unequal supplied schema, then returns the ordered source map.
            assert_eq!(
                bound
                    .component_sources(&family, &expected.execution_schema)
                    .unwrap(),
                expected.component_sources
            );
            assert!(bound
                .selection(&family)
                .unwrap()
                .has_numeric_quality_artifact());
            assert_eq!(
                bound.execution_weight_bytes(&family).unwrap(),
                receipt.payload_bytes
            );
            assert_eq!(bound.maximum_group_accounted_bytes(&family).unwrap(), 0);
            if let Some(descriptor) = &selected_descriptor {
                assert_eq!(bound.descriptor(), descriptor);
            } else {
                selected_descriptor = Some(bound.descriptor().clone());
            }

            let fused = matches!(
                profile,
                F::CUDA_FP8_FUSED_PROFILE_ID | F::METAL_FP8_FUSED_PROFILE_ID
            );
            let attention = match (phase, fused) {
                (QwenImage21TransformerPhase::PrefixDecode, true) => {
                    CROSS_ATTENTION_FUSED_BF16_OPERATION_ID
                }
                (QwenImage21TransformerPhase::PrefixDecode, false) => {
                    CROSS_ATTENTION_BF16_OPERATION_ID
                }
                (_, true) => STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID,
                (_, false) => STRUCTURED_ATTENTION_BF16_OPERATION_ID,
            };
            assert!(family
                .program()
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .any(|n| n.operation_id.as_str() == attention));
            for name in [
                "weight.qwen_image21.txt_in.text_norm.weight",
                "weight.qwen_image21.txt_in.in_layer.weight",
                "weight.qwen_image21.txt_in.out_layer.weight",
            ] {
                assert_eq!(
                    family
                        .program()
                        .weights()
                        .iter()
                        .any(|w| w.weight_id.as_str() == name),
                    phase != QwenImage21TransformerPhase::PrefixDecode
                );
                let weight = expected
                    .execution_schema
                    .tensor(&WeightId::new(name).unwrap())
                    .unwrap();
                assert!(!weight.required);
                assert!(expected
                    .execution_schema
                    .physical_component_refs(&weight.id)
                    .unwrap()
                    .iter()
                    .all(|component| component.required));
            }
            if phase == QwenImage21TransformerPhase::PrefixPrefill {
                for port in config.prefix_cache_ports().unwrap() {
                    assert!(family.program().outputs().contains(port.value_id()));
                }
            }
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
    }
    // Keeping BF16 arithmetic is not the same recipe even with the same source
    // dimensions: selected converted weights must have actual FP8 consumers.
    let incompatible = prepare(&generation, F::METAL_PROFILE_ID);
    assert!(archive.validate_recipe(&incompatible, &groups).is_err());
}
