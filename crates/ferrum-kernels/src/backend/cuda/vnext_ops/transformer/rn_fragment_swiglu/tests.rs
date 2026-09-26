use super::*;
use ferrum_interfaces::execution_cost::{AlgorithmWorkKindV1, SelectedReplayAlgorithmTemplateV1};
use ferrum_interfaces::vnext::*;

#[test]
fn rn_fragment_physical_m_boundary_preserves_selected_and_retained_recipe() {
    let identity = CublasHandleApiIdentity::fixture_identity();
    for m in [1, 7, 8, 9] {
        let shape = Shape::new(
            m,
            256,
            512,
            RnF16FragmentSourceFormatV1::Q4K,
            RnF16FragmentSourceFormatV1::Q6K,
        )
        .unwrap();
        let actual = shape
            .selected(SloStructuredCostCapture::HostSettledV1, Some(identity))
            .unwrap();
        let replay = shape.project(m, Some(identity)).unwrap();
        assert_eq!(actual, replay);
        assert_eq!(actual.algorithm_work(), replay.algorithm_work());
        actual.validate_command(m, 3, 0).unwrap();
        let work = actual.algorithm_work().unwrap().unwrap();
        assert_eq!(
            work.entries()
                .iter()
                .filter(|e| e.kind() == AlgorithmWorkKindV1::LibraryCall)
                .count(),
            if m <= 8 { 0 } else { 2 }
        );
        let template = SelectedReplayAlgorithmTemplateV1::from_selected(&actual, m, 3, 0).unwrap();
        template.validate_binding(&replay).unwrap();
        assert!(shape
            .project(if m == 8 { 9 } else { 8 }, Some(identity))
            .is_none());
        assert!(shape
            .selected(SloStructuredCostCapture::Disabled, Some(identity))
            .is_none());
        assert_eq!(
            shape
                .selected(SloStructuredCostCapture::HostSettledV1, None)
                .is_some(),
            m <= 8
        );
    }
    for args in [
        (0, 256, 512),
        (1, 255, 512),
        (1, 256, 511),
        (u64::MAX, 256, 512),
        (1, 256, i32::MAX as u64),
    ] {
        assert!(Shape::new(
            args.0,
            args.1,
            args.2,
            RnF16FragmentSourceFormatV1::Q4K,
            RnF16FragmentSourceFormatV1::Q6K
        )
        .is_err());
    }
}

fn weight_binding(
    format: RnF16FragmentSourceFormatV1,
    missing_packet: bool,
) -> Result<ResolvedValueBinding, VNextError> {
    let dims = vec![2, 256, 256];
    let plan = RnF16FragmentPlanV1::from_dimensions(format, &dims)?;
    let dense = WeightId::new("weight.fixture.dense")?;
    let packet = WeightId::new("weight.fixture.packet")?;
    let logical = WeightId::new("weight.fixture.logical")?;
    let schema = WeightSchema {
        format_id: WeightFormatId::new(
            crate::gguf_rn_fragment_materializer::GGUF_RN_FRAGMENT_FORMAT_ID,
        )?,
        layout_id: WeightLayoutId::new("layout.fixture.rn-fragment")?,
        version: ContractVersion::new(1, 0),
        components: vec![
            WeightComponentSpec {
                id: dense.clone(),
                role: WeightComponentRole::Values,
                external_names: vec!["dense".into()],
                dimensions: dims.clone(),
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            },
            WeightComponentSpec {
                id: packet.clone(),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["packet".into()],
                dimensions: plan.packed_dimensions().to_vec(),
                encoding: plan.packed_encoding(),
                required: true,
            },
        ],
        tensors: vec![WeightTensorSpec {
            id: logical.clone(),
            dimensions: dims.clone(),
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
                dense_values: PhysicalWeightComponentBinding {
                    component_id: dense.clone(),
                    storage: PhysicalStorageLayout::exact_contiguous(),
                },
                fragment_values: PhysicalWeightComponentBinding {
                    component_id: packet.clone(),
                    storage: PhysicalStorageLayout::exact_contiguous(),
                },
                source_format: format,
            },
            required: true,
        }],
    };
    schema.validate(&ModelFamilyId::new("family.fixture.rn-fragment")?)?;
    let mut storage = vec![ResolvedStorageComponent::new(
        Some(dense),
        ResourceId::new("resource.fixture.dense")?,
        128,
        plan.dense_bytes(),
        ElementType::F16,
    )?];
    if !missing_packet {
        storage.push(ResolvedStorageComponent::new(
            Some(packet),
            ResourceId::new("resource.fixture.packet")?,
            384,
            plan.packed_bytes(),
            ElementType::U8,
        )?);
    }
    ResolvedValueBinding::new(
        ProgramValueId::new("value.fixture.weight")?,
        ResolvedValueRole::Input,
        1,
        ResolvedTensorSpec::new(dims, ElementType::F16, ResolvedTensorLayout::Contiguous)?,
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        Some(ResolvedWeightBinding::from_schema(&schema, &logical)?),
        ResolvedValueStorage::composite(storage)?,
    )
}
#[test]
fn rn_fragment_consumer_validates_dual_spans_and_rejects_plain_consumer_reinterpretation() {
    for format in [
        RnF16FragmentSourceFormatV1::Q4K,
        RnF16FragmentSourceFormatV1::Q5K,
        RnF16FragmentSourceFormatV1::Q6K,
    ] {
        let value = weight_binding(format, false).unwrap();
        let plan = weights::validate(&value).unwrap();
        assert_eq!(plan.source_format(), format);
        assert_eq!((plan.n(), plan.k()), (512, 256));
        let old = OperationId::new(DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID).unwrap();
        assert!(gguf_f16_projection::validate_values(&old, &[value]).is_err());
        if let Ok(missing) = weight_binding(format, true) {
            assert!(weights::validate(&missing).is_err());
        }
    }
}
#[test]
fn rn_fragment_embedded_ptx_contains_the_production_entries() {
    let ptx = crate::ptx::VNEXT_GGUF
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    let words = ptx
        .split(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    for entry in [ENTRY, "vnext_rn_fragment_coefficients"] {
        assert!(
            words.windows(2).any(|w| w == [".entry", entry]),
            "embedded PTX missing {entry}"
        );
    }
}

mod gpu;

#[test]
fn rn_fragment_requires_the_embedded_mma_target_not_just_a_new_device() {
    for target in ["sm_80", "sm_90a", "sm_120"] {
        assert!(compiled_mma_target(&format!(
            ".version 8.0\n.target {target}\n.address_size 64"
        )));
    }
    for ptx in [
        ".target sm_75",
        "// .target sm_80",
        ".target compute_80",
        ".target sm_unknown",
        ".target sm_80\n.target sm_90",
        ".target",
        ".target\n.target sm_80",
        ".target sm_80oops",
    ] {
        assert!(!compiled_mma_target(ptx), "{ptx}");
    }
}
