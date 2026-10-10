use super::*;

// These small CPU fixtures test declaration semantics, not a registered product
// threshold or performance qualification.
fn selection(k: u64, n: u64) -> UpstreamProjectionGeometrySelection {
    UpstreamProjectionGeometrySelection::M8Q4Q5Mmq {
        minimum_input_features: k,
        minimum_output_features: n,
    }
}

fn geometry_contract(k: u64, n: u64) -> CompositeNumericalArithmetic {
    let mut contract = UpstreamMarkerV2Profile::CausalExtraAllRows.arithmetic();
    contract.schema_version = COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_GEOMETRY;
    for projection in &mut contract.projections {
        for leaf in &mut projection.leaves {
            if matches!(
                leaf.format,
                ProjectionBlockFormat::Q4K | ProjectionBlockFormat::Q5K
            ) {
                let mut policy = leaf.arithmetic.upstream_policy().unwrap().clone();
                policy.geometry_selection = Some(selection(k, n));
                leaf.arithmetic = policy.staged().unwrap();
            }
        }
    }
    contract.validate().unwrap();
    contract
}

fn geometry_values(format: ProjectionBlockFormat, k: u64, n: u64) -> Vec<ResolvedValueBinding> {
    vec![
        binding(2, &[Some(format.abi().0)], k, n, false),
        binding(3, &[None], k, n, false),
        binding(4, &[None], k, n, false),
        binding(5, &[None], k, n, false),
    ]
}

fn geometry_facts(format: ProjectionBlockFormat, k: u64, n: u64) -> UpstreamProjectionWaveFacts {
    let mut f = facts(8, UpstreamProjectionLayout::Columns, n);
    f.input_stride = k + 3;
    f.input_available_bytes = 8 * f.input_stride * 2;
    f.weight_available_bytes =
        f.retained_zero_padded_weight_rows * (k / 256) * u64::from(format.abi().2);
    f
}

fn geometry_native(k: u64, n: u64, mmq: bool) -> UpstreamNativePlanFacts {
    let mut native = native(
        8,
        UpstreamProjectionLayout::Columns,
        n.try_into().unwrap(),
        mmq,
    );
    match &mut native.geometry {
        UpstreamNativeGeometry::Mmq { padded_inputs, .. }
        | UpstreamNativeGeometry::Mmvq { padded_inputs, .. } => {
            *padded_inputs = (k.div_ceil(512) * 512).try_into().unwrap();
        }
    }
    native
}

#[test]
fn geometry_policy_preserves_legacy_wire_and_requires_explicit_new_schema() {
    let original = policy(
        ProjectionBlockFormat::Q4K,
        UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
        UpstreamProjectionLayout::Columns,
        &[1, 8],
    );
    let wire = serde_json::to_string(&original).unwrap();
    assert!(!wire.contains("geometry_selection"));
    let restored: UpstreamProjectionPolicy = serde_json::from_str(&wire).unwrap();
    assert_eq!(serde_json::to_string(&restored).unwrap(), wire);
    let old_staged = original.clone().staged().unwrap();
    let mut selected = original;
    selected.geometry_selection = Some(selection(1024, 2048));
    let staged = selected.staged().unwrap();
    assert_eq!(
        staged.schema_version,
        NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_GEOMETRY
    );
    let mut wrong = staged.clone();
    wrong.schema_version = old_staged.schema_version;
    assert!(wrong.validate().is_err());
    let mut missing = old_staged;
    missing.schema_version = staged.schema_version;
    assert!(missing.validate().is_err());
    assert_eq!(
        serde_json::from_str::<StagedNumericalArithmetic>(&serde_json::to_string(&staged).unwrap())
            .unwrap(),
        staged
    );
    let mut unknown = serde_json::to_value(selection(1024, 2048)).unwrap();
    unknown["hidden_override"] = serde_json::json!(true);
    assert!(serde_json::from_value::<UpstreamProjectionGeometrySelection>(unknown).is_err());

    let contract = geometry_contract(1024, 2048);
    let mut wrong = contract.clone();
    wrong.schema_version = COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL;
    assert!(wrong.validate().is_err());
    let mut missing = UpstreamMarkerV2Profile::CausalExtraAllRows.arithmetic();
    missing.schema_version = contract.schema_version;
    assert!(missing.validate().is_err());
}

#[test]
fn geometry_selector_is_bounded_by_format_layout_rows_and_both_leaf_dimensions() {
    for format in [ProjectionBlockFormat::Q4K, ProjectionBlockFormat::Q5K] {
        let mut p = policy(
            format,
            UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
            UpstreamProjectionLayout::Columns,
            &[1, 4, 8],
        );
        p.geometry_selection = Some(selection(1024, 2048));
        p.validate().unwrap();
        for (m, k, n, expected) in [
            (8, 1024, 2048, UpstreamProjectionArithmetic::MmqDs4MarkerV2),
            (8, 1280, 2049, UpstreamProjectionArithmetic::MmqDs4MarkerV2),
            (8, 768, 4096, UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2),
            (
                8,
                2048,
                2047,
                UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
            ),
            (
                1,
                2048,
                4096,
                UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
            ),
            (
                4,
                2048,
                4096,
                UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
            ),
        ] {
            assert_eq!(
                p.select_arithmetic(UpstreamProjectionLayout::Columns, m, k, n),
                Some(expected)
            );
        }
        assert_eq!(
            p.select_arithmetic(UpstreamProjectionLayout::Columns, 7, 2048, 4096),
            None
        );
        assert_eq!(
            p.select_arithmetic(UpstreamProjectionLayout::Channels, 8, 2048, 4096),
            None
        );
        for bad in [selection(0, 2048), selection(1024, 0)] {
            let mut invalid = p.clone();
            invalid.geometry_selection = Some(bad);
            assert!(invalid.validate().is_err());
        }
        let mut invalid = p.clone();
        invalid.format = ProjectionBlockFormat::Iq4Xs;
        assert!(invalid.validate().is_err());
        let mut invalid = p.clone();
        invalid.routes[0].local_rows.remove(&8);
        assert!(invalid.validate().is_err());
        let mut invalid = p;
        invalid.routes[0].arithmetic = UpstreamProjectionArithmetic::MmvqQ8_1V1;
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn geometry_wave_requires_matching_native_plan_bounds_and_reconstruction_identity() {
    for format in [ProjectionBlockFormat::Q4K, ProjectionBlockFormat::Q5K] {
        let contract = geometry_contract(1024, 2048);
        let values = geometry_values(format, 1024, 2048);
        let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
        let f = geometry_facts(format, 1024, 2048);
        let n = geometry_native(1024, 2048, true);
        let wave = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
        let PreparedUpstreamProjectionRoute::Selected {
            arithmetic,
            retained_weight_validation,
            ..
        } = wave.route()
        else {
            panic!("selected")
        };
        assert_eq!(*arithmetic, UpstreamProjectionArithmetic::MmqDs4MarkerV2);
        assert_eq!(
            retained_weight_validation.as_ref().unwrap().arithmetic,
            *arithmetic
        );
        wave.validate_reconstructed(&contract, &values, &f, Some(&n))
            .unwrap();
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, None).is_err());
        assert!(PreparedUpstreamProjectionWave::prepare(
            &prepared,
            &f,
            Some(&geometry_native(1024, 2048, false))
        )
        .is_err());
        let changed = geometry_contract(1280, 2048);
        let changed_prepared = PreparedProjectionNumerics::prepare(&changed, &values).unwrap();
        assert_ne!(prepared.fingerprint(), changed_prepared.fingerprint());
        let fallback = PreparedUpstreamProjectionWave::prepare(
            &changed_prepared,
            &f,
            Some(&geometry_native(1024, 2048, false)),
        )
        .unwrap();
        assert!(matches!(
            fallback.route(),
            PreparedUpstreamProjectionRoute::Selected {
                arithmetic: UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
                ..
            }
        ));
        assert_ne!(wave.fingerprint().unwrap(), fallback.fingerprint().unwrap());
        assert!(wave
            .validate_reconstructed(&changed, &values, &f, Some(&n))
            .is_err());
        let mut bad = f.clone();
        bad.input_available_bytes = 1;
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &bad, Some(&n)).is_err());
        bad = f.clone();
        bad.output_byte_offset += 1;
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &bad, Some(&n)).is_err());
        bad = f.clone();
        bad.weight_available_bytes = 1;
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &bad, Some(&n)).is_err());
    }
}

fn mixed_query(k: u64, first_n: u64, second_n: u64) -> ResolvedValueBinding {
    let first = binding(2, &[Some("quantization.gguf.q4-k")], k, first_n, false);
    let second = binding(6, &[Some("quantization.gguf.q5-k")], k, second_n, false);
    let mut components = first.weight().unwrap().components().to_vec();
    components.extend_from_slice(second.weight().unwrap().components());
    let parts = components
        .iter()
        .zip([(0, first_n), (first_n, second_n)])
        .map(|(component, (offset, n))| CompositeWeightPart {
            layout: Box::new(PhysicalWeightLayout::BlockQuantized {
                blocks: PhysicalWeightComponentBinding::exact_contiguous(
                    component.component_id().clone(),
                ),
                block_axis: 1,
                block_padding: PhysicalWeightPadding::Exact,
            }),
            logical_offsets: vec![offset, 0],
            extents: vec![n, k],
        })
        .collect();
    let weight = ResolvedWeightBinding::from_parts(
        WeightId::new("weight.mixed").unwrap(),
        WeightFormatId::new("weight-format.gguf.native-block").unwrap(),
        WeightLayoutId::new("layout.fixture").unwrap(),
        ContractVersion::new(1, 0),
        PhysicalWeightLayout::Composite { parts },
        components,
    )
    .unwrap();
    let mut storage = first.storage().components().to_vec();
    storage.extend_from_slice(second.storage().components());
    ResolvedValueBinding::new(
        first.value_id().clone(),
        first.role(),
        first.ordinal(),
        ResolvedTensorSpec::new(
            vec![first_n + second_n, k],
            ElementType::F16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        Some(weight),
        ResolvedValueStorage::composite(storage).unwrap(),
    )
    .unwrap()
}

#[test]
fn geometry_mixed_physical_leaves_use_each_leaf_n_not_the_composite_width() {
    let contract = geometry_contract(1024, 2048);
    let mut values = geometry_values(ProjectionBlockFormat::Q4K, 1024, 3072);
    values[0] = mixed_query(1024, 1024, 2048);
    let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
    assert_eq!(
        prepared
            .projection(ProjectionRole::CausalQuery)
            .unwrap()
            .output_features(),
        3072
    );
    for (component, format, n, mmq) in [
        ("component.2.0", ProjectionBlockFormat::Q4K, 1024, false),
        ("component.6.0", ProjectionBlockFormat::Q5K, 2048, true),
    ] {
        let mut facts = geometry_facts(format, 1024, n);
        facts.component_id = WeightId::new(component).unwrap();
        facts.output_stride = 3075;
        facts.output_available_bytes = 8 * facts.output_stride * 2;
        let native = geometry_native(1024, n, mmq);
        let wave =
            PreparedUpstreamProjectionWave::prepare(&prepared, &facts, Some(&native)).unwrap();
        assert!(
            matches!(wave.route(), PreparedUpstreamProjectionRoute::Selected { arithmetic, .. }
            if *arithmetic == if mmq { UpstreamProjectionArithmetic::MmqDs4MarkerV2 } else { UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2 })
        );
    }
}
