use super::*;
use crate::SafetensorsArchive;
use ferrum_runtime::{
    CanonicalRational, ElementType, HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits,
    HostMemoryLimits, WeightComponentRole, WeightEncoding,
};
use half::bf16;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

struct Tensor {
    name: &'static str,
    shape: Vec<usize>,
    dtype: Dtype,
    bytes: Vec<u8>,
}

fn bf(name: &'static str, shape: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        name,
        shape: shape.to_vec(),
        dtype: Dtype::BF16,
        bytes: values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect(),
    }
}

fn fp32(name: &'static str, shape: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        name,
        shape: shape.to_vec(),
        dtype: Dtype::F32,
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn archive(tensors: &[Tensor]) -> (tempfile::TempDir, SafetensorsArchive) {
    let directory = tempfile::tempdir().unwrap();
    let views = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name,
                TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
    let original = SafetensorsArchive::open(directory.path()).unwrap();
    (directory, original)
}

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}

fn limits() -> ProcessMappedDenseLimits {
    ProcessMappedDenseLimits {
        maximum_file_bytes: 1 << 20,
        maximum_metadata_bytes: 1 << 16,
    }
}

fn component(
    id: &str,
    names: &[&str],
    shape: &[u64],
    element_type: ElementType,
) -> WeightComponentSpec {
    WeightComponentSpec {
        id: id.to_owned().try_into().unwrap(),
        role: WeightComponentRole::Values,
        external_names: names.iter().map(|name| (*name).to_owned()).collect(),
        dimensions: shape.to_vec(),
        encoding: WeightEncoding::Dense { element_type },
        required: true,
    }
}

fn affine(
    mut spec: WeightComponentSpec,
    scale: (i64, u64),
    bias: (i64, u64),
) -> WeightComponentSpec {
    spec.encoding = WeightEncoding::DenseAffine {
        element_type: spec.physical_element_type(),
        scale: CanonicalRational::new(scale.0, scale.1).unwrap(),
        bias: CanonicalRational::new(bias.0, bias.1).unwrap(),
    };
    spec
}

#[test]
fn exact_normalization_preserves_composites_affine_once_f32_and_original_lineage() {
    let tensors = [
        bf("q", &[2, 2], &[1., -0., 0.5, -2.]),
        bf("k", &[1, 2], &[3., 4.]),
        bf("v", &[1, 2], &[5., 6.]),
        bf("z", &[1, 2], &[7., 8.]),
        fp32(
            "up",
            &[2, 2],
            &[1.00048828125, 1.00146484375, -1.00048828125, -1.00146484375],
        ),
        bf("norm", &[2], &[1., -1.]),
        bf("mandatory_f32", &[2], &[0.125, -0.]),
        fp32("raw_f32", &[2], &[f32::from_bits(0x3f80_0001), -0.]),
        fp32("cancel", &[1], &[16_777_216.]),
    ];
    let (_original_directory, original) = archive(&tensors);
    let specs = vec![
        component(
            "component.gdn",
            &["q", "k", "v", "z"],
            &[5, 2],
            ElementType::F16,
        ),
        component(
            "component.mlp_concat",
            &["up", "q"],
            &[4, 2],
            ElementType::F16,
        ),
        component(
            "component.mlp_stack",
            &["q", "up"],
            &[2, 2, 2],
            ElementType::F16,
        ),
        affine(
            component("component.norm", &["norm"], &[2], ElementType::F16),
            (1, 1),
            (1, 1),
        ),
        component(
            "component.mandatory_f32",
            &["mandatory_f32"],
            &[2],
            ElementType::F32,
        ),
        component("component.raw_f32", &["raw_f32"], &[2], ElementType::F32),
        affine(
            component("component.cancel", &["cancel"], &[1], ElementType::F32),
            (16_777_217, 16_777_216),
            (-16_777_216, 1),
        ),
    ];
    let output = tempfile::tempdir().unwrap();
    let h = host();
    let cached =
        ProcessMappedDenseWeights::build(&original, &specs, &h, output.path(), limits()).unwrap();
    assert!(cached.matches_components(&specs));
    let original_source = ManagedDenseArchiveSource::new(&original, &h);
    for spec in &specs {
        let expected = original_source.component(spec).unwrap();
        let actual = cached.component(spec).unwrap();
        assert_eq!(actual.bytes(), expected.bytes(), "{}", spec.id);
        assert_eq!(actual.component_id(), expected.component_id());
        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(actual.element_type(), expected.element_type());
        assert_eq!(actual.external_names(), expected.external_names());
        assert_eq!(actual.source_files(), expected.source_files());
        let retained = actual.retained_host_memory().unwrap();
        assert!(retained.readonly_file_mapping().is_some());
        assert_eq!(retained.managed_memory_class(), None);
        assert!(std::ptr::eq(
            actual.bytes().as_ptr(),
            retained.bytes().as_ptr()
        ));
        let staged = cached
            .component_with_upload_policy(spec, ProcessMappedDenseUploadPolicy::Staged)
            .unwrap();
        assert_eq!(staged.bytes(), actual.bytes());
        assert_eq!(staged.component_id(), actual.component_id());
        assert_eq!(staged.external_names(), actual.external_names());
        assert_eq!(staged.source_files(), actual.source_files());
        assert_eq!(staged.dimensions(), actual.dimensions());
        assert_eq!(staged.element_type(), actual.element_type());
        assert!(std::ptr::eq(
            staged.bytes().as_ptr(),
            actual.bytes().as_ptr()
        ));
        let staged_owner = staged.retained_host_memory().unwrap();
        assert!(staged_owner.readonly_file_mapping().is_none());
        assert_eq!(staged_owner.managed_memory_class(), None);
        assert!(std::ptr::eq(
            staged_owner.owner_bytes().as_ptr(),
            retained.owner_bytes().as_ptr()
        ));
    }
    let norm = cached.component(&specs[3]).unwrap();
    assert_eq!(
        norm.bytes(),
        [0x4000u16, 0]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        cached.component(&specs[6]).unwrap().bytes(),
        1.0f32.to_le_bytes()
    );
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(
        cached.artifact_sha256(),
        format!("{:x}", Sha256::digest(&cached.mapping.bytes))
    );
    assert_eq!(
        cached.requirements_report().file_bytes() as usize,
        cached.mapping.bytes.len()
    );
    cached
        .layout
        .validate_mapping(&cached.mapping.bytes)
        .unwrap();
}

#[test]
fn one_component_physical_credit_streams_all_groups_short_refunds_then_exact_retries() {
    let (_original_directory, original) = archive(&[
        bf("first", &[2, 2], &[1., 2., 3., 4.]),
        bf("second", &[3, 2], &[5., 6., 7., 8., 9., 10.]),
    ]);
    let specs = [
        component("component.first", &["first"], &[2, 2], ElementType::F16),
        component("component.second", &["second"], &[3, 2], ElementType::F16),
    ];
    let output = tempfile::tempdir().unwrap();
    let h = host();
    let report = ProcessMappedDenseWeights::requirements(&specs, limits()).unwrap();
    let peak = h
        .backing_bytes(report.maximum_component_bytes() as usize)
        .unwrap();
    assert!(
        peak < specs
            .iter()
            .map(|c| h
                .backing_bytes(c.physical_bytes().unwrap() as usize)
                .unwrap())
            .sum::<usize>()
    );
    let short = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, peak - 1)])
        .unwrap();
    assert!(ProcessMappedDenseWeights::build(
        &original,
        &specs,
        &short.account(),
        output.path(),
        limits()
    )
    .is_err());
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(h.usage().unwrap().committed_bytes(), peak - 1);
    short.close().unwrap();
    let exact = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, peak)])
        .unwrap();
    let view = exact.account();
    let cached =
        ProcessMappedDenseWeights::build(&original, &specs, &view, output.path(), limits())
            .unwrap();
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(h.usage().unwrap().committed_bytes(), peak);
    exact.close().unwrap();
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let payload = cached.component(&specs[1]).unwrap();
    let retained = payload.retained_host_memory().unwrap().clone();
    let expected = payload.bytes().to_vec();
    drop(payload);
    let weak = Arc::downgrade(&cached.mapping);
    drop(cached);
    drop(original);
    assert!(weak.upgrade().is_some());
    assert_eq!(retained.bytes(), expected);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    drop(retained);
    assert!(weak.upgrade().is_none());
}

#[test]
fn exact_file_limit_metadata_cap_and_io_failure_publish_no_source_or_host_owner() {
    let (_original_directory, original) = archive(&[bf("a", &[1, 2], &[1., -0.])]);
    let specs = [component("component.a", &["a"], &[1, 2], ElementType::F16)];
    let output = tempfile::tempdir().unwrap();
    let h = host();
    let planned = ProcessMappedDenseWeights::requirements(&specs, limits()).unwrap();
    let short = ProcessMappedDenseLimits {
        maximum_file_bytes: planned.file_bytes() - 1,
        ..limits()
    };
    assert!(ProcessMappedDenseWeights::requirements(&specs, short).is_err());
    assert!(ProcessMappedDenseWeights::build(&original, &specs, &h, output.path(), short).is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let exact = ProcessMappedDenseLimits {
        maximum_file_bytes: planned.file_bytes(),
        ..limits()
    };
    let cached =
        ProcessMappedDenseWeights::build(&original, &specs, &h, output.path(), exact).unwrap();
    let full_metadata = cached.requirements_report().metadata_bytes();
    let metadata_short = ProcessMappedDenseLimits {
        maximum_metadata_bytes: full_metadata - 1,
        ..exact
    };
    assert!(
        ProcessMappedDenseWeights::build(&original, &specs, &h, output.path(), metadata_short)
            .is_err()
    );
    let metadata_exact = ProcessMappedDenseLimits {
        maximum_metadata_bytes: full_metadata,
        ..exact
    };
    assert!(
        ProcessMappedDenseWeights::build(&original, &specs, &h, output.path(), metadata_exact)
            .is_ok()
    );
    assert!(ProcessMappedDenseWeights::build(
        &original,
        &specs,
        &h,
        &output.path().join("missing"),
        exact
    )
    .is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    drop(cached);
    // An incomplete anonymous artifact never leaves a reusable pathname.
    assert!(std::fs::read_dir(output.path()).unwrap().next().is_none());
}

#[test]
fn late_conversion_error_never_publishes_ready_and_refunds_before_retry() {
    let (_original_directory, original) = archive(&[
        bf("a", &[1, 2], &[1., -0.]),
        bf("z", &[1, 2], &[2., 65_536.]),
    ]);
    let specs = [
        component("component.a", &["a"], &[1, 2], ElementType::F16),
        component("component.z", &["z"], &[1, 2], ElementType::F16),
    ];
    let output = tempfile::tempdir().unwrap();
    let h = host();
    assert!(
        ProcessMappedDenseWeights::build(&original, &specs, &h, output.path(), limits()).is_err()
    );
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    assert!(std::fs::read_dir(output.path()).unwrap().next().is_none());
    let mut retry = specs;
    retry[1].encoding = WeightEncoding::Dense {
        element_type: ElementType::F32,
    };
    let cached =
        ProcessMappedDenseWeights::build(&original, &retry, &h, output.path(), limits()).unwrap();
    assert_eq!(
        cached.component(&retry[1]).unwrap().bytes(),
        [2., 65_536.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn source_contract_mismatch_and_invalid_layouts_fail_without_new_host_allocations() {
    let (_original_directory, original) =
        archive(&[bf("a", &[1, 2], &[1., 2.]), bf("b", &[1, 2], &[3., 4.])]);
    let spec = component("component.a", &["a", "b"], &[2, 2], ElementType::F16);
    let output = tempfile::tempdir().unwrap();
    let h = host();
    let cached = ProcessMappedDenseWeights::build(
        &original,
        std::slice::from_ref(&spec),
        &h,
        output.path(),
        limits(),
    )
    .unwrap();
    let mut changed = spec.clone();
    for variant in 0..5 {
        match variant {
            0 => changed.id = "component.foreign".to_owned().try_into().unwrap(),
            1 => changed.dimensions = vec![1, 4],
            2 => changed.external_names.reverse(),
            3 => {
                changed.encoding = WeightEncoding::Dense {
                    element_type: ElementType::F32,
                }
            }
            _ => changed.required = false,
        }
        assert!(!cached.matches_components(std::slice::from_ref(&changed)));
        assert!(cached.component(&changed).is_err());
        assert!(cached
            .component_with_upload_policy(&changed, ProcessMappedDenseUploadPolicy::Staged)
            .is_err());
        changed = spec.clone();
    }
    for invalid_spec in [
        component("component.empty", &[], &[2], ElementType::F16),
        component("component.zero", &["a"], &[0, 2], ElementType::F16),
        component(
            "component.overflow",
            &["a"],
            &[u64::MAX, 2],
            ElementType::F32,
        ),
        component("component.integer", &["a"], &[1, 2], ElementType::I8),
        component("__metadata__", &["a"], &[1, 2], ElementType::F16),
    ] {
        assert!(ProcessMappedDenseWeights::requirements(&[invalid_spec], limits()).is_err());
    }
    assert!(ProcessMappedDenseWeights::requirements(&[], limits()).is_err());
    assert!(ProcessMappedDenseWeights::requirements(&[spec.clone(), spec], limits()).is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn malformed_source_shape_and_missing_component_do_not_leave_a_partial_artifact() {
    let (_original_directory, original) =
        archive(&[bf("a", &[1, 2], &[1., 2.]), bf("b", &[2, 1], &[3., 4.])]);
    let h = host();
    let output = tempfile::tempdir().unwrap();
    for spec in [
        component("component.bad_axis", &["a", "b"], &[2, 2], ElementType::F16),
        component("component.missing", &["absent"], &[1, 2], ElementType::F16),
        affine(
            component(
                "component.affine_composite",
                &["a", "a"],
                &[2, 2],
                ElementType::F16,
            ),
            (1, 1),
            (1, 1),
        ),
    ] {
        assert!(
            ProcessMappedDenseWeights::build(&original, &[spec], &h, output.path(), limits())
                .is_err()
        );
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
    assert!(std::fs::read_dir(output.path()).unwrap().next().is_none());
}

#[test]
fn upload_policy_defaults_preserve_mapping_authority_and_staged_owner_outlives_source() {
    assert_eq!(
        ProcessMappedDenseUploadPolicy::default(),
        ProcessMappedDenseUploadPolicy::ReadOnlyMapping
    );
    for (wire, policy) in [
        (
            "read_only_mapping",
            ProcessMappedDenseUploadPolicy::ReadOnlyMapping,
        ),
        ("staged", ProcessMappedDenseUploadPolicy::Staged),
    ] {
        assert_eq!(serde_json::to_value(policy).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<ProcessMappedDenseUploadPolicy>(serde_json::json!(wire))
                .unwrap(),
            policy
        );
    }
    assert!(
        serde_json::from_value::<ProcessMappedDenseUploadPolicy>(serde_json::json!(
            "implicit_pinned"
        ))
        .is_err()
    );
    let (original_directory, original) = archive(&[bf("a", &[1, 2], &[1.125, -0.])]);
    let spec = component("component.a", &["a"], &[1, 2], ElementType::F16);
    let output = tempfile::tempdir().unwrap();
    let h = host();
    let cached = ProcessMappedDenseWeights::build(
        &original,
        std::slice::from_ref(&spec),
        &h,
        output.path(),
        limits(),
    )
    .unwrap();
    let weak = Arc::downgrade(&cached.mapping);
    let mapped = cached.component(&spec).unwrap();
    let staged = cached
        .component_with_upload_policy(&spec, ProcessMappedDenseUploadPolicy::Staged)
        .unwrap();
    let mapped_owner = mapped.retained_host_memory().unwrap().clone();
    let staged_owner = staged.retained_host_memory().unwrap().clone();
    assert!(mapped_owner.readonly_file_mapping().is_some());
    assert!(staged_owner.readonly_file_mapping().is_none());
    assert_eq!(staged_owner.managed_memory_class(), None);
    assert!(std::ptr::eq(
        mapped_owner.owner_bytes().as_ptr(),
        staged_owner.owner_bytes().as_ptr()
    ));
    drop(mapped);
    drop(staged);
    drop(cached);
    drop(original);
    original_directory.close().unwrap();
    output.close().unwrap();
    drop(mapped_owner);
    assert!(weak.upgrade().is_some());
    assert_eq!(
        staged_owner.bytes(),
        [0x3c80u16, 0x8000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    drop(staged_owner);
    assert!(weak.upgrade().is_none());
}
