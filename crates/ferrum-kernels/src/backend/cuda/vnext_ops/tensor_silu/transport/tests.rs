//! Pure provider preflight and ABI tests. No second copy implementation is
//! substituted for the managed CUDA execution path.

use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, DeviceClass, DeviceId, ProgramValueId, ResolvedTensorLayout,
    ResolvedTensorSpec, ResolvedValueStorage, ResourceId, TensorAccess,
};

fn tensor(shape: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        shape.to_vec(),
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}

fn value(
    name: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    tensor: ResolvedTensorSpec,
) -> ResolvedValueBinding {
    let storage = ResolvedValueStorage::single(
        ResourceId::new(format!("resource.{name}")).unwrap(),
        0,
        tensor.minimum_storage_bytes().unwrap(),
        tensor.element_type(),
    )
    .unwrap();
    ResolvedValueBinding::new(
        ProgramValueId::new(name).unwrap(),
        role,
        ordinal,
        tensor,
        if role == ResolvedValueRole::Input {
            TensorAccess::Read
        } else {
            TensorAccess::Write
        },
        AliasPolicy::NoAlias,
        BufferUsage::Activations,
        None,
        storage,
    )
    .unwrap()
}

fn gather_values(source: &[u64], output: &[u64]) -> Vec<ResolvedValueBinding> {
    vec![
        value("source", ResolvedValueRole::Input, 0, tensor(source)),
        value("output", ResolvedValueRole::Output, 0, tensor(output)),
    ]
}

fn attr(name: &str) -> AttributeId {
    AttributeId::new(name).unwrap()
}

fn gather_attrs() -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (attr("batch"), SemanticValue::Unsigned(2)),
        (attr("column_start"), SemanticValue::Unsigned(1)),
        (attr("column_count"), SemanticValue::Unsigned(2)),
        (attr("run_lengths"), SemanticValue::Integers(vec![2, 1, 2])),
        (
            attr("source_starts"),
            SemanticValue::Integers(vec![4, 0, 1]),
        ),
        (
            attr("source_batch_strides"),
            SemanticValue::Integers(vec![0, 3, 3]),
        ),
        (
            attr("source_row_strides"),
            SemanticValue::Integers(vec![1, 0, 1]),
        ),
    ])
}

#[test]
fn gather_run_abi_is_bounded_and_preserves_broadcast_and_batch_geometry() {
    let shape = geometry(
        Kind::RowGather,
        &gather_values(&[6, 4], &[10, 2]),
        &gather_attrs(),
    )
    .unwrap();
    assert_eq!(
        (shape.input_bytes, shape.output_bytes, shape.elements),
        ([48, 0], 40, 20)
    );
    let Parameters::Gather(runs) = shape.parameters else {
        panic!("gather parameters");
    };
    assert_eq!(
        (
            runs.run_count,
            runs.rows_per_batch,
            runs.source_width,
            runs.column_start,
            runs.column_count
        ),
        (3, 5, 4, 1, 2)
    );
    assert_eq!(&runs.run_ends[..3], &[2, 3, 5]);
    assert_eq!(&runs.source_starts[..3], &[4, 0, 1]);
    assert_eq!(&runs.source_batch_strides[..3], &[0, 3, 3]);
    assert_eq!(&runs.source_row_strides[..3], &[1, 0, 1]);
    assert!(runs.run_ends[3..].iter().all(|&value| value == 0));
    assert_eq!(runs.padding, 0);
    // Actual ABI contract, including its explicit alignment field. The full
    // parameter list additionally has two CUDA pointers and an element count.
    assert_eq!(std::mem::size_of::<NativeGather>(), 1064);
    assert_eq!(std::mem::align_of::<NativeGather>(), 8);
    assert!(std::mem::size_of::<NativeGather>() + 3 * 8 < 4096);
}

#[test]
fn gather_rejects_source_overrun_column_overrun_invalid_stride_and_port_shapes() {
    let values = gather_values(&[6, 4], &[10, 2]);
    for (key, wrong) in [
        ("source_starts", SemanticValue::Integers(vec![5, 0, 1])),
        (
            "source_batch_strides",
            SemanticValue::Integers(vec![0, i64::MAX, 3]),
        ),
        ("source_row_strides", SemanticValue::Integers(vec![2, 0, 1])),
        ("run_lengths", SemanticValue::Integers(vec![2, -1, 2])),
        ("column_start", SemanticValue::Unsigned(3)),
        ("column_count", SemanticValue::Unsigned(0)),
        ("batch", SemanticValue::Unsigned(u64::MAX)),
    ] {
        let mut attrs = gather_attrs();
        attrs.insert(attr(key), wrong);
        assert!(geometry(Kind::RowGather, &values, &attrs).is_err(), "{key}");
    }
    assert!(geometry(
        Kind::RowGather,
        &gather_values(&[6, 4], &[5, 4]),
        &gather_attrs()
    )
    .is_err());
    let mut wrong_dtype = values.clone();
    wrong_dtype[0] = value(
        "wrong",
        ResolvedValueRole::Input,
        0,
        ResolvedTensorSpec::new(
            vec![6, 4],
            ElementType::F16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
    );
    assert!(geometry(Kind::RowGather, &wrong_dtype, &gather_attrs()).is_err());
    let mut duplicate = values.clone();
    duplicate[1] = duplicate[0].clone();
    assert!(geometry(Kind::RowGather, &duplicate, &gather_attrs()).is_err());
    assert!(geometry(Kind::RowGather, &values[..1], &gather_attrs()).is_err());
    let mut unknown = gather_attrs();
    unknown.insert(attr("hidden_size"), SemanticValue::Unsigned(4));
    assert!(geometry(Kind::RowGather, &values, &unknown).is_err());
    // The whole source and output storage must be disjoint, even if only a
    // source sub-column is selected. Read/read overlap is not rejected here.
    assert!(nonoverlapping_ranges(1024, 48, 1024, 40).is_err());
    assert!(nonoverlapping_ranges(1024, 48, 1066, 40).is_err());
    assert!(nonoverlapping_ranges(1024, 48, 1072, 40).is_ok());
}

#[test]
fn gather_run_count_limit_is_an_actual_parameter_boundary() {
    let mut attrs = gather_attrs();
    attrs.insert(attr("batch"), SemanticValue::Unsigned(1));
    attrs.insert(attr("column_start"), SemanticValue::Unsigned(0));
    attrs.insert(attr("column_count"), SemanticValue::Unsigned(1));
    for count in [32, 33] {
        attrs.insert(attr("run_lengths"), SemanticValue::Integers(vec![1; count]));
        for key in [
            "source_starts",
            "source_batch_strides",
            "source_row_strides",
        ] {
            attrs.insert(attr(key), SemanticValue::Integers(vec![0; count]));
        }
        let result = geometry(
            Kind::RowGather,
            &gather_values(&[1, 1], &[count as u64, 1]),
            &attrs,
        );
        assert_eq!(result.is_ok(), count == 32);
    }
}

#[test]
fn concat_interleaves_batches_instead_of_flat_concatenation_and_rejects_broadcast() {
    let values = vec![
        value("left", ResolvedValueRole::Input, 0, tensor(&[4, 3])),
        value("right", ResolvedValueRole::Input, 1, tensor(&[6, 3])),
        value("output", ResolvedValueRole::Output, 0, tensor(&[10, 3])),
    ];
    let attrs = BTreeMap::from([(attr("batch"), SemanticValue::Unsigned(2))]);
    let shape = geometry(Kind::BatchedRowConcat, &values, &attrs).unwrap();
    assert_eq!(
        (shape.input_bytes, shape.output_bytes, shape.elements),
        ([24, 36], 60, 30)
    );
    let Parameters::Concat {
        left_rows,
        right_rows,
        columns,
    } = shape.parameters
    else {
        panic!("concat parameters");
    };
    assert_eq!((left_rows, right_rows, columns), (2, 3, 3));
    for batch in [0, 3] {
        assert!(geometry(
            Kind::BatchedRowConcat,
            &values,
            &BTreeMap::from([(attr("batch"), SemanticValue::Unsigned(batch))])
        )
        .is_err());
    }
    let mut wrong = values.clone();
    wrong[2] = value("wrong", ResolvedValueRole::Output, 0, tensor(&[6, 5]));
    assert!(geometry(Kind::BatchedRowConcat, &wrong, &attrs).is_err());
    wrong = values.clone();
    wrong[1] = value("wrong", ResolvedValueRole::Input, 1, tensor(&[6, 1]));
    assert!(geometry(Kind::BatchedRowConcat, &wrong, &attrs).is_err());
    assert!(geometry(Kind::BatchedRowConcat, &values, &BTreeMap::new()).is_err());
}

#[test]
fn transport_descriptors_require_capability_and_only_dense_weight_format() {
    let config =
        super::super::super::cuda_tensor_runtime_config(0, DeviceId::new("device.cuda.0").unwrap())
            .unwrap();
    let d = DeviceDescriptor {
        id: config.device_id,
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 4096,
        runtime_implementation_fingerprint: config.runtime_implementation_fingerprint,
        capabilities: config.capabilities,
        dynamic_storage_profiles: config.dynamic_storage_profiles,
    };
    for kind in [Kind::RowGather, Kind::BatchedRowConcat] {
        let selected = descriptor(&d, kind).unwrap();
        assert!(selected.supports_fixed_tensor_work());
        assert_eq!(selected.operation_id().as_str(), kind.operation());
        assert!(selected.accepted_quantization_formats().is_empty());
        assert_eq!(
            selected.accepted_weight_formats(),
            &BTreeSet::from([WeightFormatId::new("weight-format.safetensors.dense").unwrap()])
        );
        let mut missing = d.clone();
        missing
            .capabilities
            .remove(&CapabilityId::new(kind.capability()).unwrap());
        assert!(descriptor(&missing, kind).is_err());
    }
    let huge = tensor(&[1, isize::MAX as u64 / 2 + 1]);
    assert!(bounded_bytes(&huge).is_err());
}
