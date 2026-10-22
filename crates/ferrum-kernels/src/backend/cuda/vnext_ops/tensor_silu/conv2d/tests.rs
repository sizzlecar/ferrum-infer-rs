use super::*;
use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};

fn tensor(dims: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        dims.to_vec(),
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn attributes(pad: [i64; 4], stride: [i64; 2]) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("padding").unwrap(),
            SemanticValue::Integers(pad.to_vec()),
        ),
        (
            AttributeId::new("stride").unwrap(),
            SemanticValue::Integers(stride.to_vec()),
        ),
        (
            AttributeId::new("dilation").unwrap(),
            SemanticValue::Integers(vec![1, 1]),
        ),
        (
            AttributeId::new("groups").unwrap(),
            SemanticValue::Unsigned(1),
        ),
        (
            AttributeId::new("padding_mode").unwrap(),
            SemanticValue::Text("explicit_zero".into()),
        ),
    ])
}

#[test]
fn explicit_padding_and_library_workspace_have_disjoint_checked_layout() {
    let g = Conv2dBf16Geometry::new(
        &tensor(&[2, 3, 4, 5]),
        &tensor(&[6, 3, 3, 3]),
        &tensor(&[6]),
        &tensor(&[2, 6, 3, 4]),
        &attributes([1, 2, 0, 1], [2, 1]),
    )
    .unwrap();
    let s = Shape::new(g, 1024).unwrap();
    assert_eq!(s.native.input, [2, 3, 7, 6]);
    assert_eq!(s.native.padding, [0, 0]);
    assert_eq!(
        (s.input_bytes, s.weight_bytes, s.bias_bytes, s.output_bytes),
        (240, 324, 12, 288)
    );
    assert_eq!(
        (s.padding_bytes, s.workspace_offset, s.scratch_bytes),
        (504, 512, 1536)
    );
    assert_eq!((s.elements, s.operations), (144, 7776));
    assert_eq!(s.native.stride, [2, 1]);
}

#[test]
fn no_padding_and_zero_workspace_require_no_scratch() {
    let g = Conv2dBf16Geometry::new(
        &tensor(&[2, 3, 4, 5]),
        &tensor(&[6, 3, 1, 1]),
        &tensor(&[6]),
        &tensor(&[2, 6, 4, 5]),
        &attributes([0; 4], [1, 1]),
    )
    .unwrap();
    let s = Shape::new(g, 0).unwrap();
    assert_eq!(
        (s.padding_bytes, s.workspace_offset, s.scratch_bytes),
        (0, 0, 0)
    );
    let mut attrs = attributes([0; 4], [1, 1]);
    attrs.insert(
        AttributeId::new("padding_mode").unwrap(),
        SemanticValue::Text("implicit_zero".into()),
    );
    let g = Conv2dBf16Geometry::new(
        &tensor(&[2, 3, 4, 5]),
        &tensor(&[6, 3, 1, 1]),
        &tensor(&[6]),
        &tensor(&[2, 6, 4, 5]),
        &attrs,
    )
    .unwrap();
    assert!(Shape::new(g, 0).is_err());
}

#[test]
fn native_dimension_and_scratch_capacity_overflow_fail_without_allocating() {
    let x = tensor(&[1, 1, i32::MAX as u64 + 1, 1]);
    let g = Conv2dBf16Geometry::new(
        &x,
        &tensor(&[1, 1, 1, 1]),
        &tensor(&[1]),
        &x,
        &attributes([0; 4], [1, 1]),
    )
    .unwrap();
    assert!(Shape::new(g, 0).is_err());
    assert!(scratch_layout(u64::MAX, 0).is_err());
    assert!(scratch_layout(2, isize::MAX as u64).is_err());
    assert_eq!(scratch_layout(2, 0).unwrap(), (256, 256));
    assert_eq!(scratch_layout(0, 256).unwrap(), (0, 256));
}
