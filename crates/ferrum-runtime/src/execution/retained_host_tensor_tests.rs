use super::*;
use crate::{ElementType, HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};

fn spec(elements: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![elements],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}

#[test]
fn output_input_view_preserves_whole_backing_and_refund_after_scope_close() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            96,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 96,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let scope = host
        .reserve_scope(&[(HostMemoryClass::Output, 96)])
        .unwrap();
    let original = scope
        .account()
        .allocate_zeroed(HostMemoryClass::Output, 96)
        .unwrap()
        .freeze();
    let slice = original.slice(32..64).unwrap();
    let id = ProgramValueId::new("value.next.input").unwrap();
    assert!(ManagedHostTensor::new(id.clone(), spec(16), slice.clone()).is_err());
    let input = ManagedHostTensor::from_retained_output(id, spec(16), slice).unwrap();
    assert_eq!(
        input.bytes().bytes().as_ptr(),
        original.bytes()[32..].as_ptr()
    );
    assert_eq!(input.bytes().class(), HostMemoryClass::Output);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 96);
    drop(original);
    scope.close().unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 96);
    assert!(host.allocate_zeroed(HostMemoryClass::Output, 1).is_err());
    assert_eq!(input.bytes().bytes(), &[0; 32]);
    drop(input);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn retained_output_constructor_cannot_relabel_input_or_change_extent() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            16,
            HostMemoryClassLimits {
                input_preprocess_bytes: 8,
                output_bytes: 8,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let id = ProgramValueId::new("value.next.input").unwrap();
    let input = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
        .unwrap()
        .freeze();
    assert!(ManagedHostTensor::from_retained_output(id.clone(), spec(4), input).is_err());
    let output = host
        .allocate_zeroed(HostMemoryClass::Output, 8)
        .unwrap()
        .freeze();
    assert!(ManagedHostTensor::from_retained_output(id.clone(), spec(3), output.clone()).is_err());
    let view = ManagedHostTensor::from_retained_output(id, spec(4), output).unwrap();
    assert_eq!(host.usage().unwrap().allocated_bytes(), 8);
    drop(view);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
