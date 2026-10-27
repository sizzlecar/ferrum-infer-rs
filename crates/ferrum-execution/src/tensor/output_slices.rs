use super::*;

#[test]
fn batch_views_keep_the_entire_physical_output_charged_until_last_release() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            32,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 32,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut backing = host.allocate_zeroed(HostMemoryClass::Output, 32).unwrap();
    for (i, value) in backing.as_mut().iter_mut().enumerate() {
        *value = i as u8;
    }
    let source = TensorExecutionOutput {
        value_id: ProgramValueId::new("output.test.rgba").unwrap(),
        tensor: ResolvedTensorSpec::new(
            vec![2, 4, 1, 2],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes: backing.freeze(),
        timing: None,
    };
    assert!(source.batch_item(2).is_err());
    let first = source.batch_item(0).unwrap();
    let second = source.batch_item(1).unwrap();
    assert_eq!(first.tensor().dimensions(), [1, 4, 1, 2]);
    assert_eq!(first.bytes().bytes(), &(0u8..16).collect::<Vec<_>>());
    assert_eq!(second.bytes().bytes(), &(16u8..32).collect::<Vec<_>>());
    assert!(host.owns(first.bytes()) && host.owns(second.bytes()));
    drop(source);
    drop(first);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 32);
    assert!(host.allocate_zeroed(HostMemoryClass::Output, 1).is_err());
    let retained = second.clone();
    drop(second);
    assert_eq!(host.usage().unwrap().committed_bytes(), 32);
    drop(retained);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(host.allocate_zeroed(HostMemoryClass::Output, 32).unwrap());
}
