use super::*;
use crate::backend::HostDtype;
use std::fmt::Debug;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn actual_typed_copy<T: HostDtype + Debug + PartialEq>(source: &[T], initial: &[T]) {
    let src = MetalBackend::from_slice_typed(source);
    let mut dst = MetalBackend::from_slice_typed(initial);
    let mut expected = initial.to_vec();
    expected[2..5].copy_from_slice(&source[1..4]);
    let mut ctx = MetalBackend::new_context(&Default::default()).expect("backend context");
    MetalBackend::copy_slice(&mut ctx, &src, 1, &mut dst, 2, 3);
    MetalBackend::sync_checked(&mut ctx).expect("actual Metal copy must complete");
    assert_eq!(dst.dtype(), T::DTYPE);
    // Read all elements, including both untouched canaries, after the real
    // completion. This inspection adds no synchronization or dtype conversion.
    let copied = unsafe { std::slice::from_raw_parts(dst.raw().contents().cast::<T>(), dst.len()) };
    let original =
        unsafe { std::slice::from_raw_parts(src.raw().contents().cast::<T>(), src.len()) };
    assert_eq!(copied, expected);
    assert_eq!(original, source);
}

#[test]
#[ignore = "requires an actual Metal device"]
fn copy_slice_real_metal_f32_preserves_offsets_and_canaries() {
    actual_typed_copy(
        &[11.0_f32, -0.5, 3.25, -7.0, 29.0],
        &[101.0_f32, 102.0, 103.0, 104.0, 105.0, 106.0],
    );
}

#[test]
#[ignore = "requires an actual Metal device"]
fn copy_slice_real_metal_f16_preserves_offsets_and_canaries() {
    let source = [11.0_f32, -0.5, 3.25, -7.0, 29.0].map(f16::from_f32);
    let initial = [101.0_f32, 102.0, 103.0, 104.0, 105.0, 106.0].map(f16::from_f32);
    actual_typed_copy(&source, &initial);
}

#[test]
#[ignore = "requires an actual Metal device"]
fn copy_slice_real_metal_rejects_dtype_and_extent_errors_before_encoding() {
    let src = MetalBackend::from_slice_typed(&[1.0_f32, 2.0, 3.0, 4.0]);
    let mut wrong_dtype = MetalBackend::from_slice_typed(&[f16::from_f32(9.0); 4]);
    let mut ctx = MetalBackend::new_context(&Default::default()).expect("backend context");
    assert!(catch_unwind(AssertUnwindSafe(|| {
        MetalBackend::copy_slice(&mut ctx, &src, 0, &mut wrong_dtype, 0, 1);
    }))
    .is_err());
    assert!(MetalBackend::preparation_is_host_only(&ctx));
    let mut dst = MetalBackend::from_slice_typed(&[9.0_f32; 4]);
    for (src_offset, dst_offset, len) in [(4, 0, 1), (0, 4, 1), (usize::MAX, 0, 1)] {
        assert!(catch_unwind(AssertUnwindSafe(|| {
            MetalBackend::copy_slice(&mut ctx, &src, src_offset, &mut dst, dst_offset, len);
        }))
        .is_err());
        assert!(MetalBackend::preparation_is_host_only(&ctx));
    }
    // A valid empty range needs neither a command buffer nor a completion.
    MetalBackend::copy_slice(&mut ctx, &src, 4, &mut dst, 4, 0);
    assert!(MetalBackend::preparation_is_host_only(&ctx));
    assert_eq!(MetalBackend::to_vec(&dst, 4), vec![9.0; 4]);
}
