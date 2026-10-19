//! Actual native retains, exact mapping ranges, and quantized execution.

use super::*;
use crate::backend::GgufQuantType;
use crate::quant_linear::metal_gguf::MetalGgufLinear;
use crate::Linear;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device as CandleDevice, Tensor};
use ferrum_runtime::StableHostMemory;
use metal::objc::rc::autoreleasepool;
use std::ptr::NonNull;
use std::sync::mpsc;
use std::time::Duration;

struct PageMemory {
    pointer: NonNull<u8>,
    mapped_len: usize,
    readable_len: usize,
    released: mpsc::Sender<()>,
}

struct MappingRelease {
    owner: Weak<PageMemory>,
    released: mpsc::Receiver<()>,
    observed: bool,
}

impl MappingRelease {
    fn assert_released(&mut self, case: &str) {
        // Command completion and draining this thread's autorelease pool do
        // not prove that Metal has disposed its final native buffer retain.
        // Observe the actual host owner's destructor, without polling or
        // submitting unrelated GPU work to trigger collection.
        if !self.observed {
            self.released
                .recv_timeout(Duration::from_secs(10))
                .unwrap_or_else(|error| {
                    panic!(
                        "{case}: mapped owner was not destroyed: {error}; strong_count={}",
                        self.owner.strong_count()
                    )
                });
            self.observed = true;
        }
        assert!(
            self.owner.upgrade().is_none(),
            "{case}: destruction notification must coincide with final owner release"
        );
    }
}

impl PageMemory {
    fn new(readable_len: usize, payloads: &[(usize, &[u8])]) -> (Arc<Self>, MappingRelease) {
        let page = retained_memory::native_page_size().unwrap();
        let mapped_len = readable_len.div_ceil(page) * page;
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(pointer, libc::MAP_FAILED);
        let pointer = NonNull::new(pointer.cast::<u8>()).unwrap();
        for &(offset, payload) in payloads {
            assert!(offset + payload.len() <= readable_len);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    payload.as_ptr(),
                    pointer.as_ptr().add(offset),
                    payload.len(),
                );
            }
        }
        let (released, receiver) = mpsc::channel();
        let owner = Arc::new(Self {
            pointer,
            mapped_len,
            readable_len,
            released,
        });
        let release = MappingRelease {
            owner: Arc::downgrade(&owner),
            released: receiver,
            observed: false,
        };
        (owner, release)
    }
}

impl Drop for PageMemory {
    fn drop(&mut self) {
        assert_eq!(
            unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.mapped_len) },
            0
        );
        // A disconnected observer means its test already unwound. Never
        // panic from a native deallocator because the receiver was dropped.
        let _ = self.released.send(());
    }
}

// SAFETY: the mapping is initialized before publication and never mutated;
// its stable readable range remains mapped until this owner is dropped.
unsafe impl StableHostMemory for PageMemory {
    fn stable_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.readable_len) }
    }
}
unsafe impl Send for PageMemory {}
unsafe impl Sync for PageMemory {}

struct QuantWeights {
    kind: GgufQuantType,
    bytes: Vec<u8>,
    dense: Tensor,
    n: usize,
    k: usize,
}

impl QuantWeights {
    fn new(kind: GgufQuantType, n: usize, seed: usize) -> Self {
        let k = 256;
        let weights: Vec<_> = (0..n * k)
            .map(|i| ((i * 37 + seed * 29) % 97 + 3) as f32 / 128.0)
            .collect();
        let dtype = match kind {
            GgufQuantType::Q4K => GgmlDType::Q4K,
            GgufQuantType::Q6K => GgmlDType::Q6K,
            _ => unreachable!(),
        };
        let tensor = Tensor::from_vec(weights, (n, k), &CandleDevice::Cpu).unwrap();
        let quant = QTensor::quantize(&tensor, dtype).unwrap();
        Self {
            kind,
            bytes: quant.data().unwrap().into_owned(),
            dense: quant.dequantize(&CandleDevice::Cpu).unwrap(),
            n,
            k,
        }
    }

    fn input_and_expected(&self, m: usize) -> (Vec<f32>, Vec<f32>) {
        let input: Vec<_> = (0..m * self.k)
            .map(|i| ((i * 5 + i / self.k * 17) % 31 + 4) as f32 / 512.0)
            .collect();
        let expected = Tensor::from_vec(input.clone(), (m, self.k), &CandleDevice::Cpu)
            .unwrap()
            .matmul(&self.dense.transpose(0, 1).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(expected
            .iter()
            .all(|value| value.is_finite() && *value > 0.5));
        (input, expected)
    }

    fn load(&self, bytes: &[u8], no_copy: bool) -> MetalGgufLinear {
        let store = quant::metal_load_quant_store_helper(
            self.kind,
            bytes,
            self.n,
            self.k,
            &MetalBackendConfig::default(),
        )
        .unwrap();
        let (buffer, offset, lease) = native_weight(&store);
        assert_eq!(lease, no_copy);
        let native_address = buffer.contents() as usize + offset as usize;
        if no_copy {
            assert_eq!(
                native_address,
                bytes.as_ptr() as usize,
                "a copied weight cannot satisfy the no-copy lifetime oracle"
            );
        } else {
            assert_ne!(native_address, bytes.as_ptr() as usize);
            assert_eq!(offset, 0);
            assert_eq!(
                buffer.length(),
                bytes.len() as u64,
                "exact fallback copy bytes"
            );
        }
        MetalGgufLinear {
            store,
            in_features: self.k,
            out_features: self.n,
        }
    }
}

fn native_weight(store: &MetalQuantStore) -> (&metal::Buffer, u64, bool) {
    match store {
        MetalQuantStore::Q4K {
            blocks,
            byte_offset,
            _mmap_lease,
            ..
        }
        | MetalQuantStore::Q6K {
            blocks,
            byte_offset,
            _mmap_lease,
            ..
        } => (blocks, *byte_offset, _mmap_lease.is_some()),
        _ => unreachable!(),
    }
}

fn encode(
    ctx: &mut MetalContext,
    linear: &MetalGgufLinear,
    weights: &QuantWeights,
    m: usize,
) -> (MetalBuf, MetalBuf, Vec<f32>) {
    let (input, expected) = weights.input_and_expected(m);
    let input = MetalBackend::from_slice_in_context(ctx, &input);
    let mut output = MetalBackend::from_slice_in_context(ctx, &vec![f32::NAN; expected.len()]);
    MetalBackend::begin_execution(ctx).unwrap();
    linear.forward(ctx, &input, &mut output, m);
    assert!(ctx.cmd.is_some() && ctx.encoder.is_some() && ctx.submitted.is_none());
    (input, output, expected)
}

fn complete(ctx: &mut MetalContext, output: &MetalBuf, expected: &[f32]) {
    MetalBackend::sync_checked(ctx).unwrap();
    let actual = MetalBackend::to_vec_in_context(ctx, output, expected.len());
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "index={index}: {actual} vs {expected}"
        );
        assert!(
            (actual - expected).abs() <= 1e-3 + 1e-3 * expected.abs(),
            "index={index}: {actual} vs {expected}"
        );
    }
}

fn register(owner: &Arc<PageMemory>) -> MetalMmapRegistration {
    register_gguf_mmap(
        RetainedHostMemoryRegion::new(owner.clone(), 0, owner.readable_len).unwrap(),
        &MetalBackendConfig::default(),
    )
    .unwrap()
}

#[test]
fn native_mmap_window_checks_padding_overflow_and_alignment() {
    use retained_memory::checked_native_window as window;
    let safe = window(4096, 8192, 4096 + 128, 4096, 4096).unwrap();
    assert_eq!((safe.start, safe.len, safe.byte_offset), (4096, 8192, 128));
    assert!(window(4096, 5000, 4096 + 128, 4800, 4096).is_none());
    assert!(window(4097, 8191, 4097, 16, 4096).is_none());
    assert!(window(4096, 4096, 4096, 0, 4096).is_none());
    assert!(window(4096, 4096, 4096, 16, 0).is_none());
    assert!(window(4096, 4096, 4096, 16, 3).is_none());
    assert!(window(usize::MAX - 8, 16, usize::MAX - 8, 4, 4096).is_none());
    assert!(window(0, usize::MAX, usize::MAX - 1, 4, 4096).is_none());
    assert!(window(0, usize::MAX, usize::MAX - 7, 4, 4096).is_none());
}

#[test]
#[ignore = "requires actual Metal no-copy Q4/Q6 dispatch and native retention"]
fn metal_no_copy_quant_weights_survive_drop_before_commit() {
    for kind in [GgufQuantType::Q4K, GgufQuantType::Q6K] {
        let mut release = autoreleasepool(|| {
            let weights = QuantWeights::new(kind, 8, 1);
            let page = retained_memory::native_page_size().unwrap();
            let offset = 256;
            let len = (offset + weights.bytes.len()).div_ceil(page) * page;
            let (owner, release) = PageMemory::new(len, &[(offset, &weights.bytes)]);
            let weak = Arc::downgrade(&owner);
            let registration = register(&owner);
            let entry = Arc::downgrade(&registration.entry);
            let linear = weights.load(
                &owner.stable_bytes()[offset..offset + weights.bytes.len()],
                true,
            );
            registration.commit_residency().unwrap();
            let mut ctx = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
            let (input, output, expected) = encode(&mut ctx, &linear, &weights, 1);
            // No retained raw clone here: the actual encoded command must be
            // sufficient after all source and Rust weight ownership disappears.
            drop((linear, registration, owner, weights));
            assert!(
                entry.upgrade().is_none(),
                "native callback must not retain residency entry"
            );
            assert!(
                weak.upgrade().is_some(),
                "encoded work must retain mapped bytes before commit"
            );
            complete(&mut ctx, &output, &expected);
            MetalBackend::retire_context(&mut ctx).unwrap();
            drop((ctx, input, output));
            release
        });
        release.assert_released(&format!("drop-before-commit kind={kind:?}"));
    }
}

#[test]
#[ignore = "requires actual native Metal buffer retains"]
fn metal_no_copy_raw_buffer_clone_outlives_context_and_weight() {
    for kind in [GgufQuantType::Q4K, GgufQuantType::Q6K] {
        let (weak, mut release, raw, offset, expected_bytes) = autoreleasepool(|| {
            let weights = QuantWeights::new(kind, 8, 2);
            let page = retained_memory::native_page_size().unwrap();
            let source_offset = 256;
            let len = (source_offset + weights.bytes.len()).div_ceil(page) * page;
            let (owner, release) = PageMemory::new(len, &[(source_offset, &weights.bytes)]);
            let weak = Arc::downgrade(&owner);
            let registration = register(&owner);
            let entry = Arc::downgrade(&registration.entry);
            let linear = weights.load(
                &owner.stable_bytes()[source_offset..source_offset + weights.bytes.len()],
                true,
            );
            registration.commit_residency().unwrap();
            let (raw, offset, _) = native_weight(&linear.store);
            let raw = raw.clone();
            let mut ctx = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
            let (input, output, expected) = encode(&mut ctx, &linear, &weights, 1);
            drop((linear, registration, owner));
            assert!(entry.upgrade().is_none());
            complete(&mut ctx, &output, &expected);
            MetalBackend::retire_context(&mut ctx).unwrap();
            drop((ctx, input, output));
            (weak, release, raw, offset, weights.bytes)
        });
        assert!(
            weak.upgrade().is_some(),
            "last raw native clone still owns host memory"
        );
        let actual = unsafe {
            std::slice::from_raw_parts(
                raw.contents().cast::<u8>().add(offset as usize),
                expected_bytes.len(),
            )
        };
        assert_eq!(actual, expected_bytes);
        autoreleasepool(|| drop(raw));
        release.assert_released(&format!("raw native clone kind={kind:?}"));
        assert!(
            weak.upgrade().is_none(),
            "raw clone release must not leave a residency cycle"
        );
    }
}

#[test]
#[ignore = "requires actual Metal two-instance quant execution"]
fn metal_no_copy_peer_survives_independent_and_shared_mapping_retirement() {
    for shared in [false, true] {
        let (weak_a, weak_b, mut release_a, release_b) = autoreleasepool(|| {
            let a = QuantWeights::new(GgufQuantType::Q4K, 64, 1);
            let b = QuantWeights::new(GgufQuantType::Q6K, 64, 9);
            let page = retained_memory::native_page_size().unwrap();
            let offset_a = 256;
            let stride = (256 + a.bytes.len().max(b.bytes.len())).div_ceil(page) * page;
            let offset_b = if shared { stride + 256 } else { 256 };
            let (owner_a, owner_b, mut release_a, release_b) = if shared {
                let (owner, release) =
                    PageMemory::new(stride * 2, &[(offset_a, &a.bytes), (offset_b, &b.bytes)]);
                (owner.clone(), owner, release, None)
            } else {
                let (owner_a, release_a) = PageMemory::new(stride, &[(offset_a, &a.bytes)]);
                let (owner_b, release_b) = PageMemory::new(stride, &[(offset_b, &b.bytes)]);
                (owner_a, owner_b, release_a, Some(release_b))
            };
            let weak_a = Arc::downgrade(&owner_a);
            let weak_b = Arc::downgrade(&owner_b);
            let registration_a = register(&owner_a);
            let registration_b = register(&owner_b);
            assert_eq!(
                Arc::ptr_eq(&registration_a.entry, &registration_b.entry),
                shared
            );
            let linear_a = a.load(
                &owner_a.stable_bytes()[offset_a..offset_a + a.bytes.len()],
                true,
            );
            let linear_b = b.load(
                &owner_b.stable_bytes()[offset_b..offset_b + b.bytes.len()],
                true,
            );
            registration_a.commit_residency().unwrap();
            registration_b.commit_residency().unwrap();
            let mut ctx_a = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
            let mut ctx_b = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
            drop((owner_a, owner_b, registration_b));
            // Whole-tile prefill and subsequent decode both retain the exact
            // existing packed weights; no altered numerical route is selected.
            autoreleasepool(|| {
                let (input_a, output_a, expected_a) = encode(&mut ctx_a, &linear_a, &a, 32);
                let (input_b, output_b, expected_b) = encode(&mut ctx_b, &linear_b, &b, 32);
                drop((linear_a, registration_a));
                complete(&mut ctx_a, &output_a, &expected_a);
                MetalBackend::retire_context(&mut ctx_a).unwrap();
                drop(ctx_a);
                complete(&mut ctx_b, &output_b, &expected_b);
                drop((input_a, output_a, input_b, output_b));
            });
            if shared {
                assert!(weak_a.upgrade().is_some());
            } else {
                release_a.assert_released("peer A kind=Q4K shared=false before B decode");
                assert!(weak_a.upgrade().is_none());
            }
            assert!(weak_b.upgrade().is_some());
            let (input, output, expected) = encode(&mut ctx_b, &linear_b, &b, 1);
            drop(linear_b);
            complete(&mut ctx_b, &output, &expected);
            MetalBackend::retire_context(&mut ctx_b).unwrap();
            drop((ctx_b, input, output));
            (weak_a, weak_b, release_a, release_b)
        });
        release_a.assert_released(&format!("peer A kind=Q4K shared={shared}"));
        if let Some(mut release_b) = release_b {
            release_b.assert_released("peer B kind=Q6K shared=false");
        }
        assert!(weak_a.upgrade().is_none());
        assert!(weak_b.upgrade().is_none());
    }
}

#[test]
#[ignore = "requires actual Metal copy fallback for an unprovable enclosing page"]
fn metal_registered_partial_page_copies_exact_tensor_and_releases_source() {
    for kind in [GgufQuantType::Q4K, GgufQuantType::Q6K] {
        autoreleasepool(|| {
            let weights = QuantWeights::new(kind, 8, 4);
            let offset = 256;
            // The physical mapping has pages, but its declared readable owner
            // ends at the tensor. The extra page tail is not part of the proof.
            let (owner, mut release) =
                PageMemory::new(offset + weights.bytes.len(), &[(offset, &weights.bytes)]);
            let weak = Arc::downgrade(&owner);
            let registration = register(&owner);
            let linear = weights.load(&owner.stable_bytes()[offset..], false);
            drop((registration, owner));
            assert!(
                weak.upgrade().is_none(),
                "a copied buffer must not pin the source"
            );
            release.assert_released(&format!("partial-page copy kind={kind:?}"));
            let mut ctx = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
            let (input, output, expected) = encode(&mut ctx, &linear, &weights, 1);
            drop((linear, weights));
            complete(&mut ctx, &output, &expected);
            MetalBackend::retire_context(&mut ctx).unwrap();
            drop((ctx, input, output));
        });
    }
}
