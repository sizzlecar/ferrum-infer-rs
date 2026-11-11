//! Bounded page registration of existing, accounted static-upload allocations.
//! No private staging payload or checkpoint mapping is allocated or registered.

use super::{CudaContext, CudaDeviceRuntimeError, ManagedHostBytes};
use ferrum_runtime::{HostMemoryClass, NativeHostMemoryRegistration};
use std::{
    ops::Range,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

/// Construction-time bounds; these enter runtime identity before capacity admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaPinnedWeightUploadConfig {
    pub maximum_allocation_bytes: std::num::NonZeroUsize,
    pub maximum_registered_bytes: std::num::NonZeroUsize,
}
impl CudaPinnedWeightUploadConfig {
    pub(super) fn validate(self) -> Result<(), CudaDeviceRuntimeError> {
        if self.maximum_allocation_bytes > self.maximum_registered_bytes
            || self.maximum_registered_bytes.get() > isize::MAX as usize
        {
            return Err(CudaDeviceRuntimeError::contract(
                "pinned weight upload limits are inconsistent or unaddressable",
            ));
        }
        Ok(())
    }
}

/// The product chooses whether to create this registry and its fixed capacity.
/// Registrations retain the registry after the runtime drops, so its bound and
/// CUDA context also survive indeterminate native completion.
#[derive(Clone)]
pub(super) struct CudaPinnedHostRegistry(Arc<Registry>);

struct Registry {
    context: Arc<CudaContext>,
    maximum_bytes: usize,
    registered_bytes: AtomicUsize,
    page_size: usize,
}

impl CudaPinnedHostRegistry {
    pub(super) fn new(
        context: Arc<CudaContext>,
        maximum_bytes: usize,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if maximum_bytes == 0 {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA pinned Host capacity must be positive",
            ));
        }
        Ok(Self(Arc::new(Registry {
            context,
            maximum_bytes,
            registered_bytes: AtomicUsize::new(0),
            page_size: Self::required_alignment()?,
        })))
    }

    /// Product Host allocation alignment must include native page padding in
    /// its existing physical accounting before a streaming slot is allocated.
    pub(super) fn required_alignment() -> Result<usize, CudaDeviceRuntimeError> {
        #[cfg(unix)]
        {
            // SAFETY: sysconf reads process metadata and has no pointer inputs.
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page_size > 0 && (page_size as usize).is_power_of_two() {
                return Ok(page_size as usize);
            }
        }
        Err(CudaDeviceRuntimeError::contract(
            "CUDA pinned Host registration requires a supported native page size",
        ))
    }

    pub(super) fn register(&self, source: &ManagedHostBytes) -> Result<(), CudaDeviceRuntimeError> {
        if !matches!(
            source.class(),
            HostMemoryClass::WeightBacking | HostMemoryClass::TransferStaging
        ) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA bounded pinned upload requires accounted static weight storage",
            ));
        }
        // This Arc is retained by every successful attachment. Its address
        // cannot be reused while the allocation still has that registration.
        let key = Arc::as_ptr(&self.0) as usize as u64;
        source.try_register_native(key, |owner, logical| {
            let range = registration_range(
                owner.as_ptr() as usize,
                owner.len(),
                logical,
                self.0.page_size,
            )?;
            self.0.reserve(owner.len())?;
            let registration = (|| {
                self.0.context.bind_to_thread().map_err(|error| {
                    CudaDeviceRuntimeError::driver("bind pinned Host registration context", error)
                })?;
                // SAFETY: range encloses the original logical buffer and lies
                // wholly inside its accounted, stable allocation. The attachment
                // retires before allocation release; commands retain that same
                // allocation through their existing fence or quarantine owner.
                unsafe {
                    cudarc::driver::sys::cuMemHostRegister_v2(
                        range.start as *mut std::ffi::c_void,
                        range.len(),
                        0,
                    )
                    .result()
                }
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("register accounted pinned Host memory", error)
                })?;
                Ok(CudaHostRegistration {
                    registry: Arc::clone(&self.0),
                    pointer: range.start,
                    physical_bytes: owner.len(),
                    registered_bytes: range.len(),
                    registered: true,
                })
            })();
            if registration.is_err() {
                self.0
                    .registered_bytes
                    .fetch_sub(owner.len(), Ordering::AcqRel);
            }
            registration
        })
    }
}

impl Registry {
    fn reserve(&self, bytes: usize) -> Result<(), CudaDeviceRuntimeError> {
        reserve_capacity(&self.registered_bytes, self.maximum_bytes, bytes)
    }
}

fn reserve_capacity(
    registered: &AtomicUsize,
    maximum: usize,
    bytes: usize,
) -> Result<(), CudaDeviceRuntimeError> {
    registered
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current.checked_add(bytes).filter(|next| *next <= maximum)
        })
        .map(|_| ())
        .map_err(|current| {
            CudaDeviceRuntimeError::contract(format!(
                "CUDA pinned Host capacity exceeded: owner={bytes}, retained={current}, limit={maximum}"
            ))
        })
}

struct CudaHostRegistration {
    registry: Arc<Registry>,
    pointer: usize,
    physical_bytes: usize,
    registered_bytes: usize,
    registered: bool,
}

// SAFETY: this guard keeps only native registration/context ownership, never an
// Arc back to its allocation. Attachment retirement precedes allocation release;
// the exact original allocation also follows CUDA's command/fence/quarantine.
unsafe impl NativeHostMemoryRegistration for CudaHostRegistration {
    fn release(&mut self) -> bool {
        if !self.registered {
            return true;
        }
        let result = self.registry.context.bind_to_thread().and_then(|()| {
            // SAFETY: this is the exact successful registration's pointer, and
            // the attached allocation remains alive until this returns true.
            unsafe {
                cudarc::driver::sys::cuMemHostUnregister(self.pointer as *mut std::ffi::c_void)
                    .result()
            }
        });
        if let Err(error) = result {
            tracing::error!(
                %error,
                physical_bytes = self.physical_bytes,
                registered_bytes = self.registered_bytes,
                "CUDA Host unregister failed; original allocation, context and pinned capacity retained"
            );
            return false;
        }
        self.registered = false;
        self.registry
            .registered_bytes
            .fetch_sub(self.physical_bytes, Ordering::AcqRel);
        true
    }
}

fn registration_range(
    owner_start: usize,
    owner_length: usize,
    logical: Range<usize>,
    page_size: usize,
) -> Result<Range<usize>, CudaDeviceRuntimeError> {
    let range = (|| {
        if !page_size.is_power_of_two()
            || logical.start >= logical.end
            || logical.end > owner_length
        {
            return None;
        }
        let owner_end = owner_start.checked_add(owner_length)?;
        let logical_start = owner_start.checked_add(logical.start)?;
        let logical_end = owner_start.checked_add(logical.end)?;
        let start = logical_start & !(page_size - 1);
        let end = logical_end.checked_add(page_size - 1)? & !(page_size - 1);
        (start >= owner_start && end <= owner_end && start < end).then_some(start..end)
    })();
    range.ok_or_else(|| CudaDeviceRuntimeError::contract(
        "CUDA pinned Host pages must fit wholly inside the accounted allocation; use native page-aligned Host backing",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA runtime construction and Core capacity admission"]
    fn pinned_runtime_identity_is_final_before_core_capacity_registration() {
        use crate::backend::cuda::vnext_runtime::{
            CudaDeviceRuntime, CudaTensorRuntimeConfig, CudaTensorRuntimeOptions,
        };
        use ferrum_runtime::{DeviceCapacityOwner, DeviceId, DeviceRuntime};
        use std::num::NonZeroUsize;
        let capacity = 1 << 30;
        let runtime = CudaDeviceRuntime::new_tensor_eager_with_options(
            CudaTensorRuntimeConfig {
                device: crate::backend::cuda::vnext_ops::cuda_tensor_runtime_config(
                    0,
                    DeviceId::new("device.cuda.0").unwrap(),
                )
                .unwrap(),
                usable_memory_bytes: capacity,
                bootstrap_memory_bytes: 256 << 20,
                stream_memory_bytes: 16 << 20,
                maximum_streams: 1,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: 8,
                opaque_memory_profile: "unqualified-pinned-capacity-lifecycle-fixture".into(),
            },
            CudaTensorRuntimeOptions {
                pinned_weight_upload: Some(CudaPinnedWeightUploadConfig {
                    maximum_allocation_bytes: NonZeroUsize::new(1 << 20).unwrap(),
                    maximum_registered_bytes: NonZeroUsize::new(4 << 20).unwrap(),
                }),
                ..Default::default()
            },
        )
        .unwrap();
        runtime.descriptor().validate().unwrap();
        // Static installation opens the same owner after construction. A late
        // descriptor mutation would conflict with the already registered owner.
        let owner = DeviceCapacityOwner::open(runtime.descriptor(), capacity).unwrap();
        assert!(runtime.pinned_weight_upload.is_some());
        drop(owner);
        drop(runtime);
    }

    #[test]
    #[ignore = "requires CUDA page registration and native H2D transfer"]
    fn accounted_cuda_registration_survives_reuse_and_retires_before_refunding() {
        use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let page = CudaPinnedHostRegistry::required_alignment().unwrap();
        let logical = page + 17;
        let total = 16 * page;
        let host = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                total,
                HostMemoryClassLimits {
                    input_preprocess_bytes: total,
                    transfer_staging_bytes: total,
                    output_bytes: total,
                    weight_backing_bytes: total,
                },
            )
            .unwrap(),
            page,
        )
        .unwrap();
        let physical = host.backing_bytes(logical).unwrap();
        let registry = CudaPinnedHostRegistry::new(context, physical).unwrap();
        let mut backing = host
            .allocate_zeroed(HostMemoryClass::WeightBacking, logical)
            .unwrap();
        for (i, byte) in backing.as_mut().iter_mut().enumerate() {
            *byte = (i.wrapping_mul(37) ^ (i >> 3)) as u8;
        }
        let source = backing.freeze();
        let slice = source.slice(3..logical - 1).unwrap();
        registry.register(&slice).unwrap();
        registry.register(&source).unwrap();
        assert_eq!(
            registry.0.registered_bytes.load(Ordering::Acquire),
            physical
        );
        let source = match source.try_thaw() {
            Err(source) => source,
            Ok(_) => panic!("live slice must prevent writable recovery"),
        };
        drop(slice);

        let mut device = stream.clone_htod(source.bytes()).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(stream.clone_dtoh(&device).unwrap(), source.bytes());
        let blocked = host
            .allocate_zeroed(HostMemoryClass::WeightBacking, logical)
            .unwrap()
            .freeze();
        assert!(registry.register(&blocked).is_err());
        assert_eq!(
            registry.0.registered_bytes.load(Ordering::Acquire),
            physical
        );
        let mut backing = match source.try_thaw() {
            Ok(backing) => backing,
            Err(_) => panic!("completed transfer has no retained aliases"),
        };
        backing.as_mut().fill(0xa5);
        let source = backing.freeze();
        registry.register(&source).unwrap();
        stream.memcpy_htod(source.bytes(), &mut device).unwrap();
        stream.synchronize().unwrap();
        assert!(stream
            .clone_dtoh(&device)
            .unwrap()
            .iter()
            .all(|&byte| byte == 0xa5));
        drop(source);
        assert_eq!(registry.0.registered_bytes.load(Ordering::Acquire), 0);
        assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
        registry.register(&blocked).unwrap();
        drop(blocked);
        assert_eq!(registry.0.registered_bytes.load(Ordering::Acquire), 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }

    #[test]
    fn registration_uses_accounted_enclosing_pages_not_sliced_payloads() {
        assert_eq!(
            registration_range(0x1003, 3 * 4096, 4093..4093 + 5000, 4096).unwrap(),
            0x2000..0x4000
        );
        assert!(registration_range(0x1003, 5000, 0..4096, 4096).is_err());
        assert!(registration_range(0x1000, 5000, 0..5000, 4096).is_err());
        assert!(registration_range(0x1000, 4096, 1..1, 4096).is_err());
        assert!(registration_range(usize::MAX - 4095, 8192, 0..4096, 4096).is_err());
        assert!(registration_range(0x1000, 4096, 0..4096, 3).is_err());
    }

    #[test]
    fn physical_owner_capacity_is_shared_and_rejection_does_not_consume_credit() {
        let registered = AtomicUsize::new(0);
        reserve_capacity(&registered, 8192, 4097).unwrap();
        assert!(reserve_capacity(&registered, 8192, 4096).is_err());
        assert_eq!(registered.load(Ordering::Acquire), 4097);
        reserve_capacity(&registered, 8192, 4095).unwrap();
        assert!(reserve_capacity(&registered, 8192, 1).is_err());
        assert!(reserve_capacity(&registered, usize::MAX, usize::MAX).is_err());
        assert_eq!(registered.load(Ordering::Acquire), 8192);
        registered.fetch_sub(4097, Ordering::AcqRel);
        reserve_capacity(&registered, 8192, 4097).unwrap();
        assert_eq!(registered.load(Ordering::Acquire), 8192);
    }
}
