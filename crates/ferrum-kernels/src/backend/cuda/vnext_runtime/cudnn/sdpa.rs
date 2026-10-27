//! Fixed-shape cuDNN SDPA plans. Capacity is reserved before graph creation;
//! queried workspace is supplied by the operation's admitted scratch region.
//! Compiled Ferrum plans retain the exact native plans selected during resource
//! estimation. The finite cache reclaims only entries with no remaining lease.

use super::*;
use crate::retained_plan_cache::{PlanCacheError, RetainedPlanCache};
use std::sync::atomic::{AtomicUsize, Ordering};

mod native;
#[cfg(test)]
mod tests;

/// Element strides, not bytes. Q/O may be slices of a larger BSHD tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct CudnnSdpaSignature {
    pub batch: i64,
    pub heads: i64,
    pub query_length: i64,
    pub key_value_length: i64,
    pub head_dim: i64,
    pub query_batch_stride: i64,
    pub key_value_batch_stride: i64,
    pub output_batch_stride: i64,
    pub scale_bits: u32,
}

impl CudnnSdpaSignature {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if !(1..=16).contains(&self.batch)
            || self.heads <= 0
            || !matches!(self.head_dim, 72 | 128)
            || self.query_length <= 0
            || self.key_value_length <= 0
            || self.query_length > i32::MAX as i64
            || self.key_value_length > i32::MAX as i64
            || !f32::from_bits(self.scale_bits).is_finite()
            || f32::from_bits(self.scale_bits) <= 0.0
        {
            return Err("cuDNN SDPA requires B=1..16, positive bounded lengths/heads, D72 or D128 and finite positive scale".into());
        }
        self.extent(self.query_length, self.query_batch_stride)?;
        self.extent(self.key_value_length, self.key_value_batch_stride)?;
        self.extent(self.query_length, self.output_batch_stride)?;
        Ok(())
    }

    fn extent(&self, length: i64, batch_stride: i64) -> Result<u64, String> {
        let row = self
            .heads
            .checked_mul(self.head_dim)
            .ok_or("SDPA row extent overflows")?;
        let one = row
            .checked_mul(length)
            .ok_or("SDPA batch extent overflows")?;
        if batch_stride < one || batch_stride <= 0 {
            return Err("SDPA batch stride overlaps batches".into());
        }
        let bytes = (self.batch - 1)
            .checked_mul(batch_stride)
            .and_then(|v| v.checked_add(one))
            .and_then(|v| v.checked_mul(2))
            .filter(|&v| v > 0 && v <= isize::MAX as i64)
            .ok_or("SDPA physical extent exceeds address space")?;
        Ok(bytes as u64)
    }

    fn digest(&self, digest: &mut Sha256) {
        for value in [
            self.batch,
            self.heads,
            self.query_length,
            self.key_value_length,
            self.head_dim,
            self.query_batch_stride,
            self.key_value_batch_stride,
            self.output_batch_stride,
        ] {
            digest.update(value.to_le_bytes());
        }
        digest.update(self.scale_bits.to_le_bytes());
    }
}

/// The opaque allowance covers this bounded cache and its live variant packs.
/// It is a deployment declaration, not a measured native allocation bound.
#[derive(Clone, Copy, Debug)]
pub struct CudnnSdpaPlanPolicy {
    pub maximum_plans: usize,
    pub maximum_live_bindings: usize,
    pub opaque_memory_bytes: u64,
    pub workspace_limit_bytes: u64,
}
impl CudnnSdpaPlanPolicy {
    fn validate(self) -> Result<(), String> {
        if self.maximum_plans == 0
            || self.maximum_live_bindings == 0
            || self.opaque_memory_bytes == 0
            || self.workspace_limit_bytes > isize::MAX as u64
            || self
                .maximum_plans
                .checked_mul(std::mem::size_of::<CudnnSdpaSignature>())
                .is_none()
            || self
                .maximum_live_bindings
                .checked_mul(std::mem::size_of::<CudnnSdpaBindings>())
                .is_none()
        {
            return Err("cuDNN SDPA cache requires finite positive plan/binding/opaque limits and bounded workspace".into());
        }
        Ok(())
    }
}

struct BindingLimit {
    live: AtomicUsize,
    maximum: usize,
}
struct BindingTicket(Arc<BindingLimit>);
impl BindingLimit {
    fn acquire(self: &Arc<Self>) -> Result<BindingTicket, String> {
        self.live
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.maximum).then(|| n + 1)
            })
            .map_err(|_| "cuDNN SDPA live variant-pack limit exhausted".to_string())?;
        Ok(BindingTicket(Arc::clone(self)))
    }
}
impl Drop for BindingTicket {
    fn drop(&mut self) {
        self.0.live.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(crate) struct CudaCudnnSdpaCache {
    owner: Arc<CudaCudnnOwner>,
    policy: CudnnSdpaPlanPolicy,
    opaque: DeviceCapacityRetention,
    bindings: Arc<BindingLimit>,
    plans: Mutex<RetainedPlanCache<CudnnSdpaSignature, CudnnSdpaPlan>>,
    fingerprint: String,
}
impl CudaCudnnSdpaCache {
    pub(crate) fn new(
        runtime: &CudaDeviceRuntime,
        owner: Arc<CudaCudnnOwner>,
        policy: CudnnSdpaPlanPolicy,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        policy
            .validate()
            .map_err(CudaDeviceRuntimeError::contract)?;
        {
            let guard = owner
                .inner
                .lock()
                .map_err(|_| CudaDeviceRuntimeError::contract("cuDNN owner poisoned"))?;
            let native = guard
                .as_ref()
                .ok_or_else(|| CudaDeviceRuntimeError::contract("cuDNN owner closed"))?;
            if native.failed || !Arc::ptr_eq(&native.context, runtime.context()) {
                return Err(CudaDeviceRuntimeError::contract(
                    "SDPA owner belongs to another context or is quarantined",
                ));
            }
            let version = unsafe { (native.api.version)() };
            if !(92_000..93_000).contains(&version) {
                return Err(CudaDeviceRuntimeError::contract(format!(
                    "SDPA profile requires cuDNN 9.20.x, got {version}"
                )));
            }
        }
        let opaque = runtime.reserve_tensor_opaque(policy.opaque_memory_bytes)?;
        let mut digest = Sha256::new();
        digest.update(owner.fingerprint().as_bytes());
        digest.update((policy.maximum_plans as u64).to_le_bytes());
        digest.update((policy.maximum_live_bindings as u64).to_le_bytes());
        digest.update(policy.opaque_memory_bytes.to_le_bytes());
        digest.update(policy.workspace_limit_bytes.to_le_bytes());
        Ok(Self {
            owner,
            policy,
            opaque,
            bindings: Arc::new(BindingLimit {
                live: AtomicUsize::new(0),
                maximum: policy.maximum_live_bindings,
            }),
            plans: Mutex::new(RetainedPlanCache::new(policy.maximum_plans)),
            fingerprint: format!("{:x}", digest.finalize()),
        })
    }

    pub(crate) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Planning only. Serializes preparation, so concurrent estimates cannot
    /// create two plans or spend the cache's bounded native capacity twice.
    pub(crate) fn prepare(
        &self,
        signature: CudnnSdpaSignature,
    ) -> Result<Arc<CudnnSdpaPlan>, CudaDeviceRuntimeError> {
        signature
            .validate()
            .map_err(CudaDeviceRuntimeError::contract)?;
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("SDPA cache poisoned"))?;
        plans
            .get_or_try_insert_with(signature, || {
                self.owner.prepare_sdpa(
                    signature,
                    self.policy.workspace_limit_bytes,
                    self.opaque.clone(),
                    Arc::clone(&self.bindings),
                )
            })
            .map_err(|error| match error {
                PlanCacheError::AllPlansRetained => CudaDeviceRuntimeError::contract(
                    "cuDNN SDPA plan cache limit exhausted by retained plans",
                ),
                PlanCacheError::Prepare(error) => error,
            })
    }

    /// Encoding never compiles or reselects an engine after scratch admission.
    pub(crate) fn lookup(
        &self,
        signature: &CudnnSdpaSignature,
    ) -> Result<Arc<CudnnSdpaPlan>, CudaDeviceRuntimeError> {
        self.plans
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("SDPA cache poisoned"))?
            .get(signature)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "SDPA plan was not prepared during resource estimation",
                )
            })
    }
}

pub(crate) struct CudnnSdpaPlan {
    owner: Arc<CudaCudnnOwner>,
    native: Option<native::NativePlan>,
    signature: CudnnSdpaSignature,
    scale: f32,
    workspace_bytes: u64,
    fingerprint: String,
    opaque: DeviceCapacityRetention,
    bindings: Arc<BindingLimit>,
}
// Every native access is serialized by owner.inner and binds its CUDA context.
unsafe impl Send for CudnnSdpaPlan {}
unsafe impl Sync for CudnnSdpaPlan {}

impl CudaCudnnOwner {
    fn prepare_sdpa(
        self: &Arc<Self>,
        signature: CudnnSdpaSignature,
        cap: u64,
        opaque: DeviceCapacityRetention,
        bindings: Arc<BindingLimit>,
    ) -> Result<Arc<CudnnSdpaPlan>, CudaDeviceRuntimeError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("cuDNN owner poisoned"))?;
        let owner = guard
            .as_mut()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("cuDNN owner closed"))?;
        if owner.failed {
            return Err(CudaDeviceRuntimeError::contract(
                "cuDNN owner requires quarantine",
            ));
        }
        owner
            .context
            .bind_to_thread()
            .map_err(|e| CudaDeviceRuntimeError::driver("SDPA plan context bind", e))?;
        owner.failed = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            native::NativePlan::build(&owner.api.graph, owner.handle, signature, cap)
        }));
        let native = match result {
            Ok(Ok(plan)) => {
                owner.failed = false;
                plan
            }
            Ok(Err(error)) => {
                owner.failed = !error.cleanup_known;
                if owner.failed {
                    std::mem::forget((Arc::clone(self), opaque));
                }
                return Err(CudaDeviceRuntimeError::contract(format!(
                    "SDPA prepare signature={signature:?} workspace_cap={cap} cleanup_known={}: {}",
                    error.cleanup_known, error.message
                )));
            }
            Err(payload) => {
                std::mem::forget((Arc::clone(self), opaque));
                std::panic::resume_unwind(payload);
            }
        };
        let mut digest = Sha256::new();
        digest.update(self.fingerprint().as_bytes());
        signature.digest(&mut digest);
        digest.update(native.workspace_bytes.to_le_bytes());
        digest.update(native.identity.as_bytes());
        let workspace_bytes = native.workspace_bytes;
        Ok(Arc::new(CudnnSdpaPlan {
            owner: Arc::clone(self),
            native: Some(native),
            signature,
            scale: f32::from_bits(signature.scale_bits),
            workspace_bytes,
            fingerprint: format!("{:x}", digest.finalize()),
            opaque,
            bindings,
        }))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CudnnSdpaBindings {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
    pub seq_len_q: u64,
    pub seq_len_kv: u64,
    pub workspace: u64,
    pub workspace_bytes: u64,
}

impl CudnnSdpaPlan {
    pub(crate) fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }
    pub(crate) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(crate) fn bind(
        self: &Arc<Self>,
        bindings: CudnnSdpaBindings,
    ) -> Result<Arc<CudnnSdpaExecution>, CudaDeviceRuntimeError> {
        validate_bindings(self.signature, self.workspace_bytes, bindings)
            .map_err(CudaDeviceRuntimeError::contract)?;
        let ticket = self
            .bindings
            .acquire()
            .map_err(CudaDeviceRuntimeError::contract)?;
        let mut guard = self
            .owner
            .inner
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("cuDNN owner poisoned"))?;
        let owner = guard
            .as_mut()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("cuDNN owner closed"))?;
        if owner.failed {
            return Err(CudaDeviceRuntimeError::contract(
                "cuDNN owner requires quarantine",
            ));
        }
        owner
            .context
            .bind_to_thread()
            .map_err(|e| CudaDeviceRuntimeError::driver("SDPA binding context bind", e))?;
        owner.failed = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            native::VariantPack::build(&owner.api.graph, bindings, &self.scale)
        }));
        match result {
            Ok(Ok(pack)) => {
                owner.failed = false;
                Ok(Arc::new(CudnnSdpaExecution {
                    plan: Arc::clone(self),
                    pack: Some(pack),
                    ticket: Some(ticket),
                }))
            }
            Ok(Err(error)) => {
                owner.failed = !error.cleanup_known;
                if owner.failed {
                    std::mem::forget((Arc::clone(self), ticket));
                }
                Err(CudaDeviceRuntimeError::contract(format!(
                    "SDPA bind plan={} signature={:?} workspace_bytes={} cleanup_known={}: {}",
                    self.fingerprint,
                    self.signature,
                    self.workspace_bytes,
                    error.cleanup_known,
                    error.message
                )))
            }
            Err(payload) => {
                std::mem::forget((Arc::clone(self), ticket));
                std::panic::resume_unwind(payload)
            }
        }
    }
}

/// Capture this Arc in the retained device command, together with the actual
/// allocation regions. It must survive until terminal fence/quarantine cleanup.
pub(crate) struct CudnnSdpaExecution {
    plan: Arc<CudnnSdpaPlan>,
    pack: Option<native::VariantPack>,
    ticket: Option<BindingTicket>,
}
unsafe impl Send for CudnnSdpaExecution {}
unsafe impl Sync for CudnnSdpaExecution {}
impl CudnnSdpaExecution {
    pub(crate) fn execute(&self, stream: &CudaStream) -> Result<(), CudaDeviceRuntimeError> {
        let mut guard = self
            .plan
            .owner
            .inner
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("cuDNN owner poisoned"))?;
        let owner = guard
            .as_mut()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("cuDNN owner closed"))?;
        if owner.failed {
            return Err(CudaDeviceRuntimeError::contract(
                "cuDNN owner requires quarantine",
            ));
        }
        owner
            .context
            .bind_to_thread()
            .map_err(|e| CudaDeviceRuntimeError::driver("SDPA execution context bind", e))?;
        let plan = self
            .plan
            .native
            .as_ref()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("SDPA plan retired"))?;
        let pack = self
            .pack
            .as_ref()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("SDPA binding retired"))?;
        owner.failed = true;
        let result = unsafe {
            check(
                (owner.api.set_stream)(owner.handle, stream.cu_stream() as *mut c_void),
                "SDPA set stream",
            )
            .and_then(|()| {
                check(
                    (owner.api.graph.execute)(owner.handle, plan.descriptor, pack.descriptor),
                    "SDPA execute",
                )
            })
            .and_then(|()| {
                check(
                    (owner.api.set_stream)(owner.handle, std::ptr::null_mut()),
                    "SDPA clear stream",
                )
            })
        };
        if result.is_ok() {
            owner.failed = false;
        }
        result.map_err(|error| CudaDeviceRuntimeError::contract(format!(
            "SDPA execution plan={} signature={:?} workspace_bytes={} descriptor_kind=5 variant_pack_kind=16 attribute=none type=none count=0: {error}",
            self.plan.fingerprint, self.plan.signature, self.plan.workspace_bytes
        )))
    }
}

impl Drop for CudnnSdpaExecution {
    fn drop(&mut self) {
        let Some(mut pack) = self.pack.take() else {
            return;
        };
        let cleaned = match self.plan.owner.inner.lock() {
            Ok(mut guard) => match guard.as_mut() {
                Some(owner) if !owner.failed && owner.context.bind_to_thread().is_ok() => {
                    let result = pack.destroy(&owner.api.graph);
                    owner.failed = !result;
                    result
                }
                _ => false,
            },
            Err(_) => false,
        };
        if !cleaned {
            std::mem::forget((pack, Arc::clone(&self.plan), self.ticket.take()));
        }
    }
}
impl Drop for CudnnSdpaPlan {
    fn drop(&mut self) {
        let Some(mut plan) = self.native.take() else {
            return;
        };
        let cleaned = match self.owner.inner.lock() {
            Ok(mut guard) => match guard.as_mut() {
                Some(owner) if !owner.failed && owner.context.bind_to_thread().is_ok() => {
                    let result = plan.destroy(&owner.api.graph);
                    owner.failed = !result;
                    result
                }
                _ => false,
            },
            Err(_) => false,
        };
        if !cleaned {
            std::mem::forget((plan, Arc::clone(&self.owner), self.opaque.clone()));
        }
    }
}

fn validate_bindings(
    s: CudnnSdpaSignature,
    workspace_bytes: u64,
    b: CudnnSdpaBindings,
) -> Result<(), String> {
    s.validate()?;
    if b.workspace_bytes != workspace_bytes {
        return Err("SDPA workspace differs from prepared plan".into());
    }
    let spans = [
        (b.q, s.extent(s.query_length, s.query_batch_stride)?, 16),
        (
            b.k,
            s.extent(s.key_value_length, s.key_value_batch_stride)?,
            16,
        ),
        (
            b.v,
            s.extent(s.key_value_length, s.key_value_batch_stride)?,
            16,
        ),
        (b.o, s.extent(s.query_length, s.output_batch_stride)?, 16),
        (b.seq_len_q, s.batch as u64 * 4, 4),
        (b.seq_len_kv, s.batch as u64 * 4, 4),
        (b.workspace, workspace_bytes, 256),
    ];
    for &(pointer, bytes, alignment) in &spans {
        if bytes != 0
            && (pointer == 0 || pointer % alignment != 0 || pointer.checked_add(bytes).is_none())
        {
            return Err("SDPA binding has invalid pointer, alignment or extent".into());
        }
    }
    for index in [3, 6] {
        let (p, n, _) = spans[index];
        if n == 0 {
            continue;
        }
        for (other, &(q, m, _)) in spans.iter().enumerate() {
            if other != index && m != 0 && p < q + m && q < p + n {
                return Err(
                    "SDPA output/workspace aliases an input or another live binding".into(),
                );
            }
        }
    }
    Ok(())
}
