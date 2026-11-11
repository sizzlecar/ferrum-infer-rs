//! Owned caller phase on the original component/resource chain. An actual
//! PlanRuntimeCloseReceipt is the sole handback authority after native work.
use super::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

struct Phase<P> {
    resource: Mutex<Option<P>>,
}

struct Custody<P: Send + 'static> {
    phase: Arc<Phase<P>>,
    native_started: AtomicBool,
    bound: Option<TrustedPlanRuntimeEvidence>,
}

impl<P: Send + 'static> Custody<P> {
    fn new(resource: P) -> Self {
        Self {
            phase: Arc::new(Phase {
                resource: Mutex::new(Some(resource)),
            }),
            native_started: AtomicBool::new(false),
            bound: None,
        }
    }

    fn bind<R: DeviceRuntime>(&mut self, resources: &PlanRuntimeResources<R>) {
        self.bound = Some(resources.retain_until_closed(Arc::clone(&self.phase)));
    }

    fn take_after_close(&self, receipt: &PlanRuntimeCloseReceipt) -> Option<P> {
        if self.bound.as_ref() != Some(receipt.evidence()) {
            return None;
        }
        self.phase
            .resource
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

impl<P: Send + 'static> Drop for Custody<P> {
    fn drop(&mut self) {
        if self.native_started.load(Ordering::Acquire) && self.bound.is_none() {
            // A transaction can be consumed by a native/source panic before a
            // root or typed error is returned. Runtime's original abandonment
            // has no checked close receipt here. Preserve P with that quarantine
            // and propagate the original panic; do not manufacture retirement.
            tracing::error!("component construction ended without root handoff or original retirement proof; caller phase quarantined");
            std::mem::forget(Arc::clone(&self.phase));
        }
    }
}

impl<R: DeviceRuntime> CompiledTensorComponent<R> {
    /// Retain an actual caller-owned phase/materializer lease before native
    /// provisioning. Old borrowed initialization remains unchanged.
    pub fn initialize_retained<P: Send + 'static>(
        &self,
        runtime: Arc<R>,
        host: HostMemoryAccount,
        source: ComponentWeightSource<'_>,
        config: ComponentInitializationConfig,
        owned_phase: P,
    ) -> Result<RetainedTensorComponentSession<R, P>, RetainedComponentInitializationError<R, P>>
    {
        let mut custody = Custody::new(owned_phase);
        let initialized = initialize_resources(
            runtime,
            &self.family,
            self.executable.execution_plan(),
            source,
            &host,
            &config,
            false,
            Some(&custody.native_started),
        );
        let (resources, receipt, _) = match initialized {
            Ok(initialized) => initialized,
            Err(error) => return Err(RetainedComponentInitializationError { error, custody }),
        };
        custody.bind(resources.as_ref());
        match self.create_session(resources, receipt, host, config.executor) {
            Ok(session) => Ok(RetainedTensorComponentSession { session, custody }),
            Err(error) => Err(RetainedComponentInitializationError { error, custody }),
        }
    }
}

/// The original typed failure and phase are one owning result. Before root
/// handoff, a native failure/panic has no generic checked-close authority:
/// abandoning it quarantines the phase rather than refunding unused credit.
#[must_use = "component initialization failure retains the original native and phase owners"]
pub struct RetainedComponentInitializationError<R: DeviceRuntime, P: Send + 'static> {
    error: ComponentInitializationError<R>,
    custody: Custody<P>,
}

impl<R: DeviceRuntime, P: Send + 'static> RetainedComponentInitializationError<R, P> {
    pub fn error(&self) -> &ComponentInitializationError<R> {
        &self.error
    }

    /// Use the original static failure's checked retirement, without replacing
    /// that owner or issuing a component close receipt for a nonexistent root.
    pub fn retire_static_in_place(&mut self) -> Result<bool, FailureEnvelope> {
        let failure = match &mut self.error {
            ComponentInitializationError::Static(failure)
            | ComponentInitializationError::Handoff {
                source: failure, ..
            } => failure,
            _ => return Ok(false),
        };
        failure.retire_in_place()?;
        Ok(failure.is_retired())
    }

    /// Handback after the original static failure actually retired, or before
    /// native acquisition. Other failures remain owning and fail closed.
    pub fn try_into_retired(self) -> Result<(ComponentInitializationError<R>, P), Self> {
        let retired = match &self.error {
            ComponentInitializationError::Static(failure)
            | ComponentInitializationError::Handoff {
                source: failure, ..
            } => failure.is_retired(),
            _ => false,
        };
        if self.custody.native_started.load(Ordering::Acquire) && !retired {
            return Err(self);
        }
        let resource = self
            .custody
            .phase
            .resource
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("retired initialization owns its original phase");
        self.custody.native_started.store(false, Ordering::Release);
        Ok((self.error, resource))
    }

    /// Only a pre-native failure may hand its phase back without close evidence.
    pub fn try_into_preflight(self) -> Result<(ComponentInitializationError<R>, P), Self> {
        if self.custody.native_started.load(Ordering::Acquire) {
            return Err(self);
        }
        let resource = self
            .custody
            .phase
            .resource
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("preflight custody owns its original phase");
        Ok((self.error, resource))
    }
}

impl<R: DeviceRuntime, P: Send + 'static> fmt::Display
    for RetainedComponentInitializationError<R, P>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl<R: DeviceRuntime, P: Send + 'static> fmt::Debug
    for RetainedComponentInitializationError<R, P>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime, P: Send + 'static> std::error::Error
    for RetainedComponentInitializationError<R, P>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Execution errors keep their original resource root. Thus dropping this
/// wrapper cannot refund P while a submitted/unknown operation still owns it.
pub struct RetainedTensorComponentSession<R: DeviceRuntime, P: Send + 'static> {
    session: TensorComponentSession<R>,
    custody: Custody<P>,
}
impl<R: DeviceRuntime, P: Send + 'static> RetainedTensorComponentSession<R, P> {
    pub fn session(&self) -> &TensorComponentSession<R> {
        &self.session
    }
    pub fn session_mut(&mut self) -> &mut TensorComponentSession<R> {
        &mut self.session
    }

    pub fn close(self) -> Result<(P, PlanRuntimeCloseReceipt), RetainedComponentCloseError<R, P>> {
        let Self { session, custody } = self;
        finish_close(session.close(), custody)
    }
}

enum CloseState<R: DeviceRuntime> {
    Referenced {
        resources: Arc<PlanRuntimeResources<R>>,
        strong_count: usize,
        deferred_cleanup: DeferredDeviceCleanupStatus,
    },
    Failed(PlanRuntimeCloseFailure<R>),
    ForeignReceipt(PlanRuntimeCloseReceipt),
}

/// A non-Closed result retains the exact original root/failure plus its phase.
/// Retry uses the original close operation; it never invents a retirement ack.
#[must_use = "referenced/failed component close retains original native and phase owners"]
pub struct RetainedComponentCloseError<R: DeviceRuntime, P: Send + 'static> {
    state: CloseState<R>,
    custody: Custody<P>,
}
impl<R: DeviceRuntime, P: Send + 'static> RetainedComponentCloseError<R, P> {
    pub fn resources(&self) -> Option<&Arc<PlanRuntimeResources<R>>> {
        match &self.state {
            CloseState::Referenced { resources, .. } => Some(resources),
            _ => None,
        }
    }
    pub fn failure(&self) -> Option<&PlanRuntimeCloseFailure<R>> {
        match &self.state {
            CloseState::Failed(failure) => Some(failure),
            _ => None,
        }
    }
    pub fn retry(self) -> Result<(P, PlanRuntimeCloseReceipt), Self> {
        let Self { state, custody } = self;
        let outcome = match state {
            CloseState::Referenced { resources, .. } => PlanRuntimeResources::close(resources),
            CloseState::Failed(failure) => failure.retry().map(PlanRuntimeCloseOutcome::Closed),
            state @ CloseState::ForeignReceipt(_) => return Err(Self { state, custody }),
        };
        finish_close(outcome, custody)
    }
}
impl<R: DeviceRuntime, P: Send + 'static> fmt::Display for RetainedComponentCloseError<R, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.state {
            CloseState::Referenced {
                strong_count,
                deferred_cleanup,
                ..
            } => write!(
                f,
                "component close retains {strong_count} root references and {} deferred cleanups",
                deferred_cleanup.pending()
            ),
            CloseState::Failed(failure) => {
                write!(f, "component close failed: {:?}", failure.failure())
            }
            CloseState::ForeignReceipt(receipt) => write!(
                f,
                "component close receipt belongs to another phase root: {:?}",
                receipt.evidence()
            ),
        }
    }
}
impl<R: DeviceRuntime, P: Send + 'static> fmt::Debug for RetainedComponentCloseError<R, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime, P: Send + 'static> std::error::Error for RetainedComponentCloseError<R, P> {}

fn finish_close<R: DeviceRuntime, P: Send + 'static>(
    outcome: Result<PlanRuntimeCloseOutcome<R>, PlanRuntimeCloseFailure<R>>,
    custody: Custody<P>,
) -> Result<(P, PlanRuntimeCloseReceipt), RetainedComponentCloseError<R, P>> {
    let state = match outcome {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => match custody.take_after_close(&receipt) {
            Some(phase) => return Ok((phase, receipt)),
            None => CloseState::ForeignReceipt(receipt),
        },
        Ok(PlanRuntimeCloseOutcome::Referenced {
            resources,
            strong_count,
            deferred_cleanup,
        }) => CloseState::Referenced {
            resources,
            strong_count,
            deferred_cleanup,
        },
        Err(failure) => CloseState::Failed(failure),
    };
    Err(RetainedComponentCloseError { state, custody })
}
