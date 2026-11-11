//! Shared run/serve selection. MTP uses the same prepared requests, phase
//! account and downstream handoff as ordinary prompt expansion.
use super::cohort::{
    ExpansionCohortProgress, ExpansionParticipantResult, NativeExpansionCohort,
    PreparedExpansionCohort,
};
use super::mtp_cohort::{
    initialization::{InitializationPoll, MtpInitialization},
    MtpExpansionCohort,
};
use super::*;
use ferrum_execution::executor_lifecycle::{ExecutorShutdownObserver, ExecutorShutdownReporter};
use ferrum_runtime::HostMemoryScope;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    task::Waker,
    time::Instant,
};

pub(crate) enum ProductExpansionCohort<R: DeviceRuntime> {
    Ordinary(NativeExpansionCohort),
    Mtp(MtpOwner<R>),
}
impl<R: DeviceRuntime> ProductExpansionCohort<R> {
    pub(crate) fn initialize_with_diagnostics(
        assets: Arc<NativeTextExpansion>,
        composition: RuntimeComposition<R>,
        prepared: PreparedExpansionCohort,
        phase: HostMemoryScope,
        diagnostics: Vec<Option<DiagnosticContext>>,
    ) -> ExecutorBuildResult<Self> {
        if assets.mtp == PH3ImagePromptExpansionMtp::Off {
            return NativeExpansionCohort::initialize_with_diagnostics(
                assets,
                &composition,
                prepared,
                phase,
                diagnostics,
            )
            .map(Self::Ordinary);
        }
        if diagnostics.len() != prepared.len() {
            return Err(
                FerrumError::request_validation("MTP diagnostic membership differs").into(),
            );
        }
        let count = prepared.len();
        let started = Instant::now();
        // Construct after all original source/conversion/cache policies have
        // been configured; a stale copy of those policies is never cached.
        let mtp = Arc::new(mtp_assets::MtpExpansionAssets::from_native(&assets)?);
        let initialization = MtpInitialization::new(assets, mtp, prepared, phase);
        Ok(Self::Mtp(MtpOwner::new(
            initialization,
            composition,
            count,
            diagnostics
                .into_iter()
                .map(|context| context.map(|context| (context, started)))
                .collect(),
        )))
    }

    pub(crate) fn cleanup(&self) -> ExecutorShutdownObserver {
        match self {
            Self::Ordinary(owner) => owner.cleanup(),
            Self::Mtp(owner) => owner.observer.clone(),
        }
    }

    pub(crate) fn advance(
        &mut self,
        cancelled: &[bool],
        wake: &Waker,
    ) -> Result<ExpansionCohortProgress> {
        match self {
            Self::Ordinary(owner) => owner.advance(cancelled, wake),
            Self::Mtp(owner) => owner.advance(cancelled, wake),
        }
    }
}

enum MtpState<R: DeviceRuntime> {
    Initializing(MtpInitialization<R>),
    Running(MtpExpansionCohort<R>),
    Closed,
}
impl<R: DeviceRuntime> MtpState<R> {
    fn try_close(&mut self) -> Result<bool> {
        let closed = match self {
            Self::Initializing(owner) => owner.try_close()?,
            Self::Running(owner) => owner.try_close()?,
            Self::Closed => true,
        };
        if closed {
            *self = Self::Closed;
        }
        Ok(closed)
    }
}

/// Cleanup observations own the same machine. Only its checked Core close
/// can publish Retired. Unknown native authority remains in this custody if
/// an observer/consumer is dropped, with an explicit retry on that same owner.
pub(crate) struct MtpOwner<R: DeviceRuntime> {
    state: Arc<Mutex<MtpState<R>>>,
    reporter: ExecutorShutdownReporter,
    observer: ExecutorShutdownObserver,
    composition: RuntimeComposition<R>,
    count: usize,
    cancelled: Vec<bool>,
    closing: Arc<AtomicBool>,
    diagnostics: Vec<Option<(DiagnosticContext, Instant)>>,
}
impl<R: DeviceRuntime> MtpOwner<R> {
    fn new(
        initialization: MtpInitialization<R>,
        composition: RuntimeComposition<R>,
        count: usize,
        diagnostics: Vec<Option<(DiagnosticContext, Instant)>>,
    ) -> Self {
        let state = Arc::new(Mutex::new(MtpState::Initializing(initialization)));
        let (reporter, observer) = ExecutorShutdownObserver::channel(state.clone());
        reporter.retain_until_retired(Arc::clone(&state));
        let weak = Arc::downgrade(&state);
        let retry_reporter = reporter.clone();
        let closing = Arc::new(AtomicBool::new(false));
        let retry_closing = Arc::clone(&closing);
        let observer = observer.with_retry(Arc::new(move || {
            if !retry_closing.load(Ordering::Acquire) {
                return;
            }
            if let Some(state) = weak.upgrade() {
                close_original(&state, &retry_reporter);
            }
        }));
        Self {
            state,
            reporter,
            observer,
            composition,
            count,
            cancelled: vec![false; count],
            closing,
            diagnostics,
        }
    }

    fn advance(&mut self, cancelled: &[bool], wake: &Waker) -> Result<ExpansionCohortProgress> {
        if self.closing.load(Ordering::Acquire) {
            return Err(FerrumError::backend(
                "MTP owner is closing; execution cannot resume",
            ));
        }
        if cancelled.len() != self.count {
            let error = FerrumError::request_validation("MTP cancellation membership differs");
            self.closing.store(true, Ordering::Release);
            self.reporter.retained(error.clone());
            close_original(&self.state, &self.reporter);
            return Err(error);
        }
        for (latched, &cancel) in self.cancelled.iter_mut().zip(cancelled) {
            *latched |= cancel;
        }
        let result = (|| {
            let mut state = self
                .state
                .lock()
                .map_err(|_| FerrumError::internal("MTP owner lock poisoned"))?;
            match &mut *state {
                MtpState::Initializing(owner) if self.cancelled.iter().all(|&cancel| cancel) => {
                    if owner.try_close()? {
                        *state = MtpState::Closed;
                        Ok(ExpansionCohortProgress::Retired(
                            (0..self.count)
                                .map(|_| ExpansionParticipantResult::Cancelled)
                                .collect(),
                        ))
                    } else {
                        Err(FerrumError::backend(
                            "cancelled MTP initialization retains original native cleanup",
                        ))
                    }
                }
                MtpState::Initializing(owner) => match owner.poll(&self.composition) {
                    Ok(InitializationPoll::Pending { made_progress }) => {
                        Ok(ExpansionCohortProgress::Pending { made_progress })
                    }
                    Ok(InitializationPoll::Ready(cohort)) => {
                        *state = MtpState::Running(cohort);
                        if let MtpState::Running(cohort) = &mut *state {
                            cohort.set_diagnostics(std::mem::take(&mut self.diagnostics))?;
                        }
                        Ok(ExpansionCohortProgress::Pending {
                            made_progress: true,
                        })
                    }
                    Err(error) => Err(error),
                },
                MtpState::Running(owner) => owner.advance(&self.cancelled, wake),
                MtpState::Closed => Err(FerrumError::internal("MTP owner is already closed")),
            }
        })();
        match &result {
            Ok(ExpansionCohortProgress::Retired(_)) => {
                // The machine already checked all native owners. Drop its
                // inert metadata before refunding the outer retained phase.
                *self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = MtpState::Closed;
                self.closing.store(true, Ordering::Release);
                self.reporter.retired(None);
            }
            Err(error) => {
                self.closing.store(true, Ordering::Release);
                self.reporter.retained(error.clone());
                close_original(&self.state, &self.reporter);
            }
            _ => {}
        }
        result
    }
}
impl<R: DeviceRuntime> Drop for MtpOwner<R> {
    fn drop(&mut self) {
        if self.observer.snapshot().phase()
            == ferrum_execution::executor_lifecycle::ExecutorShutdownPhase::Retired
        {
            return;
        }
        self.closing.store(true, Ordering::Release);
        // Dropping a cancelled consumer must not execute blocking backend
        // recovery on its thread. This cold cleanup pass owns the same state;
        // its observer can retry a retained original failure afterwards.
        let state = Arc::clone(&self.state);
        let reporter = self.reporter.clone();
        if let Err(error) = std::thread::Builder::new()
            .name("ferrum-mtp-cleanup".into())
            .spawn(move || close_original(&state, &reporter))
        {
            self.reporter
                .retained(FerrumError::backend(format!("MTP cleanup worker: {error}")));
        }
    }
}
fn close_original<R: DeviceRuntime>(
    state: &Mutex<MtpState<R>>,
    reporter: &ExecutorShutdownReporter,
) {
    let result = state
        .lock()
        .map_err(|_| FerrumError::internal("MTP cleanup lock poisoned"))
        .and_then(|mut state| state.try_close());
    match result {
        Ok(true) => reporter.retired(None),
        Ok(false) => reporter.retained(FerrumError::backend(
            "MTP original native cleanup remains outstanding",
        )),
        Err(error) => reporter.retained(error),
    }
}

pub(crate) struct ProductExpansionPhase<R: DeviceRuntime> {
    cohort: ProductExpansionCohort<R>,
}
impl<R: DeviceRuntime> ProductExpansionPhase<R> {
    pub(crate) fn initialize_with_diagnostics(
        assets: Arc<NativeTextExpansion>,
        composition: RuntimeComposition<R>,
        prepared: PreparedTextExpansion,
        phase: HostMemoryScope,
        diagnostic: Option<DiagnosticContext>,
    ) -> ExecutorBuildResult<Self> {
        let prepared = if assets.mtp == PH3ImagePromptExpansionMtp::Off {
            PreparedExpansionCohort::single(prepared)
        } else {
            assets.prepare_cohort(vec![prepared])?
        };
        ProductExpansionCohort::initialize_with_diagnostics(
            assets,
            composition,
            prepared,
            phase,
            vec![diagnostic],
        )
        .map(|cohort| Self { cohort })
    }
    pub(crate) fn cleanup(&self) -> ExecutorShutdownObserver {
        self.cohort.cleanup()
    }
    pub(crate) fn advance(
        &mut self,
        cancelled: bool,
        wake: &Waker,
    ) -> Result<phase::ExpansionPhaseProgress> {
        match self.cohort.advance(&[cancelled], wake)? {
            ExpansionCohortProgress::Pending { made_progress } => {
                Ok(phase::ExpansionPhaseProgress::Pending { made_progress })
            }
            ExpansionCohortProgress::Retired(mut results) => {
                if results.len() != 1 {
                    return Err(FerrumError::internal(
                        "single MTP adapter has another membership",
                    ));
                }
                match results.pop().unwrap() {
                    ExpansionParticipantResult::Expanded(text) => {
                        Ok(phase::ExpansionPhaseProgress::Expanded(text))
                    }
                    ExpansionParticipantResult::Cancelled => {
                        Ok(phase::ExpansionPhaseProgress::Cancelled)
                    }
                    ExpansionParticipantResult::Failed(error) => Err(error),
                }
            }
        }
    }
}
