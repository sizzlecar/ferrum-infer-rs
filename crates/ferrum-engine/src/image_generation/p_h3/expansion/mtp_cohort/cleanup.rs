//! Checked retirement of the two original plans and their single static owner.
//! The controller first drains waves, failures, admissions and sequences. This
//! helper never substitutes Drop, a reference count, or an observer for a close
//! receipt, and never destroys a failed close's retry authority.
use ferrum_execution::driver::{SharedProgramWeights, SharedProgramWeightsCloseOutcome};
use ferrum_execution::CausalProgramSession;
use ferrum_runtime::{
    DeviceRuntime, PlanRuntimeCloseFailure, PlanRuntimeCloseOutcome, PlanRuntimeResources,
    SharedComponentWeightsCloseFailure, MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS,
};
use ferrum_types::{FerrumError, Result};
use std::{mem, sync::Arc};

enum Program<R: DeviceRuntime> {
    Live(CausalProgramSession<R>),
    Referenced(Arc<PlanRuntimeResources<R>>),
    Failed(PlanRuntimeCloseFailure<R>),
    Closed,
    // A consuming close that unwound has no successful retirement proof.
    Transition,
}

enum Weights<R: DeviceRuntime> {
    Live(SharedProgramWeights<R>),
    Failed(SharedComponentWeightsCloseFailure<R>),
    Closed,
    Transition,
}

#[must_use = "MTP programs and shared weights require checked close, including retries"]
pub(super) struct ProgramOwners<R: DeviceRuntime> {
    main: Program<R>,
    draft: Program<R>,
    weights: Weights<R>,
    closing: bool,
}

/// Missing slots were never acquired; acquired slots retain original close
/// authority. Failed constructor cleanup is observed separately before these
/// successful partial owners may be released.
pub(super) struct PartialProgramOwners<R: DeviceRuntime> {
    installer: Option<Program<R>>,
    main: Option<Program<R>>,
    draft: Option<Program<R>>,
    weights: Option<Weights<R>>,
}
impl<R: DeviceRuntime> PartialProgramOwners<R> {
    pub(super) fn new() -> Self {
        Self {
            installer: None,
            main: None,
            draft: None,
            weights: None,
        }
    }
    pub(super) fn installed(
        &mut self,
        installer: CausalProgramSession<R>,
        weights: SharedProgramWeights<R>,
    ) {
        assert!(
            self.installer.is_none() && self.weights.is_none(),
            "installation is a single transition"
        );
        self.installer = Some(Program::Live(installer));
        self.weights = Some(Weights::Live(weights));
    }
    pub(super) fn attached_main(&mut self, main: CausalProgramSession<R>) {
        assert!(
            self.main.is_none(),
            "main attachment is a single transition"
        );
        self.main = Some(Program::Live(main));
    }
    pub(super) fn attached_draft(&mut self, draft: CausalProgramSession<R>) {
        assert!(
            self.draft.is_none(),
            "draft attachment is a single transition"
        );
        self.draft = Some(Program::Live(draft));
    }
    pub(super) fn weights(&self) -> Result<&SharedProgramWeights<R>> {
        match &self.weights {
            Some(Weights::Live(weights)) => Ok(weights),
            _ => Err(FerrumError::internal(
                "MTP original shared installation is unavailable",
            )),
        }
    }
    pub(super) fn close_installer(&mut self) -> Result<bool> {
        self.installer
            .as_mut()
            .ok_or_else(|| FerrumError::internal("MTP installer was not acquired"))?
            .poll_close()
    }
    pub(super) fn take_ready(
        &mut self,
    ) -> Result<(
        CausalProgramSession<R>,
        CausalProgramSession<R>,
        SharedProgramWeights<R>,
    )> {
        if !matches!(self.installer, Some(Program::Closed))
            || !matches!(self.main, Some(Program::Live(_)))
            || !matches!(self.draft, Some(Program::Live(_)))
            || !matches!(self.weights, Some(Weights::Live(_)))
        {
            return Err(FerrumError::internal(
                "MTP construction has not completed all original transitions",
            ));
        }
        let Some(Program::Live(main)) = self.main.take() else {
            unreachable!()
        };
        let Some(Program::Live(draft)) = self.draft.take() else {
            unreachable!()
        };
        let Some(Weights::Live(weights)) = self.weights.take() else {
            unreachable!()
        };
        Ok((main, draft, weights))
    }
    pub(super) fn poll_close(&mut self) -> Result<bool> {
        let close =
            |slot: &mut Option<Program<R>>| slot.as_mut().map_or(Ok(true), Program::poll_close);
        let installer = close(&mut self.installer);
        let main = close(&mut self.main);
        let draft = close(&mut self.draft);
        if !(installer? & main? & draft?) {
            return Ok(false);
        }
        self.weights.as_mut().map_or(Ok(true), Weights::poll_close)
    }
}

impl<R: DeviceRuntime> ProgramOwners<R> {
    pub(super) fn new(
        main: CausalProgramSession<R>,
        draft: CausalProgramSession<R>,
        weights: SharedProgramWeights<R>,
    ) -> Self {
        Self {
            main: Program::Live(main),
            draft: Program::Live(draft),
            weights: Weights::Live(weights),
            closing: false,
        }
    }

    pub(super) fn main(&self) -> Result<&CausalProgramSession<R>> {
        self.program(&self.main)
    }

    pub(super) fn draft(&self) -> Result<&CausalProgramSession<R>> {
        self.program(&self.draft)
    }

    fn program<'a>(&self, slot: &'a Program<R>) -> Result<&'a CausalProgramSession<R>> {
        match (self.closing, slot) {
            (false, Program::Live(program)) => Ok(program),
            _ => Err(FerrumError::backend(
                "MTP programs are closing or already closed",
            )),
        }
    }

    /// One bounded recovery pass, called only after request/wave owners have
    /// been settled. As with Core close/retry this may enter backend recovery;
    /// call on the controller's cleanup/recovery thread, not in a destructor.
    /// A referenced owner returns false. Errors retain their original failure
    /// owner here; the error string is only a report, never cleanup authority.
    pub(super) fn poll_close(&mut self) -> Result<bool> {
        self.closing = true;
        // Try both, even if one fails, so an independent plan can release its
        // owners. Every failure remains in its corresponding original slot.
        let main = self.main.poll_close();
        let draft = self.draft.poll_close();
        let main_closed = main?;
        let draft_closed = draft?;
        if !main_closed || !draft_closed {
            return Ok(false);
        }
        self.weights.poll_close()
    }
}

impl<R: DeviceRuntime> Program<R> {
    fn poll_close(&mut self) -> Result<bool> {
        let resources = match self {
            Self::Live(program) => Some(program.resources()),
            Self::Referenced(resources) => Some(&*resources),
            Self::Closed => return Ok(true),
            Self::Transition => return Err(unproved("plan close")),
            Self::Failed(_) => None,
        };
        if let Some(resources) = resources {
            resources
                .maintain_deferred_cleanups(MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS)
                .map_err(|error| FerrumError::backend(format!("MTP deferred cleanup: {error}")))?;
        }
        let result = match mem::replace(self, Self::Transition) {
            Self::Live(program) => program.close(),
            Self::Referenced(resources) => PlanRuntimeResources::close(resources),
            Self::Failed(failure) => failure.retry().map(PlanRuntimeCloseOutcome::Closed),
            _ => unreachable!("terminal states returned before consuming close"),
        };
        match result {
            Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
                *self = Self::Closed;
                Ok(true)
            }
            Ok(PlanRuntimeCloseOutcome::Referenced { resources, .. }) => {
                *self = Self::Referenced(resources);
                Ok(false)
            }
            Err(failure) => {
                let error =
                    FerrumError::backend(format!("MTP plan close: {:?}", failure.failure()));
                *self = Self::Failed(failure);
                Err(error)
            }
        }
    }
}

impl<R: DeviceRuntime> Weights<R> {
    fn poll_close(&mut self) -> Result<bool> {
        let result = match mem::replace(self, Self::Transition) {
            Self::Closed => {
                *self = Self::Closed;
                return Ok(true);
            }
            Self::Transition => return Err(unproved("shared weight close")),
            Self::Live(weights) => weights.close(),
            Self::Failed(failure) => failure
                .retry()
                .map(SharedProgramWeightsCloseOutcome::Closed),
        };
        match result {
            Ok(SharedProgramWeightsCloseOutcome::Closed(_)) => {
                *self = Self::Closed;
                Ok(true)
            }
            Ok(SharedProgramWeightsCloseOutcome::Referenced { weights }) => {
                *self = Self::Live(weights);
                Ok(false)
            }
            Err(failure) => {
                let error = FerrumError::backend(format!(
                    "MTP shared weight close: {:?}",
                    failure.failure()
                ));
                *self = Self::Failed(failure);
                Err(error)
            }
        }
    }
}

fn unproved(operation: &str) -> FerrumError {
    FerrumError::internal(format!(
        "MTP {operation} unwound without a close receipt; retirement is unproved"
    ))
}
