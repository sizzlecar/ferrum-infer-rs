//! Original context observation before partial construction owners unwind.
use ferrum_kernels::backend::{Backend, BackendCompletion};
use ferrum_types::Result;

/// Construction has no recovery service yet. Declare this guard after the
/// partially constructed resource payload so an error observes the original
/// context before those resources unwind. It never substitutes another stream.
pub(crate) struct ConstructionContext<'a, B: Backend> {
    pub(crate) context: &'a mut B::Context,
    armed: bool,
}

impl<'a, B: Backend> ConstructionContext<'a, B> {
    pub(crate) fn new(context: &'a mut B::Context) -> Self {
        Self {
            context,
            armed: true,
        }
    }

    pub(crate) fn transfer(&mut self) {
        self.armed = false;
    }

    pub(crate) fn complete(&mut self) -> Result<()> {
        if B::preparation_is_host_only(self.context) {
            self.armed = false;
            return Ok(());
        }
        match B::observe_completion(self.context) {
            BackendCompletion::Succeeded => {
                self.armed = false;
                Ok(())
            }
            BackendCompletion::Failed(error) => {
                self.armed = false;
                Err(error)
            }
            BackendCompletion::Unresolved(error) => Err(error),
        }
    }
}

impl<B: Backend> Drop for ConstructionContext<'_, B> {
    fn drop(&mut self) {
        if self.armed {
            let retired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                B::retire_context(self.context)
            }));
            if !matches!(retired, Ok(Ok(()))) {
                eprintln!("cannot release partially constructed model: original context retirement failed");
                std::process::abort();
            }
        }
    }
}
