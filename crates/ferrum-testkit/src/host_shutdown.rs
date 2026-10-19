//! Consuming close for synchronous host fixtures. A unique original Arc proves
//! no borrowed execution future is active; the fixture then ends its own exact
//! token incarnations and releases its resources before publishing retirement.

use ferrum_execution::executor_lifecycle::{
    ExecutorShutdownObserver, ExecutorShutdownPermit, ExecutorShutdownReporter,
};
use ferrum_execution::ModelExecutor;
use ferrum_types::{FerrumError, Result};
use parking_lot::Mutex;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

enum Owner<T> {
    Shared(Option<Arc<T>>),
    Unique(Option<Box<T>>),
    Retired,
}

struct Close<T> {
    owner: Mutex<Owner<T>>,
    retire: fn(&mut T) -> Result<()>,
}

pub(crate) fn begin<T: ModelExecutor>(
    executor: Arc<T>,
    permit: &ExecutorShutdownPermit,
    retire: fn(&mut T) -> Result<()>,
) -> ExecutorShutdownObserver {
    if let Err(error) = permit.check(executor.as_ref()) {
        return ExecutorShutdownObserver::retained(Arc::new(executor), error);
    }
    let custody = Arc::new(Close {
        owner: Mutex::new(Owner::Shared(Some(executor))),
        retire,
    });
    let (reporter, observer) = ExecutorShutdownObserver::channel(custody.clone());
    let weak = Arc::downgrade(&custody);
    let retry_reporter = reporter.clone();
    let observer = observer.with_retry(Arc::new(move || {
        if let Some(custody) = weak.upgrade() {
            custody.advance(&retry_reporter);
        }
    }));
    custody.advance(&reporter);
    observer
}

impl<T: ModelExecutor> Close<T> {
    fn advance(&self, reporter: &ExecutorShutdownReporter) {
        let mut owner = self.owner.lock();
        let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
            if let Owner::Shared(original) = &mut *owner {
                match Arc::try_unwrap(original.take().expect("original host executor")) {
                    Ok(executor) => *owner = Owner::Unique(Some(Box::new(executor))),
                    Err(executor) => {
                        *original = Some(executor);
                        return Err(FerrumError::backend(
                            "host executor still has live references",
                        ));
                    }
                }
            }
            if let Owner::Unique(original) = &mut *owner {
                let executor = original.as_mut().ok_or_else(|| {
                    FerrumError::internal(
                        "host executor destructor unwound without a retirement receipt",
                    )
                })?;
                (self.retire)(executor)?;
                // All host work is synchronous and all exact token owners have
                // ended. Drop the original fixture before publishing Retired.
                drop(original.take());
                *owner = Owner::Retired;
            }
            Ok(())
        }));
        match result {
            Ok(Ok(())) => reporter.retired(None),
            Ok(Err(error)) => reporter.retained(error),
            Err(_) => reporter.retained(FerrumError::internal(
                "host executor retirement panicked; original custody is retained",
            )),
        }
    }
}
