//! Explicit assembly ownership and one engine's executor use scope.

use std::sync::Arc;

use ferrum_execution::executor_lifecycle::{ExecutorOwner, ExecutorScope};
use ferrum_execution::ModelExecutor;
use ferrum_types::Result;

/// An engine either owns final retirement or borrows an explicitly issued scope.
/// No shared raw executor can manufacture final-close permission.
pub struct EngineExecutor {
    pub(crate) scope: Arc<ExecutorScope>,
    pub(crate) owner: Option<ExecutorOwner>,
}

pub trait IntoEngineExecutor {
    fn into_engine_executor(self) -> Result<EngineExecutor>;
}

impl IntoEngineExecutor for ExecutorOwner {
    fn into_engine_executor(self) -> Result<EngineExecutor> {
        Ok(EngineExecutor {
            scope: Arc::new(self.scope()?),
            owner: Some(self),
        })
    }
}

impl IntoEngineExecutor for ExecutorScope {
    fn into_engine_executor(self) -> Result<EngineExecutor> {
        Ok(EngineExecutor {
            scope: Arc::new(self),
            owner: None,
        })
    }
}

impl IntoEngineExecutor for EngineExecutor {
    fn into_engine_executor(self) -> Result<EngineExecutor> {
        Ok(self)
    }
}

impl std::ops::Deref for EngineExecutor {
    type Target = dyn ModelExecutor;
    fn deref(&self) -> &Self::Target {
        self.scope.as_ref().as_ref()
    }
}

impl AsRef<dyn ModelExecutor> for EngineExecutor {
    fn as_ref(&self) -> &(dyn ModelExecutor + 'static) {
        self.scope.as_ref().as_ref()
    }
}
