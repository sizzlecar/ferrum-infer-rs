//! Protocol-independent output admission immediately before preparation.
use super::*;
use std::task::{Wake, Waker};

/// An owned output grant or subscription. The service never interprets its
/// protocol-specific payload; dropping it refunds the originating authority.
pub struct ImageOutputLease(Box<dyn Any + Send>);
impl fmt::Debug for ImageOutputLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ImageOutputLease(opaque owner)")
    }
}
impl ImageOutputLease {
    pub fn new<T: Any + Send>(owner: T) -> Self {
        Self(Box::new(owner))
    }
    pub fn into_inner<T: Any + Send>(self) -> Result<T, Self> {
        self.0.downcast::<T>().map(|owner| *owner).map_err(Self)
    }
}

#[derive(Debug)]
pub enum ImageOutputGateError {
    /// A single output exceeds an immutable configured bound.
    Capacity(String),
    Failed(String),
}

/// A job's output resource requirement, independent of any wire format.
/// `None` means another live owner must release capacity. Implementations
/// validate immutable single-output limits before returning temporary pressure.
pub trait ImageOutputGate: Send + Sync + 'static {
    /// Physical Output bytes charged to the service's Host parent by one live
    /// grant. Gates using a separate non-Host authority leave this at zero.
    fn host_output_bytes(&self, _host: &HostMemoryAccount) -> usize {
        0
    }
    fn try_acquire(&self) -> Result<Option<ImageOutputLease>, ImageOutputGateError>;
    /// Keep this subscription alive from before the admission attempt through
    /// the wait, so an intervening release cannot be missed.
    fn subscribe_capacity(&self, waker: Waker) -> ImageOutputLease;
}

pub(super) struct QueuedOutput {
    pub gate: Arc<dyn ImageOutputGate>,
    pub lease: Option<ImageOutputLease>,
    pub response: oneshot::Sender<ImageOutputLease>,
}

/// Separate from the queue mutex: resource owners are routinely dropped while
/// updating queue state, and their refund callbacks must never lock that state.
#[derive(Default)]
pub(super) struct ResourceSignal {
    epoch: Mutex<u64>,
    changed: Condvar,
}
impl ResourceSignal {
    pub fn epoch(&self) -> u64 {
        *self.epoch.lock().unwrap_or_else(|error| error.into_inner())
    }
    pub fn notify(&self) {
        let mut epoch = self.epoch.lock().unwrap_or_else(|error| error.into_inner());
        *epoch = epoch.wrapping_add(1);
        self.changed.notify_all();
    }
    pub fn wait(&self, observed: u64) {
        let mut epoch = self.epoch.lock().unwrap_or_else(|error| error.into_inner());
        while *epoch == observed {
            epoch = self
                .changed
                .wait(epoch)
                .unwrap_or_else(|error| error.into_inner());
        }
    }
    pub fn waker(self: &Arc<Self>) -> Waker {
        Waker::from(self.clone())
    }
}
impl Wake for ResourceSignal {
    fn wake(self: Arc<Self>) {
        self.notify();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.notify();
    }
}
