//! Diagnostic custody of failures that may still own submitted native work.
use std::any::Any;

pub struct Failure {
    pub stage: &'static str,
    pub message: String,
    pub owner: Box<dyn Any>,
}
impl Failure {
    pub fn retain<E: std::fmt::Display + 'static, O: 'static>(
        stage: &'static str,
        error: E,
        owner: O,
    ) -> Self {
        let owner = Box::new((error, owner));
        // Formatting diagnostics must not unwind through submitted owners.
        let message =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| owner.0.to_string()))
                .unwrap_or_else(|_| "error Display panicked; original owner retained".into());
        Self {
            stage,
            message,
            owner,
        }
    }
}
#[test]
fn diagnostic_display_panic_preserves_original_failure_owner() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    struct Owner(Arc<AtomicBool>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    struct BrokenDisplay;
    impl std::fmt::Display for BrokenDisplay {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            panic!("diagnostic formatting failure");
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let failure = Failure::retain("native", BrokenDisplay, Owner(dropped.clone()));
    assert!(!dropped.load(Ordering::SeqCst));
    assert!(failure.message.contains("Display panicked"));
    drop(failure);
    assert!(dropped.load(Ordering::SeqCst));
}
