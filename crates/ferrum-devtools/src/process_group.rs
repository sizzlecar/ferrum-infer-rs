//! Ownership of a Unix child group, shared by release and process benchmarks.
use std::time::Duration;

pub(crate) struct ProcessGroup {
    pub(crate) child: tokio::process::Child,
    pub(crate) id: i32,
    pub(crate) armed: bool,
}

impl ProcessGroup {
    pub(crate) fn signal(&self, signal: i32) -> Result<(), String> {
        // The positive ID belongs to our freshly spawned group leader. A
        // negative PID addresses that group, never the controller's group.
        let result = unsafe { libc::kill(-self.id, signal) };
        if result == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(format!("signal owned process group: {error}"))
        }
    }

    pub(crate) async fn cleanup(&mut self) -> Result<(), String> {
        // A successful leader exit does not prove its descendants exited.
        let killed = self.signal(libc::SIGKILL);
        let waited = tokio::time::timeout(Duration::from_secs(5), self.child.wait())
            .await
            .map_err(|_| "runner did not reap after process-group kill".to_string())
            .and_then(|value| value.map_err(|error| format!("reap runner: {error}")));
        if killed.is_ok() && waited.is_ok() {
            self.armed = false;
        }
        killed.and(waited.map(|_| ()))
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.signal(libc::SIGKILL);
            let _ = self.child.start_kill();
        }
    }
}
