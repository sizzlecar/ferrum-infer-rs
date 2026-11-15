//! A rendezvous, not a second device lane. At most the CPU's current readback
//! and the producer's next readback coexist. Both keep their charged owners.
use std::{io, sync::mpsc::SyncSender, thread};

pub(super) enum ProductionError<E> {
    Original(E),
    ConsumerStopped,
}

/// Failure to start the CPU thread happens before `produce` is called, so the
/// caller can use its original inline path. Every other outcome joins the CPU
/// worker before returning the producer's original owning native error.
pub(super) fn run<T: Send, E: From<anyhow::Error>>(
    produce: impl FnOnce(&SyncSender<T>) -> Result<(), ProductionError<E>>,
    mut process: impl FnMut(T) -> anyhow::Result<()> + Send,
) -> io::Result<Result<(), E>> {
    thread::scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::sync_channel(0);
        let worker = thread::Builder::new()
            .name("ferrum-video-postprocess".into())
            .spawn_scoped(scope, move || {
                for value in receiver {
                    process(value)?;
                }
                Ok::<(), anyhow::Error>(())
            })?;
        let produced = produce(&sender);
        drop(sender);
        let processed = worker
            .join()
            .unwrap_or_else(|_| Err(anyhow::anyhow!("video Host postprocessor panicked")));
        Ok(match produced {
            Err(ProductionError::Original(error)) => Err(error),
            Err(ProductionError::ConsumerStopped) => Err(processed
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("video Host consumer stopped before delivery"))
                .into()),
            Ok(()) => processed.map_err(Into::into),
        })
    })
}

#[cfg(test)]
mod tests;
