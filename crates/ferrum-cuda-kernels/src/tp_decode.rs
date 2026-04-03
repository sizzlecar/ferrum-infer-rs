//! Tensor Parallel Decode Group — persistent-thread architecture.
//!
//! Each GPU rank runs on its own persistent OS thread. The main thread sends
//! commands via per-rank channels; workers execute the full decode pipeline
//! with NCCL all-reduce providing inter-rank synchronization.
//!
//! This eliminates the per-step thread::scope overhead (was 72 spawns/step,
//! now 0 — threads are created once at init and reused for all steps).

#[cfg(feature = "tensor-parallel")]
use crate::cuda_decode::CudaDecodeRunner;
#[cfg(feature = "tensor-parallel")]
use crate::nccl_comm::NcclRank;
#[cfg(feature = "tensor-parallel")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "tensor-parallel")]
use std::collections::HashSet;
#[cfg(feature = "tensor-parallel")]
use std::sync::mpsc;
#[cfg(feature = "tensor-parallel")]
use std::thread::JoinHandle;
#[cfg(feature = "tensor-parallel")]
use std::time::Duration;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// How KV cache data is provided to a worker thread.
#[cfg(feature = "tensor-parallel")]
pub enum KvSource {
    /// Already on the correct GPU (e.g., rank 0 sharing candle's device).
    Gpu(CudaSlice<half::f16>),
    /// Host memory — worker will upload via H2D.
    Host(Vec<half::f16>),
}

// ---------------------------------------------------------------------------
// Internal command / result protocol
// ---------------------------------------------------------------------------

#[cfg(feature = "tensor-parallel")]
enum TpCommand {
    Decode {
        token_id: u32,
        position: usize,
        cache_key: String,
    },
    InitKvCache {
        cache_key: String,
        kv_data: Vec<(KvSource, KvSource)>,
        prefill_len: usize,
        max_len: usize,
    },
    ReleaseKvCache {
        cache_key: String,
    },
    Shutdown,
}

#[cfg(feature = "tensor-parallel")]
struct TpResult {
    rank: usize,
    logits: Option<CudaSlice<half::f16>>,
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// TpDecodeGroup
// ---------------------------------------------------------------------------

/// Tensor parallel decode group with persistent per-rank threads.
///
/// Workers are spawned once at construction and reused for every decode step.
/// NCCL all-reduce is the only inter-rank synchronization — no thread::scope.
#[cfg(feature = "tensor-parallel")]
pub struct TpDecodeGroup {
    cmd_txs: Vec<mpsc::SyncSender<TpCommand>>,
    result_rx: mpsc::Receiver<TpResult>,
    workers: Vec<Option<JoinHandle<()>>>,
    world_size: usize,
    /// Tracked on main thread to avoid round-trip for has_kv_cache().
    cache_keys: HashSet<String>,
}

#[cfg(feature = "tensor-parallel")]
impl TpDecodeGroup {
    /// Spawn persistent worker threads and return the group handle.
    ///
    /// Runners and NCCL ranks are moved into their respective threads.
    /// All subsequent interaction goes through channels.
    pub fn new(runners: Vec<CudaDecodeRunner>, nccl: Vec<NcclRank>) -> candle_core::Result<Self> {
        let world_size = runners.len();
        if nccl.len() != world_size || world_size == 0 {
            return Err(candle_core::Error::Msg(format!(
                "runners ({}) != nccl ({}) or zero",
                world_size,
                nccl.len()
            )));
        }

        let (result_tx, result_rx) = mpsc::sync_channel::<TpResult>(world_size);
        let mut cmd_txs = Vec::with_capacity(world_size);
        let mut workers = Vec::with_capacity(world_size);

        for (rank, (runner, nccl_rank)) in
            runners.into_iter().zip(nccl.into_iter()).enumerate()
        {
            // Bounded(1): main thread blocks until worker consumes previous cmd
            let (cmd_tx, cmd_rx) = mpsc::sync_channel::<TpCommand>(1);
            let res_tx = result_tx.clone();

            let handle = std::thread::Builder::new()
                .name(format!("tp-rank-{rank}"))
                .spawn(move || worker_loop(rank, runner, nccl_rank, cmd_rx, res_tx))
                .map_err(|e| candle_core::Error::Msg(format!("spawn rank {rank}: {e}")))?;

            cmd_txs.push(cmd_tx);
            workers.push(Some(handle));
        }

        // Drop the original sender so result_rx sees hangup when all workers exit.
        drop(result_tx);

        Ok(Self {
            cmd_txs,
            result_rx,
            workers,
            world_size,
            cache_keys: HashSet::new(),
        })
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Tensor-parallel decode step. Returns logits from rank 0.
    pub fn decode_step(
        &mut self,
        token_id: u32,
        position: usize,
        cache_key: &str,
    ) -> candle_core::Result<CudaSlice<half::f16>> {
        // Broadcast decode command to all ranks
        for tx in &self.cmd_txs {
            tx.send(TpCommand::Decode {
                token_id,
                position,
                cache_key: cache_key.to_string(),
            })
            .map_err(|_| candle_core::Error::Msg("worker dead".into()))?;
        }

        self.collect_decode_results()
    }

    /// Initialize KV cache on all ranks.
    ///
    /// Each rank receives its own shard of KV data. Workers handle H2D upload
    /// for `KvSource::Host` entries.
    pub fn init_kv_cache(
        &mut self,
        cache_key: &str,
        kv_data_per_rank: Vec<Vec<(KvSource, KvSource)>>,
        prefill_len: usize,
        max_len: usize,
    ) -> candle_core::Result<()> {
        if kv_data_per_rank.len() != self.world_size {
            return Err(candle_core::Error::Msg(format!(
                "kv_data ranks {} != world_size {}",
                kv_data_per_rank.len(),
                self.world_size
            )));
        }

        for (tx, kv_data) in self.cmd_txs.iter().zip(kv_data_per_rank.into_iter()) {
            tx.send(TpCommand::InitKvCache {
                cache_key: cache_key.to_string(),
                kv_data,
                prefill_len,
                max_len,
            })
            .map_err(|_| candle_core::Error::Msg("worker dead".into()))?;
        }

        self.collect_ack_results()?;
        self.cache_keys.insert(cache_key.to_string());
        Ok(())
    }

    /// Release KV cache on all ranks.
    pub fn release_kv_cache(&mut self, cache_key: &str) {
        self.cache_keys.remove(cache_key);
        for tx in &self.cmd_txs {
            let _ = tx.send(TpCommand::ReleaseKvCache {
                cache_key: cache_key.to_string(),
            });
        }
        // Wait for ack so results don't leak into the next decode_step.
        let _ = self.collect_ack_results();
    }

    /// Check if KV cache exists (tracked on main thread, no round-trip).
    pub fn has_kv_cache(&self, cache_key: &str) -> bool {
        self.cache_keys.contains(cache_key)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Collect world_size results, returning logits from rank 0.
    fn collect_decode_results(&self) -> candle_core::Result<CudaSlice<half::f16>> {
        let mut logits: Option<CudaSlice<half::f16>> = None;

        for _ in 0..self.world_size {
            let res = self
                .result_rx
                .recv_timeout(Duration::from_secs(30))
                .map_err(|e| candle_core::Error::Msg(format!("worker timeout/dead: {e}")))?;

            if let Some(err) = res.error {
                return Err(candle_core::Error::Msg(format!(
                    "rank {} error: {err}",
                    res.rank
                )));
            }
            if let Some(l) = res.logits {
                logits = Some(l);
            }
        }

        logits.ok_or_else(|| candle_core::Error::Msg("no logits from rank 0".into()))
    }

    /// Collect world_size ack results (no logits expected).
    fn collect_ack_results(&self) -> candle_core::Result<()> {
        for _ in 0..self.world_size {
            let res = self
                .result_rx
                .recv_timeout(Duration::from_secs(30))
                .map_err(|e| candle_core::Error::Msg(format!("worker timeout/dead: {e}")))?;

            if let Some(err) = res.error {
                return Err(candle_core::Error::Msg(format!(
                    "rank {} error: {err}",
                    res.rank
                )));
            }
        }
        Ok(())
    }
}

#[cfg(feature = "tensor-parallel")]
impl Drop for TpDecodeGroup {
    fn drop(&mut self) {
        for tx in &self.cmd_txs {
            let _ = tx.send(TpCommand::Shutdown);
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.take() {
                let _ = handle.join();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

/// Per-rank worker loop. Binds CUDA context once, then processes commands.
#[cfg(feature = "tensor-parallel")]
fn worker_loop(
    rank: usize,
    mut runner: CudaDecodeRunner,
    nccl: NcclRank,
    cmd_rx: mpsc::Receiver<TpCommand>,
    result_tx: mpsc::SyncSender<TpResult>,
) {
    // Bind CUDA context once for this thread's lifetime.
    if let Err(e) = runner.bind_context() {
        let _ = result_tx.send(TpResult {
            rank,
            logits: None,
            error: Some(format!("bind_context: {e}")),
        });
        return;
    }

    loop {
        let cmd = match cmd_rx.recv() {
            Ok(cmd) => cmd,
            Err(_) => break, // channel closed
        };

        match cmd {
            TpCommand::Decode {
                token_id,
                position,
                ref cache_key,
            } => {
                let res = run_decode_step(&mut runner, &nccl, token_id, position, cache_key);
                let (logits, error) = match res {
                    Ok(()) if rank == 0 => {
                        match runner.sync_stream().and_then(|_| runner.clone_logits()) {
                            Ok(l) => (Some(l), None),
                            Err(e) => (None, Some(format!("{e}"))),
                        }
                    }
                    Ok(()) => (None, None),
                    Err(e) => (None, Some(format!("{e}"))),
                };
                let _ = result_tx.send(TpResult {
                    rank,
                    logits,
                    error,
                });
            }
            TpCommand::InitKvCache {
                ref cache_key,
                kv_data,
                prefill_len,
                max_len,
            } => {
                let error =
                    match init_kv_on_rank(&mut runner, cache_key, kv_data, prefill_len, max_len) {
                        Ok(()) => None,
                        Err(e) => Some(format!("{e}")),
                    };
                let _ = result_tx.send(TpResult {
                    rank,
                    logits: None,
                    error,
                });
            }
            TpCommand::ReleaseKvCache { ref cache_key } => {
                runner.release_kv_cache(cache_key);
                let _ = result_tx.send(TpResult {
                    rank,
                    logits: None,
                    error: None,
                });
            }
            TpCommand::Shutdown => break,
        }
    }
}

/// Execute full decode pipeline on one rank. NCCL provides inter-rank sync.
///
/// No thread::scope needed — each rank thread calls NCCL all-reduce directly,
/// and NCCL blocks until all ranks participate (implicit barrier).
#[cfg(feature = "tensor-parallel")]
fn run_decode_step(
    runner: &mut CudaDecodeRunner,
    nccl: &NcclRank,
    token_id: u32,
    position: usize,
    cache_key: &str,
) -> candle_core::Result<()> {
    let n = runner.weight_layers().len();

    runner.tp_embed(token_id)?;
    runner.tp_first_norm()?;

    for li in 0..n {
        runner.tp_pre_o_proj(li, position, cache_key)?;
        runner.tp_o_proj(li)?;
        nccl.all_reduce_f16_inplace(runner.o_proj_out_mut())?;
        runner.tp_post_attn_norm(li)?;
        runner.tp_mlp(li)?;
        nccl.all_reduce_f16_inplace(runner.down_out_mut())?;
        runner.tp_post_mlp_norm(li)?;
    }

    runner.tp_lm_head()?;
    Ok(())
}

/// Upload KV data and initialize cache on one rank.
#[cfg(feature = "tensor-parallel")]
fn init_kv_on_rank(
    runner: &mut CudaDecodeRunner,
    cache_key: &str,
    kv_data: Vec<(KvSource, KvSource)>,
    prefill_len: usize,
    max_len: usize,
) -> candle_core::Result<()> {
    let gpu_kv: Vec<(CudaSlice<half::f16>, CudaSlice<half::f16>)> = kv_data
        .into_iter()
        .map(|(k, v)| {
            let k_gpu = match k {
                KvSource::Gpu(s) => s,
                KvSource::Host(data) => runner.upload_to_self(&data)?,
            };
            let v_gpu = match v {
                KvSource::Gpu(s) => s,
                KvSource::Host(data) => runner.upload_to_self(&data)?,
            };
            Ok((k_gpu, v_gpu))
        })
        .collect::<candle_core::Result<_>>()?;

    runner.init_kv_cache(cache_key, gpu_kv, prefill_len, max_len)
}
