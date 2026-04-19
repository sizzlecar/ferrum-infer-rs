//! Tensor Parallel Decode Group — persistent threads + barrier sync.
//!
//! Performance-critical decode path uses std::sync::Barrier (zero allocation).
//! Control operations (KV init/release) use mpsc channels (infrequent).
//! cudarc event tracking is disabled per-worker (all ops on same stream).

#[cfg(feature = "cuda")]
use crate::cuda_decode::CudaDecodeRunner;
#[cfg(feature = "cuda")]
use crate::nccl_comm::NcclRank;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
use std::collections::HashSet;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
#[cfg(feature = "cuda")]
use std::sync::{mpsc, Arc, Barrier};
#[cfg(feature = "cuda")]
use std::thread::JoinHandle;
#[cfg(feature = "cuda")]
use std::time::Duration;

// Mode constants for shared state
#[cfg(feature = "cuda")]
const MODE_DECODE: u8 = 0;
#[cfg(feature = "cuda")]
const MODE_CONTROL: u8 = 1;
#[cfg(feature = "cuda")]
const MODE_SHUTDOWN: u8 = 2;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// How KV cache data is provided to a worker thread.
#[cfg(feature = "cuda")]
pub enum KvSource {
    /// Already on the correct GPU (e.g., rank 0 sharing candle's device).
    Gpu(CudaSlice<half::f16>),
    /// Host memory — worker will upload via H2D.
    Host(Vec<half::f16>),
}

// ---------------------------------------------------------------------------
// Shared state (hot path — zero allocation)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
struct SharedDecode {
    // Main → Workers (written before start_barrier, read after)
    mode: AtomicU8,
    token_id: AtomicU32,
    position: AtomicU64,
    cache_key: std::sync::RwLock<String>,
    // Workers → Main (written before done_barrier, read after)
    logits: std::sync::Mutex<Option<CudaSlice<half::f16>>>,
    error: std::sync::Mutex<Option<String>>,
}

// ---------------------------------------------------------------------------
// Control channel types (cold path)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
enum TpCommand {
    InitKvCache {
        cache_key: String,
        kv_data: Vec<(KvSource, KvSource)>,
        prefill_len: usize,
        max_len: usize,
    },
    ReleaseKvCache {
        cache_key: String,
    },
}

// ---------------------------------------------------------------------------
// TpDecodeGroup
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub struct TpDecodeGroup {
    // Hot path: barrier sync
    start_barrier: Arc<Barrier>,
    done_barrier: Arc<Barrier>,
    shared: Arc<SharedDecode>,
    // Cold path: per-rank control channels
    cmd_txs: Vec<mpsc::SyncSender<TpCommand>>,
    // Lifecycle
    workers: Vec<Option<JoinHandle<()>>>,
    world_size: usize,
    cache_keys: HashSet<String>,
}

#[cfg(feature = "cuda")]
impl TpDecodeGroup {
    pub fn new(runners: Vec<CudaDecodeRunner>, nccl: Vec<NcclRank>) -> candle_core::Result<Self> {
        let world_size = runners.len();
        if nccl.len() != world_size || world_size == 0 {
            return Err(candle_core::Error::Msg(format!(
                "runners ({}) != nccl ({}) or zero",
                world_size,
                nccl.len()
            )));
        }

        let parties = world_size + 1; // workers + main
        let start_barrier = Arc::new(Barrier::new(parties));
        let done_barrier = Arc::new(Barrier::new(parties));
        let shared = Arc::new(SharedDecode {
            mode: AtomicU8::new(MODE_DECODE),
            token_id: AtomicU32::new(0),
            position: AtomicU64::new(0),
            cache_key: std::sync::RwLock::new(String::new()),
            logits: std::sync::Mutex::new(None),
            error: std::sync::Mutex::new(None),
        });

        let mut cmd_txs = Vec::with_capacity(world_size);
        let mut workers = Vec::with_capacity(world_size);

        for (rank, (runner, nccl_rank)) in runners.into_iter().zip(nccl.into_iter()).enumerate() {
            let (cmd_tx, cmd_rx) = mpsc::sync_channel::<TpCommand>(1);
            let sb = Arc::clone(&start_barrier);
            let db = Arc::clone(&done_barrier);
            let sh = Arc::clone(&shared);

            let handle = std::thread::Builder::new()
                .name(format!("tp-rank-{rank}"))
                .spawn(move || worker_loop(rank, runner, nccl_rank, cmd_rx, sb, db, sh))
                .map_err(|e| candle_core::Error::Msg(format!("spawn rank {rank}: {e}")))?;

            cmd_txs.push(cmd_tx);
            workers.push(Some(handle));
        }

        Ok(Self {
            start_barrier,
            done_barrier,
            shared,
            cmd_txs,
            workers,
            world_size,
            cache_keys: HashSet::new(),
        })
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Tensor-parallel decode step. Returns logits from rank 0.
    /// Hot path: barrier sync + atomics, zero allocation per step.
    pub fn decode_step(
        &mut self,
        token_id: u32,
        position: usize,
        cache_key: &str,
    ) -> candle_core::Result<CudaSlice<half::f16>> {
        // Set decode params
        self.shared.mode.store(MODE_DECODE, Ordering::Release);
        self.shared.token_id.store(token_id, Ordering::Release);
        self.shared
            .position
            .store(position as u64, Ordering::Release);
        {
            let mut key = self.shared.cache_key.write().unwrap();
            if *key != cache_key {
                key.clear();
                key.push_str(cache_key);
            }
        }
        *self.shared.error.lock().unwrap() = None;

        // Signal workers to start
        self.start_barrier.wait();
        // Wait for all workers to finish
        self.done_barrier.wait();

        // Check error
        if let Some(err) = self.shared.error.lock().unwrap().take() {
            return Err(candle_core::Error::Msg(err));
        }

        // Take logits from rank 0
        self.shared
            .logits
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| candle_core::Error::Msg("no logits from rank 0".into()))
    }

    /// Initialize KV cache on all ranks (cold path, uses channels).
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

        // Send per-rank data BEFORE barrier (workers read after barrier)
        for (tx, kv_data) in self.cmd_txs.iter().zip(kv_data_per_rank.into_iter()) {
            tx.send(TpCommand::InitKvCache {
                cache_key: cache_key.to_string(),
                kv_data,
                prefill_len,
                max_len,
            })
            .map_err(|_| candle_core::Error::Msg("worker dead".into()))?;
        }

        self.shared.mode.store(MODE_CONTROL, Ordering::Release);
        *self.shared.error.lock().unwrap() = None;

        self.start_barrier.wait();
        self.done_barrier.wait();

        if let Some(err) = self.shared.error.lock().unwrap().take() {
            return Err(candle_core::Error::Msg(err));
        }

        self.cache_keys.insert(cache_key.to_string());
        Ok(())
    }

    /// Release KV cache on all ranks (cold path).
    pub fn release_kv_cache(&mut self, cache_key: &str) {
        self.cache_keys.remove(cache_key);
        for tx in &self.cmd_txs {
            let _ = tx.send(TpCommand::ReleaseKvCache {
                cache_key: cache_key.to_string(),
            });
        }
        self.shared.mode.store(MODE_CONTROL, Ordering::Release);
        self.start_barrier.wait();
        self.done_barrier.wait();
    }

    pub fn has_kv_cache(&self, cache_key: &str) -> bool {
        self.cache_keys.contains(cache_key)
    }
}

#[cfg(feature = "cuda")]
impl Drop for TpDecodeGroup {
    fn drop(&mut self) {
        self.shared.mode.store(MODE_SHUTDOWN, Ordering::Release);
        self.start_barrier.wait(); // workers see shutdown and break
                                   // Workers do NOT call done_barrier on shutdown — just join them.
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

#[cfg(feature = "cuda")]
fn worker_loop(
    rank: usize,
    mut runner: CudaDecodeRunner,
    nccl: NcclRank,
    cmd_rx: mpsc::Receiver<TpCommand>,
    start_barrier: Arc<Barrier>,
    done_barrier: Arc<Barrier>,
    shared: Arc<SharedDecode>,
) {
    // Bind CUDA context once
    if let Err(e) = runner.bind_context() {
        eprintln!("[tp-rank-{rank}] bind_context failed: {e}");
        return;
    }

    // Disable cudarc event tracking — all ops on same stream, no cross-stream sync needed.
    // This eliminates ~500µs overhead per NCCL call from DevicePtrMut::device_ptr_mut().
    runner.disable_event_tracking();

    loop {
        start_barrier.wait(); // wait for main signal

        let mode = shared.mode.load(Ordering::Acquire);

        if mode == MODE_SHUTDOWN {
            break; // no done_barrier on shutdown
        }

        if mode == MODE_DECODE {
            let token_id = shared.token_id.load(Ordering::Acquire);
            let position = shared.position.load(Ordering::Acquire) as usize;
            let cache_key = shared.cache_key.read().unwrap();

            let res = run_decode_step(&mut runner, &nccl, token_id, position, &cache_key);
            drop(cache_key); // release read lock before writing results

            match res {
                Ok(()) if rank == 0 => {
                    match runner.sync_stream().and_then(|_| runner.clone_logits()) {
                        Ok(l) => *shared.logits.lock().unwrap() = Some(l),
                        Err(e) => *shared.error.lock().unwrap() = Some(format!("{e}")),
                    }
                }
                Err(e) => {
                    *shared.error.lock().unwrap() = Some(format!("rank {rank}: {e}"));
                }
                _ => {}
            }
        } else {
            // MODE_CONTROL: read command from channel
            if let Ok(cmd) = cmd_rx.recv_timeout(Duration::from_secs(10)) {
                let err = match cmd {
                    TpCommand::InitKvCache {
                        ref cache_key,
                        kv_data,
                        prefill_len,
                        max_len,
                    } => {
                        init_kv_on_rank(&mut runner, cache_key, kv_data, prefill_len, max_len).err()
                    }
                    TpCommand::ReleaseKvCache { ref cache_key } => {
                        runner.release_kv_cache(cache_key);
                        None
                    }
                };
                if let Some(e) = err {
                    *shared.error.lock().unwrap() = Some(format!("rank {rank}: {e}"));
                }
            }
        }

        done_barrier.wait(); // signal completion
    }
}

/// Execute full decode pipeline on one rank. NCCL provides inter-rank sync.
#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
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
