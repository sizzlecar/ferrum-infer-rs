//! Opt-in CPU command execution for engine protocol fixtures.
use super::*;
use std::sync::atomic::AtomicUsize;

/// Two real CPU implementations selected from original row contexts. This is
/// deliberately opt-in; the default fixtures retain their original algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ControlledCpuFillLayout {
    Uniform,
    Ragged,
    /// Each bit selects the packed-fill implementation for that original row.
    /// The command executes both primitives when its checked roster contains A+B.
    PerRow {
        packed_rows: u64,
    },
}
impl ControlledCpuFillLayout {
    pub(crate) fn for_contexts(contexts: impl IntoIterator<Item = usize>) -> Self {
        let mut contexts = contexts.into_iter();
        let first = contexts.next();
        if contexts.any(|context| Some(context) != first) {
            Self::Ragged
        } else {
            Self::Uniform
        }
    }

    pub(crate) fn native_op_id(self, full_logits: bool) -> &'static str {
        match (self, full_logits) {
            (Self::Uniform, true) => "fixture.controlled.full_logits_fill",
            (Self::Uniform, false) => "fixture.controlled.greedy_logits_fill",
            (Self::Ragged, true) => "fixture.controlled.ragged_full_logits_fill",
            (Self::Ragged, false) => "fixture.controlled.ragged_greedy_logits_fill",
            (Self::PerRow { .. }, true) => "fixture.controlled.row_selected_full_logits_fill",
            (Self::PerRow { .. }, false) => "fixture.controlled.row_selected_greedy_logits_fill",
        }
    }

    pub(crate) fn per_row(contexts: impl IntoIterator<Item = usize>) -> Self {
        let mut packed_rows = 0;
        for (row, context) in contexts.into_iter().enumerate() {
            assert!(row < 64, "bounded CPU row-selection fixture");
            if context > 1 {
                packed_rows |= 1 << row;
            }
        }
        Self::PerRow { packed_rows }
    }

    /// Nonempty entries are actual scalar-fill and packed-fill invocations.
    /// Keep their row partition identical in execution and cost attribution.
    pub(crate) fn primitive_rows(self, rows: usize) -> [(Self, usize); 2] {
        assert!(rows > 0);
        let packed = match self {
            Self::Uniform => 0,
            Self::Ragged => rows,
            Self::PerRow { packed_rows } => {
                assert!(rows <= 64);
                assert!(rows == 64 || packed_rows >> rows == 0);
                packed_rows.count_ones() as usize
            }
        };
        [(Self::Uniform, rows - packed), (Self::Ragged, packed)]
    }

    pub(crate) fn compute_dispatch_count(self, rows: usize) -> u64 {
        self.primitive_rows(rows)
            .iter()
            .filter(|(_, rows)| *rows != 0)
            .count() as u64
    }
}

pub(crate) struct ControlledCpuFill {
    pub(crate) id: u64,
    pub(crate) rows: usize,
    pub(crate) vocabulary: usize,
    pub(crate) full_logits: bool,
    pub(crate) layout: ControlledCpuFillLayout,
    pub(crate) executed: AtomicUsize,
    pub(crate) executed_dispatches: AtomicUsize,
    checkpoint_writes: Mutex<Vec<Vec<memory_fixture::CheckpointStateWrite>>>,
    output: Mutex<Option<Vec<Vec<f32>>>>,
}
impl ControlledCpuFill {
    pub(crate) fn new(rows: usize, vocabulary: usize, full_logits: bool) -> Arc<Self> {
        Self::with_layout(
            rows,
            vocabulary,
            full_logits,
            ControlledCpuFillLayout::Uniform,
        )
    }

    pub(crate) fn with_layout(
        rows: usize,
        vocabulary: usize,
        full_logits: bool,
        layout: ControlledCpuFillLayout,
    ) -> Arc<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        assert!(rows > 0 && vocabulary > 6);
        Arc::new(Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            rows,
            vocabulary,
            full_logits,
            layout,
            executed: AtomicUsize::new(0),
            executed_dispatches: AtomicUsize::new(0),
            checkpoint_writes: Mutex::new(Vec::new()),
            output: Mutex::new(None),
        })
    }
    pub(crate) fn execute(&self) {
        let primitives = self.layout.primitive_rows(self.rows).map(|(layout, rows)| {
            if rows == 0 {
                Vec::new()
            } else {
                self.execute_primitive(layout, rows)
            }
        });
        let [scalar, packed] = primitives;
        let mut scalar = scalar.into_iter();
        let mut packed = packed.into_iter();
        let mut rows = (0..self.rows)
            .map(|row| {
                let use_packed = match self.layout {
                    ControlledCpuFillLayout::Uniform => false,
                    ControlledCpuFillLayout::Ragged => true,
                    ControlledCpuFillLayout::PerRow { packed_rows } => {
                        packed_rows & (1 << row) != 0
                    }
                };
                if use_packed {
                    packed.next().unwrap()
                } else {
                    scalar.next().unwrap()
                }
            })
            .collect::<Vec<_>>();
        assert!(scalar.next().is_none() && packed.next().is_none());
        if !self.full_logits {
            for logits in &mut rows {
                let selected = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                *logits = vec![selected as f32];
            }
        }
        let writes = self.checkpoint_writes.lock().unwrap();
        if let Some(writes) = writes.get(self.executed.load(Ordering::Acquire)) {
            for write in writes {
                write.execute();
            }
        }
        drop(writes);
        *self.output.lock().unwrap() = Some(rows);
        self.executed.fetch_add(1, Ordering::Release);
    }

    pub(crate) fn bind_checkpoint_states(
        &self,
        invocation: &BatchedOperationInvocation<'_, TestBuffer>,
    ) {
        self.checkpoint_writes
            .lock()
            .unwrap()
            .push(memory_fixture::checkpoint_state_writes(invocation));
    }

    fn execute_primitive(&self, layout: ControlledCpuFillLayout, rows: usize) -> Vec<Vec<f32>> {
        assert!(rows > 0);
        self.executed_dispatches.fetch_add(1, Ordering::Release);
        match layout {
            ControlledCpuFillLayout::Uniform => (0..rows)
                .map(|_| {
                    let mut logits = vec![0.0_f32; self.vocabulary];
                    logits[6] = 1.0;
                    logits
                })
                .collect::<Vec<_>>(),
            ControlledCpuFillLayout::Ragged => {
                // A different executable path: stage the whole packed output,
                // scatter selected values, then materialize its per-row views.
                let mut packed = vec![0.0_f32; rows * self.vocabulary];
                for row in 0..rows {
                    packed[row * self.vocabulary + 6] = 1.0;
                }
                packed
                    .chunks_exact(self.vocabulary)
                    .map(<[f32]>::to_vec)
                    .collect()
            }
            ControlledCpuFillLayout::PerRow { .. } => unreachable!("primitive is not a selector"),
        }
    }
    pub(crate) fn take_output(&self) -> Vec<Vec<f32>> {
        self.output
            .lock()
            .unwrap()
            .take()
            .expect("actual CPU submit filled output")
    }
}
