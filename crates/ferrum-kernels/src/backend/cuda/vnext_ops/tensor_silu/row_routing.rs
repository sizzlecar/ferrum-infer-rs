//! The native argument bank limits one launch, not the logical routing table.
//! Chunk boundaries split only independent rows, never a reduction axis.
use ferrum_runtime::RowSelectedBf16Geometry;

const RUNS_PER_LAUNCH: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Routing {
    ends: [u64; RUNS_PER_LAUNCH],
    rows: [u64; RUNS_PER_LAUNCH],
    count: u64,
}
// SAFETY: Initialized pointer-free integers matching the unchanged native ABI.
unsafe impl cudarc::driver::DeviceRepr for Routing {}
const _: [(); 520] = [(); std::mem::size_of::<Routing>()];
const _: [(); 8] = [(); std::mem::align_of::<Routing>()];

pub(super) struct Chunk {
    pub byte_offset: u64,
    pub rows: u64,
    pub routing: Routing,
}

pub(super) fn chunks(g: &RowSelectedBf16Geometry) -> impl ExactSizeIterator<Item = Chunk> + '_ {
    g.run_ends()
        .chunks(RUNS_PER_LAUNCH)
        .zip(g.parameter_rows().chunks(RUNS_PER_LAUNCH))
        .enumerate()
        .map(move |(index, (ends, rows))| {
            let first = if index == 0 {
                0
            } else {
                g.run_ends()[index * RUNS_PER_LAUNCH - 1]
            };
            let mut routing = Routing {
                ends: [0; RUNS_PER_LAUNCH],
                rows: [0; RUNS_PER_LAUNCH],
                count: ends.len() as u64,
            };
            for (to, end) in routing.ends.iter_mut().zip(ends) {
                *to = end - first;
            }
            routing.rows[..rows.len()].copy_from_slice(rows);
            Chunk {
                // Geometry has already checked the complete element count and
                // at least two operations per element, so BF16 offsets fit u64.
                byte_offset: first * g.columns() * 2,
                rows: ends.last().expect("nonempty checked routing") - first,
                routing,
            }
        })
}
