//! Checked arithmetic over diagnostic timestamps. These types are not live
//! receipts and never authorize training, prediction, execution or output.
use super::MAX_COST_ROWS;
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct HostWallTimesV1 {
    pub schema_version: u32,
    pub complete_single_wave: bool,
    pub prepare_started_at_ns: Option<u64>,
    pub executor_returned_at_ns: Option<u64>,
    pub finalized_at_ns: Option<u64>,
    pub full_wall_ns: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HostRowTimesV1 {
    pub complete_single_wave: bool,
    pub host_processing_ordinal: Option<u32>,
    pub host_started_at_ns: Option<u64>,
    pub token_committed_at_ns: Option<u64>,
    pub output_published_at_ns: Option<u64>,
    pub completion_started_at_ns: Option<u64>,
    pub settled_at_ns: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostWallPartitionUnknownV1 {
    UnsupportedSchema,
    Incomplete,
    RowLimit,
    MissingTime,
    InvalidClock,
    InvalidOrdinal,
    WallMismatch,
    AllocationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HostWallRowPartitionV1 {
    /// Index in the original physical rows; rows below are in host order.
    pub physical_row_index: u32,
    pub host_processing_ordinal: u32,
    pub host_started_at_ns: u64,
    pub settled_at_ns: u64,
    pub host_ns: u64,
    pub gap_before_ns: u64,
    pub before_token_commit_ns: u64,
    pub after_token_commit_ns: u64,
    /// Nested in host_ns, never an additional wall component.
    pub terminal_completion_ns: Option<u64>,
    /// Actor handoff to settlement, not network/SSE delivery. Nested in host_ns.
    pub after_actor_handoff_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HostWallPartitionV1 {
    pub schema_version: u32,
    pub full_wall_ns: u64,
    /// Preparation through executor return: encode/device/sync/readback wall,
    /// not GPU kernel time and not a sum of overlapping device/API intervals.
    pub execution_ns: u64,
    pub serial_host_ns: u64,
    /// First-row and inter-row gaps, excluding the finalization tail.
    pub shared_gap_ns: u64,
    pub finalized_after_settled_ns: u64,
    pub rows: Vec<HostWallRowPartitionV1>,
}

/// W = E + sum(H_i) + G over one complete original observation. No new clock,
/// synthesized completion, statistical support or private receipt is created.
pub fn diagnose_host_wall_partition_v1(
    wave: HostWallTimesV1,
    rows: impl ExactSizeIterator<Item = HostRowTimesV1>,
) -> Result<HostWallPartitionV1, HostWallPartitionUnknownV1> {
    use HostWallPartitionUnknownV1 as Unknown;
    if wave.schema_version != 1 {
        return Err(Unknown::UnsupportedSchema);
    }
    if !wave.complete_single_wave {
        return Err(Unknown::Incomplete);
    }
    if rows.len() == 0 || rows.len() > MAX_COST_ROWS {
        return Err(Unknown::RowLimit);
    }
    let prepare = wave.prepare_started_at_ns.ok_or(Unknown::MissingTime)?;
    let returned = wave.executor_returned_at_ns.ok_or(Unknown::MissingTime)?;
    let finalized = wave.finalized_at_ns.ok_or(Unknown::MissingTime)?;
    let full_wall = wave.full_wall_ns.ok_or(Unknown::MissingTime)?;
    let execution = returned.checked_sub(prepare).ok_or(Unknown::InvalidClock)?;
    let mut partition = Vec::new();
    partition
        .try_reserve_exact(rows.len())
        .map_err(|_| Unknown::AllocationFailed)?;
    for (index, row) in rows.enumerate() {
        if !row.complete_single_wave {
            return Err(Unknown::Incomplete);
        }
        let ordinal = row.host_processing_ordinal.ok_or(Unknown::InvalidOrdinal)?;
        let started = row.host_started_at_ns.ok_or(Unknown::MissingTime)?;
        let committed = row.token_committed_at_ns.ok_or(Unknown::MissingTime)?;
        let settled = row.settled_at_ns.ok_or(Unknown::MissingTime)?;
        if !(returned <= started
            && started <= committed
            && committed <= settled
            && settled <= finalized)
            || [row.output_published_at_ns, row.completion_started_at_ns]
                .into_iter()
                .flatten()
                .any(|at| at < committed || at > settled)
        {
            return Err(Unknown::InvalidClock);
        }
        partition.push(HostWallRowPartitionV1 {
            physical_row_index: index as u32,
            host_processing_ordinal: ordinal,
            host_started_at_ns: started,
            settled_at_ns: settled,
            host_ns: settled - started,
            gap_before_ns: 0,
            before_token_commit_ns: committed - started,
            after_token_commit_ns: settled - committed,
            terminal_completion_ns: row.completion_started_at_ns.map(|at| settled - at),
            after_actor_handoff_ns: row.output_published_at_ns.map(|at| settled - at),
        });
    }
    partition.sort_unstable_by_key(|row| row.host_processing_ordinal);
    let mut prior = returned;
    let mut host = 0_u64;
    let mut gaps = 0_u64;
    for (ordinal, row) in partition.iter_mut().enumerate() {
        if row.host_processing_ordinal as usize != ordinal {
            return Err(Unknown::InvalidOrdinal);
        }
        row.gap_before_ns = row
            .host_started_at_ns
            .checked_sub(prior)
            .ok_or(Unknown::InvalidClock)?;
        host = host.checked_add(row.host_ns).ok_or(Unknown::InvalidClock)?;
        gaps = gaps
            .checked_add(row.gap_before_ns)
            .ok_or(Unknown::InvalidClock)?;
        prior = row.settled_at_ns;
    }
    if full_wall == 0
        || prior.checked_sub(prepare) != Some(full_wall)
        || execution
            .checked_add(host)
            .and_then(|sum| sum.checked_add(gaps))
            != Some(full_wall)
    {
        return Err(Unknown::WallMismatch);
    }
    Ok(HostWallPartitionV1 {
        schema_version: 1,
        full_wall_ns: full_wall,
        execution_ns: execution,
        serial_host_ns: host,
        shared_gap_ns: gaps,
        finalized_after_settled_ns: finalized - prior,
        rows: partition,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wave() -> HostWallTimesV1 {
        HostWallTimesV1 {
            schema_version: 1,
            complete_single_wave: true,
            prepare_started_at_ns: Some(10),
            executor_returned_at_ns: Some(20),
            finalized_at_ns: Some(100),
            full_wall_ns: Some(40),
        }
    }
    fn rows() -> [HostRowTimesV1; 2] {
        [
            HostRowTimesV1 {
                complete_single_wave: true,
                host_processing_ordinal: Some(1),
                host_started_at_ns: Some(35),
                token_committed_at_ns: Some(36),
                output_published_at_ns: Some(45),
                completion_started_at_ns: Some(37),
                settled_at_ns: Some(50),
            },
            HostRowTimesV1 {
                complete_single_wave: true,
                host_processing_ordinal: Some(0),
                host_started_at_ns: Some(22),
                token_committed_at_ns: Some(25),
                settled_at_ns: Some(30),
                ..Default::default()
            },
        ]
    }
    #[test]
    fn host_wall_partition_preserves_physical_mapping_and_excludes_finalization_tail() {
        let p = diagnose_host_wall_partition_v1(wave(), rows().into_iter()).unwrap();
        assert_eq!(
            (
                p.full_wall_ns,
                p.execution_ns,
                p.serial_host_ns,
                p.shared_gap_ns
            ),
            (40, 10, 23, 7)
        );
        assert_eq!(p.finalized_after_settled_ns, 50);
        assert_eq!(
            p.rows
                .iter()
                .map(|r| r.physical_row_index)
                .collect::<Vec<_>>(),
            [1, 0]
        );
        assert_eq!(p.rows[1].terminal_completion_ns, Some(13));
        assert_eq!(p.rows[1].after_actor_handoff_ns, Some(5));
        assert_eq!(p.rows[0].after_actor_handoff_ns, None); // nonfinal prefill
    }
    #[test]
    fn host_wall_partition_rejects_overlap_sparse_order_missing_times_and_false_wall() {
        let mut r = rows();
        r[0].host_started_at_ns = Some(29);
        assert_eq!(
            diagnose_host_wall_partition_v1(wave(), r.into_iter()),
            Err(HostWallPartitionUnknownV1::InvalidClock)
        );
        let mut r = rows();
        r[0].host_processing_ordinal = Some(0);
        assert_eq!(
            diagnose_host_wall_partition_v1(wave(), r.into_iter()),
            Err(HostWallPartitionUnknownV1::InvalidOrdinal)
        );
        let mut r = rows();
        r[0].host_processing_ordinal = Some(2);
        assert_eq!(
            diagnose_host_wall_partition_v1(wave(), r.into_iter()),
            Err(HostWallPartitionUnknownV1::InvalidOrdinal)
        );
        let mut r = rows();
        r[0].settled_at_ns = None;
        assert_eq!(
            diagnose_host_wall_partition_v1(wave(), r.into_iter()),
            Err(HostWallPartitionUnknownV1::MissingTime)
        );
        let mut w = wave();
        w.full_wall_ns = Some(90);
        assert_eq!(
            diagnose_host_wall_partition_v1(w, rows().into_iter()),
            Err(HostWallPartitionUnknownV1::WallMismatch)
        );
        let mut w = wave();
        w.complete_single_wave = false;
        assert_eq!(
            diagnose_host_wall_partition_v1(w, rows().into_iter()),
            Err(HostWallPartitionUnknownV1::Incomplete)
        );
    }
    #[test]
    fn host_wall_partition_checks_capacity_and_u64_boundary_without_time_rounding() {
        let mut w = wave();
        w.prepare_started_at_ns = Some(u64::MAX - 3);
        w.executor_returned_at_ns = Some(u64::MAX - 2);
        w.finalized_at_ns = Some(u64::MAX);
        w.full_wall_ns = Some(3);
        let row = HostRowTimesV1 {
            complete_single_wave: true,
            host_processing_ordinal: Some(0),
            host_started_at_ns: Some(u64::MAX - 1),
            token_committed_at_ns: Some(u64::MAX),
            settled_at_ns: Some(u64::MAX),
            ..Default::default()
        };
        let p = diagnose_host_wall_partition_v1(w, std::iter::once(row)).unwrap();
        assert_eq!(
            (p.execution_ns, p.serial_host_ns, p.shared_gap_ns),
            (1, 1, 1)
        );
        assert_eq!(
            diagnose_host_wall_partition_v1(wave(), (0..MAX_COST_ROWS + 1).map(|_| row)),
            Err(HostWallPartitionUnknownV1::RowLimit)
        );
    }
}
