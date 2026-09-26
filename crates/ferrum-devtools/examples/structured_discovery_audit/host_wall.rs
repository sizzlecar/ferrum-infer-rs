//! Historical JSON is parsed only into numeric diagnostic inputs. In
//! particular this never constructs HostStageEvidence or a qualified receipt.
use super::*;
use ferrum_interfaces::execution_cost::{
    diagnose_host_wall_partition_v1, HostRowTimesV1, HostWallPartitionUnknownV1,
    HostWallPartitionV1, HostWallTimesV1, MAX_COST_ROWS,
};
use serde::Deserialize;

const MAX_GROUPS: usize = 4096;
const MAX_EXAMPLES: usize = 16;

#[derive(Deserialize)]
struct RawStages {
    schema_version: u32,
    completeness: String,
    prepare_started_at_ns: Option<u64>,
    executor_returned_at_ns: Option<u64>,
    finalized_at_ns: Option<u64>,
    full_wall_ns: Option<u64>,
    rows: Vec<RawRow>,
}
#[derive(Deserialize)]
struct RawRow {
    completeness: String,
    host_processing_ordinal: Option<u32>,
    host_started_at_ns: Option<u64>,
    token_committed_at_ns: Option<u64>,
    output_published_at_ns: Option<u64>,
    completion_started_at_ns: Option<u64>,
    settled_at_ns: Option<u64>,
}

fn partition(stages: &Value) -> Result<HostWallPartitionV1, String> {
    if stages.is_null() {
        return Err("missing_host_stages".into());
    }
    let rows = stages["rows"].as_array().ok_or("missing_rows")?;
    if rows.len() > MAX_COST_ROWS {
        return Err(key(&json!(HostWallPartitionUnknownV1::RowLimit)));
    }
    let raw = RawStages::deserialize(stages).map_err(|_| "invalid_timestamp_wire")?;
    diagnose_host_wall_partition_v1(
        HostWallTimesV1 {
            schema_version: raw.schema_version,
            complete_single_wave: raw.completeness == "complete_single_wave",
            prepare_started_at_ns: raw.prepare_started_at_ns,
            executor_returned_at_ns: raw.executor_returned_at_ns,
            finalized_at_ns: raw.finalized_at_ns,
            full_wall_ns: raw.full_wall_ns,
        },
        raw.rows.into_iter().map(|row| HostRowTimesV1 {
            complete_single_wave: row.completeness == "complete_single_wave",
            host_processing_ordinal: row.host_processing_ordinal,
            host_started_at_ns: row.host_started_at_ns,
            token_committed_at_ns: row.token_committed_at_ns,
            output_published_at_ns: row.output_published_at_ns,
            completion_started_at_ns: row.completion_started_at_ns,
            settled_at_ns: row.settled_at_ns,
        }),
    )
    .map_err(|reason| label(&json!(reason)))
}

#[derive(Default, Serialize)]
struct Totals {
    waves: u64,
    rows: u64,
    full_wall_ns: u64,
    execution_ns: u64,
    serial_host_ns: u64,
    shared_gap_ns: u64,
    finalized_tail_ns: u64,
}
impl Totals {
    fn add(&mut self, p: &HostWallPartitionV1) -> Result<()> {
        for (total, value) in [
            (&mut self.waves, 1),
            (&mut self.rows, p.rows.len() as u64),
            (&mut self.full_wall_ns, p.full_wall_ns),
            (&mut self.execution_ns, p.execution_ns),
            (&mut self.serial_host_ns, p.serial_host_ns),
            (&mut self.shared_gap_ns, p.shared_gap_ns),
            (&mut self.finalized_tail_ns, p.finalized_after_settled_ns),
        ] {
            *total = total
                .checked_add(value)
                .context("wall diagnostic aggregate overflow")?;
        }
        ensure!(
            self.execution_ns
                .checked_add(self.serial_host_ns)
                .and_then(|sum| sum.checked_add(self.shared_gap_ns))
                == Some(self.full_wall_ns),
            "wall totals do not close"
        );
        Ok(())
    }
}

#[derive(Default, Serialize)]
struct RowTotals {
    rows: u64,
    host_ns: u64,
    before_token_commit_ns: u64,
    after_token_commit_ns: u64,
    /// Nested subset of host_ns; never added to host_ns.
    terminal_completion_ns: u64,
    terminal_completion_rows: u64,
    max_host_ns: u64,
}

#[derive(Default)]
pub(super) struct Audit {
    attempted: u64,
    success: BTreeMap<String, Totals>,
    row_classes: BTreeMap<String, RowTotals>,
    failures: BTreeMap<String, u64>,
    failure_examples: BTreeMap<String, Value>,
    largest_host: Vec<(u64, Value)>,
    largest_gap: Vec<(u64, Value)>,
}
fn bounded_key<T>(map: &BTreeMap<String, T>, key: &str) -> Result<()> {
    ensure!(
        map.contains_key(key) || map.len() < MAX_GROUPS,
        "wall diagnostic groups exceed bound; no partial success"
    );
    Ok(())
}
fn largest(examples: &mut Vec<(u64, Value)>, value: u64, context: Value) {
    examples.push((value, context));
    examples.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    examples.truncate(MAX_EXAMPLES);
}
impl Audit {
    pub(super) fn observe(&mut self, v: &Value, line: usize) -> Result<()> {
        self.attempted += 1;
        ensure!(
            self.attempted <= MAX_WAVES as u64,
            "wall diagnostic wave limit"
        );
        let partition = match partition(&v["host_stages"]) {
            Ok(p) => p,
            Err(reason) => {
                let k = key(&json!({"phase":v["phase"],"submission":v["submission"],
                    "partition_error":reason,"stage_completeness":v["host_stages"]["completeness"],
                    "actual_reason":v["evidence"]["reason"]}));
                bounded_key(&self.failures, &k)?;
                increment(&mut self.failures, k.clone());
                if self.failure_examples.len() < MAX_EXAMPLES {
                    self.failure_examples.entry(k).or_insert_with(
                        || json!({"line":line,"call_id":v["host_stages"]["call_id"]}),
                    );
                }
                return Ok(());
            }
        };
        let physical = array(&recipe(v)["physical_host_rows"]);
        let physical = if physical.len() == partition.rows.len()
            && physical
                .iter()
                .enumerate()
                .all(|(i, row)| row["physical_position"].as_u64() == Some(i as u64))
        {
            Some(physical)
        } else {
            None
        };
        let pending = physical.and_then(|rows| {
            rows.iter().try_fold(0_usize, |n, row| {
                row["pending_decoded_utf8"]
                    .as_bool()
                    .map(|p| n + usize::from(p))
            })
        });
        let length = physical.and_then(|rows| {
            rows.iter().try_fold(0_usize, |n, row| {
                match row["terminal_expectation"].as_str() {
                    Some("length_boundary") => n.checked_add(1),
                    Some("token_may_terminate" | "no_token_produced") => Some(n),
                    _ => None,
                }
            })
        });
        let k = key(&json!({"phase":v["phase"],"kind":shape(v)["kind"],
            "product":recipe(v)["device"]["product"],"rows":partition.rows.len(),
            "pending_rows":pending,"length_rows":length}));
        bounded_key(&self.success, &k)?;
        self.success.entry(k.clone()).or_default().add(&partition)?;
        let raw_rows = array(&v["host_stages"]["rows"]);
        for row in &partition.rows {
            let i = row.physical_row_index as usize;
            let facts = physical.map(|rows| &rows[i]);
            let raw = &raw_rows[i];
            let class = key(
                &json!({"phase":v["phase"],"work":raw["actual_work"]["kind"],
                "product":recipe(v)["device"]["product"],
                "pending_decoded_utf8":facts.map(|r|&r["pending_decoded_utf8"]),
                "terminal_expectation":facts.map(|r|&r["terminal_expectation"]),
                "terminal_finish_reason":raw["terminal"]["finish_reason"]}),
            );
            bounded_key(&self.row_classes, &class)?;
            let t = self.row_classes.entry(class).or_default();
            for (total, value) in [
                (&mut t.rows, 1),
                (&mut t.host_ns, row.host_ns),
                (&mut t.before_token_commit_ns, row.before_token_commit_ns),
                (&mut t.after_token_commit_ns, row.after_token_commit_ns),
                (
                    &mut t.terminal_completion_ns,
                    row.terminal_completion_ns.unwrap_or(0),
                ),
                (
                    &mut t.terminal_completion_rows,
                    u64::from(row.terminal_completion_ns.is_some()),
                ),
            ] {
                *total = total
                    .checked_add(value)
                    .context("row diagnostic aggregate overflow")?;
            }
            t.max_host_ns = t.max_host_ns.max(row.host_ns);
        }
        let example = json!({"line":line,"call_id":v["host_stages"]["call_id"],"class":k,"partition":partition});
        largest(
            &mut self.largest_host,
            partition.serial_host_ns,
            example.clone(),
        );
        largest(&mut self.largest_gap, partition.shared_gap_ns, example);
        Ok(())
    }
    pub(super) fn report(&self) -> Value {
        json!({"scope":"diagnostic arithmetic only; no live receipt/qualification/prediction. E includes encode/device/wait/readback, not GPU-only; H includes actor handoff and real terminal settlement, not socket/client delivery; finalized tail excluded; nested commit/terminal fields must not be added again",
            "attempted_waves":self.attempted,"partitioned_waves":self.success.values().map(|g|g.waves).sum::<u64>(),
            "unpartitioned_waves":self.failures.values().sum::<u64>(),"wave_classes":self.success,
            "row_classes":self.row_classes,"failures":self.failures,"failure_examples":self.failure_examples,
            "largest_serial_host":self.largest_host.iter().map(|e|&e.1).collect::<Vec<_>>(),
            "largest_shared_gap":self.largest_gap.iter().map(|e|&e.1).collect::<Vec<_>>()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_wall_raw_diagnostic_missing_private_times_stays_missing_not_zero() {
        let mut audit = Audit::default();
        audit
            .observe(&json!({"phase":"discovery","submission":"NotSubmitted"}), 1)
            .unwrap();
        let report = audit.report();
        assert_eq!(report["partitioned_waves"], 0);
        assert_eq!(report["unpartitioned_waves"], 1);
        let raw = json!({"schema_version":1,"completeness":"complete_single_wave",
            "prepare_started_at_ns":1,"executor_returned_at_ns":2,"finalized_at_ns":6,"full_wall_ns":4,
            "rows":[{"completeness":"complete_single_wave","host_processing_ordinal":0,
                "host_started_at_ns":3,"token_committed_at_ns":4,"settled_at_ns":5}]});
        let p = partition(&raw).unwrap();
        assert_eq!(
            (p.execution_ns, p.serial_host_ns, p.shared_gap_ns),
            (1, 2, 1)
        );
        let mut missing = raw;
        missing["rows"][0]["settled_at_ns"] = Value::Null;
        assert_eq!(partition(&missing).unwrap_err(), "missing_time");
    }

    #[test]
    fn host_wall_raw_diagnostic_maps_host_order_back_to_physical_pending_rows() {
        let mut v = json!({"phase":"discovery","event":"wave","host_stages":{
            "schema_version":1,"completeness":"complete_single_wave","call_id":1,
            "prepare_started_at_ns":0,"executor_returned_at_ns":10,"finalized_at_ns":40,"full_wall_ns":30,
            "rows":[
                {"completeness":"complete_single_wave","host_processing_ordinal":1,"actual_work":{"kind":"decode"},
                 "host_started_at_ns":20,"token_committed_at_ns":21,"settled_at_ns":30},
                {"completeness":"complete_single_wave","host_processing_ordinal":0,"actual_work":{"kind":"decode"},
                 "host_started_at_ns":10,"token_committed_at_ns":11,"settled_at_ns":12}],
            "structured_evidence":{"Ok":{"recipe":{"device":{"product":"full_logits"},"physical_host_rows":[
                {"physical_position":0,"pending_decoded_utf8":true,"terminal_expectation":"token_may_terminate"},
                {"physical_position":1,"pending_decoded_utf8":false,"terminal_expectation":"token_may_terminate"}]}}}}});
        let mut audit = Audit::default();
        audit.observe(&v, 1).unwrap();
        let pending_host = audit
            .row_classes
            .iter()
            .filter(|(k, _)| {
                serde_json::from_str::<Value>(k).unwrap()["pending_decoded_utf8"] == true
            })
            .map(|(_, t)| t.host_ns)
            .sum::<u64>();
        assert_eq!(pending_host, 10);
        assert_eq!(audit.report()["partitioned_waves"], 1);
        // A reordered/incomplete raw physical table cannot relabel the rows.
        v["host_stages"]["structured_evidence"]["Ok"]["recipe"]["physical_host_rows"][0]
            ["physical_position"] = json!(1);
        let mut unknown = Audit::default();
        unknown.observe(&v, 2).unwrap();
        assert!(unknown
            .row_classes
            .keys()
            .all(|k| serde_json::from_str::<Value>(k).unwrap()["pending_decoded_utf8"].is_null()));
        assert_eq!(unknown.report()["partitioned_waves"], 1); // arithmetic still closes; host class is unknown.
    }
}
