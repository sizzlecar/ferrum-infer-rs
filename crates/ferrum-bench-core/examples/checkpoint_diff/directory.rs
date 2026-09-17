//! Compare complete capture inventories without silently dropping failed waves.
use super::*;

type Inventory = BTreeMap<(bool, u64), (String, PathBuf)>;

fn inventory(directory: &Path) -> Result<Inventory, String> {
    let mut waves = BTreeMap::new();
    for entry in fs::read_dir(directory).map_err(|e| error("checkpoint directory", e))? {
        let entry = entry.map_err(|e| error("checkpoint directory entry", e))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let (decode, suffix) = if let Some(suffix) = name.strip_prefix("decode-wave-") {
            (true, suffix)
        } else if let Some(suffix) = name.strip_prefix("wave-") {
            (false, suffix)
        } else {
            continue;
        };
        let Some(number) = suffix.strip_suffix(".json") else {
            continue;
        };
        let index = number
            .parse::<u64>()
            .map_err(|e| error("wave file index", e))?;
        let prefix = if decode { "decode-wave" } else { "wave" };
        if name != format!("{prefix}-{index:04}.json") {
            return Err(format!("noncanonical wave manifest name {name}"));
        }
        if !entry
            .file_type()
            .map_err(|e| error("wave file type", e))?
            .is_file()
        {
            return Err(format!("wave manifest {name} is not a regular file"));
        }
        waves.insert((decode, index), (name.to_owned(), entry.path()));
    }
    if waves.is_empty() {
        return Err("checkpoint directory contains no wave manifests".into());
    }
    for decode in [false, true] {
        for (expected, &(_, observed)) in waves
            .keys()
            .filter(|&&(phase, _)| phase == decode)
            .enumerate()
        {
            if observed != expected as u64 {
                return Err("checkpoint wave inventory has a missing capture index".into());
            }
        }
    }
    Ok(waves)
}

#[derive(Deserialize, Serialize, PartialEq, Eq)]
struct TeacherPlan {
    mode: String,
    encoding: String,
    token_count: u64,
    token_ids_sha256: String,
}

#[derive(Deserialize)]
struct Plan {
    schema_version: u32,
    #[serde(default)]
    teacher_forcing: Option<TeacherPlan>,
}

fn teacher_plan(directory: &Path) -> Result<Option<TeacherPlan>, String> {
    let bytes = match fs::read(directory.join("plan.json")) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(error("checkpoint plan", err)),
    };
    let plan: Plan = serde_json::from_slice(&bytes).map_err(|e| error("checkpoint plan", e))?;
    if !matches!(plan.schema_version, 3 | 4)
        || (plan.schema_version == 4) != plan.teacher_forcing.is_some()
    {
        return Err("checkpoint plan schema and teacher forcing disagree".into());
    }
    if let Some(teacher) = &plan.teacher_forcing {
        if teacher.mode != "canonical-history"
            || teacher.encoding != "u32-le"
            || teacher.token_count == 0
            || teacher.token_ids_sha256.len() != 64
            || !teacher
                .token_ids_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("checkpoint teacher plan has invalid history evidence".into());
        }
    }
    Ok(plan.teacher_forcing)
}

pub(super) fn compare_directories(reference: &Path, candidate: &Path) -> Result<Value, String> {
    let left = inventory(reference)?;
    let right = inventory(candidate)?;
    if !left.keys().eq(right.keys()) {
        return Err("checkpoint directory wave inventories differ".into());
    }
    let teacher = teacher_plan(reference)?;
    if teacher != teacher_plan(candidate)? {
        return Err("checkpoint directory teacher plans differ".into());
    }
    if let Some(teacher) = &teacher {
        if left.len() as u64 != teacher.token_count
            || left.keys().filter(|&&(decode, _)| !decode).count() != 1
        {
            return Err("teacher-forced wave inventory is incomplete or exceeds the plan".into());
        }
    }
    let (mut waves, mut kl, mut reference_nll, mut candidate_nll, mut delta_nll) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (key, (name, left_path)) in &left {
        let report = compare(left_path, &right[key].1).map_err(|e| error(name, e))?;
        let kind = if key.0 { "decode" } else { "prefill" };
        if report["wave_kind"] != kind
            || ["capture_index", "candidate_capture_index"]
                .iter()
                .any(|field| report[*field].as_u64().is_some_and(|index| index != key.1))
        {
            return Err(format!(
                "checkpoint {name} kind or capture index differs from its file name"
            ));
        }
        let decision = &report["teacher_forced_decision"];
        if teacher.is_none()
            && (report["reference_schema_version"] == 4 || report["candidate_schema_version"] == 4)
        {
            return Err(format!(
                "checkpoint {name} teacher capture requires its complete plan"
            ));
        }
        match (&teacher, decision.is_null()) {
            (Some(_), false)
                if decision["token_index"].as_u64() == Some(if key.0 { key.1 + 1 } else { 0 }) => {}
            (None, true) => {}
            _ => {
                return Err(format!(
                    "checkpoint {name} teacher decision does not match its complete plan"
                ))
            }
        }
        for row in report["comparisons"]
            .as_array()
            .expect("compare creates a comparison array")
        {
            let distribution = &row["distribution"];
            if distribution.is_null() {
                continue;
            }
            kl.push(
                distribution["kl_reference_to_candidate_nats"]
                    .as_f64()
                    .expect("finite KL"),
            );
            if let Some(target) = distribution["teacher_forced"].as_object() {
                reference_nll.push(target["reference_nll_nats"].as_f64().expect("finite NLL"));
                candidate_nll.push(target["candidate_nll_nats"].as_f64().expect("finite NLL"));
                delta_nll.push(target["delta_nll_nats"].as_f64().expect("finite delta NLL"));
            }
        }
        waves.push(json!({"manifest":name,"report":report}));
    }
    if teacher.is_some() && (delta_nll.len() != left.len() || kl.len() != left.len()) {
        return Err(
            "teacher-forced directory does not contain one target distribution per wave".into(),
        );
    }
    let mean = |values: &[f64]| {
        (!values.is_empty()).then(|| compensated_sum(values.iter().copied()) / values.len() as f64)
    };
    Ok(
        json!({"schema_version":1,"scope":"checkpoint_directory_diagnostic","release_approved":false,
        "input_and_weight_identity_verified":false,"token_history_evidence":"matching_recorded_token_spans",
        "teacher_forcing":teacher,"wave_count":waves.len(),"waves":waves,
        "aggregate":{"distribution_count":kl.len(),"teacher_forced_target_count":delta_nll.len(),
            "mean_kl_reference_to_candidate_nats":mean(&kl),
            "mean_reference_nll_nats":mean(&reference_nll),"mean_candidate_nll_nats":mean(&candidate_nll),
            "mean_delta_nll_nats":mean(&delta_nll)}}),
    )
}
