//! Optional raw-pilot loading uses the same bounded reader/cache as the main
//! experiment. All references remain relative to the outer manifest directory.

use super::*;

pub(super) struct LoadedComparison {
    pub(super) contract: FrozenComparisonContract,
    cells: Vec<(u32, Vec<(String, LoadedArm, LoadedArm)>)>,
}

impl LoadedComparison {
    pub(super) fn inputs(&self) -> Vec<ComparisonCellInput<'_>> {
        self.cells
            .iter()
            .map(|(concurrency, pairs)| ComparisonCellInput {
                concurrency: *concurrency,
                pairs: pairs
                    .iter()
                    .map(|(pair_id, baseline, candidate)| PairedRepeatInput {
                        pair_id: pair_id.clone(),
                        baseline: baseline.input(),
                        candidate: candidate.input(),
                    })
                    .collect(),
                statistical_evidence: None,
            })
            .collect()
    }
}

pub(super) fn load_comparison(
    reader: &mut Reader<'_>,
    sidecars: &mut Sidecars,
    manifest: &ComparisonArtifactManifest,
) -> Result<LoadedComparison, ArtifactError> {
    let limits = reader.limits;
    if manifest.schema_version != 1
        || manifest.cells.len() > limits.max_cells
        || manifest
            .cells
            .iter()
            .any(|cell| cell.pairs.len() > limits.max_pairs_per_cell)
    {
        return Err(err(
            "comparison manifest version or resource bounds are invalid",
        ));
    }
    let contract: FrozenComparisonContract = parse(&reader.read(&manifest.contract)?)?;
    contract
        .validate()
        .map_err(|error| err(error.to_string()))?;
    if contract.cells.len() > limits.max_cells
        || contract.pairs.len() > limits.max_pairs_per_cell
        || contract
            .pairs
            .iter()
            .any(|pair| pair.selection.samples.len() > limits.max_requests_per_repeat)
    {
        return Err(err("frozen contract exceeds artifact resource bounds"));
    }
    let mut cells = Vec::new();
    let mut cell_ids = BTreeSet::new();
    for cell in &manifest.cells {
        if !cell_ids.insert(cell.concurrency)
            || !contract
                .cells
                .iter()
                .any(|declared| declared.concurrency == cell.concurrency)
        {
            return Err(err("duplicate or unfrozen manifest concurrency cell"));
        }
        let mut pairs = Vec::new();
        let mut pair_ids = BTreeSet::new();
        for pair in &cell.pairs {
            if !pair_ids.insert(&pair.pair_id) {
                return Err(err("duplicate manifest pair"));
            }
            let frozen = contract
                .pairs
                .iter()
                .find(|declared| declared.pair_id == pair.pair_id)
                .ok_or_else(|| err("unfrozen manifest pair"))?;
            let baseline = load_arm(
                reader,
                sidecars,
                &pair.baseline,
                &contract,
                frozen,
                cell.concurrency,
                false,
            )?;
            let candidate = load_arm(
                reader,
                sidecars,
                &pair.candidate,
                &contract,
                frozen,
                cell.concurrency,
                true,
            )?;
            pairs.push((pair.pair_id.clone(), baseline, candidate));
        }
        cells.push((cell.concurrency, pairs));
    }
    Ok(LoadedComparison { contract, cells })
}

pub(super) fn compare_using_pilot(
    reader: &mut Reader<'_>,
    sidecars: &mut Sidecars,
    refs: &EligibilityArtifactRefs,
    contract: &FrozenComparisonContract,
    inputs: &[ComparisonCellInput<'_>],
) -> Result<SloComparisonReport, ArtifactError> {
    let plan: FrozenEligibilityPlan = parse(&reader.read(&refs.plan)?)?;
    if refs.pilot_manifest.bytes > reader.limits.max_manifest_bytes {
        return Err(err("pilot manifest exceeds manifest byte bound"));
    }
    let pilot_bytes = reader.read(&refs.pilot_manifest)?;
    let pilot_source = digest(&pilot_bytes);
    let pilot_manifest: ComparisonArtifactManifest = parse(&pilot_bytes)?;
    if pilot_manifest.eligibility.is_some() {
        return Err(err(
            "pilot manifests cannot recursively supply inference eligibility",
        ));
    }
    let pilot = load_comparison(reader, sidecars, &pilot_manifest)?;
    match verify_inference_eligibility(
        contract,
        &plan,
        &pilot.contract,
        &pilot.inputs(),
        &pilot_source,
    ) {
        Ok(token) => compare_with_eligibility(contract, inputs, &token)
            .map_err(|error| err(error.to_string())),
        Err(failure) => {
            let mut report =
                compare_slo_reports(contract, inputs).map_err(|error| err(error.to_string()))?;
            report.issues.push(format!(
                "independent pilot inference eligibility is insufficient: {failure}"
            ));
            report.inference_eligibility_failure = Some(failure);
            Ok(report)
        }
    }
}

#[cfg(test)]
mod tests;
