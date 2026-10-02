use super::*;
use serde::Serialize;

#[derive(Serialize)]
struct Template<'a> {
    original_index: usize,
    request: &'a serde_json::value::RawValue,
    output: crate::AutomaticCostProbeOutput,
    response_model: &'a str,
}
#[derive(Serialize)]
struct Manifest<'a> {
    protocol: &'static str,
    prepared_prefix_prefill:
        crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan,
    excluded_original_template_indices: &'a [usize],
    prepared_prefix_unavailable: &'a [PreparedPrefixUnavailable],
    input_opportunities: Option<&'a ProbeInputOpportunityBudget>,
    checked_selection: Option<&'a layout::selection::CheckedSelection>,
    templates: Vec<Template<'a>>,
    cohorts: &'a [PreparedProbeCohort],
    token_policy_invalidation: TokenPolicyResidencyInvalidation,
}
struct Count {
    bytes: usize,
    limit: usize,
}
impl std::io::Write for Count {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .filter(|n| *n <= self.limit)
            .ok_or_else(|| {
                std::io::Error::other("probe manifest exceeds shared retained capacity")
            })?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn freeze(
    originals: &[AutomaticCostProbeTemplate],
    indices: &[usize],
    excluded: &[usize],
    prefix_unavailable: &[PreparedPrefixUnavailable],
    input_opportunities: Option<&ProbeInputOpportunityBudget>,
    checked_selection: Option<&layout::selection::CheckedSelection>,
    cohorts: &[PreparedProbeCohort],
    invalidation: TokenPolicyResidencyInvalidation,
    maximum_bytes: usize,
) -> Result<Box<serde_json::value::RawValue>> {
    let templates = originals
        .iter()
        .zip(indices)
        .map(|(t, index)| {
            Ok(Template {
                original_index: *index,
                request: serde_json::from_slice(t.serialized_request())
                    .map_err(|e| error(format!("probe original template: {e}")))?,
                output: t.output(),
                response_model: t.response_model(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let manifest = Manifest {
        protocol: "ferrum.automatic-prepared-probe-plan.v2",
        prepared_prefix_prefill: crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::PreparedSequentialV1,
        excluded_original_template_indices: excluded,
        prepared_prefix_unavailable: prefix_unavailable,
        input_opportunities,
        checked_selection,
        templates,
        cohorts,
        token_policy_invalidation: invalidation,
    };
    encode(&manifest, maximum_bytes)
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SourcePreparationChoice {
    Cold,
    NativePrivate,
    ColdFallback,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct SourceWork {
    pub planned_cycles: usize,
    pub maximum_anchor_span: Option<usize>,
    pub requests: usize,
    pub execution_actions: usize,
    pub declared_offer_row_bound: usize,
    pub serial_token_work: Option<usize>,
}

pub(super) fn freeze_source(
    parent: &serde_json::value::RawValue,
    parent_sha256: [u8; 32],
    source: usize,
    sources: usize,
    original_range: std::ops::Range<usize>,
    cohorts: &[PreparedProbeCohort],
    choice: SourcePreparationChoice,
    work: SourceWork,
    population: &StructuredServiceDeclarationV7,
    input_opportunities: Option<&ProbeInputOpportunityBudget>,
    maximum_bytes: usize,
) -> Result<Option<Box<serde_json::value::RawValue>>> {
    #[derive(Serialize)]
    struct Source<'a> {
        protocol: &'static str,
        parent: &'a serde_json::value::RawValue,
        parent_sha256: [u8; 32],
        source: usize,
        sources: usize,
        original_range: std::ops::Range<usize>,
        cohorts: &'a [PreparedProbeCohort],
        preparation_choice: SourcePreparationChoice,
        work: SourceWork,
        schedule: &'a ferrum_scheduler::implementations::continuous::cost_model::structured_v2::OwnerBlockScheduleV1,
        input_opportunities: Option<&'a ProbeInputOpportunityBudget>,
    }
    let source = Source {
        protocol: "ferrum.automatic-prepared-probe-series-source.v2",
        parent,
        parent_sha256,
        source,
        sources,
        original_range,
        cohorts,
        preparation_choice: choice,
        work,
        schedule: &population.schedule,
        input_opportunities,
    };
    // Count without allocating, so a pre-source capacity miss remains typed
    // rather than being confused with an invalid identity/serialization error.
    let mut count = Count {
        bytes: 0,
        limit: usize::MAX,
    };
    serde_json::to_writer(&mut count, &source)
        .map_err(|e| error(format!("source manifest size: {e}")))?;
    if count.bytes > maximum_bytes {
        return Ok(None);
    }
    encode(&source, maximum_bytes).map(Some)
}

/// Immutable input universe, fixed before any live inventory or numerical
/// work. A child may cover a prefix without claiming the pending inputs known.
pub(super) fn freeze_inventory_inputs(
    input: &PreparedProbeInputs,
    cases: &[layout::Case],
    template_order: &[usize],
    maximum_bytes: usize,
) -> Result<Box<serde_json::value::RawValue>> {
    #[derive(Serialize)]
    struct Inputs<'a> {
        protocol: &'static str,
        templates: Vec<Template<'a>>,
        cases: &'a [layout::Case],
        template_order: &'a [usize],
        // Additive audit data inside the existing hash-bound opaque payload.
        source_input_priorities: [u8; 3],
        #[serde(skip_serializing_if = "Option::is_none")]
        maximum_input_geometry_visits: Option<std::num::NonZeroU64>,
        excluded_original_template_indices: &'a [usize],
        required_geometry: &'a geometry::ProbeGeometryRequirements,
        settings: &'a SloAutomaticCostProbeSettingsV1,
        token_policy_invalidation: TokenPolicyResidencyInvalidation,
    }
    let maximum_bytes = input
        .templates
        .len()
        .checked_mul(std::mem::size_of::<Template<'_>>())
        .and_then(|headers| maximum_bytes.checked_sub(headers))
        .filter(|n| *n > 0)
        .ok_or_else(|| error("input manifest headers exceed retained capacity"))?;
    let mut templates = Vec::with_capacity(input.templates.len());
    for (template, &original_index) in input.templates.iter().zip(&input.original_template_indices)
    {
        templates.push(Template {
            original_index,
            request: serde_json::from_slice(template.serialized_request())
                .map_err(|e| error(format!("probe original template: {e}")))?,
            output: template.output(),
            response_model: template.response_model(),
        });
    }
    encode(
        &Inputs {
            protocol: "ferrum.automatic-prepared-probe-input-universe.v1",
            templates,
            cases,
            template_order,
            source_input_priorities: [0, 1, 2],
            maximum_input_geometry_visits: input.input_geometry_visit_limit,
            excluded_original_template_indices: &input.excluded_templates,
            required_geometry: &input.required_geometry,
            settings: &input.settings,
            token_policy_invalidation: input.invalidation,
        },
        maximum_bytes,
    )
}

pub(super) fn freeze_inventory_progress(
    parent: &serde_json::value::RawValue,
    parent_sha256: [u8; 32],
    completed_templates: &[usize],
    pending_templates: &[usize],
    selection_prefix_templates: &[usize],
    input_priority: u8,
    pending_input_units: usize,
    child: &serde_json::value::RawValue,
    maximum_bytes: usize,
) -> Result<Box<serde_json::value::RawValue>> {
    #[derive(Serialize)]
    struct Progress<'a> {
        protocol: &'static str,
        parent: &'a serde_json::value::RawValue,
        parent_sha256: [u8; 32],
        completed_templates: &'a [usize],
        pending_inventory_templates: &'a [usize],
        selection_prefix_templates: &'a [usize],
        input_priority: u8,
        pending_input_units: usize,
        child: &'a serde_json::value::RawValue,
    }
    encode(
        &Progress {
            protocol: "ferrum.automatic-prepared-probe-input-unit.v1",
            parent,
            parent_sha256,
            completed_templates,
            pending_inventory_templates: pending_templates,
            selection_prefix_templates,
            input_priority,
            pending_input_units,
            child,
        },
        maximum_bytes,
    )
}

/// A single selection over the complete input-only inventory. The source8
/// payload remains opaque; existing journals and their parsing are unchanged.
pub(super) fn freeze_global_inventory(
    parent: &serde_json::value::RawValue,
    parent_sha256: [u8; 32],
    completed_templates: &[usize],
    child: &serde_json::value::RawValue,
    maximum_bytes: usize,
) -> Result<Box<serde_json::value::RawValue>> {
    #[derive(Serialize)]
    struct Global<'a> {
        protocol: &'static str,
        selection_scope: &'static str,
        parent: &'a serde_json::value::RawValue,
        parent_sha256: [u8; 32],
        completed_templates: &'a [usize],
        pending_inventory_templates: &'a [usize],
        pending_input_units: usize,
        child: &'a serde_json::value::RawValue,
    }
    encode(
        &Global {
            protocol: "ferrum.automatic-prepared-probe-global-inputs.v1",
            selection_scope: "all_declared_inputs",
            parent,
            parent_sha256,
            completed_templates,
            pending_inventory_templates: &[],
            pending_input_units: 0,
            child,
        },
        maximum_bytes,
    )
}

fn encode(
    value: &impl Serialize,
    maximum_bytes: usize,
) -> Result<Box<serde_json::value::RawValue>> {
    // Bound actual raw storage before allocating it. No expanded Value tree.
    let mut count = Count {
        bytes: 0,
        limit: maximum_bytes,
    };
    serde_json::to_writer(&mut count, value)
        .map_err(|e| error(format!("probe manifest capacity: {e}")))?;
    let mut bytes = Vec::with_capacity(count.bytes);
    serde_json::to_writer(&mut bytes, value).map_err(|e| error(format!("probe manifest: {e}")))?;
    if bytes.len() != count.bytes {
        return Err(error(
            "immutable probe manifest changed during serialization",
        ));
    }
    serde_json::value::RawValue::from_string(
        String::from_utf8(bytes).map_err(|_| error("probe manifest UTF-8 differs"))?,
    )
    .map_err(|e| error(format!("probe manifest: {e}")))
}
