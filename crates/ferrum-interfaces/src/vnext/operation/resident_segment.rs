//! Cold physical templates and one-wave authority for a complete binding program.
//! A template owns no old slot lease, dependency authority, or participant frame.
use std::any::Any;
use std::sync::Arc;

use super::foundation::invalid_operation;
use super::retained_dependency::{RetainedPlanDependencyAuthority, RetainedPlanDependencyIdentity};
use super::segment_binding::{PreparedSegmentBindingFrame, SegmentFrameResource};
use super::segment_compile::CompiledSegmentBindingRecipe;
use crate::vnext::resource::{
    ResidentBackingTemplate, SegmentBackingContext, SegmentPlanResourceView,
};
use crate::vnext::*;

/// Non-cloneable, freshly checked permission for one complete segment. The
/// runtime may retain the owners, but cannot manufacture another wave's permit.
pub struct FreshResidentSegmentPermit<'w> {
    work_shape: &'w BatchWorkShape,
    retention: DeviceBufferRetention,
    owner_retentions: Vec<DeviceBufferRetention>,
    program_bindings: Vec<Option<ProgramBindingNodeBinding>>,
    dependencies: Vec<Vec<Option<RetainedPlanDependencyAuthority>>>,
    batch_taken: bool,
    resource_view_bytes: Vec<u64>,
}

impl FreshResidentSegmentPermit<'_> {
    pub fn work_shape(&self) -> &BatchWorkShape {
        self.work_shape
    }
    pub fn retention(&self) -> &DeviceBufferRetention {
        &self.retention
    }
    pub fn owner_retentions(&self) -> &[DeviceBufferRetention] {
        &self.owner_retentions
    }
    pub fn resource_view_bytes(&self) -> &[u64] {
        &self.resource_view_bytes
    }
    pub fn node_count(&self) -> usize {
        self.program_bindings.len()
    }
    pub fn program_binding(&self, node: usize) -> Option<&ProgramBindingNodeBinding> {
        self.program_bindings.get(node).and_then(Option::as_ref)
    }
    pub fn take_dependency(
        &mut self,
        node: usize,
        dependency: usize,
    ) -> Result<RetainedPlanDependencyAuthority, VNextError> {
        let authority = self
            .dependencies
            .get_mut(node)
            .and_then(|row| row.get_mut(dependency))
            .and_then(Option::take)
            .ok_or_else(|| {
                invalid_operation("resident dependency is absent or already consumed")
            })?;
        self.batch_taken = true;
        Ok(authority)
    }
    pub fn take_dependency_batch(
        &mut self,
    ) -> Result<Vec<Vec<RetainedPlanDependencyAuthority>>, VNextError> {
        if self.batch_taken || self.dependencies.iter().flatten().any(Option::is_none) {
            return Err(invalid_operation(
                "resident dependency batch is already consumed",
            ));
        }
        self.batch_taken = true;
        Ok(self
            .dependencies
            .iter_mut()
            .map(|row| {
                row.iter_mut()
                    .map(|entry| entry.take().expect("checked fresh dependency"))
                    .collect()
            })
            .collect())
    }
    pub(super) fn dependencies_consumed(&self) -> bool {
        self.dependencies.iter().flatten().all(Option::is_none)
    }
}

struct PlanPhysicalIdentity {
    buffer: usize,
    transaction: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    generation: u64,
    descriptor: BufferDescriptor,
    owner: DeviceBufferOwnerIdentity,
}

impl PlanPhysicalIdentity {
    fn from_view<B>(view: &SegmentPlanResourceView<'_, B>) -> Self {
        Self {
            buffer: view.leased.buffer() as *const B as usize,
            transaction: view.leased.identity().clone(),
            admission: view.leased.admission().clone(),
            generation: view.leased.generation(),
            descriptor: view.leased.committed_descriptor().clone(),
            owner: view.retention.owner_identity(),
        }
    }
    fn matches<B>(&self, view: &SegmentPlanResourceView<'_, B>) -> bool {
        // The same live Plan root owns an immutable committed slot array.
        // Weak owner identity guards the facade pointer against cross-root ABA.
        self.owner.matches(&view.retention)
            && self.buffer == view.leased.buffer() as *const B as usize
            && &self.transaction == view.leased.identity()
            && &self.admission == view.leased.admission()
            && self.generation == view.leased.generation()
            && &self.descriptor == view.leased.committed_descriptor()
    }
}

pub(super) struct ResidentSegmentTemplate<B> {
    backend: Arc<dyn Any + Send + Sync>,
    plans: Vec<PlanPhysicalIdentity>,
    backing: Option<ResidentBackingTemplate<B>>,
    dependencies: Vec<Vec<RetainedPlanDependencyIdentity>>,
    view_bounds: Vec<(u64, Option<u64>)>,
    unique_physical_buffers: u64,
}

impl<B> ResidentSegmentTemplate<B> {
    pub(super) fn capture(
        backend: Arc<dyn Any + Send + Sync>,
        frame: &PreparedSegmentBindingFrame<'_, B>,
        groups: &[&[LogicalBackingSliceAuthority]],
        unique_physical_buffers: u64,
    ) -> Result<Self, VNextError> {
        let mut view_bounds = vec![(0u64, None::<u64>); frame.recipe.resources.len()];
        for node in &frame.recipe.nodes {
            for region in &node.regions {
                let bound = &mut view_bounds[region.resource_index];
                let minimum = region
                    .offset_bytes
                    .checked_add(match region.request.extent {
                        SegmentBindingRegionExtent::Exact(bytes) => bytes,
                        SegmentBindingRegionExtent::CurrentResource => 1,
                    })
                    .ok_or_else(|| invalid_operation("resident window minimum overflows"))?;
                bound.0 = bound.0.max(minimum);
                if region.request.extent == SegmentBindingRegionExtent::CurrentResource {
                    if let Some(component) = region.component_length {
                        let maximum = component
                            .checked_sub(region.request.offset_bytes)
                            .and_then(|bytes| bytes.checked_add(region.offset_bytes))
                            .ok_or_else(|| {
                                invalid_operation("resident window maximum overflows")
                            })?;
                        bound.1 = Some(bound.1.map_or(maximum, |prior| prior.min(maximum)));
                    }
                }
            }
        }
        Ok(Self {
            backend,
            plans: frame
                .plan_views
                .iter()
                .map(PlanPhysicalIdentity::from_view)
                .collect(),
            backing: frame
                .backing
                .map(|backing| backing.resident_template(groups, frame.expected))
                .transpose()?,
            dependencies: frame
                .dependencies
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|dependency| {
                            dependency
                                .as_ref()
                                .map(|dependency| dependency.identity.clone())
                                .ok_or_else(|| {
                                    invalid_operation(
                                        "cold resident dependency was already consumed",
                                    )
                                })
                        })
                        .collect()
                })
                .collect::<Result<_, _>>()?,
            view_bounds,
            unique_physical_buffers,
        })
    }
}

pub(super) fn try_encode<'w, R: DeviceRuntime>(
    runtime: &R,
    recipe: &CompiledSegmentBindingRecipe,
    wave: &'w PreparedStepSubmissionWave<R>,
    context: &SegmentBackingContext<'_, R>,
    plans: &[SegmentPlanResourceView<'_, R::Buffer>],
    groups: &[&[LogicalBackingSliceAuthority]],
    expected: &[SegmentBackingExpectation],
    resources: &[SegmentFrameResource],
) -> Result<Option<(EncodedSegmentBindingProgram<R::Command>, Arc<()>, u64)>, VNextError> {
    let cached = recipe
        .resident_frame
        .lock()
        .map_err(|_| invalid_operation("resident frame cache is poisoned"))?
        .clone();
    let Some(cached) = cached else {
        return Ok(None);
    };
    let Some(template) = cached.downcast_ref::<ResidentSegmentTemplate<R::Buffer>>() else {
        return Err(invalid_operation(
            "resident frame belongs to another runtime buffer type",
        ));
    };
    // Plan views are fresh and share one current provisioning lease. No old
    // Plan dependency scope or retention is accepted as new authority.
    let plan_matches = plans.len() == template.plans.len()
        && plans
            .iter()
            .zip(&template.plans)
            .all(|(view, old)| old.matches(view));
    let dynamic_owners = match (&template.backing, expected.is_empty()) {
        (Some(backing), false) => context.authorize_resident(backing, groups, expected)?,
        (None, true) => Some(Vec::new()),
        _ => return Ok(None),
    };
    let Some(dynamic_owners) = dynamic_owners else {
        return Ok(None);
    };
    if !plan_matches {
        return Ok(None);
    }
    if resources.len() != template.view_bounds.len() {
        return Ok(None);
    }
    let resource_view_bytes = resources
        .iter()
        .zip(&template.view_bounds)
        .map(|(resource, &(minimum, maximum))| {
            let bytes = match resource {
                SegmentFrameResource::Plan(index) => {
                    plans[*index].leased.committed_descriptor().size_bytes
                }
                SegmentFrameResource::Dynamic(rows) => rows.row(0, expected)?.2,
            };
            if bytes < minimum || maximum.is_some_and(|maximum| bytes > maximum) {
                return Err(invalid_operation(
                    "resident current logical window exceeds its compiled component bounds",
                ));
            }
            Ok(bytes)
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    let scope = Arc::new(());
    let mut owner_retentions = plans
        .iter()
        .map(|view| view.retention.clone())
        .collect::<Vec<_>>();
    owner_retentions.extend(dynamic_owners);
    let retention = DeviceBufferRetention::lane_pair(
        wave.execution_lane_id(),
        Arc::new(owner_retentions.clone()),
        Arc::clone(&scope),
    );
    // prepare_segment_plan_views proved every participant uses the same exact
    // provisioning lease; every view retains that same live Plan root. Cold
    // issuance checked each source/destination in this immutable slot array.
    let plan_retention = plans.first().map(|view| &view.retention);
    let dependencies = template
        .dependencies
        .iter()
        .map(|row| {
            row.iter()
                .map(|identity| {
                    let retained = plan_retention.ok_or_else(|| {
                        invalid_operation("resident Plan dependency has no current Plan retention")
                    })?;
                    Ok(Some(
                        RetainedPlanDependencyAuthority::from_resident_identity(
                            identity.clone(),
                            &scope,
                            retained,
                        ),
                    ))
                })
                .collect()
        })
        .collect::<Result<Vec<Vec<_>>, VNextError>>()?;
    let mut permit = FreshResidentSegmentPermit {
        work_shape: wave
            .nodes()
            .first()
            .ok_or_else(|| invalid_operation("resident wave is empty"))?
            .work_shape(),
        retention,
        owner_retentions,
        program_bindings: recipe
            .nodes
            .iter()
            .map(|node| wave.claimed_backing().program_binding_node(node.node_index))
            .collect(),
        dependencies,
        batch_taken: false,
        resource_view_bytes,
    };
    let encoded = runtime
        .encode_resident_segment_bindings(template.backend.as_ref(), &mut permit)
        .map_err(|error| invalid_operation(format!("resident segment encoding failed: {error}")))?;
    let Some(encoded) = encoded else {
        if permit.batch_taken {
            return Err(invalid_operation(
                "resident backend declined after consuming fresh authority",
            ));
        }
        return Ok(None);
    };
    if !permit.dependencies_consumed() {
        return Err(invalid_operation(
            "resident backend did not consume every fresh dependency",
        ));
    }
    Ok(Some((encoded, scope, template.unique_physical_buffers)))
}
