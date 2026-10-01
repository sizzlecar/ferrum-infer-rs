//! Compile the Step and submission-wave physical layouts for every declared
//! reusable bucket. Retention is bounded by the immutable plan, with no runtime
//! shape-keyed cache. Demand and physical availability are evaluated separately.
use super::*;
use crate::vnext::resource::dynamic_pool::EvaluatedBackingProjection;
use crate::vnext::resource::lane_stable_arena::CompiledLaneStableLayout;

pub(in crate::vnext::resource) struct CompiledLaneStableLayouts {
    buckets: BTreeMap<ReusableExecutionBucketId, [Option<CompiledLaneStableLayout>; 2]>,
}

impl CompiledLaneStableLayouts {
    pub(super) fn compile(
        domains: &[DynamicPoolDomainSpec],
        step: &[Vec<StepSlotLayout>],
        invocation: &[Option<SubmissionWaveDomainLayout>],
        capacities: &BTreeMap<
            ReusableExecutionBucketId,
            Vec<Option<SubmissionWaveDomainCapacityLayout>>,
        >,
        reusable: Option<&ReusableExecutionMemoryPlan>,
    ) -> Result<Self, VNextError> {
        let mut buckets = BTreeMap::new();
        let Some(reusable) = reusable else {
            return Ok(Self { buckets });
        };
        if domains.len() != step.len() || domains.len() != invocation.len() {
            return Err(invalid_resource(
                "compiled lane layout domain counts differ",
            ));
        }
        for resolved in reusable.buckets() {
            let bucket = resolved.bucket().bucket_id();
            let capacity = capacities
                .get(bucket)
                .ok_or_else(|| invalid_resource("compiled lane layout bucket is missing"))?;
            if capacity.len() != domains.len() {
                return Err(invalid_resource(
                    "compiled lane capacity domain count differs",
                ));
            }
            let mut step_requests = Vec::new();
            let mut invocation_requests = Vec::new();
            for (index, domain) in domains.iter().enumerate() {
                for slot in &step[index] {
                    let sizes = slot.reusable_capacity.get(bucket).ok_or_else(|| {
                        invalid_resource("compiled Step lane capacity is missing")
                    })?;
                    if sizes.len() != slot.descriptor_indices.len() {
                        return Err(invalid_resource(
                            "compiled Step lane projection count differs",
                        ));
                    }
                    let projections = slot
                        .descriptor_indices
                        .iter()
                        .zip(sizes)
                        .map(|(&descriptor_index, &size)| {
                            let descriptor =
                                domain.descriptors.get(descriptor_index).ok_or_else(|| {
                                    invalid_resource("compiled Step lane descriptor is missing")
                                })?;
                            Ok(EvaluatedBackingProjection {
                                descriptor,
                                physical_offset_bytes: 0,
                                logical_size_bytes: size,
                                capacity_size_bytes: size,
                            })
                        })
                        .collect::<Result<Vec<_>, VNextError>>()?;
                    step_requests.push(EvaluatedBackingRequest {
                        domain,
                        claim_identity: slot.claim_identity.clone(),
                        capacity_size_bytes: sizes.iter().copied().max().unwrap_or(0),
                        reusable_execution_bucket_id: Some(bucket.clone()),
                        projections,
                    });
                }
                match (&invocation[index], &capacity[index]) {
                    (None, None) => {}
                    (Some(layout), Some(capacity)) => {
                        if layout.projection_count != capacity.projections.len() {
                            return Err(invalid_resource(
                                "compiled Invocation lane projection count differs",
                            ));
                        }
                        let mut projections = vec![None; layout.projection_count];
                        for row in &layout.rows {
                            for projection in &row.projections {
                                let descriptor = domain
                                    .descriptors
                                    .get(projection.descriptor_index)
                                    .ok_or_else(|| {
                                        invalid_resource(
                                            "compiled Invocation lane descriptor is missing",
                                        )
                                    })?;
                                let physical = capacity
                                    .projections
                                    .get(projection.projection_index)
                                    .ok_or_else(|| {
                                        invalid_resource(
                                            "compiled Invocation lane capacity is missing",
                                        )
                                    })?;
                                let output = projections
                                    .get_mut(projection.projection_index)
                                    .ok_or_else(|| {
                                        invalid_resource(
                                            "compiled Invocation lane projection is missing",
                                        )
                                    })?;
                                if output
                                    .replace(EvaluatedBackingProjection {
                                        descriptor,
                                        physical_offset_bytes: physical.physical_offset_bytes,
                                        logical_size_bytes: physical.capacity_size_bytes,
                                        capacity_size_bytes: physical.capacity_size_bytes,
                                    })
                                    .is_some()
                                {
                                    return Err(invalid_resource(
                                        "compiled Invocation lane projection is repeated",
                                    ));
                                }
                            }
                        }
                        let projections = projections
                            .into_iter()
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| {
                                invalid_resource(
                                    "compiled Invocation lane projections are incomplete",
                                )
                            })?;
                        invocation_requests.push(EvaluatedBackingRequest {
                            domain,
                            claim_identity: layout.claim_identity.clone(),
                            capacity_size_bytes: capacity.physical_size_bytes,
                            reusable_execution_bucket_id: Some(bucket.clone()),
                            projections,
                        });
                    }
                    _ => {
                        return Err(invalid_resource(
                            "compiled Invocation lane layout/capacity mismatch",
                        ))
                    }
                }
            }
            buckets.insert(
                bucket.clone(),
                [
                    CompiledLaneStableLayout::new(
                        AllocationLifetime::Step,
                        bucket,
                        &step_requests,
                    )?,
                    CompiledLaneStableLayout::new(
                        AllocationLifetime::Invocation,
                        bucket,
                        &invocation_requests,
                    )?,
                ],
            );
        }
        Ok(Self { buckets })
    }

    pub(in crate::vnext::resource) fn get(
        &self,
        bucket: Option<&ReusableExecutionBucketId>,
        lifetime: AllocationLifetime,
    ) -> Option<&CompiledLaneStableLayout> {
        let index = match lifetime {
            AllocationLifetime::Step => 0,
            AllocationLifetime::Invocation => 1,
            _ => return None,
        };
        self.buckets.get(bucket?)?[index].as_ref()
    }
}
