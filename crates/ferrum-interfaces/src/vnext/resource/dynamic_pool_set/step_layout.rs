//! Immutable Step resource relationships and reusable physical capacities.
//! Compile against the owning plan once. Per-wave logical and fit sizes are
//! evaluated afresh; no occupancy, pool availability or request state is cached.
use super::*;
use crate::vnext::resource::PhysicalBackingClaimIdentity;

pub(in crate::vnext::resource) struct StepSlotLayout {
    pub(in crate::vnext::resource) descriptor_indices: Vec<usize>,
    pub(in crate::vnext::resource) claim_identity: PhysicalBackingClaimIdentity,
    pub(in crate::vnext::resource) reusable_capacity: BTreeMap<ReusableExecutionBucketId, Vec<u64>>,
}

pub(super) fn compile_step_layouts(
    domains: &[DynamicPoolDomainSpec],
    reusable: Option<&ReusableExecutionMemoryPlan>,
) -> Result<Vec<Vec<StepSlotLayout>>, VNextError> {
    domains
        .iter()
        .map(|domain| {
            let by_resource: BTreeMap<_, _> = domain
                .descriptors
                .iter()
                .enumerate()
                .map(|(index, descriptor)| (descriptor.base_resource_id(), index))
                .collect();
            if by_resource.len() != domain.descriptors.len() {
                return Err(invalid_resource(
                    "Step layout requires unique pool descriptors",
                ));
            }
            domain
                .pool
                .step_resource_slots()
                .iter()
                .map(|slot| {
                    let descriptor_indices = slot
                        .resource_ids()
                        .iter()
                        .map(|id| {
                            let index = *by_resource.get(id).ok_or_else(|| {
                                invalid_resource(
                                    "Step slot references a descriptor outside its pool",
                                )
                            })?;
                            if domain.descriptors[index].lifetime() != AllocationLifetime::Step {
                                return Err(invalid_resource(
                                    "Step slot references a non-Step descriptor",
                                ));
                            }
                            Ok(index)
                        })
                        .collect::<Result<Vec<_>, VNextError>>()?;
                    let mut reusable_capacity = BTreeMap::new();
                    if let Some(reusable) = reusable {
                        for resolved in reusable.buckets() {
                            let bucket = resolved.bucket();
                            let capacity = bucket.capacity();
                            let shape = DynamicResourceShape::from_validated(
                                capacity.maximum_sequences(),
                                capacity.maximum_tokens(),
                                capacity.maximum_pages(),
                            );
                            let sizes = descriptor_indices
                                .iter()
                                .map(|&index| {
                                    domain.descriptors[index]
                                        .evaluate_request_bytes_for_shape(shape)
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            reusable_capacity.insert(bucket.bucket_id().clone(), sizes);
                        }
                    }
                    Ok(StepSlotLayout {
                        descriptor_indices,
                        claim_identity: PhysicalBackingClaimIdentity::new(
                            domain.pool_id().clone(),
                            slot.resource_ids().to_vec(),
                        )?,
                        reusable_capacity,
                    })
                })
                .collect()
        })
        .collect()
}
