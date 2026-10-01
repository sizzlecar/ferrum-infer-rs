//! Plan-owned physical layout identity. No logical demand, lane, occupancy,
//! allocation or resource authority is retained. The original wire digest is
//! computed once; every use compares all of its current input fields.
use super::*;

struct Projection {
    resource: ResourceId,
    offset: u64,
    capacity: u64,
}

struct Request {
    claim: PhysicalBackingClaimIdentity,
    capacity: u64,
    projections: Vec<Projection>,
}

pub(in crate::vnext::resource) struct CompiledLaneStableLayout {
    lifetime: AllocationLifetime,
    bucket: ReusableExecutionBucketId,
    requests: Vec<Request>,
    fingerprint: String,
}

impl CompiledLaneStableLayout {
    pub(in crate::vnext::resource) fn new(
        lifetime: AllocationLifetime,
        bucket: &ReusableExecutionBucketId,
        requests: &[EvaluatedBackingRequest<'_>],
    ) -> Result<Option<Self>, VNextError> {
        if requests.is_empty() {
            return Ok(None);
        }
        let mut canonical = requests.iter().collect::<Vec<_>>();
        canonical.sort_unstable_by(|a, b| a.claim_identity.cmp(&b.claim_identity));
        // Preserve the complete old key validation and digest, including the
        // ordered projection list. A lane is deliberately absent from storage.
        if canonical
            .windows(2)
            .any(|w| w[0].claim_identity >= w[1].claim_identity)
            || canonical.iter().any(|request| {
                request.reusable_execution_bucket_id.as_ref() != Some(bucket)
                    || request.projections.is_empty()
                    || request
                        .projections
                        .iter()
                        .any(|p| p.descriptor.lifetime() != lifetime)
            })
        {
            return Err(invalid_resource(
                "compiled lane layout has inconsistent immutable inputs",
            ));
        }
        let fingerprint = lane_stable_layout_fingerprint(lifetime, bucket, &canonical)?;
        let requests = canonical
            .into_iter()
            .map(|request| Request {
                claim: request.claim_identity.clone(),
                capacity: request.capacity_size_bytes,
                projections: request
                    .projections
                    .iter()
                    .map(|projection| Projection {
                        resource: projection.descriptor.base_resource_id().clone(),
                        offset: projection.physical_offset_bytes,
                        capacity: projection.capacity_size_bytes,
                    })
                    .collect(),
            })
            .collect();
        Ok(Some(Self {
            lifetime,
            bucket: bucket.clone(),
            requests,
            fingerprint,
        }))
    }

    pub(super) fn matching_fingerprint(
        &self,
        lifetime: AllocationLifetime,
        bucket: &ReusableExecutionBucketId,
        requests: &[&EvaluatedBackingRequest<'_>],
    ) -> Option<&str> {
        if self.lifetime != lifetime
            || &self.bucket != bucket
            || self.requests.len() != requests.len()
        {
            return None;
        }
        for (compiled, current) in self.requests.iter().zip(requests) {
            if compiled.claim != current.claim_identity
                || compiled.capacity != current.capacity_size_bytes
                || current.reusable_execution_bucket_id.as_ref() != Some(bucket)
                || compiled.projections.len() != current.projections.len()
            {
                return None;
            }
            for (compiled, current) in compiled.projections.iter().zip(&current.projections) {
                if &compiled.resource != current.descriptor.base_resource_id()
                    || current.descriptor.lifetime() != lifetime
                    || compiled.offset != current.physical_offset_bytes
                    || compiled.capacity != current.capacity_size_bytes
                {
                    return None;
                }
            }
        }
        Some(&self.fingerprint)
    }
}
