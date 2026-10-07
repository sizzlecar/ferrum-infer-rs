//! Retained local projection arithmetic shared by the two attention providers.
//! No numerical choice is made from token count, launch success or model name.
use super::super::binding;
use super::super::native_blocks::q8act::{self, Q8ActKernels};
use super::super::native_blocks::weights;
use crate::backend::cuda::vnext_replay::CudaCommandReplayKeyBuilder;
use crate::backend::cuda::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceRuntimeError,
};
use ferrum_interfaces::vnext::{
    BatchedOperationInvocation, PreparedProjection, PreparedProjectionNumerics, ProjectionRole,
    Q8ActAttentionProfile, Q8ActSwiGluProfile, ResolvedValueBinding, ResolvedValueRole,
};

#[derive(Clone)]
pub(super) struct PreparedAttentionProjections {
    profile: Q8ActAttentionProfile,
    numerics: PreparedProjectionNumerics,
    bytes_per_token: u64,
}

impl PreparedAttentionProjections {
    pub fn prepare(
        profile: Q8ActAttentionProfile,
        values: &[ResolvedValueBinding],
    ) -> Result<Self, String> {
        let numerics = PreparedProjectionNumerics::prepare(&profile.arithmetic(), values)?;
        let result = Self::from_numerics(profile, numerics)?;
        for projection in result.numerics.projections() {
            let value = binding(
                values,
                ResolvedValueRole::Input,
                projection.weight_input_ordinal(),
            )?;
            let parts = weights::matrix_parts(
                value
                    .weight()
                    .ok_or("attention projection has no weights")?,
                value.tensor().dimensions(),
            )?;
            result.validate_parts(projection.role(), &parts)?;
        }
        Ok(result)
    }

    fn from_numerics(
        profile: Q8ActAttentionProfile,
        numerics: PreparedProjectionNumerics,
    ) -> Result<Self, String> {
        let bytes_per_token = q8act::workspace_per_token(&numerics)?;
        Ok(Self {
            profile,
            numerics,
            bytes_per_token,
        })
    }

    pub fn from_invocation(
        profile: Q8ActAttentionProfile,
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Self, String> {
        let first = invocation
            .participants()
            .first()
            .ok_or("empty Q8act attention invocation")?;
        let numerics = first
            .projection_numerics()
            .ok_or("Q8act attention has no retained projection decision")?;
        if numerics.contract() != &profile.arithmetic() {
            return Err("Q8act attention retained policy differs from its provider".into());
        }
        for participant in invocation.participants() {
            if participant.projection_numerics() != Some(numerics) {
                return Err("Q8act attention participants disagree on retained numerics".into());
            }
        }
        // Plan compilation/import has already rebuilt this typed decision.
        // Encoding compares the actual resolved native matrices below, while
        // invocation construction validates live resources, leases and shapes.
        // Do not rebuild the entire static leaf inventory for every decode row.
        Self::from_numerics(profile, numerics.clone())
    }

    pub fn into_numerics(self) -> PreparedProjectionNumerics {
        self.numerics
    }
    pub fn bytes_per_token(&self) -> u64 {
        self.bytes_per_token
    }
    pub fn alignment_slop(&self) -> u64 {
        if self.bytes_per_token == 0 {
            0
        } else {
            15
        }
    }
    pub fn required_bytes(&self, base: u64, rows: u64) -> Result<u64, String> {
        appended_workspace(base, self.bytes_per_token, rows).map(|(_, _, end)| end)
    }
    pub fn projection(&self, role: ProjectionRole) -> Result<&PreparedProjection, String> {
        self.numerics
            .projection(role)
            .ok_or_else(|| format!("missing retained attention projection {role:?}"))
    }
    pub fn validate_parts(
        &self,
        role: ProjectionRole,
        parts: &[weights::MatrixPart],
    ) -> Result<(), String> {
        Q8ActKernels::validate_parts_for_profile(
            Q8ActSwiGluProfile::Q4KQ5KIq4Xs,
            self.projection(role)?,
            parts,
        )
    }
    pub fn needs_pack(&self, roles: &[ProjectionRole]) -> Result<bool, String> {
        roles.iter().try_fold(false, |required, &role| {
            Ok(required || self.projection(role)?.has_staged_leaf())
        })
    }
    pub fn pack_count(&self) -> Result<u64, String> {
        use ProjectionRole::*;
        let groups: &[&[ProjectionRole]] = match self.profile {
            Q8ActAttentionProfile::GatedDelta => &[&[GatedDeltaInput], &[GatedDeltaOutput]],
            Q8ActAttentionProfile::Causal => {
                &[&[CausalQuery, CausalKey, CausalValue], &[CausalOutput]]
            }
        };
        groups.iter().try_fold(0, |count, roles| {
            Ok(count + u64::from(self.needs_pack(roles)?))
        })
    }
    pub fn replay_key(
        &self,
        key: CudaCommandReplayKeyBuilder,
    ) -> Result<CudaCommandReplayKeyBuilder, String> {
        Ok(key.bytes(&self.replay_bytes()?).u64(self.bytes_per_token))
    }
    pub fn replay_bytes(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&self.numerics).map_err(|e| e.to_string())
    }

    /// Append one invocation-owned region after the ordinary attention layout.
    /// Every consumer is on the same stream; output packing follows Q/K/V use.
    pub fn workspace(
        &self,
        scratch: &CudaBufferRegion,
        base_bytes: u64,
        rows: u64,
    ) -> Result<(u64, u64), CudaDeviceRuntimeError> {
        let (offset, bytes, required) = appended_workspace(base_bytes, self.bytes_per_token, rows)
            .map_err(CudaDeviceRuntimeError::contract)?;
        if scratch.length_bytes() < required {
            return Err(CudaDeviceRuntimeError::contract(
                "attention Q8act scratch exceeds admitted region",
            ));
        }
        let pointer = scratch.device_ptr().checked_add(offset).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("attention Q8act scratch address overflows")
        })?;
        Ok((pointer, bytes))
    }
}

fn appended_workspace(base: u64, per_token: u64, rows: u64) -> Result<(u64, u64, u64), String> {
    if rows == 0 {
        return Err("Q8act attention scratch has no rows".into());
    }
    let offset = if per_token == 0 {
        base
    } else {
        base.checked_add(15)
            .ok_or("Q8act scratch alignment overflows")?
            & !15
    };
    let bytes = per_token
        .checked_mul(rows)
        .ok_or("Q8act scratch extent overflows")?;
    let end = offset
        .checked_add(bytes)
        .ok_or("Q8act scratch end overflows")?;
    Ok((offset, bytes, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attention_q8act_append_workspace_preserves_prefix_alignment_and_bounds() {
        for base in [0, 1, 15, 16, 17, 1023] {
            for rows in [1, 3, 8, 9, 1025, u64::from(u16::MAX)] {
                for k in [256, 5120, 6144] {
                    let per_token = q8act::PackLayout::new(1, k).unwrap().total_bytes;
                    let (offset, bytes, end) = appended_workspace(base, per_token, rows).unwrap();
                    assert_eq!(offset % 16, 0);
                    assert!(offset >= base && offset - base < 16);
                    assert_eq!(bytes, q8act::PackLayout::new(rows, k).unwrap().total_bytes);
                    assert_eq!(end, offset + bytes);
                    assert!(end <= base + 15 + per_token * rows);
                }
            }
        }
        assert_eq!(appended_workspace(13, 0, 1).unwrap(), (13, 0, 13));
        for (base, per_token, rows) in [(0, 1, 0), (u64::MAX, 1, 1), (0, u64::MAX, 3)] {
            assert!(appended_workspace(base, per_token, rows).is_err());
        }
    }
}
