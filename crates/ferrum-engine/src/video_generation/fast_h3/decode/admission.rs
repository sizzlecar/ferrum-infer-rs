//! Select an optional decoder group against physical Host ceilings before
//! reserving request credit. Occupancy remains the scheduler's admission work.
use crate::video_generation::fast_h3::conditioning::FastH3ConditioningReuse;
use ferrum_models::spatial_tiles::{SpatialTileBatch, SpatialTileGroups};
#[cfg(feature = "cuda-fast-h3")]
use ferrum_runtime::ModelProgram;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryError, HostMemoryScope};

type Claims = Vec<(HostMemoryClass, usize)>;

/// Each external fixed tensor is a separate physical Host allocation. This is
/// an upper bound for inputs that can instead borrow an existing immutable view.
#[cfg(feature = "cuda-fast-h3")]
pub(crate) fn input_backing_bytes(
    program: &ModelProgram,
    host: &HostMemoryAccount,
) -> anyhow::Result<usize> {
    use anyhow::Context;
    program.inputs().iter().try_fold(0usize, |total, input| {
        let spec = program
            .input_tensor_specs()
            .get(input)
            .context("fixed decoder input lacks declared geometry")?;
        total
            .checked_add(host.backing_bytes(usize::try_from(spec.byte_len()?)?)?)
            .context("decoder input Host bound overflow")
    })
}

/// Conditioning has fenced and its temporary inputs have retired. Retain the
/// actual later input bound, while preserving decoder output/transfer/weight
/// credit and every live payload charge. The same scoped account stays valid.
pub(crate) fn release_conditioning_input_credit(
    parent: &HostMemoryAccount,
    scope: &HostMemoryScope,
    original: &[(HostMemoryClass, usize)],
    later_input_bytes: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        original
            .iter()
            .filter(|(class, _)| *class == HostMemoryClass::InputPreprocess)
            .count()
            == 1,
        "request must declare one input credit bound"
    );
    let credit: Vec<_> = original
        .iter()
        .map(|&(class, bytes)| {
            (
                class,
                if class == HostMemoryClass::InputPreprocess {
                    later_input_bytes
                } else {
                    bytes
                },
            )
        })
        .collect();
    parent.shrink_scope_unused_to_at_most(scope, &credit)?;
    Ok(())
}

pub(crate) fn select_host_tile_group(
    groups: SpatialTileGroups,
    mut reuse: FastH3ConditioningReuse,
    host: &HostMemoryAccount,
    mut claims: impl FnMut(SpatialTileBatch, FastH3ConditioningReuse) -> anyhow::Result<Claims>,
) -> anyhow::Result<(SpatialTileBatch, FastH3ConditioningReuse, Claims)> {
    loop {
        for batch in groups.uniform_batches() {
            let count = batch.maximum_tiles();
            let claims = claims(batch, reuse)?;
            match host.check_scope_capacity(&claims) {
                Ok(()) => return Ok((batch, reuse, claims)),
                Err(HostMemoryError::Capacity { .. }) if count > 1 => {}
                Err(HostMemoryError::Capacity { .. })
                    if !matches!(reuse, FastH3ConditioningReuse::Disabled {}) =>
                {
                    reuse = FastH3ConditioningReuse::Disabled {};
                    break;
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

    #[test]
    fn completed_conditioning_releases_only_obsolete_input_credit_for_the_next_stage() {
        let parent = HostMemoryAccount::new(
            HostMemoryLimits::new(
                1200,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 1200,
                    output_bytes: 1200,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        );
        let claims = [
            (HostMemoryClass::InputPreprocess, 700),
            (HostMemoryClass::Output, 200),
        ];
        let scope = parent.reserve_scope(&claims).unwrap();
        let account = scope.account();
        let state = account
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 100)
            .unwrap();
        let temporary = account
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 400)
            .unwrap();
        let other = parent.allocate_zeroed(HostMemoryClass::Output, 50).unwrap();
        let batch_claim = [(HostMemoryClass::Output, 600)];
        assert!(parent
            .reserve_scope(&batch_claim)
            .unwrap_err()
            .is_temporary_capacity());
        drop(temporary);
        release_conditioning_input_credit(&parent, &scope, &claims, 150).unwrap();
        let batch = parent.reserve_scope(&batch_claim).unwrap();
        // Future solver input and decoder Output remain authorized on the same
        // account. Neither a live payload nor another scope's credit was taken.
        let successor = account
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 150)
            .unwrap();
        let decoded = account
            .allocate_zeroed(HostMemoryClass::Output, 200)
            .unwrap();
        assert!(matches!(
            account.allocate_zeroed(HostMemoryClass::InputPreprocess, 1),
            Err(HostMemoryError::ScopeCapacity { .. })
        ));
        drop(state);
        let next = account
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 100)
            .unwrap();
        batch.close().unwrap();
        scope.close().unwrap();
        drop((successor, decoded, next, other));
        assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    }

    fn host(output: usize) -> HostMemoryAccount {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                output + 16,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 16,
                    transfer_staging_bytes: 0,
                    output_bytes: output,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
    }

    fn claims(batch: SpatialTileBatch, reuse: FastH3ConditioningReuse) -> anyhow::Result<Claims> {
        Ok(vec![
            (HostMemoryClass::InputPreprocess, 16),
            (
                HostMemoryClass::Output,
                reuse.output_claim(batch.maximum_tiles() * 32, 0)?,
            ),
        ])
    }

    #[test]
    fn physical_class_boundary_shrinks_group_but_occupancy_waits() {
        let maximum = SpatialTileBatch::new(4).unwrap().groups(12).unwrap();
        let account = host(128);
        let occupied = account.allocate_zeroed(HostMemoryClass::Output, 1).unwrap();
        let disabled = FastH3ConditioningReuse::Disabled {};
        let (batch, _, claims) =
            select_host_tile_group(maximum, disabled, &account, claims).unwrap();
        assert_eq!(batch.maximum_tiles(), 4);
        assert!(account
            .reserve_scope(&claims)
            .unwrap_err()
            .is_temporary_capacity());
        drop(occupied);
        let scope = account.reserve_scope(&claims).unwrap();
        scope.close().unwrap();
        let (batch, _, _) =
            select_host_tile_group(maximum, disabled, &host(127), self::claims).unwrap();
        assert_eq!(batch.maximum_tiles(), 3);
        let (batch, _, _) =
            select_host_tile_group(maximum, disabled, &host(32), self::claims).unwrap();
        assert_eq!(batch.maximum_tiles(), 1);
    }

    #[test]
    fn larger_groups_shrink_to_real_divisors_before_reserving() {
        let groups = SpatialTileBatch::new(usize::MAX)
            .unwrap()
            .groups(12)
            .unwrap();
        let disabled = FastH3ConditioningReuse::Disabled {};
        let (batch, _, _) = select_host_tile_group(groups, disabled, &host(384), claims).unwrap();
        assert_eq!(batch.maximum_tiles(), 12);
        let (batch, _, _) = select_host_tile_group(groups, disabled, &host(383), claims).unwrap();
        assert_eq!(batch.maximum_tiles(), 6);
        let mut attempts = Vec::new();
        let (batch, _, _) = select_host_tile_group(groups, disabled, &host(191), |batch, reuse| {
            attempts.push(batch.maximum_tiles());
            claims(batch, reuse)
        })
        .unwrap();
        assert_eq!(batch.maximum_tiles(), 4);
        assert_eq!(attempts, [12, 6, 4]);
    }

    #[test]
    fn impossible_scalar_and_invalid_claims_keep_original_errors() {
        let maximum = SpatialTileBatch::new(4).unwrap().groups(12).unwrap();
        let disabled = FastH3ConditioningReuse::Disabled {};
        let error = select_host_tile_group(maximum, disabled, &host(31), claims).unwrap_err();
        let capacity = error.downcast_ref::<HostMemoryError>().unwrap();
        assert!(matches!(capacity, HostMemoryError::Capacity { .. }));
        assert!(!capacity.is_temporary_capacity());
        let error = select_host_tile_group(maximum, disabled, &host(128), |_, _| {
            Err(HostMemoryError::ArithmeticOverflow.into())
        })
        .unwrap_err();
        assert_eq!(
            error.downcast_ref::<HostMemoryError>(),
            Some(&HostMemoryError::ArithmeticOverflow)
        );
    }

    #[test]
    fn optional_retention_declines_before_rejecting_a_supported_scalar() {
        let (batch, reuse, claims) = select_host_tile_group(
            SpatialTileBatch::new(4).unwrap().groups(12).unwrap(),
            FastH3ConditioningReuse::LastResult {
                maximum_retained_bytes: 64,
            },
            &host(32),
            self::claims,
        )
        .unwrap();
        assert_eq!(batch.maximum_tiles(), 1);
        assert!(!reuse.retains(1));
        host(32).reserve_scope(&claims).unwrap().close().unwrap();
    }

    #[test]
    fn retained_conditioning_and_decoder_have_simultaneous_physical_credit() {
        let policy = FastH3ConditioningReuse::LastResult {
            maximum_retained_bytes: 64,
        };
        let claim = policy.output_claim(96, 128).unwrap();
        let account = host(claim);
        let scope = account
            .reserve_scope(&[(HostMemoryClass::Output, claim)])
            .unwrap();
        let scoped = scope.account();
        let cached = scoped
            .allocate_zeroed(HostMemoryClass::Output, 64)
            .unwrap()
            .freeze();
        let decoded = scoped.allocate_zeroed(HostMemoryClass::Output, 96).unwrap();
        assert_eq!(account.usage().unwrap().allocated_bytes(), 160);
        drop(decoded);
        scope.close().unwrap();
        assert_eq!(account.usage().unwrap().committed_bytes(), 64);
        drop(cached);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
        let short = account
            .reserve_scope(&[(HostMemoryClass::Output, claim - 1)])
            .unwrap();
        let _cached = short
            .account()
            .allocate_zeroed(HostMemoryClass::Output, 64)
            .unwrap();
        assert!(matches!(
            short.account().allocate_zeroed(HostMemoryClass::Output, 96),
            Err(HostMemoryError::ScopeCapacity { .. })
        ));
    }
}
