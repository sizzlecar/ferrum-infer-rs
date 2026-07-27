//! Shared physical-to-logical weight layout compiler for model packages.

use ferrum_interfaces::vnext::{
    PhysicalStorageLayout, PhysicalWeightComponentBinding, PhysicalWeightLayout,
    PhysicalWeightPadding, VNextError, WeightId,
};

pub(super) fn contiguous_or_reshaped_binding(
    family_id: &str,
    component_id: WeightId,
    physical_dimensions: &[u64],
    logical_dimensions: &[u64],
) -> Result<PhysicalWeightComponentBinding, VNextError> {
    match row_major_reshape_storage(family_id, physical_dimensions, logical_dimensions)? {
        None => Ok(PhysicalWeightComponentBinding::exact_contiguous(
            component_id,
        )),
        Some(storage) => Ok(PhysicalWeightComponentBinding {
            component_id,
            storage,
        }),
    }
}

pub(super) fn dense_or_reshaped_layout(
    family_id: &str,
    component_id: WeightId,
    physical_dimensions: &[u64],
    logical_dimensions: &[u64],
) -> Result<PhysicalWeightLayout, VNextError> {
    match row_major_reshape_storage(family_id, physical_dimensions, logical_dimensions)? {
        None => Ok(PhysicalWeightLayout::Dense { component_id }),
        Some(storage) => Ok(PhysicalWeightLayout::Stored {
            component: PhysicalWeightComponentBinding {
                component_id,
                storage,
            },
        }),
    }
}

fn row_major_reshape_storage(
    family_id: &str,
    physical_dimensions: &[u64],
    logical_dimensions: &[u64],
) -> Result<Option<PhysicalStorageLayout>, VNextError> {
    if physical_dimensions == logical_dimensions {
        return Ok(None);
    }
    let physical_elements = checked_dimension_product(family_id, physical_dimensions)?;
    let logical_elements = checked_dimension_product(family_id, logical_dimensions)?;
    if physical_elements != logical_elements {
        return Err(invalid_dimensions(
            family_id,
            "physical reshape changes the logical element count",
        ));
    }
    Ok(Some(PhysicalStorageLayout::Strided {
        strides_in_elements: row_major_strides(family_id, logical_dimensions)?,
        padding: PhysicalWeightPadding::Exact,
    }))
}

fn checked_dimension_product(family_id: &str, dimensions: &[u64]) -> Result<u64, VNextError> {
    dimensions.iter().try_fold(1_u64, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid_dimensions(family_id, "tensor size overflows u64"))
    })
}

fn row_major_strides(family_id: &str, dimensions: &[u64]) -> Result<Vec<u64>, VNextError> {
    let mut strides = vec![0_u64; dimensions.len()];
    let mut stride = 1_u64;
    for (axis, extent) in dimensions.iter().enumerate().rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(*extent)
            .ok_or_else(|| invalid_dimensions(family_id, "tensor stride overflows u64"))?;
    }
    Ok(strides)
}

fn invalid_dimensions(family_id: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: family_id.to_owned(),
        field: "weights.dimensions".to_owned(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id() -> WeightId {
        WeightId::new("component.test").unwrap()
    }

    #[test]
    fn exact_shape_stays_compact_and_reshape_gets_explicit_strides() {
        assert!(matches!(
            dense_or_reshaped_layout("family.test", id(), &[4, 3], &[4, 3]).unwrap(),
            PhysicalWeightLayout::Dense { .. }
        ));
        assert!(matches!(
            contiguous_or_reshaped_binding("family.test", id(), &[4, 3], &[4, 3])
                .unwrap()
                .storage,
            PhysicalStorageLayout::Contiguous {
                padding: PhysicalWeightPadding::Exact
            }
        ));
        assert!(matches!(
            dense_or_reshaped_layout("family.test", id(), &[4, 1, 3], &[4, 3]).unwrap(),
            PhysicalWeightLayout::Stored {
                component: PhysicalWeightComponentBinding {
                    storage: PhysicalStorageLayout::Strided {
                        strides_in_elements,
                        padding: PhysicalWeightPadding::Exact,
                    },
                    ..
                }
            } if strides_in_elements == [3, 1]
        ));
    }

    #[test]
    fn reshape_fails_closed_on_count_mismatch_or_overflow() {
        assert!(dense_or_reshaped_layout("family.test", id(), &[4, 4], &[4, 3]).is_err());
        assert!(dense_or_reshaped_layout(
            "family.test",
            id(),
            &[u64::MAX, 2, 1],
            &[u64::MAX, 1, 2]
        )
        .is_err());
    }
}
