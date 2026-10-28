use super::{compiled_provider, live_provider_catalog};
use ferrum_kernels::native_ops::{
    validate_compiled_native_operator_provider_catalog,
    validate_compiled_native_operator_provider_catalog_for_units as validate,
    CompiledNativeOperatorArtifact, CudaNativeBuildUnit,
};
use ferrum_types::{NativeOperatorBackend, NativeOperatorProviderCatalog};

fn artifact(
    unit: CudaNativeBuildUnit,
    catalog: &NativeOperatorProviderCatalog,
) -> CompiledNativeOperatorArtifact {
    let mut artifact = compiled_provider(catalog);
    artifact.operator = unit.artifact_operator().into();
    artifact.operator_abi_version = unit.required_operator_abi_version().unwrap_or("1").into();
    artifact.exports = unit
        .required_exports()
        .iter()
        .map(|name| (*name).into())
        .collect();
    artifact.operation_bindings[0].entrypoints = vec![unit.required_exports()[0].into()];
    artifact
}

fn tensor_catalog() -> NativeOperatorProviderCatalog {
    let mut catalog = live_provider_catalog();
    catalog.providers.truncate(1);
    catalog.providers[0].operation_id = "operation.dense_linear_fp8.bf16".into();
    catalog.providers[0].provider_id = "provider.cuda.dense_linear_fp8.bf16.cutlass".into();
    catalog
}

#[test]
fn explicit_scopes_accept_mixed_inventory_in_both_directions() {
    let text = live_provider_catalog();
    let tensor = tensor_catalog();
    let inventory = [
        artifact(CudaNativeBuildUnit::VllmPagedAttentionV2, &text),
        artifact(CudaNativeBuildUnit::CutlassFp8ScaledMm, &tensor),
    ];
    validate(
        &text,
        &inventory,
        &[CudaNativeBuildUnit::VllmPagedAttentionV2],
    )
    .unwrap();
    validate(
        &tensor,
        &inventory,
        &[CudaNativeBuildUnit::CutlassFp8ScaledMm],
    )
    .unwrap();
    // The original API still requires every compiled binding to be present.
    assert!(validate_compiled_native_operator_provider_catalog(&text, &inventory).is_err());
    assert!(validate_compiled_native_operator_provider_catalog(&tensor, &inventory).is_err());
    assert_eq!(inventory[0].operation_bindings.len(), 1);
    assert_eq!(inventory[1].operation_bindings.len(), 1);
}

#[test]
fn required_scope_does_not_hide_a_missing_live_provider() {
    let mut catalog = live_provider_catalog();
    let inventory = [artifact(
        CudaNativeBuildUnit::VllmPagedAttentionV2,
        &catalog,
    )];
    catalog.providers.remove(0);
    assert!(validate(
        &catalog,
        &inventory,
        &[CudaNativeBuildUnit::VllmPagedAttentionV2]
    )
    .unwrap_err()
    .contains("binds missing live provider"));
}

#[test]
fn a_selected_artifact_keeps_every_binding_and_shared_unit_export_requirement() {
    let text = live_provider_catalog();
    let tensor = tensor_catalog();
    let indexed = CudaNativeBuildUnit::VllmIndexedPagedAttentionV2;
    let units = [CudaNativeBuildUnit::VllmPagedAttentionV2, indexed];
    let mut selected = artifact(indexed, &text);
    validate(&text, &[selected.clone()], &units).unwrap();
    let indexed_export = indexed
        .required_exports()
        .iter()
        .find(|name| name.contains("indexed"))
        .unwrap();
    selected
        .exports
        .retain(|name| name.as_str() != *indexed_export);
    assert!(validate(&text, &[selected.clone()], &units)
        .unwrap_err()
        .contains("missing required export"));

    let mut selected = artifact(indexed, &text);
    let tensor_artifact = artifact(CudaNativeBuildUnit::CutlassFp8ScaledMm, &tensor);
    selected.exports.extend(tensor_artifact.exports);
    selected
        .operation_bindings
        .extend(tensor_artifact.operation_bindings);
    assert!(validate(&text, &[selected], &units)
        .unwrap_err()
        .contains("binds missing live provider"));
}

#[test]
fn required_units_reject_missing_ambiguous_or_incompatible_artifacts() {
    let catalog = tensor_catalog();
    let unit = CudaNativeBuildUnit::CutlassFp8ScaledMm;
    let valid = artifact(unit, &catalog);
    assert!(validate(&catalog, &[], &[unit])
        .unwrap_err()
        .contains("missing native build unit"));
    assert!(validate(&catalog, &[valid.clone(), valid.clone()], &[unit])
        .unwrap_err()
        .contains("ambiguous compiled artifacts"));
    let mut wrong_abi = valid.clone();
    wrong_abi.operator_abi_version = "9".into();
    assert!(validate(&catalog, &[wrong_abi], &[unit])
        .unwrap_err()
        .contains("operator ABI"));
    let mut missing_export = valid.clone();
    missing_export.exports.pop();
    assert!(validate(&catalog, &[missing_export], &[unit])
        .unwrap_err()
        .contains("missing required export"));
    let mut wrong_backend = valid;
    wrong_backend.backend = NativeOperatorBackend::Metal;
    assert!(validate(&catalog, &[wrong_backend], &[unit])
        .unwrap_err()
        .contains("requires a CUDA artifact"));
}

#[test]
fn selected_artifact_preserves_version_provenance_and_nonempty_binding_checks() {
    let catalog = tensor_catalog();
    let unit = CudaNativeBuildUnit::CutlassFp8ScaledMm;
    let valid = artifact(unit, &catalog);
    for provider_version in [false, true] {
        let mut incompatible = valid.clone();
        if provider_version {
            incompatible.operation_bindings[0].provider_version.major += 1;
        } else {
            incompatible.operation_bindings[0]
                .operation_contract_version
                .major += 1;
        }
        assert!(validate(&catalog, &[incompatible], &[unit])
            .unwrap_err()
            .contains("incompatible with the live versioned contract"));
    }
    let mut missing_provenance = valid.clone();
    missing_provenance.g03_catalog_sha256 = None;
    assert!(validate(&catalog, &[missing_provenance], &[unit])
        .unwrap_err()
        .contains("has no provider catalog provenance"));
    let mut invalid_schema = valid.clone();
    invalid_schema.schema_version += 1;
    assert!(validate(&catalog, &[invalid_schema], &[unit])
        .unwrap_err()
        .contains("uses schema"));
    let mut unbound = valid;
    unbound.operation_bindings.clear();
    assert!(validate(&catalog, &[unbound], &[unit])
        .unwrap_err()
        .contains("requires a live provider binding"));
    let text = live_provider_catalog();
    let text_unit = CudaNativeBuildUnit::VllmPagedAttentionV2;
    let text_artifact = artifact(text_unit, &text);
    let mut unbound_tensor = artifact(unit, &catalog);
    unbound_tensor.operation_bindings.clear();
    assert!(
        validate(&text, &[text_artifact, unbound_tensor], &[text_unit, unit])
            .unwrap_err()
            .contains("requires a live provider binding")
    );
}

#[test]
fn legacy_marlin_abi_only_scope_keeps_its_contract_without_a_fake_provider() {
    let catalog = live_provider_catalog();
    let tensor = tensor_catalog();
    let unit = CudaNativeBuildUnit::Marlin;
    let mut marlin = artifact(unit, &catalog);
    // This is the real ABI-2 package declaration, not a missing registration.
    marlin.operation_bindings.clear();
    let tensor_artifact = artifact(CudaNativeBuildUnit::CutlassFp8ScaledMm, &tensor);
    let inventory = [marlin.clone(), tensor_artifact.clone()];
    validate(&catalog, &inventory, &[unit]).unwrap();
    // The preexisting full-inventory API still rejects an unbound whole set.
    assert!(
        validate_compiled_native_operator_provider_catalog(&catalog, &[marlin.clone()])
            .unwrap_err()
            .contains("does not bind any live G03 operation/provider")
    );

    let mut wrong_abi = marlin.clone();
    wrong_abi.operator_abi_version = "1".into();
    assert!(validate(&catalog, &[wrong_abi], &[unit])
        .unwrap_err()
        .contains("operator ABI"));
    let mut missing_export = marlin.clone();
    missing_export.exports.pop();
    assert!(validate(&catalog, &[missing_export], &[unit])
        .unwrap_err()
        .contains("missing required export"));
    let mut missing_provenance = marlin.clone();
    missing_provenance.g03_catalog_sha256 = None;
    assert!(validate(&catalog, &[missing_provenance], &[unit])
        .unwrap_err()
        .contains("has no provider catalog provenance"));
    // A declared binding is never excused by the ABI-only capability.
    let mut missing_provider = marlin.clone();
    missing_provider.operation_bindings = tensor_artifact.operation_bindings;
    assert!(validate(&catalog, &[missing_provider], &[unit])
        .unwrap_err()
        .contains("binds missing live provider"));

    let mut unbound_tensor = artifact(CudaNativeBuildUnit::CutlassFp8ScaledMm, &tensor);
    unbound_tensor.operation_bindings.clear();
    assert!(validate(
        &tensor,
        &[marlin, unbound_tensor],
        &[unit, CudaNativeBuildUnit::CutlassFp8ScaledMm]
    )
    .unwrap_err()
    .contains("requires a live provider binding"));
}

#[test]
fn explicitly_empty_consumption_is_distinct_from_missing_required_artifacts() {
    let catalog = tensor_catalog();
    let unit = CudaNativeBuildUnit::CutlassFp8ScaledMm;
    let inventory = [artifact(unit, &catalog)];
    validate(&catalog, &inventory, &[]).unwrap();
    validate(&catalog, &[], &[]).unwrap();
    assert!(validate(&catalog, &[], &[unit]).is_err());
    // Empty scope still validates the composition's catalog.
    let mut invalid_catalog = catalog;
    invalid_catalog.schema_version += 1;
    assert!(validate(&invalid_catalog, &inventory, &[]).is_err());
}
