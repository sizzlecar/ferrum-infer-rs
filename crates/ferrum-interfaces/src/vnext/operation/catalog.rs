use ferrum_types::{
    NativeOperatorBackend, NativeOperatorContractVersion, NativeOperatorProviderCatalog,
    NativeOperatorProviderCatalogRow, NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use super::super::{
    DeviceDescriptor, WeightMaterializerDescriptor, WeightMaterializerId, MAX_WEIGHT_MATERIALIZERS,
};
use super::{
    invalid_operation, operation_error_for_node, ContractVersion, EngineProviderDescriptor, NodeId,
    OperationDescriptor, OperationId, OperationProviderDescriptor, OracleSpec,
    ProviderCompatibilityRejectReason, ProviderCompatibilityRejection, ProviderCompatibilityReport,
    ProviderCompatibilityRequest, ProviderId, VNextError, MAX_ENGINE_PROVIDER_ROWS,
    MAX_OPERATION_CATALOG_ROWS, MAX_OPERATION_PROVIDER_ROWS, MAX_REFERENCE_ORACLE_DEPTH,
};

/// Deterministically ordered provider capabilities consumed once by planning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CapabilityCatalog {
    device: DeviceDescriptor,
    operations: BTreeMap<OperationId, OperationDescriptor>,
    providers: BTreeMap<OperationId, Vec<OperationProviderDescriptor>>,
    engine_providers: BTreeMap<ProviderId, EngineProviderDescriptor>,
    weight_materializers: BTreeMap<WeightMaterializerId, WeightMaterializerDescriptor>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapabilityCatalogWire {
    device: DeviceDescriptor,
    operations: BTreeMap<OperationId, OperationDescriptor>,
    providers: BTreeMap<OperationId, Vec<OperationProviderDescriptor>>,
    engine_providers: BTreeMap<ProviderId, EngineProviderDescriptor>,
    #[serde(default = "identity_weight_materializer_descriptors")]
    weight_materializers: BTreeMap<WeightMaterializerId, WeightMaterializerDescriptor>,
}

impl<'de> Deserialize<'de> for CapabilityCatalog {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CapabilityCatalogWire::deserialize(deserializer)?;
        Self::from_maps(
            wire.device,
            wire.operations,
            wire.providers,
            wire.engine_providers,
            wire.weight_materializers,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl CapabilityCatalog {
    pub fn new(
        device: DeviceDescriptor,
        operations: Vec<OperationDescriptor>,
        providers: BTreeMap<OperationId, Vec<OperationProviderDescriptor>>,
        engine_providers: Vec<EngineProviderDescriptor>,
    ) -> Result<Self, VNextError> {
        let mut operation_map = BTreeMap::new();
        for operation in operations {
            let operation_id = operation.id.clone();
            if operation_map
                .insert(operation_id.clone(), operation)
                .is_some()
            {
                return Err(invalid_operation(format!(
                    "duplicate operation descriptor `{operation_id}`"
                )));
            }
        }
        let mut engine_map = BTreeMap::new();
        for engine in engine_providers {
            let provider_id = engine.provider_id().clone();
            if engine_map.insert(provider_id.clone(), engine).is_some() {
                return Err(invalid_operation(format!(
                    "duplicate engine provider `{provider_id}`"
                )));
            }
        }
        let weight_materializers = identity_weight_materializer_descriptors();
        Self::from_maps(
            device,
            operation_map,
            providers,
            engine_map,
            weight_materializers,
        )
    }

    fn from_maps(
        device: DeviceDescriptor,
        operations: BTreeMap<OperationId, OperationDescriptor>,
        mut providers: BTreeMap<OperationId, Vec<OperationProviderDescriptor>>,
        engine_providers: BTreeMap<ProviderId, EngineProviderDescriptor>,
        weight_materializers: BTreeMap<WeightMaterializerId, WeightMaterializerDescriptor>,
    ) -> Result<Self, VNextError> {
        device.validate()?;
        validate_weight_materializer_descriptors(&device, &weight_materializers)?;
        let provider_row_count = providers.values().try_fold(0_usize, |total, entries| {
            total.checked_add(entries.len()).ok_or_else(|| {
                invalid_operation("capability catalog provider row count overflows usize")
            })
        })?;
        if operations.is_empty()
            || providers.is_empty()
            || engine_providers.is_empty()
            || operations.len() > MAX_OPERATION_CATALOG_ROWS
            || provider_row_count > MAX_OPERATION_PROVIDER_ROWS
            || engine_providers.len() > MAX_ENGINE_PROVIDER_ROWS
        {
            return Err(invalid_operation(
                "capability catalog is empty or exceeds its operation/provider/engine row budget",
            ));
        }
        if operations.keys().collect::<BTreeSet<_>>() != providers.keys().collect::<BTreeSet<_>>() {
            return Err(invalid_operation(
                "capability catalog operation and provider rows do not match",
            ));
        }
        for (operation_id, operation) in &operations {
            if operation_id != &operation.id {
                return Err(invalid_operation(format!(
                    "operation descriptor `{}` is stored under `{operation_id}`",
                    operation.id
                )));
            }
            operation.validate()?;
            if !operation
                .provider
                .required_capabilities
                .is_subset(&device.capabilities)
            {
                return Err(VNextError::UnsupportedOperation {
                    node_id: None,
                    operation_id: operation_id.to_string(),
                    device_id: device.id.to_string(),
                    reason: "device does not advertise the operation's required capabilities"
                        .to_owned(),
                });
            }
        }
        validate_reference_oracle_graph(&operations)?;
        for (operation_id, entries) in &mut providers {
            if entries.is_empty() {
                return Err(VNextError::UnsupportedOperation {
                    node_id: None,
                    operation_id: operation_id.to_string(),
                    device_id: device.id.to_string(),
                    reason: "provider row is empty".to_owned(),
                });
            }
            let operation =
                operations
                    .get(operation_id)
                    .ok_or_else(|| VNextError::UnsupportedOperation {
                        node_id: None,
                        operation_id: operation_id.to_string(),
                        device_id: device.id.to_string(),
                        reason: "provider row has no operation descriptor".to_owned(),
                    })?;
            let operation_fingerprint = operation.fingerprint()?;
            for entry in entries.iter() {
                if entry.operation_id() != operation_id
                    || entry.operation_fingerprint() != operation_fingerprint
                {
                    return Err(VNextError::UnsupportedOperation {
                        node_id: None,
                        operation_id: operation_id.to_string(),
                        device_id: device.id.to_string(),
                        reason: format!(
                            "provider `{}` is bound to a different operation descriptor",
                            entry.provider_id()
                        ),
                    });
                }
                if entry.device_id() != &device.id {
                    return Err(VNextError::UnsupportedOperation {
                        node_id: None,
                        operation_id: operation_id.to_string(),
                        device_id: device.id.to_string(),
                        reason: format!(
                            "provider `{}` belongs to device `{}`",
                            entry.provider_id(),
                            entry.device_id()
                        ),
                    });
                }
                if !entry.version().satisfies(operation.version)
                    || !entry
                        .version()
                        .satisfies(operation.provider.minimum_version)
                {
                    return Err(VNextError::UnsupportedOperation {
                        node_id: None,
                        operation_id: operation_id.to_string(),
                        device_id: device.id.to_string(),
                        reason: format!(
                            "provider `{}` does not satisfy the operation version",
                            entry.provider_id()
                        ),
                    });
                }
                if !entry.capabilities().is_subset(&device.capabilities)
                    || !operation
                        .provider
                        .required_capabilities
                        .is_subset(entry.capabilities())
                {
                    return Err(VNextError::UnsupportedOperation {
                        node_id: None,
                        operation_id: operation_id.to_string(),
                        device_id: device.id.to_string(),
                        reason: format!(
                            "provider `{}` capabilities are incompatible with the device or operation",
                            entry.provider_id()
                        ),
                    });
                }
            }
            entries.sort_by(|left, right| {
                left.provider_id()
                    .cmp(right.provider_id())
                    .then(left.version().cmp(&right.version()))
            });
            let mut seen = BTreeSet::new();
            if entries
                .iter()
                .any(|entry| !seen.insert(entry.provider_id().clone()))
            {
                return Err(VNextError::UnsupportedOperation {
                    node_id: None,
                    operation_id: operation_id.to_string(),
                    device_id: device.id.to_string(),
                    reason: "duplicate provider identity".to_owned(),
                });
            }
        }
        for (provider_id, engine) in &engine_providers {
            if provider_id != engine.provider_id()
                || engine.device_id() != &device.id
                || !engine.capabilities().is_subset(&device.capabilities)
            {
                return Err(invalid_operation(format!(
                    "engine provider `{provider_id}` identity, device, or capabilities are invalid"
                )));
            }
        }
        Ok(Self {
            device,
            operations,
            providers,
            engine_providers,
            weight_materializers,
        })
    }

    pub(crate) fn with_weight_materializer_descriptors(
        mut self,
        weight_materializers: BTreeMap<WeightMaterializerId, WeightMaterializerDescriptor>,
    ) -> Result<Self, VNextError> {
        validate_weight_materializer_descriptors(&self.device, &weight_materializers)?;
        self.weight_materializers = weight_materializers;
        Ok(self)
    }

    pub fn device(&self) -> &DeviceDescriptor {
        &self.device
    }

    pub fn providers_for(
        &self,
        operation_id: &OperationId,
    ) -> Result<&[OperationProviderDescriptor], VNextError> {
        self.providers
            .get(operation_id)
            .map(Vec::as_slice)
            .ok_or_else(|| VNextError::UnsupportedOperation {
                node_id: None,
                operation_id: operation_id.to_string(),
                device_id: self.device.id.to_string(),
                reason: "no provider is registered".to_owned(),
            })
    }

    /// Resolves providers with the requesting plan node attached to failures.
    pub fn providers_for_node(
        &self,
        node_id: &NodeId,
        operation_id: &OperationId,
    ) -> Result<&[OperationProviderDescriptor], VNextError> {
        self.providers_for(operation_id)
            .map_err(|error| operation_error_for_node(error, node_id))
    }

    pub fn operation(
        &self,
        operation_id: &OperationId,
    ) -> Result<&OperationDescriptor, VNextError> {
        self.operations
            .get(operation_id)
            .ok_or_else(|| VNextError::UnsupportedOperation {
                node_id: None,
                operation_id: operation_id.to_string(),
                device_id: self.device.id.to_string(),
                reason: "operation descriptor is not registered".to_owned(),
            })
    }

    /// Resolves an operation with the requesting plan node attached to failures.
    pub fn operation_for_node(
        &self,
        node_id: &NodeId,
        operation_id: &OperationId,
    ) -> Result<&OperationDescriptor, VNextError> {
        self.operation(operation_id)
            .map_err(|error| operation_error_for_node(error, node_id))
    }

    pub fn provider_compatibility(
        &self,
        mut request: ProviderCompatibilityRequest,
    ) -> Result<ProviderCompatibilityReport, VNextError> {
        let operation = self.operation(request.operation_id())?;
        request
            .extend_required_capabilities(operation.provider.required_capabilities.iter().cloned());
        let mut compatible_provider_ids = Vec::new();
        let mut rejected = Vec::new();
        for provider in self.providers_for(request.operation_id())? {
            let mut reasons = Vec::new();
            if !operation.version.satisfies(request.required_version()) {
                reasons.push(
                    ProviderCompatibilityRejectReason::OperationVersionMismatch {
                        required: request.required_version(),
                        available: operation.version,
                    },
                );
            }
            if !provider.version().satisfies(request.required_version()) {
                reasons.push(ProviderCompatibilityRejectReason::ProviderVersionMismatch {
                    required: request.required_version(),
                    available: provider.version(),
                });
            }
            let missing_capabilities = request
                .required_capabilities()
                .difference(provider.capabilities())
                .cloned()
                .collect::<BTreeSet<_>>();
            if !missing_capabilities.is_empty() {
                reasons.push(ProviderCompatibilityRejectReason::MissingCapabilities {
                    capabilities: missing_capabilities,
                });
            }
            let missing_weight_formats = request
                .required_weight_formats()
                .difference(provider.accepted_weight_formats())
                .cloned()
                .collect::<BTreeSet<_>>();
            if !missing_weight_formats.is_empty() {
                reasons.push(
                    ProviderCompatibilityRejectReason::UnsupportedWeightFormats {
                        formats: missing_weight_formats,
                    },
                );
            }
            let missing_quantization_formats = request
                .required_quantization_formats()
                .difference(provider.accepted_quantization_formats())
                .cloned()
                .collect::<BTreeSet<_>>();
            if !missing_quantization_formats.is_empty() {
                reasons.push(
                    ProviderCompatibilityRejectReason::UnsupportedQuantizationFormats {
                        formats: missing_quantization_formats,
                    },
                );
            }
            if !request
                .execution_determinism()
                .accepts(provider.execution_semantics())
            {
                reasons.push(
                    ProviderCompatibilityRejectReason::InsufficientExecutionDeterminism {
                        required: request.execution_determinism(),
                        available: provider.execution_semantics(),
                    },
                );
            }
            if reasons.is_empty() {
                compatible_provider_ids.push(provider.provider_id().clone());
            } else {
                rejected.push(ProviderCompatibilityRejection {
                    provider_id: provider.provider_id().clone(),
                    reasons,
                });
            }
        }
        ProviderCompatibilityReport::from_classification(request, compatible_provider_ids, rejected)
    }

    pub fn operations(&self) -> &BTreeMap<OperationId, OperationDescriptor> {
        &self.operations
    }

    pub fn providers(&self) -> &BTreeMap<OperationId, Vec<OperationProviderDescriptor>> {
        &self.providers
    }

    pub fn engine_providers(&self) -> &BTreeMap<ProviderId, EngineProviderDescriptor> {
        &self.engine_providers
    }

    pub fn weight_materializers(
        &self,
    ) -> &BTreeMap<WeightMaterializerId, WeightMaterializerDescriptor> {
        &self.weight_materializers
    }

    pub fn weight_materializer(
        &self,
        materializer_id: &WeightMaterializerId,
    ) -> Result<&WeightMaterializerDescriptor, VNextError> {
        self.weight_materializers
            .get(materializer_id)
            .ok_or_else(|| {
                invalid_operation(format!(
                    "weight materializer `{materializer_id}` is absent from the capability catalog"
                ))
            })
    }

    pub fn engine_provider(
        &self,
        provider_id: &ProviderId,
        required_version: ContractVersion,
    ) -> Result<&EngineProviderDescriptor, VNextError> {
        let provider = self.engine_providers.get(provider_id).ok_or_else(|| {
            invalid_operation(format!("engine provider `{provider_id}` is not registered"))
        })?;
        if !provider.contract_version().satisfies(required_version) {
            return Err(invalid_operation(format!(
                "engine provider `{provider_id}` version {} does not satisfy {required_version}",
                provider.contract_version()
            )));
        }
        Ok(provider)
    }

    /// Projects the exact live registry into the device-independent identity
    /// catalog consumed by native artifact packaging. Physical entrypoint
    /// ownership remains in the native package definition.
    pub fn native_operator_provider_catalog(
        &self,
        backend: NativeOperatorBackend,
    ) -> Result<NativeOperatorProviderCatalog, VNextError> {
        let mut providers = Vec::new();
        for (operation_id, descriptors) in &self.providers {
            let operation = self.operations.get(operation_id).ok_or_else(|| {
                invalid_operation(
                    "capability catalog provider row lacks its operation while exporting native identities",
                )
            })?;
            let operation_fingerprint = operation.fingerprint()?;
            for provider in descriptors {
                providers.push(NativeOperatorProviderCatalogRow {
                    operation_id: operation_id.to_string(),
                    operation_contract_version: NativeOperatorContractVersion::new(
                        operation.version.major,
                        operation.version.minor,
                    ),
                    operation_fingerprint: operation_fingerprint.clone(),
                    provider_id: provider.provider_id().to_string(),
                    provider_version: NativeOperatorContractVersion::new(
                        provider.version().major,
                        provider.version().minor,
                    ),
                    provider_implementation_fingerprint: provider
                        .provider_implementation_fingerprint()
                        .to_owned(),
                });
            }
        }
        providers.sort();
        let catalog = NativeOperatorProviderCatalog {
            schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
            backend,
            providers,
        };
        catalog.validate().map_err(invalid_operation)?;
        Ok(catalog)
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        let bytes = serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize capability catalog",
            message: error.to_string(),
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

fn validate_weight_materializer_descriptors(
    device: &DeviceDescriptor,
    descriptors: &BTreeMap<WeightMaterializerId, WeightMaterializerDescriptor>,
) -> Result<(), VNextError> {
    if descriptors.is_empty() || descriptors.len() > MAX_WEIGHT_MATERIALIZERS {
        return Err(invalid_operation(
            "capability catalog weight materializers are empty or exceed their row budget",
        ));
    }
    for (id, descriptor) in descriptors {
        if id != descriptor.id() {
            return Err(invalid_operation(format!(
                "weight materializer `{}` is stored under `{id}`",
                descriptor.id()
            )));
        }
        descriptor.validate_for_device(device)?;
    }
    let identity = WeightMaterializerDescriptor::identity()?;
    if descriptors.get(identity.id()) != Some(&identity) {
        return Err(invalid_operation(
            "capability catalog lacks the canonical identity weight materializer",
        ));
    }
    Ok(())
}

fn identity_weight_materializer_descriptors(
) -> BTreeMap<WeightMaterializerId, WeightMaterializerDescriptor> {
    let identity = WeightMaterializerDescriptor::identity()
        .expect("the built-in identity weight materializer descriptor is valid");
    BTreeMap::from([(identity.id().clone(), identity)])
}

fn validate_reference_oracle_graph(
    operations: &BTreeMap<OperationId, OperationDescriptor>,
) -> Result<(), VNextError> {
    for (operation_id, operation) in operations {
        if let OracleSpec::ReferenceOperation {
            operation_id: reference_id,
            version,
        } = &operation.oracle
        {
            let reference = operations.get(reference_id).ok_or_else(|| {
                invalid_operation(format!(
                    "operation `{operation_id}` references missing oracle `{reference_id}`"
                ))
            })?;
            if !reference.version.satisfies(*version) {
                return Err(invalid_operation(format!(
                    "operation `{operation_id}` oracle `{reference_id}` version {} does not satisfy {version}",
                    reference.version
                )));
            }
            if operation.inputs != reference.inputs
                || operation.outputs != reference.outputs
                || operation.attributes != reference.attributes
            {
                return Err(invalid_operation(format!(
                    "operation `{operation_id}` oracle `{reference_id}` has an incompatible input/output/attribute contract"
                )));
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum VisitState {
        Visiting,
        Visited,
    }

    let mut states = BTreeMap::<OperationId, VisitState>::new();
    for root in operations.keys() {
        if states.get(root) == Some(&VisitState::Visited) {
            continue;
        }
        let mut path = Vec::<OperationId>::new();
        let mut current = root.clone();
        loop {
            match states.get(&current) {
                Some(VisitState::Visited) => break,
                Some(VisitState::Visiting) => {
                    return Err(invalid_operation(format!(
                        "reference-oracle graph contains a cycle at `{current}`"
                    )));
                }
                None => {}
            }
            if path.len() >= MAX_REFERENCE_ORACLE_DEPTH {
                return Err(invalid_operation(format!(
                    "reference-oracle chain from `{root}` exceeds depth {MAX_REFERENCE_ORACLE_DEPTH}"
                )));
            }
            states.insert(current.clone(), VisitState::Visiting);
            path.push(current.clone());
            let Some(OperationDescriptor {
                oracle:
                    OracleSpec::ReferenceOperation {
                        operation_id: reference_id,
                        ..
                    },
                ..
            }) = operations.get(&current)
            else {
                break;
            };
            current = reference_id.clone();
        }
        for operation_id in path {
            states.insert(operation_id, VisitState::Visited);
        }
    }
    Ok(())
}
