use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentInitializationReceipt,
        ComponentWeightSource, TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use std::sync::atomic::{AtomicUsize, Ordering};
mod schedule;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct EmptyConfig {}
struct JoinedFamily {
    id: ModelFamilyId,
    graph: StoredComposition,
}
impl ModelFamilyProvider for JoinedFamily {
    type Config = EmptyConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.joined.test")])
    }
    fn validate_config_identity(&self, raw: &Value, _: &EmptyConfig) -> Result<(), VNextError> {
        if raw.as_object().is_some_and(|m| m.is_empty()) {
            Ok(())
        } else {
            Err(invalid("unit fixture"))
        }
    }
    fn validated_external_metadata_id(
        &self,
        r: &Value,
        c: &EmptyConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(r, c)?;
        Ok(id("metadata.joined.test"))
    }
    fn parse_config(&self, r: &Value) -> Result<EmptyConfig, VNextError> {
        self.validate_config_identity(r, &EmptyConfig {})?;
        Ok(EmptyConfig {})
    }
    fn weights(&self, _: &EmptyConfig) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(self.graph.schema.clone()))
    }
    fn numerical_profiles(&self, _: &EmptyConfig) -> Result<FamilyNumericalProfiles, VNextError> {
        let p = self.graph.profile.clone();
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![p.clone()],
            vec![p.id],
        )
    }
    fn semantic_program(
        &self,
        _: &EmptyConfig,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if p != &self.graph.profile {
            return Err(invalid("profile differs"));
        }
        Ok(self.graph.program.clone())
    }
    fn semantic_metadata(&self, _: &EmptyConfig) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
struct Source {
    component: WeightComponentSpec,
    bytes: ManagedHostBytes,
    reads: AtomicUsize,
}
impl WeightComponentSource for Source {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if c != &self.component {
            return Err(invalid("source differs"));
        }
        self.reads.fetch_add(1, Ordering::Relaxed);
        WeightComponentPayload::from_managed_host(
            c,
            c.external_names.clone(),
            vec!["fixture.shared.f32".into()],
            c.dimensions.clone(),
            ElementType::F32,
            self.bytes.clone(),
        )
    }
}
#[test]
fn shared_physical_source_is_uploaded_once_and_both_logical_uses_execute_after_source_drop() {
    let graph = joined(shared_chain(), 8).unwrap();
    let options = ProgramPlanCompileOptions::new(graph.options.tensor_specs().clone()).unwrap();
    let account = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    );
    let mut bytes = account
        .allocate_zeroed(HostMemoryClass::WeightBacking, 8)
        .unwrap();
    for (chunk, value) in bytes.as_mut().chunks_exact_mut(4).zip([0.25f32, -0.5]) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    let source = Source {
        component: graph.schema.components[0].clone(),
        bytes: bytes.freeze(),
        reads: AtomicUsize::new(0),
    };
    let family = JoinedFamily {
        id: graph.program.family_id().clone(),
        graph,
    };
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(&serde_json::json!({}), &id("joined.test.f32.v1"))
        .unwrap();
    let c = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.shared_fragment"),
        1 << 20,
    )
    .unwrap();
    let d = c.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        c.registry(),
        c.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        c.weight_materializers(),
        &options,
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            c.runtime().clone(),
            account.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.shared.init"),
                request_id: id("request.shared.init"),
                transaction_id: id("transaction.shared.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let ComponentInitializationReceipt::Initialized { receipt } = session.initialization_receipt()
    else {
        panic!("Stored receipt")
    };
    assert_eq!(receipt.uploaded_component_count(), 1);
    assert_eq!(receipt.uploaded_bytes(), 8);
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    drop(source);
    let mut retained = None;
    for _ in 0..2 {
        let mut input = account
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
            .unwrap();
        for (chunk, value) in input.as_mut().chunks_exact_mut(4).zip([1f32, 2.]) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        let input = ManagedHostTensor::new(
            id("value.input"),
            ResolvedTensorSpec::new(
                vec![1, 2],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            input.freeze(),
        )
        .unwrap();
        let out = session
            .execute(
                id("run.shared.step"),
                id("request.shared.step"),
                vec![input],
            )
            .unwrap();
        assert_eq!(
            out[0].bytes().bytes(),
            [1.5f32, 1.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        retained = Some(out);
    }
    assert!(matches!(
        session.close().unwrap_or_else(|e| {
            let why = format!("{:?}", e.failure());
            std::mem::forget(e);
            panic!("close owner retained: {why}")
        }),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
    assert!(account.owns(retained.as_ref().unwrap()[0].bytes()));
    drop(retained);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
