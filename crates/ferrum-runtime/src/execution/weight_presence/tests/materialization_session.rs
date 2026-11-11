//! A validated installation view retains the original per-group contracts.
use super::*;
use crate::test_support::vnext_core_contract::{catalog, plan_fixture, sha, OrderedSchemaFamily};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn stored_family(reverse_sources: bool) -> PreparedModelFamily {
    TypedFamilyRegistration::new(OrderedSchemaFamily {
        reverse: false,
        reverse_sources,
    })
    .prepare_with_profile(
        &json!({"width": 4}),
        &NumericalProfileId::new("fixture.f32").unwrap(),
    )
    .unwrap()
}

#[derive(Default)]
struct Source {
    called: AtomicBool,
}

fn bytes(component: &WeightComponentSpec) -> Vec<u8> {
    // Preserve low F32 bits and negative zero through the identity source.
    let words = [0x8000_0000u32, 0x3f80_0001, 0xbf00_0001, 0x4000_0001];
    (0..usize::try_from(component.physical_bytes().unwrap()).unwrap() / 4)
        .flat_map(|i| words[i % words.len()].to_le_bytes())
        .collect()
}

fn payload(component: &WeightComponentSpec) -> Result<WeightComponentPayload<'static>, VNextError> {
    WeightComponentPayload::from_ordered_sources(
        component,
        component.external_names.clone(),
        vec!["fixture.safetensors".to_owned(); component.external_names.len()],
        component.dimensions.clone(),
        component.physical_element_type(),
        bytes(component),
    )
}

impl WeightComponentSource for Source {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.called.store(true, Ordering::SeqCst);
        payload(component)
    }
}

#[test]
fn validated_materialization_session_consumes_distinct_original_source_groups() {
    let family = stored_family(false);
    let authority = TrustedExecutionWeights::for_family(&family).unwrap();
    let session = authority.materialization_session(&family).unwrap();
    let source = Source::default();
    let components = &family.stored_weight_schema().unwrap().components;
    // The shared family has separate physical components and ordered external
    // sources, rather than repeatedly exercising a single component's lookup.
    assert!(components.windows(2).any(|pair| pair[0].id != pair[1].id));
    for component in components.iter().rev() {
        let loaded = session
            .materialize_components(&source, &[component], None)
            .unwrap();
        assert_eq!(loaded[0].component_id(), &component.id);
        assert_eq!(loaded[0].external_names(), component.external_names);
        assert_eq!(loaded[0].bytes(), bytes(component));
    }
}

#[test]
fn range_authority_requires_the_original_live_identity_schema() {
    let family = stored_family(false);
    let authority = TrustedExecutionWeights::for_family(&family).unwrap();
    let session = authority.materialization_session(&family).unwrap();
    for component in &family.stored_weight_schema().unwrap().components {
        session
            .validate_identity_range_component(component)
            .unwrap();
        let mut changed = component.clone();
        changed.dimensions[0] += 1;
        assert!(session.validate_identity_range_component(&changed).is_err());
        changed = component.clone();
        changed.external_names.reverse();
        assert!(session.validate_identity_range_component(&changed).is_err());
    }
}

#[test]
fn validated_materialization_session_rejects_foreign_family_and_absent_weights() {
    let original = stored_family(false);
    let other = stored_family(true);
    assert_eq!(original.family_id(), other.family_id());
    assert_ne!(original.weights(), other.weights());
    let authority = TrustedExecutionWeights::for_family(&original).unwrap();
    assert!(authority.materialization_session(&other).is_err());

    let none = prepared(4, "node.main", "fixture.f32");
    let absent = TrustedExecutionWeights::for_family(&none).unwrap();
    // A None proof remains valid for weightless execution but cannot mint a
    // stored materialization view, including with its own correct family.
    absent.validate_against_family(&none).unwrap();
    assert!(absent.materialization_session(&none).is_err());
    assert!(absent.materialization_session(&original).is_err());
}

#[test]
fn validated_materialization_session_checks_changed_component_before_source_access() {
    let family = stored_family(false);
    let authority = TrustedExecutionWeights::for_family(&family).unwrap();
    let session = authority.materialization_session(&family).unwrap();
    let source = Source::default();
    let original = &family.stored_weight_schema().unwrap().components[0];
    for mutate in [0, 1, 2] {
        let mut changed = original.clone();
        match mutate {
            0 => changed.external_names.reverse(),
            1 => changed.dimensions[0] += 1,
            _ => {
                changed.encoding = WeightEncoding::Dense {
                    element_type: ElementType::F16,
                }
            }
        }
        assert!(session
            .materialize_components(&source, &[&changed], None)
            .is_err());
        assert!(!source.called.load(Ordering::SeqCst));
    }
    assert!(session.materialize_components(&source, &[], None).is_err());
    let mut foreign = original.clone();
    foreign.id = id("weight.component.foreign");
    assert!(session
        .materialize_components(&source, &[&foreign], None)
        .is_err());
    assert!(!source.called.load(Ordering::SeqCst));
}

struct WrongSource {
    wrong_shape: bool,
}
impl WeightComponentSource for WrongSource {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let mut wrong = component.clone();
        if self.wrong_shape {
            wrong.dimensions[0] += 1;
        } else {
            wrong.encoding = WeightEncoding::Dense {
                element_type: ElementType::F16,
            };
        }
        // This payload satisfies a different, internally consistent source
        // schema. The trusted materializer must still reject it for this plan.
        let count = usize::try_from(wrong.physical_bytes()?).unwrap();
        WeightComponentPayload::from_ordered_sources(
            &wrong,
            wrong.external_names.clone(),
            vec!["fixture.safetensors".to_owned(); wrong.external_names.len()],
            wrong.dimensions.clone(),
            wrong.physical_element_type(),
            vec![0; count],
        )
    }
}

#[test]
fn validated_materialization_session_rejects_a_source_payload_for_another_schema() {
    let family = stored_family(false);
    let authority = TrustedExecutionWeights::for_family(&family).unwrap();
    let session = authority.materialization_session(&family).unwrap();
    let component = &family.stored_weight_schema().unwrap().components[0];
    for wrong_shape in [false, true] {
        let error = session
            .materialize_components(&WrongSource { wrong_shape }, &[component], None)
            .err()
            .expect("foreign source output must fail");
        assert!(error.to_string().contains("invalid or reordered payload"));
    }
    // A prior rejected source does not poison the immutable view.
    let source = Source::default();
    assert_eq!(
        session
            .materialize_components(&source, &[component], None)
            .unwrap()[0]
            .bytes(),
        bytes(component)
    );
}

struct FailingSource;
impl WeightComponentSource for FailingSource {
    fn component<'a>(
        &'a self,
        _: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        Err(VNextError::InvalidExecutionPlan {
            reason: "original source read failed".into(),
        })
    }
}

#[test]
fn validated_materialization_session_preserves_original_source_failure() {
    let family = stored_family(false);
    let authority = TrustedExecutionWeights::for_family(&family).unwrap();
    let session = authority.materialization_session(&family).unwrap();
    let component = &family.stored_weight_schema().unwrap().components[0];
    let error = session
        .materialize_components(&FailingSource, &[component], None)
        .err()
        .expect("source error must remain an error");
    assert!(
        matches!(error, VNextError::InvalidExecutionPlan { reason } if reason == "original source read failed")
    );
}

struct SwitchingMaterializer {
    original: WeightMaterializerDescriptor,
    changed: WeightMaterializerDescriptor,
    switched: Arc<AtomicBool>,
    invoked: Arc<AtomicBool>,
}
impl WeightMaterializer for SwitchingMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        if self.switched.load(Ordering::SeqCst) {
            &self.changed
        } else {
            &self.original
        }
    }
    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        Ok(family.stored_weight_schema()?.clone())
    }
    fn materialize_component<'a>(
        &self,
        source: &'a dyn WeightComponentSource,
        sources: &[&WeightComponentSpec],
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.invoked.store(true, Ordering::SeqCst);
        assert_eq!(sources, &[component]);
        source.component(component)
    }
}

#[test]
fn validated_materialization_session_still_checks_live_materializer_descriptor() {
    let family = stored_family(false);
    let descriptor = |fingerprint| {
        WeightMaterializerDescriptor::new(
            id("weight-materializer.session-fixture"),
            ContractVersion::new(1, 0),
            fingerprint,
            WeightMaterializationFidelity::Exact,
            BTreeSet::new(),
        )
        .unwrap()
    };
    let original = descriptor(sha('a'));
    let changed = descriptor(sha('b'));
    let switched = Arc::new(AtomicBool::new(false));
    let invoked = Arc::new(AtomicBool::new(false));
    let registry = WeightMaterializerRegistry::new(vec![Box::new(SwitchingMaterializer {
        original: original.clone(),
        changed,
        switched: Arc::clone(&switched),
        invoked: Arc::clone(&invoked),
    })])
    .unwrap();
    let catalog = registry.augment_catalog(catalog()).unwrap();
    let authority = TrustedExecutionWeights::from(
        registry
            .select_exact(&family, &catalog, original.id())
            .unwrap(),
    );
    let session = authority.materialization_session(&family).unwrap();
    let source = Source::default();
    let component = &family.stored_weight_schema().unwrap().components[0];
    session
        .materialize_components(&source, &[component], None)
        .unwrap();
    assert!(invoked.load(Ordering::SeqCst));
    invoked.store(false, Ordering::SeqCst);
    switched.store(true, Ordering::SeqCst);
    let error = session
        .materialize_components(&source, &[component], None)
        .err()
        .expect("changed descriptor must fail before invoking materializer");
    assert!(error.to_string().contains("runtime authority differs"));
    assert!(!invoked.load(Ordering::SeqCst));
}

#[test]
fn validated_materialization_session_is_consumed_by_an_actual_execution_plan() {
    let fixture = plan_fixture(0);
    let session = fixture
        .plan
        .weight_materialization_session(&fixture.family)
        .unwrap();
    let source = Source::default();
    let component = &fixture.family.stored_weight_schema().unwrap().components[0];
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            // Accounts require a positive total; every allocation class still
            // has zero capacity, so this source cannot acquire Host backing.
            1,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let loaded = session
        .materialize_components(&source, &[component], Some(&host))
        .unwrap();
    assert_eq!(loaded[0].bytes(), bytes(component));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let foreign = stored_family(false);
    assert!(fixture
        .plan
        .weight_materialization_session(&foreign)
        .is_err());
}
