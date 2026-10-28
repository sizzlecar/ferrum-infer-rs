use super::*;
use ferrum_runtime::*;
use half::bf16;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use std::sync::Arc;

fn fixture() -> (tempfile::TempDir, LinearAdapterSource) {
    let directory = tempfile::tempdir().unwrap();
    let a = [1_f32, 2., 3., 4.]
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let b = [1_f32, 0., 0., 1., 1., 1., -1., 1.]
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let file = directory.path().join("adapter.safetensors");
    serialize_to_file(
        [
            (
                "diffusion_model.fused.lora_A.weight",
                TensorView::new(Dtype::F32, vec![2, 2], &a).unwrap(),
            ),
            (
                "diffusion_model.fused.lora_B.weight",
                TensorView::new(Dtype::F32, vec![4, 2], &b).unwrap(),
            ),
        ],
        &None,
        &file,
    )
    .unwrap();
    (
        directory,
        LinearAdapterSource::Diffusers {
            file,
            base_model: "fixture/base".into(),
            scaling: LinearAdapterScaling::Unit,
        },
    )
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    )
}
fn bindings() -> Vec<LinearAdapterBinding> {
    vec![
        LinearAdapterBinding {
            module: "fused".into(),
            external_weight: "gate.weight".into(),
            base_dimensions: [2, 2],
            adapter_output_rows: 0..2,
        },
        LinearAdapterBinding {
            module: "fused".into(),
            external_weight: "up.weight".into(),
            base_dimensions: [2, 2],
            adapter_output_rows: 2..4,
        },
    ]
}
fn component(name: &str) -> WeightComponentSpec {
    WeightComponentSpec {
        id: WeightId::new(format!("component.{name}")).unwrap(),
        role: WeightComponentRole::Values,
        external_names: vec![format!("{name}.weight")],
        dimensions: vec![2, 2],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::Bf16,
        },
        required: true,
    }
}

#[test]
fn fused_rows_compose_real_matrices_and_release_all_temporary_claims() {
    let (_directory, source) = fixture();
    let imported = Arc::new(ImportedLinearAdapter::open(&source).unwrap());
    let prepared = Arc::new(PreparedLinearAdapter::new(imported, bindings(), 0.5).unwrap());
    let host = host();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let managed = prepared.manage(&host).unwrap();
    let mapped = host.usage().unwrap().committed_bytes();
    assert!(mapped >= managed.mapped_backing_bytes());
    let base = [1_f32; 4]
        .into_iter()
        .flat_map(|v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    for (name, expected) in [("gate", [1.5, 2., 2.5, 3.]), ("up", [3., 4., 2., 2.])] {
        let component = component(name);
        let budget = managed.workspace_bytes(&host, &component).unwrap()
            + managed.output_backing_bytes(&host, &component).unwrap();
        let scope = host
            .reserve_scope(&[(HostMemoryClass::WeightBacking, budget)])
            .unwrap();
        let output = managed
            .compose(&scope.account(), &component, &base)
            .unwrap();
        let actual = output
            .bytes()
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(
            host.usage().unwrap().allocated_bytes(),
            output.backing_bytes()
        );
        drop(output);
        drop(scope);
        assert_eq!(host.usage().unwrap().committed_bytes(), mapped);
    }
    assert!(base
        .chunks_exact(2)
        .all(|b| b == bf16::ONE.to_bits().to_le_bytes()));
    drop(managed);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn identity_uses_adapter_contents_and_strength_instead_of_mapping_addresses() {
    let (_directory, source) = fixture();
    let host = host();
    let make = |strength| {
        Arc::new(
            PreparedLinearAdapter::new(
                Arc::new(ImportedLinearAdapter::open(&source).unwrap()),
                bindings(),
                strength,
            )
            .unwrap(),
        )
        .manage(&host)
        .unwrap()
    };
    let first = make(0.5);
    let independently_mapped = make(0.5);
    assert_eq!(first.identity(), independently_mapped.identity());
    assert_ne!(first.identity(), make(1.0).identity());
    assert!(!make(0.0).affects(&component("gate")));
    assert!(first
        .compose(&self::host(), &component("gate"), &[0; 8])
        .is_err());
}

#[test]
fn import_and_binding_reject_variants_missing_pairs_and_incomplete_routes() {
    let (_directory, source) = fixture();
    let imported = Arc::new(ImportedLinearAdapter::open(&source).unwrap());
    assert!(PreparedLinearAdapter::new(imported.clone(), vec![bindings().remove(0)], 1.0).is_err());
    assert!(PreparedLinearAdapter::new(imported.clone(), bindings(), f32::NAN).is_err());
    let mut wrong = bindings();
    wrong[1].adapter_output_rows = 1..3;
    assert!(PreparedLinearAdapter::new(imported, wrong, 1.0).is_err());
    let config = serde_json::json!({"peft_type":"LORA", "r":2, "lora_alpha":2,
        "base_model_name_or_path":"fixture/base", "target_modules":["fused"]});
    assert!(super::config::PeftConfig::parse(&serde_json::to_vec(&config).unwrap()).is_ok());
    for key in ["use_dora", "use_rslora", "fan_in_fan_out"] {
        let mut bad = config.clone();
        bad[key] = true.into();
        assert!(super::config::PeftConfig::parse(&serde_json::to_vec(&bad).unwrap()).is_err());
    }
    let directory = tempfile::tempdir().unwrap();
    let bytes = [0u8; 4];
    let file = directory.path().join("missing.safetensors");
    serialize_to_file(
        [(
            "fused.lora_A.weight",
            TensorView::new(Dtype::F32, vec![1, 1], &bytes).unwrap(),
        )],
        &None,
        &file,
    )
    .unwrap();
    assert!(
        ImportedLinearAdapter::open(&LinearAdapterSource::Diffusers {
            file,
            base_model: "fixture/base".into(),
            scaling: LinearAdapterScaling::Unit,
        })
        .is_err()
    );
}
