use super::*;

#[test]
fn original_euler_preserves_negative_head_sign_and_terminal_step() {
    // Returned -2 means positive model velocity +2. Half a unit of sigma
    // therefore raises x by 1; the final step remains an Euler evaluation.
    assert_eq!(
        euler(1.25, -2.0, 1.0, 0.5).unwrap().to_bits(),
        2.25f32.to_bits()
    );
    assert_eq!(
        euler(2.25, -1.0, 0.5, 0.0).unwrap().to_bits(),
        2.75f32.to_bits()
    );
    assert_eq!(
        euler(-1.25, 2.0, 1.0, 0.5).unwrap().to_bits(),
        (-2.25f32).to_bits()
    );
    assert_eq!(
        euler(-0.0, 0.0, 1.0, 0.0).unwrap().to_bits(),
        (-0.0f32).to_bits()
    );
    for (x, head, sigma, next) in [
        (f32::NAN, 0., 1., 0.),
        (0., f32::INFINITY, 1., 0.),
        (0., 0., 0., 0.),
        (0., 0., 1., 1.),
        (0., 0., 1., -1.),
    ] {
        assert!(euler(x, head, sigma, next).is_err());
    }
}

#[test]
fn patch_mapping_keeps_channel_and_patch_axes_and_rejects_partial_patches() {
    let l = ImageLayout::new([1, 2, 1, 2, 4], [1, 2, 2]).unwrap();
    let mapped: Vec<_> = (0..l.elements).map(|i| l.nchw(i)).collect();
    assert_eq!(
        mapped,
        [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]
    );
    assert!(ImageLayout::new([1, 2, 1, 3, 4], [1, 2, 2]).is_err());
    assert!(ImageLayout::new([2, 2, 1, 2, 4], [1, 2, 2]).is_err());
    assert!(validate_options(Trajectory::OriginalEuler, InputBoundary::Mod1, false).is_err());
    assert!(validate_options(Trajectory::OriginalEuler, InputBoundary::Packed, false).is_err());
    assert!(validate_options(Trajectory::OriginalEuler, InputBoundary::Prepared, true).is_err());
    validate_options(Trajectory::OriginalEuler, InputBoundary::Prepared, false).unwrap();
    validate_options(Trajectory::Single, InputBoundary::Mod1, true).unwrap();
}

fn fixture(path: &std::path::Path, corrupt_step: bool, scaled_final: bool) {
    use safetensors::{
        tensor::{serialize_to_file, TensorView},
        Dtype,
    };
    let floats = |values: Vec<f32>| {
        values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let mut tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = vec![
        (
            "trajectory_sigmas".into(),
            Dtype::F32,
            vec![9],
            floats((0..9).map(|i| 1. - i as f32 / 8.).collect()),
        ),
        (
            "time_basis".into(),
            Dtype::F32,
            vec![1, 2],
            floats(vec![1., 2.]),
        ),
        (
            "trajectory_time_basis".into(),
            Dtype::F32,
            vec![8, 1, 2],
            floats([1., 2.].repeat(8)),
        ),
    ];
    for (name, sign, shape) in [
        ("image", 1f32, vec![1, 1, 1, 2, 2]),
        ("audio", -1., vec![1, 1, 2, 2]),
    ] {
        let states: Vec<f32> = (0..9)
            .flat_map(|i| [sign * (1. + i as f32 / 4.); 4])
            .collect();
        let mut bytes = floats(states);
        if corrupt_step && name == "image" {
            bytes[4 * 4..4 * 4 + 4].copy_from_slice(&1.5f32.to_le_bytes());
        }
        tensors.push((
            format!("trajectory_states_{name}"),
            Dtype::F32,
            [vec![9], shape.clone()].concat(),
            bytes,
        ));
        let bits = if sign > 0. { 0xc000u16 } else { 0x4000u16 }; // -2 / +2, independent BF16 bits.
        let returned: Vec<u8> = (0..4).flat_map(|_| bits.to_le_bytes()).collect();
        tensors.push((
            format!("returned_{name}"),
            Dtype::BF16,
            shape.clone(),
            returned.clone(),
        ));
        tensors.push((
            format!("trajectory_returned_{name}"),
            Dtype::BF16,
            [vec![8], shape.clone()].concat(),
            returned.repeat(8),
        ));
        tensors.push((
            format!("trajectory_output_{name}"),
            Dtype::F32,
            shape,
            floats(vec![sign * if scaled_final { 6. } else { 3. }; 4]),
        ));
        tensors.push((
            format!("{name}_patches"),
            Dtype::F32,
            if name == "image" {
                vec![1, 4]
            } else {
                vec![4, 1]
            },
            floats(vec![sign; 4]),
        ));
    }
    let views = tensors.iter().map(|(name, dtype, shape, bytes)| {
        (
            name.as_str(),
            TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
        )
    });
    serialize_to_file(views, &None, path).unwrap();
}

fn capture() -> data::Capture {
    serde_json::from_value(json!({"schema":"ferrum.ph3.dit_input.v1","phase":"refined",
        "target_shape":[1,1,1,2,2],"audio_shape":[1,1,2,2],"reference_shape":null,
        "text_tags":[],"mod_segments":[],"layout_segments":[],"img_update":[true],"audio_update":[true,true,true,true],
        "sigma":1.,"timestep":1000.,"shift_video":1.,"shift_audio":1.,"audio_scale":1.,
        "denoise_mask_present":false,"audio_denoise_mask_present":false,"pdd_head_count":1,
        "block_replacements_present":false,"checkpoint_sha256":"0".repeat(64),
        "provenance":{"trajectory":{"schema":"ferrum.ph3.original_euler.v1","solver":"euler",
            "cfg":1.,"steps":8,"original_f32_state":true,"routing_unchanged":true}}})).unwrap()
}

#[test]
fn capture_cpu_replay_checks_each_actual_state_and_refuses_scaling_before_gpu() {
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4096,
                output_bytes: 4096,
                transfer_staging_bytes: 4096,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let dir = tempfile::tempdir().unwrap();
    for (name, corrupt, scaled) in [
        ("good", false, false),
        ("bad", true, false),
        ("scaled", false, true),
    ] {
        let path = dir.path().join(format!("{name}.safetensors"));
        fixture(&path, corrupt, scaled);
        let archive = data::open_archive(&path, 16384).unwrap();
        let artifacts = artifacts::Artifacts::create(&dir.path().join(name)).unwrap();
        let result = EulerRun::prepare(&archive, &capture(), [1, 2, 2], &host, &artifacts, 16);
        assert_eq!(result.is_ok(), !corrupt && !scaled);
        if !scaled {
            let report: serde_json::Value = serde_json::from_slice(
                &std::fs::read(artifacts.directory().join("official-euler-cpu-replay.json"))
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(report["steps"].as_array().unwrap().len(), 16);
            assert_eq!(report["bit_mismatches"].as_u64().unwrap() > 0, corrupt);
        }
        drop(result);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

type TensorFixture = (String, safetensors::Dtype, Vec<usize>, Vec<u8>);
fn write_fixture(path: &std::path::Path, values: &[TensorFixture]) {
    use safetensors::tensor::{serialize_to_file, TensorView};
    serialize_to_file(
        values
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), TensorView::new(*d, s.clone(), b).unwrap())),
        &None,
        path,
    )
    .unwrap();
}
fn override_payloads() -> Vec<TensorFixture> {
    use safetensors::Dtype;
    let f = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let b = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect::<Vec<_>>()
    };
    vec![
        (
            "image_patches".into(),
            Dtype::F32,
            vec![2, 4],
            f(&[51., 52., 53., 54., 999., 999., 999., 999.]),
        ),
        (
            "text".into(),
            Dtype::BF16,
            vec![2, 2],
            b(&[7., 8., 9., 10.]),
        ),
        (
            "initial_image_state".into(),
            Dtype::F32,
            vec![1, 1, 1, 2, 2],
            f(&[1.00390625, 2., 3., 4.]),
        ),
        (
            "initial_audio_state".into(),
            Dtype::F32,
            vec![1, 1, 2, 2],
            f(&[5., 6., 7., 8.]),
        ),
        // This incompatible production representation must never replace the
        // original one-row per-step basis in this independent fixture.
        ("time_basis".into(), Dtype::F32, vec![3, 2], f(&[999.; 6])),
    ]
}
fn prepared_fixture(path: &std::path::Path) {
    use safetensors::{Dtype, SafeTensors};
    fixture(path, false, false);
    let bytes = std::fs::read(path).unwrap();
    let archive = SafeTensors::deserialize(&bytes).unwrap();
    let mut values: Vec<TensorFixture> = archive
        .tensors()
        .into_iter()
        .map(|(n, t)| (n, t.dtype(), t.shape().to_vec(), t.data().to_vec()))
        .collect();
    let f = |v: &[f32]| v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    for (name, _, shape, bytes) in &mut values {
        if name == "image_patches" {
            *shape = vec![2, 4];
            *bytes = f(&[101., 102., 103., 104., 1., 1., 1., 1.]);
        }
        if name == "trajectory_time_basis" {
            *bytes = f(&(0..8)
                .flat_map(|i| [i as f32 + 1., i as f32 + 2.])
                .collect::<Vec<_>>());
        }
    }
    let b = |v: &[f32]| {
        v.iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect::<Vec<_>>()
    };
    values.extend([
        ("text".into(), Dtype::BF16, vec![2, 2], b(&[1., 2., 3., 4.])),
        ("rope_cos".into(), Dtype::BF16, vec![4, 2], b(&[1.; 8])),
        ("rope_sin".into(), Dtype::BF16, vec![4, 2], b(&[0.; 8])),
    ]);
    write_fixture(path, &values);
}
fn test_host() -> HostMemoryAccount {
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4096,
                output_bytes: 4096,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn prepared_templates(
    host: &HostMemoryAccount,
    archive: &SafetensorsArchive,
) -> Vec<ManagedHostTensor> {
    use ferrum_runtime::{ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec};
    data::input_bindings(InputBoundary::Prepared)
        .iter()
        .map(|(name, id)| {
            let tensor = archive.tensor(name).unwrap();
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, tensor.bytes().len())
                .unwrap();
            bytes.as_mut().copy_from_slice(tensor.bytes());
            ManagedHostTensor::new(
                ProgramValueId::new(*id).unwrap(),
                ResolvedTensorSpec::new(
                    tensor.shape().to_vec(),
                    tensor.element_type().unwrap(),
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes.freeze(),
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn one_input_override_changes_only_selected_actual_bytes_and_keeps_official_replay() {
    let dir = tempfile::tempdir().unwrap();
    let official_path = dir.path().join("official.safetensors");
    prepared_fixture(&official_path);
    let override_path = dir.path().join("production.safetensors");
    write_fixture(&override_path, &override_payloads());
    let archive = data::open_archive(&official_path, 16384).unwrap();
    let mut c = capture();
    c.reference_shape = Some(c.target_shape);
    let host = test_host();
    let mut baseline = None;
    for (i, group) in [
        None,
        Some(OverrideGroup::Reference),
        Some(OverrideGroup::Text),
        Some(OverrideGroup::InitialState),
    ]
    .into_iter()
    .enumerate()
    {
        let artifacts = artifacts::Artifacts::create(&dir.path().join(format!("case{i}"))).unwrap();
        let mut run = EulerRun::prepare(&archive, &c, [1, 2, 2], &host, &artifacts, 16).unwrap();
        let replay: serde_json::Value = serde_json::from_slice(
            &std::fs::read(artifacts.directory().join("official-euler-cpu-replay.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(replay["bit_mismatches"], 0);
        if let Some(group) = group {
            run.bind_override(
                data::open_archive(&override_path, 16384).unwrap(),
                group,
                json!({"path":override_path}),
            )
            .unwrap();
            assert_eq!(
                run.override_receipt().unwrap()["official_replay_preserved"],
                true
            );
        }
        let templates = run
            .override_templates(prepared_templates(&host, &archive))
            .unwrap();
        let state = run.initial_state().unwrap();
        let ports = run.step_ports(&templates, &state, 3).unwrap();
        let actual: std::collections::BTreeMap<_, _> =
            data::input_bindings(InputBoundary::Prepared)
                .iter()
                .map(|(name, id)| {
                    (
                        *name,
                        ports
                            .iter()
                            .find(|p| p.value_id().as_str() == *id)
                            .unwrap()
                            .bytes()
                            .bytes()
                            .to_vec(),
                    )
                })
                .collect();
        if group.is_none() {
            baseline = Some(actual.clone());
        }
        let expected = baseline.as_ref().unwrap();
        for name in [
            "text",
            "image_patches",
            "audio_patches",
            "time_basis",
            "rope_cos",
            "rope_sin",
        ] {
            let changed = match group {
                Some(OverrideGroup::Reference) => name == "image_patches",
                Some(OverrideGroup::Text) => name == "text",
                Some(OverrideGroup::InitialState) => {
                    matches!(name, "image_patches" | "audio_patches")
                }
                None => false,
            };
            assert_eq!(
                actual[name] != expected[name],
                changed,
                "only the selected group may change: {name}"
            );
        }
        assert_eq!(
            actual["time_basis"],
            [4f32, 5.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        match group {
            Some(OverrideGroup::Reference) => {
                assert_eq!(
                    &actual["image_patches"][..16],
                    &override_payloads()[0].3[..16]
                );
                assert_eq!(
                    &actual["image_patches"][16..],
                    &expected["image_patches"][16..]
                );
                assert_eq!(f32_at(state[0].bytes(), 0), 1.0);
            }
            Some(OverrideGroup::InitialState) => {
                assert_eq!(
                    f32_at(state[0].bytes(), 0).to_bits(),
                    1.00390625f32.to_bits()
                );
                assert_eq!(
                    &actual["image_patches"][..16],
                    &expected["image_patches"][..16]
                );
                assert_eq!(f32_at(&actual["image_patches"], 4), 1.0); // Model ingress only rounds here.
                assert_eq!(state[1].bytes(), override_payloads()[3].3);
            }
            _ => (),
        }
        drop(ports);
        drop(state);
        drop(templates);
        drop(run);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn overrides_reject_incomplete_options_wrong_shapes_and_wrong_precision() {
    use super::super::InputOverride;
    use std::ffi::OsString;
    let mode = Trajectory::OriginalEuler;
    let boundary = InputBoundary::Prepared;
    assert!(
        InputOverride::parse(None, None, Trajectory::Single, boundary, false)
            .unwrap()
            .is_none()
    );
    assert!(InputOverride::parse(Some("input".into()), None, mode, boundary, false).is_err());
    assert!(InputOverride::parse(None, Some("text".into()), mode, boundary, false).is_err());
    for (mode, boundary, taps, group) in [
        (Trajectory::Single, boundary, false, "text"),
        (mode, InputBoundary::Packed, false, "text"),
        (mode, boundary, true, "text"),
        (mode, boundary, false, "text,reference"),
    ] {
        assert!(InputOverride::parse(
            Some("input".into()),
            Some(group.into()),
            mode,
            boundary,
            taps
        )
        .is_err());
    }
    for group in ["reference", "text", "initial-state"] {
        assert!(InputOverride::parse(
            Some("input".into()),
            Some(OsString::from(group)),
            mode,
            boundary,
            false
        )
        .unwrap()
        .is_some());
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("official.safetensors");
    prepared_fixture(&path);
    let archive = data::open_archive(&path, 16384).unwrap();
    let host = test_host();
    let mut c = capture();
    c.reference_shape = Some(c.target_shape);
    for (index, group) in [
        (0, OverrideGroup::Reference),
        (1, OverrideGroup::Text),
        (3, OverrideGroup::InitialState),
    ] {
        let mut values = override_payloads();
        if index == 1 {
            values[index].1 = safetensors::Dtype::F32;
            values[index].3 = vec![0; 16];
        } else {
            values[index].2.reverse();
        }
        let path = dir.path().join(format!("bad{index}.safetensors"));
        write_fixture(&path, &values);
        assert!(data::open_archive(&path, 1).is_err());
        let artifacts =
            artifacts::Artifacts::create(&dir.path().join(format!("error{index}"))).unwrap();
        let mut run = EulerRun::prepare(&archive, &c, [1, 2, 2], &host, &artifacts, 16).unwrap();
        assert!(run
            .bind_override(data::open_archive(&path, 16384).unwrap(), group, json!({}))
            .is_err());
        assert!(run.override_receipt().is_none());
        drop(run);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
