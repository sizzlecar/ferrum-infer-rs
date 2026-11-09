use super::*;
use clap::error::ErrorKind;
use ferrum_cli::commands::image_product::ImageProductSource;
use ferrum_engine::image_generation::ImageCudaPrecisionPolicy;

fn parse(options: &[&str]) -> std::result::Result<Arguments, clap::Error> {
    Arguments::try_parse_from(
        [
            "profile_image_pipeline",
            "--run",
            "--output",
            "profile.jsonl",
            "--prompt",
            "A ceramic teapot",
        ]
        .into_iter()
        .chain(options.iter().copied()),
    )
}

#[test]
fn cli_requires_exactly_one_image_source() {
    assert_eq!(
        parse(&[]).err().expect("source is required").kind(),
        ErrorKind::MissingRequiredArgument
    );
    assert_eq!(
        parse(&["--model", "qwen", "--image-config", "h3.json"])
            .err()
            .expect("sources conflict")
            .kind(),
        ErrorKind::ArgumentConflict
    );
}

#[test]
fn dit_capture_requires_explicit_budget_b1_and_excludes_other_captures() {
    assert!(parse(&["--image-config", "h3.json"])
        .unwrap()
        .dit_input_capture_dir
        .is_none());
    let options = [
        "--image-config",
        "h3.json",
        "--dit-input-capture-dir",
        "inputs",
        "--dit-input-capture-max-bytes",
        "536870912",
    ];
    validate_arguments(&parse(&options).unwrap()).unwrap();
    assert!(parse(&options[..4]).is_err());
    let mut invalid = options.to_vec();
    invalid[5] = "0";
    assert!(parse(&invalid).is_err());
    let mut mixed = options.to_vec();
    mixed.extend(["--batch", "2"]);
    assert!(validate_arguments(&parse(&mixed).unwrap()).is_err());
    let mut conflict = options.to_vec();
    conflict.extend([
        "--final-latent-capture-dir",
        "latent",
        "--final-latent-capture-max-bytes",
        "1000",
    ]);
    assert_eq!(
        parse(&conflict).err().unwrap().kind(),
        ErrorKind::ArgumentConflict
    );
}

#[test]
fn explicit_config_selects_the_document_for_kernel_and_off_without_loading_assets() {
    for timing in ["kernel", "off"] {
        let args = parse(&["--image-config", "h3.json", "--device-timing", timing]).unwrap();
        let ImageProductSource::ExplicitConfig(path) = args.source().unwrap() else {
            panic!("explicit configuration must use the declared document");
        };
        assert_eq!(path, PathBuf::from("h3.json"));
        assert!(args.gpu_memory_utilization.is_none());
        assert!(args.cudnn_library.is_none());
        match timing {
            "kernel" => assert!(matches!(args.device_timing, Timing::Kernel)),
            "off" => assert!(matches!(args.device_timing, Timing::Off)),
            _ => unreachable!(),
        }
    }
}

#[test]
fn explicit_config_rejects_directory_only_options_including_explicit_defaults() {
    for option in [
        ["--gpu-memory-utilization", "0.9"],
        ["--cudnn-library", "libcudnn.so.9"],
    ] {
        let error = parse(&["--image-config", "h3.json", option[0], option[1]])
            .err()
            .expect("directory options must not be silently ignored");
        assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
    }
}

#[test]
fn directory_source_keeps_existing_fp8_defaults() {
    let args = parse(&["--model", "qwen"]).unwrap();
    let ImageProductSource::LocalDirectory { path, options } = args.source().unwrap() else {
        panic!("model directory must keep its existing product path");
    };
    assert_eq!(path, PathBuf::from("qwen"));
    assert_eq!(options.utilization, f64::from(0.9_f32));
    assert_eq!(options.maximum_batch_size, 1);
    assert!(options.cudnn_library.is_none());
    assert_eq!(
        options.precision,
        ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1
    );
}

#[test]
fn directory_source_preserves_explicit_memory_library_and_batch_capacity() {
    let args = parse(&[
        "--model",
        "qwen",
        "--gpu-memory-utilization",
        "0.7",
        "--cudnn-library",
        "libcudnn.so.9",
        "--batch",
        "1,2",
    ])
    .unwrap();
    let ImageProductSource::LocalDirectory { options, .. } = args.source().unwrap() else {
        panic!("expected directory options");
    };
    assert_eq!(options.utilization, f64::from(0.7_f32));
    assert_eq!(options.maximum_batch_size, 2);
    assert_eq!(options.cudnn_library, Some(PathBuf::from("libcudnn.so.9")));
    assert_eq!(
        options.precision,
        ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1
    );
    let invalid = parse(&["--model", "qwen", "--gpu-memory-utilization", "0"]).unwrap();
    assert!(invalid.source().is_err());
}

#[test]
fn capture_is_disabled_by_default_and_explicit_bytes_cover_the_directory() {
    let defaults = parse(&["--image-config", "h3.json"]).unwrap();
    assert!(defaults.main_vision_capture_dir.is_none());
    assert!(defaults.main_vision_capture_max_bytes.is_none());
    let args = parse(&[
        "--image-config",
        "h3.json",
        "--image",
        "input.png",
        "--main-vision-capture-dir",
        "features",
        "--main-vision-capture-max-bytes",
        "72",
    ])
    .unwrap();
    assert_eq!(
        args.main_vision_capture_dir,
        Some(PathBuf::from("features"))
    );
    assert_eq!(args.main_vision_capture_max_bytes.unwrap().get(), 72);
}

#[test]
fn capture_requires_a_paired_nonzero_budget_and_an_explicit_edit_source() {
    for options in [
        vec![
            "--image-config",
            "h3.json",
            "--image",
            "input.png",
            "--main-vision-capture-dir",
            "features",
        ],
        vec![
            "--image-config",
            "h3.json",
            "--image",
            "input.png",
            "--main-vision-capture-max-bytes",
            "72",
        ],
        vec![
            "--image-config",
            "h3.json",
            "--main-vision-capture-dir",
            "features",
            "--main-vision-capture-max-bytes",
            "72",
        ],
    ] {
        assert_eq!(
            parse(&options).err().unwrap().kind(),
            ErrorKind::MissingRequiredArgument
        );
    }
    let zero = parse(&[
        "--image-config",
        "h3.json",
        "--image",
        "input.png",
        "--main-vision-capture-dir",
        "features",
        "--main-vision-capture-max-bytes",
        "0",
    ])
    .err()
    .unwrap();
    assert_eq!(zero.kind(), ErrorKind::ValueValidation);
    let directory = parse(&[
        "--model",
        "qwen",
        "--image",
        "input.png",
        "--main-vision-capture-dir",
        "features",
        "--main-vision-capture-max-bytes",
        "72",
    ])
    .err()
    .unwrap();
    assert_eq!(directory.kind(), ErrorKind::ArgumentConflict);
}

#[test]
fn text_capture_requires_explicit_edit_budget_and_excludes_vision_capture() {
    let defaults = parse(&["--image-config", "h3.json"]).unwrap();
    assert!(defaults.main_text_capture_dir.is_none());
    assert!(defaults.main_text_capture_max_bytes.is_none());
    let options = [
        "--image-config",
        "h3.json",
        "--image",
        "input.png",
        "--main-text-capture-dir",
        "text",
        "--main-text-capture-max-bytes",
        "134217728",
    ];
    let args = parse(&options).unwrap();
    validate_arguments(&args).unwrap();
    assert_eq!(args.main_text_capture_max_bytes.unwrap().get(), 134217728);
    for omitted_pair in [2, 4, 6] {
        let missing: Vec<_> = options
            .iter()
            .enumerate()
            .filter_map(|(i, arg)| (!(omitted_pair..omitted_pair + 2).contains(&i)).then_some(*arg))
            .collect();
        assert_eq!(
            parse(&missing).err().unwrap().kind(),
            ErrorKind::MissingRequiredArgument
        );
    }
    let mut zero = options;
    zero[7] = "0";
    assert_eq!(
        parse(&zero).err().unwrap().kind(),
        ErrorKind::ValueValidation
    );
    let mut both = options.to_vec();
    both.extend([
        "--main-vision-capture-dir",
        "vision",
        "--main-vision-capture-max-bytes",
        "134217728",
    ]);
    assert_eq!(
        parse(&both).err().unwrap().kind(),
        ErrorKind::ArgumentConflict
    );
    let mut batched = options.to_vec();
    batched.extend(["--batch", "2"]);
    assert!(validate_arguments(&parse(&batched).unwrap()).is_err());
    let mut directory = options;
    directory[0] = "--model";
    assert_eq!(
        parse(&directory).err().unwrap().kind(),
        ErrorKind::ArgumentConflict
    );
}

#[test]
fn final_latent_capture_is_explicit_b1_and_excludes_conditioning_modes() {
    let defaults = parse(&["--image-config", "h3.json"]).unwrap();
    assert!(defaults.final_latent_capture_dir.is_none());
    assert!(defaults.final_latent_capture_max_bytes.is_none());
    let options = [
        "--image-config",
        "h3.json",
        "--final-latent-capture-dir",
        "final",
        "--final-latent-capture-max-bytes",
        "8388608",
    ];
    let args = parse(&options).unwrap();
    validate_arguments(&args).unwrap();
    assert_eq!(args.final_latent_capture_max_bytes.unwrap().get(), 8388608);
    // The same complete-target diagnostic supports both generation and edit.
    let mut edit = options.to_vec();
    edit.extend(["--image", "input.png"]);
    validate_arguments(&parse(&edit).unwrap()).unwrap();
    for pair in [2, 4] {
        let missing: Vec<_> = options
            .iter()
            .enumerate()
            .filter_map(|(i, arg)| (!(pair..pair + 2).contains(&i)).then_some(*arg))
            .collect();
        assert_eq!(
            parse(&missing).err().unwrap().kind(),
            ErrorKind::MissingRequiredArgument
        );
    }
    let mut zero = options;
    zero[5] = "0";
    assert_eq!(
        parse(&zero).err().unwrap().kind(),
        ErrorKind::ValueValidation
    );
    let mut directory = options;
    directory[0] = "--model";
    assert_eq!(
        parse(&directory).err().unwrap().kind(),
        ErrorKind::ArgumentConflict
    );
    for (dir, limit) in [
        ("--main-text-capture-dir", "--main-text-capture-max-bytes"),
        (
            "--main-vision-capture-dir",
            "--main-vision-capture-max-bytes",
        ),
    ] {
        let mut both = edit.clone();
        both.extend([dir, "other", limit, "134217728"]);
        assert_eq!(
            parse(&both).err().unwrap().kind(),
            ErrorKind::ArgumentConflict
        );
    }
    for batch in ["2", "1,1"] {
        let mut batched = options.to_vec();
        batched.extend(["--batch", batch]);
        assert!(validate_arguments(&parse(&batched).unwrap()).is_err());
    }
}
