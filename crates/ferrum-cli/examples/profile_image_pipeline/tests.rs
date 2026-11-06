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
