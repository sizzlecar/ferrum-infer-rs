use clap::{CommandFactory, Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Run(crate::commands::run::RunCliCommand),
    Serve(crate::commands::serve::ServeCliCommand),
}
fn validate(arguments: &[&str]) -> anyhow::Result<()> {
    let definition = Cli::command();
    let matches = definition.clone().try_get_matches_from(arguments)?;
    crate::commands::image_product::validate_cli_matches(&definition, &matches)
}

#[test]
fn ordered_media_cli_keeps_type_order_and_rejects_two_reference_representations() {
    let args = [
        "ferrum",
        "run",
        "--video-config",
        "video.json",
        "--reference",
        "audio=first=take.wav",
        "--reference",
        "image=second.png",
        "--reference",
        "video=third.mp4",
    ];
    validate(&args).unwrap();
    let matches = Cli::command().try_get_matches_from(args).unwrap();
    let run = matches.subcommand_matches("run").unwrap();
    let references = run
        .get_many::<super::VideoReferencePath>("references")
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(
        references[0].kind,
        ferrum_engine::video_generation::VideoReferenceKind::Audio
    );
    assert_eq!(
        references[0].path,
        std::path::PathBuf::from("first=take.wav")
    );
    assert_eq!(
        references[1].kind,
        ferrum_engine::video_generation::VideoReferenceKind::Image
    );
    let mut mixed = args.to_vec();
    mixed.extend(["--image", "other.png"]);
    assert!(validate(&mixed).is_err());
    assert_eq!(
        references[2].kind,
        ferrum_engine::video_generation::VideoReferenceKind::Video
    );
    assert_eq!(references[2].path, std::path::PathBuf::from("third.mp4"));
    for input in ["audio=", "video=", "image.png"] {
        assert!(input.parse::<super::VideoReferencePath>().is_err());
    }
}

#[test]
fn video_controls_select_both_entrypoints_and_reject_unimplemented_controls() {
    assert!(validate(&[
        "ferrum",
        "run",
        "--video-config",
        "video.json",
        "--prompt",
        "wave",
        "--image",
        "reference.png",
        "--output",
        "clip.mp4",
        "--width",
        "832",
        "--height",
        "480",
        "--num-frames",
        "124",
        "--fps",
        "24",
        "--seed",
        "1101"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "serve",
        "--video-config",
        "video.json",
        "--host",
        "127.0.0.1",
        "--port",
        "8092",
        "--served-model-name",
        "video",
        "--api-key",
        "test-only-key"
    ])
    .is_ok());
    for unsupported in ["--max-tokens", "--max-num-seqs", "--image-config"] {
        assert!(validate(&[
            "ferrum",
            "run",
            "--video-config",
            "video.json",
            unsupported,
            "2"
        ])
        .is_err());
    }
    assert!(validate(&[
        "ferrum",
        "run",
        "unrelated-text-model",
        "--video-config",
        "video.json"
    ])
    .is_err());
    assert!(validate(&[
        "ferrum",
        "run",
        "unrelated-text-model",
        "--num-frames",
        "124"
    ])
    .is_err());
}

#[test]
fn video_controls_leave_image_and_text_dispatch_intact() {
    assert!(validate(&[
        "ferrum",
        "run",
        "text-model",
        "--prompt",
        "hello",
        "--max-tokens",
        "10"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "run",
        "--image-config",
        "image.json",
        "--prompt",
        "lake",
        "--output",
        "image.png"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "run",
        "--image-config",
        "image.json",
        "--num-frames",
        "124"
    ])
    .is_err());
}
