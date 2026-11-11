//! Export actual H3 main-vision pixels and optional normalized patch inputs.
//! Pixel differences are diagnostics, not a quality verdict or kernel attribution.
use anyhow::{ensure, Context, Result};
use ferrum_models::{
    p_h3_image::{PH3ImageEditPixelPlan, PH3ImageEditPixelPolicy, PH3_IMAGE_EDIT_PIXEL_PROFILE},
    qwen3_vl_vision::{
        comfy::{
            Qwen3VLVisionComfyImagePlan, Qwen3VLVisionComfyPixelPolicy,
            QWEN3_VL_COMFY_IMAGE_PREPROCESS_PROFILE,
        },
        Qwen3VLVisionArchitecture,
    },
    qwen_image21::image_input::{ImageInputLimits, ManagedRgba8Image},
};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const USAGE: &str = "export_h3_vision_input --input PNG --generation-width N --generation-height N --output-vision NEW_PNG [--maximum-host-bytes N (536870912)] [--text-encoder-config JSON --output-patches NEW_F32_RAW]\nCPU-only production H3 pixel preprocessing. Does not run expansion, vision inference, VAE or denoising.";

struct PatchExport {
    config: PathBuf,
    output: PathBuf,
}

struct Arguments {
    input: PathBuf,
    output: PathBuf,
    width: u32,
    height: u32,
    host_bytes: usize,
    patches: Option<PatchExport>,
}

fn arguments() -> Result<Option<Arguments>> {
    parse_arguments(std::env::args_os().skip(1))
}

fn parse_arguments(args: impl IntoIterator<Item = OsString>) -> Result<Option<Arguments>> {
    let mut args = args.into_iter().peekable();
    if args.peek().is_some_and(|v| v == "--help" || v == "-h") {
        args.next();
        ensure!(args.next().is_none(), "--help accepts no other arguments");
        return Ok(None);
    }
    let mut values = BTreeMap::<String, OsString>::new();
    while let Some(flag) = args.next() {
        let flag = flag
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 option name"))?;
        let value = args.next().context("every option requires a value")?;
        ensure!(
            values.insert(flag.clone(), value).is_none(),
            "duplicate option {flag}"
        );
    }
    let input = values.remove("--input").context(USAGE)?.into();
    let output = values.remove("--output-vision").context(USAGE)?.into();
    let patches = match (
        values.remove("--text-encoder-config"),
        values.remove("--output-patches"),
    ) {
        (None, None) => None,
        (Some(config), Some(output)) => Some(PatchExport {
            config: config.into(),
            output: output.into(),
        }),
        _ => anyhow::bail!("--text-encoder-config and --output-patches must be supplied together"),
    };
    let number = |value: OsString| -> Result<u64> {
        value
            .to_str()
            .context("numeric option requires UTF-8")?
            .parse()
            .context("invalid unsigned integer")
    };
    let width = u32::try_from(number(values.remove("--generation-width").context(USAGE)?)?)?;
    let height = u32::try_from(number(
        values.remove("--generation-height").context(USAGE)?,
    )?)?;
    let host_bytes = usize::try_from(number(
        values
            .remove("--maximum-host-bytes")
            .unwrap_or_else(|| "536870912".into()),
    )?)?;
    ensure!(values.is_empty(), "unknown options: {:?}", values.keys());
    ensure!(
        width > 0 && height > 0 && host_bytes > 0 && host_bytes <= isize::MAX as usize,
        "dimensions and Host limit must be positive and addressable"
    );
    Ok(Some(Arguments {
        input,
        output,
        width,
        height,
        host_bytes,
        patches,
    }))
}

fn create_output(path: &Path) -> Result<File> {
    Ok(OpenOptions::new().create_new(true).write(true).open(path)?)
}

fn export_patches(
    host: &HostMemoryAccount,
    image: &ManagedRgba8Image,
    args: &PatchExport,
) -> Result<serde_json::Value> {
    let mut source = File::open(&args.config)?;
    let metadata = source.metadata()?;
    let length = usize::try_from(metadata.len())?;
    ensure!(
        metadata.is_file() && length > 0 && length <= 4 * 1024 * 1024,
        "text encoder config must be a nonempty regular file of at most 4 MiB"
    );
    let mut config = host.allocate_zeroed(HostMemoryClass::InputPreprocess, length)?;
    source.read_exact(config.as_mut())?;
    ensure!(
        source.read(&mut [0_u8; 1])? == 0,
        "config changed length while reading"
    );
    let config = config.freeze();
    let config_sha256 = format!("{:x}", Sha256::digest(config.bytes()));
    let architecture =
        Qwen3VLVisionArchitecture::from_hf_value(&serde_json::from_slice(config.bytes())?)?;
    drop(config);
    let plan = Qwen3VLVisionComfyImagePlan::new(
        architecture,
        image.width(),
        image.height(),
        Qwen3VLVisionComfyPixelPolicy::default(),
    )?;
    // This is the same managed input constructor used by H3 execute_vision_inputs.
    let patches = plan.managed_patches(host, image)?;
    let bytes = patches.bytes().bytes();
    let mut destination = create_output(&args.output)?;
    destination.write_all(bytes)?;
    destination.flush()?;
    let grid = plan.config().grid();
    Ok(json!({
        "path":args.output, "config_path":args.config, "config_sha256":config_sha256,
        "profile":QWEN3_VL_COMFY_IMAGE_PREPROCESS_PROFILE,
        "value_id":patches.value_id().as_str(), "dtype":"F32", "byte_order":"little_endian",
        "layout":"contiguous", "shape":patches.tensor().dimensions(),
        "grid_thw":[grid.temporal(),grid.height(),grid.width()],
        "resized_dimensions_wh":plan.resized_dimensions(),
        "raw_bytes":bytes.len(), "raw_sha256":format!("{:x}",Sha256::digest(bytes)),
        "host_backing_bytes":patches.bytes().backing_bytes()
    }))
}

fn run(args: Arguments) -> Result<()> {
    let limits = ImageInputLimits::default();
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        args.host_bytes,
        HostMemoryClassLimits {
            input_preprocess_bytes: args.host_bytes,
            output_bytes: 0,
            transfer_staging_bytes: 0,
            weight_backing_bytes: 0,
        },
    )?);
    let mut input = File::open(&args.input)?;
    let metadata = input.metadata()?;
    let length = usize::try_from(metadata.len())?;
    ensure!(
        metadata.is_file() && length > 0 && length <= limits.maximum_encoded_bytes(),
        "input must be a regular PNG within the production encoded bound"
    );
    let mut encoded = host.allocate_zeroed(HostMemoryClass::InputPreprocess, length)?;
    input.read_exact(encoded.as_mut())?;
    ensure!(
        input.read(&mut [0_u8; 1])? == 0,
        "input changed length while reading"
    );
    let encoded = encoded.freeze();
    let input_sha256 = format!("{:x}", Sha256::digest(encoded.bytes()));
    let decoded = ManagedRgba8Image::decode_png(&host, &encoded, limits)?;
    let plan = PH3ImageEditPixelPlan::new(
        decoded.width(),
        decoded.height(),
        args.width,
        args.height,
        PH3ImageEditPixelPolicy::default(),
        limits,
    )?;
    let prepared = plan.prepare(&host, &decoded)?;
    let vision = &prepared.vision;
    // Write the original managed logical bytes, without another resize or copy.
    let mut destination = BufWriter::new(create_output(&args.output)?);
    {
        let mut encoder = png::Encoder::new(&mut destination, vision.width(), vision.height());
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(vision.bytes().bytes())?;
        writer.finish()?;
    }
    destination.flush()?;
    let mut report = json!({
        "complete":true, "profile":PH3_IMAGE_EDIT_PIXEL_PROFILE,
        "source":{"path":args.input,"png_sha256":input_sha256,"width":decoded.width(),"height":decoded.height()},
        "generation":{"width":args.width,"height":args.height},
        "vision":{"path":args.output,"width":vision.width(),"height":vision.height(),"layout":"HWC_RGBA8","logical_bytes":vision.bytes().bytes().len(),"rgba8_sha256":format!("{:x}",Sha256::digest(vision.bytes().bytes()))},
        "host_limit_bytes":args.host_bytes, "host_limit_is_process_rss":false,
        "model_loaded":false,"gpu_used":false,"quality_status":"not_evaluated"
    });
    if let Some(patches) = &args.patches {
        report["normalized_patches"] = export_patches(&host, vision, patches)?;
    }
    drop((prepared, decoded, encoded));
    let remaining = host.usage()?.committed_bytes();
    ensure!(
        remaining == 0,
        "original preprocessing owners were not retired"
    );
    report["host_committed_after_drop"] = json!(remaining);
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_export_options_are_paired_and_legacy_arguments_remain_valid() {
        let base = [
            "--input",
            "source.png",
            "--generation-width",
            "2912",
            "--generation-height",
            "2912",
            "--output-vision",
            "new.png",
        ];
        let parse = |extra: &[&str]| parse_arguments(base.iter().chain(extra).map(OsString::from));
        assert!(parse(&[]).unwrap().unwrap().patches.is_none());
        assert!(parse(&["--text-encoder-config", "config.json"]).is_err());
        assert!(parse(&["--output-patches", "patches.f32"]).is_err());
        let args = parse(&[
            "--text-encoder-config",
            "config.json",
            "--output-patches",
            "patches.f32",
        ])
        .unwrap()
        .unwrap();
        let patches = args.patches.unwrap();
        assert_eq!(patches.config, PathBuf::from("config.json"));
        assert_eq!(patches.output, PathBuf::from("patches.f32"));
    }

    #[test]
    fn output_creation_never_truncates_an_existing_capture() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ferrum-vision-export-{}-{unique}",
            std::process::id()
        ));
        let mut file = create_output(&path).unwrap();
        file.write_all(b"original capture").unwrap();
        drop(file);
        let rejected = create_output(&path).is_err();
        let retained = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(rejected);
        assert_eq!(retained, b"original capture");
    }
}

fn main() -> Result<()> {
    match arguments()? {
        Some(args) => run(args),
        None => {
            println!("{USAGE}");
            Ok(())
        }
    }
}
