use super::{ImageProductError, ImageProductSource};
use std::path::Path;

/// One process, one runtime and one Host account. This is a cold serial
/// functional path, not concurrent serving or a throughput qualification.
pub fn execute_run(
    config_path: &Path,
    prompt: String,
    width: u32,
    height: u32,
    seed: u64,
    destination: &Path,
) -> Result<(), ImageProductError> {
    execute_run_source(
        ImageProductSource::ExplicitConfig(config_path.to_owned()),
        prompt,
        width,
        height,
        seed,
        destination,
    )
}

pub fn execute_run_source(
    source: ImageProductSource,
    prompt: String,
    width: u32,
    height: u32,
    seed: u64,
    destination: &Path,
) -> Result<(), ImageProductError> {
    execute_run_source_with_image(source, prompt, width, height, seed, destination, None)
}

pub fn execute_run_source_with_image(
    source: ImageProductSource,
    prompt: String,
    width: u32,
    height: u32,
    seed: u64,
    destination: &Path,
    image: Option<&Path>,
) -> Result<(), ImageProductError> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (source, prompt, width, height, seed, destination, image);
        Err(ImageProductError::Unsupported(
            "image generation currently requires a Ferrum binary built with the cuda feature",
        ))
    }
    #[cfg(feature = "cuda")]
    {
        use ferrum_engine::image_generation::{ImageEditRequest, ImageGenerationRequest};
        use ferrum_models::qwen_image21::pipeline_adapters::QwenImage21RgbaOutput;
        use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
        use ferrum_runtime::HostMemoryClass;
        use std::{fs::OpenOptions, io};

        let product = source.prepare()?;
        // Reject absent/oversized input before device construction. Keep this
        // open handle and recheck the read extent rather than trusting a path.
        let image_file = image.map(open_image).transpose()?;
        if width == 0
            || height == 0
            || width > product.cuda.pipeline.maximum_width
            || height > product.cuda.pipeline.maximum_height
        {
            return Err(
                anyhow::anyhow!("requested image dimensions exceed configured bounds").into(),
            );
        }
        let grid_multiple = product
            .assets
            .spatial_scale()?
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("image spatial multiple overflows"))?;
        if grid_multiple == 0
            || width % grid_multiple != 0
            || height % grid_multiple != 0
            || prompt.len() > 1024 * 1024
        {
            return Err(anyhow::anyhow!("image dimensions must produce even latent axes and the prompt must fit the 1 MiB bound").into());
        }
        let dimensions =
            PngDimensions::new(width.into(), height.into()).map_err(anyhow::Error::from)?;
        let png_limits = PngLimits::default();
        png_limits
            .validate(dimensions)
            .map_err(anyhow::Error::from)?;
        // Opening archives and validating their actual metadata precedes CUDA.
        let (pipeline, composition) = product.cuda.create_pipeline_with_receipt(product.assets)?;
        let mut pipeline = Box::new(pipeline);
        tracing::warn!(
            profile = product.profile,
            residency = ?pipeline.residency_policy(),
            "experimental image run; no throughput qualification"
        );
        let request = ImageGenerationRequest {
            prompt,
            width,
            height,
            seed,
            n: 1,
        };
        let result = if let Some((file, length)) = image_file {
            if !pipeline.supports_edits() {
                return Err(anyhow::anyhow!(
                    "this image package does not declare complete editing assets"
                )
                .into());
            }
            let encoded = match read_image(pipeline.host_account(), file, length) {
                Ok(encoded) => encoded,
                Err(error) => {
                    if let Err(source) = pipeline.shutdown() {
                        return Err(ImageProductError::Generation { source, pipeline });
                    }
                    return Err(error.into());
                }
            };
            pipeline.edit(ImageEditRequest {
                generation: request,
                image: encoded,
            })
        } else {
            pipeline.generate(request)
        };
        let output = match result {
            Ok(output) => output,
            Err(source) => return Err(ImageProductError::Generation { source, pipeline }),
        };
        // Close native sessions before fallible output encoding. The returned
        // image keeps its independent managed Host backing after device close.
        let shutdown = match pipeline.shutdown() {
            Ok(receipts) => receipts,
            Err(source) => return Err(ImageProductError::Generation { source, pipeline }),
        };
        // The model adapter checks every pixel is finite before touching a sink.
        let rgba = QwenImage21RgbaOutput::new(
            pipeline.host_account(),
            output.rgba.tensor(),
            output.rgba.bytes(),
        )?;
        if rgba.dimensions() != [1, 4, height as usize, width as usize] {
            return Err(anyhow::anyhow!(
                "image pipeline output dimensions differ from the request"
            )
            .into());
        }
        let mut scratch = pipeline
            .host_account()
            .allocate_zeroed(HostMemoryClass::Output, 16 * 1024)
            .map_err(anyhow::Error::from)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)
            .map_err(anyhow::Error::from)?;
        let png = write_rgba8_png(&mut file, dimensions, png_limits, |writer| {
            rgba.write_frame_rgba8(0, scratch.as_mut(), writer)
                .map_err(io::Error::other)
        })
        .map_err(anyhow::Error::from)?;
        file.sync_all().map_err(anyhow::Error::from)?;
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "model": product.model,
                "execution_profile": product.profile,
                "composition": composition,
                "output": destination,
                "request_kind": if image.is_some() {"edit"} else {"generate"},
                "png": {"width": png.dimensions().width(), "height": png.dimensions().height(),
                    "encoded_bytes": png.encoded_bytes(), "color_type": "rgba8"},
                "generation": output.receipt,
                "shutdown": shutdown,
            }))
            .map_err(anyhow::Error::from)?
        );
        Ok(())
    }
}

#[cfg(any(feature = "cuda", test))]
fn open_image(path: &Path) -> anyhow::Result<(std::fs::File, usize)> {
    use ferrum_models::qwen_image21::image_input::ImageInputLimits;
    let file = std::fs::File::open(path)?;
    let metadata = file.metadata()?;
    anyhow::ensure!(metadata.is_file(), "edit image must be a regular file");
    let length = usize::try_from(metadata.len())?;
    let maximum = ImageInputLimits::default().maximum_encoded_bytes();
    anyhow::ensure!(
        length > 0 && length <= maximum,
        "edit image must contain 1..={maximum} encoded bytes"
    );
    Ok((file, length))
}

#[cfg(any(feature = "cuda", test))]
fn read_image(
    host: &ferrum_runtime::HostMemoryAccount,
    mut file: std::fs::File,
    length: usize,
) -> anyhow::Result<ferrum_runtime::ManagedHostBytes> {
    use std::io::Read;
    let mut bytes =
        host.allocate_zeroed(ferrum_runtime::HostMemoryClass::InputPreprocess, length)?;
    file.read_exact(bytes.as_mut())?;
    let mut extra = [0u8; 1];
    anyhow::ensure!(
        file.read(&mut extra)? == 0,
        "edit image changed length during read"
    );
    anyhow::ensure!(
        bytes.as_ref().starts_with(b"\x89PNG\r\n\x1a\n"),
        "edit input must be PNG"
    );
    Ok(bytes.freeze())
}

#[cfg(test)]
mod edit_tests;
