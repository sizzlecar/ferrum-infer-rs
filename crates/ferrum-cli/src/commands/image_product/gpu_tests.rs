//! Explicit, ignored CUDA functional qualification of a real shared B2 forward.
//!
//! Required test inputs:
//! - `FERRUM_TEST_IMAGE_PRODUCT_CONFIG`: the ordinary typed product JSON.
//! - `FERRUM_TEST_IMAGE_ARTIFACT_DIR`: a directory that must not already exist.
//! - Optional `FERRUM_TEST_IMAGE_REQUESTS_JSON`: JSON array of exactly two normal
//!   ImageGenerationRequests. Otherwise use the product's explicit serving
//!   default dimensions, two simple prompts, and seeds 42/43.
//!
//! These are test fixture inputs, not hidden product configuration. The test
//! performs one `generate_batch` call on one CUDA runtime/Host account. It does
//! not claim warm residency, HTTP batching, Torch parity, or G3 throughput.

use super::{configuration::ImageBackendConfig, source, ImageProductConfig, ImageProductError};
use anyhow::{ensure, Context, Result};
use ferrum_engine::image_generation::{ImageGenerationOutput, ImageGenerationRequest};
use ferrum_models::qwen_image21::pipeline_adapters::QwenImage21RgbaOutput;
use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};

fn required_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .with_context(|| format!("set explicit test input {name}"))
}

fn requests(config: &ImageProductConfig) -> Result<[ImageGenerationRequest; 2]> {
    let requests = if let Some(path) = std::env::var_os("FERRUM_TEST_IMAGE_REQUESTS_JSON") {
        let bytes = source::read_bounded(Path::new(&path), 2 * 1024 * 1024 + 4096)?;
        serde_json::from_slice::<[ImageGenerationRequest; 2]>(&bytes)?
    } else {
        let serving = config.serving.as_ref().context(
            "provide FERRUM_TEST_IMAGE_REQUESTS_JSON or explicit serving default dimensions",
        )?;
        [
            ImageGenerationRequest {
                prompt: "A red kite above the sea".into(),
                width: serving.default_width,
                height: serving.default_height,
                seed: 42,
                n: 1,
            },
            ImageGenerationRequest {
                prompt: "A blue ceramic bowl on a wooden table".into(),
                width: serving.default_width,
                height: serving.default_height,
                seed: 43,
                n: 1,
            },
        ]
    };
    ensure!(
        requests[0].prompt != requests[1].prompt && requests[0].seed != requests[1].seed,
        "B2 fixture requires distinct prompts and independent seeds"
    );
    ensure!(
        requests[0].width == requests[1].width && requests[0].height == requests[1].height,
        "B2 fixture requires compatible dimensions"
    );
    for request in &requests {
        ensure!(request.n == 1, "each batch request returns one image");
        PngLimits::default().validate(PngDimensions::new(
            request.width.into(),
            request.height.into(),
        )?)?;
    }
    Ok(requests)
}

struct HashedWriter<W> {
    writer: W,
    sha256: Sha256,
}
impl<W: Write> Write for HashedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.writer.write(bytes)?;
        self.sha256.update(&bytes[..written]);
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

fn write_output(
    host: &HostMemoryAccount,
    directory: &Path,
    index: usize,
    request: &ImageGenerationRequest,
    output: &ImageGenerationOutput,
) -> Result<Value> {
    ensure!(
        output.receipt.batch_size == 2 && output.receipt.batch_index == index,
        "receipt does not describe the actual B2 item"
    );
    ensure!(
        output.receipt.seed == request.seed
            && output.receipt.width == request.width
            && output.receipt.height == request.height,
        "receipt differs from the submitted request"
    );
    // This validates all BF16 elements for finiteness before creating the PNG.
    let pixels = QwenImage21RgbaOutput::new(host, output.rgba.tensor(), output.rgba.bytes())?;
    ensure!(
        pixels.dimensions() == [1, 4, request.height as usize, request.width as usize],
        "returned item is not exact [1,4,H,W]"
    );
    let dimensions = PngDimensions::new(request.width.into(), request.height.into())?;
    let mut scratch = host.allocate_zeroed(HostMemoryClass::Output, 16 * 1024)?;
    let filename = format!("image-{index}.png");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(&filename))?;
    let mut writer = HashedWriter {
        writer: file,
        sha256: Sha256::new(),
    };
    let png = write_rgba8_png(&mut writer, dimensions, PngLimits::default(), |sink| {
        pixels
            .write_frame_rgba8(0, scratch.as_mut(), sink)
            .map_err(io::Error::other)
    })?;
    writer.writer.sync_all()?;
    let sha256 = format!("{:x}", writer.sha256.finalize());
    let record = json!({
        "request": request, "generation": output.receipt,
        "rgba_bf16": {"shape":output.rgba.tensor().dimensions(), "finite":true,
            "bytes":output.rgba.bytes().len(), "sha256":format!("{:x}",Sha256::digest(output.rgba.bytes().bytes()))},
        "png": {"file":filename,"width":png.dimensions().width(),"height":png.dimensions().height(),
            "color_type":"rgba8","encoded_bytes":png.encoded_bytes(),"sha256":sha256},
    });
    write_json_new(&directory.join(format!("image-{index}.json")), &record)?;
    Ok(record)
}

fn write_json_new(path: &Path, value: &Value) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

#[test]
#[ignore = "requires explicit real assets, CUDA/cuDNN budgets and an exclusive GPU window"]
fn cuda_product_one_real_batch_of_two_preserves_outputs_and_retires_host(
) -> Result<(), ImageProductError> {
    let config_path = required_path("FERRUM_TEST_IMAGE_PRODUCT_CONFIG")?;
    let artifact_dir = required_path("FERRUM_TEST_IMAGE_ARTIFACT_DIR")?;
    let config = ImageProductConfig::load(&config_path)?;
    let requests = requests(&config)?;
    let ImageBackendConfig::Cuda(cuda) = &config.backend;
    let cuda = cuda.resolved()?;
    cuda.validate()?;
    let assets = config.prepare_assets()?;
    // Atomic exclusive creation rejects old evidence before device construction.
    fs::create_dir(&artifact_dir).map_err(anyhow::Error::from)?;
    write_json_new(
        &artifact_dir.join("request.json"),
        &json!({
            "config":config_path,"requests":requests,"execution":"one cold generate_batch call",
            "qualification":"functional B2 only; not G3, throughput or Torch parity"
        }),
    )?;
    let mut pipeline = Box::new(cuda.create_pipeline(assets)?);
    let host = pipeline.host_account().clone();
    let outputs = match pipeline.generate_batch(Vec::from(requests.clone())) {
        Ok(outputs) => outputs,
        Err(source) => return Err(ImageProductError::Generation { source, pipeline }),
    };
    let inspect = (|| -> Result<(Vec<Value>, usize)> {
        ensure!(
            outputs.len() == requests.len(),
            "batch output count mismatch"
        );
        ensure!(
            outputs[0].receipt.job == outputs[1].receipt.job,
            "outputs did not originate in one admitted batch"
        );
        ensure!(
            !outputs[0].rgba.bytes().same_region(outputs[1].rgba.bytes()),
            "independent outputs alias the same batch slice"
        );
        let mut records = Vec::with_capacity(2);
        let mut bytes = 0usize;
        for (index, (request, output)) in requests.iter().zip(&outputs).enumerate() {
            bytes = bytes
                .checked_add(output.rgba.bytes().len())
                .context("output byte sum overflows")?;
            records.push(write_output(&host, &artifact_dir, index, request, output)?);
        }
        let usage = host.usage()?;
        ensure!(
            usage.committed_bytes() == bytes
                && usage.allocated_bytes() == bytes
                && usage.class_committed_bytes(HostMemoryClass::Output) == bytes,
            "non-output Host owners remain after all component stages closed"
        );
        Ok((records, bytes))
    })()?;
    let (records, batch_bytes) = inspect;
    let final_alias = outputs[1].rgba.bytes().clone();
    let final_alias_sha = format!("{:x}", Sha256::digest(final_alias.bytes()));
    drop(outputs);
    drop(pipeline);
    let finish = (|| -> Result<()> {
        let retained = host.usage()?;
        ensure!(
            host.owns(&final_alias)
                && retained.committed_bytes() == batch_bytes
                && retained.allocated_bytes() == batch_bytes,
            "last output slice must retain its entire batch backing across pipeline drop"
        );
        ensure!(
            format!("{:x}", Sha256::digest(final_alias.bytes())) == final_alias_sha,
            "retained output changed after pipeline destruction"
        );
        drop(final_alias);
        let released = host.usage()?;
        ensure!(
            released.committed_bytes() == 0 && released.allocated_bytes() == 0,
            "Host backing/claims survived the last output owner"
        );
        write_json_new(
            &artifact_dir.join("result.json"),
            &json!({
                "status":"passed", "qualification":"functional cold B2; no G3/performance/Torch parity claim",
                "batch_size":2,"outputs":records,"retained_batch_bytes":batch_bytes,
                "host_final_committed_bytes":released.committed_bytes(),
                "host_final_allocated_bytes":released.allocated_bytes(),
            }),
        )
    })();
    finish.map_err(Into::into)
}
