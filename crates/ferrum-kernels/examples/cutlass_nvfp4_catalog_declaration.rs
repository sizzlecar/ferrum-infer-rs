//! CPU-only source declaration for the first locked native package.
use ferrum_kernels::native_ops::cutlass_nvfp4::{provider_catalog_declaration, CutlassNvfp4Config};
use std::{error::Error, fs::OpenOptions, io::Write};
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: cutlass_nvfp4_catalog_declaration <opaque-bytes> <opaque-profile> <output> [base-catalog]".into());
    }
    let mut catalog = provider_catalog_declaration(&CutlassNvfp4Config {
        opaque_memory_bytes: args[1].parse()?,
        opaque_memory_profile: args[2].clone(),
    })?;
    if let Some(path) = args.get(4) {
        let base: ferrum_types::NativeOperatorProviderCatalog =
            serde_json::from_slice(&std::fs::read(path)?)?;
        base.validate()?;
        if base.schema_version != catalog.schema_version || base.backend != catalog.backend {
            return Err("base catalog schema/backend mismatch".into());
        }
        catalog.providers.extend(base.providers);
        catalog.providers.sort_by(|a, b| {
            (&a.operation_id, &a.provider_id).cmp(&(&b.operation_id, &b.provider_id))
        });
        catalog.validate()?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    file.write_all(&catalog.canonical_json_bytes()?)?;
    file.sync_all()?;
    println!(
        "source declaration only: sha256={}",
        catalog.canonical_sha256()?
    );
    Ok(())
}
