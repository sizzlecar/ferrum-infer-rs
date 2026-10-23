//! CPU-only source declaration for first native package construction.
use ferrum_kernels::native_ops::cutlass_fp8::{
    provider_catalog_declaration, CutlassFp8CatalogConfig,
};
use std::{error::Error, fs::OpenOptions, io::Write, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if !(5..=6).contains(&args.len()) {
        return Err("usage: cutlass_fp8_catalog_declaration <opaque-bytes> <maximum-plans> <opaque-profile> <catalog-output> [base-catalog]".into());
    }
    let config = CutlassFp8CatalogConfig {
        opaque_memory_bytes: args[1].parse()?,
        maximum_plans: args[2].parse()?,
        opaque_memory_profile: args[3].clone(),
    };
    let mut catalog = provider_catalog_declaration(&config)?;
    if let Some(path) = args.get(5) {
        let base: ferrum_types::NativeOperatorProviderCatalog =
            serde_json::from_slice(&std::fs::read(path)?)?;
        base.validate()?;
        if base.schema_version != catalog.schema_version || base.backend != catalog.backend {
            return Err("base catalog schema/backend differs from FP8 declaration".into());
        }
        catalog.providers.extend(base.providers);
        catalog.providers.sort_by(|a, b| {
            (&a.operation_id, &a.provider_id).cmp(&(&b.operation_id, &b.provider_id))
        });
        catalog.validate()?;
    }
    let bytes = catalog.canonical_json_bytes()?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(Path::new(&args[4]))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    println!(
        "source declaration only: sha256={} output={}",
        catalog.canonical_sha256()?,
        args[4]
    );
    Ok(())
}
