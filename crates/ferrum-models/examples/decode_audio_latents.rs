//! Original FastH3 audio VAE -> interleaved stereo IEEE-F32 WAVE artifacts.
#[path = "decode_audio_latents/config.rs"]
mod config;
#[cfg(feature = "cuda")]
#[path = "decode_audio_latents/execution.rs"]
mod execution;
#[path = "support/retained_failure.rs"]
mod retained_failure;
use retained_failure::Failure;
fn main() {
    let result = (|| -> Result<(), Failure> {
        let mut args = std::env::args().skip(1);
        if args.next().as_deref() != Some("--configuration") {
            return Err(Failure::retain(
                "arguments",
                "usage: decode_audio_latents --configuration FILE.json",
                (),
            ));
        }
        let path = args
            .next()
            .ok_or_else(|| Failure::retain("arguments", "missing configuration", ()))?;
        if args.next().is_some() {
            return Err(Failure::retain("arguments", "unexpected argument", ()));
        }
        let c = config::Configuration::load(std::path::Path::new(&path))
            .map_err(|e| Failure::retain("configuration", e, ()))?;
        #[cfg(feature = "cuda")]
        {
            let report = execution::run(c)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).map_err(|e| Failure::retain(
                    "report",
                    e,
                    ()
                ))?
            );
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = c;
            Err(Failure::retain("feature", "requires --features cuda", ()))
        }
    })();
    if let Err(f) = result {
        std::mem::forget(f.owner);
        eprintln!(
            "{}",
            serde_json::json!({"stage":f.stage,"error":f.message,"retirement":"not claimed; original owners retained until process exit"})
        );
        std::process::exit(1);
    }
}
