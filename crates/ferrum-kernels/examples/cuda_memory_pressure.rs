//! Hold external device memory while exercising a server's live capacity policy.
//! Usage: cuda_memory_pressure <device ordinal> <bytes> <seconds>.
//! The allocation retires automatically on normal exit; no server is restarted.

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::CudaContext;
    use std::{io::Write, time::Duration};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("expected device ordinal, allocation bytes, and hold seconds".into());
    }
    let ordinal: usize = args[0].parse()?;
    let bytes: usize = args[1].parse()?;
    let seconds: u64 = args[2].parse()?;
    if bytes == 0 || seconds == 0 {
        return Err("allocation bytes and hold seconds must be positive".into());
    }
    let context = CudaContext::new(ordinal)?;
    let stream = context.default_stream();
    let allocation = stream.alloc_zeros::<u8>(bytes)?;
    stream.synchronize()?;
    println!("Holding {bytes} device bytes on ordinal {ordinal} for {seconds} seconds");
    std::io::stdout().flush()?;
    std::thread::sleep(Duration::from_secs(seconds));
    drop(allocation);
    stream.synchronize()?;
    println!("Device memory released");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("cuda_memory_pressure requires the cuda feature");
    std::process::exit(1);
}
