//! CPU diagnostic only. Both paths allocate through the same Host account;
//! source construction and equality checks are outside the timed intervals.
use super::*;

fn scalar_rgb(chunk: &FastH3VideoFrameChunk, host: &HostMemoryAccount) -> ManagedHostBytes {
    let [batch, _, frames, height, width] = chunk.shape;
    let plane = frames * height * width;
    let mut output = host
        .allocate_zeroed(HostMemoryClass::Output, batch * plane * 3)
        .unwrap();
    for b in 0..batch {
        for c in 0..3 {
            for pixel in 0..plane {
                let src = 2 * ((b * 3 + c) * plane + pixel);
                let x = bf16::from_bits(u16::from_le_bytes([
                    chunk.bytes.bytes()[src],
                    chunk.bytes.bytes()[src + 1],
                ]))
                .to_f32();
                assert!(x.is_finite());
                let scaled = x * [0.229f32, 0.224, 0.225][c];
                let shifted = scaled + [0.485f32, 0.456, 0.406][c];
                output.as_mut()[(b * plane + pixel) * 3 + c] =
                    (shifted.clamp(0., 1.) * 255f32).clamp(0., 255.) as u8;
            }
        }
    }
    output.freeze()
}

#[test]
#[ignore = "CPU performance diagnostic; use an optimized profile on the inference host"]
fn rgb_480p_host_transfer_benchmark() {
    let budget = 1 << 30;
    let host = host(budget);
    let shape = [1, 3, 124, 480, 832];
    let mut input = host
        .allocate_zeroed(HostMemoryClass::Output, shape.iter().product::<usize>() * 2)
        .unwrap();
    for (i, word) in input.as_mut().chunks_exact_mut(2).enumerate() {
        let x = ((i % 1301) as f32 - 650.) / 128.;
        word.copy_from_slice(&bf16::from_f32(x).to_bits().to_le_bytes());
    }
    let chunk = FastH3VideoFrameChunk {
        start: 0,
        shape,
        bytes: input.freeze(),
    };
    let mut measurements = Vec::new();
    // Alternate order so neither path always receives the first allocation.
    for repeat in 0..4 {
        let mut outputs = Vec::new();
        let order = if repeat % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for table in order {
            let start = std::time::Instant::now();
            let output = if table {
                chunk.managed_rgb8(&host, budget).unwrap().bytes
            } else {
                scalar_rgb(&chunk, &host)
            };
            measurements.push(serde_json::json!({
                "repeat": repeat, "lookup": table, "seconds": start.elapsed().as_secs_f64()
            }));
            outputs.push(output);
        }
        assert_eq!(outputs[0].bytes(), outputs[1].bytes());
    }
    eprintln!(
        "{}",
        serde_json::json!({
            "scope": "CPU original BF16 to RGB24 transfer only, no inference claim",
            "shape": shape, "measurements": measurements
        })
    );
    drop(chunk);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}
