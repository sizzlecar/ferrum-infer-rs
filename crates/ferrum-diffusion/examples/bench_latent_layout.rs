//! CPU diagnostic for opaque patch unpacking, without model execution.
//! Run optimized: `cargo run --release -p ferrum-diffusion --example bench_latent_layout`
//! Optional positional arguments: N C T H W PT PH PW REPETITIONS.
use ferrum_diffusion::SpatiotemporalLatentShape;
use std::{error::Error, hint::black_box, time::Instant};

fn copy_elements(shape: SpatiotemporalLatentShape, input: &[u8], output: &mut [u8]) {
    shape.for_each_packed_index(|p, d| {
        output[2 * d..2 * d + 2].copy_from_slice(&input[2 * p..2 * p + 2]);
    });
}

fn copy_spans(shape: SpatiotemporalLatentShape, input: &[u8], output: &mut [u8]) {
    shape.for_each_packed_span(|p, d| {
        output[2 * d.start..2 * d.end].copy_from_slice(&input[2 * p.start..2 * p.end]);
    });
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args()
        .skip(1)
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let values: [usize; 9] = if args.is_empty() {
        // One production spatial decoder tile, including its temporal halo.
        [1, 3, 28, 256, 256, 4, 16, 16, 10]
    } else {
        args.try_into()
            .map_err(|_| "expected N C T H W PT PH PW REPETITIONS")?
    };
    let [n, c, t, h, w, pt, ph, pw, repeats] = values;
    if repeats == 0 {
        return Err("repetitions must be positive".into());
    }
    let shape = SpatiotemporalLatentShape::new([n, c, t, h, w], [pt, ph, pw])?;
    let bytes = shape.byte_len::<u16>()?;
    let input: Vec<u8> = (0..bytes).map(|i| (i ^ (i >> 8)) as u8).collect();
    let mut elements = vec![0; bytes];
    let mut spans = vec![0; bytes];
    copy_elements(shape, &input, &mut elements);
    copy_spans(shape, &input, &mut spans);
    assert_eq!(elements, spans, "opaque layout copies differ");
    eprintln!(
        "CPU layout diagnostic only; dense={:?}, patch={:?}, bytes={bytes}; preallocated buffers, one warmup per method, alternating order",
        shape.ncthw_dims(), shape.patch_dims()
    );
    println!("repeat,method,elapsed_ns");
    for repeat in 0..repeats {
        for span_first in [repeat % 2 == 0, repeat % 2 != 0] {
            let (name, output, copy): (_, _, fn(_, _, _)) = if span_first {
                ("spans", &mut spans, copy_spans)
            } else {
                ("elements", &mut elements, copy_elements)
            };
            let started = Instant::now();
            copy(black_box(shape), black_box(&input), black_box(output));
            let elapsed = started.elapsed();
            println!("{repeat},{name},{}", elapsed.as_nanos());
        }
    }
    assert_eq!(elements, spans, "repeated opaque layout copies differ");
    black_box((elements, spans));
    Ok(())
}
