use super::super::test_support::Guarded;
use super::*;
use cudarc::driver::CudaContext;
use half::f16;

#[test]
#[ignore = "requires an actual CUDA device"]
fn segmented_projection_stitch_preserves_rows_offsets_and_unwritten_columns_on_cuda() {
    let context = CudaContext::new(0).expect("projection stitching requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::LINEAR_ATTENTION))
        .unwrap();
    let function = module.load_function(PROJECTION_STITCH_FUNCTION).unwrap();
    for rows in [1, 3, 33] {
        // A quantized-width segment and an unaligned floating segment. Leave
        // holes on each row so a flat copy or incorrect pitch cannot pass.
        let width = 209;
        let sentinel = f16::from_f32(-97.0);
        let mut expected = vec![sentinel; rows * width];
        let output = Guarded::new(&stream, &expected, f16::from_f32(-117.0));
        for (offset, part_width) in [(5, 128), (139, 17)] {
            let values = (0..rows * part_width)
                .map(|i| f16::from_f32(((i * 13 + offset * 3) % 997) as f32 / 8.0 - 61.0))
                .collect::<Vec<_>>();
            let input = Guarded::new(&stream, &values, f16::from_f32(-119.0));
            launch_projection_stitch(
                &stream,
                &function,
                input.pointer(&stream),
                output.pointer(&stream),
                rows as i32,
                part_width as i32,
                width as i32,
                offset as i32,
                "stitch conformance",
            )
            .unwrap();
            for row in 0..rows {
                expected[row * width + offset..][..part_width]
                    .copy_from_slice(&values[row * part_width..][..part_width]);
            }
            assert_eq!(output.read(&stream), expected);
            input.assert_unchanged(&stream);
        }
    }
}
