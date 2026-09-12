use super::tests::{bytes, overwrite, region};
use super::*;
use crate::backend::metal::vnext_ops::linear::microbench::{weights, Shape};
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::DeviceId;
use half::f16;
use metal::MTLCommandBufferStatus;

#[test]
fn staged_gated_delta_projections_preserve_offsets_fallback_and_workspace_reuse() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("device.gated-delta.staging").unwrap())
            .unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    let hidden = 1024_u64;
    let leaves = [
        (2048_u32, GgufBlockFormat::Q5K, LinearPhysicalFormat::Q5K),
        (1024, GgufBlockFormat::Q4K, LinearPhysicalFormat::Q4K),
        (8, GgufBlockFormat::Q8_0, LinearPhysicalFormat::Q8_0),
        (8, GgufBlockFormat::Q8_0, LinearPhysicalFormat::Q8_0),
    ];
    let output_width = leaves
        .iter()
        .map(|(width, _, _)| u64::from(*width))
        .sum::<u64>();
    let guard = f16::from_f32(123.0);
    let staging_bytes = hidden * u64::from(leaves[0].0) * 2;
    // 767/768 straddle the candidate policy; decode remains on the old route.
    for rows in [1_u64, 767, 768] {
        let normalized_offset = 64_u64;
        let output_offset = normalized_offset + rows * hidden * 2;
        let staging_offset = output_offset + rows * output_width * 2;
        let mut initial = vec![guard; ((staging_offset + staging_bytes + 64) / 2) as usize];
        initial[32..(staging_offset / 2) as usize].fill(f16::NAN);
        initial[(staging_offset / 2) as usize..((staging_offset + staging_bytes) / 2) as usize]
            .fill(f16::NAN);
        for index in 0..rows * hidden {
            initial[(normalized_offset / 2 + index) as usize] =
                f16::from_f32((index as f32 * 0.019).sin() * 0.0625);
        }
        let mut regions = vec![region(
            runtime,
            "projection-scratch",
            &initial,
            ElementType::U8,
        )];
        for (index, (width, format, _)) in leaves.iter().enumerate() {
            let values = weights(Shape {
                name: "gated_delta_leaf",
                input: hidden as u32,
                output: *width,
                format: *format,
            });
            regions.push(region(
                runtime,
                &format!("projection-weight-{index}"),
                &values,
                ElementType::U8,
            ));
        }
        let immutable_weights = regions[1..].iter().map(bytes).collect::<Vec<_>>();
        let mut column_offset = 0;
        let launches = leaves
            .iter()
            .enumerate()
            .map(|(index, (width, _, format))| {
                let launch = linear_launch(
                    PreparedLinearPart {
                        region: index + 1,
                        format: *format,
                        output_offset: column_offset,
                        out_features: *width,
                    },
                    0,
                    0,
                    rows,
                    hidden,
                    output_width,
                    normalized_offset,
                    output_offset,
                )
                .unwrap();
                column_offset += *width;
                launch
            })
            .collect::<Vec<_>>();
        validate_launch_regions_with_raw_workspace(&regions, &launches, &[0]).unwrap();
        let workspace = Workspace::with_policy(
            &regions,
            0,
            staging_offset,
            staging_bytes,
            launches.iter().copied(),
            StagingPolicy::GatedDelta,
        )
        .unwrap();
        if rows == 768 {
            for (offset, size) in [
                (staging_offset, staging_bytes - 2),
                (staging_offset - 2, staging_bytes),
                (staging_offset, staging_bytes + 66),
            ] {
                assert!(Workspace::with_policy(
                    &regions,
                    0,
                    offset,
                    size,
                    launches.iter().copied(),
                    StagingPolicy::GatedDelta
                )
                .is_err());
            }
            assert_eq!(bytes(&regions[0]), half_bytes(&initial));
        }
        let mut baseline: Option<Vec<u8>> = None;
        for candidate in [false, true] {
            overwrite(&regions[0], &initial);
            let active = if candidate { workspace } else { None };
            assert_eq!(
                launches
                    .iter()
                    .map(|launch| dispatch_count(*launch, active))
                    .sum::<u64>(),
                4 + if candidate && rows >= 768 { 2 } else { 0 }
            );
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            for launch in &launches {
                dispatch(&pipelines, encoder, &regions, *launch, active);
            }
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let actual = bytes(&regions[0]);
            assert_eq!(
                &actual[..output_offset as usize],
                &half_bytes(&initial)[..output_offset as usize]
            );
            let output = &actual[output_offset as usize..staging_offset as usize];
            assert!(output
                .chunks_exact(2)
                .all(|pair| f16::from_le_bytes(pair.try_into().unwrap()).is_finite()));
            if let Some(baseline) = &baseline {
                assert_eq!(
                    output,
                    baseline.as_slice(),
                    "all four projection leaves rows={rows}"
                );
            } else {
                baseline = Some(output.to_vec());
            }
            assert!(actual[(staging_offset + staging_bytes) as usize..]
                .chunks_exact(2)
                .all(|pair| pair == guard.to_le_bytes()));
            let staged =
                &actual[staging_offset as usize..(staging_offset + staging_bytes) as usize];
            assert!(staged.chunks_exact(2).all(|pair| {
                let value = f16::from_le_bytes(pair.try_into().unwrap());
                if candidate && rows >= 768 {
                    value.is_finite()
                } else {
                    value.is_nan()
                }
            }));
        }
        assert_eq!(
            regions[1..].iter().map(bytes).collect::<Vec<_>>(),
            immutable_weights
        );
    }
}

fn half_bytes(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn gated_delta_staging_does_not_enable_new_swiglu_or_f32_routes() {
    let launch = LinearLaunch {
        input_region: 0,
        weight_region: 1,
        output_region: 2,
        input_offset_bytes: 0,
        output_offset_bytes: 0,
        activation_type: ElementType::F16,
        format: LinearPhysicalFormat::Q5K,
        params: LinearParams {
            rows: 768,
            in_features: 4096,
            out_features: 8192,
            output_stride: 12352,
            output_column_offset: 0,
        },
    };
    assert!(selected_for(launch, StagingPolicy::GatedDelta));
    assert!(
        !selected(launch),
        "Q5 support must not expand the SwiGLU route"
    );
    assert!(!selected_for(
        LinearLaunch {
            activation_type: ElementType::F32,
            ..launch
        },
        StagingPolicy::GatedDelta
    ));
    assert!(!selected_for(
        LinearLaunch {
            format: LinearPhysicalFormat::Q6K,
            ..launch
        },
        StagingPolicy::GatedDelta
    ));
}
