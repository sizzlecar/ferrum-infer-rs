use super::{arguments::Arguments, fixture, Result};
use cudarc::driver::{result, sys, CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr};
use ferrum_kernels::native_ops::{
    compiled_native_operator_artifacts,
    cutlass_fp8::{
        Bindings, Device, DeviceSpan, PreparedCutlassFp8, Request, CUTLASS_FP8_OPERATOR,
    },
};
use half::bf16;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{io::Write, mem::ManuallyDrop, sync::Arc, time::Instant};

// Raw native calls do not update cudarc slice tracking. Everything uses this
// one stream. Even an enqueue error retains the entire owner until its fence;
// a failed fence intentionally leaks ownership until process/context teardown.
struct Owners {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    buffers: ManuallyDrop<Vec<CudaSlice<u8>>>,
    events: ManuallyDrop<Vec<CudaEvent>>,
    requested_payload_bytes: u64,
}

impl Drop for Owners {
    fn drop(&mut self) {
        match self.stream.synchronize() {
            Ok(()) => unsafe {
                ManuallyDrop::drop(&mut self.events);
                ManuallyDrop::drop(&mut self.buffers);
            },
            Err(error) => {
                eprintln!("CUDA fence failed; retaining diagnostic allocations/events: {error}");
            }
        }
    }
}

impl Owners {
    fn allocate(&mut self, bytes: usize) -> Result<usize> {
        let index = self.buffers.len();
        // No uninitialized value is read: setup writes input/weights/scales,
        // native quantization overwrites scratch and GEMM overwrites outputs.
        self.buffers
            .push(unsafe { self.stream.alloc::<u8>(bytes)? });
        self.requested_payload_bytes = self
            .requested_payload_bytes
            .checked_add(bytes as u64)
            .ok_or("allocation byte counter overflow")?;
        self.stream.synchronize()?;
        Ok(index)
    }

    fn span(&self, index: usize) -> DeviceSpan {
        let buffer = &self.buffers[index];
        let (address, _usage) = buffer.device_ptr(&self.stream);
        DeviceSpan {
            address,
            bytes: buffer.len() as u64,
        }
    }

    fn upload(&mut self, bytes: &[u8]) -> Result<usize> {
        let index = self.allocate(bytes.len())?;
        self.context.bind_to_thread()?;
        // Setup uses blocking copies so a borrowed Host payload is never
        // released while an asynchronous upload may still reference it.
        unsafe { result::memcpy_htod_sync(self.span(index).address, bytes)? };
        Ok(index)
    }

    fn read(&self, index: usize) -> Result<Vec<u8>> {
        self.stream.synchronize()?;
        self.context.bind_to_thread()?;
        let span = self.span(index);
        let mut bytes = vec![0; span.bytes.try_into()?];
        unsafe { result::memcpy_dtoh_sync(&mut bytes, span.address)? };
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Variant {
    Separate,
    Shared,
}

#[derive(Clone, Debug, Serialize)]
struct Sample {
    iteration: u32,
    position: usize,
    variant: Variant,
    device_event_ms: f32,
    enqueue_and_wait_us: u128,
}

fn emit(writer: &mut impl Write, value: serde_json::Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, &value)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn enqueue(
    owners: &Owners,
    plan: &PreparedCutlassFp8,
    bindings: &[Bindings],
    variant: Variant,
) -> Result<()> {
    owners.context.bind_to_thread()?;
    // All spans, native context, and stream remain in Owners until completion.
    // Failure is not treated as evidence that nothing was enqueued.
    unsafe {
        match variant {
            Variant::Separate => {
                for binding in bindings {
                    plan.run(binding, owners.stream.cu_stream().cast())?;
                }
            }
            Variant::Shared => plan.run_group(bindings, owners.stream.cu_stream().cast())?,
        }
    }
    Ok(())
}

fn sample(
    owners: &Owners,
    plan: &PreparedCutlassFp8,
    bindings: &[Bindings],
    variant: Variant,
    iteration: u32,
    position: usize,
) -> Result<Sample> {
    owners.events[0].record(&owners.stream)?;
    let started = Instant::now();
    enqueue(owners, plan, bindings, variant)?;
    owners.events[1].record(&owners.stream)?;
    owners.events[1].synchronize()?;
    let enqueue_and_wait_us = started.elapsed().as_micros();
    let device_event_ms = owners.events[0].elapsed_ms(&owners.events[1])?;
    if !device_event_ms.is_finite() || device_event_ms < 0.0 {
        return Err("invalid CUDA event duration".into());
    }
    Ok(Sample {
        iteration,
        position,
        variant,
        device_event_ms,
        enqueue_and_wait_us,
    })
}

fn check_finite_and_zero_row(bytes: &[u8], columns: u32) -> Result<u64> {
    let mut nonzero = 0;
    for (index, element) in bytes.chunks_exact(2).enumerate() {
        let value = bf16::from_bits(u16::from_le_bytes([element[0], element[1]])).to_f32();
        if !value.is_finite() {
            return Err(format!("nonfinite output element {index}").into());
        }
        if index < columns as usize && value != 0.0 {
            return Err(format!("zero-input row produced nonzero element {index}").into());
        }
        nonzero += u64::from(value != 0.0);
    }
    if nonzero == 0 {
        return Err("fixture produced an entirely zero output".into());
    }
    Ok(nonzero)
}

fn read_baseline(owners: &Owners, outputs: &[usize], args: &Arguments) -> Result<Vec<Vec<u8>>> {
    let mut baseline = Vec::with_capacity(outputs.len());
    for &index in outputs {
        let bytes = owners.read(index)?;
        check_finite_and_zero_row(&bytes, args.n)?;
        baseline.push(bytes);
    }
    if baseline.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("distinct fixture branches unexpectedly have equal outputs".into());
    }
    Ok(baseline)
}

fn verify(
    owners: &Owners,
    outputs: &[usize],
    baseline: &[Vec<u8>],
    args: &Arguments,
) -> Result<()> {
    for (branch, (&index, expected)) in outputs.iter().zip(baseline).enumerate() {
        let actual = owners.read(index)?;
        check_finite_and_zero_row(&actual, args.n)?;
        if actual != *expected {
            let first = actual.iter().zip(expected).position(|(a, b)| a != b);
            return Err(format!("output differs in branch {branch}, first byte {first:?}").into());
        }
    }
    Ok(())
}

pub fn run(args: &Arguments, writer: &mut impl Write) -> Result<()> {
    let extents = args.extents()?;
    let context = CudaContext::new(args.device)?;
    // A single manually fenced stream owns every allocation and native launch.
    // No slice or event escapes Owners, including an unsuccessful submission.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream()?;
    let mut owners = Owners {
        context,
        stream,
        buffers: ManuallyDrop::new(Vec::new()),
        events: ManuallyDrop::new(Vec::new()),
        requested_payload_bytes: 0,
    };
    use sys::CUdevice_attribute::*;
    let attribute = |key| -> Result<u32> { Ok(owners.context.attribute(key)?.try_into()?) };
    let device = Device {
        ordinal: args.device.try_into()?,
        architecture: attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? * 100
            + attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)? * 10,
        multiprocessors: attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        maximum_dynamic_shared_bytes: attribute(
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )?
        .into(),
    };
    owners.context.bind_to_thread()?;
    let plan = PreparedCutlassFp8::new(Request::new(args.m, args.k, args.n, device)?)?;
    let payload_bytes = args.device_payload_bytes(plan.scratch_bytes())?;
    let required = payload_bytes
        .checked_add(args.native_reserve_bytes)
        .ok_or("device budget overflow")?;
    let (free_before, total) = owners.context.mem_get_info()?;
    if required > args.device_bytes || required > free_before as u64 {
        return Err(format!("payload {payload_bytes} + native/context reserve {} = {required}; budget {}, observed free {free_before}", args.native_reserve_bytes, args.device_bytes).into());
    }
    let uuid = owners
        .context
        .uuid()?
        .bytes
        .iter()
        .map(|b| format!("{:02x}", *b as u8))
        .collect::<String>();
    emit(
        writer,
        json!({
            "phase":"declaration", "schema":"ferrum.fp8_shared_quant_micro.v1",
            "arguments":args, "gpu_name":owners.context.name()?, "gpu_uuid_hex":uuid,
            "architecture":device.architecture, "multiprocessors":device.multiprocessors,
            "observed_free_before_allocations":free_before, "device_total_bytes":total,
            "extents":extents, "scratch_bytes":plan.scratch_bytes(),
            "planned_owned_device_payload_bytes":payload_bytes,
            "device_budget_including_declared_reserve_bytes":args.device_bytes,
            "native_context_reserve_bytes":args.native_reserve_bytes,
            "native_artifacts":compiled_native_operator_artifacts().iter().filter(|x| x.operator==CUTLASS_FP8_OPERATOR).collect::<Vec<_>>(),
            "fixture":"finite-bf16-row-ranges-and-zero-rows/independent-e4m3-weights-and-scales.v1",
            "schedule":"warmup A/B; each measured iteration A/B/B/A; full validation before and after",
            "measurement":"CUDA event interval brackets native calls and can include host enqueue gaps; wall includes enqueue and fence wait. Upload/readback, fixture generation, validation and logging are excluded. Optional inner phase events are a separate diagnostic, not ABBA measurements.",
            "memory_scope":"Explicit diagnostic-owned requested payload bytes, not Ferrum runtime admission or allocator-reserved/whole-GPU peak. Native/context reserve is an allowance, not a measured peak; free memory is a snapshot. Host bound covers tensor vectors, not process RSS.",
            "not_measured":["image throughput", "HTTP latency", "latency SLO", "model quality", "peak VRAM", "peak process RSS"]
        }),
    )?;

    let input = owners.upload(&fixture::input(args.m, args.k))?;
    let scratch = owners.allocate(plan.scratch_bytes().try_into()?)?;
    let mut outputs = Vec::new();
    let mut bindings = Vec::new();
    for branch in 0..args.branches {
        let weight = owners.upload(&fixture::weight(args.n, args.k, branch))?;
        let scale = owners.upload(&fixture::scale(branch))?;
        let output = owners.allocate(extents.output)?;
        outputs.push(output);
        bindings.push(Bindings {
            input: owners.span(input),
            weight: owners.span(weight),
            weight_scale: owners.span(scale),
            output: owners.span(output),
            scratch: owners.span(scratch),
        });
    }
    if owners.requested_payload_bytes != payload_bytes {
        return Err("owned allocations differ from declared payload".into());
    }
    plan.validate_group(&bindings)?;
    for binding in &bindings {
        plan.validate(binding)?;
    }
    for _ in 0..if args.profile_phases { 5 } else { 2 } {
        owners.events.push(
            owners
                .context
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?,
        );
    }
    enqueue(&owners, &plan, &bindings, Variant::Separate)?;
    let baseline = read_baseline(&owners, &outputs, args)?;
    enqueue(&owners, &plan, &bindings, Variant::Shared)?;
    verify(&owners, &outputs, &baseline, args)?;
    let hashes = baseline
        .iter()
        .map(|v| format!("{:x}", Sha256::digest(v)))
        .collect::<Vec<_>>();
    emit(
        writer,
        json!({"phase":"initial_validation","full_output_bitwise_equal":true,"finite":true,"zero_row_oracle":true,"output_sha256":hashes,"elements_compared":u64::from(args.m)*u64::from(args.n)*args.branches as u64}),
    )?;

    for _ in 0..args.warmup {
        for variant in [Variant::Separate, Variant::Shared] {
            enqueue(&owners, &plan, &bindings, variant)?;
            owners.stream.synchronize()?;
        }
    }
    let mut samples = Vec::new();
    for iteration in 0..args.iterations {
        for (position, variant) in [
            Variant::Separate,
            Variant::Shared,
            Variant::Shared,
            Variant::Separate,
        ]
        .into_iter()
        .enumerate()
        {
            let sample = sample(&owners, &plan, &bindings, variant, iteration, position)?;
            emit(writer, json!({"phase":"measured","sample":sample}))?;
            samples.push(sample);
        }
    }
    if args.profile_phases {
        for iteration in 0..args.iterations {
            // CUDA Driver/Runtime event handles share the same CUDA context.
            // Native code only records these retained caller-owned events.
            unsafe {
                plan.run_group_profiled(
                    &bindings,
                    owners.stream.cu_stream().cast(),
                    [
                        owners.events[2].cu_event().cast(),
                        owners.events[3].cu_event().cast(),
                        owners.events[4].cu_event().cast(),
                    ],
                )?;
            }
            owners.events[4].synchronize()?;
            let quant_ms = owners.events[2].elapsed_ms(&owners.events[3])?;
            let gemms_ms = owners.events[3].elapsed_ms(&owners.events[4])?;
            emit(
                writer,
                json!({"phase":"group_phase_diagnostic","iteration":iteration,"quantization_ms":quant_ms,"all_gemms_ms":gemms_ms,"scope":"Inner event overhead and any host enqueue gaps are included. Separate diagnostic, excluded from ABBA aggregate; not pure kernel-active timing."}),
            )?;
        }
    }
    for variant in [Variant::Separate, Variant::Shared] {
        enqueue(&owners, &plan, &bindings, variant)?;
        verify(&owners, &outputs, &baseline, args)?;
    }
    let mean = |variant| {
        let rows = samples
            .iter()
            .filter(|x| x.variant == variant)
            .collect::<Vec<_>>();
        rows.iter()
            .map(|x| f64::from(x.device_event_ms))
            .sum::<f64>()
            / rows.len() as f64
    };
    let separate_ms = mean(Variant::Separate);
    let shared_ms = mean(Variant::Shared);
    let (free_after, _) = owners.context.mem_get_info()?;
    owners.stream.synchronize()?;
    owners.context.check_err()?;
    emit(
        writer,
        json!({
            "phase":"summary", "full_output_bitwise_equal":true, "finite":true,
            "validation":"All elements compared before and after timing; zero-input row independently checked. This is parity against the existing scalar operator, not a checkpoint/image-quality oracle.",
            "samples_per_variant":u64::from(args.iterations)*2,
            "separate_mean_device_event_ms":separate_ms,
            "shared_mean_device_event_ms":shared_ms,
            "shared_duration_change_percent":(shared_ms/separate_ms-1.0)*100.0,
            "owned_device_payload_bytes":owners.requested_payload_bytes,
            "host_tensor_payload_peak_bound_bytes":extents.host_payload_peak_bound,
            "observed_free_after_measurement":free_after,
            "peak_vram_measured":false, "peak_rss_measured":false,
            "service_or_end_to_end_performance_claim":false
        }),
    )?;
    Ok(())
}
