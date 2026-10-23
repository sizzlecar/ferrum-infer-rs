//! Hand-written narrow constants/prototypes, checked against deployed NVIDIA
//! cuDNN 9.20.0 cudnn_graph.h (R52 header receipt). No copied header, C++
//! frontend, native allocator or timing/autotuning call is required.

use super::super::ffi::{check, Descriptor, GraphApi, Handle, SUCCESS};
use super::{CudnnSdpaBindings, CudnnSdpaSignature};
use std::ffi::{c_int, c_void};

const ENGINE: c_int = 2;
const ENGINE_CONFIG: c_int = 3;
const HEURISTICS: c_int = 4;
const PLAN: c_int = 5;
const KNOB_CHOICE: c_int = 7;
const GRAPH: c_int = 15;
const VARIANT_PACK: c_int = 16;
const TENSOR: c_int = 17;
const SDPA_FORWARD: c_int = 41;
const TYPE_HANDLE: c_int = 0;
const TYPE_DATA: c_int = 1;
const TYPE_BOOL: c_int = 2;
const TYPE_I64: c_int = 3;
const TYPE_POINTER: c_int = 6;
const TYPE_HEURISTIC: c_int = 8;
const TYPE_KNOB: c_int = 9;
const TYPE_DESCRIPTOR: c_int = 15;
const MAX_CANDIDATES: usize = 16;
const MAX_KNOBS: usize = 32;
const UIDS: [i64; 7] = [1, 2, 3, 4, 5, 6, 7];

pub(super) struct BuildError {
    pub message: String,
    pub cleanup_known: bool,
}

pub(super) struct NativePlan {
    pub descriptor: Descriptor,
    pub workspace_bytes: u64,
    pub identity: String,
    descriptors: Vec<Descriptor>,
}
pub(super) struct VariantPack {
    pub descriptor: Descriptor,
    descriptors: Vec<Descriptor>,
}

struct Builder<'a> {
    api: &'a GraphApi,
    descriptors: Vec<Descriptor>,
    // Diagnostic metadata only; native ownership remains in `descriptors`.
    descriptor_kinds: Vec<(Descriptor, c_int)>,
}
impl<'a> Builder<'a> {
    fn new(api: &'a GraphApi) -> Self {
        Self {
            api,
            descriptors: Vec::new(),
            descriptor_kinds: Vec::new(),
        }
    }
    fn create(&mut self, kind: c_int) -> Result<Descriptor, String> {
        let mut descriptor = std::ptr::null_mut();
        let result = unsafe { (self.api.create)(kind, &mut descriptor) };
        // Even a failed native call may have returned a partially-created owner.
        if !descriptor.is_null() {
            self.descriptors.push(descriptor);
            self.descriptor_kinds.push((descriptor, kind));
        }
        check(result, "create graph descriptor").map_err(|error| {
            format!("{error}; descriptor_kind={kind} attribute=none type=none count=0")
        })?;
        if descriptor.is_null() {
            return Err(format!("cuDNN returned a null graph descriptor; descriptor_kind={kind} attribute=none type=none count=0"));
        }
        Ok(descriptor)
    }
    fn kind(&self, descriptor: Descriptor) -> Option<c_int> {
        self.descriptor_kinds
            .iter()
            .find_map(|&(d, kind)| (d == descriptor).then_some(kind))
    }
    fn set<T: std::fmt::Debug>(
        &self,
        d: Descriptor,
        name: c_int,
        kind: c_int,
        values: &[T],
    ) -> Result<(), String> {
        let count = i64::try_from(values.len()).map_err(|_| "cuDNN attribute count overflows")?;
        unsafe {
            check(
                (self.api.set)(d, name, kind, count, values.as_ptr().cast()),
                "set graph attribute",
            )
        }
        .map_err(|error| {
            let values = if matches!(kind, TYPE_HANDLE | TYPE_POINTER | TYPE_DESCRIPTOR) {
                "<addresses omitted>".to_owned()
            } else {
                format!("{:?}", &values[..values.len().min(8)])
            };
            format!("{error}; descriptor_kind={:?} attribute={name} type={kind} count={count} values={values}", self.kind(d))
        })
    }
    fn finish(&self, d: Descriptor) -> Result<(), String> {
        unsafe { check((self.api.finalize)(d), "finalize graph descriptor") }.map_err(|error| {
            format!(
                "{error}; descriptor_kind={:?} attribute=none type=none count=0",
                self.kind(d)
            )
        })
    }
    fn get<T>(
        &self,
        d: Descriptor,
        name: c_int,
        kind: c_int,
        values: &mut [T],
    ) -> Result<usize, String> {
        let requested =
            i64::try_from(values.len()).map_err(|_| "cuDNN attribute count overflows")?;
        let mut count = 0_i64;
        let pointer = if values.is_empty() {
            std::ptr::null_mut()
        } else {
            values.as_mut_ptr().cast()
        };
        unsafe {
            check(
                (self.api.get)(d, name, kind, requested, &mut count, pointer),
                "get graph attribute",
            )
            .map_err(|error| {
                format!(
                    "{error}; descriptor_kind={:?} attribute={name} type={kind} count={requested}",
                    self.kind(d)
                )
            })?;
        }
        usize::try_from(count).map_err(|_| format!("cuDNN returned negative count={count}; descriptor_kind={:?} attribute={name} type={kind} requested={requested}", self.kind(d)))
    }
    fn i64(&self, d: Descriptor, name: c_int) -> Result<i64, String> {
        let mut value = [0_i64];
        let count = self.get(d, name, TYPE_I64, &mut value)?;
        if count != 1 {
            return Err(format!("cuDNN scalar attribute count is not one; descriptor_kind={:?} attribute={name} type={TYPE_I64} count={count} requested=1", self.kind(d)));
        }
        Ok(value[0])
    }
    fn tensor(
        &mut self,
        uid: i64,
        dtype: c_int,
        dims: [i64; 4],
        strides: [i64; 4],
        alignment: i64,
        by_value: bool,
    ) -> Result<Descriptor, String> {
        (|| {
            let d = self.create(TENSOR)?;
            self.set(d, 900, TYPE_I64, &[alignment])?;
            self.set(d, 901, TYPE_DATA, &[dtype])?;
            self.set(d, 902, TYPE_I64, &dims)?;
            self.set(d, 903, TYPE_I64, &strides)?;
            self.set(d, 906, TYPE_I64, &[uid])?;
            self.set(d, 907, TYPE_BOOL, &[false])?;
            if by_value {
                self.set(d, 908, TYPE_BOOL, &[true])?;
            }
            self.finish(d)?;
            Ok(d)
        })()
        .map_err(|error: String| format!("{error}; tensor_uid={uid} dtype={dtype} dims={dims:?} strides={strides:?} alignment={alignment} by_value={by_value}"))
    }
    fn cleanup(&mut self) -> bool {
        destroy_descriptors(self.api, &mut self.descriptors)
    }
}

impl NativePlan {
    pub(super) fn build(
        api: &GraphApi,
        handle: Handle,
        s: CudnnSdpaSignature,
        cap: u64,
    ) -> Result<Self, BuildError> {
        let mut b = Builder::new(api);
        let built = (|| -> Result<(Descriptor, u64, String), String> {
            s.validate()?;
            let row = s
                .heads
                .checked_mul(s.head_dim)
                .ok_or("SDPA row overflows")?;
            let qdim = [s.batch, s.heads, s.query_length, s.head_dim];
            let kvdim = [s.batch, s.heads, s.key_value_length, s.head_dim];
            let q = b.tensor(
                1,
                9,
                qdim,
                [s.query_batch_stride, s.head_dim, row, 1],
                16,
                false,
            )?;
            let k = b.tensor(
                2,
                9,
                kvdim,
                [s.key_value_batch_stride, s.head_dim, row, 1],
                16,
                false,
            )?;
            let v = b.tensor(
                3,
                9,
                kvdim,
                [s.key_value_batch_stride, s.head_dim, row, 1],
                16,
                false,
            )?;
            let o = b.tensor(
                4,
                9,
                qdim,
                [s.output_batch_stride, s.head_dim, row, 1],
                16,
                false,
            )?;
            let scale = b.tensor(5, 0, [1; 4], [1; 4], 4, true)?;
            let qlen = b.tensor(6, 4, [s.batch, 1, 1, 1], [1; 4], 4, false)?;
            let kvlen = b.tensor(7, 4, [s.batch, 1, 1, 1], [1; 4], 4, false)?;
            let op = b.create(SDPA_FORWARD)?;
            for (name, value) in [
                (2800, q),
                (2801, k),
                (2802, v),
                (2803, o),
                (2805, scale),
                (2809, qlen),
                (2810, kvlen),
            ] {
                b.set(op, name, TYPE_DESCRIPTOR, &[value])?;
            }
            b.finish(op)?;
            let graph = b.create(GRAPH)?;
            // These attributes are deprecated but still present in the pinned
            // 9.20 ABI. They bind plans to the already-admitted context/handle.
            b.set(graph, 800, TYPE_HANDLE, &[handle])?;
            b.set(graph, 801, TYPE_DESCRIPTOR, &[op])?;
            b.finish(graph)?;
            let mut mode_failures = Vec::with_capacity(3);
            // Ranked metadata only; no candidate is executed or timed. The
            // bounded result prefix keeps native graph construction finite.
            for mode in [3 as c_int, 2, 1] {
                let mut summary = "heuristic setup".to_owned();
                let attempt = (|| -> Result<Option<(Descriptor, u64, String)>, String> {
                    let heuristic = b.create(HEURISTICS)?;
                    b.set(heuristic, 201, TYPE_DESCRIPTOR, &[graph])?;
                    b.set(heuristic, 200, TYPE_HEURISTIC, &[mode])?;
                    if let Err(error) = b.finish(heuristic) {
                        summary = error;
                        return Ok(None);
                    }
                    let count = b
                        .get::<Descriptor>(heuristic, 202, TYPE_DESCRIPTOR, &mut [])?
                        .min(MAX_CANDIDATES);
                    let mut candidates = Vec::with_capacity(count);
                    for _ in 0..count {
                        candidates.push(b.create(ENGINE_CONFIG)?);
                    }
                    summary = format!("candidates={count}");
                    if count == 0 {
                        return Ok(None);
                    }
                    let found = b.get(heuristic, 202, TYPE_DESCRIPTOR, &mut candidates)?;
                    // cuDNN can report the full available count when the requested
                    // result prefix is shorter. Only initialized requested slots
                    // are examined.
                    for (ordinal, candidate) in
                        candidates.into_iter().take(found.min(count)).enumerate()
                    {
                        summary = format!("candidates={} candidate={ordinal}", found.min(count));
                        let (identity, engine) = engine_identity(&mut b, candidate)?;
                        let Some(identity) = identity else {
                            summary.push_str(&format!(
                                " engine={engine} declares nondeterministic execution"
                            ));
                            continue;
                        };
                        summary.push_str(&format!(" engine={engine}"));
                        let plan = b.create(PLAN)?;
                        b.set(plan, 400, TYPE_HANDLE, &[handle])?;
                        b.set(plan, 401, TYPE_DESCRIPTOR, &[candidate])?;
                        if let Err(error) = b.finish(plan) {
                            summary.push_str(&format!(" {error}"));
                            continue;
                        }
                        let bytes = b.i64(plan, 402)?;
                        let Ok(workspace) = u64::try_from(bytes) else {
                            return Err(format!("cuDNN returned negative workspace={bytes}; descriptor_kind={PLAN} attribute=402 type={TYPE_I64} count=1"));
                        };
                        if workspace > cap || workspace > isize::MAX as u64 {
                            summary.push_str(&format!(" workspace={workspace} exceeds cap={cap}"));
                            continue;
                        }
                        return Ok(Some((
                            plan,
                            workspace,
                            format!("heuristic={mode};{identity}"),
                        )));
                    }
                    Ok(None)
                })();
                match attempt {
                    Ok(Some(plan)) => return Ok(plan),
                    Ok(None) => mode_failures.push(format!("mode={mode}: {summary}")),
                    Err(error) => {
                        mode_failures.push(format!("mode={mode}: {summary}; {error}"));
                        // Preserve the existing early-failure behavior; only
                        // diagnostic context changes, not candidate selection.
                        return Err(format!(
                            "SDPA preparation failed: {}",
                            mode_failures.join(" | ")
                        ));
                    }
                }
            }
            Err(format!(
                "no supported SDPA engine: {}",
                mode_failures.join(" | ")
            ))
        })();
        match built {
            Ok((descriptor, workspace_bytes, identity)) => Ok(Self {
                descriptor,
                workspace_bytes,
                identity,
                descriptors: b.descriptors,
            }),
            Err(message) => Err(BuildError {
                message,
                cleanup_known: b.cleanup(),
            }),
        }
    }
    pub(super) fn destroy(&mut self, api: &GraphApi) -> bool {
        destroy_descriptors(api, &mut self.descriptors)
    }
}

fn engine_identity(
    b: &mut Builder<'_>,
    candidate: Descriptor,
) -> Result<(Option<String>, i64), String> {
    let engine = b.create(ENGINE)?;
    let engine_count = b.get(candidate, 300, TYPE_DESCRIPTOR, &mut [engine])?;
    if engine_count != 1 {
        return Err(format!("cuDNN engine identity missing; descriptor_kind={ENGINE_CONFIG} attribute=300 type={TYPE_DESCRIPTOR} count={engine_count} requested=1"));
    }
    let index = b.i64(engine, 1301)?;
    // The admitted component policy requires BitwiseSameRuntime. Never rely
    // on the heuristic rank to imply determinism: inspect the selected engine.
    let note_count = b.get::<c_int>(engine, 1303, 11, &mut [])?;
    if note_count > 16 {
        return Err(format!("cuDNN numerical notes exceed the bounded ABI; engine={index} descriptor_kind={ENGINE} attribute=1303 type=11 count={note_count} maximum=16"));
    }
    let mut notes = [0 as c_int; 16];
    if note_count != 0 {
        let actual = b.get(engine, 1303, 11, &mut notes[..note_count])?;
        if actual != note_count {
            return Err(format!("cuDNN numerical note count mismatch; engine={index} descriptor_kind={ENGINE} attribute=1303 type=11 count={actual} requested={note_count}"));
        }
    }
    if notes[..note_count].contains(&4) {
        return Ok((None, index));
    }
    notes[..note_count].sort_unstable();
    let count = b.get::<Descriptor>(candidate, 302, TYPE_DESCRIPTOR, &mut [])?;
    if count > MAX_KNOBS {
        return Err(format!("cuDNN knob choices exceed the bounded ABI; engine={index} descriptor_kind={ENGINE_CONFIG} attribute=302 type={TYPE_DESCRIPTOR} count={count} maximum={MAX_KNOBS}"));
    }
    let mut knobs = Vec::with_capacity(count);
    for _ in 0..count {
        knobs.push(b.create(KNOB_CHOICE)?);
    }
    if count != 0 {
        let actual = b.get(candidate, 302, TYPE_DESCRIPTOR, &mut knobs)?;
        if actual != count {
            return Err(format!("cuDNN knob identity count mismatch; engine={index} descriptor_kind={ENGINE_CONFIG} attribute=302 type={TYPE_DESCRIPTOR} count={actual} requested={count}"));
        }
    }
    let mut choices = Vec::with_capacity(count);
    for knob in knobs {
        let mut kind = [0 as c_int];
        let count = b.get(knob, 600, TYPE_KNOB, &mut kind)?;
        if count != 1 {
            return Err(format!("cuDNN knob identity missing; engine={index} descriptor_kind={KNOB_CHOICE} attribute=600 type={TYPE_KNOB} count={count} requested=1"));
        }
        choices.push((kind[0], b.i64(knob, 601)?));
    }
    choices.sort_unstable();
    Ok((
        Some(format!(
            "engine={index};knobs={choices:?};notes={:?}",
            &notes[..note_count]
        )),
        index,
    ))
}

impl VariantPack {
    pub(super) fn build(
        api: &GraphApi,
        v: CudnnSdpaBindings,
        scale: &f32,
    ) -> Result<Self, BuildError> {
        let mut b = Builder::new(api);
        let built = (|| -> Result<Descriptor, String> {
            let d = b.create(VARIANT_PACK)?;
            let pointers = [
                v.q as *mut c_void,
                v.k as *mut c_void,
                v.v as *mut c_void,
                v.o as *mut c_void,
                std::ptr::from_ref(scale).cast_mut().cast(),
                v.seq_len_q as *mut c_void,
                v.seq_len_kv as *mut c_void,
            ];
            b.set(d, 1000, TYPE_I64, &UIDS)?;
            b.set(d, 1001, TYPE_POINTER, &pointers)?;
            b.set(d, 1003, TYPE_POINTER, &[v.workspace as *mut c_void])?;
            b.finish(d)?;
            Ok(d)
        })();
        match built {
            Ok(descriptor) => Ok(Self {
                descriptor,
                descriptors: b.descriptors,
            }),
            Err(message) => Err(BuildError {
                message,
                cleanup_known: b.cleanup(),
            }),
        }
    }
    pub(super) fn destroy(&mut self, api: &GraphApi) -> bool {
        destroy_descriptors(api, &mut self.descriptors)
    }
}

fn destroy_descriptors(api: &GraphApi, descriptors: &mut Vec<Descriptor>) -> bool {
    while let Some(&descriptor) = descriptors.last() {
        if unsafe { (api.destroy)(descriptor) } != SUCCESS {
            return false;
        }
        descriptors.pop();
    }
    true
}
