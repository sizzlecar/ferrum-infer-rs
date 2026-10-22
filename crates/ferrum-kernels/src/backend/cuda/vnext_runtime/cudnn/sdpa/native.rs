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
}
impl<'a> Builder<'a> {
    fn new(api: &'a GraphApi) -> Self {
        Self {
            api,
            descriptors: Vec::new(),
        }
    }
    fn create(&mut self, kind: c_int) -> Result<Descriptor, String> {
        let mut descriptor = std::ptr::null_mut();
        let result = unsafe { (self.api.create)(kind, &mut descriptor) };
        // Even a failed native call may have returned a partially-created owner.
        if !descriptor.is_null() {
            self.descriptors.push(descriptor);
        }
        check(result, "create graph descriptor")?;
        if descriptor.is_null() {
            return Err("cuDNN returned a null graph descriptor".into());
        }
        Ok(descriptor)
    }
    fn set<T>(&self, d: Descriptor, name: c_int, kind: c_int, values: &[T]) -> Result<(), String> {
        let count = i64::try_from(values.len()).map_err(|_| "cuDNN attribute count overflows")?;
        unsafe {
            check(
                (self.api.set)(d, name, kind, count, values.as_ptr().cast()),
                "set graph attribute",
            )
        }
    }
    fn finish(&self, d: Descriptor) -> Result<(), String> {
        unsafe { check((self.api.finalize)(d), "finalize graph descriptor") }
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
            )?;
        }
        usize::try_from(count).map_err(|_| "cuDNN returned a negative attribute count".into())
    }
    fn i64(&self, d: Descriptor, name: c_int) -> Result<i64, String> {
        let mut value = [0_i64];
        if self.get(d, name, TYPE_I64, &mut value)? != 1 {
            return Err("cuDNN scalar attribute count is not one".into());
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
            let mut last_failure = "no supported SDPA engine".to_string();
            // Ranked metadata only; no candidate is executed or timed. The
            // bounded result prefix keeps native graph construction finite.
            for mode in [3 as c_int, 2, 1] {
                let heuristic = b.create(HEURISTICS)?;
                b.set(heuristic, 201, TYPE_DESCRIPTOR, &[graph])?;
                b.set(heuristic, 200, TYPE_HEURISTIC, &[mode])?;
                if let Err(error) = b.finish(heuristic) {
                    last_failure = error;
                    continue;
                }
                let count = b
                    .get::<Descriptor>(heuristic, 202, TYPE_DESCRIPTOR, &mut [])?
                    .min(MAX_CANDIDATES);
                let mut candidates = Vec::with_capacity(count);
                for _ in 0..count {
                    candidates.push(b.create(ENGINE_CONFIG)?);
                }
                if count == 0 {
                    continue;
                }
                let found = b.get(heuristic, 202, TYPE_DESCRIPTOR, &mut candidates)?;
                // cuDNN can report the full available count when the requested
                // result prefix is shorter. Only initialized requested slots
                // are examined.
                for candidate in candidates.into_iter().take(found.min(count)) {
                    let Some(identity) = engine_identity(&mut b, candidate)? else {
                        last_failure =
                            "cuDNN SDPA engine declares nondeterministic execution".into();
                        continue;
                    };
                    let plan = b.create(PLAN)?;
                    b.set(plan, 400, TYPE_HANDLE, &[handle])?;
                    b.set(plan, 401, TYPE_DESCRIPTOR, &[candidate])?;
                    if let Err(error) = b.finish(plan) {
                        last_failure = error;
                        continue;
                    }
                    let bytes = b.i64(plan, 402)?;
                    let Ok(workspace) = u64::try_from(bytes) else {
                        return Err("cuDNN returned negative workspace".into());
                    };
                    if workspace > cap || workspace > isize::MAX as u64 {
                        last_failure = format!("SDPA plan workspace {workspace} exceeds cap {cap}");
                        continue;
                    }
                    return Ok((plan, workspace, format!("heuristic={mode};{identity}")));
                }
            }
            Err(last_failure)
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

fn engine_identity(b: &mut Builder<'_>, candidate: Descriptor) -> Result<Option<String>, String> {
    let engine = b.create(ENGINE)?;
    if b.get(candidate, 300, TYPE_DESCRIPTOR, &mut [engine])? != 1 {
        return Err("cuDNN engine identity missing".into());
    }
    let index = b.i64(engine, 1301)?;
    // The admitted component policy requires BitwiseSameRuntime. Never rely
    // on the heuristic rank to imply determinism: inspect the selected engine.
    let note_count = b.get::<c_int>(engine, 1303, 11, &mut [])?;
    if note_count > 16 {
        return Err("cuDNN numerical notes exceed the bounded ABI".into());
    }
    let mut notes = [0 as c_int; 16];
    if note_count != 0 && b.get(engine, 1303, 11, &mut notes[..note_count])? != note_count {
        return Err("cuDNN numerical note count mismatch".into());
    }
    if notes[..note_count].contains(&4) {
        return Ok(None);
    }
    notes[..note_count].sort_unstable();
    let count = b.get::<Descriptor>(candidate, 302, TYPE_DESCRIPTOR, &mut [])?;
    if count > MAX_KNOBS {
        return Err("cuDNN engine has more knob choices than the bounded ABI".into());
    }
    let mut knobs = Vec::with_capacity(count);
    for _ in 0..count {
        knobs.push(b.create(KNOB_CHOICE)?);
    }
    if count != 0 && b.get(candidate, 302, TYPE_DESCRIPTOR, &mut knobs)? != count {
        return Err("cuDNN knob identity count mismatch".into());
    }
    let mut choices = Vec::with_capacity(count);
    for knob in knobs {
        let mut kind = [0 as c_int];
        if b.get(knob, 600, TYPE_KNOB, &mut kind)? != 1 {
            return Err("cuDNN knob identity missing".into());
        }
        choices.push((kind[0], b.i64(knob, 601)?));
    }
    choices.sort_unstable();
    Ok(Some(format!(
        "engine={index};knobs={choices:?};notes={:?}",
        &notes[..note_count]
    )))
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
