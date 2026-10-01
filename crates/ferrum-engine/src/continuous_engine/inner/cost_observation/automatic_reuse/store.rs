//! One nonblocking lease and one byte ledger across old commit and new staging.
use super::*;
use std::fs::TryLockError;

const OWNER: &[u8] = b"ferrum.automatic.reuse.v1\n";
const DIRTY: &[u8] = b"ferrum.automatic.reuse.active.v1\n";
const CONTROL: &[&str] = &[
    "owner",
    "lease",
    "dirty",
    "manifest.json",
    "manifest.pending",
];

pub(in crate::continuous_engine::inner::cost_observation) struct CacheOpen {
    pub session: CacheSession,
    /// Untrusted bounded metadata; strict independent source/profile/feedback
    /// replay must finish under transaction before this can become a seed.
    pub candidate: std::result::Result<CacheManifest, CacheMiss>,
    pub transaction: ColdTransaction,
}
#[derive(Clone)]
pub(in crate::continuous_engine::inner::cost_observation) struct CacheSession(
    pub(super) Arc<Shared>,
);
pub(super) struct Shared {
    pub root: PathBuf,
    pub generation_path: PathBuf,
    pub generation: [u8; 16],
    pub identity: CacheIdentity,
    pub limits: CacheLimits,
    pub ledger: Mutex<Ledger>,
    resumed_feedback: Mutex<Option<ResumedFeedbackArtifact>>,
    // Every source observer retains this lease, including after runtime Drop.
    _lease: File,
}
struct ResumedFeedbackArtifact {
    index: usize,
    maximum_bytes: u64,
}
pub(super) struct Ledger {
    pub used: u64,
    pub next_artifact: usize,
    pub writing: [bool; 258],
    pub sources: usize,
    pub clean: bool,
    pub closing: bool,
    pub active_writers: usize,
    pub codec_retained: usize,
}
impl Drop for Shared {
    fn drop(&mut self) {
        // Release at the final Rust owner boundary. A concurrent process spawn
        // can briefly duplicate this open-file-description before exec closes
        // CLOEXEC handles; relying only on close can keep its flock alive.
        // Source observers still retain this Shared, so they keep the lease.
        let _ = self._lease.unlock();
    }
}
impl Shared {
    pub fn charge(&self, bytes: u64) -> Result<()> {
        let mut ledger = self.ledger.lock();
        if ledger.clean || ledger.closing {
            return Err(CacheMiss::Dirty);
        }
        let next = ledger.used.checked_add(bytes).ok_or(CacheMiss::Capacity)?;
        if next > self.limits.maximum_bytes {
            return Err(CacheMiss::Capacity);
        }
        ledger.used = next;
        Ok(())
    }
    pub fn artifact(&self) -> Result<(usize, PathBuf)> {
        let mut ledger = self.ledger.lock();
        let maximum = self
            .limits
            .maximum_sources
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            .ok_or(CacheMiss::Capacity)?;
        if ledger.clean || ledger.closing {
            return Err(CacheMiss::Capacity);
        }
        // Freed, unreferenced original journals release their actual disk
        // bytes. Reuse vacant slots; the bound is simultaneous retention,
        // never lifetime generation count.
        let index = (0..maximum)
            .find(|i| !ledger.writing[*i] && !self.generation_path.join(artifact_name(*i)).exists())
            .ok_or(CacheMiss::Capacity)?;
        ledger.next_artifact = ledger.next_artifact.max(index + 1);
        ledger.writing[index] = true;
        ledger.active_writers += 1;
        Ok((index, self.generation_path.join(artifact_name(index))))
    }
    pub fn close_writer(&self, index: usize) {
        let mut ledger = self.ledger.lock();
        assert!(ledger.writing[index]);
        ledger.writing[index] = false;
        ledger.active_writers -= 1;
    }
}
impl CacheSession {
    #[cfg(test)]
    pub(super) fn duplicate_lease_for_test(&self) -> File {
        self.0._lease.try_clone().unwrap()
    }

    pub fn open(
        policy: &SloAutomaticCalibrationReuseV1,
        identity: CacheIdentity,
        now: u64,
        mut limits: CacheLimits,
    ) -> Result<CacheOpen> {
        let SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 {
            location,
            limits: policy_limits,
        } = policy
        else {
            return Err(CacheMiss::Disabled);
        };
        identity
            .clock
            .validate()
            .map_err(|_| CacheMiss::UnsupportedClock)?;
        limits.maximum_bytes = limits
            .maximum_bytes
            .min(policy_limits.maximum_total_bytes.get());
        limits.maximum_duration = limits.maximum_duration.min(Duration::from_millis(
            policy_limits.maximum_operation_duration_ms.get(),
        ));
        Self::open_at(
            &location
                .resolve_directory()
                .map_err(|_| CacheMiss::Location)?,
            identity,
            now,
            limits,
        )
    }

    pub(super) fn open_at(
        directory: &Path,
        identity: CacheIdentity,
        now: u64,
        limits: CacheLimits,
    ) -> Result<CacheOpen> {
        limits.validate()?;
        // Paths, source handles and serialization/decode buffers can coexist.
        // Authorize them before creating any path-owning cache structures.
        let paths = directory
            .as_os_str()
            .len()
            .checked_add(128)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(4096))
            .and_then(|n| n.checked_mul(limits.maximum_sources))
            .ok_or(CacheMiss::Capacity)?;
        let peak = limits
            .manifest_limit()?
            .checked_mul(16)
            .and_then(|n| n.checked_add(paths))
            .and_then(|n| n.checked_add(8192))
            .ok_or(CacheMiss::Capacity)?;
        if peak > limits.maximum_retained_bytes {
            return Err(CacheMiss::Capacity);
        }
        let transaction = ColdTransaction::new(limits)?;
        fs::create_dir_all(directory).map_err(io)?;
        let root = directory.join("cache-v1");
        match fs::symlink_metadata(&root) {
            Ok(m) if m.file_type().is_dir() => {}
            Ok(_) => return Err(CacheMiss::Corrupt),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&root).map_err(io)?;
            }
            Err(e) => return Err(io(e)),
        }
        // Exporters bind canonical absolute source paths. Standard macOS
        // temporary/cache parents may use /var -> /private/var aliases; keep
        // one actual directory identity instead of comparing path spellings.
        let root = root.canonicalize().map_err(io)?;
        let lease_path = root.join("lease");
        if fs::symlink_metadata(&lease_path).is_ok_and(|m| !m.file_type().is_file()) {
            return Err(CacheMiss::Corrupt);
        }
        let lease = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lease_path)
            .map_err(io)?;
        match lease.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(CacheMiss::Busy),
            Err(TryLockError::Error(e)) => return Err(io(e)),
        }
        let owner = root.join("owner");
        if !owner.exists() {
            for e in fs::read_dir(&root).map_err(io)? {
                if e.map_err(io)?.file_name() != "lease" {
                    return Err(CacheMiss::Corrupt);
                }
            }
            if OWNER.len() as u64 + DIRTY.len() as u64 > limits.maximum_bytes {
                return Err(CacheMiss::Capacity);
            }
            write_new(&owner, OWNER)?;
            sync_directory(&root)?;
        }
        if regular_bytes(&owner)? != OWNER.len() as u64 || fs::read(&owner).map_err(io)? != OWNER {
            return Err(CacheMiss::Corrupt);
        }
        let mut used = scan(&root, limits, &transaction)?;
        let was_dirty = root.join("dirty").exists();
        let candidate = if was_dirty {
            Err(CacheMiss::Dirty)
        } else {
            read_candidate(&root, &identity, now, limits, &transaction)
        };
        // Dirty/corrupt/cross-boot/expired data is never used as fallback. Once
        // its exclusive lease is ours, remove only the fully scanned owned tree
        // so a later clean session can recover automatically from a cache miss.
        if candidate.is_err() {
            clear_payloads(&root, &transaction)?;
            used = regular_bytes(&owner)?
                .checked_add(regular_bytes(&lease_path)?)
                .ok_or(CacheMiss::Capacity)?;
        }
        transaction.poll()?;
        let dirty_path = root.join("dirty");
        used = used
            .checked_add(DIRTY.len() as u64)
            .ok_or(CacheMiss::Capacity)?;
        if used > limits.maximum_bytes {
            return Err(CacheMiss::Capacity);
        }
        write_new(&dirty_path, DIRTY)?;
        sync_directory(&root)?;
        // Paths never move after source/profile metadata binds them.
        let generation = *uuid::Uuid::new_v4().as_bytes();
        let generation_path = root.join(format!("g-{}", hex(&generation)));
        fs::create_dir(&generation_path).map_err(io)?;
        sync_directory(&root)?;
        transaction.poll()?;
        Ok(CacheOpen {
            session: Self(Arc::new(Shared {
                root,
                generation_path,
                generation,
                identity,
                limits,
                ledger: Mutex::new(Ledger {
                    used,
                    next_artifact: 0,
                    writing: [false; 258],
                    sources: 0,
                    clean: false,
                    closing: false,
                    active_writers: 0,
                    codec_retained: 0,
                }),
                resumed_feedback: Mutex::new(None),
                _lease: lease,
            })),
            candidate,
            transaction,
        })
    }

    pub fn original_source(&self) -> Result<OriginalSourceJournal> {
        self.original_source_with_encoding(true)
    }
    pub fn adopt_source(
        &self,
        generation: [u8; 16],
        source: &CachedSource,
        transaction: &ColdTransaction,
    ) -> Result<OriginalSourceJournal> {
        {
            let mut ledger = self.0.ledger.lock();
            if ledger.sources >= self.0.limits.maximum_sources || ledger.clean || ledger.closing {
                return Err(CacheMiss::Capacity);
            }
            ledger.sources += 1;
        }
        match OriginalSourceJournal::adopt(
            self.clone(),
            &self.artifact_path(generation, &source.journal),
            source.journal,
            source.encoding,
            transaction,
        ) {
            Ok(source) => Ok(source),
            Err(reason) => {
                self.0.ledger.lock().sources -= 1;
                Err(reason)
            }
        }
    }
    #[cfg(test)]
    pub fn original_source_raw(&self) -> Result<OriginalSourceJournal> {
        self.original_source_with_encoding(false)
    }
    fn original_source_with_encoding(&self, compressed: bool) -> Result<OriginalSourceJournal> {
        {
            let mut ledger = self.0.ledger.lock();
            if ledger.sources >= self.0.limits.maximum_sources || ledger.clean || ledger.closing {
                return Err(CacheMiss::Capacity);
            }
            ledger.sources += 1;
        }
        match OriginalSourceJournal::new(self.clone(), compressed) {
            Ok(journal) => Ok(journal),
            Err(reason) => {
                self.0.ledger.lock().sources -= 1;
                Err(reason)
            }
        }
    }
    pub fn used_bytes(&self) -> u64 {
        self.0.ledger.lock().used
    }
    pub fn artifact_path(&self, generation: [u8; 16], receipt: &ArtifactReceipt) -> PathBuf {
        self.0
            .root
            .join(format!("g-{}", hex(&generation)))
            .join(receipt.file_name())
    }
    pub fn transaction(&self) -> Result<ColdTransaction> {
        ColdTransaction::new(self.0.limits)
    }

    /// At the final replay-proven handoff, retire only this session's resumed
    /// mutable store. Its in-memory state still feeds the original handoff.
    /// Releasing the physical file before final allocation preserves the
    /// existing 2*sources+2 bound, even with every source and seed slot occupied.
    pub(super) fn retire_resumed_feedback(
        &self,
        monitor: &mut super::super::selected_feedback::Monitor,
        transaction: &ColdTransaction,
    ) -> Result<()> {
        let mut retained = self.0.resumed_feedback.lock();
        let Some(artifact) = retained.as_ref() else {
            return Ok(());
        };
        transaction.poll()?;
        let path = self.0.generation_path.join(artifact_name(artifact.index));
        let receipt = monitor
            .retire_restart_storage(&path)
            .map_err(|_| CacheMiss::Corrupt)?;
        if receipt.path != path || receipt.bytes > artifact.maximum_bytes {
            return Err(CacheMiss::Corrupt);
        }
        for suffix in [".session", ".pending"] {
            let mut marker = path.as_os_str().to_owned();
            marker.push(suffix);
            if Path::new(&marker).exists() {
                return Err(CacheMiss::Dirty);
            }
        }
        transaction.poll()?;
        fs::remove_file(&path).map_err(io)?;
        sync_directory(&self.0.generation_path)?;
        let mut ledger = self.0.ledger.lock();
        ledger.used = ledger
            .used
            .checked_sub(artifact.maximum_bytes)
            .ok_or(CacheMiss::Corrupt)?;
        *retained = None;
        transaction.poll()
    }

    /// Reserve the whole external writer's simultaneous final/pending/marker
    /// bound before it allocates or writes. No refund on failure or Drop.
    pub fn reserve_external(
        &self,
        maximum_bytes: u64,
        transaction: &ColdTransaction,
    ) -> Result<ExternalArtifact> {
        transaction.poll()?;
        self.0.charge(maximum_bytes)?;
        let (index, path) = self.0.artifact()?;
        Ok(ExternalArtifact {
            session: self.clone(),
            index,
            path,
            maximum_bytes,
        })
    }

    /// Called only after original checkpoint replay and feedback handoff have
    /// succeeded. This operation never qualifies an input or changes sample age.
    pub fn commit(
        &self,
        sources: Vec<CachedSource>,
        feedback: ArtifactReceipt,
        expires_at_ns: Option<u64>,
        now: u64,
        transaction: &ColdTransaction,
    ) -> Result<()> {
        self.commit_with_validation(
            sources,
            feedback,
            expires_at_ns,
            now,
            transaction,
            || Ok(()),
        )
    }
    pub(super) fn commit_with_validation(
        &self,
        sources: Vec<CachedSource>,
        feedback: ArtifactReceipt,
        expires_at_ns: Option<u64>,
        now: u64,
        transaction: &ColdTransaction,
        validate: impl Fn() -> Result<()>,
    ) -> Result<()> {
        self.prepare_commit(
            sources,
            feedback,
            None,
            expires_at_ns,
            now,
            transaction,
            &validate,
        )?;
        self.complete_prepared_commit(transaction, validate)
    }
    pub(super) fn prepare_commit(
        &self,
        sources: Vec<CachedSource>,
        feedback: ArtifactReceipt,
        startup_seed: Option<ArtifactReceipt>,
        expires_at_ns: Option<u64>,
        now: u64,
        transaction: &ColdTransaction,
        validate: impl Fn() -> Result<()>,
    ) -> Result<()> {
        transaction.poll()?;
        validate()?;
        {
            let mut ledger = self.0.ledger.lock();
            if ledger.active_writers != 0 || ledger.clean || ledger.closing {
                return Err(CacheMiss::Dirty);
            }
            // Linearize closing with writer creation. A concurrent owner can
            // neither appear after the drain nor append beneath a clean marker.
            ledger.closing = true;
        }
        let manifest = CacheManifest {
            schema_version: 2,
            generation: self.0.generation,
            identity: self.0.identity.clone(),
            sources,
            feedback,
            startup_seed,
            expires_at_ns,
        };
        manifest.validate(&self.0.identity, now, self.0.limits)?;
        for receipt in manifest.artifacts() {
            verify_file(
                &self.0.generation_path.join(receipt.file_name()),
                receipt,
                transaction,
            )?;
        }
        let limit = self.0.limits.manifest_limit()?;
        let mut encoded = BoundedEncoding::new(limit)?;
        serde_json::to_writer(&mut encoded, &manifest).map_err(|_| CacheMiss::Capacity)?;
        transaction.poll()?;
        {
            let mut ledger = self.0.ledger.lock();
            let next = ledger
                .used
                .checked_add(encoded.bytes.len() as u64)
                .ok_or(CacheMiss::Capacity)?;
            if next > self.0.limits.maximum_bytes {
                return Err(CacheMiss::Capacity);
            }
            ledger.used = next;
        }
        let pending = self.0.root.join("manifest.pending");
        write_new(&pending, &encoded.bytes)?;
        transaction.poll()?;
        fs::rename(&pending, self.0.root.join("manifest.json")).map_err(io)?;
        sync_directory(&self.0.root)?;
        // Keep all old committed files charged until they have actually gone.
        for entry in fs::read_dir(&self.0.root).map_err(io)? {
            transaction.poll()?;
            let entry = entry.map_err(io)?;
            if entry.path() != self.0.generation_path
                && generation_name(&entry.file_name().to_string_lossy())
            {
                fs::remove_dir_all(entry.path()).map_err(io)?;
            }
        }
        transaction.poll()?;
        validate()?;
        // The worker has prepared durable replay/feedback, but only the outer
        // engine can acknowledge a successful worker join and full shutdown.
        // Until then this remains an unconditionally dirty cache candidate.
        Ok(())
    }
    pub(super) fn complete_prepared_commit(
        &self,
        transaction: &ColdTransaction,
        validate: impl Fn() -> Result<()>,
    ) -> Result<()> {
        {
            let ledger = self.0.ledger.lock();
            if !ledger.closing || ledger.clean || ledger.active_writers != 0 {
                return Err(CacheMiss::Dirty);
            }
        }
        transaction.poll()?;
        validate()?;
        fs::remove_file(self.0.root.join("dirty")).map_err(io)?;
        if let Err(reason) = sync_directory(&self.0.root)
            .and_then(|()| transaction.poll())
            .and_then(|()| validate())
        {
            // A slow/failed final fsync cannot publish a cache after its cold
            // deadline. The exclusive lease stays held while restoring dirty.
            let _ = write_new(&self.0.root.join("dirty"), DIRTY);
            let _ = sync_directory(&self.0.root);
            return Err(reason);
        }
        self.0.ledger.lock().clean = true;
        Ok(())
    }
}

pub(super) struct ExternalArtifact {
    session: CacheSession,
    pub path: PathBuf,
    index: usize,
    maximum_bytes: u64,
}
impl ExternalArtifact {
    /// Keep the mutable file's complete reservation until its original Monitor
    /// has stopped writing. No self-owning Arc is retained in CacheSession.
    pub(super) fn retain_resumed_feedback(self) -> Result<()> {
        let mut retained = self.session.0.resumed_feedback.lock();
        if retained.is_some() {
            return Err(CacheMiss::Dirty);
        }
        *retained = Some(ResumedFeedbackArtifact {
            index: self.index,
            maximum_bytes: self.maximum_bytes,
        });
        Ok(())
    }

    /// Copy immutable committed bytes into this transaction's private artifact.
    /// The caller has already reserved the entire destination lifetime bound;
    /// old-generation bytes remain charged until a new commit collects them.
    pub(super) fn copy_verified_from(
        &self,
        source: &Path,
        receipt: &ArtifactReceipt,
        transaction: &ColdTransaction,
    ) -> Result<()> {
        transaction.poll()?;
        if receipt.bytes == 0 || receipt.bytes > self.maximum_bytes {
            return Err(CacheMiss::Capacity);
        }
        if regular_bytes(source)? != receipt.bytes {
            return Err(CacheMiss::Corrupt);
        }
        let mut input = File::open(source).map_err(io)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.path)
            .map_err(io)?;
        let mut buffer = [0u8; 8192];
        let mut copied = 0u64;
        let mut hash = Sha256::new();
        loop {
            transaction.poll()?;
            let n = input.read(&mut buffer).map_err(io)?;
            if n == 0 {
                break;
            }
            copied = copied
                .checked_add(n as u64)
                .filter(|n| *n <= receipt.bytes)
                .ok_or(CacheMiss::Corrupt)?;
            hash.update(&buffer[..n]);
            output.write_all(&buffer[..n]).map_err(io)?;
        }
        if copied != receipt.bytes || <[u8; 32]>::from(hash.finalize()) != receipt.sha256 {
            return Err(CacheMiss::Corrupt);
        }
        output.sync_all().map_err(io)?;
        sync_directory(&self.session.0.generation_path)?;
        transaction.poll()
    }

    pub fn finish(self, transaction: &ColdTransaction) -> Result<ArtifactReceipt> {
        transaction.poll()?;
        // Feedback may use these paths transiently, but must close cleanly.
        for suffix in [".pending", ".session"] {
            let mut p = self.path.as_os_str().to_owned();
            p.push(suffix);
            if Path::new(&p).exists() {
                return Err(CacheMiss::Dirty);
            }
        }
        let bytes = regular_bytes(&self.path)?;
        if bytes == 0 || bytes > self.maximum_bytes {
            return Err(CacheMiss::Capacity);
        }
        let sha256 = hash_file(&self.path, transaction)?;
        // Successful writers may release their now-absent pending-name reserve.
        // Failure/Drop never refunds a partially written artifact. This is
        // disk accounting only; execution/sample budgets are untouched.
        for entry in fs::read_dir(&self.session.0.generation_path).map_err(io)? {
            let entry = entry.map_err(io)?;
            if entry.file_name().to_string_lossy().ends_with(".tmp") {
                return Err(CacheMiss::Dirty);
            }
        }
        {
            let mut ledger = self.session.0.ledger.lock();
            ledger.used = ledger
                .used
                .checked_sub(self.maximum_bytes - bytes)
                .ok_or(CacheMiss::Corrupt)?;
        }
        let receipt = ArtifactReceipt {
            index: self.index,
            bytes,
            sha256,
        };
        // Retain the global lease through verification; the conservative
        // reservation stays charged until this session completes.
        Ok(receipt)
    }
}
impl Drop for ExternalArtifact {
    fn drop(&mut self) {
        self.session.0.close_writer(self.index);
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io)?;
    file.write_all(bytes).map_err(io)?;
    file.sync_all().map_err(io)
}
fn generation_name(name: &str) -> bool {
    name.len() == 34
        && name.starts_with("g-")
        && name.as_bytes()[2..].iter().all(u8::is_ascii_hexdigit)
}
fn artifact_name_is_owned(name: &str) -> bool {
    // Existing strict profile exporter uses an atomic same-directory temp.
    if name.starts_with(".ferrum-structured-") && name.ends_with(".tmp") {
        return name
            .strip_prefix(".ferrum-structured-")
            .and_then(|n| n.strip_suffix(".tmp"))
            .is_some_and(|n| uuid::Uuid::parse_str(n).is_ok());
    }
    let name = name
        .strip_suffix(".pending")
        .or_else(|| name.strip_suffix(".session"))
        .unwrap_or(name);
    name.strip_prefix("artifact-")
        .and_then(|n| n.strip_suffix(".bin"))
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}
fn scan(root: &Path, limits: CacheLimits, transaction: &ColdTransaction) -> Result<u64> {
    let maximum_entries = limits
        .maximum_sources
        .checked_mul(8)
        .and_then(|n| n.checked_add(16))
        .ok_or(CacheMiss::Capacity)?;
    let mut entries = 0usize;
    let mut bytes = 0u64;
    for e in fs::read_dir(root).map_err(io)? {
        transaction.poll()?;
        entries += 1;
        if entries > maximum_entries {
            return Err(CacheMiss::Capacity);
        }
        let e = e.map_err(io)?;
        let name = e.file_name();
        let name = name.to_str().ok_or(CacheMiss::Corrupt)?;
        let m = fs::symlink_metadata(e.path()).map_err(io)?;
        if CONTROL.contains(&name) && m.file_type().is_file() {
            bytes = bytes.checked_add(m.len()).ok_or(CacheMiss::Capacity)?;
        } else if generation_name(name) && m.file_type().is_dir() {
            for f in fs::read_dir(e.path()).map_err(io)? {
                transaction.poll()?;
                entries += 1;
                if entries > maximum_entries {
                    return Err(CacheMiss::Capacity);
                }
                let f = f.map_err(io)?;
                if !artifact_name_is_owned(f.file_name().to_str().ok_or(CacheMiss::Corrupt)?) {
                    return Err(CacheMiss::Corrupt);
                }
                bytes = bytes
                    .checked_add(regular_bytes(&f.path())?)
                    .ok_or(CacheMiss::Capacity)?;
            }
        } else {
            return Err(CacheMiss::Corrupt);
        }
        if bytes > limits.maximum_bytes {
            return Err(CacheMiss::Capacity);
        }
    }
    Ok(bytes)
}
fn clear_payloads(root: &Path, transaction: &ColdTransaction) -> Result<()> {
    for e in fs::read_dir(root).map_err(io)? {
        transaction.poll()?;
        let e = e.map_err(io)?;
        let n = e.file_name();
        if n == "lease" || n == "owner" {
            continue;
        }
        if generation_name(&n.to_string_lossy()) {
            fs::remove_dir_all(e.path()).map_err(io)?;
        } else {
            fs::remove_file(e.path()).map_err(io)?;
        }
    }
    sync_directory(root)
}
fn read_candidate(
    root: &Path,
    identity: &CacheIdentity,
    now: u64,
    limits: CacheLimits,
    transaction: &ColdTransaction,
) -> Result<CacheManifest> {
    let path = root.join("manifest.json");
    if !path.exists() {
        return Err(CacheMiss::Missing);
    }
    if root.join("manifest.pending").exists() {
        return Err(CacheMiss::Dirty);
    }
    let limit = limits.manifest_limit()?;
    let bytes = regular_bytes(&path)?;
    if bytes > limit as u64 {
        return Err(CacheMiss::Capacity);
    }
    // Fixed capacity is authorized by manifest_limit's 16x decode allowance.
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(bytes as usize)
        .map_err(|_| CacheMiss::Capacity)?;
    if encoded.capacity() != bytes as usize {
        return Err(CacheMiss::Capacity);
    }
    encoded.resize(bytes as usize, 0);
    let mut file = File::open(path).map_err(io)?;
    file.read_exact(&mut encoded).map_err(io)?;
    if file.read(&mut [0; 1]).map_err(io)? != 0 {
        return Err(CacheMiss::Capacity);
    }
    let manifest: CacheManifest =
        serde_json::from_slice(&encoded).map_err(|_| CacheMiss::Corrupt)?;
    transaction.poll()?;
    manifest.validate(identity, now, limits)?;
    let directory = root.join(manifest.generation_name());
    if !fs::symlink_metadata(&directory)
        .map_err(io)?
        .file_type()
        .is_dir()
    {
        return Err(CacheMiss::Corrupt);
    }
    for receipt in manifest.artifacts() {
        verify_file(&directory.join(receipt.file_name()), receipt, transaction)?;
    }
    transaction.poll()?;
    Ok(manifest)
}
pub(super) fn verify_file(
    path: &Path,
    receipt: &ArtifactReceipt,
    transaction: &ColdTransaction,
) -> Result<()> {
    if regular_bytes(path)? != receipt.bytes || hash_file(path, transaction)? != receipt.sha256 {
        return Err(CacheMiss::Corrupt);
    }
    Ok(())
}
fn hash_file(path: &Path, transaction: &ColdTransaction) -> Result<[u8; 32]> {
    let mut file = File::open(path).map_err(io)?;
    let mut buffer = [0u8; 8192];
    let mut hash = Sha256::new();
    loop {
        transaction.poll()?;
        let n = file.read(&mut buffer).map_err(io)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    transaction.poll()?;
    Ok(hash.finalize().into())
}
struct BoundedEncoding {
    bytes: Vec<u8>,
    maximum: usize,
}
impl BoundedEncoding {
    fn new(maximum: usize) -> Result<Self> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(maximum)
            .map_err(|_| CacheMiss::Capacity)?;
        if bytes.capacity() != maximum {
            return Err(CacheMiss::Capacity);
        }
        Ok(Self { bytes, maximum })
    }
}
impl Write for BoundedEncoding {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.maximum - self.bytes.len() {
            return Err(std::io::ErrorKind::OutOfMemory.into());
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
