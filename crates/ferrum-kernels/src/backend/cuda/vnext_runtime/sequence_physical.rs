//! Backend physical facts owned by a Sequence catalog, never by an old wave.
//! The core grants a fresh logical window before these pre-sliced rows are used.
use super::segment_owners::CudaRegionView;
use super::*;
use ferrum_interfaces::vnext::{DynamicStorageView, SequencePhysicalBindingTable};
use std::collections::HashMap;

pub(crate) struct CudaSequencePhysicalTable {
    runtime_instance: u64,
    rows: Vec<Option<CudaPhysicalRow>>,
    // Independent logical/capacity owner, released after every physical region.
    _lifetime: Arc<dyn Send + Sync>,
}

struct CudaPhysicalRow {
    length_bytes: u64,
    storage: CudaPhysicalStorage,
}

enum CudaPhysicalStorage {
    Contiguous(CudaBufferRegion),
    Paged {
        block_bytes: u64,
        pages: Vec<CudaBufferRegion>,
    },
}

impl CudaSequencePhysicalTable {
    pub(super) fn compile(
        table: &SequencePhysicalBindingTable<CudaDeviceBuffer>,
        runtime_instance: u64,
    ) -> Result<Option<Arc<Self>>, CudaDeviceRuntimeError> {
        let invalid = CudaDeviceRuntimeError::contract;
        let Some(maximum_ordinal) = table.rows().map(|row| row.resource_ordinal()).max() else {
            return Ok(None);
        };
        let count = maximum_ordinal
            .checked_add(1)
            .ok_or_else(|| invalid("physical table resource ordinal overflows"))?;
        let mut rows: Vec<Option<CudaPhysicalRow>> = (0..count).map(|_| None).collect();
        for row in table.rows() {
            let descriptor = row.descriptor();
            let mut regions = Vec::new();
            let mut covered = 0u64;
            for physical in row.physical_regions() {
                if physical.logical_offset_bytes() != covered || physical.length_bytes() == 0 {
                    return Err(invalid("physical table row has a gap or empty region"));
                }
                let (buffer, range, retention) = physical.borrowed_buffer_and_physical_range();
                let region = buffer.borrowed_region(range, retention)?;
                if region.runtime_instance != runtime_instance
                    || region.element_type() != descriptor.element_type()
                    || region.length_bytes() != physical.length_bytes()
                {
                    return Err(invalid(
                        "physical table row has foreign runtime or metadata",
                    ));
                }
                covered = covered
                    .checked_add(region.length_bytes())
                    .ok_or_else(|| invalid("physical table coverage overflows"))?;
                regions.push(region.to_owned());
            }
            if covered == 0 || covered != descriptor.view_size_bytes() {
                return Err(invalid("physical table does not cover its declared row"));
            }
            let storage = match row.storage_profile().view() {
                DynamicStorageView::Contiguous => {
                    if regions.len() != 1 {
                        return Ok(None);
                    }
                    CudaPhysicalStorage::Contiguous(regions.pop().expect("one physical region"))
                }
                DynamicStorageView::PagedRegions { block_bytes } => {
                    let mut pages = Vec::new();
                    for region in regions {
                        // A provider can decline a legal storage profile whose
                        // physical segments cannot be borrowed as whole pages.
                        if block_bytes == 0 || region.length_bytes() % block_bytes != 0 {
                            return Ok(None);
                        }
                        let mut offset = 0;
                        while offset < region.length_bytes() {
                            pages.push(region.subregion(offset, block_bytes)?);
                            offset = offset
                                .checked_add(block_bytes)
                                .ok_or_else(|| invalid("physical page offset overflows"))?;
                        }
                    }
                    CudaPhysicalStorage::Paged { block_bytes, pages }
                }
            };
            let slot = &mut rows[row.resource_ordinal()];
            if slot.is_some() {
                return Err(invalid("physical table repeats a resource ordinal"));
            }
            *slot = Some(CudaPhysicalRow {
                length_bytes: covered,
                storage,
            });
        }
        Ok(Some(Arc::new(Self {
            runtime_instance,
            rows,
            _lifetime: Arc::clone(table.lifetime_keeper()),
        })))
    }

    fn row(&self, ordinal: usize) -> Result<&CudaPhysicalRow, CudaDeviceRuntimeError> {
        self.rows
            .get(ordinal)
            .and_then(Option::as_ref)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("physical resource row is absent"))
    }
}

/// One current authorized resource window; page ownership is held once in its
/// cold table. These windows preserve the ordinary flattened retention order.
#[derive(Clone)]
pub(crate) struct CudaPhysicalRowWindow {
    table_index: usize,
    resource_ordinal: usize,
    offset_bytes: u64,
    length_bytes: u64,
    region_count: usize,
}

impl CudaPhysicalRowWindow {
    pub(crate) fn table_index(&self) -> usize {
        self.table_index
    }

    /// Only a typed, Core-verified append refresh may grow this window. Page
    /// addresses come from the new table, never from extrapolating old pointers.
    pub(crate) fn refreshed(
        &self,
        tables: &[Arc<CudaSequencePhysicalTable>],
        length_bytes: u64,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let invalid = CudaDeviceRuntimeError::contract;
        let row = tables
            .get(self.table_index)
            .ok_or_else(|| invalid("refreshed physical table is absent"))?
            .row(self.resource_ordinal)?;
        checked_subregion_address(0, row.length_bytes, self.offset_bytes, length_bytes)?;
        let region_count = match &row.storage {
            CudaPhysicalStorage::Contiguous(_) if length_bytes == self.length_bytes => 1,
            CudaPhysicalStorage::Paged { block_bytes, .. }
                if length_bytes >= self.length_bytes
                    && self.offset_bytes % block_bytes == 0
                    && length_bytes % block_bytes == 0 =>
            {
                usize::try_from(length_bytes / block_bytes)
                    .map_err(|_| invalid("refreshed page count exceeds usize"))?
            }
            _ => return Err(invalid("refreshed physical window is not append-only")),
        };
        Ok(Self {
            length_bytes,
            region_count,
            ..self.clone()
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.region_count
    }

    pub(crate) fn region<'a>(
        &self,
        tables: &'a [Arc<CudaSequencePhysicalTable>],
        index: usize,
    ) -> Result<CudaRegionView<'a>, CudaDeviceRuntimeError> {
        let invalid = CudaDeviceRuntimeError::contract;
        if index >= self.region_count {
            return Err(invalid("physical window region index is absent"));
        }
        let row = tables
            .get(self.table_index)
            .ok_or_else(|| invalid("physical owner table is absent"))?
            .row(self.resource_ordinal)?;
        match &row.storage {
            CudaPhysicalStorage::Contiguous(region) => {
                let mut view = region.borrowed();
                view.device_ptr = checked_subregion_address(
                    view.device_ptr,
                    view.length_bytes,
                    self.offset_bytes,
                    self.length_bytes,
                )?;
                view.length_bytes = self.length_bytes;
                Ok(view)
            }
            CudaPhysicalStorage::Paged { block_bytes, pages } => {
                let start = usize::try_from(self.offset_bytes / block_bytes)
                    .map_err(|_| invalid("physical page offset exceeds usize"))?;
                pages
                    .get(
                        start
                            .checked_add(index)
                            .ok_or_else(|| invalid("physical page index overflows"))?,
                    )
                    .map(CudaBufferRegion::borrowed)
                    .ok_or_else(|| invalid("physical page is absent"))
            }
        }
    }

    pub(crate) fn validate_runtime(
        &self,
        tables: &[Arc<CudaSequencePhysicalTable>],
        runtime: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        if tables
            .get(self.table_index)
            .is_none_or(|table| table.runtime_instance != runtime)
        {
            return Err(CudaDeviceRuntimeError::contract(
                "physical window has a foreign runtime",
            ));
        }
        Ok(())
    }
}

pub(crate) struct CudaPhysicalOwnerBuilder {
    tables: Vec<Arc<CudaSequencePhysicalTable>>,
    indices: HashMap<usize, usize>,
}

impl CudaPhysicalOwnerBuilder {
    pub(crate) fn new() -> Self {
        Self {
            tables: Vec::new(),
            indices: HashMap::new(),
        }
    }

    pub(crate) fn window(
        &mut self,
        table: &SequencePhysicalBindingTable<CudaDeviceBuffer>,
        ordinal: usize,
        offset: u64,
        length: u64,
        page_bytes: Option<u64>,
    ) -> Result<CudaPhysicalRowWindow, CudaDeviceRuntimeError> {
        let invalid = CudaDeviceRuntimeError::contract;
        let backend = table
            .backend_bindings()
            .ok_or_else(|| invalid("Sequence physical backend is absent"))?;
        let key = Arc::as_ptr(backend) as *const () as usize;
        let index = if let Some(index) = self.indices.get(&key) {
            *index
        } else {
            let physical = Arc::clone(backend)
                .downcast::<CudaSequencePhysicalTable>()
                .map_err(|_| invalid("Sequence physical backend has another provider type"))?;
            let index = self.tables.len();
            self.tables.push(physical);
            self.indices.insert(key, index);
            index
        };
        let row = self.tables[index].row(ordinal)?;
        checked_subregion_address(0, row.length_bytes, offset, length)?;
        let region_count = match (&row.storage, page_bytes) {
            (CudaPhysicalStorage::Contiguous(_), None) => 1,
            (CudaPhysicalStorage::Paged { block_bytes, .. }, Some(expected))
                if expected == *block_bytes
                    && offset % block_bytes == 0
                    && length % block_bytes == 0 =>
            {
                usize::try_from(length / block_bytes)
                    .map_err(|_| invalid("physical window page count exceeds usize"))?
            }
            _ => {
                return Err(invalid(
                    "physical window storage differs from its requested geometry",
                ))
            }
        };
        Ok(CudaPhysicalRowWindow {
            table_index: index,
            resource_ordinal: ordinal,
            offset_bytes: offset,
            length_bytes: length,
            region_count,
        })
    }

    pub(crate) fn region(
        &self,
        window: &CudaPhysicalRowWindow,
        index: usize,
    ) -> Result<CudaRegionView<'_>, CudaDeviceRuntimeError> {
        window.region(&self.tables, index)
    }

    pub(crate) fn finish(self) -> Arc<CudaSegmentOwnerTable> {
        Arc::new(CudaSegmentOwnerTable::physical(self.tables))
    }
}

pub(crate) enum CudaBindingProgramRanges {
    Indexed(Vec<CudaSegmentRange>),
    Physical(Vec<CudaPhysicalRowWindow>),
}

impl CudaBindingProgramRanges {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Indexed(ranges) => ranges.len(),
            Self::Physical(rows) => rows.iter().map(CudaPhysicalRowWindow::len).sum(),
        }
    }

    pub(crate) fn region<'a>(
        &self,
        owners: &'a CudaSegmentOwnerTable,
        mut index: usize,
    ) -> Result<CudaRegionView<'a>, CudaDeviceRuntimeError> {
        match self {
            Self::Indexed(ranges) => owners.region(
                ranges
                    .get(index)
                    .ok_or_else(|| CudaDeviceRuntimeError::contract("indexed region is absent"))?,
            ),
            Self::Physical(rows) => {
                for row in rows {
                    if index < row.len() {
                        return row.region(owners.physical_tables(), index);
                    }
                    index -= row.len();
                }
                Err(CudaDeviceRuntimeError::contract(
                    "physical region is absent",
                ))
            }
        }
    }

    pub(crate) fn validate_runtime(
        &self,
        owners: &CudaSegmentOwnerTable,
        runtime: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        match self {
            Self::Indexed(ranges) => {
                for range in ranges {
                    if owners.region(range)?.runtime_instance != runtime {
                        return Err(CudaDeviceRuntimeError::contract(
                            "compiled binding owner has a foreign runtime",
                        ));
                    }
                }
            }
            Self::Physical(rows) => {
                for row in rows {
                    row.validate_runtime(owners.physical_tables(), runtime)?;
                }
            }
        }
        Ok(())
    }
}
