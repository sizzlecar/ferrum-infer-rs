//! Scalar page projections belong to an immutable Core physical part. These
//! values own no allocation or authority; a current approved part still selects
//! every range retained by the submission. The complete payload is uploaded.
use super::{visit_binding_addresses, CausalKvLayout, VNEXT_KV_PAGE_BYTES};
use ferrum_interfaces::vnext::{
    SegmentBindingPageAddressLayout, SegmentBindingPageTemplate, SegmentBindingPageTemplateKey,
    SegmentBindingTemplatePage,
};
use std::ops::Range;
use std::sync::Arc;

pub(super) struct ColdOwner {
    pub local_owner: usize,
    pub physical_range: Range<u64>,
    pub address: u64,
}

pub(super) struct Selection {
    pub pages: Range<usize>,
    pub addresses: Range<usize>,
}

/// Only scalar templates are retained here, never the enclosing Core row.
pub(super) struct AddressPart {
    pub template: Arc<SegmentBindingPageTemplate>,
    pub addresses: Range<usize>,
}

pub(super) fn address_layout(layout: CausalKvLayout) -> SegmentBindingPageAddressLayout {
    match layout {
        CausalKvLayout::TokenMajorPages => SegmentBindingPageAddressLayout::Flat,
        CausalKvLayout::VllmBlocks16 {
            combined_block_bytes,
            blocks_per_page,
        } => SegmentBindingPageAddressLayout::Blocked {
            combined_block_bytes,
            blocks_per_page,
        },
    }
}

fn causal_layout(layout: SegmentBindingPageAddressLayout) -> CausalKvLayout {
    match layout {
        SegmentBindingPageAddressLayout::Flat => CausalKvLayout::TokenMajorPages,
        SegmentBindingPageAddressLayout::Blocked {
            combined_block_bytes,
            blocks_per_page,
        } => CausalKvLayout::VllmBlocks16 {
            combined_block_bytes,
            blocks_per_page,
        },
    }
}

fn entries_per_page(key: &SegmentBindingPageTemplateKey) -> Result<usize, String> {
    if key.physical_page_bytes != VNEXT_KV_PAGE_BYTES {
        return Err("segment causal template page size differs from its ABI".into());
    }
    match key.address_layout {
        SegmentBindingPageAddressLayout::Flat => Ok(1),
        SegmentBindingPageAddressLayout::Blocked {
            combined_block_bytes,
            blocks_per_page,
        } => {
            if combined_block_bytes == 0
                || blocks_per_page == 0
                || combined_block_bytes
                    .checked_mul(blocks_per_page)
                    .is_none_or(|bytes| bytes > key.physical_page_bytes)
            {
                return Err("segment causal template block geometry is invalid".into());
            }
            usize::try_from(blocks_per_page)
                .map_err(|_| "segment causal template entries exceed usize".into())
        }
    }
}

/// The caller supplies bases derived from the actual checked CUDA buffers.
/// An unsupported full-part geometry returns None so the ordinary window
/// encoder can still handle a valid clipped view.
pub(super) fn build(
    key: SegmentBindingPageTemplateKey,
    owners: impl IntoIterator<Item = ColdOwner>,
) -> Result<Option<SegmentBindingPageTemplate>, String> {
    let per_page = entries_per_page(&key)?;
    let mut pages = Vec::new();
    let mut physical_bytes = 0u64;
    for owner in owners {
        let length = owner
            .physical_range
            .end
            .checked_sub(owner.physical_range.start)
            .filter(|length| *length != 0)
            .ok_or("segment causal template owner range is empty or reversed")?;
        if length % key.physical_page_bytes != 0 {
            return Ok(None);
        }
        physical_bytes = physical_bytes
            .checked_add(length)
            .ok_or("segment causal template coverage overflows")?;
        let mut offset = 0u64;
        while offset < length {
            let physical_offset_bytes = owner
                .physical_range
                .start
                .checked_add(offset)
                .ok_or("segment causal template physical offset overflows")?;
            let address = owner
                .address
                .checked_add(offset)
                .filter(|address| {
                    *address != 0 && address.checked_add(key.physical_page_bytes).is_some()
                })
                .ok_or("segment causal template native address overflows")?;
            pages.push(SegmentBindingTemplatePage {
                local_owner: owner.local_owner,
                physical_offset_bytes,
                length_bytes: key.physical_page_bytes,
                address,
            });
            offset = offset
                .checked_add(key.physical_page_bytes)
                .ok_or("segment causal template page cursor overflows")?;
        }
    }
    if pages.is_empty() {
        return Ok(None);
    }
    let count = pages
        .len()
        .checked_mul(per_page)
        .and_then(|count| i32::try_from(count).ok())
        .ok_or("segment causal template address count exceeds i32")?;
    let mut addresses = Vec::with_capacity(count as usize);
    visit_binding_addresses(
        causal_layout(key.address_layout),
        count,
        pages.len(),
        |index| {
            pages
                .get(index)
                .map(|page| page.address)
                .ok_or("segment causal template page is absent".into())
        },
        |address| addresses.push(address),
    )?;
    Ok(Some(SegmentBindingPageTemplate {
        key,
        pages,
        addresses,
        physical_bytes,
    }))
}

pub(super) fn select(
    template: &SegmentBindingPageTemplate,
    start: u64,
    length: u64,
) -> Result<Selection, String> {
    let per_page = entries_per_page(&template.key)?;
    let bytes = template.key.physical_page_bytes;
    let end = start
        .checked_add(length)
        .ok_or("segment causal template selected window overflows")?;
    if length == 0 || start % bytes != 0 || length % bytes != 0 || end > template.physical_bytes {
        return Err("segment causal template selection is outside its complete pages".into());
    }
    let first = usize::try_from(start / bytes)
        .map_err(|_| "segment causal template first page exceeds usize")?;
    let last = usize::try_from(end / bytes)
        .map_err(|_| "segment causal template last page exceeds usize")?;
    let physical_bytes = u64::try_from(template.pages.len())
        .ok()
        .and_then(|pages| pages.checked_mul(bytes));
    if physical_bytes != Some(template.physical_bytes)
        || template.pages.len().checked_mul(per_page) != Some(template.addresses.len())
        || last > template.pages.len()
    {
        return Err("segment causal template scalar coverage is inconsistent".into());
    }
    let address_first = first
        .checked_mul(per_page)
        .ok_or("segment causal template first address overflows")?;
    let address_last = last
        .checked_mul(per_page)
        .ok_or("segment causal template last address overflows")?;
    Ok(Selection {
        pages: first..last,
        addresses: address_first..address_last,
    })
}

/// Copies already-derived addresses in the original order, including a partial
/// last KV page. Controls, active entry count and the full upload stay current.
pub(super) fn append_addresses(
    layout: CausalKvLayout,
    table_entries: i32,
    page_count: usize,
    parts: &[AddressPart],
    payload: &mut Vec<u8>,
) -> Result<(), String> {
    let entries = usize::try_from(table_entries)
        .map_err(|_| "causal attention address-table count is negative")?;
    let expected_layout = address_layout(layout);
    let mut available = 0usize;
    let mut selected_pages = 0usize;
    for part in parts {
        if part.template.key.address_layout != expected_layout {
            return Err("segment causal template address layout changed".into());
        }
        let per_page = entries_per_page(&part.template.key)?;
        let count = part
            .addresses
            .end
            .checked_sub(part.addresses.start)
            .filter(|count| *count != 0 && *count % per_page == 0)
            .ok_or("segment causal template address selection is invalid")?;
        if part.addresses.end > part.template.addresses.len()
            || part.addresses.start % per_page != 0
        {
            return Err("segment causal template address selection exceeds its part".into());
        }
        available = available
            .checked_add(count)
            .ok_or("segment causal template available addresses overflow")?;
        selected_pages = selected_pages
            .checked_add(count / per_page)
            .ok_or("segment causal template selected pages overflow")?;
    }
    if selected_pages != page_count
        || entries > available
        || (matches!(layout, CausalKvLayout::TokenMajorPages) && entries != page_count)
    {
        return Err("segment causal template table does not match current retained pages".into());
    }
    let mut remaining = entries;
    for part in parts {
        let addresses = &part.template.addresses[part.addresses.clone()];
        let take = remaining.min(addresses.len());
        for address in &addresses[..take] {
            payload.extend_from_slice(&address.to_ne_bytes());
        }
        remaining -= take;
        if remaining == 0 {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{binding_addresses, BINDING_CONTROL_BYTES};
    use super::*;
    use ferrum_interfaces::vnext::ElementType;

    fn key(layout: CausalKvLayout) -> SegmentBindingPageTemplateKey {
        SegmentBindingPageTemplateKey {
            physical_page_bytes: VNEXT_KV_PAGE_BYTES,
            element_type: ElementType::F16,
            address_layout: address_layout(layout),
            runtime_instance: 1,
            provider_fingerprint: [1; 32],
            numerical_profile: [2; 32],
        }
    }

    fn template(layout: CausalKvLayout, addresses: &[u64]) -> Arc<SegmentBindingPageTemplate> {
        Arc::new(
            build(
                key(layout),
                addresses
                    .iter()
                    .enumerate()
                    .map(|(local, address)| ColdOwner {
                        local_owner: local,
                        physical_range: (local as u64) * VNEXT_KV_PAGE_BYTES
                            ..((local as u64) + 1) * VNEXT_KV_PAGE_BYTES,
                        address: *address,
                    }),
            )
            .unwrap()
            .unwrap(),
        )
    }

    fn part(template: Arc<SegmentBindingPageTemplate>, start: u64, bytes: u64) -> AddressPart {
        let selection = select(&template, start, bytes).unwrap();
        AddressPart {
            template,
            addresses: selection.addresses,
        }
    }

    #[test]
    fn causal_segment_page_template_matches_full_payload_and_partial_last_block() {
        for layout in [
            CausalKvLayout::TokenMajorPages,
            CausalKvLayout::VllmBlocks16 {
                combined_block_bytes: VNEXT_KV_PAGE_BYTES / 2,
                blocks_per_page: 2,
            },
        ] {
            let bases = (0..25)
                .map(|i| 0x10_0000 + i * 2 * VNEXT_KV_PAGE_BYTES)
                .collect::<Vec<_>>();
            let cached = template(layout, &bases);
            let entry_counts = if matches!(layout, CausalKvLayout::TokenMajorPages) {
                vec![1, 2, 25]
            } else {
                vec![1, 2, 3, 49, 50]
            };
            for entries in entry_counts {
                let pages = if matches!(layout, CausalKvLayout::TokenMajorPages) {
                    entries as usize
                } else {
                    (entries as usize).div_ceil(2)
                };
                let controls = [entries, 0, 1, entries * 16, 7, 0];
                let mut actual = Vec::new();
                for control in controls {
                    actual.extend_from_slice(&control.to_ne_bytes());
                }
                append_addresses(
                    layout,
                    entries,
                    pages,
                    &[part(
                        Arc::clone(&cached),
                        0,
                        pages as u64 * VNEXT_KV_PAGE_BYTES,
                    )],
                    &mut actual,
                )
                .unwrap();
                let mut expected = Vec::new();
                for control in controls {
                    expected.extend_from_slice(&control.to_ne_bytes());
                }
                for address in binding_addresses(layout, entries, &bases[..pages]).unwrap() {
                    expected.extend_from_slice(&address.to_ne_bytes());
                }
                assert_eq!(actual, expected);
                assert_eq!(
                    actual.len(),
                    BINDING_CONTROL_BYTES as usize + entries as usize * 8
                );
            }
        }
    }

    #[test]
    fn causal_segment_page_template_keeps_growth_shortening_and_reordered_parts_exact() {
        let layout = CausalKvLayout::VllmBlocks16 {
            combined_block_bytes: VNEXT_KV_PAGE_BYTES / 2,
            blocks_per_page: 2,
        };
        let first = template(layout, &[0x10_0000, 0x40_0000]);
        let tail = template(layout, &[0x90_0000]);
        for (parts, bases, entries) in [
            (
                vec![part(Arc::clone(&first), 0, VNEXT_KV_PAGE_BYTES)],
                vec![0x10_0000],
                1,
            ),
            (
                vec![
                    part(Arc::clone(&first), 0, 2 * VNEXT_KV_PAGE_BYTES),
                    part(Arc::clone(&tail), 0, VNEXT_KV_PAGE_BYTES),
                ],
                vec![0x10_0000, 0x40_0000, 0x90_0000],
                5,
            ),
            (
                vec![
                    part(Arc::clone(&tail), 0, VNEXT_KV_PAGE_BYTES),
                    part(Arc::clone(&first), VNEXT_KV_PAGE_BYTES, VNEXT_KV_PAGE_BYTES),
                ],
                vec![0x90_0000, 0x40_0000],
                3,
            ),
        ] {
            let mut actual = Vec::new();
            append_addresses(layout, entries, bases.len(), &parts, &mut actual).unwrap();
            let expected = binding_addresses(layout, entries, &bases)
                .unwrap()
                .into_iter()
                .flat_map(u64::to_ne_bytes)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
        let replacement = template(layout, &[0x70_0000, 0x80_0000]);
        let independent = template(layout, &[0x10_0000, 0x40_0000]);
        assert!(!Arc::ptr_eq(&first, &replacement));
        assert!(!Arc::ptr_eq(&first, &independent));
        assert_eq!(independent.addresses, first.addresses);
        assert_ne!(replacement.addresses, first.addresses);
    }

    #[test]
    fn causal_segment_page_template_rejects_geometry_overflow_and_outside_selection() {
        let layout = CausalKvLayout::TokenMajorPages;
        let cached = template(layout, &[0x10_0000]);
        assert!(select(&cached, 1, VNEXT_KV_PAGE_BYTES).is_err());
        assert!(select(&cached, 0, 0).is_err());
        assert!(select(&cached, VNEXT_KV_PAGE_BYTES, VNEXT_KV_PAGE_BYTES).is_err());
        assert!(select(&cached, u64::MAX, 1).is_err());
        assert!(build(
            key(layout),
            [ColdOwner {
                local_owner: 0,
                physical_range: 0..VNEXT_KV_PAGE_BYTES,
                address: u64::MAX - 1
            }]
        )
        .is_err());
        assert!(build(
            key(layout),
            [ColdOwner {
                local_owner: 0,
                physical_range: 0..(VNEXT_KV_PAGE_BYTES - 1),
                address: 0x10_0000
            }]
        )
        .unwrap()
        .is_none());
        let selected = [part(cached, 0, VNEXT_KV_PAGE_BYTES)];
        assert!(append_addresses(layout, 2, 1, &selected, &mut Vec::new()).is_err());
        assert!(append_addresses(layout, -1, 1, &selected, &mut Vec::new()).is_err());
        let blocked = CausalKvLayout::VllmBlocks16 {
            combined_block_bytes: VNEXT_KV_PAGE_BYTES,
            blocks_per_page: 2,
        };
        assert!(build(
            key(blocked),
            [ColdOwner {
                local_owner: 0,
                physical_range: 0..VNEXT_KV_PAGE_BYTES,
                address: 0x10_0000
            }]
        )
        .is_err());
    }
}
