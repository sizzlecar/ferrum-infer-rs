//! Bounded physical packing; static provisioning still owns each full resource.
use super::*;

#[derive(Default)]
struct ArenaCursor {
    ordinal: u64,
    end_bytes: u64,
    occupied: bool,
}

pub(super) fn bounded_locations(
    components: Vec<&crate::WeightComponentSpec>,
    execution_weight_fingerprint: &str,
    provider_alignment_bytes: u64,
    target_bytes: NonZeroU64,
) -> Result<BTreeMap<WeightId, WeightComponentLocation>, VNextError> {
    let mut arenas = BTreeMap::<ElementType, ArenaCursor>::new();
    let mut locations = BTreeMap::new();
    for component in components {
        let element_type = component.physical_element_type();
        let alignment = provider_alignment_bytes.max(element_type.size_bytes());
        // Validate even an empty or oversized arena's alignment.
        checked_align_up(0, alignment)?;
        let length_bytes = component.physical_bytes()?;
        let arena = arenas.entry(element_type).or_default();
        let aligned = checked_align_up(arena.end_bytes, alignment).ok();
        let end = aligned.and_then(|offset| offset.checked_add(length_bytes));
        let offset_bytes = if arena.occupied
            && (arena.end_bytes > target_bytes.get()
                || end.is_none_or(|end| end > target_bytes.get()))
        {
            arena.ordinal = arena
                .ordinal
                .checked_add(1)
                .ok_or_else(|| invalid_plan("weight arena chunk ordinal overflows"))?;
            0
        } else {
            aligned.ok_or_else(|| invalid_plan("weight arena alignment overflows u64"))?
        };
        arena.end_bytes = offset_bytes
            .checked_add(length_bytes)
            .ok_or_else(|| invalid_plan("weight arena byte range overflows u64"))?;
        arena.occupied = true;
        let semantic_identity = serde_json::to_string(&(
            "bounded-v1",
            element_type,
            target_bytes.get(),
            arena.ordinal,
        ))
        .map_err(|error| VNextError::Serialization {
            context: "serialize bounded weight arena identity",
            message: error.to_string(),
        })?;
        let resource_id = hashed_resource_id(
            "weight-arena",
            execution_weight_fingerprint,
            &semantic_identity,
        )?;
        locations.insert(
            component.id.clone(),
            WeightComponentLocation {
                resource_id,
                offset_bytes,
                length_bytes,
                element_type,
            },
        );
    }
    Ok(locations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WeightComponentRole, WeightComponentSpec, WeightEncoding};

    fn component(name: &str, element_type: ElementType, elements: u64) -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new(name).unwrap(),
            role: WeightComponentRole::Values,
            external_names: vec![name.to_owned()],
            dimensions: vec![elements],
            encoding: WeightEncoding::Dense { element_type },
            required: true,
        }
    }

    fn pack(
        components: &[WeightComponentSpec],
        alignment: u64,
        target: u64,
    ) -> BTreeMap<WeightId, WeightComponentLocation> {
        bounded_locations(
            components.iter().collect(),
            "schema-fingerprint",
            alignment,
            NonZeroU64::new(target).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn alignment_exact_target_and_oversized_components_keep_whole_ranges() {
        let c = [
            component("a", ElementType::F32, 3),
            component("b", ElementType::F32, 4),
            component("c", ElementType::F32, 9),
            component("d", ElementType::F32, 1),
            component("e", ElementType::F32, 1),
        ];
        let p = pack(&c, 16, 32);
        let get = |i: usize| p.get(&c[i].id).unwrap();
        assert_eq!(get(0).offset_bytes, 0);
        assert_eq!(get(1).offset_bytes, 16);
        assert_eq!(get(0).resource_id, get(1).resource_id);
        assert_ne!(get(1).resource_id, get(2).resource_id);
        assert_eq!((get(2).offset_bytes, get(2).length_bytes), (0, 36));
        assert_ne!(get(2).resource_id, get(3).resource_id);
        assert_eq!(get(3).resource_id, get(4).resource_id);
        assert_eq!((get(3).offset_bytes, get(4).offset_bytes), (0, 16));
        for component in &c {
            let location = &p[&component.id];
            assert_eq!(location.length_bytes, component.physical_bytes().unwrap());
            assert_eq!(location.offset_bytes % 16, 0);
        }
    }

    #[test]
    fn dtype_cursors_are_independent_and_ids_bind_target_and_ordinal() {
        let c = [
            component("a", ElementType::F32, 4),
            component("b", ElementType::F16, 4),
            component("c", ElementType::F32, 4),
            component("d", ElementType::F32, 1),
        ];
        let p = pack(&c, 8, 32);
        let q = pack(&c, 8, 64);
        assert_eq!(p[&c[0].id].resource_id, p[&c[2].id].resource_id);
        assert_ne!(p[&c[0].id].resource_id, p[&c[1].id].resource_id);
        assert_ne!(p[&c[0].id].resource_id, p[&c[3].id].resource_id);
        assert_ne!(p[&c[0].id].resource_id, q[&c[0].id].resource_id);
        assert_eq!(p[&c[1].id].offset_bytes, 0);
    }

    #[test]
    fn alignment_overflow_starts_a_new_chunk_but_component_overflow_rejects() {
        let c = [
            component("a", ElementType::U8, u64::MAX),
            component("b", ElementType::U8, 1),
        ];
        let p = pack(&c, 256, u64::MAX);
        assert_eq!(p[&c[1].id].offset_bytes, 0);
        assert_ne!(p[&c[0].id].resource_id, p[&c[1].id].resource_id);
        let c = [component("bad", ElementType::F32, u64::MAX)];
        assert!(bounded_locations(
            c.iter().collect(),
            "schema",
            256,
            NonZeroU64::new(32).unwrap()
        )
        .is_err());
        assert!(
            serde_json::from_str::<WeightArenaPacking>(r#"{"bounded":{"target_bytes":0}}"#)
                .is_err()
        );
    }
}
