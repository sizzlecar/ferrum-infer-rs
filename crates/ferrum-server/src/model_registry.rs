use ferrum_types::ModelId;
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServedModelKind {
    Llm,
    Embedding,
    Transcription,
    Speech,
}

impl ServedModelKind {
    pub fn modalities(self) -> &'static [&'static str] {
        match self {
            Self::Llm => &["text"],
            Self::Embedding => &["text", "image"],
            Self::Transcription => &["audio"],
            Self::Speech => &["text", "audio"],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ServedModelName(String);

impl ServedModelName {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ServedModelName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraAdapterModel {
    pub name: String,
    pub model_id: String,
    pub path: String,
}

impl LoraAdapterModel {
    pub fn new(
        name: impl Into<String>,
        model_id: impl Into<String>,
        path: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            model_id: model_id.into(),
            path: path.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ServedModelEntry {
    public_name: ServedModelName,
    engine_model_id: ModelId,
    kind: ServedModelKind,
    parent_public_name: Option<ServedModelName>,
    adapter: Option<LoraAdapterModel>,
}

impl ServedModelEntry {
    pub fn public_name(&self) -> &ServedModelName {
        &self.public_name
    }

    pub fn engine_model_id(&self) -> &ModelId {
        &self.engine_model_id
    }

    pub fn kind(&self) -> ServedModelKind {
        self.kind
    }

    pub fn parent_public_name(&self) -> Option<&ServedModelName> {
        self.parent_public_name.as_ref()
    }

    pub fn adapter(&self) -> Option<&LoraAdapterModel> {
        self.adapter.as_ref()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ServedModelRegistry {
    entries: Vec<ServedModelEntry>,
    entry_by_public_name: HashMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServedModelRegistryError(String);

impl fmt::Display for ServedModelRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ServedModelRegistryError {}

impl ServedModelRegistry {
    pub fn try_new(
        engine_model_id: impl Into<ModelId>,
        kind: ServedModelKind,
        served_model_names: Vec<String>,
        adapters: Vec<LoraAdapterModel>,
    ) -> Result<Self, ServedModelRegistryError> {
        let engine_model_id = engine_model_id.into();
        validate_identifier("engine model id", &engine_model_id.0)?;
        if served_model_names.is_empty() {
            return Err(ServedModelRegistryError(
                "at least one served model name is required".to_string(),
            ));
        }
        if kind != ServedModelKind::Llm && !adapters.is_empty() {
            return Err(ServedModelRegistryError(
                "LoRA adapters may only be registered for LLM models".to_string(),
            ));
        }

        let mut registry = Self {
            entries: Vec::with_capacity(served_model_names.len() + adapters.len()),
            entry_by_public_name: HashMap::with_capacity(served_model_names.len() + adapters.len()),
        };
        for public_name in served_model_names {
            registry.push_entry(public_name, engine_model_id.clone(), kind, None, None)?;
        }
        let primary_public_name = registry.entries[0].public_name.clone();
        let mut adapter_name_indexes = HashMap::with_capacity(adapters.len());
        for adapter in adapters {
            validate_identifier("LoRA adapter name", &adapter.name)?;
            validate_identifier("LoRA public model id", &adapter.model_id)?;
            if adapter.path.trim().is_empty() {
                return Err(ServedModelRegistryError(format!(
                    "LoRA adapter path must not be empty: {}",
                    adapter.name
                )));
            }
            if adapter_name_indexes
                .insert(adapter.name.clone(), registry.entries.len())
                .is_some()
            {
                return Err(ServedModelRegistryError(format!(
                    "duplicate LoRA adapter name: {}",
                    adapter.name
                )));
            }
            registry.push_entry(
                adapter.model_id.clone(),
                engine_model_id.clone(),
                kind,
                Some(primary_public_name.clone()),
                Some(adapter),
            )?;
        }
        Ok(registry)
    }

    fn push_entry(
        &mut self,
        public_name: String,
        engine_model_id: ModelId,
        kind: ServedModelKind,
        parent_public_name: Option<ServedModelName>,
        adapter: Option<LoraAdapterModel>,
    ) -> Result<(), ServedModelRegistryError> {
        validate_identifier("served model name", &public_name)?;
        if self.entry_by_public_name.contains_key(&public_name) {
            return Err(ServedModelRegistryError(format!(
                "duplicate or colliding served model name: {public_name}"
            )));
        }
        let index = self.entries.len();
        self.entry_by_public_name.insert(public_name.clone(), index);
        self.entries.push(ServedModelEntry {
            public_name: ServedModelName(public_name),
            engine_model_id,
            kind,
            parent_public_name,
            adapter,
        });
        Ok(())
    }

    pub fn entries(&self) -> &[ServedModelEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn primary_model_name(&self) -> Option<&ServedModelName> {
        self.entries
            .iter()
            .find(|entry| entry.parent_public_name.is_none())
            .map(|entry| &entry.public_name)
    }

    pub fn adapter_models(&self) -> impl Iterator<Item = &LoraAdapterModel> {
        self.entries.iter().filter_map(ServedModelEntry::adapter)
    }

    pub fn adapter_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.adapter.is_some())
            .count()
    }

    pub fn try_with_lora_adapters(
        &self,
        expected_base: &str,
        adapters: Vec<LoraAdapterModel>,
    ) -> Result<Self, ServedModelRegistryError> {
        let base_entries = self
            .entries
            .iter()
            .filter(|entry| entry.parent_public_name.is_none())
            .collect::<Vec<_>>();
        let primary = base_entries.first().ok_or_else(|| {
            ServedModelRegistryError("cannot attach LoRA adapters to an empty registry".to_string())
        })?;
        if primary.kind != ServedModelKind::Llm {
            return Err(ServedModelRegistryError(
                "LoRA adapters may only be registered for LLM models".to_string(),
            ));
        }
        require_same_base_model(&base_entries)?;
        let expected_base_matches = primary.engine_model_id.0 == expected_base
            || base_entries
                .iter()
                .any(|entry| entry.public_name.as_str() == expected_base);
        if !expected_base_matches {
            return Err(ServedModelRegistryError(format!(
                "LoRA base {expected_base} does not match the registered engine or public model"
            )));
        }
        Self::try_new(
            primary.engine_model_id.clone(),
            primary.kind,
            base_entries
                .iter()
                .map(|entry| entry.public_name.to_string())
                .collect(),
            adapters,
        )
    }

    pub fn resolve(
        &self,
        request_model: &str,
        required_kind: ServedModelKind,
    ) -> Option<&ServedModelEntry> {
        let entry = self
            .entry_by_public_name
            .get(request_model)
            .and_then(|index| self.entries.get(*index))?;
        (entry.kind == required_kind).then_some(entry)
    }
}

fn require_same_base_model(entries: &[&ServedModelEntry]) -> Result<(), ServedModelRegistryError> {
    let Some(first) = entries.first() else {
        return Ok(());
    };
    if entries
        .iter()
        .any(|entry| entry.engine_model_id != first.engine_model_id || entry.kind != first.kind)
    {
        return Err(ServedModelRegistryError(
            "served model aliases do not resolve to one engine model and kind".to_string(),
        ));
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<(), ServedModelRegistryError> {
    if value.is_empty() || value.trim() != value {
        return Err(ServedModelRegistryError(format!(
            "{label} must be non-empty and have no surrounding whitespace"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_public_aliases_to_one_engine_model() {
        let registry = ServedModelRegistry::try_new(
            "internal-model",
            ServedModelKind::Llm,
            vec!["public-model".to_string(), "short-alias".to_string()],
            vec![],
        )
        .unwrap();

        for name in ["public-model", "short-alias"] {
            let entry = registry.resolve(name, ServedModelKind::Llm).unwrap();
            assert_eq!(entry.engine_model_id(), &ModelId::new("internal-model"));
            assert!(entry.adapter().is_none());
        }
        assert!(registry
            .resolve("internal-model", ServedModelKind::Llm)
            .is_none());
        assert!(registry
            .resolve("public-model", ServedModelKind::Embedding)
            .is_none());
    }

    #[test]
    fn resolves_lora_model_to_internal_id_and_public_parent() {
        let registry = ServedModelRegistry::try_new(
            "internal-model",
            ServedModelKind::Llm,
            vec!["public-model".to_string()],
            vec![LoraAdapterModel::new(
                "sql",
                "public-model:sql",
                "/models/sql",
            )],
        )
        .unwrap();

        let entry = registry
            .resolve("public-model:sql", ServedModelKind::Llm)
            .unwrap();
        assert_eq!(entry.engine_model_id(), &ModelId::new("internal-model"));
        assert_eq!(entry.adapter().unwrap().name, "sql");
        assert_eq!(
            entry.parent_public_name().map(ServedModelName::as_str),
            Some("public-model")
        );
    }

    #[test]
    fn adding_lora_preserves_all_public_base_aliases() {
        let base = ServedModelRegistry::try_new(
            "internal-model",
            ServedModelKind::Llm,
            vec!["public-model".to_string(), "short-alias".to_string()],
            vec![],
        )
        .unwrap();
        let registry = base
            .try_with_lora_adapters(
                "public-model",
                vec![LoraAdapterModel::new(
                    "sql",
                    "public-model:sql",
                    "/models/sql",
                )],
            )
            .unwrap();

        assert!(registry
            .resolve("short-alias", ServedModelKind::Llm)
            .is_some());
        assert_eq!(
            registry
                .resolve("public-model:sql", ServedModelKind::Llm)
                .unwrap()
                .parent_public_name()
                .map(ServedModelName::as_str),
            Some("public-model")
        );
    }

    #[test]
    fn rejects_ambiguous_or_invalid_public_names() {
        for result in [
            ServedModelRegistry::try_new("internal", ServedModelKind::Llm, vec![], vec![]),
            ServedModelRegistry::try_new(
                "internal",
                ServedModelKind::Llm,
                vec!["same".to_string(), "same".to_string()],
                vec![],
            ),
            ServedModelRegistry::try_new(
                "internal",
                ServedModelKind::Llm,
                vec!["base".to_string()],
                vec![LoraAdapterModel::new("sql", "base", "/models/sql")],
            ),
        ] {
            assert!(result.is_err());
        }
    }
}
