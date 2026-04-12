// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! ClassRouter — deterministic Rust routing, not inference.
//! Maps class name → model path + diffusion hyperparams.
//! Loaded from class_config.toml.

use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Per-class diffusion config.
#[derive(Clone, Debug)]
pub struct DiffusionConfig {
    pub model_path: String,
    pub steps: usize,
    pub guidance_scale: f64,
    pub scheduler: String,
    pub channels: Vec<usize>,
}

#[derive(Deserialize)]
struct TomlConfig {
    defaults: TomlDefaults,
    #[serde(default)]
    class: HashMap<String, TomlClass>,
}

#[derive(Deserialize)]
struct TomlDefaults {
    steps: usize,
    guidance_scale: f64,
    scheduler: String,
}

#[derive(Deserialize)]
struct TomlClass {
    model: String,
    #[serde(default)]
    classes: Vec<String>,
    steps: Option<usize>,
    guidance_scale: Option<f64>,
    scheduler: Option<String>,
    channels: Option<Vec<usize>>,
}

/// ClassRouter — loads once, routes many.
pub struct ClassRouter {
    /// class_name → silo_name
    class_to_silo: HashMap<String, String>,
    /// silo_name → DiffusionConfig
    silo_configs: HashMap<String, DiffusionConfig>,
    defaults: TomlDefaults,
}

impl ClassRouter {
    /// Load from a TOML config file.
    pub fn load(path: &str) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let config: TomlConfig = toml::from_str(&text)?;

        let mut class_to_silo = HashMap::new();
        let mut silo_configs = HashMap::new();

        for (silo_name, entry) in &config.class {
            let channels = entry.channels.clone().unwrap_or_else(|| vec![16, 32]);
            let dc = DiffusionConfig {
                model_path: entry.model.clone(),
                steps: entry.steps.unwrap_or(config.defaults.steps),
                guidance_scale: entry.guidance_scale.unwrap_or(config.defaults.guidance_scale),
                scheduler: entry.scheduler.clone().unwrap_or_else(|| config.defaults.scheduler.clone()),
                channels,
            };
            silo_configs.insert(silo_name.clone(), dc);

            // Map each class in this silo to the silo name
            for class_name in &entry.classes {
                class_to_silo.insert(class_name.clone(), silo_name.clone());
            }
            // The silo name itself is also a valid class
            class_to_silo.insert(silo_name.clone(), silo_name.clone());
        }

        Ok(Self { class_to_silo, silo_configs, defaults: config.defaults })
    }

    /// Route a class name to its DiffusionConfig.
    /// Falls back to "misc" silo, then returns None.
    pub fn route(&self, class_name: &str) -> Option<&DiffusionConfig> {
        let silo = self.class_to_silo.get(class_name)
            .or_else(|| self.class_to_silo.get("misc"))?;
        self.silo_configs.get(silo)
    }

    /// List all silo names.
    pub fn silos(&self) -> Vec<&str> {
        self.silo_configs.keys().map(|s| s.as_str()).collect()
    }

    /// List all classes that route to a given silo.
    pub fn classes_for_silo(&self, silo: &str) -> Vec<&str> {
        self.class_to_silo.iter()
            .filter(|(_, v)| v.as_str() == silo)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Check if a config file exists at the given path.
    pub fn exists(path: &str) -> bool {
        Path::new(path).exists()
    }
}
