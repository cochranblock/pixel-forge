// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Ring buffer for swipe data — bincode+zstd on disk.
//! Stores last 200 swipes (sprite pixel data + good/bad label).
//! ~154 KB max (200 × 769 bytes).
//!
//! Atomic writes: temp file + rename prevents corruption on crash.
//! Self-healing: corrupted store loads as empty (user just re-swipes).

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Pixel stride for 32×32 RGB channel-first: 3 × 32 × 32 = 3072 floats.
pub const PIXEL_STRIDE: usize = 3072;

/// Single swipe: 32×32×3 RGB f32 pixels + bool label.
#[derive(Clone, Serialize, Deserialize)]
pub struct Swipe {
    /// Channel-first f32 pixels [R..., G..., B...], len = PIXEL_STRIDE
    pub pixels: Vec<f32>,
    /// true = good (swipe right), false = bad (swipe left)
    pub good: bool,
    /// Class ID of the sprite (0-14)
    pub class_id: u32,
    /// Timestamp (seconds since UNIX epoch) — for recency weighting
    pub timestamp: u64,
}

/// Stats about the swipe buffer state.
#[derive(Debug, Clone)]
pub struct SwipeStats {
    pub buffered: usize,
    pub total_ever: usize,
    pub good: usize,
    pub bad: usize,
    pub oldest_ts: Option<u64>,
    pub newest_ts: Option<u64>,
    /// Per-class counts: (class_id, good_count, bad_count)
    pub by_class: Vec<(u32, usize, usize)>,
}

/// Ring buffer of swipes, persisted as bincode+zstd.
///
/// Capacity is fixed at construction. When full, the oldest swipe is evicted.
/// All disk writes are atomic (write to .tmp, rename over target).
#[derive(Serialize, Deserialize)]
pub struct SwipeStore {
    swipes: Vec<Swipe>,
    capacity: usize,
    /// Total swipes ever recorded (not just those in buffer)
    pub total: usize,
    /// Monotonic counter — bumps on every mutation for dirty-checking
    generation: u64,
}

impl SwipeStore {
    pub fn new(capacity: usize) -> Self {
        Self {
            swipes: Vec::with_capacity(capacity),
            capacity,
            total: 0,
            generation: 0,
        }
    }

    /// Load from disk. Returns empty store on missing file or corruption.
    /// Never fails — a fresh store is always valid.
    pub fn load_or_default(path: &str, capacity: usize) -> Self {
        match Self::try_load(path) {
            Ok(mut store) => {
                // Adjust capacity if it changed between versions
                store.capacity = capacity;
                store
            }
            Err(_) => Self::new(capacity),
        }
    }

    fn try_load(path: &str) -> Result<Self> {
        let compressed = std::fs::read(Path::new(path))?;
        let decompressed = zstd::decode_all(compressed.as_slice())?;
        let store: SwipeStore = bincode::deserialize(&decompressed)?;
        // Validate pixel data integrity
        for swipe in &store.swipes {
            if swipe.pixels.len() != PIXEL_STRIDE {
                anyhow::bail!("corrupt swipe: pixel len {} != {}", swipe.pixels.len(), PIXEL_STRIDE);
            }
            if swipe.class_id >= crate::tiny_unet::NUM_CLASSES as u32 {
                anyhow::bail!("corrupt swipe: class_id {} out of range", swipe.class_id);
            }
        }
        Ok(store)
    }

    /// Atomic save: write to .tmp then rename. Can't corrupt the real file.
    pub fn save(&self, path: &str) -> Result<()> {
        let target = PathBuf::from(path);
        let tmp = target.with_extension("tmp");

        let encoded = bincode::serialize(self)?;
        let compressed = zstd::encode_all(encoded.as_slice(), 3)?;
        std::fs::write(&tmp, &compressed)?;
        std::fs::rename(&tmp, &target)?;
        Ok(())
    }

    /// Add a swipe. Drops oldest if at capacity. Returns new buffer length.
    pub fn push(&mut self, swipe: Swipe) -> usize {
        if self.swipes.len() >= self.capacity {
            self.swipes.remove(0);
        }
        self.swipes.push(swipe);
        self.total += 1;
        self.generation += 1;
        self.swipes.len()
    }

    /// Add a swipe from raw parts (convenience for the UI layer).
    pub fn record(
        &mut self,
        pixels: Vec<f32>,
        good: bool,
        class_id: u32,
    ) -> usize {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.push(Swipe { pixels, good, class_id, timestamp })
    }

    /// All current swipes.
    pub fn swipes(&self) -> &[Swipe] {
        &self.swipes
    }

    /// Number of swipes in buffer.
    pub fn len(&self) -> usize {
        self.swipes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.swipes.is_empty()
    }

    /// Whether the Judge should activate (minimum viable training data).
    pub fn judge_ready(&self) -> bool {
        let good = self.good_count();
        let bad = self.bad_count();
        // Need at least 5 of each for meaningful binary classification
        good >= 5 && bad >= 5
    }

    /// Whether it's time to retrain (every 5 new swipes).
    pub fn should_retrain(&self) -> bool {
        self.total >= 10 && self.total % 5 == 0
    }

    /// Whether it's time to update LoRA (every 20 swipes).
    pub fn should_update_lora(&self) -> bool {
        self.total >= 20 && self.total % 20 == 0
    }

    /// Current generation counter for dirty-checking.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Count of good swipes in buffer.
    pub fn good_count(&self) -> usize {
        self.swipes.iter().filter(|s| s.good).count()
    }

    /// Count of bad swipes in buffer.
    pub fn bad_count(&self) -> usize {
        self.swipes.iter().filter(|s| !s.good).count()
    }

    /// Full stats snapshot.
    pub fn stats(&self) -> SwipeStats {
        let mut by_class: std::collections::HashMap<u32, (usize, usize)> =
            std::collections::HashMap::new();
        for s in &self.swipes {
            let entry = by_class.entry(s.class_id).or_insert((0, 0));
            if s.good { entry.0 += 1; } else { entry.1 += 1; }
        }
        let mut class_vec: Vec<(u32, usize, usize)> =
            by_class.into_iter().map(|(k, (g, b))| (k, g, b)).collect();
        class_vec.sort_by_key(|&(id, _, _)| id);

        SwipeStats {
            buffered: self.swipes.len(),
            total_ever: self.total,
            good: self.good_count(),
            bad: self.bad_count(),
            oldest_ts: self.swipes.first().map(|s| s.timestamp),
            newest_ts: self.swipes.last().map(|s| s.timestamp),
            by_class: class_vec,
        }
    }

    /// Extract training tensors: (pixel_batch, label_batch) suitable for the Judge.
    /// Returns (pixels: [N, 3, 16, 16] flat, labels: [N] as f32 0.0/1.0).
    pub fn training_data(&self) -> (Vec<f32>, Vec<f32>) {
        let mut pixels = Vec::with_capacity(self.swipes.len() * PIXEL_STRIDE);
        let mut labels = Vec::with_capacity(self.swipes.len());
        for s in &self.swipes {
            pixels.extend_from_slice(&s.pixels);
            labels.push(if s.good { 1.0f32 } else { 0.0f32 });
        }
        (pixels, labels)
    }

    /// Get only the good sprites, grouped by class — input for the Combiner.
    pub fn accepted_by_class(&self) -> std::collections::HashMap<u32, Vec<&[f32]>> {
        let mut map: std::collections::HashMap<u32, Vec<&[f32]>> =
            std::collections::HashMap::new();
        for s in &self.swipes {
            if s.good {
                map.entry(s.class_id).or_default().push(&s.pixels);
            }
        }
        map
    }
}

impl std::fmt::Display for SwipeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "swipes: {}/{} buffered, {} total | good: {} bad: {}",
            self.buffered, 200, self.total_ever, self.good, self.bad)?;
        if !self.by_class.is_empty() {
            write!(f, " | classes: ")?;
            for (i, &(cid, g, b)) in self.by_class.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "c{}:{}/{}", cid, g, b)?;
            }
        }
        Ok(())
    }
}
