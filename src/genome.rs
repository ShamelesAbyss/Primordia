use anyhow::Result;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

use crate::chronicle::unix_now;

const GENOME_DIR: &str = "saves/genomes";
const GENOME_INDEX: &str = "saves/genomes/index.json";
const MAX_GENOMES: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelTapGenome {
    pub dx: i32,
    pub dy: i32,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleGenome {
    pub from: usize,
    pub to: usize,
    pub mu: f32,
    pub sigma: f32,
    pub weight: f32,
    pub taps: Vec<KernelTapGenome>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeSnapshot {
    pub version: u32,
    pub seed: u64,
    pub saved_at_unix: u64,
    pub reason: String,
    pub tick: u64,
    pub field_w: usize,
    pub field_h: usize,
    pub channels: usize,
    pub base_rules: usize,
    pub kernel_radius: i32,
    pub motion_score: f32,
    pub entropy_score: f32,
    pub mass: f32,
    pub rules: Vec<RuleGenome>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeIndexEntry {
    pub id: String,
    pub seed: u64,
    pub saved_at_unix: u64,
    pub reason: String,
    pub channels: usize,
    pub rules: usize,
    pub kernel_radius: i32,
    pub motion_score: f32,
    pub entropy_score: f32,
    pub mass: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeVault {
    pub version: u32,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
    pub total_saved: u64,
    pub entries: Vec<GenomeIndexEntry>,
}

impl GenomeVault {
    pub fn load_or_new() -> Self {
        if let Ok(data) = std::fs::read_to_string(GENOME_INDEX) {
            if let Ok(mut vault) = serde_json::from_str::<GenomeVault>(&data) {
                vault.version = 2;
                return vault;
            }
        }

        let now = unix_now();
        Self {
            version: 2,
            created_at_unix: now,
            updated_at_unix: now,
            total_saved: 0,
            entries: Vec::new(),
        }
    }

    pub fn save_snapshot(&mut self, snapshot: &GenomeSnapshot) -> Result<String> {
        fs::create_dir_all(GENOME_DIR)?;

        let id = genome_id(
            snapshot.seed,
            snapshot.tick,
            self.total_saved + 1,
            &snapshot.reason,
        );
        let path = format!("{}/{}.json", GENOME_DIR, id);

        let json = serde_json::to_string_pretty(snapshot)?;
        let mut file = File::create(&path)?;
        file.write_all(json.as_bytes())?;

        self.entries.push(GenomeIndexEntry {
            id: id.clone(),
            seed: snapshot.seed,
            saved_at_unix: snapshot.saved_at_unix,
            reason: snapshot.reason.clone(),
            channels: snapshot.channels,
            rules: snapshot.rules.len(),
            kernel_radius: snapshot.kernel_radius,
            motion_score: snapshot.motion_score,
            entropy_score: snapshot.entropy_score,
            mass: snapshot.mass,
        });

        self.entries.sort_by(|a, b| {
            let a_score = a.motion_score * 0.45 + a.entropy_score * 0.35 + a.mass * 0.20;
            let b_score = b.motion_score * 0.45 + b.entropy_score * 0.35 + b.mass * 0.20;
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.entries.truncate(MAX_GENOMES);
        self.total_saved += 1;
        self.updated_at_unix = unix_now();
        self.save()?;

        Ok(id)
    }

    pub fn load_random_snapshot(&self) -> Result<Option<(String, GenomeSnapshot)>> {
        if self.entries.is_empty() {
            return Ok(None);
        }

        let seed = unix_now() ^ self.total_saved ^ self.entries.len() as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let index = rng.gen_range(0..self.entries.len());
        self.load_snapshot_by_index(index)
    }

    pub fn load_best_snapshot(&self) -> Result<Option<(String, GenomeSnapshot)>> {
        if self.entries.is_empty() {
            return Ok(None);
        }

        self.load_snapshot_by_index(0)
    }

    fn load_snapshot_by_index(&self, index: usize) -> Result<Option<(String, GenomeSnapshot)>> {
        let Some(entry) = self.entries.get(index) else {
            return Ok(None);
        };

        let path = format!("{}/{}.json", GENOME_DIR, entry.id);
        let data = std::fs::read_to_string(path)?;
        let snapshot = serde_json::from_str::<GenomeSnapshot>(&data)?;
        Ok(Some((entry.id.clone(), snapshot)))
    }

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = Path::new(GENOME_INDEX).parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(GENOME_INDEX)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn status(&self) -> String {
        format!("genomes={}", self.entries.len())
    }
}

fn genome_id(seed: u64, tick: u64, count: u64, reason: &str) -> String {
    let clean_reason = reason
        .to_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");

    format!(
        "genome-{}-{:03}-{:x}-t{}",
        clean_reason,
        count,
        seed & 0xffff,
        tick
    )
}
