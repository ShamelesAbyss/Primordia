use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
};

use crate::chronicle::{unix_now, RunRecord};

const BESTIARY_DIR: &str = "saves/bestiary";
const BESTIARY_INDEX: &str = "saves/bestiary/index.json";
const MAX_ENTRIES: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestiaryEntry {
    pub id: String,
    pub species_name: String,
    pub discovered_at_unix: u64,
    pub promotion_reason: String,
    pub record: RunRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestiaryIndexEntry {
    pub id: String,
    pub species_name: String,
    pub seed: u64,
    pub score: f32,
    pub motion_score: f32,
    pub entropy_score: f32,
    pub channels: usize,
    pub rules: usize,
    pub kernel_radius: i32,
    pub discovered_at_unix: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bestiary {
    pub version: u32,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
    pub total_discoveries: u64,
    pub entries: Vec<BestiaryIndexEntry>,
}

impl Bestiary {
    pub fn load_or_new() -> Self {
        if let Ok(mut file) = File::open(BESTIARY_INDEX) {
            let mut data = String::new();
            if file.read_to_string(&mut data).is_ok() {
                if let Ok(index) = serde_json::from_str::<Bestiary>(&data) {
                    return index;
                }
            }
        }

        let now = unix_now();
        Self {
            version: 1,
            created_at_unix: now,
            updated_at_unix: now,
            total_discoveries: 0,
            entries: Vec::new(),
        }
    }

    pub fn consider(&mut self, record: &RunRecord) -> Result<Option<String>> {
        if self.entries.iter().any(|entry| entry.seed == record.seed) {
            return Ok(None);
        }

        let Some(reason) = promotion_reason(record) else {
            return Ok(None);
        };

        fs::create_dir_all(BESTIARY_DIR)?;

        let now = unix_now();
        let species_name = species_name(
            record.seed,
            record.channels,
            record.motion_score,
            record.entropy_score,
        );
        let id = species_id(&species_name, record.seed, self.total_discoveries + 1);

        let entry = BestiaryEntry {
            id: id.clone(),
            species_name: species_name.clone(),
            discovered_at_unix: now,
            promotion_reason: reason.clone(),
            record: record.clone(),
        };

        let path = format!("{}/{}.json", BESTIARY_DIR, id);
        let json = serde_json::to_string_pretty(&entry)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;

        self.entries.push(BestiaryIndexEntry {
            id: id.clone(),
            species_name: species_name.clone(),
            seed: record.seed,
            score: record.score,
            motion_score: record.motion_score,
            entropy_score: record.entropy_score,
            channels: record.channels,
            rules: record.rules,
            kernel_radius: record.kernel_radius,
            discovered_at_unix: now,
        });

        self.entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.entries.truncate(MAX_ENTRIES);

        self.total_discoveries += 1;
        self.updated_at_unix = now;
        self.save()?;

        Ok(Some(format!(
            "bestiary discovered: {} [{}]",
            species_name, reason
        )))
    }

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = Path::new(BESTIARY_INDEX).parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(BESTIARY_INDEX)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn status(&self) -> String {
        format!("bestiary species={}", self.entries.len())
    }
}

fn promotion_reason(record: &RunRecord) -> Option<String> {
    if record.score >= 0.72 {
        return Some("high overall survival score".to_string());
    }

    if record.motion_score >= 0.060 && record.entropy_score >= 0.300 && record.score >= 0.48 {
        return Some("mobile complex behavior".to_string());
    }

    if record.entropy_score >= 0.520 && record.score >= 0.52 {
        return Some("high living complexity".to_string());
    }

    if record.motion_score >= 0.110 && record.score >= 0.42 {
        return Some("strong drift signature".to_string());
    }

    None
}

fn species_name(seed: u64, channels: usize, motion: f32, entropy: f32) -> String {
    let prefixes = [
        "Abyssal",
        "Neon",
        "Crystal",
        "Void",
        "Radiant",
        "Lunar",
        "Echo",
        "Plasma",
        "Ghost",
        "Prismatic",
        "Ancient",
        "Iridescent",
        "Feral",
        "Oceanic",
        "Violet",
        "Solar",
    ];

    let bodies = [
        "Drifter", "Manta", "Wisp", "Ray", "Nautilus", "Medusa", "Bloom", "Serpent", "Tide",
        "Orbium", "Spore", "Lantern", "Glider", "Mote", "Reef", "Pulse",
    ];

    let suffixes = [
        "Prime", "Minor", "Major", "Vesper", "Aster", "Umbra", "Pelagic", "Nocturne", "Genesis",
        "Eidolon", "Fractal", "Lucent", "Abyss", "Halo", "Myriad", "Nova",
    ];

    let a = prefixes[(seed as usize) % prefixes.len()];
    let b = bodies[((seed >> 11) as usize + channels) % bodies.len()];
    let c = suffixes[((seed >> 23) as usize
        + (motion * 1000.0) as usize
        + (entropy * 1000.0) as usize)
        % suffixes.len()];

    format!("{} {} {}", a, b, c)
}

fn species_id(name: &str, seed: u64, count: u64) -> String {
    let slug = name
        .to_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");

    format!("{}-{:03}-{:x}", slug, count, seed & 0xffff)
}
