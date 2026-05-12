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
    pub version: u32,
    pub id: String,
    pub species_name: String,
    pub discovered_at_unix: u64,
    pub discovery_rank: u64,
    pub promotion_reason: String,
    pub morphology: String,
    pub rarity: String,
    pub tags: Vec<String>,
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
    pub mass: f32,
    pub channels: usize,
    pub rules: usize,
    pub kernel_radius: i32,
    pub morphology: String,
    pub rarity: String,
    pub tags: Vec<String>,
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
                if let Ok(mut index) = serde_json::from_str::<Bestiary>(&data) {
                    index.version = 2;
                    return index;
                }
            }
        }

        let now = unix_now();
        Self {
            version: 2,
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
        let morphology = morphology(record);
        let rarity = rarity(record);
        let tags = tags(record);
        let species_name = species_name(
            record.seed,
            record.channels,
            record.motion_score,
            record.entropy_score,
            &morphology,
        );
        let id = species_id(&species_name, record.seed, self.total_discoveries + 1);

        let entry = BestiaryEntry {
            version: 2,
            id: id.clone(),
            species_name: species_name.clone(),
            discovered_at_unix: now,
            discovery_rank: self.total_discoveries + 1,
            promotion_reason: reason.clone(),
            morphology: morphology.clone(),
            rarity: rarity.clone(),
            tags: tags.clone(),
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
            mass: record.mass,
            channels: record.channels,
            rules: record.rules,
            kernel_radius: record.kernel_radius,
            morphology,
            rarity,
            tags,
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
            "bestiary discovered: {} [{} / {}]",
            species_name, reason, entry.rarity
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
        let rare = self
            .entries
            .iter()
            .filter(|entry| entry.rarity == "rare" || entry.rarity == "mythic")
            .count();

        format!("bestiary species={} rare+={}", self.entries.len(), rare)
    }
}

fn promotion_reason(record: &RunRecord) -> Option<String> {
    if record.score >= 0.72 {
        return Some("high survival score".to_string());
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

fn morphology(record: &RunRecord) -> String {
    if record.motion_score >= 0.140 && record.entropy_score >= 0.420 {
        "swimmer".to_string()
    } else if record.motion_score >= 0.095 {
        "drifter".to_string()
    } else if record.entropy_score >= 0.600 {
        "bloom".to_string()
    } else if record.mass >= 0.340 {
        "reef".to_string()
    } else if record.mass <= 0.070 && record.entropy_score >= 0.250 {
        "spore".to_string()
    } else if record.positive_rules >= record.negative_rules * 2 {
        "radiant".to_string()
    } else if record.negative_rules >= record.positive_rules * 2 {
        "shadow".to_string()
    } else {
        "orbium".to_string()
    }
}

fn rarity(record: &RunRecord) -> String {
    if record.score >= 0.84 || (record.motion_score >= 0.18 && record.entropy_score >= 0.55) {
        "mythic".to_string()
    } else if record.score >= 0.72 || record.motion_score >= 0.12 {
        "rare".to_string()
    } else if record.score >= 0.58 || record.entropy_score >= 0.42 {
        "uncommon".to_string()
    } else {
        "common".to_string()
    }
}

fn tags(record: &RunRecord) -> Vec<String> {
    let mut tags = Vec::new();

    if record.motion_score >= 0.100 {
        tags.push("mobile".to_string());
    }
    if record.entropy_score >= 0.450 {
        tags.push("complex".to_string());
    }
    if record.mass >= 0.300 {
        tags.push("dense".to_string());
    }
    if record.mass <= 0.080 {
        tags.push("sparse".to_string());
    }
    if record.channels >= 6 {
        tags.push("six-channel".to_string());
    }
    if record.rules >= 14 {
        tags.push("rule-rich".to_string());
    }
    if record.kernel_radius >= 6 {
        tags.push("wide-kernel".to_string());
    }
    if record.positive_rules > record.negative_rules {
        tags.push("growth-biased".to_string());
    }
    if record.negative_rules > record.positive_rules {
        tags.push("decay-biased".to_string());
    }

    if tags.is_empty() {
        tags.push("stable".to_string());
    }

    tags
}

fn species_name(seed: u64, channels: usize, motion: f32, entropy: f32, morphology: &str) -> String {
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

    let bodies = match morphology {
        "swimmer" => [
            "Manta", "Ray", "Glider", "Serpent", "Nautilus", "Medusa", "Drifter", "Tide",
        ],
        "drifter" => [
            "Drifter", "Wisp", "Mote", "Lantern", "Pulse", "Orbium", "Ray", "Tide",
        ],
        "bloom" => [
            "Bloom", "Spore", "Reef", "Halo", "Pulse", "Medusa", "Wisp", "Orbium",
        ],
        "reef" => [
            "Reef", "Bloom", "Nautilus", "Halo", "Orbium", "Lantern", "Spore", "Medusa",
        ],
        "spore" => [
            "Spore", "Mote", "Wisp", "Lantern", "Pulse", "Bloom", "Orbium", "Drifter",
        ],
        "shadow" => [
            "Umbra", "Ghost", "Wraith", "Nocturne", "Abyss", "Mote", "Wisp", "Serpent",
        ],
        _ => [
            "Orbium", "Pulse", "Bloom", "Drifter", "Wisp", "Halo", "Nautilus", "Mote",
        ],
    };

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
