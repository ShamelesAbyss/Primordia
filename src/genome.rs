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
    #[serde(default)]
    pub genome_id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub co_parent_id: Option<String>,
    #[serde(default)]
    pub generation: u32,
    #[serde(default)]
    pub branch_label: String,
    #[serde(default)]
    pub mutation_strength: f32,

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

    #[serde(default)]
    pub cells: Vec<f32>,
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
    #[serde(default)]
    pub has_body_snapshot: bool,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub co_parent_id: Option<String>,
    #[serde(default)]
    pub generation: u32,
    #[serde(default)]
    pub branch_label: String,
    #[serde(default)]
    pub mutation_strength: f32,
    #[serde(default)]
    pub children_count: u32,
    #[serde(default)]
    pub times_loaded: u32,
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
                vault.version = 5;
                for entry in &mut vault.entries {
                    if entry.branch_label.is_empty() {
                        entry.branch_label = branch_label(entry.seed);
                    }
                }
                return vault;
            }
        }

        let now = unix_now();
        Self {
            version: 5,
            created_at_unix: now,
            updated_at_unix: now,
            total_saved: 0,
            entries: Vec::new(),
        }
    }

    pub fn save_snapshot(&mut self, snapshot: &GenomeSnapshot) -> Result<String> {
        fs::create_dir_all(GENOME_DIR)?;

        let id = if snapshot.genome_id.is_empty() {
            genome_id(
                snapshot.seed,
                snapshot.tick,
                self.total_saved + 1,
                &snapshot.reason,
            )
        } else {
            snapshot.genome_id.clone()
        };

        let mut snapshot = snapshot.clone();
        snapshot.version = 5;
        snapshot.genome_id = id.clone();
        snapshot.saved_at_unix = unix_now();

        if snapshot.branch_label.is_empty() {
            snapshot.branch_label = branch_label(snapshot.seed);
        }

        let path = format!("{}/{}.json", GENOME_DIR, id);
        let json = serde_json::to_string_pretty(&snapshot)?;
        let mut file = File::create(&path)?;
        file.write_all(json.as_bytes())?;

        for parent_id in [snapshot.parent_id.as_ref(), snapshot.co_parent_id.as_ref()]
            .into_iter()
            .flatten()
        {
            if let Some(parent) = self.entries.iter_mut().find(|entry| &entry.id == parent_id) {
                parent.children_count = parent.children_count.saturating_add(1);
            }
        }

        if let Some(existing) = self.entries.iter_mut().find(|entry| entry.id == id) {
            existing.saved_at_unix = snapshot.saved_at_unix;
            existing.reason = snapshot.reason.clone();
            existing.channels = snapshot.channels;
            existing.rules = snapshot.rules.len();
            existing.kernel_radius = snapshot.kernel_radius;
            existing.motion_score = snapshot.motion_score;
            existing.entropy_score = snapshot.entropy_score;
            existing.mass = snapshot.mass;
            existing.has_body_snapshot = !snapshot.cells.is_empty();
            existing.parent_id = snapshot.parent_id.clone();
            existing.co_parent_id = snapshot.co_parent_id.clone();
            existing.generation = snapshot.generation;
            existing.branch_label = snapshot.branch_label.clone();
            existing.mutation_strength = snapshot.mutation_strength;
        } else {
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
                has_body_snapshot: !snapshot.cells.is_empty(),
                parent_id: snapshot.parent_id.clone(),
                co_parent_id: snapshot.co_parent_id.clone(),
                generation: snapshot.generation,
                branch_label: snapshot.branch_label.clone(),
                mutation_strength: snapshot.mutation_strength,
                children_count: 0,
                times_loaded: 0,
            });
        }

        self.entries.sort_by(|a, b| {
            score_entry(b)
                .partial_cmp(&score_entry(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.entries.truncate(MAX_GENOMES);
        self.total_saved += 1;
        self.updated_at_unix = unix_now();
        self.save()?;

        Ok(id)
    }

    pub fn load_random_snapshot(&mut self) -> Result<Option<(String, GenomeSnapshot)>> {
        if self.entries.is_empty() {
            return Ok(None);
        }

        let seed = unix_now() ^ self.total_saved ^ self.entries.len() as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let index = rng.gen_range(0..self.entries.len());
        self.load_snapshot_by_index(index)
    }

    pub fn load_best_snapshot(&mut self) -> Result<Option<(String, GenomeSnapshot)>> {
        if self.entries.is_empty() {
            return Ok(None);
        }

        self.load_snapshot_by_index(0)
    }

    pub fn mutated_best_snapshot(&mut self) -> Result<Option<(String, GenomeSnapshot)>> {
        let Some((parent_id, parent)) = self.load_best_snapshot()? else {
            return Ok(None);
        };

        let seed =
            unix_now() ^ parent.seed ^ parent.tick ^ self.total_saved ^ 0xA17E_5EED_DA7A_BA5E;

        let mut rng = StdRng::seed_from_u64(seed);
        let strength = rng.gen_range(0.015..0.085);
        let mut child = parent.clone();

        child.version = 5;
        child.genome_id = genome_id(seed, 0, self.total_saved + 1, "mutated_offspring");
        child.parent_id = Some(parent_id.clone());
        child.co_parent_id = None;
        child.generation = parent.generation.saturating_add(1);
        child.branch_label = if parent.branch_label.is_empty() {
            branch_label(parent.seed)
        } else {
            parent.branch_label.clone()
        };
        child.mutation_strength = strength;
        child.seed = seed;
        child.reason = "mutated_offspring".to_string();
        child.tick = 0;
        child.motion_score = 0.0;
        child.entropy_score = 0.0;
        child.mass = 0.0;
        child.saved_at_unix = unix_now();

        mutate_rules(&mut child, &mut rng, strength);

        if rng.gen_bool(0.55) {
            child.cells.clear();
        } else {
            for cell in &mut child.cells {
                let drift = rng.gen_range(-strength..strength) * 0.35;
                *cell = (*cell + drift).clamp(0.0, 1.0);
            }
        }

        Ok(Some((parent_id, child)))
    }

    pub fn mutated_current_snapshot(
        &mut self,
        snapshot: &GenomeSnapshot,
    ) -> Result<GenomeSnapshot> {
        let seed = unix_now()
            ^ snapshot.seed
            ^ snapshot.tick
            ^ self.total_saved
            ^ 0xC0DE_C0DE_A11F_EE17_u64;

        let mut rng = StdRng::seed_from_u64(seed);
        let strength = rng.gen_range(0.012..0.075);
        let mut child = snapshot.clone();

        child.version = 5;
        child.genome_id = genome_id(seed, 0, self.total_saved + 1, "mutated_current");
        child.parent_id = if snapshot.genome_id.is_empty() {
            None
        } else {
            Some(snapshot.genome_id.clone())
        };
        child.co_parent_id = None;
        child.generation = snapshot.generation.saturating_add(1);
        child.branch_label = if snapshot.branch_label.is_empty() {
            branch_label(snapshot.seed)
        } else {
            snapshot.branch_label.clone()
        };
        child.mutation_strength = strength;
        child.seed = seed;
        child.reason = "mutated_current".to_string();
        child.tick = 0;
        child.motion_score = 0.0;
        child.entropy_score = 0.0;
        child.mass = 0.0;
        child.saved_at_unix = unix_now();

        mutate_rules(&mut child, &mut rng, strength);

        child.cells.clear();

        Ok(child)
    }
    pub fn breed_best_two(&mut self) -> Result<Option<(String, String, GenomeSnapshot)>> {
        if self.entries.len() < 2 {
            return Ok(None);
        }

        let Some((id_a, parent_a)) = self.load_snapshot_by_index(0)? else {
            return Ok(None);
        };
        let Some((id_b, parent_b)) = self.load_snapshot_by_index(1)? else {
            return Ok(None);
        };

        let child = breed_snapshots(
            &id_a,
            &parent_a,
            &id_b,
            &parent_b,
            self.total_saved + 1,
            false,
        );
        Ok(Some((id_a, id_b, child)))
    }

    pub fn breed_random_two(&mut self) -> Result<Option<(String, String, GenomeSnapshot)>> {
        if self.entries.len() < 2 {
            return Ok(None);
        }

        let seed = unix_now() ^ self.total_saved ^ 0xBEE5_B1EED;
        let mut rng = StdRng::seed_from_u64(seed);

        let a = rng.gen_range(0..self.entries.len());
        let mut b = rng.gen_range(0..self.entries.len());
        while b == a {
            b = rng.gen_range(0..self.entries.len());
        }

        let Some((id_a, parent_a)) = self.load_snapshot_by_index(a)? else {
            return Ok(None);
        };
        let Some((id_b, parent_b)) = self.load_snapshot_by_index(b)? else {
            return Ok(None);
        };

        let child = breed_snapshots(
            &id_a,
            &parent_a,
            &id_b,
            &parent_b,
            self.total_saved + 1,
            true,
        );
        Ok(Some((id_a, id_b, child)))
    }

    fn load_snapshot_by_index(&mut self, index: usize) -> Result<Option<(String, GenomeSnapshot)>> {
        let Some(entry) = self.entries.get_mut(index) else {
            return Ok(None);
        };

        entry.times_loaded = entry.times_loaded.saturating_add(1);
        let id = entry.id.clone();

        let path = format!("{}/{}.json", GENOME_DIR, id);
        let data = std::fs::read_to_string(path)?;
        let mut snapshot = serde_json::from_str::<GenomeSnapshot>(&data)?;
        snapshot.version = snapshot.version.max(5);

        if snapshot.genome_id.is_empty() {
            snapshot.genome_id = id.clone();
        }
        if snapshot.branch_label.is_empty() {
            snapshot.branch_label = branch_label(snapshot.seed);
        }

        self.updated_at_unix = unix_now();
        self.save()?;

        Ok(Some((id, snapshot)))
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
        let body_count = self
            .entries
            .iter()
            .filter(|entry| entry.has_body_snapshot)
            .count();

        let max_generation = self
            .entries
            .iter()
            .map(|entry| entry.generation)
            .max()
            .unwrap_or(0);

        let hybrid_count = self
            .entries
            .iter()
            .filter(|entry| entry.co_parent_id.is_some())
            .count();

        format!(
            "genomes={} bodies={} hybrids={} max_gen={}",
            self.entries.len(),
            body_count,
            hybrid_count,
            max_generation
        )
    }
}

fn breed_snapshots(
    id_a: &str,
    parent_a: &GenomeSnapshot,
    id_b: &str,
    parent_b: &GenomeSnapshot,
    count: u64,
    random_pair: bool,
) -> GenomeSnapshot {
    let seed = unix_now()
        ^ parent_a.seed.rotate_left(13)
        ^ parent_b.seed.rotate_right(7)
        ^ count
        ^ if random_pair {
            0xBADC_0FFEE
        } else {
            0xE117_EC7E
        };

    let mut rng = StdRng::seed_from_u64(seed);
    let strength = rng.gen_range(0.018..0.095);

    let channels = parent_a.channels.max(parent_b.channels).clamp(3, 6);
    let base_rules = ((parent_a.base_rules + parent_b.base_rules) / 2).clamp(4, 12);
    let kernel_radius = ((parent_a.kernel_radius + parent_b.kernel_radius) / 2).clamp(4, 7);

    let mut rules = Vec::new();
    let max_rules = parent_a.rules.len().max(parent_b.rules.len()).clamp(4, 24);

    for i in 0..max_rules {
        let source = match (parent_a.rules.get(i), parent_b.rules.get(i)) {
            (Some(a), Some(b)) => {
                if rng.gen_bool(0.45) {
                    blend_rule(a, b, channels, &mut rng, strength)
                } else if rng.gen_bool(0.5) {
                    clone_rule_for_channels(a, channels)
                } else {
                    clone_rule_for_channels(b, channels)
                }
            }
            (Some(a), None) => clone_rule_for_channels(a, channels),
            (None, Some(b)) => clone_rule_for_channels(b, channels),
            (None, None) => continue,
        };

        rules.push(source);
    }

    if rules.is_empty() {
        rules.extend(
            parent_a
                .rules
                .iter()
                .map(|rule| clone_rule_for_channels(rule, channels)),
        );
    }

    let branch_label = hybrid_branch_label(parent_a, parent_b);
    let generation = parent_a
        .generation
        .max(parent_b.generation)
        .saturating_add(1);

    let mut child = GenomeSnapshot {
        version: 5,
        genome_id: genome_id(
            seed,
            0,
            count,
            if random_pair {
                "random_hybrid"
            } else {
                "elite_hybrid"
            },
        ),
        parent_id: Some(id_a.to_string()),
        co_parent_id: Some(id_b.to_string()),
        generation,
        branch_label,
        mutation_strength: strength,
        seed,
        saved_at_unix: unix_now(),
        reason: if random_pair {
            "random_hybrid".to_string()
        } else {
            "elite_hybrid".to_string()
        },
        tick: 0,
        field_w: parent_a.field_w.max(parent_b.field_w),
        field_h: parent_a.field_h.max(parent_b.field_h),
        channels,
        base_rules,
        kernel_radius,
        motion_score: 0.0,
        entropy_score: 0.0,
        mass: 0.0,
        rules,
        cells: Vec::new(),
    };

    mutate_rules(&mut child, &mut rng, strength * 0.75);
    child
}

fn blend_rule(
    a: &RuleGenome,
    b: &RuleGenome,
    channels: usize,
    rng: &mut StdRng,
    strength: f32,
) -> RuleGenome {
    let mut taps = if rng.gen_bool(0.5) {
        a.taps.clone()
    } else {
        b.taps.clone()
    };

    if !a.taps.is_empty() && !b.taps.is_empty() && rng.gen_bool(0.35) {
        taps.clear();
        let max_len = a.taps.len().max(b.taps.len()).min(96);
        for i in 0..max_len {
            match (a.taps.get(i), b.taps.get(i)) {
                (Some(ta), Some(tb)) => {
                    taps.push(KernelTapGenome {
                        dx: if rng.gen_bool(0.5) { ta.dx } else { tb.dx },
                        dy: if rng.gen_bool(0.5) { ta.dy } else { tb.dy },
                        weight: blend_f32(ta.weight, tb.weight, rng.gen_range(0.35..0.65)),
                    });
                }
                (Some(ta), None) => taps.push(ta.clone()),
                (None, Some(tb)) => taps.push(tb.clone()),
                (None, None) => {}
            }
        }
    }

    for tap in &mut taps {
        if rng.gen_bool(0.20) {
            tap.weight = (tap.weight + rng.gen_range(-strength..strength) * 0.12).max(0.0);
        }
    }
    normalize_taps(&mut taps);

    RuleGenome {
        from: if rng.gen_bool(0.5) { a.from } else { b.from }.min(channels - 1),
        to: if rng.gen_bool(0.5) { a.to } else { b.to }.min(channels - 1),
        mu: blend_f32(a.mu, b.mu, rng.gen_range(0.35..0.65)).clamp(0.08, 0.55),
        sigma: blend_f32(a.sigma, b.sigma, rng.gen_range(0.35..0.65)).clamp(0.012, 0.140),
        weight: blend_f32(a.weight, b.weight, rng.gen_range(0.35..0.65)).clamp(-0.85, 0.85),
        taps,
    }
}

fn clone_rule_for_channels(rule: &RuleGenome, channels: usize) -> RuleGenome {
    let mut cloned = rule.clone();
    cloned.from = cloned.from.min(channels - 1);
    cloned.to = cloned.to.min(channels - 1);
    cloned
}

fn mutate_rules(snapshot: &mut GenomeSnapshot, rng: &mut StdRng, strength: f32) {
    for rule in &mut snapshot.rules {
        rule.mu = (rule.mu + rng.gen_range(-strength..strength)).clamp(0.08, 0.55);
        rule.sigma = (rule.sigma + rng.gen_range(-strength..strength) * 0.35).clamp(0.012, 0.140);
        rule.weight = (rule.weight + rng.gen_range(-strength..strength) * 2.0).clamp(-0.85, 0.85);

        for tap in &mut rule.taps {
            if rng.gen_bool(0.18) {
                tap.weight = (tap.weight + rng.gen_range(-strength..strength) * 0.15).max(0.0);
            }
        }

        normalize_taps(&mut rule.taps);
    }

    if rng.gen_bool((strength * 1.6).clamp(0.02, 0.18) as f64) && snapshot.rules.len() > 2 {
        let remove_at = rng.gen_range(0..snapshot.rules.len());
        snapshot.rules.remove(remove_at);
    }

    if rng.gen_bool((strength * 1.9).clamp(0.02, 0.22) as f64) && !snapshot.rules.is_empty() {
        let template = snapshot.rules[rng.gen_range(0..snapshot.rules.len())].clone();

        let mut new_rule = template;
        new_rule.from = rng.gen_range(0..snapshot.channels);
        new_rule.to = rng.gen_range(0..snapshot.channels);
        new_rule.mu = (new_rule.mu + rng.gen_range(-strength..strength)).clamp(0.08, 0.55);
        new_rule.sigma =
            (new_rule.sigma + rng.gen_range(-strength..strength) * 0.35).clamp(0.012, 0.140);
        new_rule.weight =
            (new_rule.weight + rng.gen_range(-strength..strength) * 2.0).clamp(-0.85, 0.85);

        snapshot.rules.push(new_rule);
    }
}

fn normalize_taps(taps: &mut [KernelTapGenome]) {
    let total = taps.iter().map(|tap| tap.weight).sum::<f32>().max(0.0001);
    for tap in taps {
        tap.weight /= total;
    }
}

fn score_entry(entry: &GenomeIndexEntry) -> f32 {
    let performance = entry.motion_score * 0.38 + entry.entropy_score * 0.34 + entry.mass * 0.18;
    let lineage = entry.generation as f32 * 0.003;
    let hybrid_bonus = if entry.co_parent_id.is_some() {
        0.018
    } else {
        0.0
    };
    let body_bonus = if entry.has_body_snapshot { 0.006 } else { 0.0 };
    let overuse_penalty = entry.times_loaded as f32 * 0.004 + entry.children_count as f32 * 0.002;

    performance + lineage + hybrid_bonus + body_bonus - overuse_penalty
}
fn blend_f32(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn branch_label(seed: u64) -> String {
    let names = [
        "Abyss", "Tide", "Bloom", "Orbium", "Manta", "Reef", "Halo", "Vesper", "Nautilus",
        "Drifter", "Pulse", "Medusa", "Prism", "Ghost", "Lantern", "Nova",
    ];

    names[(seed as usize) % names.len()].to_string()
}

fn hybrid_branch_label(a: &GenomeSnapshot, b: &GenomeSnapshot) -> String {
    let left = if a.branch_label.is_empty() {
        branch_label(a.seed)
    } else {
        a.branch_label.clone()
    };

    let right = if b.branch_label.is_empty() {
        branch_label(b.seed)
    } else {
        b.branch_label.clone()
    };

    if left == right {
        format!("{} Hybrid", left)
    } else {
        format!("{}-{} Hybrid", left, right)
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
