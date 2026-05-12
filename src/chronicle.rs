use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

const CHRONICLE_PATH: &str = "saves/chronicle.json";
const MAX_RECENT: usize = 128;
const MAX_BEST: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub seed: u64,
    pub saved_at_unix: u64,
    pub reason: String,
    pub tick: u64,
    pub field_w: usize,
    pub field_h: usize,
    pub channels: usize,
    pub rules: usize,
    pub base_rules: usize,
    pub kernel_radius: i32,
    pub mass: f32,
    pub mass_variance_hint: f32,
    pub channel_masses: Vec<f32>,
    pub avg_rule_mu: f32,
    pub avg_rule_sigma: f32,
    pub avg_rule_weight: f32,
    pub positive_rules: usize,
    pub negative_rules: usize,
    pub avg_kernel_taps: f32,
    #[serde(default)]
    pub center_x: f32,
    #[serde(default)]
    pub center_y: f32,
    #[serde(default)]
    pub motion_score: f32,
    #[serde(default)]
    pub entropy_score: f32,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct ChronicleBias {
    pub target_channels: usize,
    pub target_base_rules: usize,
    pub target_radius: i32,
    pub target_mu: f32,
    pub target_sigma: f32,
    pub target_weight: f32,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chronicle {
    pub version: u32,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
    pub total_runs_recorded: u64,
    pub recent: Vec<RunRecord>,
    pub best: Vec<RunRecord>,
    pub avg_channels: f32,
    pub avg_rules: f32,
    pub avg_kernel_radius: f32,
    pub avg_mass: f32,
    #[serde(default)]
    pub avg_motion_score: f32,
    #[serde(default)]
    pub avg_entropy_score: f32,
    pub best_score_seen: f32,
    pub last_seed: Option<u64>,
}

impl Chronicle {
    pub fn load_or_new() -> Self {
        if let Ok(mut file) = File::open(CHRONICLE_PATH) {
            let mut data = String::new();
            if file.read_to_string(&mut data).is_ok() {
                if let Ok(mut memory) = serde_json::from_str::<Chronicle>(&data) {
                    memory.version = 3;
                    return memory;
                }
            }
        }

        let now = unix_now();
        Self {
            version: 3,
            created_at_unix: now,
            updated_at_unix: now,
            total_runs_recorded: 0,
            recent: Vec::new(),
            best: Vec::new(),
            avg_channels: 0.0,
            avg_rules: 0.0,
            avg_kernel_radius: 0.0,
            avg_mass: 0.0,
            avg_motion_score: 0.0,
            avg_entropy_score: 0.0,
            best_score_seen: 0.0,
            last_seed: None,
        }
    }

    pub fn suggest_bias(&self) -> Option<ChronicleBias> {
        if self.best.len() < 3 {
            return None;
        }

        let take = self.best.len().min(8);
        let mut score_total = 0.0;
        let mut channels = 0.0;
        let mut base_rules = 0.0;
        let mut radius = 0.0;
        let mut mu = 0.0;
        let mut sigma = 0.0;
        let mut weight = 0.0;

        for record in self.best.iter().take(take) {
            let score = record.score.max(0.05);
            score_total += score;
            channels += record.channels as f32 * score;
            base_rules += record.base_rules as f32 * score;
            radius += record.kernel_radius as f32 * score;
            mu += record.avg_rule_mu * score;
            sigma += record.avg_rule_sigma * score;
            weight += record.avg_rule_weight * score;
        }

        if score_total <= 0.0 {
            return None;
        }

        let strength = (self.total_runs_recorded as f32 / 32.0).clamp(0.08, 0.38);

        Some(ChronicleBias {
            target_channels: (channels / score_total).round().clamp(3.0, 6.0) as usize,
            target_base_rules: (base_rules / score_total).round().clamp(4.0, 12.0) as usize,
            target_radius: (radius / score_total).round().clamp(4.0, 7.0) as i32,
            target_mu: (mu / score_total).clamp(0.12, 0.46),
            target_sigma: (sigma / score_total).clamp(0.022, 0.095),
            target_weight: (weight / score_total).clamp(-0.46, 0.60),
            strength,
        })
    }

    pub fn record(&mut self, mut record: RunRecord) {
        record.saved_at_unix = unix_now();
        self.updated_at_unix = record.saved_at_unix;
        self.total_runs_recorded += 1;
        self.last_seed = Some(record.seed);
        self.best_score_seen = self.best_score_seen.max(record.score);

        let n = self.total_runs_recorded as f32;
        self.avg_channels = rolling_avg(self.avg_channels, record.channels as f32, n);
        self.avg_rules = rolling_avg(self.avg_rules, record.rules as f32, n);
        self.avg_kernel_radius =
            rolling_avg(self.avg_kernel_radius, record.kernel_radius as f32, n);
        self.avg_mass = rolling_avg(self.avg_mass, record.mass, n);
        self.avg_motion_score = rolling_avg(self.avg_motion_score, record.motion_score, n);
        self.avg_entropy_score = rolling_avg(self.avg_entropy_score, record.entropy_score, n);

        self.recent.push(record.clone());
        if self.recent.len() > MAX_RECENT {
            self.recent.remove(0);
        }

        self.best.push(record);
        self.best.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.best.truncate(MAX_BEST);
    }

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = Path::new(CHRONICLE_PATH).parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(CHRONICLE_PATH)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn status(&self) -> String {
        let mode = if self.suggest_bias().is_some() {
            "memory-bias active"
        } else {
            "pure-random learning"
        };

        format!(
            "chronicle runs={} best={} best={:.3} motion={:.3} entropy={:.3} {}",
            self.total_runs_recorded,
            self.best.len(),
            self.best_score_seen,
            self.avg_motion_score,
            self.avg_entropy_score,
            mode
        )
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn rolling_avg(old: f32, new: f32, n: f32) -> f32 {
    if n <= 1.0 {
        new
    } else {
        old + (new - old) / n
    }
}
