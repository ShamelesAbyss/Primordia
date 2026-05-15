mod bestiary;
mod chronicle;
mod genome;
mod gpu;

use anyhow::Result;
use bestiary::Bestiary;
use chronicle::{unix_now, Chronicle, ChronicleBias, RunRecord};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use genome::{GenomeSnapshot, GenomeVault, KernelTapGenome, RuleGenome};
use gpu::GpuStatus;
use rand::{rngs::StdRng, Rng, SeedableRng};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use rayon::prelude::*;
use std::{
    io,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MIN_CHANNELS: usize = 3;
const MAX_CHANNELS: usize = 10;
const MIN_RULES: usize = 4;
const MAX_RULES: usize = 24;
const MIN_RADIUS: i32 = 4;
const MAX_RADIUS: i32 = 8;
const DT: f32 = 0.048;

#[derive(Clone)]
struct KernelTap {
    dx: i32,
    dy: i32,
    weight: f32,
}

#[derive(Clone)]
struct Rule {
    from: usize,
    to: usize,
    mu: f32,
    sigma: f32,
    weight: f32,
    taps: Vec<KernelTap>,
}

struct World {
    seed: u64,
    tick: u64,
    w: usize,
    h: usize,
    channels: usize,
    base_rules: usize,
    radius: i32,
    cells: Vec<f32>,
    next: Vec<f32>,
    rules: Vec<Rule>,
    rng: StdRng,
    last_center_x: f32,
    last_center_y: f32,
    motion_score: f32,
    entropy_score: f32,
}

#[derive(Clone, Copy)]
enum KernelArchetype {
    Orbium,
    Manta,
    Medusa,
    Reef,
    Spiral,
    Predator,
}

impl KernelArchetype {
    fn random(rng: &mut StdRng) -> Self {
        match rng.gen_range(0..6) {
            0 => Self::Orbium,
            1 => Self::Manta,
            2 => Self::Medusa,
            3 => Self::Reef,
            4 => Self::Spiral,
            _ => Self::Predator,
        }
    }

    fn shape(&self, rng: &mut StdRng) -> (usize, f32, f32, f32, f32, f32) {
        match self {
            Self::Orbium => (
                rng.gen_range(2..5),
                rng.gen_range(0.02..0.12),
                rng.gen_range(0.03..0.08),
                rng.gen_range(0.00..0.16),
                rng.gen_range(0.00..0.08),
                1.0,
            ),
            Self::Manta => (
                rng.gen_range(3..6),
                rng.gen_range(0.08..0.28),
                rng.gen_range(0.035..0.095),
                rng.gen_range(0.18..0.46),
                rng.gen_range(0.00..0.12),
                2.0,
            ),
            Self::Medusa => (
                rng.gen_range(2..5),
                rng.gen_range(0.00..0.18),
                rng.gen_range(0.065..0.140),
                rng.gen_range(0.04..0.24),
                rng.gen_range(0.00..0.10),
                1.0,
            ),
            Self::Reef => (
                rng.gen_range(3..6),
                rng.gen_range(0.00..0.10),
                rng.gen_range(0.030..0.075),
                rng.gen_range(0.00..0.12),
                rng.gen_range(0.02..0.18),
                3.0,
            ),
            Self::Spiral => (
                rng.gen_range(3..6),
                rng.gen_range(0.06..0.24),
                rng.gen_range(0.035..0.090),
                rng.gen_range(0.22..0.55),
                rng.gen_range(0.02..0.16),
                rng.gen_range(3.0..6.0),
            ),
            Self::Predator => (
                rng.gen_range(2..5),
                rng.gen_range(0.10..0.34),
                rng.gen_range(0.022..0.060),
                rng.gen_range(0.16..0.42),
                rng.gen_range(0.04..0.22),
                rng.gen_range(2.0..5.0),
            ),
        }
    }

    fn ring_weight(&self, ring: usize, rng: &mut StdRng) -> f32 {
        match self {
            Self::Orbium => rng.gen_range(0.18..1.00),
            Self::Manta => {
                let sign = if ring % 3 == 1 { -1.0 } else { 1.0 };
                sign * rng.gen_range(0.20..1.10)
            }
            Self::Medusa => {
                let falloff = 1.0 / (ring as f32 + 1.0).sqrt();
                rng.gen_range(0.25..1.15) * falloff
            }
            Self::Reef => {
                let sign = if ring % 2 == 0 { 1.0 } else { -0.45 };
                sign * rng.gen_range(0.25..0.95)
            }
            Self::Spiral => {
                let sign = if ring % 2 == 0 { 1.0 } else { -1.0 };
                sign * rng.gen_range(0.25..1.20)
            }
            Self::Predator => {
                let sign = if ring == 0 { -1.0 } else { 1.0 };
                sign * rng.gen_range(0.35..1.35)
            }
        }
    }
}

impl Rule {
    fn random(
        rng: &mut StdRng,
        channels: usize,
        radius: i32,
        bias: Option<&ChronicleBias>,
    ) -> Self {
        let from = rng.gen_range(0..channels);
        let to = rng.gen_range(0..channels);
        let archetype = KernelArchetype::random(rng);
        let (ring_count, center_jitter, base_width, asymmetry, noise_mix, angular_lobes) =
            archetype.shape(rng);

        let phase = rng.gen_range(0.0..std::f32::consts::TAU);
        let swirl = rng.gen_range(-1.0..1.0);

        let mut ring_weights = Vec::with_capacity(ring_count);
        for ring in 0..ring_count {
            ring_weights.push(archetype.ring_weight(ring, rng));
        }

        let mut taps = Vec::new();
        let mut total_abs = 0.0;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let dist_cells = ((dx * dx + dy * dy) as f32).sqrt();
                let dist = dist_cells / radius as f32;

                if dist > 1.0 {
                    continue;
                }

                let angle = (dy as f32).atan2(dx as f32);
                let mut weight = 0.0;

                for ring in 0..ring_count {
                    let center = ((ring as f32 + 0.55) / ring_count as f32)
                        + rng.gen_range(-center_jitter..center_jitter);
                    let width = (base_width * rng.gen_range(0.72..1.34)).max(0.012);
                    let shell = (-((dist - center).powi(2)) / (2.0 * width * width)).exp();

                    let angular_wave = 1.0
                        + asymmetry * ((angle * angular_lobes + phase + ring as f32 * swirl).cos());

                    let directional_bias = match archetype {
                        KernelArchetype::Manta => 1.0 + asymmetry * 0.65 * angle.cos(),
                        KernelArchetype::Spiral => {
                            1.0 + asymmetry * 0.45 * (angle + dist * 6.0).sin()
                        }
                        KernelArchetype::Predator => 1.0 + asymmetry * 0.55 * (angle * 2.0).cos(),
                        _ => 1.0,
                    };

                    weight += shell * ring_weights[ring] * angular_wave * directional_bias;
                }

                if rng.gen_bool(noise_mix as f64) {
                    weight += rng.gen_range(-0.20..0.20);
                }

                if weight.abs() > 0.0001 {
                    taps.push(KernelTap { dx, dy, weight });
                    total_abs += weight.abs();
                }
            }
        }

        for tap in taps.iter_mut() {
            tap.weight /= total_abs.max(0.0001);
        }

        let mut mu = rng.gen_range(0.12..0.46);
        let mut sigma = rng.gen_range(0.022..0.095);
        let mut weight = rng.gen_range(-0.46..0.60);

        match archetype {
            KernelArchetype::Orbium => {
                sigma *= rng.gen_range(0.85..1.10);
            }
            KernelArchetype::Manta => {
                mu *= rng.gen_range(0.82..1.05);
                weight += rng.gen_range(0.02..0.12);
            }
            KernelArchetype::Medusa => {
                sigma *= rng.gen_range(1.05..1.35);
                weight *= rng.gen_range(0.78..1.05);
            }
            KernelArchetype::Reef => {
                weight *= rng.gen_range(0.65..0.95);
            }
            KernelArchetype::Spiral => {
                mu *= rng.gen_range(0.90..1.18);
                sigma *= rng.gen_range(0.85..1.18);
            }
            KernelArchetype::Predator => {
                weight += rng.gen_range(-0.16..0.08);
                sigma *= rng.gen_range(0.70..0.98);
            }
        }

        if let Some(memory) = bias {
            if rng.gen_bool(memory.strength as f64) {
                mu = blend(mu, memory.target_mu, memory.strength).clamp(0.12, 0.46);
                sigma = blend(sigma, memory.target_sigma, memory.strength).clamp(0.022, 0.095);
                weight = blend(weight, memory.target_weight, memory.strength).clamp(-0.46, 0.60);
            }
        }

        Self {
            from,
            to,
            mu: mu.clamp(0.08, 0.55),
            sigma: sigma.clamp(0.012, 0.140),
            weight: weight.clamp(-0.85, 0.85),
            taps,
        }
    }
}
impl World {
    fn new(w: usize, h: usize, bias: Option<ChronicleBias>) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self::from_seed(seed, w, h, bias)
    }

    fn from_seed(seed: u64, w: usize, h: usize, bias: Option<ChronicleBias>) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut channels = rng.gen_range(MIN_CHANNELS..=MAX_CHANNELS);
        let mut base_rules = rng.gen_range(MIN_RULES..=MAX_RULES);
        let mut radius = rng.gen_range(MIN_RADIUS..=MAX_RADIUS);

        if let Some(memory) = &bias {
            if rng.gen_bool(memory.strength as f64) {
                channels = memory.target_channels.clamp(MIN_CHANNELS, MAX_CHANNELS);
            }
            if rng.gen_bool(memory.strength as f64) {
                base_rules = memory.target_base_rules.clamp(MIN_RULES, MAX_RULES);
            }
            if rng.gen_bool(memory.strength as f64) {
                radius = memory.target_radius.clamp(MIN_RADIUS, MAX_RADIUS);
            }
        }

        let w = w.max(24);
        let h = h.max(12);

        let mut rules = Vec::new();

        for _ in 0..base_rules {
            rules.push(Rule::random(&mut rng, channels, radius, bias.as_ref()));
        }

        for c in 0..channels {
            let mut self_rule = Rule::random(&mut rng, channels, radius, bias.as_ref());
            self_rule.from = c;
            self_rule.to = c;
            rules.push(self_rule);
        }

        let mut world = Self {
            seed,
            tick: 0,
            w,
            h,
            channels,
            base_rules,
            radius,
            cells: vec![0.0; w * h * channels],
            next: vec![0.0; w * h * channels],
            rules,
            rng,
            last_center_x: 0.0,
            last_center_y: 0.0,
            motion_score: 0.0,
            entropy_score: 0.0,
        };

        world.seed_life();
        world.refresh_motion_baseline();
        world
    }

    fn from_genome_snapshot(snapshot: GenomeSnapshot, w: usize, h: usize) -> Self {
        let channels = snapshot.channels.clamp(MIN_CHANNELS, MAX_CHANNELS);

        let saved_w = snapshot.field_w.max(24);
        let saved_h = snapshot.field_h.max(12);
        let w = if snapshot.cells.is_empty() {
            w.max(24)
        } else {
            saved_w
        };
        let h = if snapshot.cells.is_empty() {
            h.max(12)
        } else {
            saved_h
        };

        let rng = StdRng::seed_from_u64(snapshot.seed ^ 0x6D65_6D6F_7279);

        let rules = snapshot
            .rules
            .into_iter()
            .map(|rule| Rule {
                from: rule.from.min(channels - 1),
                to: rule.to.min(channels - 1),
                mu: rule.mu,
                sigma: rule.sigma,
                weight: rule.weight,
                taps: rule
                    .taps
                    .into_iter()
                    .map(|tap| KernelTap {
                        dx: tap.dx,
                        dy: tap.dy,
                        weight: tap.weight,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();

        let expected_len = w * h * channels;
        let mut world = Self {
            seed: snapshot.seed,
            tick: snapshot.tick,
            w,
            h,
            channels,
            base_rules: snapshot.base_rules,
            radius: snapshot.kernel_radius,
            cells: if snapshot.cells.len() == expected_len {
                snapshot.cells
            } else {
                vec![0.0; expected_len]
            },
            next: vec![0.0; expected_len],
            rules,
            rng,
            last_center_x: 0.0,
            last_center_y: 0.0,
            motion_score: snapshot.motion_score,
            entropy_score: snapshot.entropy_score,
        };

        if world.cells.iter().all(|v| *v <= 0.0) {
            world.seed_life();
        }

        world.refresh_motion_baseline();
        world
    }

    fn resize(&mut self, new_w: usize, new_h: usize) {
        let new_w = new_w.max(24).max(self.w);
        let new_h = new_h.max(12).max(self.h);

        if new_w == self.w && new_h == self.h {
            return;
        }

        let mut new_cells = vec![0.0; new_w * new_h * self.channels];
        let copy_w = self.w.min(new_w);
        let copy_h = self.h.min(new_h);

        for y in 0..copy_h {
            for x in 0..copy_w {
                for c in 0..self.channels {
                    let old_idx = self.idx(x, y, c);
                    let new_idx = (y * new_w + x) * self.channels + c;
                    new_cells[new_idx] = self.cells[old_idx];
                }
            }
        }

        self.w = new_w;
        self.h = new_h;
        self.cells = new_cells;
        self.next = vec![0.0; self.w * self.h * self.channels];
        self.seed_life();
        self.refresh_motion_baseline();
    }

    fn idx(&self, x: usize, y: usize, c: usize) -> usize {
        (y * self.w + x) * self.channels + c
    }

    fn wrap_x(&self, x: i32) -> usize {
        x.rem_euclid(self.w as i32) as usize
    }

    fn wrap_y(&self, y: i32) -> usize {
        y.rem_euclid(self.h as i32) as usize
    }

    fn step(&mut self) {
        let w = self.w;
        let h = self.h;
        let channels = self.channels;
        let tick = self.tick;
        let seed = self.seed;
        let rules = &self.rules;
        let cells = &self.cells;

        self.next
            .par_chunks_mut(channels)
            .enumerate()
            .for_each(|(cell_i, out)| {
                let x = cell_i % w;
                let y = cell_i / w;
                let mut delta = vec![0.0_f32; channels];

                for rule in rules {
                    let mut conv = 0.0;

                    for tap in &rule.taps {
                        let nx = wrap_dim(x as i32 + tap.dx, w);
                        let ny = wrap_dim(y as i32 + tap.dy, h);
                        let idx = (ny * w + nx) * channels + rule.from;
                        conv += cells[idx] * tap.weight;
                    }

                    let growth = bell(conv, rule.mu, rule.sigma) * 2.0 - 1.0;
                    delta[rule.to] += growth * rule.weight;
                }

                for c in 0..channels {
                    let idx = (y * w + x) * channels + c;
                    let old = cells[idx];

                    let up = wrap_dim(y as i32 - 1, h);
                    let down = wrap_dim(y as i32 + 1, h);
                    let left = wrap_dim(x as i32 - 1, w);
                    let right = wrap_dim(x as i32 + 1, w);

                    let mut lap = 0.0;
                    lap += cells[(up * w + x) * channels + c];
                    lap += cells[(down * w + x) * channels + c];
                    lap += cells[(y * w + left) * channels + c];
                    lap += cells[(y * w + right) * channels + c];
                    lap -= old * 4.0;

                    let pressure = old * old * 0.060;
                    let noise = deterministic_noise(seed, tick, x, y, c) * 0.0007;

                    out[c] = (old + DT * delta[c] + lap * 0.007 + noise - pressure).clamp(0.0, 1.0);
                }
            });

        std::mem::swap(&mut self.cells, &mut self.next);
        self.update_motion_memory();
        self.tick += 1;

        if self.tick % 2400 == 0 && !self.field_is_zero() {
            self.seed_life();
        }
    }

    fn seed_life(&mut self) {
        let clusters = self.rng.gen_range(4..10);

        for _ in 0..clusters {
            let cx = self.rng.gen_range(4..self.w - 4) as i32;
            let cy = self.rng.gen_range(3..self.h - 3) as i32;
            let radius = self.rng.gen_range(3..9) as i32;

            for y in -radius..=radius {
                for x in -radius..=radius {
                    let d = ((x * x + y * y) as f32).sqrt();
                    if d <= radius as f32 {
                        let softness = 1.0 - d / radius as f32;
                        let px = self.wrap_x(cx + x);
                        let py = self.wrap_y(cy + y);

                        for c in 0..self.channels {
                            if self.rng.gen_bool(0.72) {
                                let idx = self.idx(px, py, c);
                                let v = self.rng.gen_range(0.06..0.85) * softness;
                                self.cells[idx] = (self.cells[idx] + v).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
            }
        }
    }

    fn mass(&self) -> f32 {
        self.cells.iter().sum::<f32>() / self.cells.len() as f32
    }

    fn channel_mass(&self, c: usize) -> f32 {
        let mut total = 0.0;
        for y in 0..self.h {
            for x in 0..self.w {
                total += self.cells[self.idx(x, y, c)];
            }
        }
        total / (self.w * self.h) as f32
    }

    fn center_of_mass(&self) -> (f32, f32, f32) {
        let mut total = 0.0;
        let mut sx = 0.0;
        let mut sy = 0.0;

        for y in 0..self.h {
            for x in 0..self.w {
                let mut cell_mass = 0.0;
                for c in 0..self.channels {
                    cell_mass += self.cells[self.idx(x, y, c)];
                }

                total += cell_mass;
                sx += x as f32 * cell_mass;
                sy += y as f32 * cell_mass;
            }
        }

        if total <= 0.0001 {
            (self.w as f32 * 0.5, self.h as f32 * 0.5, 0.0)
        } else {
            (
                sx / total,
                sy / total,
                total / (self.w * self.h * self.channels) as f32,
            )
        }
    }

    fn refresh_motion_baseline(&mut self) {
        let (cx, cy, _) = self.center_of_mass();
        self.last_center_x = cx;
        self.last_center_y = cy;
        self.motion_score = 0.0;
        self.entropy_score = 0.0;
    }

    fn field_is_zero(&self) -> bool {
        self.cells.iter().all(|v| *v == 0.0)
    }

    fn update_motion_memory(&mut self) {
        if self.field_is_zero() {
            self.motion_score = 0.0;
            self.entropy_score = 0.0;
            let (cx, cy, _) = self.center_of_mass();
            self.last_center_x = cx;
            self.last_center_y = cy;
            return;
        }

        let (cx, cy, _) = self.center_of_mass();

        let dx = (cx - self.last_center_x).abs();
        let dy = (cy - self.last_center_y).abs();
        let raw_motion = ((dx * dx + dy * dy).sqrt() / self.w.max(self.h) as f32) * 20.0;

        self.motion_score = self.motion_score * 0.985 + raw_motion.clamp(0.0, 1.0) * 0.015;
        self.last_center_x = cx;
        self.last_center_y = cy;

        let mut active = 0usize;
        let mut saturated = 0usize;
        let mut dead = 0usize;

        for v in &self.cells {
            if *v > 0.03 {
                active += 1;
            }
            if *v > 0.82 {
                saturated += 1;
            }
            if *v < 0.01 {
                dead += 1;
            }
        }

        let len = self.cells.len().max(1) as f32;
        let active_ratio = active as f32 / len;
        let saturated_ratio = saturated as f32 / len;
        let dead_ratio = dead as f32 / len;

        let lively = (active_ratio * 2.2).clamp(0.0, 1.0);
        let not_saturated = (1.0 - saturated_ratio * 4.0).clamp(0.0, 1.0);
        let not_dead = (1.0 - dead_ratio * 0.85).clamp(0.0, 1.0);

        let raw_entropy = (lively * 0.45 + not_saturated * 0.35 + not_dead * 0.20).clamp(0.0, 1.0);
        self.entropy_score = self.entropy_score * 0.985 + raw_entropy * 0.015;
    }

    fn is_extinct(&self) -> bool {
        if !self.field_is_zero() {
            return false;
        }

        if self.mass() != 0.0 {
            return false;
        }

        if self.motion_score != 0.0 {
            return false;
        }

        for c in 0..self.channels {
            if self.channel_mass(c) != 0.0 {
                return false;
            }
        }

        true
    }

    fn channel_name(c: usize) -> &'static str {
        match c % 12 {
            0 => "cyan",
            1 => "green",
            2 => "magenta",
            3 => "red",
            4 => "blue",
            5 => "gold",
            6 => "violet",
            7 => "orange",
            8 => "teal",
            9 => "lime",
            10 => "pink",
            _ => "white",
        }
    }

    fn channel_color(c: usize) -> Color {
        match c % 12 {
            0 => Color::Cyan,
            1 => Color::Green,
            2 => Color::Magenta,
            3 => Color::Red,
            4 => Color::Blue,
            5 => Color::Yellow,
            6 => Color::Rgb(170, 90, 255),
            7 => Color::Rgb(255, 140, 40),
            8 => Color::Rgb(0, 220, 180),
            9 => Color::LightGreen,
            10 => Color::LightMagenta,
            _ => Color::White,
        }
    }

    fn cell_visual(&self, x: usize, y: usize) -> (&'static str, Color) {
        let mut dominant = 0usize;
        let mut strongest = 0.0f32;
        let mut total = 0.0f32;
        let mut second = 0.0f32;

        for c in 0..self.channels {
            let v = self.cells[self.idx(x, y, c)];
            total += v;

            if v > strongest {
                second = strongest;
                strongest = v;
                dominant = c;
            } else if v > second {
                second = v;
            }
        }

        let glyph = match strongest {
            v if v < 0.008 => " ",
            v if v < 0.020 => "·",
            v if v < 0.035 => "•",
            v if v < 0.050 => "◦",
            v if v < 0.070 => "○",
            v if v < 0.095 => "◌",
            v if v < 0.120 => "◍",
            v if v < 0.150 => "◎",
            v if v < 0.180 => "●",
            v if v < 0.210 => "◇",
            v if v < 0.240 => "◈",
            v if v < 0.275 => "◆",
            v if v < 0.310 => "△",
            v if v < 0.345 => "▲",
            v if v < 0.380 => "□",
            v if v < 0.420 => "▣",
            v if v < 0.460 => "■",
            v if v < 0.500 => "✦",
            v if v < 0.540 => "✧",
            v if v < 0.580 => "✶",
            v if v < 0.620 => "✸",
            v if v < 0.660 => "✹",
            v if v < 0.700 => "❀",
            v if v < 0.740 => "✿",
            v if v < 0.780 => "❂",
            v if v < 0.820 => "⬡",
            v if v < 0.860 => "⬢",
            v if v < 0.900 => "◐",
            v if v < 0.935 => "◑",
            v if v < 0.970 => "◒",
            _ => "◓",
        };

        let blend = second > strongest * 0.72 && total > 0.25;
        let color = if total > self.channels as f32 * 0.70 {
            Color::White
        } else if blend && dominant == 0 {
            Color::LightCyan
        } else if blend && dominant == 1 {
            Color::LightGreen
        } else if blend && dominant == 2 {
            Color::LightMagenta
        } else if blend && dominant == 3 {
            Color::LightRed
        } else if blend && dominant == 4 {
            Color::LightBlue
        } else {
            Self::channel_color(dominant)
        };

        (glyph, color)
    }

    #[allow(dead_code)]
    fn inject_genome_snapshot(&mut self, snapshot: GenomeSnapshot) {
        let target_channels = self
            .channels
            .max(snapshot.channels)
            .clamp(MIN_CHANNELS, MAX_CHANNELS);

        if target_channels != self.channels {
            let mut new_cells = vec![0.0; self.w * self.h * target_channels];

            for y in 0..self.h {
                for x in 0..self.w {
                    for c in 0..self.channels {
                        let old_idx = self.idx(x, y, c);
                        let new_idx = (y * self.w + x) * target_channels + c;
                        new_cells[new_idx] = self.cells[old_idx];
                    }
                }
            }

            self.channels = target_channels;
            self.cells = new_cells;
            self.next = vec![0.0; self.w * self.h * self.channels];
        }

        let available_slots = 72usize.saturating_sub(self.rules.len());
        let take_rules = available_slots.min(snapshot.rules.len()).min(18);

        for rule in snapshot.rules.into_iter().take(take_rules) {
            self.rules.push(Rule {
                from: rule.from.min(self.channels - 1),
                to: rule.to.min(self.channels - 1),
                mu: rule.mu,
                sigma: rule.sigma,
                weight: rule.weight,
                taps: rule
                    .taps
                    .into_iter()
                    .map(|tap| KernelTap {
                        dx: tap.dx,
                        dy: tap.dy,
                        weight: tap.weight,
                    })
                    .collect(),
            });
        }

        self.base_rules = self
            .base_rules
            .max(self.rules.len().saturating_sub(self.channels));
        self.radius = self
            .radius
            .max(snapshot.kernel_radius)
            .clamp(MIN_RADIUS, MAX_RADIUS);

        for _ in 0..8 {
            self.seed_life();
        }

        self.refresh_motion_baseline();
    }

    fn genome_snapshot(&self, reason: &str) -> GenomeSnapshot {
        let mut rules = Vec::with_capacity(self.rules.len());

        for rule in &self.rules {
            let taps = rule
                .taps
                .iter()
                .map(|tap| KernelTapGenome {
                    dx: tap.dx,
                    dy: tap.dy,
                    weight: tap.weight,
                })
                .collect();

            rules.push(RuleGenome {
                from: rule.from,
                to: rule.to,
                mu: rule.mu,
                sigma: rule.sigma,
                weight: rule.weight,
                taps,
            });
        }

        GenomeSnapshot {
            version: 4,
            genome_id: String::new(),
            parent_id: None,
            co_parent_id: None,
            generation: 0,
            branch_label: String::new(),
            mutation_strength: 0.0,
            seed: self.seed,
            saved_at_unix: unix_now(),
            reason: reason.to_string(),
            tick: self.tick,
            field_w: self.w,
            field_h: self.h,
            channels: self.channels,
            base_rules: self.base_rules,
            kernel_radius: self.radius,
            motion_score: self.motion_score,
            entropy_score: self.entropy_score,
            mass: self.mass(),
            rules,
            cells: self.cells.clone(),
        }
    }

    fn chronicle_record(&self, reason: &str) -> RunRecord {
        let mut channel_masses = Vec::with_capacity(self.channels);
        for c in 0..self.channels {
            channel_masses.push(self.channel_mass(c));
        }

        let mut mu_total = 0.0;
        let mut sigma_total = 0.0;
        let mut weight_total = 0.0;
        let mut positive_rules = 0;
        let mut negative_rules = 0;
        let mut tap_total = 0usize;

        for rule in &self.rules {
            mu_total += rule.mu;
            sigma_total += rule.sigma;
            weight_total += rule.weight;
            tap_total += rule.taps.len();

            if rule.weight >= 0.0 {
                positive_rules += 1;
            } else {
                negative_rules += 1;
            }
        }

        let rule_count = self.rules.len().max(1) as f32;
        let mass = self.mass();
        let ideal_mass = 0.18;
        let mass_health = (1.0 - (mass - ideal_mass).abs() * 4.0).clamp(0.0, 1.0);
        let channel_balance = if channel_masses.is_empty() {
            0.0
        } else {
            let avg = channel_masses.iter().sum::<f32>() / channel_masses.len() as f32;
            let variance = channel_masses
                .iter()
                .map(|v| (v - avg).powi(2))
                .sum::<f32>()
                / channel_masses.len() as f32;
            (1.0 - variance * 30.0).clamp(0.0, 1.0)
        };

        let activity = if self.tick > 300 {
            1.0
        } else {
            self.tick as f32 / 300.0
        };

        let score = (mass_health * 0.28
            + channel_balance * 0.22
            + activity * 0.14
            + self.motion_score * 0.20
            + self.entropy_score * 0.16)
            .clamp(0.0, 1.0);

        RunRecord {
            seed: self.seed,
            saved_at_unix: unix_now(),
            reason: reason.to_string(),
            tick: self.tick,
            field_w: self.w,
            field_h: self.h,
            channels: self.channels,
            rules: self.rules.len(),
            base_rules: self.base_rules,
            kernel_radius: self.radius,
            mass,
            mass_variance_hint: 1.0 - channel_balance,
            channel_masses,
            avg_rule_mu: mu_total / rule_count,
            avg_rule_sigma: sigma_total / rule_count,
            avg_rule_weight: weight_total / rule_count,
            positive_rules,
            negative_rules,
            avg_kernel_taps: tap_total as f32 / rule_count,
            center_x: self.last_center_x,
            center_y: self.last_center_y,
            motion_score: self.motion_score,
            entropy_score: self.entropy_score,
            score,
        }
    }
}

fn wrap_dim(v: i32, max: usize) -> usize {
    v.rem_euclid(max as i32) as usize
}

fn deterministic_noise(seed: u64, tick: u64, x: usize, y: usize, c: usize) -> f32 {
    let mut n = seed
        ^ tick.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (x as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (y as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
        ^ (c as u64).wrapping_mul(0xD6E8_FD9D_AA35_558D);

    n ^= n >> 30;
    n = n.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    n ^= n >> 27;
    n = n.wrapping_mul(0x94D0_49BB_1331_11EB);
    n ^= n >> 31;

    let unit = (n as f32 / u64::MAX as f32).clamp(0.0, 1.0);
    unit * 2.0 - 1.0
}

fn blend(a: f32, b: f32, amount: f32) -> f32 {
    a * (1.0 - amount) + b * amount
}

fn bell(x: f32, mu: f32, sigma: f32) -> f32 {
    (-((x - mu).powi(2)) / (2.0 * sigma * sigma)).exp()
}

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run(&mut terminal);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut chronicle = Chronicle::load_or_new();
    let mut bestiary = Bestiary::load_or_new();
    let mut genome_vault = GenomeVault::load_or_new();
    let _gpu_status: GpuStatus = gpu::probe_gpu();
    let mut world = World::new(132, 72, chronicle.suggest_bias());

    let sim_step = Duration::from_millis(33);
    let render_step = Duration::from_millis(33);

    let mut last_sim_tick = Instant::now();
    let mut last_render = Instant::now();
    let mut status_note = format!(
        "{}  {}  {}",
        chronicle.status(),
        bestiary.status(),
        genome_vault.status(),
    );
    let mut gpu_live_enabled = false;
    #[cfg(feature = "gpu")]
    let mut real_lenia_gpu: Option<gpu::RealLeniaGpuEngine> = None;
    let mut extinction_ticks: u64 = 0;
    let extinction_threshold: u64 = 1000;
    let mut extinction_rebirths: u64 = 0;

    loop {
        while event::poll(Duration::from_millis(1))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        let record = world.chronicle_record("quit_autosave");
                        chronicle.record(record.clone());

                        if bestiary.consider(&record)?.is_some() {
                            let _ = genome_vault
                                .save_snapshot(&world.genome_snapshot("bestiary_quit_autosave"))?;
                        }

                        chronicle.save()?;
                        bestiary.save()?;
                        genome_vault.save()?;
                        return Ok(());
                    }
                    KeyCode::Char('s') => {
                        let record = world.chronicle_record("manual_save");
                        chronicle.record(record.clone());

                        let discovery = bestiary.consider(&record)?;
                        let genome_id =
                            genome_vault.save_snapshot(&world.genome_snapshot("manual_save"))?;

                        chronicle.save()?;
                        bestiary.save()?;
                        genome_vault.save()?;

                        status_note = if let Some(note) = discovery {
                            format!("{} genome={}", note, genome_id)
                        } else {
                            format!(
                                "saved genome={}  {}  {}  {}",
                                genome_id,
                                chronicle.status(),
                                bestiary.status(),
                                genome_vault.status(),
                            )
                        };
                    }
                    KeyCode::Char('r') => {
                        let record = world.chronicle_record("rebirth");
                        chronicle.record(record.clone());

                        let discovery = bestiary.consider(&record)?;
                        if discovery.is_some() {
                            let _ = genome_vault
                                .save_snapshot(&world.genome_snapshot("bestiary_rebirth"))?;
                        }

                        chronicle.save()?;
                        bestiary.save()?;
                        genome_vault.save()?;

                        world = World::new(world.w, world.h, chronicle.suggest_bias());

                        status_note = if let Some(note) = discovery {
                            format!("{}  {}", note, genome_vault.status())
                        } else {
                            format!(
                                "reborn {}  {}  {}",
                                chronicle.status(),
                                bestiary.status(),
                                genome_vault.status(),
                            )
                        };
                    }
                    KeyCode::Char('l') => match genome_vault.load_random_snapshot()? {
                        Some((genome_id, snapshot)) => {
                            world = World::from_genome_snapshot(snapshot, world.w, world.h);
                            status_note = format!(
                                "loaded genome={}  {}  {}  {}",
                                genome_id,
                                chronicle.status(),
                                bestiary.status(),
                                genome_vault.status(),
                            );
                        }
                        None => {
                            status_note =
                                "no saved genomes yet, press s to save one first".to_string();
                        }
                    },
                    KeyCode::Char('L') => match genome_vault.load_best_snapshot()? {
                        Some((genome_id, snapshot)) => {
                            world = World::from_genome_snapshot(snapshot, world.w, world.h);
                            status_note = format!(
                                "loaded best genome={}  {}  {}  {}",
                                genome_id,
                                chronicle.status(),
                                bestiary.status(),
                                genome_vault.status(),
                            );
                        }
                        None => {
                            status_note =
                                "no saved genomes yet, press s to save one first".to_string();
                        }
                    },
                    KeyCode::Char('m') => {
                        let current = world.genome_snapshot("mutated_current_source");
                        let child_snapshot = genome_vault.mutated_current_snapshot(&current)?;
                        let child_id = genome_vault.save_snapshot(&child_snapshot)?;

                        world = World::from_genome_snapshot(child_snapshot, world.w, world.h);
                        for _ in 0..5 {
                            world.seed_life();
                        }
                        world.refresh_motion_baseline();

                        chronicle.save()?;
                        bestiary.save()?;
                        genome_vault.save()?;

                        status_note = format!(
                            "mutated current={}  {}  {}  {}",
                            child_id,
                            chronicle.status(),
                            bestiary.status(),
                            genome_vault.status()
                        );
                    }
                    KeyCode::Char('n') => match genome_vault.mutated_best_snapshot()? {
                        Some((parent_id, child_snapshot)) => {
                            let child_id = genome_vault.save_snapshot(&child_snapshot)?;
                            world = World::from_genome_snapshot(child_snapshot, world.w, world.h);
                            for _ in 0..5 {
                                world.seed_life();
                            }
                            world.refresh_motion_baseline();

                            chronicle.save()?;
                            bestiary.save()?;
                            genome_vault.save()?;

                            status_note = format!(
                                "spawned mutation={} parent={} {}",
                                child_id,
                                parent_id,
                                genome_vault.status(),
                            );
                        }
                        None => {
                            status_note =
                                "no genome available to mutate yet, press s first".to_string();
                        }
                    },
                    KeyCode::Char('b') => match genome_vault.breed_best_two()? {
                        Some((parent_a, parent_b, child_snapshot)) => {
                            let child_id = genome_vault.save_snapshot(&child_snapshot)?;
                            world = World::from_genome_snapshot(child_snapshot, world.w, world.h);
                            for _ in 0..5 {
                                world.seed_life();
                            }
                            world.refresh_motion_baseline();

                            chronicle.save()?;
                            bestiary.save()?;
                            genome_vault.save()?;

                            status_note = format!(
                                "bred elite hybrid={} parents={} + {} {}",
                                child_id,
                                parent_a,
                                parent_b,
                                genome_vault.status(),
                            );
                        }
                        None => {
                            status_note =
                                "need at least two saved genomes before breeding".to_string();
                        }
                    },
                    KeyCode::Char('B') => match genome_vault.breed_random_two()? {
                        Some((parent_a, parent_b, child_snapshot)) => {
                            let child_id = genome_vault.save_snapshot(&child_snapshot)?;
                            world = World::from_genome_snapshot(child_snapshot, world.w, world.h);
                            for _ in 0..5 {
                                world.seed_life();
                            }
                            world.refresh_motion_baseline();

                            chronicle.save()?;
                            bestiary.save()?;
                            genome_vault.save()?;

                            status_note = format!(
                                "bred random hybrid={} parents={} + {} {}",
                                child_id,
                                parent_a,
                                parent_b,
                                genome_vault.status(),
                            );
                        }
                        None => {
                            status_note =
                                "need at least two saved genomes before breeding".to_string();
                        }
                    },
                    KeyCode::Char('g') => {
                        gpu_live_enabled = !gpu_live_enabled;
                        status_note = if gpu_live_enabled {
                            "GPU LIVE MODE ENABLED: prototype shader backend active".to_string()
                        } else {
                            "CPU RAYON MODE ENABLED: full Primordia engine active".to_string()
                        };
                    }
                    _ => {}
                }
            }
        }

        let mut catchup = 0;
        while last_sim_tick.elapsed() >= sim_step && catchup < 2 {
            if gpu_live_enabled {
                #[cfg(feature = "gpu")]
                {
                    let mut gpu_rules = Vec::with_capacity(world.rules.len());
                    let mut gpu_taps = Vec::new();

                    for rule in &world.rules {
                        let tap_start = gpu_taps.len() as u32;

                        for tap in &rule.taps {
                            gpu_taps.push(gpu::GpuTapData {
                                dx: tap.dx,
                                dy: tap.dy,
                                weight: tap.weight,
                                _pad: 0.0,
                            });
                        }

                        gpu_rules.push(gpu::GpuRuleData {
                            src_ch: rule.from as u32,
                            to: rule.to as u32,
                            tap_start,
                            tap_count: rule.taps.len() as u32,
                            mu: rule.mu,
                            sigma: rule.sigma,
                            weight: rule.weight,
                            _pad: 0.0,
                        });
                    }

                    let needs_engine = real_lenia_gpu
                        .as_ref()
                        .map(|engine| {
                            !engine.matches(
                                world.w as u32,
                                world.h as u32,
                                world.channels as u32,
                                world.cells.len(),
                                gpu_rules.len(),
                                gpu_taps.len(),
                            )
                        })
                        .unwrap_or(true);

                    if needs_engine {
                        match gpu::RealLeniaGpuEngine::new(
                            world.w as u32,
                            world.h as u32,
                            world.channels as u32,
                            world.cells.len(),
                            gpu_rules.len(),
                            gpu_taps.len(),
                        ) {
                            Ok(engine) => {
                                real_lenia_gpu = Some(engine);
                            }
                            Err(err) => {
                                gpu_live_enabled = false;
                                real_lenia_gpu = None;
                                status_note = format!("GPU init failed: {}, returned to CPU", err);
                                world.step();
                                last_sim_tick += sim_step;
                                catchup += 1;
                                continue;
                            }
                        }
                    }

                    let result = real_lenia_gpu
                        .as_mut()
                        .expect("GPU engine should exist after initialization")
                        .step(&world.cells, &gpu_rules, &gpu_taps);

                    match result {
                        Ok(next_cells) if next_cells.len() == world.cells.len() => {
                            world.cells = next_cells;
                            world.tick = world.tick.saturating_add(1);
                            world.update_motion_memory();
                        }
                        Ok(_) => {
                            gpu_live_enabled = false;
                            real_lenia_gpu = None;
                            status_note =
                                "GPU live disabled: readback size mismatch, returned to CPU"
                                    .to_string();
                            world.step();
                        }
                        Err(err) => {
                            gpu_live_enabled = false;
                            real_lenia_gpu = None;
                            status_note = format!("GPU live disabled: {}, returned to CPU", err);
                            world.step();
                        }
                    }
                }

                #[cfg(not(feature = "gpu"))]
                {
                    gpu_live_enabled = false;
                    status_note = "GPU live unavailable in CPU build, returned to CPU".to_string();
                    world.step();
                }
            } else {
                world.step();
            }
            if world.is_extinct() {
                extinction_ticks = extinction_ticks.saturating_add(1);
            } else {
                extinction_ticks = 0;
            }

            if extinction_ticks >= extinction_threshold {
                let record = world.chronicle_record("extinction_resuscitation");
                chronicle.record(record.clone());

                let discovery = bestiary.consider(&record)?;
                if discovery.is_some() {
                    let _ = genome_vault.save_snapshot(
                        &world.genome_snapshot("bestiary_extinction_resuscitation"),
                    )?;
                }

                extinction_rebirths = extinction_rebirths.saturating_add(1);

                let recovery_cause;

                if let Some((parent_id, child_snapshot)) = genome_vault.mutated_best_snapshot()? {
                    let child_id = genome_vault.save_snapshot(&child_snapshot)?;
                    world.inject_genome_snapshot(child_snapshot);

                    recovery_cause = format!(
                        "memory mutation injection child={} parent={}",
                        child_id, parent_id
                    );
                } else if let Some((parent_a, parent_b, child_snapshot)) =
                    genome_vault.breed_best_two()?
                {
                    let child_id = genome_vault.save_snapshot(&child_snapshot)?;
                    world.inject_genome_snapshot(child_snapshot);

                    recovery_cause = format!(
                        "hybrid memory injection child={} parents={}+{}",
                        child_id, parent_a, parent_b
                    );
                } else if let Some((genome_id, snapshot)) = genome_vault.load_best_snapshot()? {
                    world.inject_genome_snapshot(snapshot);

                    recovery_cause = format!("best genome injection genome={}", genome_id);
                } else {
                    world = World::new(world.w, world.h, chronicle.suggest_bias());
                    for _ in 0..8 {
                        world.seed_life();
                    }
                    world.refresh_motion_baseline();

                    recovery_cause = "fresh memory-biased rebirth".to_string();
                }

                extinction_ticks = 0;

                chronicle.save()?;
                bestiary.save()?;
                genome_vault.save()?;

                status_note = if let Some(note) = discovery {
                    format!(
                        "CAUSE=EXTINCTION RECOVERY #{}  {}  {}  {}",
                        extinction_rebirths,
                        recovery_cause,
                        note,
                        genome_vault.status(),
                    )
                } else {
                    format!(
                        "CAUSE=EXTINCTION RECOVERY #{}  {}  runs={}  {}  {}  {}",
                        extinction_rebirths,
                        recovery_cause,
                        chronicle.total_runs_recorded,
                        chronicle.status(),
                        bestiary.status(),
                        genome_vault.status(),
                    )
                };
            }

            last_sim_tick += sim_step;
            catchup += 1;
        }
        if last_render.elapsed() < render_step {
            continue;
        }
        last_render = Instant::now();

        terminal.draw(|frame| {
            let area = frame.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(8),
                    Constraint::Min(10),
                    Constraint::Length(3),
                ])
                .split(area);

            let canvas_w = chunks[1].width.saturating_sub(2) as usize;
            let canvas_h = chunks[1].height.saturating_sub(2) as usize;
            world.resize(canvas_w, canvas_h);

            let mut mass_spans = Vec::new();
            for c in 0..world.channels {
                mass_spans.push(Span::styled(
                    format!("{}={:.3}  ", World::channel_name(c), world.channel_mass(c)),
                    Style::default().fg(World::channel_color(c)),
                ));
            }

            let header = Paragraph::new(vec![
                Line::from(vec![
                    Span::styled("PRIMORDIA", Style::default().fg(Color::Magenta)),
                    Span::raw("  |  randomized multi-channel Lenia genome"),
                ]),
                Line::from(format!(
                    "seed={}  tick={}  mass={:.4}  motion={:.3}  entropy={:.3}  field={}x{}",
                    world.seed,
                    world.tick,
                    world.mass(),
                    world.motion_score,
                    world.entropy_score,
                    world.w,
                    world.h
                )),
                Line::from(format!(
                    "channels={}  rules={}  base_rules={}  kernel_radius={}",
                    world.channels,
                    world.rules.len(),
                    world.base_rules,
                    world.radius
                )),
                Line::from(mass_spans),
                Line::from(status_note.clone()),
            ])
            .block(Block::default().borders(Borders::ALL).title("Genesis Core"));
            frame.render_widget(header, chunks[0]);

            let mut lines = Vec::with_capacity(world.h);
            for y in 0..world.h {
                let mut spans = Vec::with_capacity(world.w);
                for x in 0..world.w {
                    let (glyph, color) = world.cell_visual(x, y);
                    spans.push(Span::styled(glyph, Style::default().fg(color)));
                }
                lines.push(Line::from(spans));
            }

            let canvas = Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title("Living Field"));
            frame.render_widget(canvas, chunks[1]);

            let footer = Paragraph::new(vec![
                Line::from(vec![
                    Span::styled("q", Style::default().fg(Color::Red)),
                    Span::raw(" quit  "),
                    Span::styled("s", Style::default().fg(Color::Green)),
                    Span::raw(" save  "),
                    Span::styled("r", Style::default().fg(Color::Yellow)),
                    Span::raw(" rebirth  "),
                    Span::styled("g", Style::default().fg(Color::Cyan)),
                    Span::raw(" cpu/gpu"),
                ]),
                Line::from(vec![
                    Span::styled("l", Style::default().fg(Color::LightBlue)),
                    Span::raw(" load random  "),
                    Span::styled("L", Style::default().fg(Color::LightBlue)),
                    Span::raw(" load best  "),
                    Span::styled("m", Style::default().fg(Color::Magenta)),
                    Span::raw(" mutate current  "),
                    Span::styled("n", Style::default().fg(Color::Magenta)),
                    Span::raw(" mutate best"),
                ]),
                Line::from(vec![
                    Span::styled("b", Style::default().fg(Color::LightGreen)),
                    Span::raw(" breed best two  "),
                    Span::styled("B", Style::default().fg(Color::LightGreen)),
                    Span::raw(" breed random two"),
                ]),
            ])
            .block(Block::default().borders(Borders::ALL).title("Controls"));
            frame.render_widget(footer, chunks[2]);
        })?;
    }
}
