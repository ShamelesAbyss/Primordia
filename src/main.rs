mod bestiary;
mod chronicle;
mod genome;

use anyhow::Result;
use bestiary::Bestiary;
use chronicle::{unix_now, Chronicle, ChronicleBias, RunRecord};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use genome::{GenomeSnapshot, GenomeVault, KernelTapGenome, RuleGenome};
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
const MAX_CHANNELS: usize = 6;
const MIN_RULES: usize = 4;
const MAX_RULES: usize = 12;
const MIN_RADIUS: i32 = 4;
const MAX_RADIUS: i32 = 7;
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

impl Rule {
    fn random(
        rng: &mut StdRng,
        channels: usize,
        radius: i32,
        bias: Option<&ChronicleBias>,
    ) -> Self {
        let from = rng.gen_range(0..channels);
        let to = rng.gen_range(0..channels);
        let ring_count = rng.gen_range(2..6);
        let sparsity = rng.gen_range(0.25..0.58);
        let mut rings = Vec::new();

        for _ in 0..ring_count {
            rings.push(rng.gen_range(0.04..1.0));
        }

        let mut taps = Vec::new();
        let mut total = 0.0;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx == 0 && dy == 0 {
                    continue;
                }

                if rng.gen_bool(sparsity) {
                    continue;
                }

                let dist = ((dx * dx + dy * dy) as f32).sqrt() / radius as f32;
                if dist > 1.0 {
                    continue;
                }

                let ring = ((dist * ring_count as f32).floor() as usize).min(ring_count - 1);
                let center = (ring as f32 + 0.5) / ring_count as f32;
                let shell = (-((dist - center).powi(2)) / rng.gen_range(0.010..0.040)).exp();
                let weight = shell * rings[ring];

                if weight > 0.0001 {
                    taps.push(KernelTap { dx, dy, weight });
                    total += weight;
                }
            }
        }

        for tap in taps.iter_mut() {
            tap.weight /= total.max(0.0001);
        }

        let mut mu = rng.gen_range(0.12..0.46);
        let mut sigma = rng.gen_range(0.022..0.095);
        let mut weight = rng.gen_range(-0.46..0.60);

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
            mu,
            sigma,
            weight,
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
        let w = w.max(24);
        let h = h.max(12);
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

        let mut world = Self {
            seed: snapshot.seed,
            tick: 0,
            w,
            h,
            channels,
            base_rules: snapshot.base_rules,
            radius: snapshot.kernel_radius,
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

    fn resize(&mut self, new_w: usize, new_h: usize) {
        let new_w = new_w.max(24);
        let new_h = new_h.max(12);

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

        if self.tick % 2400 == 0 {
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

    fn update_motion_memory(&mut self) {
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

    fn channel_name(c: usize) -> &'static str {
        match c {
            0 => "cyan",
            1 => "green",
            2 => "magenta",
            3 => "red",
            4 => "blue",
            _ => "yellow",
        }
    }

    fn channel_color(c: usize) -> Color {
        match c {
            0 => Color::Cyan,
            1 => Color::Green,
            2 => Color::Magenta,
            3 => Color::Red,
            4 => Color::Blue,
            _ => Color::Yellow,
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
            v if v < 0.018 => " ",
            v if v < 0.050 => "·",
            v if v < 0.090 => "∙",
            v if v < 0.145 => "•",
            v if v < 0.220 => "○",
            v if v < 0.320 => "◌",
            v if v < 0.440 => "◉",
            v if v < 0.570 => "●",
            v if v < 0.700 => "◆",
            v if v < 0.850 => "⬢",
            _ => "✦",
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
            version: 1,
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
    let mut world = World::new(88, 36, chronicle.suggest_bias());

    let sim_step = Duration::from_millis(16);
    let render_step = Duration::from_millis(33);

    let mut last_sim_tick = Instant::now();
    let mut last_render = Instant::now();
    let mut status_note = format!(
        "{}  {}  {}",
        chronicle.status(),
        bestiary.status(),
        genome_vault.status()
    );

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
                                genome_vault.status()
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
                                genome_vault.status()
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
                                genome_vault.status()
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
                                genome_vault.status()
                            );
                        }
                        None => {
                            status_note =
                                "no saved genomes yet, press s to save one first".to_string();
                        }
                    },
                    _ => {}
                }
            }
        }

        let mut catchup = 0;
        while last_sim_tick.elapsed() >= sim_step && catchup < 4 {
            world.step();
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

            let footer = Paragraph::new(
                "q / esc = save + quit    s = save genome    r = rebirth    l = load random genome    L = load best genome",
            )
            .block(Block::default().borders(Borders::ALL).title("Controls"));
            frame.render_widget(footer, chunks[2]);
        })?;
    }
}
