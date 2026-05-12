use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
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
}

impl Rule {
    fn random(rng: &mut StdRng, channels: usize, radius: i32) -> Self {
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

        Self {
            from,
            to,
            mu: rng.gen_range(0.12..0.46),
            sigma: rng.gen_range(0.022..0.095),
            weight: rng.gen_range(-0.46..0.60),
            taps,
        }
    }
}

impl World {
    fn new(w: usize, h: usize) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self::from_seed(seed, w, h)
    }

    fn from_seed(seed: u64, w: usize, h: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let channels = rng.gen_range(MIN_CHANNELS..=MAX_CHANNELS);
        let base_rules = rng.gen_range(MIN_RULES..=MAX_RULES);
        let radius = rng.gen_range(MIN_RADIUS..=MAX_RADIUS);

        let w = w.max(24);
        let h = h.max(12);

        let mut rules = Vec::new();

        for _ in 0..base_rules {
            rules.push(Rule::random(&mut rng, channels, radius));
        }

        for c in 0..channels {
            let mut self_rule = Rule::random(&mut rng, channels, radius);
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
        };

        world.seed_life();
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
        self.next.copy_from_slice(&self.cells);

        for y in 0..self.h {
            for x in 0..self.w {
                let mut delta = vec![0.0_f32; self.channels];

                for rule in &self.rules {
                    let mut conv = 0.0;

                    for tap in &rule.taps {
                        let nx = self.wrap_x(x as i32 + tap.dx);
                        let ny = self.wrap_y(y as i32 + tap.dy);
                        conv += self.cells[self.idx(nx, ny, rule.from)] * tap.weight;
                    }

                    let growth = bell(conv, rule.mu, rule.sigma) * 2.0 - 1.0;
                    delta[rule.to] += growth * rule.weight;
                }

                for c in 0..self.channels {
                    let idx = self.idx(x, y, c);
                    let old = self.cells[idx];

                    let mut lap = 0.0;
                    lap += self.cells[self.idx(x, self.wrap_y(y as i32 - 1), c)];
                    lap += self.cells[self.idx(x, self.wrap_y(y as i32 + 1), c)];
                    lap += self.cells[self.idx(self.wrap_x(x as i32 - 1), y, c)];
                    lap += self.cells[self.idx(self.wrap_x(x as i32 + 1), y, c)];
                    lap -= old * 4.0;

                    let pressure = old * old * 0.060;
                    let noise = self.rng.gen_range(-0.0007..0.0007);

                    self.next[idx] =
                        (old + DT * delta[c] + lap * 0.007 + noise - pressure).clamp(0.0, 1.0);
                }
            }
        }

        std::mem::swap(&mut self.cells, &mut self.next);
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
    let mut world = World::new(88, 36);

    // Simulation advances at ~60 updates/sec.
    let sim_step = Duration::from_millis(16);

    // Terminal redraws at ~30 FPS.
    let render_step = Duration::from_millis(33);

    let mut last_sim_tick = Instant::now();
    let mut last_render = Instant::now();

    loop {
        while event::poll(Duration::from_millis(1))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('r') => world = World::new(world.w, world.h),
                    _ => {}
                }
            }
        }

        // Advance the simulation independently of rendering.
        let mut catchup = 0;
        while last_sim_tick.elapsed() >= sim_step && catchup < 4 {
            world.step();
            last_sim_tick += sim_step;
            catchup += 1;
        }

        // Skip drawing until the next render interval.
        if last_render.elapsed() < render_step {
            continue;
        }
        last_render = Instant::now();

        terminal.draw(|frame| {
            let area = frame.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(7),
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
                    format!(
                        "{}={:.3}  ",
                        World::channel_name(c),
                        world.channel_mass(c)
                    ),
                    Style::default().fg(World::channel_color(c)),
                ));
            }

            let header = Paragraph::new(vec![
                Line::from(vec![
                    Span::styled("PRIMORDIA", Style::default().fg(Color::Magenta)),
                    Span::raw("  |  randomized multi-channel Lenia genome"),
                ]),
                Line::from(format!(
                    "seed={}  tick={}  mass={:.4}  field={}x{}",
                    world.seed,
                    world.tick,
                    world.mass(),
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
            ])
            .block(Block::default().borders(Borders::ALL).title("Genesis Core"));
            frame.render_widget(header, chunks[0]);

            let mut lines = Vec::with_capacity(world.h);
            for y in 0..world.h {
                let mut spans = Vec::with_capacity(world.w);
                for x in 0..world.w {
                    let (glyph, color) = world.cell_visual(x, y);
                    spans.push(Span::styled(
                        glyph,
                        Style::default().fg(color),
                    ));
                }
                lines.push(Line::from(spans));
            }

            let canvas = Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title("Living Field"));
            frame.render_widget(canvas, chunks[1]);

            let footer = Paragraph::new(
                "q / esc = quit    r = rebirth universe    60 simulation ticks/sec, ~30 FPS rendering",
            )
            .block(Block::default().borders(Borders::ALL).title("Controls"));
            frame.render_widget(footer, chunks[2]);
        })?;
    }
}
