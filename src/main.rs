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

const CHANNELS: usize = 3;
const RULES: usize = 4;
const KERNEL_RADIUS: i32 = 5;
const DT: f32 = 0.055;

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
    cells: Vec<f32>,
    next: Vec<f32>,
    rules: Vec<Rule>,
    rng: StdRng,
}

impl Rule {
    fn random(rng: &mut StdRng, from: usize, to: usize) -> Self {
        let ring_count = rng.gen_range(2..5);
        let mut rings = Vec::new();

        for _ in 0..ring_count {
            rings.push(rng.gen_range(0.05..1.0));
        }

        let mut taps = Vec::new();
        let mut total = 0.0;

        for dy in -KERNEL_RADIUS..=KERNEL_RADIUS {
            for dx in -KERNEL_RADIUS..=KERNEL_RADIUS {
                if dx == 0 && dy == 0 {
                    continue;
                }

                if rng.gen_bool(0.42) {
                    continue;
                }

                let dist = ((dx * dx + dy * dy) as f32).sqrt() / KERNEL_RADIUS as f32;
                if dist > 1.0 {
                    continue;
                }

                let ring = ((dist * ring_count as f32).floor() as usize).min(ring_count - 1);
                let center = (ring as f32 + 0.5) / ring_count as f32;
                let shell = (-((dist - center).powi(2)) / rng.gen_range(0.010..0.035)).exp();
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
            mu: rng.gen_range(0.14..0.42),
            sigma: rng.gen_range(0.025..0.090),
            weight: rng.gen_range(-0.42..0.55),
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

        let mut rng = StdRng::seed_from_u64(seed);
        let mut rules = Vec::new();

        for _ in 0..RULES {
            let from = rng.gen_range(0..CHANNELS);
            let to = rng.gen_range(0..CHANNELS);
            rules.push(Rule::random(&mut rng, from, to));
        }

        for c in 0..CHANNELS {
            rules.push(Rule::random(&mut rng, c, c));
        }

        let mut world = Self {
            seed,
            tick: 0,
            w: w.max(24),
            h: h.max(12),
            cells: vec![0.0; w.max(24) * h.max(12) * CHANNELS],
            next: vec![0.0; w.max(24) * h.max(12) * CHANNELS],
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

        let mut new_cells = vec![0.0; new_w * new_h * CHANNELS];
        let copy_w = self.w.min(new_w);
        let copy_h = self.h.min(new_h);

        for y in 0..copy_h {
            for x in 0..copy_w {
                for c in 0..CHANNELS {
                    let old_idx = self.idx(x, y, c);
                    let new_idx = (y * new_w + x) * CHANNELS + c;
                    new_cells[new_idx] = self.cells[old_idx];
                }
            }
        }

        self.w = new_w;
        self.h = new_h;
        self.cells = new_cells;
        self.next = vec![0.0; self.w * self.h * CHANNELS];
        self.seed_life();
    }

    fn idx(&self, x: usize, y: usize, c: usize) -> usize {
        (y * self.w + x) * CHANNELS + c
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
                let mut delta = [0.0_f32; CHANNELS];

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

                for c in 0..CHANNELS {
                    let idx = self.idx(x, y, c);
                    let old = self.cells[idx];

                    let mut lap = 0.0;
                    lap += self.cells[self.idx(x, self.wrap_y(y as i32 - 1), c)];
                    lap += self.cells[self.idx(x, self.wrap_y(y as i32 + 1), c)];
                    lap += self.cells[self.idx(self.wrap_x(x as i32 - 1), y, c)];
                    lap += self.cells[self.idx(self.wrap_x(x as i32 + 1), y, c)];
                    lap -= old * 4.0;

                    let pressure = old * old * 0.055;
                    let noise = self.rng.gen_range(-0.0008..0.0008);

                    self.next[idx] =
                        (old + DT * delta[c] + lap * 0.008 + noise - pressure).clamp(0.0, 1.0);
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
        let min_w = self.w.max(24);
        let min_h = self.h.max(12);

        for _ in 0..self.rng.gen_range(4..9) {
            let cx = self.rng.gen_range(4..min_w - 4) as i32;
            let cy = self.rng.gen_range(3..min_h - 3) as i32;
            let radius = self.rng.gen_range(3..8) as i32;

            for y in -radius..=radius {
                for x in -radius..=radius {
                    let d = ((x * x + y * y) as f32).sqrt();
                    if d <= radius as f32 {
                        let softness = 1.0 - d / radius as f32;
                        let px = self.wrap_x(cx + x);
                        let py = self.wrap_y(cy + y);

                        for c in 0..CHANNELS {
                            let idx = self.idx(px, py, c);
                            let v = self.rng.gen_range(0.08..0.85) * softness;
                            self.cells[idx] = (self.cells[idx] + v).clamp(0.0, 1.0);
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

    fn cell_visual(&self, x: usize, y: usize) -> (&'static str, Color) {
        let a = self.cells[self.idx(x, y, 0)];
        let b = self.cells[self.idx(x, y, 1)];
        let c = self.cells[self.idx(x, y, 2)];
        let m = a.max(b).max(c);
        let total = a + b + c;
        let spread = (a - b).abs() + (b - c).abs() + (c - a).abs();

        let glyph = match m {
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

        let color = if total > 2.15 {
            Color::White
        } else if spread < 0.09 && total > 0.32 {
            Color::Yellow
        } else if a > b * 1.25 && a > c * 1.25 {
            Color::Cyan
        } else if b > a * 1.25 && b > c * 1.25 {
            Color::Green
        } else if c > a * 1.25 && c > b * 1.25 {
            Color::Magenta
        } else if a + b > c * 1.45 {
            Color::LightCyan
        } else if b + c > a * 1.45 {
            Color::LightGreen
        } else if a + c > b * 1.45 {
            Color::LightMagenta
        } else if m > 0.72 {
            Color::Red
        } else {
            Color::Blue
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
    let mut last_tick = Instant::now();

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

        if last_tick.elapsed() >= Duration::from_millis(16) {
            world.step();
            last_tick = Instant::now();
        }

        terminal.draw(|frame| {
            let area = frame.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(6),
                    Constraint::Min(10),
                    Constraint::Length(3),
                ])
                .split(area);

            let canvas_w = chunks[1].width.saturating_sub(2) as usize;
            let canvas_h = chunks[1].height.saturating_sub(2) as usize;
            world.resize(canvas_w, canvas_h);

            let header = Paragraph::new(vec![
                Line::from(vec![
                    Span::styled("PRIMORDIA", Style::default().fg(Color::Magenta)),
                    Span::raw("  |  expanded randomized multi-channel Lenia"),
                ]),
                Line::from(format!(
                    "seed={}  tick={}  mass={:.4}  rules={}  channels={}  field={}x{}",
                    world.seed,
                    world.tick,
                    world.mass(),
                    world.rules.len(),
                    CHANNELS,
                    world.w,
                    world.h
                )),
                Line::from(format!(
                    "channel mass  cyan={:.4}  green={:.4}  magenta={:.4}",
                    world.channel_mass(0),
                    world.channel_mass(1),
                    world.channel_mass(2)
                )),
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
                "q / esc = quit    r = rebirth universe    dynamically resizes to terminal",
            )
            .block(Block::default().borders(Borders::ALL).title("Controls"));
            frame.render_widget(footer, chunks[2]);
        })?;
    }
}
