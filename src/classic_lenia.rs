#![allow(dead_code)]

#[derive(Clone, Copy, Debug)]
pub enum ClassicKernelShell {
    Exponential,
    Polynomial,
    Step,
}

#[derive(Clone, Debug)]
pub struct ClassicKernel {
    pub radius: usize,
    pub peaks: Vec<f32>,
    pub shell: ClassicKernelShell,
    pub weights: Vec<f32>,
    pub size: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ClassicGrowth {
    pub mu: f32,
    pub sigma: f32,
    pub weight: f32,
    pub dt: f32,
}

#[derive(Clone, Debug)]
pub struct ClassicLeniaField {
    pub w: usize,
    pub h: usize,
    pub cells: Vec<f32>,
}

impl ClassicLeniaField {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            cells: vec![0.0; w * h],
        }
    }

    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }

    pub fn get_wrap(&self, x: i32, y: i32) -> f32 {
        let xx = x.rem_euclid(self.w as i32) as usize;
        let yy = y.rem_euclid(self.h as i32) as usize;
        self.cells[self.idx(xx, yy)]
    }

    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        let idx = self.idx(x, y);
        self.cells[idx] = v.clamp(0.0, 1.0);
    }
}

impl ClassicKernel {
    pub fn new(radius: usize, peaks: Vec<f32>, shell: ClassicKernelShell) -> Self {
        let radius = radius.max(1);
        let size = radius * 2 + 1;
        let mut weights = vec![0.0; size * size];
        let mut sum = 0.0f32;

        let peak_count = peaks.len().max(1);

        for ky in 0..size {
            for kx in 0..size {
                let dx = kx as i32 - radius as i32;
                let dy = ky as i32 - radius as i32;
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let r = dist / radius as f32;

                if r >= 1.0 {
                    continue;
                }

                let band_pos = r * peak_count as f32;
                let band = band_pos.floor() as usize;
                let band = band.min(peak_count - 1);
                let local_r = band_pos.fract();

                let peak = peaks.get(band).copied().unwrap_or(1.0);
                let shell_v = kernel_shell(shell, local_r);
                let v = peak * shell_v;

                let idx = ky * size + kx;
                weights[idx] = v;
                sum += v;
            }
        }

        if sum > 0.0 {
            for v in &mut weights {
                *v /= sum;
            }
        }

        Self {
            radius,
            peaks,
            shell,
            weights,
            size,
        }
    }

    pub fn sample(&self, field: &ClassicLeniaField, x: usize, y: usize) -> f32 {
        let mut acc = 0.0f32;

        for ky in 0..self.size {
            for kx in 0..self.size {
                let weight = self.weights[ky * self.size + kx];
                if weight == 0.0 {
                    continue;
                }

                let dx = kx as i32 - self.radius as i32;
                let dy = ky as i32 - self.radius as i32;
                acc += field.get_wrap(x as i32 + dx, y as i32 + dy) * weight;
            }
        }

        acc
    }
}

pub fn kernel_shell(shell: ClassicKernelShell, r: f32) -> f32 {
    let r = r.clamp(0.0, 1.0);

    match shell {
        ClassicKernelShell::Exponential => {
            let k = 4.0 * r * (1.0 - r);
            if k <= 0.0 {
                0.0
            } else {
                (4.0 * (1.0 - 1.0 / k)).exp()
            }
        }
        ClassicKernelShell::Polynomial => {
            let k = 4.0 * r * (1.0 - r);
            k.max(0.0).powf(4.0)
        }
        ClassicKernelShell::Step => {
            if (0.25..=0.75).contains(&r) {
                1.0
            } else {
                0.0
            }
        }
    }
}

pub fn growth_gaussian(u: f32, mu: f32, sigma: f32) -> f32 {
    let sigma = sigma.max(0.0001);
    let d = u - mu;
    2.0 * (-(d * d) / (2.0 * sigma * sigma)).exp() - 1.0
}

pub fn step_classic_lenia(
    field: &ClassicLeniaField,
    kernel: &ClassicKernel,
    growth: ClassicGrowth,
) -> ClassicLeniaField {
    let mut next = ClassicLeniaField::new(field.w, field.h);

    for y in 0..field.h {
        for x in 0..field.w {
            let idx = field.idx(x, y);
            let u = kernel.sample(field, x, y);
            let g = growth_gaussian(u, growth.mu, growth.sigma);
            let v = field.cells[idx] + growth.dt * growth.weight * g;
            next.cells[idx] = v.clamp(0.0, 1.0);
        }
    }

    next
}

pub fn orbium_like_kernel() -> (ClassicKernel, ClassicGrowth) {
    (
        ClassicKernel::new(13, vec![1.0], ClassicKernelShell::Exponential),
        ClassicGrowth {
            mu: 0.15,
            sigma: 0.015,
            weight: 1.0,
            dt: 0.1,
        },
    )
}

pub fn stamp_orbium_like(field: &mut ClassicLeniaField, cx: usize, cy: usize, scale: f32) {
    let pattern: &[&[f32]] = &[
        &[0.00, 0.00, 0.08, 0.22, 0.08, 0.00, 0.00],
        &[0.00, 0.15, 0.42, 0.70, 0.42, 0.15, 0.00],
        &[0.10, 0.50, 0.95, 0.65, 0.95, 0.50, 0.10],
        &[0.20, 0.75, 0.70, 0.20, 0.70, 0.75, 0.20],
        &[0.10, 0.50, 0.95, 0.65, 0.95, 0.50, 0.10],
        &[0.00, 0.15, 0.42, 0.70, 0.42, 0.15, 0.00],
        &[0.00, 0.00, 0.08, 0.22, 0.08, 0.00, 0.00],
    ];

    let half = 3i32;

    for (py, row) in pattern.iter().enumerate() {
        for (px, value) in row.iter().enumerate() {
            let x = (cx as i32 + px as i32 - half).rem_euclid(field.w as i32) as usize;
            let y = (cy as i32 + py as i32 - half).rem_euclid(field.h as i32) as usize;
            let idx = field.idx(x, y);
            field.cells[idx] = (field.cells[idx] + value * scale).clamp(0.0, 1.0);
        }
    }
}
