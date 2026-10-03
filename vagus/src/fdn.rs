//! Self-resampling grain feedback delay network.
//!
//! Each of the eight lines records its own output into a ring buffer, tags
//! every hop of that buffer with loudness and brightness, and replays it
//! through a small grain scheduler. Grain choice is a tournament against a
//! brightness target, so the loop evolves toward the target each pass.

use crate::corpus::{HOP, Slot, brightness};
use crate::util::{Rng, hann, read_wrapped};
use std::f64::consts::TAU;

pub const FDN_LINES: usize = 8;
const FDN_GRAINS: usize = 6;
const FDN_DENSITY: f64 = 3.0;
const RING_SECONDS: f64 = 3.5;

/// Pitch shift each line applies, as a multiple of the Shift control.
const LINE_SHIFT: [f64; FDN_LINES] = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0];
/// Fraction of the Memory control each line may reach back.
const LINE_MEM: [f64; FDN_LINES] = [1.0, 0.89, 0.79, 0.71, 0.63, 0.56, 0.5, 0.45];
const IN_GAIN: f64 = 0.35;

#[derive(Clone, Copy, Debug)]
pub struct FdnSettings {
    pub mix: f64,
    pub feedback: f64,
    pub memory_s: f64,
    pub grain_s: f64,
    pub shift_semi: f64,
    pub tone: f64,
    pub focus: u32,
    pub damp_hz: f64,
}

impl Default for FdnSettings {
    fn default() -> Self {
        FdnSettings {
            mix: 0.35,
            feedback: 0.7,
            memory_s: 1.0,
            grain_s: 0.09,
            shift_semi: 0.0,
            tone: 0.5,
            focus: 3,
            damp_hz: 7000.0,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct FGrain {
    pos: f64,
    inc: f64,
    age: u32,
    len: u32,
    on: bool,
}

struct FdnLine {
    ring: Vec<f32>,
    slots: Vec<Slot>,
    write: usize,
    grains: [FGrain; FDN_GRAINS],
    countdown: f64,
    lp: f64,
    dc_x: f64,
    dc_y: f64,
    prev: f64,
    acc_e: f64,
    acc_d: f64,
    peak: f32,
    rng: Rng,
}

impl FdnLine {
    fn new(ring_len: usize, seed: u64) -> Self {
        FdnLine {
            ring: vec![0.0; ring_len],
            slots: vec![Slot::default(); ring_len / HOP],
            write: 0,
            grains: [FGrain::default(); FDN_GRAINS],
            countdown: 0.0,
            lp: 0.0,
            dc_x: 0.0,
            dc_y: 0.0,
            prev: 0.0,
            acc_e: 0.0,
            acc_d: 0.0,
            peak: 0.0,
            rng: Rng::new(seed),
        }
    }

    fn read(&mut self) -> f64 {
        let g = 1.0 / FDN_DENSITY.sqrt();
        let rl = self.ring.len() as f64;
        let mut sum = 0.0;
        for gr in self.grains.iter_mut() {
            if !gr.on {
                continue;
            }
            if gr.age >= gr.len {
                gr.on = false;
                continue;
            }
            let w = hann(gr.age as f64 / gr.len as f64);
            sum += w * read_wrapped(&self.ring, gr.pos);
            gr.pos += gr.inc;
            if gr.pos >= rl {
                gr.pos -= rl;
            }
            gr.age += 1;
        }
        sum * g
    }

    fn maybe_spawn(&mut self, s: &FdnSettings, ratio: f64, mem_frac: f64, sr: f64) {
        self.countdown -= 1.0;
        if self.countdown > 0.0 {
            return;
        }
        let len = ((s.grain_s * sr) as usize).max(64);
        self.countdown += len as f64 / FDN_DENSITY * (0.75 + 0.5 * self.rng.unit());

        let ring = self.ring.len();
        // The read head must never be overtaken by the write head. When the
        // grain plays faster than 1x it closes the gap by (ratio-1) per sample.
        let min_age = HOP * 2 + (len as f64 * (ratio - 1.0).max(0.0)) as usize + 8;
        let cap = ring.saturating_sub(len + 8);
        let mut max_age = ((s.memory_s * mem_frac * sr) as usize).min(cap);
        if max_age < min_age + HOP {
            max_age = (min_age + HOP).min(cap);
        }
        if max_age <= min_age {
            return;
        }

        let tb = s.tone as f32;
        let mut best_age = min_age;
        let mut best_cost = f32::MAX;
        for _ in 0..s.focus.max(1) {
            let age = min_age + self.rng.below(max_age - min_age);
            let pos = (self.write + ring - age) % ring;
            let sl = self.slots[pos / HOP];
            let ln = (sl.loud / (self.peak + 1e-9)).min(1.0);
            let db = sl.bright - tb;
            let dl = 1.0 - ln;
            let cost = db * db + 0.5 * dl * dl;
            if cost < best_cost {
                best_cost = cost;
                best_age = age;
            }
        }
        let start = ((self.write + ring - best_age) % ring) as f64;
        if let Some(slot) = self.grains.iter_mut().find(|g| !g.on) {
            *slot = FGrain {
                pos: start,
                inc: ratio,
                age: 0,
                len: len as u32,
                on: true,
            };
        }
    }

    fn write(&mut self, x: f64, damp_a: f64, sr: f64) {
        self.lp += damp_a * (x - self.lp);
        let y = self.lp;
        let dc = y - self.dc_x + 0.995 * self.dc_y;
        self.dc_x = y;
        self.dc_y = dc;
        let mut v = dc.tanh();
        if v.abs() < 1e-15 {
            v = 0.0;
        }
        self.ring[self.write] = v as f32;
        let d = v - self.prev;
        self.prev = v;
        self.acc_e += v * v;
        self.acc_d += d * d;
        self.write += 1;
        if self.write % HOP == 0 {
            let rms = (self.acc_e / HOP as f64).sqrt() as f32;
            let rmsd = (self.acc_d / HOP as f64).sqrt() as f32;
            let n = self.slots.len();
            let idx = (self.write / HOP + n - 1) % n;
            self.slots[idx] = Slot {
                loud: rms,
                bright: brightness(rms, rmsd, sr as f32),
            };
            self.peak = (self.peak * 0.9995).max(rms);
            self.acc_e = 0.0;
            self.acc_d = 0.0;
            if self.write >= self.ring.len() {
                self.write = 0;
            }
        }
    }
}

pub struct Fdn {
    lines: Vec<FdnLine>,
    sr: f64,
    level: f64,
}

impl Fdn {
    pub fn new(sr: f64) -> Self {
        let ring_len = (((RING_SECONDS * sr) as usize) / HOP + 1) * HOP;
        let lines = (0..FDN_LINES)
            .map(|i| FdnLine::new(ring_len, 0xF0D0 + i as u64 * 7919))
            .collect();
        Fdn {
            lines,
            sr,
            level: 0.0,
        }
    }

    /// True once the network has decayed to silence.
    pub fn is_silent(&self) -> bool {
        self.level < 1e-5
    }

    /// Process one sample. Returns the wet stereo signal.
    pub fn tick(&mut self, input: f64, s: &FdnSettings) -> (f64, f64) {
        let mut y = [0.0f64; FDN_LINES];
        for (i, line) in self.lines.iter_mut().enumerate() {
            y[i] = line.read();
        }

        let (mut l, mut r) = (0.0, 0.0);
        for (i, v) in y.iter().enumerate() {
            if i % 2 == 0 {
                l += v;
            } else {
                r += v;
            }
        }
        l *= 0.5;
        r *= 0.5;

        // Householder reflection, an orthogonal mixing matrix.
        let sum: f64 = y.iter().sum::<f64>() * (2.0 / FDN_LINES as f64);
        for v in y.iter_mut() {
            *v -= sum;
        }

        let damp_a = 1.0 - (-TAU * s.damp_hz.clamp(200.0, self.sr * 0.45) / self.sr).exp();
        let fb = s.feedback.clamp(0.0, 1.0) * 0.98;
        for (i, line) in self.lines.iter_mut().enumerate() {
            let ratio = (LINE_SHIFT[i] * s.shift_semi / 12.0).exp2();
            line.maybe_spawn(s, ratio, LINE_MEM[i], self.sr);
            line.write(input * IN_GAIN + y[i] * fb, damp_a, self.sr);
        }

        self.level = (self.level * 0.9999).max(l.abs()).max(r.abs());
        (l, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(fdn: &mut Fdn, s: &FdnSettings, secs_on: f64, secs_total: f64) -> Vec<f64> {
        let sr = 44100.0;
        let mut out = Vec::new();
        for i in 0..(secs_total * sr) as usize {
            let t = i as f64 / sr;
            let x = if t < secs_on {
                0.4 * (TAU * 220.0 * t).sin()
            } else {
                0.0
            };
            let (l, r) = fdn.tick(x, s);
            assert!(l.is_finite() && r.is_finite(), "non finite at {i}");
            out.push(0.5 * (l + r));
        }
        out
    }

    fn rms(x: &[f64]) -> f64 {
        (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn rings_then_decays_with_moderate_feedback() {
        let mut fdn = Fdn::new(44100.0);
        let s = FdnSettings {
            feedback: 0.6,
            ..Default::default()
        };
        let out = drive(&mut fdn, &s, 1.0, 9.0);
        let early = rms(&out[44100..2 * 44100]);
        let late = rms(&out[8 * 44100..]);
        assert!(early > 1e-3, "no sound, early rms {early}");
        assert!(late < early * 0.1, "no decay, early {early} late {late}");
    }

    #[test]
    fn high_feedback_stays_bounded() {
        let mut fdn = Fdn::new(44100.0);
        let s = FdnSettings {
            feedback: 1.0,
            shift_semi: 7.0,
            focus: 8,
            ..Default::default()
        };
        let out = drive(&mut fdn, &s, 2.0, 10.0);
        let peak = out.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(peak < 4.0, "peak {peak}");
    }

    #[test]
    fn extreme_settings_do_not_panic() {
        let mut fdn = Fdn::new(48000.0);
        let s = FdnSettings {
            memory_s: 3.0,
            grain_s: 0.3,
            shift_semi: 12.0,
            focus: 16,
            ..Default::default()
        };
        drive(&mut fdn, &s, 1.0, 6.0);
        let s2 = FdnSettings {
            memory_s: 0.1,
            grain_s: 0.02,
            shift_semi: -12.0,
            ..Default::default()
        };
        drive(&mut fdn, &s2, 0.5, 2.0);
    }

    #[test]
    fn goes_silent_when_idle() {
        let mut fdn = Fdn::new(44100.0);
        let s = FdnSettings {
            feedback: 0.3,
            ..Default::default()
        };
        drive(&mut fdn, &s, 0.2, 12.0);
        assert!(fdn.is_silent());
    }
}
