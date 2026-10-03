//! The analysed audio the grain clouds draw from.
//!
//! A `Corpus` is immutable once built. It is created off the audio thread
//! (by the loader, or by `Corpus::builtin`) and shared with the audio thread
//! through an `Arc`.

use crate::util::{Rng, read_clamped};
use std::f64::consts::TAU;

/// Samples per analysis slot.
pub const HOP: usize = 512;

/// Per-slot descriptors, both normalised to roughly 0..1.
#[derive(Clone, Copy, Default, Debug)]
pub struct Slot {
    pub loud: f32,
    pub bright: f32,
}

/// Cheap brightness estimate from the energy of a signal and of its first
/// difference. For a sine the ratio is 2*sin(pi*f/sr), which we invert to a
/// frequency, then place on a log scale between 50 Hz and 16 kHz.
pub fn brightness(rms: f32, rms_diff: f32, sample_rate: f32) -> f32 {
    if rms < 1e-6 {
        return 0.0;
    }
    let r = (rms_diff / (rms * 2.0)).clamp(0.0, 1.0);
    let f = (r.asin() / std::f32::consts::PI * sample_rate).max(20.0);
    ((f / 50.0).log2() / (16000.0f32 / 50.0).log2()).clamp(0.0, 1.0)
}

/// One column of a waveform overview, for drawing in an editor.
#[derive(Clone, Copy, Debug, Default)]
pub struct Overview {
    pub min: f32,
    pub max: f32,
    /// Mean brightness 0..1, handy for colouring the waveform.
    pub bright: f32,
}

pub struct Corpus {
    pub name: String,
    pub samples: Vec<f32>,
    pub sample_rate: f64,
    pub slots: Vec<Slot>,
}

impl Corpus {
    /// Build a corpus from mono audio. Peak-normalises and analyses.
    pub fn from_mono(name: impl Into<String>, mut samples: Vec<f32>, sample_rate: f64) -> Self {
        let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        if peak > 1e-9 {
            let g = 0.9 / peak;
            for s in samples.iter_mut() {
                *s *= g;
            }
        }
        let n_slots = (samples.len() / HOP).max(1);
        let mut raw = Vec::with_capacity(n_slots);
        let mut max_rms = 1e-9f32;
        for s in 0..n_slots {
            let a = s * HOP;
            let b = ((s + 1) * HOP).min(samples.len());
            let chunk = &samples[a.min(samples.len())..b];
            let mut e = 0.0f64;
            let mut d = 0.0f64;
            let mut prev = 0.0f64;
            for &x in chunk {
                let x = x as f64;
                e += x * x;
                d += (x - prev) * (x - prev);
                prev = x;
            }
            let n = chunk.len().max(1) as f64;
            let rms = (e / n).sqrt() as f32;
            let rmsd = (d / n).sqrt() as f32;
            max_rms = max_rms.max(rms);
            raw.push((rms, rmsd));
        }
        let slots = raw
            .into_iter()
            .map(|(rms, rmsd)| Slot {
                loud: (rms / max_rms).sqrt(),
                bright: brightness(rms, rmsd, sample_rate as f32),
            })
            .collect();
        Corpus {
            name: name.into(),
            samples,
            sample_rate,
            slots,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    #[inline]
    pub fn read(&self, pos: f64) -> f64 {
        read_clamped(&self.samples, pos)
    }

    /// Min, max and brightness for `bins` equal columns across the file.
    pub fn overview(&self, bins: usize) -> Vec<Overview> {
        let n = self.samples.len();
        if bins == 0 || n == 0 {
            return Vec::new();
        }
        (0..bins)
            .map(|b| {
                let a = b * n / bins;
                let z = (((b + 1) * n / bins).max(a + 1)).min(n);
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for &x in &self.samples[a..z] {
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
                let s0 = (a / HOP).min(self.slots.len() - 1);
                let s1 = ((z - 1) / HOP).min(self.slots.len() - 1).max(s0);
                let bright = self.slots[s0..=s1].iter().map(|s| s.bright).sum::<f32>()
                    / (s1 - s0 + 1) as f32;
                Overview {
                    min: lo,
                    max: hi,
                    bright,
                }
            })
            .collect()
    }

    /// Tournament selection. Draws `k` random slots from a window around
    /// `center` (0..1 across the file, `spread` is the window width) and
    /// returns the one closest to the target descriptors.
    /// k = 1 is plain random granulation. Large k locks onto the target.
    pub fn choose(
        &self,
        center: f64,
        spread: f64,
        k: u32,
        target_bright: f32,
        target_loud: f32,
        rng: &mut Rng,
    ) -> usize {
        let n = self.slots.len();
        let half = (spread.clamp(0.0, 1.0) * 0.5 * n as f64).max(1.0);
        let c = center.clamp(0.0, 1.0) * n as f64;
        let lo = ((c - half).floor().max(0.0) as usize).min(n - 1);
        let hi = (((c + half).ceil()) as usize).min(n).max(lo + 1);
        let mut best = lo;
        let mut best_cost = f32::MAX;
        for _ in 0..k.max(1) {
            let s = lo + rng.below(hi - lo);
            let sl = self.slots[s];
            let db = sl.bright - target_bright;
            let dl = sl.loud - target_loud;
            let cost = db * db + 0.5 * dl * dl;
            if cost < best_cost {
                best_cost = cost;
                best = s;
            }
        }
        best
    }

    /// A small procedural corpus so the synth makes sound before any file is
    /// imported. It sweeps from dark to bright over time, so the Position
    /// and Tone controls have something meaningful to select between.
    pub fn builtin() -> Self {
        let sr = 44100.0f64;
        let secs = 4.0;
        let n = (sr * secs) as usize;
        let mut out = Vec::with_capacity(n);
        let mut rng = Rng::new(12345);
        let f0 = 110.0;
        let mut phase = 0.0f64;
        let mut lp = 0.0f64;
        for i in 0..n {
            let u = i as f64 / n as f64;
            let limit = 1.0 + 22.0 * u * u;
            phase += f0 * (1.0 + 0.003 * (TAU * 0.7 * u * secs).sin()) / sr;
            if phase >= 1.0 {
                phase -= 1.0;
            }
            let mut s = 0.0;
            let kmax = limit.ceil() as usize;
            for k in 1..=kmax {
                let a = (limit - (k as f64 - 1.0)).clamp(0.0, 1.0) / k as f64;
                s += a * (TAU * phase * k as f64).sin();
            }
            // breathy noise that grows toward the end
            lp += 0.2 * (rng.bipolar() - lp);
            let pulse = 0.5 + 0.5 * (TAU * 2.0 * u * secs).sin();
            s += (rng.bipolar() - 0.6 * lp) * 0.35 * u * pulse;
            out.push(s as f32);
        }
        Corpus::from_mono("Built-in", out, sr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(f: f64, sr: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (TAU * f * i as f64 / sr).sin() as f32)
            .collect()
    }

    #[test]
    fn brightness_is_monotonic_in_frequency() {
        let sr = 44100.0;
        let lo = Corpus::from_mono("lo", sine(200.0, sr, 4096), sr);
        let mid = Corpus::from_mono("mid", sine(2000.0, sr, 4096), sr);
        let hi = Corpus::from_mono("hi", sine(8000.0, sr, 4096), sr);
        let b = |c: &Corpus| c.slots[2].bright;
        assert!(
            b(&lo) < b(&mid) && b(&mid) < b(&hi),
            "{} {} {}",
            b(&lo),
            b(&mid),
            b(&hi)
        );
    }

    #[test]
    fn choose_follows_target() {
        let sr = 44100.0;
        let mut s = sine(200.0, sr, HOP * 40);
        s.extend(sine(8000.0, sr, HOP * 40));
        let c = Corpus::from_mono("two", s, sr);
        let mut rng = Rng::new(1);
        let mut hi_hits = 0;
        for _ in 0..200 {
            let slot = c.choose(0.5, 1.0, 8, 0.9, 1.0, &mut rng);
            if slot >= 40 {
                hi_hits += 1;
            }
        }
        assert!(hi_hits > 190, "only {hi_hits} bright picks");
        let mut lo_hits = 0;
        for _ in 0..200 {
            if c.choose(0.5, 1.0, 8, 0.05, 1.0, &mut rng) < 40 {
                lo_hits += 1;
            }
        }
        assert!(lo_hits > 190, "only {lo_hits} dark picks");
    }

    #[test]
    fn choose_respects_window() {
        let sr = 44100.0;
        let c = Corpus::from_mono("n", sine(440.0, sr, HOP * 100), sr);
        let mut rng = Rng::new(3);
        for _ in 0..500 {
            let s = c.choose(0.2, 0.1, 1, 0.5, 0.5, &mut rng);
            assert!((12..=28).contains(&s), "slot {s} outside window");
        }
    }

    #[test]
    fn overview_covers_file() {
        let sr = 44100.0;
        let mut s = sine(200.0, sr, HOP * 20);
        s.extend(sine(8000.0, sr, HOP * 20));
        let c = Corpus::from_mono("o", s, sr);
        let o = c.overview(40);
        assert_eq!(o.len(), 40);
        assert!(o.iter().all(|c| c.min <= c.max));
        assert!(o[35].bright > o[5].bright);
        assert!(c.overview(0).is_empty());
        assert_eq!(c.overview(100_000).len(), 100_000);
    }

    #[test]
    fn builtin_has_range() {
        let c = Corpus::builtin();
        assert!(c.slots.first().unwrap().bright < c.slots.last().unwrap().bright);
        assert!(c.samples.iter().all(|s| s.is_finite()));
    }
}
