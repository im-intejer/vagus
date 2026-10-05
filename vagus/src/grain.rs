//! Per-voice grain cloud that reads from an imported (or built-in) corpus.

use crate::corpus::{Corpus, HOP};
use crate::util::{Rng, hann};
use std::f64::consts::FRAC_PI_4;

pub const MAX_GRAINS: usize = 24;

#[derive(Clone, Copy, Default)]
struct Grain {
    pos: f64,
    inc: f64,
    age: u32,
    len: u32,
    gl: f64,
    gr: f64,
    on: bool,
}

/// Settings shared by every voice's cloud.
#[derive(Clone, Copy, Debug)]
pub struct CloudSettings {
    /// Grain length in seconds.
    pub grain_s: f64,
    /// Average number of overlapping grains.
    pub density: f64,
    /// Scan position in the file, 0..1.
    pub position: f64,
    /// Width of the window grains are drawn from, 0..1.
    pub spread: f64,
    /// Random pitch scatter per grain in cents.
    pub detune_cents: f64,
    /// Stereo scatter, 0..1.
    pub width: f64,
    /// Brightness target for descriptor selection, 0..1.
    pub tone: f64,
    /// Tournament size. 1 is random, higher locks onto the targets.
    pub focus: u32,
    /// Randomness of grain onset timing, 0..1. 0 is perfectly regular,
    /// 1 is fully random (exponentially distributed gaps, like rain).
    pub timing_jitter: f64,
}

impl Default for CloudSettings {
    fn default() -> Self {
        CloudSettings {
            grain_s: 0.08,
            density: 3.0,
            position: 0.3,
            spread: 0.2,
            detune_cents: 8.0,
            width: 0.5,
            tone: 0.5,
            focus: 4,
            timing_jitter: 0.35,
        }
    }
}

/// Samples until the next grain onset.
///
/// The mean gap is always `grain_len / density`, so Density keeps its meaning
/// at every jitter setting. The jitter blends a fixed gap (0) with an
/// exponentially distributed gap (1), which is what a Poisson process, the
/// statistics of rain, produces. Gaps are capped at 6x the mean so one unlucky
/// draw cannot leave a long hole.
pub fn next_interval(grain_len: f64, density: f64, jitter: f64, rng: &mut Rng) -> f64 {
    let mean = grain_len / density.max(0.1);
    let j = jitter.clamp(0.0, 1.0);
    if j <= 0.0 {
        return mean;
    }
    let expo = (-(1.0 - rng.unit()).ln()).min(6.0);
    mean * ((1.0 - j) + j * expo)
}

pub struct GrainCloud {
    grains: [Grain; MAX_GRAINS],
    countdown: f64,
}

impl Default for GrainCloud {
    fn default() -> Self {
        Self::new()
    }
}

impl GrainCloud {
    pub fn new() -> Self {
        GrainCloud {
            grains: [Grain::default(); MAX_GRAINS],
            countdown: 0.0,
        }
    }

    pub fn reset(&mut self) {
        for g in self.grains.iter_mut() {
            g.on = false;
        }
        self.countdown = 0.0;
    }

    /// Render one stereo sample.
    /// `ratio` is the playback ratio relative to the file's original pitch.
    pub fn render(
        &mut self,
        corpus: &Corpus,
        s: &CloudSettings,
        ratio: f64,
        target_loud: f64,
        sr: f64,
        rng: &mut Rng,
    ) -> (f64, f64) {
        self.countdown -= 1.0;
        if self.countdown <= 0.0 {
            let len = (s.grain_s * sr).max(32.0);
            self.countdown += next_interval(len, s.density, s.timing_jitter, rng);
            if !corpus.is_empty() {
                self.spawn(corpus, s, ratio, target_loud, sr, len as u32, rng);
            }
        }

        let n = corpus.len() as f64;
        let (mut l, mut r) = (0.0, 0.0);
        for g in self.grains.iter_mut() {
            if !g.on {
                continue;
            }
            if g.age >= g.len || g.pos < 0.0 || g.pos >= n {
                g.on = false;
                continue;
            }
            let w = hann(g.age as f64 / g.len as f64);
            let x = corpus.read(g.pos) * w;
            l += x * g.gl;
            r += x * g.gr;
            g.pos += g.inc;
            g.age += 1;
        }
        (l, r)
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn(
        &mut self,
        corpus: &Corpus,
        s: &CloudSettings,
        ratio: f64,
        target_loud: f64,
        sr: f64,
        len: u32,
        rng: &mut Rng,
    ) {
        let Some(slot) = self.grains.iter_mut().find(|g| !g.on) else {
            return;
        };
        let idx = corpus.choose(
            s.position,
            s.spread,
            s.focus,
            s.tone as f32,
            target_loud as f32,
            rng,
        );
        let start = (idx * HOP + rng.below(HOP)) as f64;
        let detune = (rng.bipolar() * s.detune_cents / 1200.0).exp2();
        let pan = rng.bipolar() * s.width;
        let angle = (pan + 1.0) * FRAC_PI_4;
        let gain = 1.0 / s.density.max(1.0).sqrt();
        *slot = Grain {
            pos: start,
            inc: ratio * detune * corpus.sample_rate / sr,
            age: 0,
            len,
            gl: angle.cos() * gain,
            gr: angle.sin() * gain,
            on: true,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_makes_bounded_finite_sound() {
        let corpus = Corpus::builtin();
        let mut cloud = GrainCloud::new();
        let mut rng = Rng::new(9);
        let s = CloudSettings::default();
        let mut peak = 0.0f64;
        let mut energy = 0.0f64;
        for _ in 0..44100 {
            let (l, r) = cloud.render(&corpus, &s, 1.0, 0.8, 44100.0, &mut rng);
            assert!(l.is_finite() && r.is_finite());
            peak = peak.max(l.abs()).max(r.abs());
            energy += l * l + r * r;
        }
        assert!(energy > 1.0, "silent cloud");
        assert!(peak < 3.0, "peak {peak}");
    }

    #[test]
    fn pitch_ratio_changes_content() {
        let sr = 44100.0;
        let sine: Vec<f32> = (0..44100 * 2)
            .map(|i| (std::f64::consts::TAU * 200.0 * i as f64 / sr).sin() as f32)
            .collect();
        let corpus = Corpus::from_mono("sine", sine, sr);
        let run = |ratio: f64| {
            let mut cloud = GrainCloud::new();
            let mut rng = Rng::new(4);
            let s = CloudSettings {
                detune_cents: 0.0,
                density: 2.0,
                spread: 0.5,
                ..Default::default()
            };
            let mut prev = 0.0;
            let mut crossings = 0;
            for _ in 0..44100 {
                let (l, _) = cloud.render(&corpus, &s, ratio, 0.8, sr, &mut rng);
                if (l >= 0.0) != (prev >= 0.0) {
                    crossings += 1;
                }
                prev = l;
            }
            crossings as f64
        };
        let r = run(2.0) / run(1.0);
        assert!((1.7..2.3).contains(&r), "crossing ratio {r}");
    }

    #[test]
    fn zero_jitter_is_perfectly_regular() {
        let mut rng = Rng::new(5);
        for _ in 0..100 {
            assert_eq!(next_interval(4410.0, 3.0, 0.0, &mut rng), 1470.0);
        }
    }

    #[test]
    fn jitter_keeps_mean_and_widens_spread() {
        let stats = |j: f64| {
            let mut rng = Rng::new(11);
            let v: Vec<f64> = (0..50_000)
                .map(|_| next_interval(4410.0, 3.0, j, &mut rng))
                .collect();
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64;
            assert!(v.iter().all(|x| *x >= 0.0 && x.is_finite()));
            (mean, var.sqrt() / mean)
        };
        let (m0, _) = stats(0.0);
        let (m5, cv5) = stats(0.5);
        let (m1, cv1) = stats(1.0);
        for m in [m0, m5, m1] {
            assert!((m / 1470.0 - 1.0).abs() < 0.03, "mean drifted to {m}");
        }
        assert!(
            cv5 > 0.3 && cv1 > cv5,
            "spread should grow, cv {cv5} then {cv1}"
        );
    }

    #[test]
    fn regular_timing_produces_steady_pulse() {
        // Density 1 with zero jitter means one grain per grain length, so the
        // output energy repeats with that period.
        let sr = 44100.0;
        let sine: Vec<f32> = (0..44100 * 2)
            .map(|i| (std::f64::consts::TAU * 200.0 * i as f64 / sr).sin() as f32)
            .collect();
        let corpus = Corpus::from_mono("sine", sine, sr);
        let env = |jitter: f64| {
            let mut cloud = GrainCloud::new();
            let mut rng = Rng::new(2);
            let s = CloudSettings {
                grain_s: 0.05,
                density: 1.0,
                timing_jitter: jitter,
                detune_cents: 0.0,
                spread: 0.5,
                ..Default::default()
            };
            // energy per 5 ms block
            let mut blocks = Vec::new();
            let mut e = 0.0;
            for i in 0..44100 {
                let (l, r) = cloud.render(&corpus, &s, 1.0, 0.8, sr, &mut rng);
                e += l * l + r * r;
                if (i + 1) % 220 == 0 {
                    blocks.push(e);
                    e = 0.0;
                }
            }
            let mean = blocks.iter().sum::<f64>() / blocks.len() as f64;
            blocks
                .iter()
                .map(|b| (b - mean) * (b - mean))
                .sum::<f64>()
                .sqrt()
                / mean
        };
        // Both are bursty at density 1, but random timing is clearly more uneven.
        assert!(
            env(1.0) > env(0.0) * 1.1,
            "random {} regular {}",
            env(1.0),
            env(0.0)
        );
    }

    #[test]
    fn empty_corpus_is_safe() {
        let corpus = Corpus::from_mono("empty", vec![], 44100.0);
        let mut cloud = GrainCloud::new();
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            let (l, r) = cloud.render(
                &corpus,
                &CloudSettings::default(),
                1.0,
                0.5,
                44100.0,
                &mut rng,
            );
            assert_eq!((l, r), (0.0, 0.0));
        }
    }
}
