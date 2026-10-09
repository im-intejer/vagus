//! Per-voice grain cloud that reads from an imported (or built-in) corpus.

use truce::params::FloatParamReadF64;
use truce::prelude64::ParamEnum;

use crate::corpus::{Corpus, HOP};
use crate::engine::Settings;
use crate::util::{Rng, hann};
use std::f64::consts::FRAC_PI_4;

pub const MAX_GRAINS: usize = 24;

/// Tempo used when the host reports none, or something unusable.
pub const FALLBACK_TEMPO_BPM: f64 = 120.0;

/// Grain size note values. `Free` uses the Grain Size knob in milliseconds,
/// every other choice sets the grain length from the host tempo. Grains
/// always start when the note is played, only their length is synced.
#[derive(ParamEnum)]
pub enum SizeSync {
    Free,
    #[name = "1/32"]
    Div32,
    #[name = "1/16T"]
    Div16T,
    #[name = "1/16"]
    Div16,
    #[name = "1/16D"]
    Div16D,
    #[name = "1/8T"]
    Div8T,
    #[name = "1/8"]
    Div8,
    #[name = "1/8D"]
    Div8D,
    #[name = "1/4T"]
    Div4T,
    #[name = "1/4"]
    Div4,
    #[name = "1/4D"]
    Div4D,
    #[name = "1/2"]
    Div2,
    #[name = "1 bar"]
    Bar1,
}

impl SizeSync {
    /// Length of the note value in quarter note beats, or `None` for Free.
    pub fn beats(&self) -> Option<f64> {
        Some(match self {
            SizeSync::Free => return None,
            SizeSync::Div32 => 0.125,
            SizeSync::Div16T => 1.0 / 6.0,
            SizeSync::Div16 => 0.25,
            SizeSync::Div16D => 0.375,
            SizeSync::Div8T => 1.0 / 3.0,
            SizeSync::Div8 => 0.5,
            SizeSync::Div8D => 0.75,
            SizeSync::Div4T => 2.0 / 3.0,
            SizeSync::Div4 => 1.0,
            SizeSync::Div4D => 1.5,
            SizeSync::Div2 => 2.0,
            SizeSync::Bar1 => 4.0,
        })
    }
}

/// Host tempo made safe to divide by.
pub fn usable_tempo(bpm: f64) -> f64 {
    if bpm.is_finite() && bpm >= 1.0 {
        bpm
    } else {
        FALLBACK_TEMPO_BPM
    }
}

/// Grain length in seconds for a note value at a tempo.
pub fn synced_length_s(beats: f64, bpm: f64) -> f64 {
    (beats * 60.0 / usable_tempo(bpm)).clamp(0.005, 4.0)
}

#[derive(Clone, Copy, Default)]
struct Grain {
    position: f64,
    increment: f64,
    age: u32,
    length: u32,
    gain_left: f64,
    gain_right: f64,
    is_on: bool,
}

/// Settings shared by every voice's cloud.
#[derive(Clone, Copy, Debug)]
pub struct CloudSettings {
    /// Grain length in seconds.
    pub grain_length_s: f64,
    /// Average number of overlapping grains.
    pub density: f64,
    /// Scan position in the file, 0..1.
    pub start_position_percent: f64,
    /// Width of the window grains are drawn from, 0..1.
    pub start_position_spread: f64,
    /// Random pitch scatter per grain in cents.
    pub detune_cents: f64,
    /// Stereo scatter, 0..1.
    pub stereo_width: f64,
    /// Brightness target for descriptor selection, 0..1.
    pub brightness: f64,
    /// Tournament size. 1 is random, higher locks onto the targets.
    pub focus: u32,
    /// Randomness of grain onset timing, 0..1. 0 is perfectly regular,
    /// 1 is fully random (exponentially distributed gaps, like rain).
    pub timing_jitter: f64,
}

impl Default for CloudSettings {
    fn default() -> Self {
        CloudSettings {
            grain_length_s: 0.08,
            density: 3.0,
            start_position_percent: 0.3,
            start_position_spread: 0.2,
            detune_cents: 8.0,
            stereo_width: 0.5,
            brightness: 0.5,
            focus: 4,
            timing_jitter: 0.35,
        }
    }
}

impl Settings for CloudSettings {
    fn update_block_settings(
        &mut self,
        p: &crate::SynthParams,
        transport: &truce::prelude::TransportInfo,
    ) {
        self.start_position_spread = p.grains.spread.value();
        self.grain_length_s = match p.grains.size_sync.value().beats() {
            Some(beats) => synced_length_s(beats, transport.tempo),
            None => p.grains.size.value() * 0.001,
        };
        self.density = p.grains.density.value();
        self.timing_jitter = p.grains.jitter.value();
        self.detune_cents = p.grains.detune.value();
        self.stereo_width = p.grains.width.value();
        self.focus = p.grains.focus.value().round().max(1.0) as u32;
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
            g.is_on = false;
        }
        self.countdown = 0.0;
    }

    /// Render one stereo sample.
    /// `ratio` is the playback ratio relative to the file's original pitch.
    pub fn render(
        &mut self,
        corpus: &Corpus,
        cloud_settings: &CloudSettings,
        ratio: f64,
        target_loud: f64,
        sample_rate: f64,
        rng: &mut Rng,
    ) -> (f64, f64) {
        self.countdown -= 1.0;
        if self.countdown <= 0.0 {
            let len = (cloud_settings.grain_length_s * sample_rate).max(32.0);
            self.countdown += next_interval(
                len,
                cloud_settings.density,
                cloud_settings.timing_jitter,
                rng,
            );
            if !corpus.is_empty() {
                self.spawn(
                    corpus,
                    cloud_settings,
                    ratio,
                    target_loud,
                    sample_rate,
                    len as u32,
                    rng,
                );
            }
        }

        let n = corpus.len() as f64;
        let (mut l, mut r) = (0.0, 0.0);
        for g in self.grains.iter_mut() {
            if !g.is_on {
                continue;
            }
            if g.age >= g.length || g.position < 0.0 || g.position >= n {
                g.is_on = false;
                continue;
            }
            let w = hann(g.age as f64 / g.length as f64);
            let x = corpus.read(g.position) * w;
            l += x * g.gain_left;
            r += x * g.gain_right;
            g.position += g.increment;
            g.age += 1;
        }
        (l, r)
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn(
        &mut self,
        corpus: &Corpus,
        cloud_settings: &CloudSettings,
        ratio: f64,
        target_loudness: f64,
        engine_sample_rate: f64,
        length: u32,
        rng: &mut Rng,
    ) {
        let Some(slot) = self.grains.iter_mut().find(|g| !g.is_on) else {
            return;
        };
        let idx = corpus.choose(
            cloud_settings.start_position_percent,
            cloud_settings.start_position_spread,
            cloud_settings.focus,
            cloud_settings.brightness as f32,
            target_loudness as f32,
            rng,
        );
        let start = (idx * HOP + rng.below(HOP)) as f64;
        let detune = (rng.bipolar() * cloud_settings.detune_cents / 1200.0).exp2();
        let pan = rng.bipolar() * cloud_settings.stereo_width;
        let angle = (pan + 1.0) * FRAC_PI_4;
        let gain = 1.0 / cloud_settings.density.max(1.0).sqrt();
        *slot = Grain {
            position: start,
            increment: ratio * detune * corpus.sample_rate / engine_sample_rate,
            age: 0,
            length,
            gain_left: angle.cos() * gain,
            gain_right: angle.sin() * gain,
            is_on: true,
        };
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::TEST_SAMPLE_RATE;

    use super::*;

    #[test]
    fn cloud_makes_bounded_finite_sound() {
        let corpus = Corpus::builtin(TEST_SAMPLE_RATE);
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
                start_position_spread: 0.5,
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
                grain_length_s: 0.05,
                density: 1.0,
                timing_jitter: jitter,
                detune_cents: 0.0,
                start_position_spread: 0.5,
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
    #[test]
    fn note_values_have_the_right_length() {
        assert_eq!(SizeSync::Free.beats(), None);
        assert_eq!(SizeSync::Div4.beats(), Some(1.0));
        assert_eq!(SizeSync::Div8.beats(), Some(0.5));
        assert_eq!(SizeSync::Bar1.beats(), Some(4.0));
        // A dotted note is 1.5x, a triplet 2/3 of the plain value.
        assert_eq!(SizeSync::Div8D.beats(), Some(0.75));
        assert!((SizeSync::Div8T.beats().unwrap() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn synced_length_follows_tempo() {
        assert!((synced_length_s(1.0, 120.0) - 0.5).abs() < 1e-12);
        assert!((synced_length_s(1.0, 60.0) - 1.0).abs() < 1e-12);
        assert!((synced_length_s(0.5, 240.0) - 0.125).abs() < 1e-12);
        // Missing or broken tempo falls back to 120 BPM.
        assert_eq!(synced_length_s(1.0, 0.0), synced_length_s(1.0, 120.0));
        assert_eq!(synced_length_s(1.0, f64::NAN), synced_length_s(1.0, 120.0));
        assert!(synced_length_s(4.0, 1.0) <= 4.0);
    }

    /// Sample indices at which a grain starts.
    fn onsets(s: &CloudSettings, samples: usize) -> Vec<usize> {
        let sr = 44100.0;
        let corpus = Corpus::builtin(sr);
        let mut cloud = GrainCloud::new();
        let mut rng = Rng::new(3);
        let mut out = Vec::new();
        for i in 0..samples {
            cloud.render(&corpus, s, 1.0, 0.8, sr, &mut rng);
            if cloud.grains.iter().any(|g| g.is_on && g.age == 1) {
                out.push(i);
            }
        }
        out
    }

    #[test]
    fn synced_grains_start_immediately() {
        // 1/2 note at 120 BPM is one second, yet the first grain must start
        // on the very first sample, not at the next half note.
        let s = CloudSettings {
            grain_length_s: synced_length_s(2.0, 120.0),
            density: 2.0,
            timing_jitter: 0.0,
            ..Default::default()
        };
        let o = onsets(&s, 44100);
        assert_eq!(o.first(), Some(&0), "first grain waited until {o:?}");
    }

    #[test]
    fn synced_length_sets_the_grain_spacing() {
        // 1/8 at 120 BPM is 0.25 s. At density 2 and no jitter a grain
        // starts every 0.125 s.
        let s = CloudSettings {
            grain_length_s: synced_length_s(0.5, 120.0),
            density: 2.0,
            timing_jitter: 0.0,
            ..Default::default()
        };
        let o = onsets(&s, 44100);
        assert!(o.len() >= 7, "{} onsets", o.len());
        for w in o.windows(2) {
            assert!((w[1] as f64 - w[0] as f64 - 5512.5).abs() <= 1.0, "{w:?}");
        }
    }
}
