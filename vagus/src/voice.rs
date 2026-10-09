//! Polyphonic voices, envelope and filter.

use crate::corpus::Corpus;
use crate::grain::{CloudSettings, GrainCloud};
use crate::util::Rng;
use std::f64::consts::PI;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Stage {
    Attack,
    Decay,
    Sustain,
    Release,
    Done,
}

pub struct Adsr {
    stage: Stage,
    level: f64,
    attack_inc: f64,
    decay_coef: f64,
    sustain: f64,
    release_coef: f64,
}

impl Adsr {
    fn new() -> Self {
        Adsr {
            stage: Stage::Done,
            level: 0.0,
            attack_inc: 0.01,
            decay_coef: 0.001,
            sustain: 0.7,
            release_coef: 0.001,
        }
    }

    /// Starts from the current level so a stolen voice does not click.
    fn start(&mut self, a: f64, d: f64, s: f64, sr: f64) {
        self.attack_inc = 1.0 / (a.max(0.001) * sr);
        self.decay_coef = 1.0 - (-4.6 / (d.max(0.001) * sr)).exp();
        self.sustain = s.clamp(0.0, 1.0);
        self.stage = Stage::Attack;
    }

    fn release(&mut self, r: f64, sr: f64) {
        self.release_coef = 1.0 - (-4.6 / (r.max(0.005) * sr)).exp();
        if self.stage != Stage::Done {
            self.stage = Stage::Release;
        }
    }

    #[inline]
    fn tick(&mut self) -> f64 {
        match self.stage {
            Stage::Attack => {
                self.level += self.attack_inc;
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.stage = Stage::Decay;
                }
            }
            Stage::Decay => {
                self.level += (self.sustain - self.level) * self.decay_coef;
                if (self.level - self.sustain).abs() < 1e-4 {
                    self.stage = Stage::Sustain;
                }
            }
            Stage::Sustain => {
                self.level = self.sustain;
                if self.level < 1e-5 {
                    self.stage = Stage::Done;
                }
            }
            Stage::Release => {
                self.level -= self.level * self.release_coef;
                if self.level < 1e-5 {
                    self.level = 0.0;
                    self.stage = Stage::Done;
                }
            }
            Stage::Done => self.level = 0.0,
        }
        self.level
    }

    fn is_done(&self) -> bool {
        self.stage == Stage::Done
    }
}

/// Coefficients for a TPT state variable filter, computed once per sample and
/// shared by every voice because the cutoff is global.
#[derive(Clone, Copy)]
pub struct SvfCoeffs {
    a1: f64,
    a2: f64,
    a3: f64,
}

impl SvfCoeffs {
    pub fn new(cutoff: f64, resonance: f64, sr: f64) -> Self {
        let fc = cutoff.clamp(20.0, sr * 0.45);
        let g = (PI * fc / sr).tan();
        let k = 2.0 * (1.0 - 0.97 * resonance.clamp(0.0, 1.0));
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        SvfCoeffs { a1, a2, a3 }
    }
}

#[derive(Clone, Copy, Default)]
struct SvfState {
    ic1: f64,
    ic2: f64,
}

impl SvfState {
    #[inline]
    fn lowpass(&mut self, x: f64, c: &SvfCoeffs) -> f64 {
        let v3 = x - self.ic2;
        let v1 = c.a1 * self.ic1 + c.a2 * v3;
        let v2 = self.ic2 + c.a2 * self.ic1 + c.a3 * v3;
        self.ic1 = 2.0 * v1 - self.ic1;
        self.ic2 = 2.0 * v2 - self.ic2;
        v2
    }
}

pub struct Voice {
    pub active: bool,
    pub note: u8,
    /// True while the key is physically held.
    pub key_down: bool,
    pub releasing: bool,
    pub serial: u64,
    velocity: f64,
    gain: f64,
    base_ratio: f64,
    env: Adsr,
    cloud: GrainCloud,
    flt: [SvfState; 2],
    rng: Rng,
}

impl Voice {
    pub fn new() -> Self {
        Voice {
            active: false,
            note: 0,
            key_down: false,
            releasing: false,
            serial: 0,
            velocity: 0.0,
            gain: 0.0,
            base_ratio: 1.0,
            env: Adsr::new(),
            cloud: GrainCloud::new(),
            flt: [SvfState::default(); 2],
            rng: Rng::new(1),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &mut self,
        note: u8,
        velocity: f64,
        serial: u64,
        base_ratio: f64,
        (a, d, s): (f64, f64, f64),
        sr: f64,
    ) {
        let stolen = self.active;
        self.active = true;
        self.note = note;
        self.key_down = true;
        self.releasing = false;
        self.serial = serial;
        self.velocity = velocity.clamp(0.0, 1.0);
        self.gain = 0.15 + 0.85 * self.velocity;
        self.base_ratio = base_ratio;
        self.rng = Rng::new(serial.wrapping_mul(2654435761));
        if !stolen {
            self.env = Adsr::new();
            self.cloud.reset();
            self.flt = [SvfState::default(); 2];
        }
        self.env.start(a, d, s, sr);
    }

    pub fn release(&mut self, r: f64, sr: f64) {
        self.releasing = true;
        self.env.release(r, sr);
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn render(
        &mut self,
        corpus: &Corpus,
        cs: &CloudSettings,
        flt: &SvfCoeffs,
        pitch_mod: f64,
        sr: f64,
        beat: f64,
    ) -> (f64, f64) {
        let e = self.env.tick();
        if self.env.is_done() {
            self.active = false;
            return (0.0, 0.0);
        }
        let ratio = self.base_ratio * pitch_mod;
        let (l, r) = self
            .cloud
            .render(corpus, cs, ratio, self.velocity, sr, beat, &mut self.rng);
        let g = e * self.gain;
        (
            self.flt[0].lowpass(l * g, flt),
            self.flt[1].lowpass(r * g, flt),
        )
    }
}

impl Default for Voice {
    fn default() -> Self {
        Self::new()
    }
}
