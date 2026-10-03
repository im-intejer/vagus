
//! The complete polysynth with no plugin framework dependency.
//!
//! The plugin layer in lib.rs only translates host parameters and MIDI into
//! calls on `Engine`, which keeps all of the DSP testable on its own.

use crate::corpus::Corpus;
use crate::fdn::{Fdn, FdnSettings};
use crate::grain::CloudSettings;
use crate::loader::Loader;
use crate::voice::{SvfCoeffs, Voice};
use std::f64::consts::TAU;
use std::sync::Arc;

pub const MAX_VOICES: usize = 16;
pub const PITCH_BEND_RANGE: f64 = 2.0;
pub const VIBRATO_RATE_HZ: f64 = 5.0;
pub const VIBRATO_DEPTH_SEMITONES: f64 = 0.5;
const VOICE_TRIM: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct EngineSettings {
    pub cutoff: f64,
    pub resonance: f64,
    /// Linear gain.
    pub volume: f64,
    /// MIDI note at which the file plays at its original pitch.
    pub root: f64,
    /// 0 keeps grains at original pitch, 1 tracks the keyboard.
    pub follow: f64,
    pub attack: f64,
    pub decay: f64,
    pub sustain: f64,
    pub release: f64,
    pub cloud: CloudSettings,
    pub fdn: FdnSettings,
}

impl Default for EngineSettings {
    fn default() -> Self {
        EngineSettings {
            cutoff: 8000.0,
            resonance: 0.0,
            volume: 0.5,
            root: 60.0,
            follow: 1.0,
            attack: 0.01,
            decay: 0.1,
            sustain: 0.7,
            release: 0.3,
            cloud: CloudSettings::default(),
            fdn: FdnSettings::default(),
        }
    }
}

pub struct Engine {
    sr: f64,
    voices: Vec<Voice>,
    serial: u64,
    corpus: Arc<Corpus>,
    loader: Loader,
    last_slot: i64,
    fdn: Fdn,
    pedal: bool,
    bend_mult: f64,
    mod_wheel: f64,
    lfo_phase: f64,
}

impl Engine {
    pub fn new(sr: f64) -> Self {
        let builtin = Arc::new(Corpus::builtin());
        let loader = Loader::spawn(builtin.clone());
        Engine {
            sr,
            voices: (0..MAX_VOICES).map(|_| Voice::new()).collect(),
            serial: 0,
            corpus: builtin,
            loader,
            last_slot: 0,
            fdn: Fdn::new(sr),
            pedal: false,
            bend_mult: 1.0,
            mod_wheel: 0.0,
            lfo_phase: 0.0,
        }
    }

    /// Call from `reset`. Allocation is allowed there.
    pub fn reset(&mut self, sr: f64) {
        self.sr = sr;
        self.fdn = Fdn::new(sr);
        for v in self.voices.iter_mut() {
            v.active = false;
        }
        self.pedal = false;
        self.bend_mult = 1.0;
        self.mod_wheel = 0.0;
        self.lfo_phase = 0.0;
    }

    pub fn loader(&self) -> &Loader {
        &self.loader
    }

    pub fn corpus_name(&self) -> &str {
        &self.corpus.name
    }

    /// Call once per block. Picks up a freshly loaded corpus and forwards
    /// slot changes to the loader thread.
    pub fn begin_block(&mut self, slot: i64) {
        if slot != self.last_slot {
            self.last_slot = slot;
            self.loader.request_slot(slot);
        }
        self.loader.poll(&mut self.corpus);
    }

    pub fn is_idle(&self) -> bool {
        !self.voices.iter().any(|v| v.active) && self.fdn.is_silent()
    }

    pub fn active_voices(&self) -> usize {
        self.voices.iter().filter(|v| v.active).count()
    }

    pub fn set_pitch_bend(&mut self, normalized: f64) {
        self.bend_mult = (normalized * PITCH_BEND_RANGE / 12.0).exp2();
    }

    pub fn set_mod_wheel(&mut self, v: f64) {
        self.mod_wheel = v.clamp(0.0, 1.0);
    }

    fn alloc_voice(&self) -> usize {
        if let Some(i) = self.voices.iter().position(|v| !v.active) {
            return i;
        }
        // Prefer the oldest releasing voice, then the oldest voice overall.
        let oldest = |releasing_only: bool| {
            self.voices
                .iter()
                .enumerate()
                .filter(|(_, v)| !releasing_only || v.releasing)
                .min_by_key(|(_, v)| v.serial)
                .map(|(i, _)| i)
        };
        oldest(true).or_else(|| oldest(false)).unwrap_or(0)
    }

    pub fn note_on(&mut self, s: &EngineSettings, note: u8, velocity: f64) {
        let idx = self.alloc_voice();
        self.serial += 1;
        let base_ratio = (((note as f64 - s.root) * s.follow) / 12.0).exp2();
        let serial = self.serial;
        let sr = self.sr;
        self.voices[idx].start(note, velocity, serial, base_ratio, (s.attack, s.decay, s.sustain), sr);
    }

    pub fn note_off(&mut self, s: &EngineSettings, note: u8) {
        let (pedal, sr) = (self.pedal, self.sr);
        for v in self.voices.iter_mut().filter(|v| v.active && v.note == note && v.key_down) {
            v.key_down = false;
            if !pedal {
                v.release(s.release, sr);
            }
        }
    }

    pub fn sustain_pedal(&mut self, s: &EngineSettings, down: bool) {
        self.pedal = down;
        if !down {
            let sr = self.sr;
            for v in self.voices.iter_mut().filter(|v| v.active && !v.key_down && !v.releasing) {
                v.release(s.release, sr);
            }
        }
    }

    pub fn all_notes_off(&mut self, s: &EngineSettings) {
        let sr = self.sr;
        for v in self.voices.iter_mut().filter(|v| v.active && !v.releasing) {
            v.key_down = false;
            v.release(s.release, sr);
        }
    }

    /// Render one stereo frame.
    pub fn process_frame(&mut self, s: &EngineSettings) -> (f64, f64) {
        let vib = self.mod_wheel * VIBRATO_DEPTH_SEMITONES * (self.lfo_phase * TAU).sin();
        self.lfo_phase += VIBRATO_RATE_HZ / self.sr;
        if self.lfo_phase >= 1.0 {
            self.lfo_phase -= 1.0;
        }
        // Bend and vibrato are scaled by Key Follow once, not once per voice.
        let pitch_mod = (self.bend_mult * (vib / 12.0).exp2()).powf(s.follow);
        let flt = SvfCoeffs::new(s.cutoff, s.resonance, self.sr);

        let corpus: &Corpus = &self.corpus;
        let (mut l, mut r) = (0.0, 0.0);
        for v in self.voices.iter_mut().filter(|v| v.active) {
            let (a, b) = v.render(corpus, &s.cloud, &flt, pitch_mod, self.sr);
            l += a;
            r += b;
        }
        l *= VOICE_TRIM;
        r *= VOICE_TRIM;

        let (wl, wr) = self.fdn.tick(0.5 * (l + r), &s.fdn);
        let mix = s.fdn.mix.clamp(0.0, 1.0);
        let ol = (l * (1.0 - mix) + wl * mix) * s.volume;
        let or = (r * (1.0 - mix) + wr * mix) * s.volume;
        (ol.tanh(), or.tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;

    fn render(e: &mut Engine, s: &EngineSettings, n: usize) -> Vec<(f64, f64)> {
        (0..n).map(|_| e.process_frame(s)).collect()
    }

    fn rms(x: &[(f64, f64)]) -> f64 {
        (x.iter().map(|(l, r)| l * l + r * r).sum::<f64>() / (2.0 * x.len() as f64)).sqrt()
    }

    #[test]
    fn silent_without_notes() {
        let mut e = Engine::new(SR);
        let s = EngineSettings::default();
        e.begin_block(0);
        let out = render(&mut e, &s, 4096);
        assert!(rms(&out) < 1e-9);
        assert!(e.is_idle());
    }

    #[test]
    fn note_makes_sound_then_releases_to_idle() {
        let mut e = Engine::new(SR);
        let mut s = EngineSettings::default();
        s.fdn.mix = 0.0;
        s.release = 0.05;
        e.begin_block(0);
        e.note_on(&s, 60, 0.9);
        let on = render(&mut e, &s, 8000);
        assert!(rms(&on) > 1e-3, "rms {}", rms(&on));
        assert!(on.iter().all(|(l, r)| l.is_finite() && r.is_finite()));
        e.note_off(&s, 60);
        render(&mut e, &s, 44100);
        assert_eq!(e.active_voices(), 0);
        // The FDN keeps listening to the voice for a few seconds.
        render(&mut e, &s, 44100 * 8);
        assert!(e.is_idle());
    }

    #[test]
    fn polyphony_and_voice_stealing() {
        let mut e = Engine::new(SR);
        let s = EngineSettings::default();
        for n in 40..70u8 {
            e.note_on(&s, n, 0.8);
            render(&mut e, &s, 64);
        }
        assert_eq!(e.active_voices(), MAX_VOICES);
        // Newest notes survive, oldest were stolen.
        let alive: Vec<u8> = e.voices.iter().filter(|v| v.active).map(|v| v.note).collect();
        assert!(alive.contains(&69) && !alive.contains(&40));
        let out = render(&mut e, &s, 4096);
        assert!(out.iter().all(|(l, r)| l.abs() <= 1.0 && r.abs() <= 1.0));
    }

    #[test]
    fn sustain_pedal_holds_notes() {
        let mut e = Engine::new(SR);
        let mut s = EngineSettings::default();
        s.release = 0.02;
        e.note_on(&s, 60, 0.8);
        e.sustain_pedal(&s, true);
        e.note_off(&s, 60);
        render(&mut e, &s, 22050);
        assert_eq!(e.active_voices(), 1, "pedal should hold the note");
        e.sustain_pedal(&s, false);
        render(&mut e, &s, 22050);
        assert_eq!(e.active_voices(), 0, "release after pedal up");
    }

    fn sine_engine() -> Engine {
        let mut e = Engine::new(SR);
        let sine: Vec<f32> = (0..44100 * 2)
            .map(|i| (TAU * 200.0 * i as f64 / SR).sin() as f32)
            .collect();
        e.corpus = Arc::new(Corpus::from_mono("sine", sine, SR));
        e
    }

    fn crossings(e: &mut Engine, s: &EngineSettings) -> f64 {
        let out = render(e, s, 44100);
        out.windows(2).filter(|w| (w[0].0 >= 0.0) != (w[1].0 >= 0.0)).count() as f64
    }

    fn quiet_settings() -> EngineSettings {
        let mut s = EngineSettings::default();
        s.fdn.mix = 0.0;
        s.cutoff = 20000.0;
        s.cloud.density = 2.0;
        s.cloud.detune_cents = 0.0;
        s.cloud.spread = 0.5;
        s
    }

    #[test]
    fn pitch_bend_and_follow() {
        let s = quiet_settings();
        let mut plain = sine_engine();
        plain.note_on(&s, 60, 0.9);
        let base = crossings(&mut plain, &s);

        let mut bent = sine_engine();
        bent.set_pitch_bend(1.0);
        bent.note_on(&s, 60, 0.9);
        let up = crossings(&mut bent, &s);
        assert!(up > base * 1.08, "bend up {up} vs {base}");

        let mut nofollow = EngineSettings { follow: 0.0, ..s.clone() };
        nofollow.follow = 0.0;
        let mut a = sine_engine();
        a.set_pitch_bend(1.0);
        a.note_on(&nofollow, 72, 0.9);
        let mut b = sine_engine();
        b.note_on(&nofollow, 48, 0.9);
        assert_eq!(crossings(&mut a, &nofollow), crossings(&mut b, &nofollow), "follow 0 ignores keys and bend");
    }

    #[test]
    fn octave_up_doubles_frequency() {
        let s = quiet_settings();
        let mut a = sine_engine();
        a.note_on(&s, 60, 0.9);
        let mut b = sine_engine();
        b.note_on(&s, 72, 0.9);
        let r = crossings(&mut b, &s) / crossings(&mut a, &s);
        assert!((1.7..2.3).contains(&r), "octave ratio {r}");
    }

    #[test]
    fn fdn_tail_outlives_the_note() {
        let mut e = Engine::new(SR);
        let mut s = EngineSettings::default();
        s.fdn.mix = 0.8;
        s.fdn.feedback = 0.85;
        s.release = 0.02;
        e.note_on(&s, 60, 0.9);
        render(&mut e, &s, 22050);
        e.note_off(&s, 60);
        render(&mut e, &s, 22050);
        assert_eq!(e.active_voices(), 0);
        let tail = render(&mut e, &s, 22050);
        assert!(rms(&tail) > 1e-3, "tail rms {}", rms(&tail));
        assert!(!e.is_idle());
    }

    #[test]
    fn imported_file_is_used_after_slot_change() {
        use crate::loader::tests_support::write_test_wav;
        let dir = std::env::temp_dir().join(format!("truce_grain_engine_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_test_wav(&dir.join("tone.wav"));
        std::env::set_var("TRUCE_GRAIN_DIR", &dir);

        let mut e = Engine::new(SR);
        let s = EngineSettings::default();
        e.begin_block(1);
        let t0 = std::time::Instant::now();
        while e.corpus_name() == "Built-in" && t0.elapsed().as_secs() < 5 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            e.begin_block(1);
        }
        assert_eq!(e.corpus_name(), "tone");
        e.note_on(&s, 60, 0.9);
        let out = render(&mut e, &s, 8000);
        assert!(rms(&out) > 1e-3);
    }
}
