// Vagus is a wavelet convolution synth.
// White Noise -> Wavelet array -> Final output

// Synth runs ADSR + voice rendering in f64 for cumulative-state
// stability (phase accumulator, envelope coefficients); the f64
// prelude makes that the buffer precision too - the format wrapper
// widens the host's f32 audio buffer to f64 at the block boundary
// and narrows on the way out.
use std::f64::consts::TAU;

use std::sync::Arc;

use truce::core::midi::{norm_7bit, norm_pitch_bend};
use truce::prelude64::*;
use truce_gui::IntoLayoutEditor;
use truce_gui_types::layout::{GridLayout, dropdown, knob, section};

mod voice;
use voice::Voice;

// --- Waveform enum ---

#[derive(ParamEnum)]
pub enum Exciter {
    Harmonic,
    Chaotic,
}

#[derive(ParamEnum)]
pub enum Mesh {
    Nonlinear,
}

// --- Parameters ---

use VagusParamsParamId as P;

use crate::voice::PolyVoice;

#[derive(Params)]
pub struct VagusParams {
    #[param(name = "Algorithm", short_name = "Algorithm", default = 0)]
    pub algorithm: EnumParam<Exciter>,

    #[param(
        name = "Depth",
        short_name = "Depth",
        range = "linear(0, 64)",
        default = 0
    )]
    pub bin_count: IntParam,

    #[param(
        name = "Spectral Tilt",
        short_name = "SpecTilt",
        range = "linear(0, 360)",
        default = 0.0,
        unit = "deg",
        smooth = "exp(5)"
    )]
    pub spectral_tilt: FloatParam,

    #[param(name = "Damping", default = 0.5, smooth = "exp(5)")]
    pub damping: FloatParam,

    #[param(name = "Mesh", short_name = "Mesh", default = 0)]
    pub mesh: EnumParam<Mesh>,

    #[param(name = "Volume", short_name = "Vol",
            range = "linear(-60, 0)", default = -6.0,
            unit = "dB", smooth = "exp(5)")]
    pub volume: FloatParam,

    /// Pitch-bend target. VST3 has no native pitch-bend input event,
    /// so the host routes the pitch wheel to this parameter via
    /// `IMidiMapping`; the wrapper bridges the resulting change back
    /// into an `EventBody::PitchBend`. Hidden because it is a MIDI
    /// proxy, not a knob the user reaches for. AU / CLAP deliver pitch
    /// bend as raw MIDI and ignore this binding.
    #[param(
        name = "Pitch Bend",
        short_name = "Bend",
        range = "linear(-1, 1)",
        default = 0.0,
        flags = "hidden | automatable",
        midi_source = "pitchbend"
    )]
    pub bend: FloatParam,

    /// Mod-wheel (CC1) target, driving vibrato depth. Same story as
    /// `bend`: VST3 routes the wheel here via `IMidiMapping` and the
    /// wrapper bridges it back to an `EventBody::ControlChange`, while
    /// AU / CLAP deliver the CC as raw MIDI. Hidden MIDI proxy.
    #[param(
        name = "Mod Wheel",
        short_name = "Mod",
        range = "linear(0, 1)",
        default = 0.0,
        flags = "hidden | automatable",
        midi_cc = 1
    )]
    pub mod_wheel: FloatParam,
}

// --- Plugin ---
const MAX_VOICES: usize = 16;

/// Pitch-bend range in semitones at full deflection, matching the
/// MIDI default of +/-2 semitones.
const PITCH_BEND_RANGE: f64 = 2.0;

/// Vibrato LFO rate in Hz, and its depth in semitones at a fully
/// raised mod wheel.
const VIBRATO_RATE_HZ: f64 = 5.0;
const VIBRATO_DEPTH_SEMITONES: f64 = 0.5;

pub struct VagusDspState {
    poly_manager: PolyManager,
    sample_rate: f64,
    /// Channel pitch bend as a frequency multiplier. `1.0` is centered.
    pitch_bend_mult: f64,
    /// Mod-wheel position (CC1), `0.0..=1.0`, scaling vibrato depth.
    mod_wheel: f64,
    /// Vibrato LFO phase, `0.0..1.0`.
    lfo_phase: f64,
}

impl Default for VagusDspState {
    fn default() -> Self {
        let sample_rate = 44100.0;
        VagusDspState {
            // Pre-sized so `process` never grows the pool on the
            // audio thread.
            poly_manager: PolyManager::new(sample_rate),
            sample_rate,
            pitch_bend_mult: 1.0,
            mod_wheel: 0.0,
            lfo_phase: 0.0,
        }
    }
}

pub struct PolyManager {
    voices: Vec<PolyVoice>,
    time_counter: u64, // Increments every note_on to track relative age
}

impl PolyManager {
    fn new(sample_rate: f64) -> Self {
        Self {
            voices: vec![
                PolyVoice {
                    voice: Voice::new(sample_rate),
                    is_released: false,
                    midi_note: 0,
                    triggered_at: 0
                };
                MAX_VOICES
            ],
            time_counter: 0,
        }
    }

    pub fn note_on(&mut self, params: &VagusParams, note: u8, velocity: f32) {
        self.time_counter = self.time_counter.wrapping_add(1);

        let note_freq: f64 = midi_note_to_freq(note);
        let target_idx = self.find_voice_to_steal(note);

        // Hard-reset or gracefully fade the stolen voice, then trigger it
        let voice = &mut self.voices[target_idx];
        voice.voice.exciter.tilt = params.spectral_tilt.read();
        voice.voice.note_on(note_freq);
        voice.is_released = false;
        voice.midi_note = note;
        voice.triggered_at = self.time_counter;
    }

    pub fn note_off(&mut self, note: u8) {
        if let Some(v) = self.find_voice_with_note(note) {
            self.voices[v].voice.note_off();
        }
    }

    fn find_voice_to_steal(&self, new_note: u8) -> usize {
        // 1. Look for a completely free voice
        if let Some(idx) = self.voices.iter().position(|v| !v.voice.active) {
            return idx;
        }

        // 2. Look for a voice playing the exact same note
        if let Some(value) = self.find_voice_with_note(new_note) {
            return value;
        }

        // 3. Look for the oldest voice in the RELEASE phase
        let mut oldest_released_idx = None;
        let mut oldest_released_time = u64::MAX;

        for (i, v) in self.voices.iter().enumerate() {
            if v.is_released && v.triggered_at < oldest_released_time {
                oldest_released_time = v.triggered_at;
                oldest_released_idx = Some(i);
            }
        }

        if let Some(idx) = oldest_released_idx {
            return idx;
        }

        // 4. Fallback: Steal the absolute oldest active voice
        let mut oldest_idx = 0;
        let mut oldest_time = u64::MAX;

        for (i, v) in self.voices.iter().enumerate() {
            if v.triggered_at < oldest_time {
                oldest_time = v.triggered_at;
                oldest_idx = i;
            }
        }

        oldest_idx
    }

    fn find_voice_with_note(&self, note: u8) -> Option<usize> {
        if let Some(idx) = self.voices.iter().position(|v| v.midi_note == note) {
            return Some(idx);
        }
        None
    }

    pub fn none_playing(&self) -> bool {
        self.voices.iter().all(|v| !v.voice.active)
    }
}

impl VagusDspState {
    /// Map a 14-bit pitch-bend code to a frequency multiplier.
    fn pitch_bend(&mut self, value: u16) {
        let semitones = f64::from(norm_pitch_bend(value)) * PITCH_BEND_RANGE;
        self.pitch_bend_mult = 2.0_f64.powf(semitones / 12.0);
    }

    fn note_on(&mut self, params: &VagusParams, note: u8, velocity: f32) {
        self.poly_manager.note_on(params, note, velocity);
    }

    fn note_off(&mut self, note: u8) {
        self.poly_manager.note_off(note);
    }
}

/// Stateless descriptor - the synth's per-block DSP state is [`SynthDspState`].
#[derive(Default)]
pub struct Vagus;

impl PluginLogic for Vagus {
    type Params = VagusParams;
    type DspState = VagusDspState;

    fn bus_layouts() -> Vec<BusLayout> {
        BusLayout::stereo_and_mono_output()
    }

    fn reset(state: &mut VagusDspState, _params: &VagusParams, config: &AudioConfig) {
        let sample_rate = config.sample_rate;
        state.sample_rate = sample_rate;
        state.pitch_bend_mult = 1.0;
        state.mod_wheel = 0.0;
        state.lfo_phase = 0.0;
    }

    fn process(
        state: &mut VagusDspState,
        params: &VagusParams,
        buffer: &mut AudioBuffer,
        events: &EventList,
        _context: &mut ProcessContext,
    ) -> ProcessStatus {
        let mut next_event = 0;
        // A mono or multi-mono host instance hands us a single output
        // channel; writing a second would be out of bounds.
        let out_channels = buffer.num_output_channels();

        for i in 0..buffer.num_samples() {
            while let Some(event) = events.get(next_event) {
                if event.sample_offset as usize > i {
                    break;
                }
                match &event.body {
                    EventBody::NoteOn { note, velocity, .. } => {
                        state.note_on(params, *note, norm_7bit(*velocity));
                    }
                    EventBody::NoteOff { note, .. } => state.note_off(*note),
                    EventBody::PitchBend { value, .. } => state.pitch_bend(*value),
                    // CC1 is the mod wheel; steer vibrato depth from it.
                    EventBody::ControlChange { cc: 1, value, .. } => {
                        state.mod_wheel = f64::from(norm_7bit(*value));
                    }
                    _ => {}
                }
                next_event += 1;
            }

            let algo_idx = params.algorithm.index();
            let bin_count = params.bin_count.value_usize();
            let volume = db_to_linear(params.volume.read());

            // Advance the shared vibrato LFO and fold it into the
            // channel pitch bend, so every voice gets one combined
            // pitch multiplier this sample.
            let vibrato_semitones =
                state.mod_wheel * VIBRATO_DEPTH_SEMITONES * (state.lfo_phase * TAU).sin();
            state.lfo_phase += VIBRATO_RATE_HZ / state.sample_rate;
            if state.lfo_phase >= 1.0 {
                state.lfo_phase -= 1.0;
            }
            let pitch_mult = state.pitch_bend_mult * 2.0_f64.powf(vibrato_semitones / 12.0);

            let mut sample = 0.0f64;

            for voice in &mut state.poly_manager.voices {
                sample += voice.voice.process();
            }
            sample *= volume;

            let out = sample.clamp(-1.0, 1.0);
            buffer.output(0)[i] = out;
            if out_channels > 1 {
                buffer.output(1)[i] = out;
            }
        }

        if state.poly_manager.none_playing() {
            ProcessStatus::Tail(0)
        } else {
            ProcessStatus::Normal
        }
    }

    fn editor(params: Arc<VagusParams>) -> Box<dyn Editor> {
        GridLayout::build(vec![section(
            "Engine",
            vec![
                dropdown(P::Algorithm, "Algorithm").cols(2),
                knob(P::Volume, "Volume"),
                knob(params.bin_count.id(), "Depth"),
                knob(params.spectral_tilt.id(), "Tilt"),
            ],
        )])
        .with_title("Vagus")
        .into_editor(&params)
    }
}

truce::plugin! {
    logic: Vagus,
    params: VagusParams,
}

// Installs the real-time allocation checker under `--features rt-paranoid`
// (a no-op otherwise). Wrap a driver run in `assert_no_audio_alloc` to
// fail a test if `process` ever allocates. See the audio-testing guide.
truce::enable_rt_paranoid!();
