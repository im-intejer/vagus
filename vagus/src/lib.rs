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
use truce_gui_types::layout::{GridLayout, dropdown, knob, section, widgets};

mod voice;
use voice::Voice;

// --- Waveform enum ---

#[derive(ParamEnum)]
pub enum Waveform {
    Sine,
    Saw,
    Square,
    Triangle,
}

// --- Parameters ---

use VagusParamsParamId as P;

#[derive(Params)]
pub struct FilterParams {
    #[param(
        name = "Filter Cutoff",
        short_name = "Cutoff",
        group = "Filter",
        range = "log(20, 20000)",
        default = 8000.0,
        unit = "Hz",
        smooth = "exp(5)"
    )]
    pub cutoff: FloatParam,

    #[param(
        name = "Filter Resonance",
        short_name = "Reso",
        group = "Filter",
        range = "linear(0, 1)",
        smooth = "exp(5)"
    )]
    pub resonance: FloatParam,
}

#[derive(Params)]
pub struct EnvParams {
    #[param(
        name = "Attack",
        short_name = "Atk",
        group = "Envelope",
        range = "log(0.001, 5)",
        default = 0.01,
        unit = "s"
    )]
    pub attack: FloatParam,

    #[param(
        name = "Decay",
        short_name = "Dec",
        group = "Envelope",
        range = "log(0.001, 5)",
        default = 0.1,
        unit = "s"
    )]
    pub decay: FloatParam,

    #[param(
        name = "Sustain",
        short_name = "Sus",
        group = "Envelope",
        range = "linear(0, 1)",
        default = 0.7
    )]
    pub sustain: FloatParam,

    #[param(
        name = "Release",
        short_name = "Rel",
        group = "Envelope",
        range = "log(0.01, 10)",
        default = 0.3,
        unit = "s"
    )]
    pub release: FloatParam,
}

#[derive(Params)]
pub struct VagusParams {
    #[param(name = "Waveform", short_name = "Wave", default = 1)]
    pub waveform: EnumParam<Waveform>,

    #[nested]
    pub filter: FilterParams,

    #[nested]
    pub envelope: EnvParams,

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
    voices: Vec<Voice>,
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
        VagusDspState {
            // Pre-sized so `process` never grows the pool on the
            // audio thread.
            voices: Vec::with_capacity(MAX_VOICES),
            sample_rate: 44100.0,
            pitch_bend_mult: 1.0,
            mod_wheel: 0.0,
            lfo_phase: 0.0,
        }
    }
}

impl VagusDspState {
    /// Map a 14-bit pitch-bend code to a frequency multiplier.
    fn pitch_bend(&mut self, value: u16) {
        let semitones = f64::from(norm_pitch_bend(value)) * PITCH_BEND_RANGE;
        self.pitch_bend_mult = 2.0_f64.powf(semitones / 12.0);
    }

    fn note_on(&mut self, params: &VagusParams, note: u8, velocity: f32) {
        let freq = midi_note_to_freq(note);
        let attack = params.envelope.attack.value();
        let decay = params.envelope.decay.value();
        let sustain = params.envelope.sustain.value();
        let release = params.envelope.release.value();

        // Steal the oldest voice *before* pushing, so `len` never exceeds
        // the pre-reserved capacity - a push past capacity would reallocate
        // on the audio thread.
        if self.voices.len() >= MAX_VOICES {
            self.voices.remove(0);
        }
        self.voices.push(Voice::new(
            note,
            freq,
            velocity,
            self.sample_rate,
            attack,
            decay,
            sustain,
            release,
        ));
    }

    fn note_off(&mut self, note: u8) {
        for voice in &mut self.voices {
            if voice.note == note && !voice.releasing {
                voice.release();
            }
        }
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
        state.voices.clear();
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

            let waveform_idx = params.waveform.index();
            let cutoff = params.filter.cutoff.read();
            let resonance = params.filter.resonance.read();
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
            for voice in &mut state.voices {
                sample += voice.render(
                    waveform_idx,
                    cutoff,
                    resonance,
                    state.sample_rate,
                    pitch_mult,
                );
            }
            sample *= volume;

            let out = sample.clamp(-1.0, 1.0);
            buffer.output(0)[i] = out;
            if out_channels > 1 {
                buffer.output(1)[i] = out;
            }
        }

        state.voices.retain(|v| !v.is_done());
        if state.voices.is_empty() {
            ProcessStatus::Tail(0)
        } else {
            ProcessStatus::Normal
        }
    }

    fn editor(params: Arc<VagusParams>) -> Box<dyn Editor> {
        GridLayout::build(vec![
            widgets(vec![
                dropdown(P::Waveform, "Wave").cols(2),
                knob(P::Volume, "Volume"),
            ]),
            section(
                "FILTER",
                vec![
                    knob(params.filter.cutoff.id(), "Cutoff"),
                    knob(params.filter.resonance.id(), "Reso"),
                ],
            ),
            section(
                "ENVELOPE",
                vec![
                    knob(params.envelope.attack.id(), "Attack"),
                    knob(params.envelope.decay.id(), "Decay"),
                    knob(params.envelope.sustain.id(), "Sustain"),
                    knob(params.envelope.release.id(), "Release"),
                ],
            ),
        ])
        .with_title("TRUCE SYNTH")
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
