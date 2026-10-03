// Granular polysynth for truce.
//
// Each voice is a grain cloud reading from an imported audio file. The summed
// voices excite an eight line self-resampling grain FDN. All DSP lives in the
// framework-free modules below. This file only maps parameters and MIDI onto
// `engine::Engine`.
//
// Importing audio. Files live in a managed library folder, ~/TruceGrain or the
// folder named by TRUCE_GRAIN_DIR, as NNNN_name.ext. The Sample parameter holds
// the file's stable id, with 0 meaning the built-in sound. `library::import`
// copies a file in and returns its id, which is what a drop handler sets.

use std::sync::Arc;

use truce::core::midi::{norm_7bit, norm_pitch_bend};
use truce::prelude64::*;
use truce_gui::IntoLayoutEditor;
use truce_gui_types::layout::{GridLayout, knob, section};

pub mod corpus;
pub mod engine;
pub mod fdn;
pub mod grain;
pub mod library;
pub mod loader;
pub mod util;
pub mod voice;

use engine::{Engine, EngineSettings};

// --- Parameters ---

#[derive(Params)]
pub struct SourceParams {
    #[param(
        name = "Sample",
        short_name = "Sample",
        group = "Source",
        range = "linear(0, 9999)",
        default = 0.0
    )]
    pub sample: FloatParam,

    #[param(
        name = "Root Note",
        short_name = "Root",
        group = "Source",
        range = "linear(21, 108)",
        default = 60.0,
        unit = "st"
    )]
    pub root: IntParam,
}

#[derive(Params)]
pub struct GrainParams {
    #[param(
        name = "Position",
        short_name = "Pos",
        group = "Grains",
        range = "linear(0, 1)",
        default = 0.3,
        smooth = "exp(10)"
    )]
    pub position: FloatParam,

    #[param(
        name = "Spread",
        short_name = "Spread",
        group = "Grains",
        range = "linear(0, 1)",
        default = 0.2
    )]
    pub spread: FloatParam,

    #[param(
        name = "Grain Size",
        short_name = "Size",
        group = "Grains",
        range = "log(10, 500)",
        default = 80.0,
        unit = "ms"
    )]
    pub size: FloatParam,

    #[param(
        name = "Density",
        short_name = "Dens",
        group = "Grains",
        range = "log(0.5, 12)",
        default = 3.0
    )]
    pub density: FloatParam,

    #[param(
        name = "Detune",
        short_name = "Detune",
        group = "Grains",
        range = "linear(0, 100)",
        default = 8.0,
        // unit = "ct"
    )]
    pub detune: FloatParam,

    #[param(
        name = "Key Follow",
        short_name = "Follow",
        group = "Grains",
        range = "linear(0, 1)",
        default = 1.0
    )]
    pub follow: FloatParam,

    #[param(
        name = "Width",
        short_name = "Width",
        group = "Grains",
        range = "linear(0, 1)",
        default = 0.5
    )]
    pub width: FloatParam,

    #[param(
        name = "Tone",
        short_name = "Tone",
        group = "Grains",
        range = "linear(0, 1)",
        default = 0.5,
        smooth = "exp(10)"
    )]
    pub tone: FloatParam,

    #[param(
        name = "Focus",
        short_name = "Focus",
        group = "Grains",
        range = "linear(1, 16)",
        default = 4.0
    )]
    pub focus: FloatParam,
}

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
pub struct FdnParams {
    #[param(
        name = "Memory Mix",
        short_name = "Mix",
        group = "Memory",
        range = "linear(0, 1)",
        default = 0.35,
        smooth = "exp(10)"
    )]
    pub mix: FloatParam,

    #[param(
        name = "Memory Feedback",
        short_name = "Fdbk",
        group = "Memory",
        range = "linear(0, 1)",
        default = 0.7,
        smooth = "exp(10)"
    )]
    pub feedback: FloatParam,

    #[param(
        name = "Memory Length",
        short_name = "Length",
        group = "Memory",
        range = "log(0.1, 3)",
        default = 1.0,
        unit = "s"
    )]
    pub length: FloatParam,

    #[param(
        name = "Memory Grain",
        short_name = "Grain",
        group = "Memory",
        range = "log(20, 300)",
        default = 90.0,
        unit = "ms"
    )]
    pub grain: FloatParam,

    #[param(
        name = "Memory Shift",
        short_name = "Shift",
        group = "Memory",
        range = "linear(-12, 12)",
        default = 0.0,
        unit = "st"
    )]
    pub shift: FloatParam,

    #[param(
        name = "Memory Tone",
        short_name = "MTone",
        group = "Memory",
        range = "linear(0, 1)",
        default = 0.5
    )]
    pub tone: FloatParam,

    #[param(
        name = "Memory Focus",
        short_name = "MFocus",
        group = "Memory",
        range = "linear(1, 16)",
        default = 3.0
    )]
    pub focus: FloatParam,

    #[param(
        name = "Memory Damping",
        short_name = "Damp",
        group = "Memory",
        range = "log(500, 16000)",
        default = 7000.0,
        unit = "Hz"
    )]
    pub damp: FloatParam,
}

#[derive(Params)]
pub struct SynthParams {
    #[nested]
    pub source: SourceParams,

    #[nested]
    pub grains: GrainParams,

    #[nested]
    pub filter: FilterParams,

    #[nested]
    pub envelope: EnvParams,

    #[nested]
    pub memory: FdnParams,

    #[param(name = "Volume", short_name = "Vol",
            range = "linear(-60, 0)", default = -6.0,
            unit = "dB", smooth = "exp(5)")]
    pub volume: FloatParam,

    /// Hidden MIDI proxy for pitch bend, see the stock polysynth example.
    #[param(
        name = "Pitch Bend",
        short_name = "Bend",
        range = "linear(-1, 1)",
        default = 0.0,
        flags = "hidden | automatable",
        midi_source = "pitchbend"
    )]
    pub bend: FloatParam,

    /// Hidden MIDI proxy for the mod wheel, drives vibrato depth.
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

pub struct Synth;

pub struct SynthDspState {
    engine: Engine,
    settings: EngineSettings,
}

impl Default for SynthDspState {
    fn default() -> Self {
        SynthDspState {
            engine: Engine::new(44100.0),
            settings: EngineSettings::default(),
        }
    }
}

impl PluginLogic for Synth {
    type Params = SynthParams;
    type DspState = SynthDspState;

    fn bus_layouts() -> Vec<BusLayout> {
        BusLayout::stereo_and_mono_output()
    }

    fn reset(state: &mut SynthDspState, _params: &SynthParams, config: &AudioConfig) {
        state.engine.reset(config.sample_rate);
    }

    fn process(
        state: &mut SynthDspState,
        params: &SynthParams,
        buffer: &mut AudioBuffer,
        events: &EventList,
        _context: &mut ProcessContext,
    ) -> ProcessStatus {
        let engine = &mut state.engine;

        let settings = &mut state.settings;
        settings.update_block_settings(params);

        engine.begin_block(params.source.sample.value().round() as i64);

        let mut next_event = 0;
        let out_channels = buffer.num_output_channels();

        for i in 0..buffer.num_samples() {
            // Smoothed parameters are read per sample.
            settings.cutoff = params.filter.cutoff.read();
            settings.resonance = params.filter.resonance.read();
            settings.volume = db_to_linear(params.volume.read());
            settings.cloud.position = params.grains.position.read();
            settings.cloud.tone = params.grains.tone.read();
            settings.fdn.mix = params.memory.mix.read();
            settings.fdn.feedback = params.memory.feedback.read();

            while let Some(event) = events.get(next_event) {
                if event.sample_offset as usize > i {
                    break;
                }
                match &event.body {
                    EventBody::NoteOn { note, velocity, .. } => {
                        engine.note_on(settings, *note, f64::from(norm_7bit(*velocity)));
                    }
                    EventBody::NoteOff { note, .. } => engine.note_off(settings, *note),
                    EventBody::PitchBend { value, .. } => {
                        engine.set_pitch_bend(f64::from(norm_pitch_bend(*value)));
                    }
                    EventBody::ControlChange { cc: 1, value, .. } => {
                        engine.set_mod_wheel(f64::from(norm_7bit(*value)));
                    }
                    EventBody::ControlChange { cc: 64, value, .. } => {
                        engine.sustain_pedal(settings, norm_7bit(*value) >= 0.5);
                    }
                    EventBody::ControlChange { cc: 123, .. } => engine.all_notes_off(settings),
                    _ => {}
                }
                next_event += 1;
            }

            let (l, r) = engine.process_frame(settings);
            buffer.output(0)[i] = l;
            if out_channels > 1 {
                buffer.output(1)[i] = r;
            }
        }

        if engine.is_idle() {
            ProcessStatus::Tail(0)
        } else {
            ProcessStatus::Normal
        }
    }

    fn editor(params: Arc<SynthParams>) -> Box<dyn Editor> {
        GridLayout::build(vec![
            section(
                "SOURCE",
                vec![
                    knob(params.source.sample.id(), "Sample"),
                    knob(params.source.root.id(), "Root"),
                    knob(params.volume.id(), "Volume"),
                ],
            ),
            section(
                "GRAINS",
                vec![
                    knob(params.grains.position.id(), "Position"),
                    knob(params.grains.spread.id(), "Spread"),
                    knob(params.grains.size.id(), "Size"),
                    knob(params.grains.density.id(), "Density"),
                    knob(params.grains.detune.id(), "Detune"),
                    knob(params.grains.follow.id(), "Follow"),
                    knob(params.grains.width.id(), "Width"),
                    knob(params.grains.tone.id(), "Tone"),
                    knob(params.grains.focus.id(), "Focus"),
                ],
            ),
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
            section(
                "MEMORY",
                vec![
                    knob(params.memory.mix.id(), "Mix"),
                    knob(params.memory.feedback.id(), "Feedback"),
                    knob(params.memory.length.id(), "Length"),
                    knob(params.memory.grain.id(), "Grain"),
                    knob(params.memory.shift.id(), "Shift"),
                    knob(params.memory.tone.id(), "Tone"),
                    knob(params.memory.focus.id(), "Focus"),
                    knob(params.memory.damp.id(), "Damp"),
                ],
            ),
        ])
        .with_title("VAGUS")
        .into_editor(&params)
    }
}

truce::plugin! {
    logic: Synth,
    params: SynthParams,
}

truce::enable_rt_paranoid!();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_is_allocation_free() {
        use std::time::Duration;
        use truce_test::{assert_no_audio_alloc, driver};
        assert_no_audio_alloc(|| {
            driver!(Plugin)
                .duration(Duration::from_millis(60))
                .script(|s| {
                    for n in 48..72u8 {
                        s.note_on(n, 0.8);
                    }
                    s.wait_ms(20);
                    s.note_off(60);
                    s.cc(64, 1.0);
                    s.wait_ms(20);
                })
                .run()
        });
    }

    #[test]
    fn info_is_valid() {
        truce_test::assert_valid_info::<Plugin>();
    }

    #[test]
    fn silence_without_midi() {
        use std::time::Duration;
        use truce_test::{assertions, driver};
        let result = driver!(Plugin).duration(Duration::from_millis(12)).run();
        assertions::assert_silence(&result);
    }

    #[test]
    fn produces_sound_on_note_on() {
        use std::time::Duration;
        use truce_test::{assertions, driver};
        let result = driver!(Plugin)
            .duration(Duration::from_millis(200))
            .script(|s| s.note_on(60, 100.0 / 127.0))
            .run();
        assertions::assert_nonzero(&result);
        assertions::assert_no_nans(&result);
    }

    #[test]
    fn has_editor() {
        truce_test::assert_has_editor::<Plugin>();
    }

    #[test]
    fn state_round_trips() {
        truce_test::assert_state_round_trip::<Plugin>();
    }

    #[test]
    fn param_defaults_match() {
        truce_test::assert_param_defaults_match::<Plugin>();
    }

    #[test]
    fn param_normalized_clamped() {
        truce_test::assert_param_normalized_clamped::<Plugin>();
    }

    #[test]
    fn param_normalized_roundtrip() {
        truce_test::assert_param_normalized_roundtrip::<Plugin>();
    }

    #[test]
    fn no_duplicate_param_ids() {
        truce_test::assert_no_duplicate_param_ids::<Plugin>();
    }

    #[test]
    fn editor_lifecycle() {
        truce_test::assert_editor_lifecycle::<Plugin>();
    }
}
