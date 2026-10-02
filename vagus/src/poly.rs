use truce::{core::midi_note_to_freq, params::FloatParamReadF64};

use super::VagusParams;

use crate::{
    MAX_VOICES,
    voice::{PolyVoice, Voice},
};

pub struct PolyManager {
    pub(crate) voices: Vec<PolyVoice>,
    pub(crate) time_counter: u64, // Increments every note_on to track relative age
}

impl PolyManager {
    pub(crate) fn new(sample_rate: f64) -> Self {
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

    pub(crate) fn find_voice_to_steal(&self, new_note: u8) -> usize {
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

    pub(crate) fn find_voice_with_note(&self, note: u8) -> Option<usize> {
        if let Some(idx) = self.voices.iter().position(|v| v.midi_note == note) {
            return Some(idx);
        }
        None
    }

    pub fn none_playing(&self) -> bool {
        self.voices.iter().all(|v| !v.voice.active)
    }
}
