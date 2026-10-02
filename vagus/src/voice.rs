/// Fast tanh approximation for nonlinear scattering
#[inline]
fn fast_tanh(x: f64) -> f64 {
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}

mod delayline;
mod exciter;
mod mesh;

/// The Combined Truce Voice processing struct
#[derive(Clone)]
pub struct Voice {
    pub exciter: exciter::SpectralExciter,
    pub mesh: mesh::NonlinearMesh,
    pub active: bool,
}

impl Voice {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            exciter: exciter::SpectralExciter::new(32, sample_rate), // 32 spectral bins
            mesh: mesh::NonlinearMesh::new(),
            active: false,
        }
    }

    pub fn note_on(&mut self, note_freq: f64) {
        self.active = true;
        self.exciter.trigger(note_freq);
        // Here you would also map velocity/keytrack to mesh delay times or drive
    }

    pub fn note_off(&mut self) {
        self.active = false;
    }

    /// Called by your audio thread every sample, per voice
    #[inline]
    pub fn process(&mut self) -> f64 {
        if !self.active {
            return 0.0;
        }

        let excitation = self.exciter.process();
        self.mesh.process(excitation)
    }
}

#[derive(Clone)]
pub struct PolyVoice {
    pub voice: Voice,
    pub midi_note: u8,
    pub is_released: bool,
    pub triggered_at: u64,
}
