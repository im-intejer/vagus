use std::f64::consts::PI;

/// Fast tanh approximation for nonlinear scattering
#[inline]
fn fast_tanh(x: f64) -> f64 {
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}

/// A basic fractional delay line
#[derive(Clone)]
pub struct DelayLine {
    buffer: Vec<f64>,
    mask: usize,
    write_idx: usize,
}

impl DelayLine {
    pub fn new(max_delay_samples: usize) -> Self {
        // Power of two for fast masking
        let size = max_delay_samples.next_power_of_two();
        Self {
            buffer: vec![0.0; size],
            mask: size - 1,
            write_idx: 0,
        }
    }

    #[inline]
    pub fn read(&self, delay_samples: f64) -> f64 {
        let read_idx = self.write_idx as f64 - delay_samples;

        // Linear interpolation for sub-sample delay modulation
        let idx_int = read_idx.floor() as usize;
        let frac = read_idx - read_idx.floor();

        let val_a = self.buffer[idx_int & self.mask];
        let val_b = self.buffer[(idx_int + 1) & self.mask];

        val_a + frac * (val_b - val_a)
    }

    #[inline]
    pub fn push(&mut self, sample: f64) {
        self.buffer[self.write_idx] = sample;
        self.write_idx = (self.write_idx + 1) & self.mask;
    }
}

/// Component 1: Spectral Exciter
/// Acts as the "bow" or "strike", manipulating frequency bins directly.
#[derive(Clone)]
pub struct SpectralExciter {
    phases: Vec<f64>,
    phase_increments: Vec<f64>,
    amplitudes: Vec<f64>,
    sample_rate: f64,
}

impl SpectralExciter {
    pub fn new(bins: usize, sample_rate: f64) -> Self {
        Self {
            phases: vec![0.0; bins],
            phase_increments: vec![0.0; bins],
            amplitudes: vec![0.0; bins],
            sample_rate,
        }
    }

    pub fn trigger(&mut self, fundamental: f64, tilt: f64) {
        for (i, (inc, amp)) in self
            .phase_increments
            .iter_mut()
            .zip(self.amplitudes.iter_mut())
            .enumerate()
        {
            let partial = (i + 1) as f64;
            let freq = fundamental * partial;

            // Randomize starting phase for organic "clank" per voice
            self.phases[i] = rand::random::<f64>() * 2.0 * PI;
            *inc = (freq * 2.0 * PI) / self.sample_rate;

            // Spectral tilt: adjust bin amplitudes (e.g., negative tilt = lowpass, positive = highpass)
            *amp = partial.powf(tilt);
        }
    }

    #[inline]
    pub fn process(&mut self) -> f64 {
        let mut out = 0.0;
        for (phase, (&inc, &amp)) in self
            .phases
            .iter_mut()
            .zip(self.phase_increments.iter().zip(&self.amplitudes))
        {
            out += phase.sin() * amp;
            *phase += inc;
            if *phase > 2.0 * PI {
                *phase -= 2.0 * PI;
            }
        }
        out
    }
}

/// Component 2 & 3: Nonlinear Waveguide Mesh
/// A 2x2 grid of scattering junctions with non-linear saturation.
#[derive(Clone)]
pub struct NonlinearMesh {
    nodes: [DelayLine; 4],
    delay_times: [f64; 4],
    damping: f64,
    drive: f64,
}

impl NonlinearMesh {
    pub fn new() -> Self {
        Self {
            nodes: [
                DelayLine::new(2048),
                DelayLine::new(2048),
                DelayLine::new(2048),
                DelayLine::new(2048),
            ],
            // In a real synth, these delay times dictate the physical "size" and "shape" of the resonator
            delay_times: [142.3, 211.5, 345.1, 411.7],
            damping: 0.98, // High frequency loss in the "material"
            drive: 2.0,    // Pushes the mesh into non-linear chaos
        }
    }

    #[inline]
    pub fn process(&mut self, exciter_in: f64) -> f64 {
        // Read current state from the delay lines
        let v0 = self.nodes[0].read(self.delay_times[0]);
        let v1 = self.nodes[1].read(self.delay_times[1]);
        let v2 = self.nodes[2].read(self.delay_times[2]);
        let v3 = self.nodes[3].read(self.delay_times[3]);

        // Calculate nonlinear scattering junctions (cross-coupling nodes)
        // Node 0 connects to 1 and 2, Node 3 connects to 1 and 2.
        let s0 = fast_tanh((v1 + v2 + exciter_in) * self.drive) * self.damping;
        let s1 = fast_tanh((v0 + v3) * self.drive) * self.damping;
        let s2 = fast_tanh((v0 + v3) * self.drive) * self.damping;
        let s3 = fast_tanh((v1 + v2) * self.drive) * self.damping;

        // Push new energy into the delay network
        self.nodes[0].push(s0);
        self.nodes[1].push(s1);
        self.nodes[2].push(s2);
        self.nodes[3].push(s3);

        // Tap the mesh at node 3 for output
        v3
    }
}

/// The Combined Truce Voice processing struct
#[derive(Clone)]
pub struct Voice {
    exciter: SpectralExciter,
    mesh: NonlinearMesh,
    pub active: bool,
}

impl Voice {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            exciter: SpectralExciter::new(32, sample_rate), // 32 spectral bins
            mesh: NonlinearMesh::new(),
            active: false,
        }
    }

    pub fn note_on(&mut self, note_freq: f64, spectral_tilt: f64) {
        self.active = true;
        self.exciter.trigger(note_freq, spectral_tilt);
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
