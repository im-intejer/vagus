use std::f64::consts::TAU;

/// Component 1: Spectral Exciter
/// Acts as the "bow" or "strike", manipulating frequency bins directly.
#[derive(Clone)]
pub struct SpectralExciter {
    pub(crate) phases: Vec<f64>,
    pub(crate) phase_increments: Vec<f64>,
    pub(crate) amplitudes: Vec<f64>,
    pub(crate) sample_rate: f64,
    pub tilt: f64,
}

impl SpectralExciter {
    pub fn new(bins: usize, sample_rate: f64) -> Self {
        Self {
            phases: vec![0.0; bins],
            phase_increments: vec![0.0; bins],
            amplitudes: vec![0.0; bins],
            sample_rate,
            tilt: 0.0,
        }
    }

    pub fn trigger(&mut self, fundamental: f64) {
        for (i, (inc, amp)) in self
            .phase_increments
            .iter_mut()
            .zip(self.amplitudes.iter_mut())
            .enumerate()
        {
            let partial = (i + 1) as f64;
            let freq = fundamental * partial;

            // Randomize starting phase for organic "clank" per voice
            self.phases[i] = rand::random::<f64>() * TAU;

            *inc = (freq * TAU) / self.sample_rate;

            // Spectral tilt: adjust bin amplitudes (e.g., negative tilt = lowpass, positive = highpass)
            *amp = partial.powf(self.tilt);
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
            if *phase > TAU {
                *phase -= TAU;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::voice::exciter::SpectralExciter;

    #[test]
    fn spectral_exciter_dump() {
        let mut m = SpectralExciter::new(32, 44100.0);

        m.trigger(440.0);

        let v: Vec<_> = (0..127).map(|_| m.process()).collect();

        println!("{v:#?}");
    }
}
