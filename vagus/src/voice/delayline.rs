/// A basic fractional delay line
#[derive(Clone)]
pub struct DelayLine {
    pub(crate) buffer: Vec<f64>,
    pub(crate) mask: usize,
    pub(crate) write_idx: usize,
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
