//! Small DSP helpers shared by the grain engine. No dependencies.

use std::f64::consts::TAU;

/// xorshift64* generator. Cheap, allocation free, good enough for grain jitter.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in 0.0..1.0
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in -1.0..1.0
    #[inline]
    pub fn bipolar(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }

    /// Uniform integer in 0..n (returns 0 when n <= 1)
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            0
        } else {
            ((self.unit() * n as f64) as usize).min(n - 1)
        }
    }
}

/// Hann window. `phase` runs 0.0..1.0 across the grain.
#[inline]
pub fn hann(phase: f64) -> f64 {
    0.5 - 0.5 * (TAU * phase).cos()
}

/// 4-point Hermite interpolation between y1 and y2.
#[inline]
pub fn hermite(y0: f64, y1: f64, y2: f64, y3: f64, t: f64) -> f64 {
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
    ((c3 * t + c2) * t + c1) * t + y1
}

/// Interpolated read from a linear buffer. Out of range reads return silence.
#[inline]
pub fn read_clamped(buf: &[f32], pos: f64) -> f64 {
    let n = buf.len() as isize;
    if pos < 0.0 || pos >= n as f64 {
        return 0.0;
    }
    let fl = pos.floor();
    let t = pos - fl;
    let i = fl as isize;
    let g = |k: isize| -> f64 {
        if k < 0 || k >= n {
            0.0
        } else {
            buf[k as usize] as f64
        }
    };
    hermite(g(i - 1), g(i), g(i + 1), g(i + 2), t)
}

/// Interpolated read from a circular buffer.
#[inline]
pub fn read_wrapped(buf: &[f32], pos: f64) -> f64 {
    let n = buf.len() as isize;
    let fl = pos.floor();
    let t = pos - fl;
    let i = fl as isize;
    let g = |k: isize| -> f64 { buf[k.rem_euclid(n) as usize] as f64 };
    hermite(g(i - 1), g(i), g(i + 1), g(i + 2), t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_ranges() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
            assert!(r.below(5) < 5);
            assert!(r.bipolar().abs() <= 1.0);
        }
    }

    #[test]
    fn hermite_hits_endpoints() {
        assert!((hermite(0.0, 1.0, 2.0, 3.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((hermite(0.0, 1.0, 2.0, 3.0, 1.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn wrapped_read_wraps() {
        let b = [0.0f32, 1.0, 2.0, 3.0];
        assert!((read_wrapped(&b, 3.0) - 3.0).abs() < 1e-9);
        assert!((read_wrapped(&b, 4.0) - 0.0).abs() < 1e-9);
        assert_eq!(read_clamped(&b, 4.5), 0.0);
    }
}
