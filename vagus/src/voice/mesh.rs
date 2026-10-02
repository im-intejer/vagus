use super::{delayline, fast_tanh};

/// Component 2 & 3: Nonlinear Waveguide Mesh
/// A 2x2 grid of scattering junctions with non-linear saturation.
#[derive(Clone)]
pub struct NonlinearMesh {
    pub(crate) nodes: [delayline::DelayLine; 4],
    pub(crate) delay_times: [f64; 4],
    pub(crate) damping: f64,
    pub(crate) drive: f64,
}

impl NonlinearMesh {
    pub fn new() -> Self {
        Self {
            nodes: [
                delayline::DelayLine::new(2048),
                delayline::DelayLine::new(2048),
                delayline::DelayLine::new(2048),
                delayline::DelayLine::new(2048),
            ],
            // In a real synth, these delay times dictate the physical "size" and "shape" of the resonator
            delay_times: [142.3, 211.5, 345.1, 411.7],
            damping: 0.2, // High frequency loss in the "material"
            drive: 1.0,   // Pushes the mesh into non-linear chaos
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
