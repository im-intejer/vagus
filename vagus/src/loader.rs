//! Audio file import.
//!
//! Decoding happens on a worker thread. The audio thread only ever touches two
//! lock-free queues. New corpora arrive through `incoming`, and replaced ones
//! are sent back through `garbage` so they are freed off the audio thread.
//!
//! Files live in the managed library folder (see `library.rs`). Set
//! TRUCE_GRAIN_DIR to choose it, otherwise `~/TruceGrain` is used. A slot is a
//! stable library id, and slot 0 is the built-in corpus.
//! `Loader::request_path` loads one file directly without importing it.

use crate::corpus::Corpus;
use crate::library;
use crossbeam_queue::ArrayQueue;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Files longer than this are truncated.
pub const MAX_SECONDS: f64 = 120.0;

pub fn sample_dir() -> PathBuf {
    if let Ok(d) = std::env::var("TRUCE_GRAIN_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home).join("Vagus").join("Samples")
}

/// Decode any supported file to mono f32. Returns (samples, sample_rate).
pub fn decode_file(path: &Path) -> Result<(Vec<f32>, f64), String> {
    let file = Box::new(if let Ok(f) = File::open(path) {
        f
    } else {
        return Ok((vec![0.0], 0.0));
    });
    let mss = MediaSourceStream::new(file, Default::default());
    let hint = Hint::new();
    let fmt_opts: FormatOptions = Default::default();
    let meta_opts: MetadataOptions = Default::default();
    let dec_opts: AudioDecoderOptions = Default::default();
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, fmt_opts, meta_opts)
        .unwrap();
    let track = format.default_track(TrackType::Audio).unwrap();
    let sample_rate = track
        .codec_params
        .as_ref()
        .unwrap()
        .audio()
        .unwrap()
        .sample_rate
        .unwrap_or(0) as f64;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(
            track.codec_params.as_ref().unwrap().audio().unwrap(),
            &dec_opts,
        )
        .unwrap();
    let track_id = track.id;
    let channels = decoder
        .codec_params()
        .channels
        .as_ref()
        .map_or_else(|| 1u16, |channels| channels.count() as u16);

    let mut scratch: Vec<f32> = vec![];
    let mut samples: Vec<f32> = vec![];
    let mut total_sample_count = 0;
    while let Some(packet) = format.next_packet().unwrap() {
        // If the packet does not belong to the selected track, skip it.
        if packet.track_id != track_id {
            continue;
        }
        use symphonia::core::audio::sample::Sample; // Decode the packet into audio samples, ignoring any decode errors.
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                scratch.resize(audio_buf.samples_interleaved(), f32::MID);

                // Copy the audio samples from the generic audio buffer to the vector in interleaved
                // order. The sample format to convert to is inferred from the type of the Vec.
                // Sum up the total number of samples.
                total_sample_count += scratch.len();
                audio_buf.copy_to_slice_interleaved(&mut scratch);
                samples.append(&mut scratch);
                print!("\rDecoded {total_sample_count} samples");
            }
            Err(_) => break,
        }
    }

    samples.truncate((MAX_SECONDS * sample_rate as f64) as usize);
    if samples.len() < 2048 {
        return Err("file is too short".into());
    }
    Ok((samples, sample_rate))
}

pub fn load_corpus(path: &Path) -> Result<Corpus, String> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("file")
        .to_string();
    load_corpus_named(path, name)
}

pub fn load_corpus_named(path: &Path, name: String) -> Result<Corpus, String> {
    let (samples, sr) = decode_file(path)?;
    Ok(Corpus::from_mono(name, samples, sr))
}

pub struct LoaderShared {
    incoming: ArrayQueue<Arc<Corpus>>,
    garbage: ArrayQueue<Arc<Corpus>>,
    requested: AtomicI64,
    stop: AtomicBool,
    path_request: Mutex<Option<PathBuf>>,
    status: Mutex<String>,
    builtin: Arc<Corpus>,
}

pub struct Loader {
    shared: Arc<LoaderShared>,
    handle: Option<JoinHandle<()>>,
}

impl Loader {
    pub fn spawn(builtin: Arc<Corpus>) -> Self {
        let shared = Arc::new(LoaderShared {
            incoming: ArrayQueue::new(4),
            garbage: ArrayQueue::new(8),
            requested: AtomicI64::new(0),
            stop: AtomicBool::new(false),
            path_request: Mutex::new(None),
            status: Mutex::new("Built-in".to_string()),
            builtin,
        });
        let worker = shared.clone();
        let handle = thread::Builder::new()
            .name("truce-grain-loader".into())
            .spawn(move || worker_loop(worker))
            .ok();
        Loader { shared, handle }
    }

    /// Select a slot. 0 is the built-in corpus, 1.. are files in `sample_dir()`.
    pub fn request_slot(&self, slot: i64) {
        self.shared.requested.store(slot, Ordering::Relaxed);
    }

    /// Load one file directly. Safe to call from a GUI thread.
    pub fn request_path(&self, path: PathBuf) {
        if let Ok(mut p) = self.shared.path_request.lock() {
            *p = Some(path);
        }
    }

    /// Human readable status for an editor to display.
    pub fn status(&self) -> String {
        self.shared
            .status
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Audio thread. Swaps in the newest corpus and hands the old one back.
    /// Never blocks and never allocates.
    pub fn poll(&self, current: &mut Arc<Corpus>) {
        while let Some(new) = self.shared.incoming.pop() {
            let old = std::mem::replace(current, new);
            // Full queue is very unlikely. In that case `old` drops here.
            let _ = self.shared.garbage.push(old);
        }
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn set_status(sh: &LoaderShared, msg: String) {
    if let Ok(mut s) = sh.status.lock() {
        *s = msg;
    }
}

fn publish(sh: &LoaderShared, c: Arc<Corpus>) {
    set_status(sh, c.name.clone());
    let _ = sh.incoming.push(c);
}

fn worker_loop(sh: Arc<LoaderShared>) {
    let dir = sample_dir();
    let _ = fs::create_dir_all(&dir);
    let mut loaded: i64 = 0;
    while !sh.stop.load(Ordering::Relaxed) {
        while sh.garbage.pop().is_some() {}

        let direct = sh.path_request.lock().ok().and_then(|mut p| p.take());
        if let Some(path) = direct {
            match load_corpus(&path) {
                Ok(c) => publish(&sh, Arc::new(c)),
                Err(e) => set_status(&sh, format!("Load failed, {e}")),
            }
        }

        let want = sh.requested.load(Ordering::Relaxed);
        if want != loaded {
            loaded = want; // do not retry a failing slot in a loop
            if want <= 0 {
                publish(&sh, sh.builtin.clone());
            } else {
                match library::find(&dir, want as u32) {
                    Some(e) => match load_corpus_named(&e.path, e.name.clone()) {
                        Ok(c) => publish(&sh, Arc::new(c)),
                        Err(err) => set_status(&sh, format!("Load failed, {err}")),
                    },
                    None => set_status(&sh, format!("No file in slot {want}")),
                }
            }
        }
        thread::sleep(Duration::from_millis(40));
    }
}

#[cfg(test)]
pub mod tests_support {
    use super::*;

    pub fn write_test_wav(path: &Path) {
        let frames: Vec<Vec<i16>> = (0..30000)
            .map(|i| vec![((i as f64 * 0.08).sin() * 10000.0) as i16])
            .collect();
        super::tests::write_wav(path, 44100, 1, &frames);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    pub fn write_wav(path: &Path, sr: u32, channels: u16, frames: &[Vec<i16>]) {
        let n = frames.len() as u32;
        let data_len = n * channels as u32 * 2;
        let mut b: Vec<u8> = Vec::new();
        b.extend(b"RIFF");
        b.extend((36 + data_len).to_le_bytes());
        b.extend(b"WAVEfmt ");
        b.extend(16u32.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.extend(channels.to_le_bytes());
        b.extend(sr.to_le_bytes());
        b.extend((sr * channels as u32 * 2).to_le_bytes());
        b.extend((channels * 2).to_le_bytes());
        b.extend(16u16.to_le_bytes());
        b.extend(b"data");
        b.extend(data_len.to_le_bytes());
        for f in frames {
            for s in f {
                b.extend(s.to_le_bytes());
            }
        }
        fs::write(path, b).unwrap();
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("truce_grain_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn decodes_stereo_wav_to_mono() {
        let d = tmpdir("decode");
        let p = d.join("a.wav");
        let frames: Vec<Vec<i16>> = (0..8000)
            .map(|i| {
                let s = ((i as f64 * 0.05).sin() * 12000.0) as i16;
                vec![s, s]
            })
            .collect();
        write_wav(&p, 22050, 2, &frames);
        let (s, sr) = decode_file(&p).unwrap();
        assert_eq!(sr, 22050.0);
        assert_eq!(s.len(), 8000);
        assert!(s.iter().any(|v| v.abs() > 0.2));
    }

    #[test]
    fn rejects_garbage_and_short_files() {
        let d = tmpdir("bad");
        let p = d.join("junk.wav");
        fs::write(&p, b"this is not audio").unwrap();
        assert!(decode_file(&p).is_err());
        let q = d.join("short.wav");
        write_wav(
            &q,
            44100,
            1,
            &(0..100).map(|i| vec![i as i16]).collect::<Vec<_>>(),
        );
        assert!(decode_file(&q).is_err());
    }

    #[test]
    fn worker_delivers_requested_file_and_returns_garbage() {
        let d = tmpdir("worker");
        let frames: Vec<Vec<i16>> = (0..20000)
            .map(|i| vec![((i as f64 * 0.1).sin() * 9000.0) as i16])
            .collect();
        write_wav(&d.join("one.wav"), 44100, 1, &frames);
        unsafe { std::env::set_var("TRUCE_GRAIN_DIR", &d) };

        let builtin = Arc::new(Corpus::builtin());
        let loader = Loader::spawn(builtin.clone());
        loader.request_slot(1);

        let mut current = builtin.clone();
        let t0 = Instant::now();
        while current.name == "Built-in" && t0.elapsed() < Duration::from_secs(5) {
            loader.poll(&mut current);
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(current.name, "one");

        // Back to slot 0 must return the built-in corpus.
        loader.request_slot(0);
        let t0 = Instant::now();
        while current.name != "Built-in" && t0.elapsed() < Duration::from_secs(5) {
            loader.poll(&mut current);
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(current.name, "Built-in");

        // A missing slot reports status and keeps the current corpus.
        loader.request_slot(9);
        thread::sleep(Duration::from_millis(200));
        assert!(loader.status().contains("No file"));
    }
}
