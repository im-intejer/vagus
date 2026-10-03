//! The plugin's sample library, a managed folder of audio files.
//!
//! Every file is named `NNNN_name.ext`. The number is a stable id that never
//! changes when other files are added or removed, so a saved project can store
//! just the id (the Sample parameter) and find the same file again.
//!
//! Importing copies the file in, so the library does not depend on the
//! original location. Files that a user drops straight into the folder are
//! adopted automatically by giving them the next free id.

use crate::loader::decode_file;
use std::fs;
use std::path::{Path, PathBuf};

pub const EXTENSIONS: [&str; 4] = ["wav", "flac", "mp3", "ogg"];

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub id: u32,
    /// Display name without id prefix or extension.
    pub name: String,
    pub path: PathBuf,
}

pub fn is_audio(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// "0007_my sound" -> (7, "my sound")
fn parse_prefixed(stem: &str) -> Option<(u32, String)> {
    let (num, rest) = stem.split_once('_')?;
    if num.len() != 4 || !num.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let id: u32 = num.parse().ok()?;
    (id >= 1).then(|| (id, rest.to_string()))
}

fn sanitize(stem: &str) -> String {
    let s: String = stem
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(48)
        .collect();
    let s = s.trim().to_string();
    if s.is_empty() { "sample".into() } else { s }
}

fn file_name(id: u32, name: &str, ext: &str) -> String {
    format!("{id:04}_{name}.{}", ext.to_ascii_lowercase())
}

/// All library entries sorted by id. Unnumbered audio files found in the
/// folder are adopted (renamed with the next free id).
pub fn list(dir: &Path) -> Vec<Entry> {
    let mut paths: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| is_audio(p))
            .collect(),
        Err(_) => return Vec::new(),
    };
    paths.sort();

    let mut entries: Vec<Entry> = Vec::new();
    let mut loose: Vec<PathBuf> = Vec::new();
    for p in paths {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        match parse_prefixed(stem) {
            Some((id, name)) if !entries.iter().any(|e| e.id == id) => {
                entries.push(Entry { id, name, path: p })
            }
            _ => loose.push(p),
        }
    }

    let mut next = entries.iter().map(|e| e.id).max().unwrap_or(0) + 1;
    for p in loose {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("sample");
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("wav");
        let name = sanitize(
            parse_prefixed(stem)
                .map(|(_, n)| n)
                .as_deref()
                .unwrap_or(stem),
        );
        let dest = dir.join(file_name(next, &name, ext));
        // If the rename fails (read only folder, race with another thread) the
        // file is simply skipped until the next scan.
        if fs::rename(&p, &dest).is_ok() {
            entries.push(Entry {
                id: next,
                name,
                path: dest,
            });
            next += 1;
        }
    }
    entries.sort_by_key(|e| e.id);
    entries
}

pub fn find(dir: &Path, id: u32) -> Option<Entry> {
    list(dir).into_iter().find(|e| e.id == id)
}

/// Copy a file into the library after checking that it decodes.
/// Importing the same file twice returns the existing entry.
/// Decoding the file makes this slow for long files, so call it from a
/// worker thread rather than directly in a UI callback.
pub fn import(dir: &Path, src: &Path) -> Result<Entry, String> {
    if !is_audio(src) {
        return Err("unsupported file type".into());
    }
    decode_file(src).map_err(|e| format!("cannot read audio, {e}"))?;
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;

    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("sample");
    let name = sanitize(stem);
    let existing = list(dir);

    let bytes = fs::read(src).map_err(|e| e.to_string())?;
    if let Some(dup) = existing
        .iter()
        .find(|e| e.name == name && fs::read(&e.path).map(|b| b == bytes).unwrap_or(false))
    {
        return Ok(dup.clone());
    }

    let id = existing.iter().map(|e| e.id).max().unwrap_or(0) + 1;
    let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("wav");
    let dest = dir.join(file_name(id, &name, ext));
    fs::write(&dest, &bytes).map_err(|e| e.to_string())?;
    Ok(Entry {
        id,
        name,
        path: dest,
    })
}

/// Import several dropped files. Returns one result per file, in order.
pub fn import_many(dir: &Path, srcs: &[PathBuf]) -> Vec<Result<Entry, String>> {
    srcs.iter().map(|p| import(dir, p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::tests_support::write_test_wav;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("truce_grain_lib_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn import_assigns_stable_ids() {
        let lib = tmp("ids");
        let src = tmp("ids_src");
        write_test_wav(&src.join("kick drum!.wav"));
        write_test_wav(&src.join("pad.wav"));
        let a = import(&lib, &src.join("kick drum!.wav")).unwrap();
        let b = import(&lib, &src.join("pad.wav")).unwrap();
        assert_eq!((a.id, b.id), (1, 2));
        assert_eq!(a.name, "kick drum_");
        assert!(a.path.ends_with("0001_kick drum_.wav"));

        // Removing the first file must not renumber the second.
        fs::remove_file(&a.path).unwrap();
        assert_eq!(find(&lib, 2).unwrap().name, "pad");
        assert!(find(&lib, 1).is_none());

        // A new import gets a fresh id, never a reused one.
        write_test_wav(&src.join("new.wav"));
        assert_eq!(import(&lib, &src.join("new.wav")).unwrap().id, 3);
    }

    #[test]
    fn duplicate_import_returns_existing() {
        let lib = tmp("dup");
        let src = tmp("dup_src");
        write_test_wav(&src.join("tone.wav"));
        let a = import(&lib, &src.join("tone.wav")).unwrap();
        let b = import(&lib, &src.join("tone.wav")).unwrap();
        assert_eq!(a, b);
        assert_eq!(list(&lib).len(), 1);
    }

    #[test]
    fn rejects_bad_files() {
        let lib = tmp("bad");
        let src = tmp("bad_src");
        fs::write(src.join("notes.txt"), b"hi").unwrap();
        fs::write(src.join("fake.wav"), b"not audio at all").unwrap();
        assert!(import(&lib, &src.join("notes.txt")).is_err());
        assert!(import(&lib, &src.join("fake.wav")).is_err());
        assert!(list(&lib).is_empty());
    }

    #[test]
    fn loose_files_are_adopted() {
        let lib = tmp("adopt");
        write_test_wav(&lib.join("b.wav"));
        write_test_wav(&lib.join("a.wav"));
        let l = list(&lib);
        assert_eq!(l.iter().map(|e| e.id).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(l[0].name, "a");
        // Second scan is stable.
        assert_eq!(list(&lib), l);
    }
}
