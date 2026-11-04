//! Housekeeping for `~/.yak/tmp`: the scratch dir the editor drops pasted
//! clipboard images and ctrl+g buffers into. Thread files keep an attachment's
//! path as provenance, never its bytes, so nothing else ever revisits these
//! files — without a prune the directory grows for the life of the install.
//! Each agent start sweeps what has aged past the retention window.

use std::path::Path;
use std::time::{Duration, SystemTime};

/// A week: long enough that resuming a thread from earlier in the week still
/// finds the images it references, short enough that the dir stays bounded.
pub const RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Retrieved originals (`tmp/retrieve`, written by `agent::compact` before a
/// tool result's middle is cut) outlive the scratch files: the thread that
/// carries the marker can stay resumable for months, and a marker pointing at
/// a deleted archive is a dead end. A quarter is the compromise between "the
/// transcript still works" and "the directory stays bounded".
pub const RETRIEVE_RETENTION: Duration = Duration::from_secs(90 * 24 * 60 * 60);

/// Delete regular files in `dir` last modified more than `retention` before
/// `now`; returns how many went. A missing dir is simply nothing to do, and
/// an entry that cannot be stat'ed or removed is left alone rather than
/// failing the session that is only doing housekeeping.
pub fn prune_before(dir: &Path, now: SystemTime, retention: Duration) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let Ok(modified) = meta.modified() else {
            continue;
        };
        let age = now.duration_since(modified).unwrap_or_default();
        if age > retention && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Prune the user's tmp dir against the wall clock. The top level is
/// non-recursive, so `retrieve/` (content-addressed originals) is swept on its
/// own, longer clock instead of dying with the pasted images it sits next to.
pub fn prune_user_tmp() -> usize {
    let tmp = crate::core::config::user_dir().join("tmp");
    let now = SystemTime::now();
    prune_before(&tmp, now, RETENTION)
        + prune_before(&tmp.join("retrieve"), now, RETRIEVE_RETENTION)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_drops_aged_files_and_keeps_fresh_ones() {
        let dir = crate::core::testutil::scratch_dir("tmp-prune");
        std::fs::write(dir.join("paste-fresh.png"), b"x").unwrap();
        std::fs::write(dir.join("paste-old.png"), b"x").unwrap();
        std::fs::create_dir(dir.join("inner")).unwrap();
        // the fresh file is created now, so a clock one retention past it is
        // exactly what "aged" means — no sleeping, no mtime surgery
        let now = SystemTime::now() + RETENTION + Duration::from_secs(60);
        assert_eq!(prune_before(&dir, now, RETENTION), 2);
        assert!(!dir.join("paste-fresh.png").exists());
        assert!(!dir.join("paste-old.png").exists());
        assert!(
            dir.join("inner").is_dir(),
            "directories are not ours to cut"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn retrieved_originals_outlive_the_scratch_window() {
        // a pruned tool result's archive is what a resumed thread's marker
        // points at: the week that clears a pasted screenshot must not clear it
        let dir = crate::core::testutil::scratch_dir("tmp-retrieve");
        std::fs::write(dir.join("abc.txt"), b"full output").unwrap();
        let past_week = SystemTime::now() + RETENTION + Duration::from_secs(60);
        assert_eq!(
            prune_before(&dir, past_week, RETRIEVE_RETENTION),
            0,
            "still inside the archive's own window"
        );
        assert!(dir.join("abc.txt").exists());
        let past_quarter = SystemTime::now() + RETRIEVE_RETENTION + Duration::from_secs(60);
        assert_eq!(prune_before(&dir, past_quarter, RETRIEVE_RETENTION), 1);
        assert!(!dir.join("abc.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nothing_to_prune_is_not_an_error() {
        let dir = crate::core::testutil::scratch_path("tmp-absent");
        assert_eq!(prune_before(&dir, SystemTime::now(), RETENTION), 0);
    }

    #[test]
    fn a_file_inside_the_window_survives() {
        let dir = crate::core::testutil::scratch_dir("tmp-keep");
        std::fs::write(dir.join("paste-now.png"), b"x").unwrap();
        assert_eq!(prune_before(&dir, SystemTime::now(), RETENTION), 0);
        assert!(dir.join("paste-now.png").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
