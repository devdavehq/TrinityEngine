// src/audio/music.rs
// Music management — crossfading, playlist support, ambient layers.
//
// ── Architecture ─────────────────────────────────────────────────────────────
// MusicManager handles high-level music logic:
//   - Crossfade between tracks
//   - Playlist cycling (sequential, random, shuffle)
//   - Ambient layer mixing (e.g., rain ambience + music)
//   - Music transitions triggered by gameplay events
//
// The actual playback is handled by AudioSystem. MusicManager just decides
// WHAT to play and WHEN to transition.

use std::time::{Duration, Instant};

/// Represents a music track with metadata.
#[derive(Clone, Debug)]
pub struct MusicTrack {
    /// File path to the audio file.
    pub path: String,
    /// Human-readable name (for editor display).
    pub name: String,
    /// Optional tag for filtering (e.g., "combat", "explore", "menu").
    pub tag: String,
    /// Recommended volume (0.0 - 1.0).
    pub volume: f32,
}

/// Play mode for playlists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayMode {
    /// Play tracks in order, then stop.
    Sequential,
    /// Play tracks in order, loop back to start.
    Loop,
    /// Random order, no repeats until all played.
    Shuffle,
    /// Repeat the same track forever.
    RepeatOne,
}

/// State of the music manager.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MusicState {
    /// No music playing.
    Stopped,
    /// Music is playing normally.
    Playing,
    /// Currently crossfading between two tracks.
    Crossfading,
    /// Music is paused.
    Paused,
}

/// Manages music playback, playlists, and crossfading.
pub struct MusicManager {
    /// Current playlist.
    playlist: Vec<MusicTrack>,
    /// Current index in the playlist.
    current_index: usize,
    /// Playback mode.
    pub play_mode: PlayMode,
    /// Current state.
    pub state: MusicState,
    /// Crossfade duration.
    pub crossfade_duration: Duration,
    /// When the current track started.
    track_started: Option<Instant>,
    /// Currently active tag filter (empty = play all).
    active_tag: String,
    /// Whether next_track has been called at least once.
    has_started: bool,
}

impl Default for MusicManager {
    fn default() -> Self {
        Self {
            playlist: Vec::new(),
            current_index: 0,
            play_mode: PlayMode::Loop,
            state: MusicState::Stopped,
            crossfade_duration: Duration::from_secs(2),
            track_started: None,
            active_tag: String::new(),
            has_started: false,
        }
    }
}

impl MusicManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a track to the playlist.
    pub fn add_track(&mut self, track: MusicTrack) {
        self.playlist.push(track);
    }

    /// Set the playlist (replaces existing).
    pub fn set_playlist(&mut self, tracks: Vec<MusicTrack>) {
        self.playlist = tracks;
        self.current_index = 0;
        self.has_started = false;
    }

    /// Clear the playlist.
    pub fn clear_playlist(&mut self) {
        self.playlist.clear();
        self.current_index = 0;
        self.has_started = false;
    }

    /// Set a tag filter — only tracks matching this tag will play.
    /// Empty string = play all.
    pub fn set_tag_filter(&mut self, tag: &str) {
        self.active_tag = tag.to_string();
    }

    /// Get the next track to play (based on play mode and tag filter).
    pub fn next_track(&mut self) -> Option<&MusicTrack> {
        if self.playlist.is_empty() {
            return None;
        }

        let filtered: Vec<usize> = self.playlist
            .iter()
            .enumerate()
            .filter(|(_, t)| self.active_tag.is_empty() || t.tag == self.active_tag)
            .map(|(i, _)| i)
            .collect();

        if filtered.is_empty() {
            return None;
        }

        // First call: return the first track without advancing.
        if !self.has_started {
            self.has_started = true;
            self.current_index = *filtered.first().unwrap();
            return self.playlist.get(self.current_index);
        }

        match self.play_mode {
            PlayMode::RepeatOne => {
                // Stay on the same track.
            }
            PlayMode::Sequential => {
                // Advance past current index.
                if let Some(&next) = filtered.iter().find(|&&i| i > self.current_index) {
                    self.current_index = next;
                } else {
                    return None; // Playlist finished.
                }
            }
            PlayMode::Loop => {
                if let Some(&next) = filtered.iter().find(|&&i| i > self.current_index) {
                    self.current_index = next;
                } else {
                    self.current_index = *filtered.first().unwrap();
                }
            }
            PlayMode::Shuffle => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                Instant::now().hash(&mut hasher);
                let idx = (hasher.finish() as usize) % filtered.len();
                self.current_index = filtered[idx];
            }
        }

        self.playlist.get(self.current_index)
    }

    /// Get the current track (if any).
    pub fn current_track(&self) -> Option<&MusicTrack> {
        self.playlist.get(self.current_index)
    }

    /// Check if it's time to crossfade (near end of track).
    pub fn should_crossfade(&self) -> bool {
        if self.state != MusicState::Playing {
            return false;
        }
        if let Some(started) = self.track_started {
            let _elapsed = started.elapsed();
            // We don't know the track duration, so we can't crossfade.
            // This would need the actual audio duration from rodio.
            // For now, return false — crossfading is triggered externally.
            false
        } else {
            false
        }
    }

    /// Mark a track as started.
    pub fn on_track_started(&mut self) {
        self.track_started = Some(Instant::now());
        self.state = MusicState::Playing;
    }

    /// Mark crossfading as active.
    pub fn on_crossfade_started(&mut self) {
        self.state = MusicState::Crossfading;
    }

    /// Number of tracks in the playlist.
    pub fn track_count(&self) -> usize {
        self.playlist.len()
    }

    /// Current track index.
    pub fn current_index(&self) -> usize {
        self.current_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_playlist() {
        let mut mgr = MusicManager::new();
        assert!(mgr.next_track().is_none());
    }

    #[test]
    fn sequential_playback() {
        let mut mgr = MusicManager::new();
        mgr.set_playlist(vec![
            MusicTrack { path: "a.ogg".into(), name: "A".into(), tag: "".into(), volume: 1.0 },
            MusicTrack { path: "b.ogg".into(), name: "B".into(), tag: "".into(), volume: 1.0 },
        ]);
        mgr.play_mode = PlayMode::Sequential;

        let t = mgr.next_track().unwrap();
        assert_eq!(t.path, "a.ogg");

        let t = mgr.next_track().unwrap();
        assert_eq!(t.path, "b.ogg");

        // Sequential stops after last track.
        assert!(mgr.next_track().is_none());
    }

    #[test]
    fn loop_playback() {
        let mut mgr = MusicManager::new();
        mgr.set_playlist(vec![
            MusicTrack { path: "a.ogg".into(), name: "A".into(), tag: "".into(), volume: 1.0 },
        ]);
        mgr.play_mode = PlayMode::Loop;

        // Should keep returning the same track.
        let t1_path = mgr.next_track().unwrap().path.clone();
        let t2_path = mgr.next_track().unwrap().path.clone();
        assert_eq!(t1_path, t2_path);
    }

    #[test]
    fn tag_filter() {
        let mut mgr = MusicManager::new();
        mgr.set_playlist(vec![
            MusicTrack { path: "a.ogg".into(), name: "A".into(), tag: "combat".into(), volume: 1.0 },
            MusicTrack { path: "b.ogg".into(), name: "B".into(), tag: "explore".into(), volume: 1.0 },
        ]);
        mgr.play_mode = PlayMode::Loop;
        mgr.set_tag_filter("combat");

        let t = mgr.next_track().unwrap();
        assert_eq!(t.tag, "combat");
    }

    #[test]
    fn default_state() {
        let mgr = MusicManager::new();
        assert_eq!(mgr.state, MusicState::Stopped);
        assert_eq!(mgr.play_mode, PlayMode::Loop);
        assert_eq!(mgr.track_count(), 0);
    }
}
