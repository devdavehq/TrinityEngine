// src/editor_ui/panels/script_editor.rs
// ──────────────────────────────────────────────────────────────────────────────
// In-engine Lua script editor dock tab.
//
// Features:
//   • File tree (recursive) with script + folder icons, filtered by search.
//   • Folder picker — defaults to the project's Content/Scripts folder, but the
//     user can switch to any folder (quick jumps, editable path, native dialog).
//   • Create files and folders directly from the editor.
//   • Debounced AUTO-SAVE: edits are written to disk ~0.8 s after you stop
//     typing, so nothing is ever lost. Ctrl+S forces an immediate save.
//   • Autocomplete / recommendations tailored to the engine's Lua API: it uses
//     ScriptEngine::api_catalogue() (every flat global + `ns.fn` namespaced
//     function) plus Lua keywords and the Lua stdlib.
//   • Enter / Tab completes the highlighted suggestion (generates the code for
//     you); Arrow keys navigate, Esc dismisses.
//   • Opening a script from the Content Browser (double-click) loads it here.
//
// State is stored in the egui context so it survives frames/tab switches and is
// shared between the dock tab and any floating copies.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::editor_ui::UiFrameArgs;
use egui::{Color32, RichText};

const STATE_ID: &str = "script_editor_state_v1";
const OPEN_REQUEST_ID: &str = "script_editor_open";

// The words we complete against. Engine API names come from the live
// ScriptEngine catalogue so suggestions always match what a script can actually
// call.
const LUA_KEYWORDS: &[&str] = &[
    "function", "end", "if", "then", "else", "elseif", "for", "while", "do",
    "return", "local", "nil", "true", "false", "not", "and", "or", "repeat",
    "until", "break", "in", "self",
];

// Common Lua stdlib globals + members most scripts reach for.
const LUA_STDLIB: &[&str] = &[
    "print", "require", "tonumber", "tostring", "pairs", "ipairs", "type",
    "select", "next", "pcall", "xpcall", "assert", "error", "setmetatable",
    "getmetatable", "unpack", "math", "math.sin", "math.cos", "math.sqrt",
    "math.floor", "math.ceil", "math.abs", "math.max", "math.min", "math.pi",
    "math.random", "string", "string.format", "string.sub", "string.len",
    "string.find", "string.gsub", "string.lower", "string.upper", "string.rep",
    "table", "table.insert", "table.remove", "table.concat", "table.sort",
    "table.unpack", "os", "os.time", "os.clock", "os.date", "io", "io.open",
    "io.write", "io.read", "coroutine",
];

const NEW_FILE_TEMPLATE: &str = "function update(entity, dt)\n    -- TODO: your logic\nend\n";

#[derive(Clone)]
struct ScriptEditorState {
    /// Folder currently being browsed / edited (absolute or relative path).
    root_dir: String,
    /// Absolute path of the open file, if any.
    current_file: Option<String>,
    /// Editor buffer for the open file.
    buffer: String,
    dirty: bool,
    /// egui time (seconds) of the last edit — drives the autosave debounce.
    last_edit_time: f64,
    last_saved_time: f64,
    /// Search filter for the file tree.
    search: String,
    /// New-file / new-folder name inputs.
    new_file_name: String,
    new_folder_name: String,
    /// Live autocomplete state.
    suggestions: Vec<String>,
    selected_suggestion: usize,
    /// True while the completion popup should be visible.
    show_suggestions: bool,
    /// Screen-space anchor for the popup, recomputed each frame.
    popup_pos: egui::Pos2,
    /// Cursor (char index) + token captured when suggestions were computed, so
    /// the completion popup can replace exactly the typed token.
    cursor_char: usize,
    token: String,
}

impl Default for ScriptEditorState {
    fn default() -> Self {
        Self {
            root_dir: "Content/Scripts".to_string(),
            current_file: None,
            buffer: String::new(),
            dirty: false,
            last_edit_time: 0.0,
            last_saved_time: 0.0,
            search: String::new(),
            new_file_name: String::new(),
            new_folder_name: String::new(),
            suggestions: Vec::new(),
            selected_suggestion: 0,
            show_suggestions: false,
            popup_pos: egui::Pos2::ZERO,
            cursor_char: 0,
            token: String::new(),
        }
    }
}

/// Render the Script Editor dock tab.
pub fn render_script_editor_panel(
    ui: &mut egui::Ui,
    args: &mut UiFrameArgs<'_>,
    icon_texture_cache: &HashMap<String, egui::TextureHandle>,
) {
    let state_id = egui::Id::new(STATE_ID);
    let mut state: ScriptEditorState = ui
        .ctx()
        .data_mut(|d| d.get_temp(state_id))
        .unwrap_or_default();

    // If the Content Browser asked us to open a script, load it now.
    let open_request: Option<String> = ui
        .ctx()
        .data_mut(|d| d.get_temp(egui::Id::new(OPEN_REQUEST_ID)));
    if let Some(path) = open_request {
        ui.ctx()
            .data_mut(|d| d.remove_temp::<String>(egui::Id::new(OPEN_REQUEST_ID)));
        open_file(&mut state, args, &path);
    }

    let now = ui.input(|i| i.time);
    egui::Frame::new()
        .fill(Color32::from_rgb(14, 17, 22))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(33, 39, 49)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Folder").small().strong().color(Color32::from_rgb(168, 176, 188)));
                // Quick-jump to common folders.
                egui::ComboBox::from_id_salt("script_editor_folder_jump")
                    .selected_text(
                        if state.root_dir == "Content/Scripts" {
                            "Scripts"
                        } else if state.root_dir == "Content" {
                            "Content"
                        } else if state.root_dir == "Content/Scripts/plugins" {
                            "Plugins"
                        } else {
                            "…"
                        },
                    )
                    .width(90.0)
                    .show_ui(ui, |ui| {
                        if ui.button("Scripts").clicked() {
                            state.root_dir = "Content/Scripts".to_string();
                            ui.close();
                        }
                        if ui.button("Plugins").clicked() {
                            state.root_dir = "Content/Scripts/plugins".to_string();
                            ui.close();
                        }
                        if ui.button("Content").clicked() {
                            state.root_dir = "Content".to_string();
                            ui.close();
                        }
                    });
                ui.add(
                    egui::TextEdit::singleline(&mut state.root_dir)
                        .desired_width(220.0)
                        .hint_text("path (e.g. Content/Scripts)"),
                );
                if ui.button("Browse…").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        state.root_dir = dir.to_string_lossy().to_string();
                    }
                }
                ui.separator();
                ui.label(RichText::new("New file").small().color(Color32::from_rgb(147, 158, 172)));
                ui.add(
                    egui::TextEdit::singleline(&mut state.new_file_name)
                        .desired_width(140.0)
                        .hint_text("my_script.lua"),
                );
                if ui.button("Create").clicked() {
                    create_file(&mut state, args);
                }
                ui.separator();
                ui.label(RichText::new("New folder").small().color(Color32::from_rgb(147, 158, 172)));
                ui.add(
                    egui::TextEdit::singleline(&mut state.new_folder_name)
                        .desired_width(120.0)
                        .hint_text("folder name"),
                );
                if ui.button("Make").clicked() {
                    create_folder(&mut state, args);
                }
            });
        });
    ui.add_space(6.0);

    // ── Body: file tree (left) + editor (right) ───────────────────────────
    ui.columns(2, |cols| {
        // Left column — file tree.
        egui::Frame::new()
            .fill(Color32::from_rgb(12, 15, 20))
            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(32, 38, 48)))
            .corner_radius(8.0)
            .inner_margin(egui::Margin::same(6))
            .show(&mut cols[0], |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Files").small().strong().color(Color32::from_rgb(168, 176, 188)));
                    ui.add(
                        egui::TextEdit::singleline(&mut state.search)
                            .desired_width(120.0)
                            .hint_text("filter…"),
                    );
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let query = state.search.to_ascii_lowercase();
                        let root_dir = state.root_dir.clone();
                        draw_tree(
                            ui,
                            Path::new(&root_dir),
                            &query,
                            icon_texture_cache,
                            &mut state,
                            args,
                        );
                    });
            });

        // Right column — code editor.
        egui::Frame::new()
            .fill(Color32::from_rgb(12, 15, 20))
            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(32, 38, 48)))
            .corner_radius(8.0)
            .inner_margin(egui::Margin::same(6))
            .show(&mut cols[1], |ui| {
                draw_editor(ui, args, &mut state, now);
            });
    });

    // Persist the panel state (buffer, open file, folder, autocomplete…) so it
    // survives frames and tab switches.
    ui.ctx().data_mut(|d| d.insert_temp(state_id, state));
}

// ── File tree ─────────────────────────────────────────────────────────────────

fn draw_tree(
    ui: &mut egui::Ui,
    dir: &Path,
    query: &str,
    icon_texture_cache: &HashMap<String, egui::TextureHandle>,
    state: &mut ScriptEditorState,
    args: &mut UiFrameArgs<'_>,
) {
    let Ok(rd) = fs::read_dir(dir) else {
        ui.colored_label(Color32::from_rgb(235, 170, 120), "Folder not found");
        return;
    };

    // Folders first, then files; sorted by name.
    let mut folders: Vec<PathBuf> = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            folders.push(p);
        } else if p.extension().map(|x| x == "lua").unwrap_or(false) {
            files.push(p);
        }
    }
    folders.sort();
    files.sort();

    for folder in &folders {
        let name = folder
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let matches = query.is_empty() || name.to_ascii_lowercase().contains(query);
        egui::CollapsingHeader::new(name)
            .default_open(query.is_empty())
            .show(ui, |ui| {
                if matches || query.is_empty() {
                    draw_tree(ui, folder, query, icon_texture_cache, state, args);
                }
            });
    }

    for file in &files {
        let name = file
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if !query.is_empty() && !name.to_ascii_lowercase().contains(query) {
            continue;
        }
        let path = file.to_string_lossy().to_string();
        let is_open = state.current_file.as_deref() == Some(path.as_str());
        let row = ui.horizontal(|ui| {
            icon(ui, icon_texture_cache, "script");
            ui.selectable_label(is_open, name)
        });
        // Double-click also opens the file.
        if row.inner.clicked() || row.inner.double_clicked() {
            open_file(state, args, &path);
        }
    }
}

fn icon(ui: &mut egui::Ui, cache: &HashMap<String, egui::TextureHandle>, stem: &str) {
    if let Some(tex) = cache.get(stem) {
        ui.add(egui::Image::new((tex.id(), egui::vec2(14.0, 14.0))));
    } else {
        let color = match stem {
            "script" => Color32::from_rgb(126, 100, 168),
            _ => Color32::from_rgb(90, 124, 148),
        };
        ui.colored_label(color, "■");
    }
}

// ── Editor ────────────────────────────────────────────────────────────────────

fn draw_editor(
    ui: &mut egui::Ui,
    args: &mut UiFrameArgs<'_>,
    state: &mut ScriptEditorState,
    now: f64,
) {
    // Header row: file name, dirty/saved indicator, action buttons.
    ui.horizontal(|ui| {
        match &state.current_file {
            Some(path) => {
                let name = Path::new(path)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                ui.label(RichText::new(name).strong().color(Color32::from_rgb(229, 232, 238)));
                ui.separator();
                ui.label(
                    RichText::new(path)
                        .small()
                        .color(Color32::from_rgb(130, 140, 155)),
                );
            }
            None => {
                ui.label(
                    RichText::new("No file open — pick one from the tree or create a new one.")
                        .italics()
                        .color(Color32::from_rgb(120, 130, 145)),
                );
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if state.dirty {
                ui.colored_label(Color32::from_rgb(230, 180, 90), "● Unsaved");
            } else if state.current_file.is_some() {
                let secs = state.last_saved_time as i64;
                ui.colored_label(
                    Color32::from_rgb(120, 200, 150),
                    format!("✓ Saved {:02}:{:02}:{:02}", secs / 3600, (secs / 60) % 60, secs % 60),
                );
            }
            if ui.button("Save").clicked() {
                save_file(state, args);
            }
            if ui.button("Save + Reload").clicked() {
                save_file(state, args);
                if let Some(path) = state.current_file.clone() {
                    if let Err(err) = args.scripts.reload_script(&path) {
                        args.error_log.push(format!("[Script] Reload error: {}", err));
                    }
                }
            }
            if state.current_file.is_some() {
                if let Some(entity) = args.selected_renderable.as_ref().copied() {
                    if ui.button("Attach To Selected").clicked() {
                        if let Some(path) = state.current_file.clone() {
                            let _ = args
                                .world
                                .insert(entity, (crate::components::Script { path },));
                        }
                    }
                }
            }
        });
    });
    ui.separator();

    if state.current_file.is_none() {
        return;
    }

    // ── Autocomplete input plumbing ─────────────────────────────────────
    // When the completion popup is showing, consume Enter/Tab to complete and
    // Arrow keys to move, so the TextEdit never sees them.
    if state.show_suggestions && !state.suggestions.is_empty() {
        let mut nav = None;
        let mut accept = false;
        let mut dismiss = false;
        ui.input_mut(|i| {
            if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown) {
                nav = Some(1usize);
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp) {
                nav = Some(usize::MAX); // sentinel for -1
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
                || i.consume_key(egui::Modifiers::NONE, egui::Key::Tab)
            {
                accept = true;
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::Escape) {
                dismiss = true;
            }
        });
        if dismiss {
            state.show_suggestions = false;
        }
        if let Some(dir) = nav {
            let n = state.suggestions.len();
            if n > 0 {
                let step = if dir == usize::MAX { n.saturating_sub(1) } else { 1 };
                state.selected_suggestion =
                    (state.selected_suggestion + step) % n;
            }
        }
        if accept && !state.suggestions.is_empty() {
            let chosen = state.suggestions[state.selected_suggestion].clone();
            state.show_suggestions = false;
            apply_completion(state, &chosen);
            state.dirty = true;
            state.last_edit_time = now;
        }
    }

    // Ctrl+S forces an immediate save.
    ui.input_mut(|i| {
        if i.modifiers.command && i.key_pressed(egui::Key::S) {
            i.consume_key(egui::Modifiers::COMMAND, egui::Key::S);
            save_file(state, args);
        }
    });

    // Ctrl+Space forces the full API list (even with an empty token) so you can
    // always browse the complete engine surface, not just what you've typed.
    let mut force_complete = false;
    ui.input_mut(|i| {
        if i.modifiers.command && i.key_pressed(egui::Key::Space) {
            i.consume_key(egui::Modifiers::COMMAND, egui::Key::Space);
            force_complete = true;
        }
    });

    // ── The code editor ─────────────────────────────────────────────────
    let text_edit = egui::TextEdit::multiline(&mut state.buffer)
        .code_editor()
        .desired_rows(24)
        .desired_width(f32::INFINITY);
    let output = text_edit.show(ui);

    if output.response.changed() {
        state.dirty = true;
        state.last_edit_time = now;
    }

    let has_focus = output.response.has_focus();

    // Recompute suggestions from the token under the cursor.
    let api_names = args.scripts.api_catalogue();
    if has_focus {
        if let Some(range) = output.cursor_range {
            let token = token_at_cursor(&state.buffer, range.primary.index);
            state.cursor_char = range.primary.index;
            state.token = token.clone();
            if token.is_empty() && !force_complete {
                state.show_suggestions = false;
            } else {
                let mut s = if token.is_empty() {
                    // Ctrl+Space: browse the entire engine API (flat globals +
                    // namespaced members + component names), all of it.
                    let mut all: Vec<String> = api_names.clone();
                    all.sort();
                    all.dedup();
                    all
                } else {
                    compute_suggestions(&token, &api_names)
                };
                // Keep a stable selection index within range.
                if state.selected_suggestion >= s.len() {
                    state.selected_suggestion = 0;
                }
                s.truncate(40);
                state.suggestions = s;
                state.show_suggestions = !state.suggestions.is_empty();
                if state.show_suggestions {
                    // pos_from_cursor returns galley-local coords (relative to
                    // the galley's top-left), so galley_pos + that offset gives
                    // the on-screen position.
                    let cursor_rect = output.galley.pos_from_cursor(range.primary);
                    state.popup_pos =
                        output.galley_pos + cursor_rect.min.to_vec2() + egui::vec2(0.0, 6.0);
                }
            }
        }
    } else {
        state.show_suggestions = false;
    }

    // Render the completion popup (screen space, drawn on top).
    if state.show_suggestions && !state.suggestions.is_empty() {
        let popup_pos = state.popup_pos;
        let sel = state.selected_suggestion;
        let suggestions = state.suggestions.clone();
        let ctx = ui.ctx().clone();
        let mut completed = None;
        egui::Area::new(egui::Id::new("script_editor_completion_popup"))
            .fixed_pos(popup_pos)
            .order(egui::Order::Foreground)
            .show(&ctx, |ui| {
                egui::Frame::new()
                    .fill(Color32::from_rgb(28, 32, 40))
                    .stroke(egui::Stroke::new(1.0, Color32::from_rgb(70, 82, 100)))
                    .corner_radius(6.0)
                    .inner_margin(egui::Margin::symmetric(4, 4))
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .max_height(280.0)
                            .auto_shrink([true, true])
                            .show(ui, |ui| {
                                for (i, s) in suggestions.iter().enumerate() {
                                    let is_sel = i == sel;
                                    let text = if is_sel {
                                        RichText::new(s).monospace().color(Color32::from_rgb(20, 22, 28))
                                    } else {
                                        RichText::new(s).monospace().color(Color32::from_rgb(210, 216, 226))
                                    };
                                    if ui
                                        .add(egui::Button::new(text).fill(if is_sel {
                                            Color32::from_rgb(120, 160, 230)
                                        } else {
                                            Color32::TRANSPARENT
                                        }))
                                        .clicked()
                                    {
                                        completed = Some(s.clone());
                                    }
                                }
                            });
                        ui.label(
                            RichText::new("Enter = complete  •  Esc = dismiss  •  Ctrl+Space = full API")
                                .small()
                                .weak(),
                        );
                    });
            });
        if let Some(s) = completed {
            state.show_suggestions = false;
            apply_completion(state, &s);
            state.dirty = true;
            state.last_edit_time = now;
        }
    }

    // ── Debounced autosave ──────────────────────────────────────────────
    if state.dirty && state.current_file.is_some() && now - state.last_edit_time > 0.8 {
        save_file(state, args);
    }
}

// ── File / folder operations ─────────────────────────────────────────────────

fn open_file(state: &mut ScriptEditorState, args: &mut UiFrameArgs<'_>, path: &str) {
    // If the user had unsaved edits, save them before switching.
    if state.dirty && state.current_file.is_some() {
        save_file(state, args);
    }
    match fs::read_to_string(path) {
        Ok(contents) => {
            state.current_file = Some(path.to_string());
            state.buffer = contents;
            state.dirty = false;
            state.last_edit_time = 0.0;
            if let Some(dir) = Path::new(path).parent() {
                state.root_dir = dir.to_string_lossy().to_string();
            }
        }
        Err(err) => {
            args.error_log
                .push(format!("[Script Editor] Open failed ({}): {}", path, err));
        }
    }
}

fn save_file(state: &mut ScriptEditorState, args: &mut UiFrameArgs<'_>) {
    let Some(path) = state.current_file.clone() else {
        return;
    };
    match fs::write(&path, state.buffer.as_bytes()) {
        Ok(()) => {
            state.dirty = false;
            state.last_saved_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            args.error_log.push(format!("[Script Editor] Saved {}", path));
        }
        Err(err) => {
            args.error_log
                .push(format!("[Script Editor] Save failed ({}): {}", path, err));
        }
    }
}

fn create_file(state: &mut ScriptEditorState, args: &mut UiFrameArgs<'_>) {
    let name = state.new_file_name.trim().to_string();
    if name.is_empty() {
        args.error_log.push("[Script Editor] No file name given".to_string());
        return;
    }
    let mut name = name;
    if !name.ends_with(".lua") {
        name.push_str(".lua");
    }
    let path = Path::new(&state.root_dir).join(&name);
    if path.exists() {
        args.error_log
            .push(format!("[Script Editor] {} already exists", path.display()));
        return;
    }
    match fs::write(&path, NEW_FILE_TEMPLATE) {
        Ok(()) => {
            args.error_log
                .push(format!("[Script Editor] Created {}", path.display()));
            state.new_file_name.clear();
            // Open the freshly-created file so the user can edit it immediately.
            open_file(state, args, &path.to_string_lossy());
        }
        Err(err) => {
            args.error_log
                .push(format!("[Script Editor] Create failed ({}): {}", path.display(), err));
        }
    }
}

fn create_folder(state: &mut ScriptEditorState, args: &mut UiFrameArgs<'_>) {
    let name = state.new_folder_name.trim().to_string();
    if name.is_empty() {
        args.error_log.push("[Script Editor] No folder name given".to_string());
        return;
    }
    let path = Path::new(&state.root_dir).join(&name);
    match fs::create_dir_all(&path) {
        Ok(()) => {
            args.error_log
                .push(format!("[Script Editor] Created folder {}", path.display()));
            state.new_folder_name.clear();
        }
        Err(err) => {
            args.error_log
                .push(format!("[Script Editor] Folder create failed ({}): {}", path.display(), err));
        }
    }
}

// ── Autocomplete helpers ──────────────────────────────────────────────────────

/// Convert a char index to a byte offset (egui cursors are char-based).
fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

/// The identifier (with dots for namespaces) under the cursor.
fn token_at_cursor(buffer: &str, char_idx: usize) -> String {
    let byte_idx = char_to_byte(buffer, char_idx.min(buffer.chars().count()));
    let before = &buffer[..byte_idx];
    let start = before
        .rfind(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
        .map(|i| i + 1)
        .unwrap_or(0);
    before[start..].to_string()
}

fn compute_suggestions(token: &str, api_names: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let lower = token.to_ascii_lowercase();

    // Namespaced: "ns.prefix" — complete members of that namespace.
    if let Some(dot) = token.find('.') {
        let ns = &token[..dot];
        let member = &token[dot + 1..];
        for name in api_names {
            if name.starts_with(ns)
                && name.len() > dot
                && name.as_bytes()[dot] == b'.'
            {
                let member_part = &name[dot + 1..];
                if member_part.to_ascii_lowercase().starts_with(&member.to_ascii_lowercase())
                {
                    out.push(name.clone());
                }
            }
        }
        return out;
    }

    // Flat names + Lua keywords + stdlib.
    for name in api_names {
        if !name.contains('.') && name.to_ascii_lowercase().starts_with(&lower) {
            out.push(name.clone());
        }
    }
    for kw in LUA_KEYWORDS {
        if kw.to_ascii_lowercase().starts_with(&lower) {
            out.push((*kw).to_string());
        }
    }
    for s in LUA_STDLIB {
        if s.to_ascii_lowercase().starts_with(&lower) {
            out.push((*s).to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Replace the token under the cursor (captured while suggestions were shown)
/// with the chosen suggestion.
fn apply_completion(state: &mut ScriptEditorState, suggestion: &str) {
    let byte_idx = char_to_byte(&state.buffer, state.cursor_char.min(state.buffer.chars().count()));
    let before = &state.buffer[..byte_idx];
    let start = before
        .rfind(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
        .map(|i| i + 1)
        .unwrap_or(0);
    state
        .buffer
        .replace_range(start..byte_idx, suggestion);
    // Move the cursor to the end of the completed token.
    state.cursor_char = state.buffer[..start + suggestion.len()].chars().count();
}