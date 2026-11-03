//! Prompt packs: named, swappable overrides of the compression surface.
//!
//! The built-in prompts are tuned for one shape of work. A team that compresses
//! mostly logs wants different rules than one compressing design discussion, and
//! neither wants to fork the binary. A pack is a name plus a set of overrides;
//! sources are consulted in order and the first hit wins, so a host composes
//! its own chain — project directory, then user directory, then built-in.
//!
//! Sanitising is the whole safety story. A pack is user-authored JSON on disk,
//! so every field is optional and any value of the wrong shape is **dropped**,
//! not coerced: a bad override must never clobber a good default. A missing
//! `how_to_compress_rules` means "use the built-in", never "use an empty
//! string" — an empty rules block would silently produce summaries that drop
//! exactly what the rules exist to preserve.
//!
//! Identity is the *filename*, never the `name` field inside the file. A config
//! that says `prompt_pack: "team"` must keep resolving to `team.json` even
//! after someone edits the file's internal name.

use std::path::{Path, PathBuf};

use crate::agent::bigctx::rules::Prompts;

/// A sanitised set of overrides. Every field is "an override, or absent".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackSurface {
    /// Individual rule texts. `None` means keep the built-in.
    pub prompts: PromptOverrides,
    /// Tool description overrides, keyed by tool name.
    pub tool_prompts: std::collections::BTreeMap<String, ToolPromptOverride>,
}

/// One prompt text override per rule. `None` = use the built-in.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PromptOverrides {
    pub compress_philosophy: Option<String>,
    pub how_to_compress_rules: Option<String>,
    pub tier2_distill_rules: Option<String>,
    pub tier3_condense_rules: Option<String>,
}

impl PromptOverrides {
    pub fn is_empty(&self) -> bool {
        self.compress_philosophy.is_none()
            && self.how_to_compress_rules.is_none()
            && self.tier2_distill_rules.is_none()
            && self.tier3_condense_rules.is_none()
    }

    /// Overlay these onto `base`, keeping every field this pack does not set.
    pub fn apply_to(&self, base: &Prompts) -> Prompts {
        use std::borrow::Cow;
        let pick = |over: &Option<String>, default: &Cow<'static, str>| match over {
            Some(text) => Cow::Owned(text.clone()),
            None => default.clone(),
        };
        Prompts {
            compress_philosophy: pick(&self.compress_philosophy, &base.compress_philosophy),
            how_to_compress_rules: pick(&self.how_to_compress_rules, &base.how_to_compress_rules),
            tier2_distill_rules: pick(&self.tier2_distill_rules, &base.tier2_distill_rules),
            tier3_condense_rules: pick(&self.tier3_condense_rules, &base.tier3_condense_rules),
        }
    }
}

/// A tool's description override.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolPromptOverride {
    pub description: Option<String>,
    /// Per-parameter descriptions, keyed by parameter name.
    pub param_descriptions: std::collections::BTreeMap<String, String>,
}

/// A resolved pack: a name, its sanitised surface, and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Pack {
    pub name: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub surface: PackSurface,
    /// Provenance, e.g. `builtin:default` or `file:/home/u/.config/yak/packs/x.json`.
    /// Carried so a user asking "which pack am I running?" gets an answer that
    /// points at a file they can edit.
    pub source: String,
}

/// A place packs can come from.
pub trait PackSource: Send + Sync {
    fn id(&self) -> &str;
    fn resolve(&self, name: &str) -> Option<Pack>;
    fn list(&self) -> Vec<Pack>;
}

/// A pack name is safe to turn into a filename.
///
/// Rejects `..` and anything outside a conservative alphabet because the name is
/// concatenated into a path: `../secrets` must never become `<dir>/../secrets.json`.
pub fn is_valid_pack_name(name: &str) -> bool {
    if name.is_empty() || name.contains("..") {
        return false;
    }
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphanumeric() {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

fn sanitize_prompts(raw: Option<&serde_json::Value>) -> PromptOverrides {
    let mut out = PromptOverrides::default();
    let Some(object) = raw.and_then(serde_json::Value::as_object) else {
        return out;
    };
    let text = |key: &str| {
        object
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    };
    out.compress_philosophy = text("compress_philosophy");
    out.how_to_compress_rules = text("how_to_compress_rules");
    out.tier2_distill_rules = text("tier2_distill_rules");
    out.tier3_condense_rules = text("tier3_condense_rules");
    out
}

fn sanitize_tool_prompts(
    raw: Option<&serde_json::Value>,
) -> std::collections::BTreeMap<String, ToolPromptOverride> {
    let mut out = std::collections::BTreeMap::new();
    let Some(object) = raw.and_then(serde_json::Value::as_object) else {
        return out;
    };
    for (name, value) in object {
        let Some(entry) = value.as_object() else {
            continue;
        };
        let mut over = ToolPromptOverride::default();
        if let Some(desc) = entry.get("description").and_then(serde_json::Value::as_str) {
            over.description = Some(desc.to_string());
        }
        if let Some(params) = entry
            .get("param_descriptions")
            .and_then(serde_json::Value::as_object)
        {
            for (param, desc) in params {
                if let Some(text) = desc.as_str() {
                    over.param_descriptions
                        .insert(param.clone(), text.to_string());
                }
            }
        }
        // An override that sets nothing is a typo, not an instruction to
        // blank the description.
        if over.description.is_some() || !over.param_descriptions.is_empty() {
            out.insert(name.clone(), over);
        }
    }
    out
}

/// Sanitise a raw pack file body. Malformed values are dropped, never coerced.
pub fn sanitize(raw: Option<&serde_json::Value>) -> PackSurface {
    let Some(object) = raw.and_then(serde_json::Value::as_object) else {
        return PackSurface::default();
    };
    PackSurface {
        prompts: sanitize_prompts(object.get("prompts")),
        tool_prompts: sanitize_tool_prompts(object.get("tool_prompts")),
    }
}

/// The identity pack: built-in defaults everywhere.
pub fn default_pack() -> Pack {
    Pack {
        name: "default".into(),
        version: Some("1.0.0".into()),
        description: Some("Built-in defaults (no overrides).".into()),
        surface: PackSurface::default(),
        source: "builtin:default".into(),
    }
}

/// A built-in registry.
pub struct BuiltinSource;

impl PackSource for BuiltinSource {
    fn id(&self) -> &str {
        "builtin"
    }
    fn resolve(&self, name: &str) -> Option<Pack> {
        match name {
            "default" => Some(default_pack()),
            _ => None,
        }
    }
    fn list(&self) -> Vec<Pack> {
        vec![default_pack()]
    }
}

/// A directory of `<name>.json` packs.
pub struct DirSource {
    id: String,
    dir: PathBuf,
}

impl DirSource {
    pub fn new(id: impl Into<String>, dir: impl Into<PathBuf>) -> DirSource {
        DirSource {
            id: id.into(),
            dir: dir.into(),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

fn read_pack_file(path: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&text).ok()?;
    // A pack body must be an object. Returning early keeps the value owned, so
    // the caller's `get(..)` lookups do not borrow across a move.
    if !parsed.is_object() {
        return None;
    }
    Some(parsed)
}

impl PackSource for DirSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn resolve(&self, name: &str) -> Option<Pack> {
        if !is_valid_pack_name(name) {
            return None;
        }
        let path = self.dir.join(format!("{name}.json"));
        let raw = read_pack_file(&path)?;
        let text = |key: &str| {
            raw.get(key)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        };
        Some(Pack {
            // The filename is canonical; the file's internal `name` is ignored.
            name: name.to_string(),
            version: text("version"),
            description: text("description"),
            surface: sanitize(Some(&raw)),
            source: format!("file:{}", path.display()),
        })
    }

    fn list(&self) -> Vec<Pack> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if let Some(pack) = self.resolve(stem) {
                out.push(pack);
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }
}

/// Ordered resolution: the first source that has the name wins.
pub struct Resolver {
    sources: Vec<Box<dyn PackSource>>,
}

impl Resolver {
    pub fn new(sources: Vec<Box<dyn PackSource>>) -> Resolver {
        Resolver { sources }
    }

    /// The default chain: project, then user, then built-in.
    pub fn with_dirs(project: Option<PathBuf>, user: Vec<PathBuf>) -> Resolver {
        let mut sources: Vec<Box<dyn PackSource>> = Vec::new();
        if let Some(dir) = project {
            sources.push(Box::new(DirSource::new("project", dir)));
        }
        for (i, dir) in user.into_iter().enumerate() {
            sources.push(Box::new(DirSource::new(format!("user{i}"), dir)));
        }
        sources.push(Box::new(BuiltinSource));
        Resolver::new(sources)
    }

    pub fn resolve(&self, name: &str) -> Option<Pack> {
        if !is_valid_pack_name(name) {
            return None;
        }
        self.sources.iter().find_map(|s| s.resolve(name))
    }

    /// Every pack, first-wins by name so a project override shadows the builtin.
    pub fn list(&self) -> Vec<Pack> {
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        for source in &self.sources {
            for pack in source.list() {
                if seen.insert(pack.name.clone()) {
                    out.push(pack);
                }
            }
        }
        out
    }

    pub fn source_ids(&self) -> Vec<&str> {
        self.sources.iter().map(|s| s.id()).collect()
    }
}

/// The standard source chain for this process: the project's `.yak/packs`, the
/// user's `~/.yak/packs`, then the built-ins.
///
/// Project first so a repository can pin its own conventions over the user's
/// global choice — the more specific scope wins, which is what a developer
/// expects when they check a pack into the repo.
///
/// The directories are not created here. A missing packs directory is the
/// normal case for everyone who has never made one, and creating it on every
/// session would leave an empty directory behind for a feature they may never
/// use.
pub fn default_resolver(cwd: &Path) -> Resolver {
    Resolver::with_dirs(
        Some(cwd.join(".yak").join("packs")),
        vec![crate::core::config::user_dir().join("packs")],
    )
}

/// Resolve a named pack into a usable prompt set over `base`.
///
/// An unknown name falls back to the built-in defaults rather than failing: a
/// typo in a config file should cost a warning, not a session that cannot
/// compress.
pub fn resolve_prompts(resolver: &Resolver, name: Option<&str>) -> (Prompts, Option<String>) {
    let base = crate::agent::bigctx::rules::default_prompts();
    let Some(name) = name.filter(|n| !n.trim().is_empty()) else {
        return (base, None);
    };
    match resolver.resolve(name.trim()) {
        Some(pack) => (pack.surface.prompts.apply_to(&base), None),
        None => (
            base,
            Some(format!(
                "prompt pack `{name}` was not found in any source ({}) — using built-in defaults",
                resolver.source_ids().join(", ")
            )),
        ),
    }
}

/// Fill in `config.prompts` from `config.prompt_pack`.
///
/// Called once, where a config is built with filesystem access. Returns any
/// warning to show the user — an unresolvable pack name falls back to the
/// built-ins rather than failing, because a typo should not leave a session
/// unable to compress.
pub fn apply_prompt_pack(config: &mut crate::agent::bigctx::config::BigctxConfig, cwd: &Path) -> Option<String> {
    let Some(name) = config.prompt_pack.clone() else {
        return None;
    };
    let resolver = default_resolver(cwd);
    let (prompts, warning) = resolve_prompts(&resolver, Some(&name));
    config.prompts = Some(prompts);
    // The tool descriptions travel with the rule texts: a pack that sets one
    // and not the other would be applied halfway, which reads as a broken file
    // rather than a partial override.
    if let Some(pack) = resolver.resolve(&name) {
        config.tool_prompts = pack.surface.tool_prompts;
    }
    warning
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "yak-pack-test-{tag}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("tmpdir");
        dir
    }

    // ── name validation ─────────────────────────────────────────────────────

    #[test]
    fn ordinary_names_are_valid() {
        for name in ["default", "lean", "team", "a", "my-pack_2.0"] {
            assert!(is_valid_pack_name(name), "{name} must be valid");
        }
    }

    #[test]
    fn a_traversal_name_is_rejected() {
        // The name becomes a filename: this is the one that matters.
        for name in ["../secrets", "a/../b", "..", "dir/name", "/etc/passwd"] {
            assert!(!is_valid_pack_name(name), "{name} must be rejected");
        }
    }

    #[test]
    fn a_name_starting_with_punctuation_is_rejected() {
        for name in ["-x", ".x", "_x"] {
            assert!(!is_valid_pack_name(name), "{name} must be rejected");
        }
    }

    #[test]
    fn an_empty_name_is_rejected() {
        assert!(!is_valid_pack_name(""));
    }

    // ── sanitising ──────────────────────────────────────────────────────────

    #[test]
    fn a_well_formed_pack_sanitises_cleanly() {
        let raw = json!({
            "prompts": {"how_to_compress_rules": "custom rules"},
            "tool_prompts": {"compress": {"description": "d", "param_descriptions": {"content": "c"}}}
        });
        let surface = sanitize(Some(&raw));
        assert_eq!(
            surface.prompts.how_to_compress_rules.as_deref(),
            Some("custom rules")
        );
        assert_eq!(
            surface.tool_prompts["compress"].description.as_deref(),
            Some("d")
        );
        assert_eq!(
            surface.tool_prompts["compress"].param_descriptions["content"],
            "c"
        );
    }

    #[test]
    fn a_bad_override_is_dropped_not_coerced() {
        // A number where a string belongs must not blank the built-in.
        let raw = json!({"prompts": {"how_to_compress_rules": 42}});
        let surface = sanitize(Some(&raw));
        assert!(surface.prompts.how_to_compress_rules.is_none());
    }

    #[test]
    fn a_non_object_pack_body_yields_no_overrides() {
        for raw in [json!("string"), json!(42), json!([1, 2])] {
            let surface = sanitize(Some(&raw));
            assert_eq!(surface, PackSurface::default());
        }
        assert_eq!(sanitize(None), PackSurface::default());
    }

    #[test]
    fn an_empty_tool_override_is_not_recorded() {
        // `{"compress": {}}` is a typo, not an instruction to blank the text.
        let raw = json!({"tool_prompts": {"compress": {}}});
        assert!(sanitize(Some(&raw)).tool_prompts.is_empty());
    }

    #[test]
    fn an_array_where_an_object_belongs_is_ignored() {
        let raw = json!({"prompts": ["nope"], "tool_prompts": ["nope"]});
        assert_eq!(sanitize(Some(&raw)), PackSurface::default());
    }

    // ── overlay ─────────────────────────────────────────────────────────────

    #[test]
    fn an_override_replaces_only_what_it_sets() {
        let base = crate::agent::bigctx::rules::default_prompts();
        let over = PromptOverrides {
            how_to_compress_rules: Some("replaced".into()),
            ..Default::default()
        };
        let merged = over.apply_to(&base);
        assert_eq!(&*merged.how_to_compress_rules, "replaced");
        assert_eq!(
            merged.compress_philosophy,
            base.compress_philosophy,
            "unset fields must keep the built-in"
        );
        assert_eq!(merged.tier2_distill_rules, base.tier2_distill_rules);
    }

    #[test]
    fn an_empty_overlay_changes_nothing() {
        let base = crate::agent::bigctx::rules::default_prompts();
        let merged = PromptOverrides::default().apply_to(&base);
        assert_eq!(merged, base);
    }

    // ── sources and resolution ──────────────────────────────────────────────

    #[test]
    fn the_builtin_source_resolves_default() {
        let source = BuiltinSource;
        let pack = source.resolve("default").expect("default exists");
        assert_eq!(pack.source, "builtin:default");
        assert!(pack.surface.prompts.is_empty());
        assert!(source.resolve("nope").is_none());
    }

    #[test]
    fn a_directory_source_reads_a_pack() {
        let dir = tmpdir("read");
        std::fs::write(
            dir.join("team.json"),
            r#"{"version":"2.0","description":"team pack","prompts":{"tier2_distill_rules":"t2"}}"#,
        )
        .unwrap();
        let source = DirSource::new("project", &dir);
        let pack = source.resolve("team").expect("pack must load");
        assert_eq!(pack.name, "team");
        assert_eq!(pack.version.as_deref(), Some("2.0"));
        assert_eq!(
            pack.surface.prompts.tier2_distill_rules.as_deref(),
            Some("t2")
        );
        assert!(pack.source.starts_with("file:"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_filename_is_canonical_not_the_internal_name() {
        // Config says "team"; someone renames the internal field. Resolution
        // must keep working, because a config value points at the file.
        let dir = tmpdir("canonical");
        std::fs::write(dir.join("team.json"), r#"{"name":"totally-different"}"#).unwrap();
        let source = DirSource::new("project", &dir);
        let pack = source.resolve("team").expect("must resolve by filename");
        assert_eq!(pack.name, "team");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_file_resolves_to_nothing() {
        let dir = tmpdir("missing");
        let source = DirSource::new("project", &dir);
        assert!(source.resolve("nope").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_malformed_file_resolves_to_nothing_rather_than_panicking() {
        let dir = tmpdir("malformed");
        std::fs::write(dir.join("broken.json"), "{not json").unwrap();
        let source = DirSource::new("project", &dir);
        assert!(source.resolve("broken").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_traversal_name_never_reads_outside_the_directory() {
        let dir = tmpdir("traversal");
        let source = DirSource::new("project", &dir);
        assert!(source.resolve("../etc/passwd").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_first_source_wins() {
        let project = tmpdir("first-project");
        let user = tmpdir("first-user");
        std::fs::write(
            project.join("team.json"),
            r#"{"prompts":{"compress_philosophy":"project"}}"#,
        )
        .unwrap();
        std::fs::write(
            user.join("team.json"),
            r#"{"prompts":{"compress_philosophy":"user"}}"#,
        )
        .unwrap();
        let resolver = Resolver::with_dirs(Some(project.clone()), vec![user.clone()]);
        let pack = resolver.resolve("team").expect("must resolve");
        assert_eq!(
            pack.surface.prompts.compress_philosophy.as_deref(),
            Some("project"),
            "the project pack must shadow the user pack"
        );
        std::fs::remove_dir_all(&project).ok();
        std::fs::remove_dir_all(&user).ok();
    }

    #[test]
    fn the_builtin_is_always_reachable_as_the_last_source() {
        let resolver = Resolver::with_dirs(None, Vec::new());
        assert!(resolver.resolve("default").is_some());
    }

    #[test]
    fn listing_deduplicates_by_name_first_wins() {
        let project = tmpdir("list-project");
        let user = tmpdir("list-user");
        std::fs::write(project.join("team.json"), "{}").unwrap();
        std::fs::write(user.join("team.json"), "{}").unwrap();
        std::fs::write(user.join("other.json"), "{}").unwrap();
        let resolver = Resolver::with_dirs(Some(project.clone()), vec![user.clone()]);
        let names: Vec<String> = resolver.list().into_iter().map(|p| p.name).collect();
        assert_eq!(
            names.iter().filter(|n| *n == "team").count(),
            1,
            "no duplicate names in the listing"
        );
        assert!(names.contains(&"other".to_string()));
        assert!(names.contains(&"default".to_string()));
        std::fs::remove_dir_all(&project).ok();
        std::fs::remove_dir_all(&user).ok();
    }

    #[test]
    fn listing_a_missing_directory_is_empty_not_an_error() {
        let resolver = Resolver::with_dirs(Some(PathBuf::from("/nonexistent/yak/packs")), vec![]);
        // Only the builtin remains.
        let names: Vec<String> = resolver.list().into_iter().map(|p| p.name).collect();
        assert_eq!(names, vec!["default".to_string()]);
    }

    // ── end to end ──────────────────────────────────────────────────────────

    #[test]
    fn resolving_a_pack_produces_overlaid_prompts() {
        let dir = tmpdir("e2e");
        std::fs::write(
            dir.join("team.json"),
            r#"{"prompts":{"how_to_compress_rules":"team rules"}}"#,
        )
        .unwrap();
        let resolver = Resolver::with_dirs(Some(dir.clone()), Vec::new());
        let (prompts, warning) = resolve_prompts(&resolver, Some("team"));
        assert!(warning.is_none());
        assert_eq!(&*prompts.how_to_compress_rules, "team rules");
        assert!(
            !prompts.compress_philosophy.is_empty(),
            "the unset philosophy must still be the built-in"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unknown_pack_falls_back_with_a_warning() {
        let resolver = Resolver::with_dirs(None, Vec::new());
        let (prompts, warning) = resolve_prompts(&resolver, Some("nope"));
        let base = crate::agent::bigctx::rules::default_prompts();
        assert_eq!(prompts, base, "must fall back, not fail");
        let warning = warning.expect("a miss must warn");
        assert!(warning.contains("nope"));
    }

    // ── the tests that prove the pack is actually wired in ───────────────────

    fn temp_cwd(tag: &str) -> PathBuf {
        let dir = tmpdir(tag);
        std::fs::create_dir_all(dir.join(".yak").join("packs")).unwrap();
        dir
    }

    #[test]
    fn a_pack_changes_the_rendered_nudge() {
        // The point of packs: the rule text inside the nudge the model reads.
        // A pack that resolves but never reaches a renderer is dead weight.
        let cwd = temp_cwd("nudge");
        std::fs::write(
            cwd.join(".yak").join("packs").join("team.json"),
            r#"{"prompts":{"how_to_compress_rules":"TEAM_RULE_SENTINEL"}}"#,
        )
        .unwrap();

        let config =
            crate::agent::bigctx::config_from_json_in(200_000, Some(&json!({"prompt_pack":"team"})), &cwd);
        assert!(
            config.prompts.is_some(),
            "the pack must be resolved into the config"
        );
        let prompts = config
            .prompts
            .clone()
            .expect("resolved");
        assert_eq!(&*prompts.how_to_compress_rules, "TEAM_RULE_SENTINEL");

        // And the built-in default still holds for everything unset.
        let base = crate::agent::bigctx::rules::default_prompts();
        assert_eq!(prompts.compress_philosophy, base.compress_philosophy);

        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn a_pack_overrides_a_tool_description_through_the_registry() {
        // A tool prompt that never reaches a `Tool::description` is inert.
        let cwd = temp_cwd("tool");
        std::fs::write(
            cwd.join(".yak").join("packs").join("team.json"),
            r#"{"tool_prompts":{"compress":{"description":"TEAM_TOOL_SENTINEL"}}}"#,
        )
        .unwrap();

        let config =
            crate::agent::bigctx::config_from_json_in(200_000, Some(&json!({"prompt_pack":"team"})), &cwd);
        assert_eq!(
            config.tool_prompts["compress"].description.as_deref(),
            Some("TEAM_TOOL_SENTINEL")
        );

        let tools = crate::agent::bigctx::bigctx_tools_with(
            crate::agent::bigctx::BigctxHandle::new(),
            &config.tool_prompts,
        );
        let compress = tools
            .iter()
            .find(|t| t.name() == "compress")
            .expect("compress tool");
        assert_eq!(compress.description(), "TEAM_TOOL_SENTINEL");
        // A tool the pack did not mention keeps its built-in text.
        let status = tools.iter().find(|t| t.name() == "acp_status").unwrap();
        assert!(
            status.description().len() > 20,
            "an unmentioned tool must keep its built-in description"
        );

        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn the_project_pack_shadows_a_user_pack() {
        let cwd = temp_cwd("shadow");
        std::fs::write(
            cwd.join(".yak").join("packs").join("team.json"),
            r#"{"prompts":{"how_to_compress_rules":"PROJECT_RULE"}}"#,
        )
        .unwrap();
        let resolver = default_resolver(&cwd);
        let pack = resolver.resolve("team").expect("project pack");
        assert_eq!(
            pack.surface.prompts.how_to_compress_rules.as_deref(),
            Some("PROJECT_RULE")
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn no_pack_configured_leaves_the_builtins_untouched() {
        let cwd = temp_cwd("none");
        let config = crate::agent::bigctx::config_from_json_in(200_000, None, &cwd);
        assert!(
            config.prompts.is_none(),
            "no pack named means no resolution at all"
        );
        assert!(config.tool_prompts.is_empty());
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn an_unknown_pack_name_still_leaves_a_working_config() {
        // A typo must cost a warning, not a session that cannot compress.
        let cwd = temp_cwd("unknown");
        let mut config = crate::agent::bigctx::config_from_json(
            200_000,
            Some(&json!({"prompt_pack":"does-not-exist"})),
        );
        let warning = apply_prompt_pack(&mut config, &cwd);
        assert!(warning.is_some(), "a miss must warn");
        let prompts = config.prompts.expect("falls back to built-ins");
        assert_eq!(
            prompts,
            crate::agent::bigctx::rules::default_prompts(),
            "the fallback must be the built-in set, not empty strings"
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn no_pack_named_means_the_builtin_and_no_warning() {
        let resolver = Resolver::with_dirs(None, Vec::new());
        for name in [None, Some(""), Some("   ")] {
            let (prompts, warning) = resolve_prompts(&resolver, name);
            assert_eq!(prompts, crate::agent::bigctx::rules::default_prompts());
            assert!(warning.is_none());
        }
    }
}
