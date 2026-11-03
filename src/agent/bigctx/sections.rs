//! Tri-state overrides for named prompt sections.
//!
//! The load-bearing rule texts live in [`super::rules`] and a pack overrides
//! them wholesale. Sections are the other half of a prompt — the headings and
//! connective prose around those rules — and a pack overrides them *piecewise*,
//! with three states per section:
//!
//! - a string replaces the section's text
//! - `null` drops the section entirely
//! - absent keeps the default
//!
//! The third state is what makes this worth a type. A pack that wants to trim
//! the prompt has to say "remove this section", and it cannot say that with a
//! string — an empty string still emits a blank pair, and omitting the key means
//! "keep it". Without an explicit null there is no way to express removal, and
//! packs end up copying whole default texts just to delete a paragraph.
//!
//! Malformed values are dropped, never coerced: a number where a section
//! belongs must not blank it. A blanked section is the worst failure mode here
//! because the prompt still assembles and still runs — it just quietly stops
//! telling the model something it needs to know.

use std::collections::BTreeMap;

/// One section's override.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SectionOverride {
    /// Replace the default text.
    Replace(String),
    /// Drop the section.
    Remove,
}

impl SectionOverride {
    /// Apply to a default section, returning what should be emitted.
    fn resolve<'a>(&'a self, default: &'a str) -> Option<&'a str> {
        match self {
            SectionOverride::Replace(text) => Some(text),
            SectionOverride::Remove => None,
        }
    }
}

/// A set of section overrides, keyed by section name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SectionOverrides(BTreeMap<String, SectionOverride>);

impl SectionOverrides {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether this set names `key` at all, in any state.
    pub fn names(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    /// Read a parsed pack object into overrides, applying the tri-state rules.
    ///
    /// `allowed` bounds which keys are meaningful. A section name the host does
    /// not render would otherwise sit in the map looking configured while
    /// changing nothing — the same silent no-op that made the whole pack
    /// feature inert before.
    pub fn from_json(
        raw: Option<&serde_json::Value>,
        allowed: &[&str],
    ) -> SectionOverrides {
        let mut out = SectionOverrides::default();
        let Some(object) = raw.and_then(serde_json::Value::as_object) else {
            return out;
        };
        for (key, value) in object {
            if !allowed.contains(&key.as_str()) {
                continue;
            }
            if value.is_null() {
                out.0.insert(key.clone(), SectionOverride::Remove);
            } else if let Some(text) = value.as_str() {
                out.0.insert(key.clone(), SectionOverride::Replace(text.to_string()));
            }
            // Any other type is a malformed override: dropped, so the default
            // survives.
        }
        out
    }

    /// Render an ordered list of `(name, default_text)` sections, applying
    /// overrides and dropping removed ones.
    ///
    /// Order is the caller's, not the pack's: a pack replaces what a section
    /// says, never where it goes. Letting a pack reorder the prompt would let it
    /// move the directive that the model reads first, which is not a
    /// presentation concern.
    pub fn render(&self, sections: &[(&str, &str)]) -> Vec<String> {
        let mut out = Vec::new();
        for (name, default) in sections {
            match self.0.get(*name) {
                Some(over) => {
                    if let Some(text) = over.resolve(default) {
                        out.push(text.to_string());
                    }
                }
                None => out.push((*default).to_string()),
            }
        }
        out
    }

    /// Render with owned defaults, for a caller that builds its section texts
    /// from the resolved rule set rather than from literals.
    pub fn render_owned(&self, sections: &[(&str, String)]) -> Vec<String> {
        let mut out = Vec::new();
        for (name, default) in sections {
            match self.0.get(*name) {
                Some(over) => {
                    if let Some(text) = over.resolve(default.as_str()) {
                        out.push(text.to_string());
                    }
                }
                None => out.push(default.clone()),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const ALLOWED: &[&str] = &["alpha", "beta"];

    #[test]
    fn an_absent_key_keeps_the_default() {
        let over = SectionOverrides::from_json(Some(&json!({})), ALLOWED);
        assert_eq!(
            over.render(&[("alpha", "DEFAULT_A"), ("beta", "DEFAULT_B")]),
            vec!["DEFAULT_A", "DEFAULT_B"]
        );
    }

    #[test]
    fn a_string_replaces_the_section() {
        let over = SectionOverrides::from_json(Some(&json!({"alpha": "CUSTOM"})), ALLOWED);
        assert_eq!(
            over.render(&[("alpha", "DEFAULT_A"), ("beta", "DEFAULT_B")]),
            vec!["CUSTOM", "DEFAULT_B"]
        );
    }

    #[test]
    fn null_removes_the_section() {
        // The state that is impossible to express with strings alone, and the
        // whole reason this type exists.
        let over = SectionOverrides::from_json(Some(&json!({"alpha": null})), ALLOWED);
        assert_eq!(
            over.render(&[("alpha", "DEFAULT_A"), ("beta", "DEFAULT_B")]),
            vec!["DEFAULT_B"]
        );
    }

    #[test]
    fn a_malformed_value_leaves_the_default() {
        // A number must not blank the section: a blanked section still
        // assembles, so the failure would be silent.
        for bad in [json!(42), json!(true), json!([1, 2]), json!({"x": 1})] {
            let over = SectionOverrides::from_json(Some(&json!({"alpha": bad})), ALLOWED);
            assert_eq!(
                over.render(&[("alpha", "DEFAULT_A")]),
                vec!["DEFAULT_A"],
                "a malformed override must not clobber the default"
            );
            assert!(!over.names("alpha"));
        }
    }

    #[test]
    fn a_non_object_block_yields_no_overrides() {
        for raw in [json!("text"), json!(7), json!([1])] {
            let over = SectionOverrides::from_json(Some(&raw), ALLOWED);
            assert!(over.is_empty());
        }
        assert!(SectionOverrides::from_json(None, ALLOWED).is_empty());
    }

    #[test]
    fn an_unknown_section_name_is_ignored() {
        // A key the host never renders would look configured while changing
        // nothing — the failure mode that made packs inert once already.
        let over = SectionOverrides::from_json(
            Some(&json!({"not_a_section": "TEXT", "alpha": "OK"})),
            ALLOWED,
        );
        assert!(!over.names("not_a_section"));
        assert!(over.names("alpha"));
        assert_eq!(over.len(), 1);
    }

    #[test]
    fn order_follows_the_caller() {
        // A pack replaces what a section says, never where it goes.
        let over = SectionOverrides::from_json(Some(&json!({"beta": "B2"})), ALLOWED);
        assert_eq!(
            over.render(&[("alpha", "A1"), ("beta", "B1")]),
            vec!["A1", "B2"]
        );
    }

    #[test]
    fn an_empty_string_still_emits_the_section() {
        // Distinct from removal: a pack can blank a heading but keep the slot.
        let over = SectionOverrides::from_json(Some(&json!({"alpha": ""})), ALLOWED);
        assert_eq!(over.render(&[("alpha", "A")]), vec![""]);
    }

    #[test]
    fn owned_defaults_render_the_same_way() {
        let over = SectionOverrides::from_json(Some(&json!({"beta": null})), ALLOWED);
        let out = over.render_owned(&[
            ("alpha", "A".to_string()),
            ("beta", "B".to_string()),
        ]);
        assert_eq!(out, vec!["A".to_string()]);
    }
}
