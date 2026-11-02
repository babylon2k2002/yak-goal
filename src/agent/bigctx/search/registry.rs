//! Algorithm registry.
//!
//! Builtins are registered once, lazily. `hybrid` is the default (see
//! [`DEFAULT_ALGORITHM`]). A name that is not registered yields no results
//! rather than an error: a model that guesses an algorithm name should get an
//! empty list it can retry past, not a failed tool call.

use std::sync::LazyLock;

use crate::agent::bigctx::search::bm25::Bm25;
use crate::agent::bigctx::search::fuzzy::Fuzzy;
use crate::agent::bigctx::search::hybrid::Hybrid;
use crate::agent::bigctx::search::substring::Substring;
use crate::agent::bigctx::search::types::SearchAlgorithm;

static REGISTRY: LazyLock<Vec<Box<dyn SearchAlgorithm>>> = LazyLock::new(|| {
    vec![
        Box::new(Substring),
        Box::new(Bm25),
        Box::new(Fuzzy),
        Box::new(Hybrid),
    ]
});

/// The algorithm registered under `name`.
pub fn get_algorithm(name: &str) -> Option<&'static dyn SearchAlgorithm> {
    REGISTRY
        .iter()
        .find(|a| a.name() == name)
        .map(|a| a.as_ref())
}

/// Every registered algorithm, for the help text.
pub fn list_algorithms() -> Vec<(&'static str, &'static str)> {
    REGISTRY
        .iter()
        .map(|a| (a.name(), a.description()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_is_registered() {
        for name in ["substring", "bm25", "fuzzy", "hybrid"] {
            assert!(get_algorithm(name).is_some(), "{name} must be registered");
        }
    }

    #[test]
    fn the_default_is_registered() {
        assert!(get_algorithm(crate::agent::bigctx::search::types::DEFAULT_ALGORITHM).is_some());
    }

    #[test]
    fn an_unknown_name_yields_none() {
        assert!(get_algorithm("semantic").is_none());
        assert!(get_algorithm("").is_none());
    }

    #[test]
    fn each_algorithm_reports_a_name_and_description() {
        for (name, description) in list_algorithms() {
            assert!(!name.is_empty());
            assert!(!description.is_empty(), "{name} needs a description");
        }
    }
}
