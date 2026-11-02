//! Per-doc derived features, memoized across search calls.
//!
//! A search over the compressed history re-scores the SAME immutable docs on
//! every call — block summaries never change once written, and folded message
//! text is append-only. Without this cache every `search_context` call would
//! re-tokenize the whole corpus and rebuild the bigram set: a multi-megabyte
//! session costs seconds per call, growing linearly with session length. With
//! the cache the corpus is processed once; later searches are
//! O(docs × query-terms).
//!
//! Keyed by doc text (immutable), bounded by total cached SOURCE CHARS.
//! Eviction is least-recently-used: a hit re-inserts the doc at the tail, so
//! the docs a session keeps re-touching survive.
//!
//! Sizing consequence: if the corpus exceeds the cap and each search rescans
//! it whole, the only setting that stops re-tokenization is a cap at or above
//! the corpus size. Lowering the cap to save memory trades bounded memory for
//! a full re-tokenize of the excess on EVERY call.
//!
//! The cache is module-global, like the kernel's. One yak process runs one
//! session at a time, so a per-session scope would add plumbing for no gain.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{LazyLock, Mutex};

use crate::agent::bigctx::search::tokenizer::{char_bigrams, tf_map};

/// Derived, immutable features of one doc's text.
pub struct DocFeatures {
    /// Stemmed term frequencies (the BM25 channel).
    pub tf: BTreeMap<String, u32>,
    /// Total term count (BM25 length normalization).
    pub len: u32,
    /// Lower-cased text (substring and fuzzy channels).
    pub lower: String,
    /// Unique char bigrams of `lower` (the fuzzy channel).
    pub grams: BTreeSet<String>,
}

const DEFAULT_CAP_CHARS: usize = 8 * 1024 * 1024;

struct Cache {
    cap_chars: usize,
    cached_chars: usize,
    /// Insertion order is recency order: a hit moves the entry to the back.
    map: BTreeMap<String, std::sync::Arc<DocFeatures>>,
    /// Recency order, oldest first.
    order: Vec<String>,
}

impl Cache {
    fn new() -> Cache {
        Cache {
            cap_chars: DEFAULT_CAP_CHARS,
            cached_chars: 0,
            map: BTreeMap::new(),
            order: Vec::new(),
        }
    }

    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push(key.to_string());
    }

    fn evict_until_fits(&mut self, needed: usize) {
        while self.cached_chars + needed > self.cap_chars && !self.order.is_empty() {
            let victim = self.order.remove(0);
            if self.map.remove(&victim).is_some() {
                self.cached_chars -= victim.len();
            }
        }
    }
}

static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(|| Mutex::new(Cache::new()));

fn build(text: &str) -> DocFeatures {
    let tf = tf_map(text, true);
    let len = tf.values().copied().sum();
    let lower = text.to_lowercase();
    let grams = char_bigrams(&lower).into_iter().collect();
    DocFeatures {
        tf,
        len,
        lower,
        grams,
    }
}

/// Features for `text`, from the cache when present.
///
/// Returns an `Arc` so a caller can hold the features without keeping the
/// cache lock — the lock is taken once here, not once per channel.
pub fn doc_features(text: &str) -> std::sync::Arc<DocFeatures> {
    {
        let mut cache = match CACHE.lock() {
            Ok(c) => c,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(hit) = cache.map.get(text).cloned() {
            cache.touch(text);
            return hit;
        }
    }
    // Built outside the lock: tokenizing a large doc under a global mutex would
    // serialize every other lookup behind it.
    let features = std::sync::Arc::new(build(text));
    let mut cache = match CACHE.lock() {
        Ok(c) => c,
        Err(poisoned) => poisoned.into_inner(),
    };
    if text.len() > 0 && text.len() <= cache.cap_chars && !cache.map.contains_key(text) {
        let needed = text.len();
        cache.evict_until_fits(needed);
        cache.map.insert(text.to_string(), features.clone());
        cache.order.push(text.to_string());
        cache.cached_chars += needed;
    }
    features
}

/// Drop every cached feature set (session shutdown / context switch).
pub fn clear_doc_features() {
    let mut cache = match CACHE.lock() {
        Ok(c) => c,
        Err(poisoned) => poisoned.into_inner(),
    };
    cache.map.clear();
    cache.order.clear();
    cache.cached_chars = 0;
}

/// Set the cache cap in source chars. A doc larger than the cap is never
/// cached. Pass a value at or above the corpus size to cache all of it.
pub fn set_doc_cache_cap(chars: usize) {
    let mut cache = match CACHE.lock() {
        Ok(c) => c,
        Err(poisoned) => poisoned.into_inner(),
    };
    cache.cap_chars = chars.max(1);
    while cache.cached_chars > cache.cap_chars && !cache.order.is_empty() {
        let victim = cache.order.remove(0);
        if cache.map.remove(&victim).is_some() {
            cache.cached_chars -= victim.len();
        }
    }
}

/// Cache occupancy, for diagnostics.
pub fn doc_cache_info() -> (usize, usize) {
    let cache = match CACHE.lock() {
        Ok(c) => c,
        Err(poisoned) => poisoned.into_inner(),
    };
    (cache.map.len(), cache.cached_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The cache is module-global, so tests that touch the cap must not run
    // concurrently with each other. Serialize them behind one lock.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
        match LOCK.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        }
    }

    #[test]
    fn features_carry_tf_len_and_bigrams() {
        let _g = lock();
        let f = doc_features("cache cache login");
        assert_eq!(f.tf.get("cache"), Some(&2));
        assert_eq!(f.len, 3);
        assert!(f.grams.contains("ca"));
        assert!(f.lower.starts_with("cache"));
    }

    #[test]
    fn a_repeat_lookup_returns_the_same_allocation() {
        let _g = lock();
        let a = doc_features("identical text for both calls");
        let b = doc_features("identical text for both calls");
        assert!(
            std::sync::Arc::ptr_eq(&a, &b),
            "the second lookup must hit the cache, not rebuild"
        );
    }

    #[test]
    fn the_cap_evicts_and_the_entry_is_rebuilt() {
        let _g = lock();
        clear_doc_features();
        set_doc_cache_cap(40);
        let big = "x".repeat(30);
        let first = doc_features(&big);
        // A second doc that cannot fit alongside the first evicts it.
        let _other = doc_features(&"y".repeat(30));
        let (entries, _chars) = doc_cache_info();
        assert!(entries <= 1, "cap must bound the entry count, got {entries}");
        let again = doc_features(&big);
        assert!(
            !std::sync::Arc::ptr_eq(&first, &again),
            "an evicted doc must be rebuilt"
        );
        set_doc_cache_cap(DEFAULT_CAP_CHARS);
        clear_doc_features();
    }

    #[test]
    fn a_doc_larger_than_the_cap_is_never_cached() {
        let _g = lock();
        clear_doc_features();
        set_doc_cache_cap(10);
        let a = doc_features(&"z".repeat(50));
        let b = doc_features(&"z".repeat(50));
        assert!(!std::sync::Arc::ptr_eq(&a, &b));
        set_doc_cache_cap(DEFAULT_CAP_CHARS);
        clear_doc_features();
    }
}
