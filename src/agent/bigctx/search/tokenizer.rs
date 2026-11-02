//! Search tokenizer — mixed Latin + CJK.
//!
//! The single biggest quality lever over plain substring search. Latin splits
//! on non-word boundaries and is lightly stemmed; CJK has no spaces, so it is
//! word-segmented and falls back to overlapping bigrams when no multi-char
//! word is found (out-of-vocabulary text), so a query like "身份验证" still
//! scores against doc text "身份验证流程".
//!
//! The upstream kernel segments CJK with `Intl.Segmenter` (an ICU dictionary).
//! Rust has no dependency-free equivalent, so this module does without: CJK
//! runs become overlapping bigrams plus single characters. That is the
//! upstream's *fallback* path, applied unconditionally. The trade is recall
//! for precision: bigrams do not distinguish 国际化 from 国际+化, so a query
//! matches more docs than a dictionary segmenter would. For a retrieval aid
//! over folded history that is the right side to err on — a missed hit is
//! worse than a weak hit, because the model can read the preview and decide.

/// A CJK ideograph, kana or hangul character. One shared definition, so the
/// fuzzy gate in `fuzzy.rs` relaxes its short-query rule for exactly the range
/// this module segments.
///
/// Latin is deliberately absent: two-character English tokens ("to", "of")
/// carry no meaning, while nearly all CJK words are two-character atomic
/// units (登录/缓存) — the two scripts need opposite rules.
pub fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x3400..=0x9FFF     // CJK Unified Ideographs (+ Extension A)
        | 0xF900..=0xFAFF   // CJK Compatibility Ideographs
        | 0x3040..=0x30FF   // Hiragana + Katakana
        | 0xAC00..=0xD7AF   // Hangul Syllables
    )
}

/// True when the text contains any CJK character.
pub fn has_cjk(text: &str) -> bool {
    text.chars().any(is_cjk)
}

/// Tokens for `text`, optionally stemmed.
///
/// Latin: maximal runs of `[a-z0-9_]`, lowercased, dropping single characters
/// (they are noise). CJK: each contiguous run becomes overlapping bigrams plus
/// its single characters.
pub fn tokenize(text: &str, stem: bool) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut tokens: Vec<String> = Vec::new();

    // Latin and CJK are split in one pass: a char is either "word" material or
    // a separator, and CJK chars are their own class so a mixed run such as
    // "缓存cache" yields both halves instead of one merged token.
    let mut latin = String::new();
    let mut cjk_run: Vec<char> = Vec::new();

    let flush_latin = |buf: &mut String, out: &mut Vec<String>| {
        if buf.len() >= 2 {
            let word = std::mem::take(buf);
            out.push(if stem { stem_word(&word) } else { word });
        } else {
            buf.clear();
        }
    };
    let flush_cjk = |run: &mut Vec<char>, out: &mut Vec<String>| {
        if run.is_empty() {
            return;
        }
        // Overlapping bigrams keep "身份验证" matching "身份验证流程"; single
        // chars are added too so a one-character query still hits.
        for i in 0..run.len().saturating_sub(1) {
            out.push(run[i..i + 2].iter().collect());
        }
        for ch in run.iter() {
            out.push(ch.to_string());
        }
        run.clear();
    };

    for ch in lower.chars() {
        if is_cjk(ch) {
            flush_latin(&mut latin, &mut tokens);
            cjk_run.push(ch);
        } else if ch.is_alphanumeric() || ch == '_' {
            flush_cjk(&mut cjk_run, &mut tokens);
            latin.push(ch);
        } else {
            flush_latin(&mut latin, &mut tokens);
            flush_cjk(&mut cjk_run, &mut tokens);
        }
    }
    flush_latin(&mut latin, &mut tokens);
    flush_cjk(&mut cjk_run, &mut tokens);

    tokens
}

/// Character bigrams over arbitrary text — the fuzzy channel's unit.
///
/// Bigrams that span whitespace are skipped: they would let two unrelated
/// words in a sentence match each other.
pub fn char_bigrams(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut grams = Vec::new();
    for pair in chars.windows(2) {
        if !pair[0].is_whitespace() && !pair[1].is_whitespace() {
            grams.push(pair.iter().collect());
        }
    }
    grams
}

/// Term-frequency map over `text`.
pub fn tf_map(text: &str, stem: bool) -> std::collections::BTreeMap<String, u32> {
    let mut map = std::collections::BTreeMap::new();
    for token in tokenize(text, stem) {
        *map.entry(token).or_insert(0) += 1;
    }
    map
}

/// Lightweight English stemmer (suffix stripping, Porter-inspired).
///
/// Not a full Porter stemmer — deliberately simpler, and the point is only
/// that compress/compressed/compression collapse together for matching. CJK
/// never reaches here (bigram tokenization handles it, stemming would mangle
/// it), which is why the width check is on chars, not bytes.
pub fn stem_word(word: &str) -> String {
    let mut w = word.to_string();
    let len = |s: &str| s.chars().count();
    if len(&w) <= 3 {
        return w;
    }
    if w.ends_with("ies") && len(&w) > 4 {
        w.truncate(w.len() - 3);
        w.push('y');
    } else if w.ends_with("sses") || w.ends_with("xes") || w.ends_with("zes") {
        w.truncate(w.len() - 2);
    } else if w.ends_with("ches") || w.ends_with("shes") {
        w.truncate(w.len() - 2);
    } else if w.ends_with('s') && !w.ends_with("ss") && !w.ends_with("us") {
        w.truncate(w.len() - 1);
    }
    if w.ends_with("ing") && len(&w) > 5 {
        w.truncate(w.len() - 3);
    }
    if w.ends_with("ed") && len(&w) > 4 {
        w.truncate(w.len() - 2);
    }
    if w.ends_with("ation") && len(&w) > 6 {
        w.truncate(w.len() - 3);
    } else if w.ends_with("tion") && len(&w) > 5 {
        w.truncate(w.len() - 4);
        w.push('t');
    } else if w.ends_with("ion") && len(&w) > 4 {
        w.truncate(w.len() - 3);
    }
    if w.ends_with("ment") && len(&w) > 6 {
        w.truncate(w.len() - 4);
    }
    if w.ends_with("ness") && len(&w) > 6 {
        w.truncate(w.len() - 4);
    }
    if w.ends_with("ly") && len(&w) > 4 {
        w.truncate(w.len() - 2);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin_is_split_and_lowercased() {
        let t = tokenize("The Compressor runs", false);
        assert!(t.contains(&"the".to_string()));
        assert!(t.contains(&"compressor".to_string()));
    }

    #[test]
    fn latin_single_chars_are_dropped() {
        // "a" and "I" carry no retrieval signal
        let t = tokenize("a b c", false);
        assert!(t.is_empty(), "got {t:?}");
    }

    #[test]
    fn stemming_collapses_morphology() {
        assert_eq!(stem_word("compressed"), "compress");
        assert_eq!(stem_word("compression"), "compress");
        assert_eq!(stem_word("compress"), "compress");
        assert_eq!(stem_word("handling"), "handl");
        assert_eq!(stem_word("subagents"), "subagent");
    }

    #[test]
    fn short_words_are_not_stemmed() {
        assert_eq!(stem_word("was"), "was");
    }

    #[test]
    fn cjk_runs_become_bigrams_and_single_chars() {
        let t = tokenize("身份验证流程", false);
        assert!(t.contains(&"身份".to_string()), "got {t:?}");
        assert!(t.contains(&"验证".to_string()));
        // single chars too, so a one-char query still matches
        assert!(t.contains(&"身".to_string()));
    }

    #[test]
    fn mixed_script_splits_at_the_boundary() {
        let t = tokenize("缓存cache", false);
        assert!(t.contains(&"cache".to_string()), "got {t:?}");
        assert!(t.contains(&"缓存".to_string()));
    }

    #[test]
    fn bigrams_skip_pairs_spanning_whitespace() {
        let g = char_bigrams("ab cd");
        // "ab", "b " (skipped), " c" (skipped), "cd"
        assert_eq!(g, vec!["ab".to_string(), "cd".to_string()]);
    }

    #[test]
    fn an_empty_string_yields_nothing() {
        assert!(tokenize("", false).is_empty());
        assert!(char_bigrams("").is_empty());
        assert!(tf_map("", true).is_empty());
    }

    #[test]
    fn tf_counts_repeats() {
        let m = tf_map("cache cache login", true);
        assert_eq!(m.get("cache"), Some(&2));
        assert_eq!(m.get("login"), Some(&1));
    }
}
