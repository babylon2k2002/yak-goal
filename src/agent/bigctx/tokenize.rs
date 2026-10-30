//! Token estimation, CJK-aware.
//!
//! The algorithm's thresholds are all token counts, so the estimator's *shape*
//! matters more than its absolute accuracy: a plain `chars / 4` badly
//! under-counts CJK (where one character is roughly one token), which would
//! make every threshold fire far too late for Chinese conversations.

use crate::agent::bigctx::types::CoreMessage;

/// Rough token count for a string.
///
/// CJK characters count 1:1; the rest is estimated at 4 characters per token
/// so that code and JSON punctuation still contribute.
pub fn default_count_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let mut cjk = 0u64;
    let mut other = 0u64;
    for ch in text.chars() {
        if is_cjk(ch) {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    cjk + other.div_ceil(4)
}

/// The CJK ranges that tokenize roughly one character per token: Han,
/// Hiragana/Katakana, and Hangul.
fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x4E00..=0x9FFF     // CJK Unified Ideographs
        | 0x3400..=0x4DBF   // CJK Extension A
        | 0x3040..=0x30FF   // Hiragana + Katakana
        | 0xAC00..=0xD7AF   // Hangul Syllables
        | 0xF900..=0xFAFF   // CJK Compatibility Ideographs
    )
}

/// Tokens for one core: its text plus any tool payload it carries.
///
/// A tool call's arguments live in `text` on this projection (the host puts
/// them there), so this stays a single term.
pub fn count_message_tokens(message: &CoreMessage) -> u64 {
    default_count_tokens(message.text_or_empty())
}

/// Tokens for a slice of cores.
pub fn count_messages_tokens(messages: &[CoreMessage]) -> u64 {
    messages.iter().map(count_message_tokens).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(default_count_tokens(""), 0);
    }

    #[test]
    fn ascii_is_about_four_chars_per_token() {
        assert_eq!(default_count_tokens("abcd"), 1);
        assert_eq!(default_count_tokens("abcde"), 2);
        assert_eq!(default_count_tokens("a".repeat(400).as_str()), 100);
    }

    #[test]
    fn cjk_is_one_token_per_char() {
        // 10 Han characters: 10 tokens, not 2 (chars/4) and not 4
        assert_eq!(default_count_tokens("上下文压缩管理器测试"), 10);
    }

    #[test]
    fn mixed_text_counts_both_sides() {
        // 4 Han + 8 ascii -> 4 + 2
        assert_eq!(default_count_tokens("中文压缩abcdwxyz"), 6);
    }

    #[test]
    fn hiragana_and_hangul_are_also_cjk() {
        assert_eq!(default_count_tokens("あいうえお"), 5);
        assert_eq!(default_count_tokens("한국어텍스트"), 6);
    }

    #[test]
    fn a_message_counts_only_its_text() {
        let m = CoreMessage::new("t1.1", crate::agent::bigctx::types::MessageRole::User,
                                 crate::agent::bigctx::types::MessageContentType::Text)
            .with_text("abcd");
        assert_eq!(count_message_tokens(&m), 1);
    }

    #[test]
    fn a_message_without_text_counts_zero() {
        let m = CoreMessage::new("t1.1", crate::agent::bigctx::types::MessageRole::Assistant,
                                 crate::agent::bigctx::types::MessageContentType::ToolCall);
        assert_eq!(count_message_tokens(&m), 0);
    }
}
