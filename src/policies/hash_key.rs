//! Shared hash key extraction for consistent hashing policies
//!
//! Extracts routing keys from HTTP headers and request bodies.
//! Used by both EngineLoadPolicy and RendezvousHashPolicy.

use super::RequestHeaders;

use tracing::debug;

/// HTTP header names to check for session ID (case-insensitive, checked in order)
pub(crate) const SESSION_HEADER_NAMES: &[&str] = &[
    "x-session-id",
    "x-user-id",
    "x-tenant-id",
    "x-correlation-id", // per-session — check before per-request
    "x-request-id",
    "x-trace-id",
];

/// Extract hash key with priority: HTTP headers > body fields > request content hash
///
/// Priority order:
/// 1. HTTP Headers: x-session-id, x-user-id, x-tenant-id, x-correlation-id, x-request-id, x-trace-id
/// 2. Body: session_params.session_id (nested)
/// 3. Body: user field (OpenAI format)
/// 4. Body: session_id (legacy)
/// 5. Body: user_id (legacy)
/// 6. Fallback: hash of request body (long) or raw text (short)
pub(crate) fn extract_hash_key(
    request_text: Option<&str>,
    headers: Option<&RequestHeaders>,
) -> String {
    // 1. First priority: HTTP headers
    if let Some(hdrs) = headers {
        if let Some(key) = extract_hash_key_from_headers(hdrs) {
            return key;
        }
    }

    // 2. Second priority: Body fields
    if let Some(key) = extract_hash_key_from_body(request_text) {
        return key;
    }

    // 3. Final fallback: hash of request body
    let text = request_text.unwrap_or("");
    if text.len() > 100 {
        format!("request_hash:{:016x}", StableHash::fbi_hash(text))
    } else {
        format!("request:{}", text)
    }
}

/// Extract hash key from HTTP headers
pub(crate) fn extract_hash_key_from_headers(headers: &RequestHeaders) -> Option<String> {
    for header_name in SESSION_HEADER_NAMES {
        if let Some(value) = headers.get(*header_name) {
            if !value.is_empty() {
                debug!(
                    "Hash key extraction: found session key in header '{}': {}",
                    header_name, value
                );
                return Some(format!("header:{}:{}", header_name, value));
            }
        }
    }
    None
}

/// Extract hash key from request body fields
///
/// Priority: session_params.session_id > user > session_id > user_id
pub(crate) fn extract_hash_key_from_body(request_text: Option<&str>) -> Option<String> {
    let text = request_text.unwrap_or("");
    if text.is_empty() {
        return None;
    }

    // 1. Try to extract session_params.session_id first (highest priority in body)
    if let Some(session_id) = extract_nested_field_value(text, "session_params", "session_id") {
        debug!(
            "Hash key extraction: found session_params.session_id: {}",
            session_id
        );
        return Some(format!("session:{}", session_id));
    }

    // 2. Try to extract direct user field (from OpenAI ChatCompletion/Completion requests)
    if let Some(user) = extract_field_value(text, "user") {
        debug!("Hash key extraction: found user field: {}", user);
        return Some(format!("user:{}", user));
    }

    // 3. Fallback: try legacy session_id field
    if let Some(session_id) = extract_field_value(text, "session_id") {
        return Some(format!("session:{}", session_id));
    }

    // 4. Fallback: try legacy user_id field
    if let Some(user_id) = extract_field_value(text, "user_id") {
        return Some(format!("user:{}", user_id));
    }

    None
}

/// Extract nested field value like session_params.session_id from JSON text
pub(crate) fn extract_nested_field_value(
    text: &str,
    parent_field: &str,
    child_field: &str,
) -> Option<String> {
    if let Some(parent_start) = find_field_start(text, parent_field) {
        if let Some(obj_start) = text[parent_start..].find('{') {
            let obj_start_pos = parent_start + obj_start;
            if let Some(obj_content) = extract_json_object(&text[obj_start_pos..]) {
                return extract_field_value(&obj_content, child_field);
            }
        }
    }
    None
}

/// Find the start position after the colon of a field in JSON text
pub(crate) fn find_field_start(text: &str, field_name: &str) -> Option<usize> {
    let patterns = [format!("\"{}\"", field_name), format!("'{}'", field_name)];

    for pattern in &patterns {
        if let Some(field_pos) = text.find(pattern) {
            let after_field = &text[field_pos + pattern.len()..];
            for (i, ch) in after_field.char_indices() {
                if ch == ':' {
                    return Some(field_pos + pattern.len() + i + 1);
                } else if !ch.is_whitespace() {
                    break;
                }
            }
        }
    }
    None
}

/// Extract JSON object content (simple brace matching)
pub(crate) fn extract_json_object(text: &str) -> Option<String> {
    if !text.starts_with('{') {
        return None;
    }

    let mut brace_count = 0;
    let mut end_pos = 0;

    for (i, ch) in text.char_indices() {
        match ch {
            '{' => brace_count += 1,
            '}' => {
                brace_count -= 1;
                if brace_count == 0 {
                    end_pos = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    if brace_count == 0 && end_pos > 0 {
        Some(text[0..end_pos].to_string())
    } else {
        None
    }
}

/// Extract field value from JSON-like text (simple parser)
///
/// Supports double-quoted, single-quoted, and unquoted values.
pub(crate) fn extract_field_value(text: &str, field_name: &str) -> Option<String> {
    let patterns = [
        format!("\"{}\"", field_name),
        format!("'{}'", field_name),
        field_name.to_string(),
    ];

    for pattern in &patterns {
        if let Some(field_pos) = text.find(pattern) {
            let after_field = &text[field_pos + pattern.len()..];

            // Skip whitespace and look for colon
            let mut colon_pos = None;
            for (i, ch) in after_field.char_indices() {
                if ch == ':' {
                    colon_pos = Some(i);
                    break;
                } else if !ch.is_whitespace() {
                    break;
                }
            }

            if let Some(colon_idx) = colon_pos {
                let after_colon = &after_field[colon_idx + 1..];
                let trimmed = after_colon.trim_start();

                // Extract quoted string (double or single quotes)
                if trimmed.starts_with('"') {
                    if let Some(stripped) = trimmed.strip_prefix('"') {
                        if let Some(end_quote) = stripped.find('"') {
                            return Some(stripped[..end_quote].to_string());
                        }
                    }
                } else if trimmed.starts_with('\'') {
                    if let Some(stripped) = trimmed.strip_prefix('\'') {
                        if let Some(end_quote) = stripped.find('\'') {
                            return Some(stripped[..end_quote].to_string());
                        }
                    }
                } else {
                    // Unquoted value - extract until delimiter
                    let end_pos = trimmed
                        .find(&[',', ' ', '}', ']', '\n', '\r', '\t'][..])
                        .unwrap_or(trimmed.len());
                    if end_pos > 0 {
                        return Some(trimmed[..end_pos].to_string());
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // === extract_field_value tests ===

    #[test]
    fn test_extract_field_value_double_quoted() {
        let text = r#"{"session_id": "abc123", "prompt": "hello"}"#;
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("abc123".to_string())
        );
    }

    #[test]
    fn test_extract_field_value_single_quoted() {
        let text = r#"{'session_id': 'def456', 'prompt': 'world'}"#;
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("def456".to_string())
        );
    }

    #[test]
    fn test_extract_field_value_unquoted() {
        let text = r#"{"count": 42, "name": "test"}"#;
        assert_eq!(extract_field_value(text, "count"), Some("42".to_string()));
    }

    #[test]
    fn test_extract_field_value_missing() {
        let text = r#"{"other": "val"}"#;
        assert_eq!(extract_field_value(text, "session_id"), None);
    }

    #[test]
    fn test_extract_field_value_no_space_after_colon() {
        let text = r#"{"session_id":"compact_value"}"#;
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("compact_value".to_string())
        );
    }

    #[test]
    fn test_extract_field_value_multiple_fields() {
        let text = r#"{"user": "bob", "prompt": "hi", "session_id": "sess1"}"#;
        assert_eq!(extract_field_value(text, "user"), Some("bob".to_string()));
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("sess1".to_string())
        );
    }

    // === extract_nested_field_value tests ===

    #[test]
    fn test_extract_nested_field_value() {
        let text = r#"{"session_params": {"session_id": "nested123"}, "prompt": "hi"}"#;
        assert_eq!(
            extract_nested_field_value(text, "session_params", "session_id"),
            Some("nested123".to_string())
        );
    }

    #[test]
    fn test_extract_nested_field_value_missing_parent() {
        let text = r#"{"prompt": "hi"}"#;
        assert_eq!(
            extract_nested_field_value(text, "session_params", "session_id"),
            None
        );
    }

    #[test]
    fn test_extract_nested_field_value_missing_child() {
        let text = r#"{"session_params": {"other": "val"}, "prompt": "hi"}"#;
        assert_eq!(
            extract_nested_field_value(text, "session_params", "session_id"),
            None
        );
    }

    // === extract_json_object tests ===

    #[test]
    fn test_extract_json_object_simple() {
        let text = r#"{"key": "value"} trailing"#;
        assert_eq!(
            extract_json_object(text),
            Some(r#"{"key": "value"}"#.to_string())
        );
    }

    #[test]
    fn test_extract_json_object_nested() {
        let text = r#"{"outer": {"inner": "val"}} trailing"#;
        assert_eq!(
            extract_json_object(text),
            Some(r#"{"outer": {"inner": "val"}}"#.to_string())
        );
    }

    #[test]
    fn test_extract_json_object_not_object() {
        assert_eq!(extract_json_object("not json"), None);
    }

    #[test]
    fn test_extract_json_object_unclosed() {
        assert_eq!(extract_json_object("{unclosed"), None);
    }

    // === extract_hash_key_from_headers tests ===

    #[test]
    fn test_header_extraction_priority() {
        let mut headers = HashMap::new();
        headers.insert("x-request-id".to_string(), "req-1".to_string());
        headers.insert("x-session-id".to_string(), "sess-1".to_string());

        // x-session-id has higher priority than x-request-id
        let key = extract_hash_key_from_headers(&headers).unwrap();
        assert_eq!(key, "header:x-session-id:sess-1");
    }

    #[test]
    fn test_header_extraction_skips_empty() {
        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "".to_string());
        headers.insert("x-user-id".to_string(), "user-1".to_string());

        let key = extract_hash_key_from_headers(&headers).unwrap();
        assert_eq!(key, "header:x-user-id:user-1");
    }

    #[test]
    fn test_header_extraction_no_match() {
        let mut headers = HashMap::new();
        headers.insert("x-custom-header".to_string(), "val".to_string());
        assert_eq!(extract_hash_key_from_headers(&headers), None);
    }

    // === extract_hash_key_from_body tests ===

    #[test]
    fn test_body_extraction_session_params_priority() {
        // session_params.session_id takes priority over direct session_id
        let text = r#"{"session_params": {"session_id": "nested"}, "session_id": "direct"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "session:nested");
    }

    #[test]
    fn test_body_extraction_user_field() {
        let text = r#"{"user": "alice", "prompt": "hi"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "user:alice");
    }

    #[test]
    fn test_body_extraction_legacy_session_id() {
        let text = r#"{"session_id": "legacy123", "prompt": "hi"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "session:legacy123");
    }

    #[test]
    fn test_body_extraction_legacy_user_id() {
        let text = r#"{"user_id": "uid456", "prompt": "hi"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "user:uid456");
    }

    #[test]
    fn test_body_extraction_empty() {
        assert_eq!(extract_hash_key_from_body(None), None);
        assert_eq!(extract_hash_key_from_body(Some("")), None);
    }

    #[test]
    fn test_body_extraction_no_known_fields() {
        let text = r#"{"prompt": "hello", "model": "llama"}"#;
        assert_eq!(extract_hash_key_from_body(Some(text)), None);
    }

    // === extract_hash_key (top-level) tests ===

    #[test]
    fn test_hash_key_headers_over_body() {
        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "from-header".to_string());
        let body = r#"{"session_id": "from-body"}"#;

        let key = extract_hash_key(Some(body), Some(&headers));
        assert_eq!(key, "header:x-session-id:from-header");
    }

    #[test]
    fn test_hash_key_fallback_short_text() {
        let key = extract_hash_key(Some("short"), None);
        assert_eq!(key, "request:short");
    }

    #[test]
    fn test_hash_key_fallback_long_text() {
        let long_text = "x".repeat(200);
        let key = extract_hash_key(Some(&long_text), None);
        assert!(key.starts_with("request_hash:"));
        assert_eq!(key.len(), "request_hash:".len() + 16); // 16 hex chars

        // Same long text should produce same hash
        let key2 = extract_hash_key(Some(&long_text), None);
        assert_eq!(key, key2);
    }

    #[test]
    fn test_hash_key_fallback_none() {
        let key = extract_hash_key(None, None);
        assert_eq!(key, "request:");
    }

    // === find_field_start tests ===

    #[test]
    fn test_find_field_start_double_quoted() {
        let text = r#"{"field": "value"}"#;
        let pos = find_field_start(text, "field");
        assert!(pos.is_some());
        // Should point to after the colon
        let after = &text[pos.unwrap()..];
        assert!(after.trim_start().starts_with('"'));
    }

    #[test]
    fn test_find_field_start_single_quoted() {
        let text = r#"{'field': 'value'}"#;
        let pos = find_field_start(text, "field");
        assert!(pos.is_some());
    }

    #[test]
    fn test_find_field_start_missing() {
        let text = r#"{"other": "value"}"#;
        assert_eq!(find_field_start(text, "field"), None);
    }
}

/// Stable key primitive; no placement or worker ownership lives here.
pub(crate) struct StableHash;
impl StableHash {
    fn murmur_hash_64a(key: &[u8], seed: u32) -> u64 {
        const M: u64 = 0xc6a4a7935bd1e995;
        const R: i32 = 47;

        let mut h = (seed as u64) ^ ((key.len() as u64).wrapping_mul(M));

        // Process 8-byte chunks
        let chunks = key.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut k = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);

            k = k.wrapping_mul(M);
            k ^= k >> R;
            k = k.wrapping_mul(M);

            h ^= k;
            h = h.wrapping_mul(M);
        }

        // Process remaining bytes
        match remainder.len() {
            7 => {
                h ^= (remainder[6] as u64) << 48;
                h ^= (remainder[5] as u64) << 40;
                h ^= (remainder[4] as u64) << 32;
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            6 => {
                h ^= (remainder[5] as u64) << 40;
                h ^= (remainder[4] as u64) << 32;
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            5 => {
                h ^= (remainder[4] as u64) << 32;
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            4 => {
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            3 => {
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            2 => {
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            1 => {
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            _ => {}
        }

        h ^= h >> R;
        h = h.wrapping_mul(M);
        h ^= h >> R;

        h
    }

    /// Optimized MurmurHash64A for rehashing uint64_t keys (from hash.c)
    fn murmur_rehash_64a(k: u64) -> u64 {
        const M: u64 = 0xc6a4a7935bd1e995;
        const R: i32 = 47;
        const SEED: u64 = 4193360111; // From hash.c

        let mut h = SEED ^ (8u64.wrapping_mul(M)); // sizeof(uint64_t) * M

        let mut k = k;
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);

        h ^= k;
        h = h.wrapping_mul(M);

        h ^= h >> R;
        h = h.wrapping_mul(M);
        h ^= h >> R;

        h
    }

    /// FurcHash implementation from Facebook's mcrouter/lib/fbi/hash.c
    /// This is the actual consistent hash function used by Facebook
    fn furc_hash(key: &str, m: u32) -> u32 {
        const MAX_TRIES: u32 = 32;
        const FURC_SHIFT: u32 = 23;
        const FURC_CACHE_SIZE: usize = 1024;

        if m <= 1 {
            return 0;
        }

        let key_bytes = key.as_bytes();
        let mut hash_cache: Vec<u64> = Vec::with_capacity(FURC_CACHE_SIZE);
        let mut old_ord: i32 = -1;

        // Calculate d = ceil(log2(m))
        let mut d = 0;
        while m > (1u32 << d) {
            d += 1;
        }

        let mut a = d;
        for _try in 0..MAX_TRIES {
            // Generate bits until we find a 1 bit or run out of depth
            while !Self::furc_get_bit(key_bytes, a, &mut hash_cache, &mut old_ord) {
                if d == 0 {
                    return 0;
                }
                d -= 1;
                a = d;
            }

            a += FURC_SHIFT;
            let mut num = 1u32;

            // Build the number bit by bit
            for _ in 0..d.saturating_sub(1) {
                num = (num << 1)
                    | Self::furc_get_bit(key_bytes, a, &mut hash_cache, &mut old_ord) as u32;
                a += FURC_SHIFT;
            }

            if num < m {
                return num;
            }
        }

        // Give up and return 0, which is always a legal value
        0
    }

    /// Bit generator function from furc_hash (from hash.c)
    fn furc_get_bit(key: &[u8], idx: u32, hash_cache: &mut Vec<u64>, old_ord: &mut i32) -> bool {
        const SEED: u32 = 4193360111;

        let ord = (idx >> 6) as i32; // idx / 64

        // Extend hash cache if needed
        if *old_ord < ord {
            for n in (*old_ord + 1)..=ord {
                let hash_value = if n == 0 {
                    Self::murmur_hash_64a(key, SEED)
                } else {
                    Self::murmur_rehash_64a(hash_cache[(n - 1) as usize])
                };

                // Ensure cache has enough capacity
                if hash_cache.len() <= n as usize {
                    hash_cache.resize((n as usize) + 1, 0);
                }
                hash_cache[n as usize] = hash_value;
            }
            *old_ord = ord;
        }

        // Extract the bit
        let hash_val = hash_cache[ord as usize];
        let bit_pos = idx & 0x3f; // idx % 64
        (hash_val >> bit_pos) & 0x1 != 0
    }

    /// Facebook-style hash function using furc_hash for consistent hashing
    pub fn fbi_hash(key: &str) -> u64 {
        // Use furc_hash with a large modulus to get good distribution
        // Then expand to u64 for our hash ring
        const LARGE_MODULUS: u32 = (1u32 << 23) - 1; // Max furc_hash modulus

        let furc_result = Self::furc_hash(key, LARGE_MODULUS);

        // Expand the 23-bit furc result to 64-bit by using MurmurHash
        // This gives us the full 64-bit space while preserving consistency
        Self::murmur_hash_64a(
            &furc_result.to_le_bytes(),
            4193360111, // Same seed as furc_hash
        )
    }
}
