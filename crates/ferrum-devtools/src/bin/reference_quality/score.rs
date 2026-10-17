//! Exact scoring rules from the frozen quality_gate's final_marker_exact_rational_v1
//! and trimmed_uppercase_letter_v1. No statistical gate is evaluated here.
use super::input::Task;

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
fn decimal(text: &str) -> Option<(i128, i128)> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let (negative, text) = if let Some(s) = text.strip_prefix('-') {
        (true, s)
    } else {
        (false, text.strip_prefix('+').unwrap_or(text))
    };
    let (integer, fraction) = text.split_once('.').map_or((text, ""), |(i, f)| (i, f));
    if text.matches('.').count() > 1
        || (integer.is_empty() && fraction.is_empty())
        || !fraction.bytes().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let chunks: Vec<_> = integer.split(',').collect();
    if chunks.len() > 1
        && (chunks[0].is_empty()
            || chunks[0].len() > 3
            || chunks.iter().skip(1).any(|c| c.len() != 3))
    {
        return None;
    }
    if !chunks.iter().all(|c| c.bytes().all(|b| b.is_ascii_digit())) {
        return None;
    }
    let digits = format!("{}{}", chunks.concat(), fraction);
    if digits.is_empty() || digits.len() > 36 {
        return None;
    }
    let mut n: i128 = digits.parse().ok()?;
    if negative {
        n = -n;
    }
    let d = 10_i128.checked_pow(fraction.len().try_into().ok()?)?;
    Some((n, d))
}
fn rational(text: &str) -> Option<String> {
    let parts: Vec<_> = text.trim().split('/').collect();
    if parts.len() > 2 {
        return None;
    }
    let (mut n, mut d) = decimal(parts[0])?;
    if parts.len() == 2 {
        let (a, b) = decimal(parts[1])?;
        if a == 0 {
            return None;
        }
        n = n.checked_mul(b)?;
        d = d.checked_mul(a)?;
    }
    if d < 0 {
        n = n.checked_neg()?;
        d = d.checked_neg()?;
    }
    let g = gcd(n.checked_abs()?, d);
    Some(format!("{}/{}", n / g, d / g))
}
fn math_answer(text: &str) -> Option<String> {
    if text.matches("####").count() != 1 {
        return None;
    }
    let line = text.lines().rev().find(|l| !l.trim().is_empty())?.trim();
    rational(line.strip_prefix("####")?.trim())
}
fn choice_answer(text: &str) -> Option<String> {
    match text.trim() {
        "A" | "B" | "C" | "D" => Some(text.trim().to_owned()),
        _ => None,
    }
}
pub fn grade(task: &Task, text: &str, finish: &str, expected: &str) -> bool {
    finish == "stop"
        && match task {
            Task::Math => math_answer(text),
            Task::Choice => choice_answer(text),
        }
        .as_deref()
            == Some(expected)
}
