//! The source accepts two forms. In particular it does not require every
//! section to be nonempty, ASCII-only entity numbers, or a particular prose
//! language. Keep those parser rules separate from any product preference.
use super::python_space;
use std::sync::OnceLock;
use tokenizers::utils::SysRegex;

const SECTIONS: [&str; 6] = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PH3ImageEditPromptForm {
    Brief,
    Instruction,
}

pub fn ph3_image_edit_prompt_form(prompt: &str) -> Option<PH3ImageEditPromptForm> {
    let text = prompt.trim_matches(python_space);
    // Python's MULTILINE ^/$ recognize LF here. Retain CR instead of using
    // Rust lines(), which would silently accept CRLF headers rejected by re.
    let mut cursor = 0;
    let mut headers = Vec::with_capacity(SECTIONS.len());
    for line in text.split_inclusive('\n') {
        let body = line.strip_suffix('\n').unwrap_or(line);
        let name = body.trim_end_matches([' ', '\t']).strip_suffix(':');
        if let Some(name) = name.filter(|name| SECTIONS.contains(name)) {
            if headers.len() == SECTIONS.len() {
                return None;
            }
            headers.push((name, cursor, cursor + body.len()));
        }
        cursor += line.len();
    }
    if headers.is_empty() {
        return (text.starts_with("Edit <Picture 1>.")
            && text.contains("Keep")
            && !text.contains('\n'))
        .then_some(PH3ImageEditPromptForm::Instruction);
    }
    if headers.len() != SECTIONS.len()
        || !headers
            .iter()
            .zip(SECTIONS)
            .all(|((name, _, _), expected)| *name == expected)
        || text.matches("[keyframe completion]").count() != 1
    {
        return None;
    }
    let retention = &text[headers[2].2..headers[3].1];
    let mut lines = retention
        .split(python_line_break)
        .map(|line| line.trim_matches(python_space))
        .filter(|line| !line.is_empty());
    let first = lines.next()?;
    (retention_line(first) && lines.all(retention_line)).then_some(PH3ImageEditPromptForm::Brief)
}
fn python_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}'..='\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}
fn retention_line(line: &str) -> bool {
    static PATTERN: OnceLock<SysRegex> = OnceLock::new();
    // Nd preserves Python's Unicode \d. Its \s additionally includes the
    // information separators, rather than imposing a new ASCII-only grammar.
    let pattern = PATTERN.get_or_init(|| SysRegex::new(
        r"\A<(Picture|Subject) \p{Nd}+>(?: \([^)]*\))?:[\s\x1c-\x1f]*(fully_preserved|partially_preserved|attribute_transfer|weak_reference)[\s\x1c-\x1f]*[-–—][\s\x1c-\x1f]*.*\z"
    ).expect("static official retention grammar"));
    pattern.find_iter(line).next() == Some((0, line.len()))
}
