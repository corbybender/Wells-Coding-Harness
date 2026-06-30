use crate::detect::Language;
use crate::graph::{FileIndex, Symbol, SymbolKind};
use regex::Regex;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Lazy-compiled regex patterns (one set per language, compiled once)
// ---------------------------------------------------------------------------

struct LangPatterns {
    /// Each entry: (regex, capture_group_for_name, kind_if_indented, kind_if_toplevel)
    /// kind_if_indented is used when indent > 0 (e.g. Python method vs function).
    defs: Vec<(Regex, usize, SymbolKind, SymbolKind)>,
}

static PY_PATS: OnceLock<LangPatterns> = OnceLock::new();
static JS_PATS: OnceLock<LangPatterns> = OnceLock::new();
static TS_PATS: OnceLock<LangPatterns> = OnceLock::new();
static RS_PATS: OnceLock<LangPatterns> = OnceLock::new();
static GO_PATS: OnceLock<LangPatterns> = OnceLock::new();
static JAVA_PATS: OnceLock<LangPatterns> = OnceLock::new();
static C_PATS: OnceLock<LangPatterns> = OnceLock::new();
static CPP_PATS: OnceLock<LangPatterns> = OnceLock::new();

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

fn py_patterns() -> &'static LangPatterns {
    PY_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            // class Foo / class Foo(Bar):
            (
                Regex::new(r"^(\s*)class\s+([A-Za-z_]\w*)").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // def foo / async def foo
            (
                Regex::new(r"^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
        ],
    })
}

fn js_patterns() -> &'static LangPatterns {
    JS_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            // class Foo / export class Foo
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$]\w*)").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // function foo() / export function foo() / async function foo()
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$]\w*)\s*\(").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
            // const foo = function / const foo = async function
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$]\w*)\s*=\s*(?:async\s+)?function").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
            // const foo = (...) => / const foo = async (...) =>
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$]\w*)\s*=\s*(?:async\s+)?\(?[^=]*\)?\s*=>").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
            // method shorthand inside object/class: foo(...) { or async foo() {
            (
                Regex::new(r"^(\s+)(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?([A-Za-z_$]\w*)\s*\([^)]*\)\s*\{").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
        ],
    })
}

fn ts_patterns() -> &'static LangPatterns {
    TS_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            // class / abstract class
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$]\w*)").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // interface
            (
                Regex::new(r"^(\s*)(?:export\s+)?interface\s+([A-Za-z_$]\w*)").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // type alias: export type Foo =
            (
                Regex::new(r"^(\s*)(?:export\s+)?type\s+([A-Za-z_$]\w*)\s*(?:<[^>]*>)?\s*=").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // function / export function / async function
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$]\w*)\s*[\(<]").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
            // const foo = function / arrow
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$]\w*)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?(?:function|\()").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
            // class method shorthand
            (
                Regex::new(r"^(\s+)(?:public|private|protected|static|abstract|readonly|async|override|\s)*\s+([A-Za-z_$]\w*)\s*\([^)]*\)").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
        ],
    })
}

fn rs_patterns() -> &'static LangPatterns {
    RS_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            // struct Foo / pub struct Foo / pub(crate) struct Foo
            (
                Regex::new(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?struct\s+([A-Za-z_]\w*)").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // enum Foo
            (
                Regex::new(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?enum\s+([A-Za-z_]\w*)").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // trait Foo
            (
                Regex::new(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?trait\s+([A-Za-z_]\w*)").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // impl Foo / impl<T> Foo<T> (record as Class so "find_symbol Foo" finds the impl)
            (
                Regex::new(r"^(\s*)impl(?:<[^>]*>)?\s+(?:[A-Za-z_]\w*(?:<[^>]*>)?\s+for\s+)?([A-Za-z_]\w*)").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // fn foo / pub fn foo / pub(crate) async fn foo
            (
                Regex::new(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
        ],
    })
}

fn go_patterns() -> &'static LangPatterns {
    GO_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            // type Foo struct / type Foo interface
            (
                Regex::new(r"^(\s*)type\s+([A-Za-z_]\w*)\s+(?:struct|interface)").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // func Foo(...) / func (r *Recv) Foo(...)
            (
                Regex::new(r"^(\s*)func\s+(?:\([^)]+\)\s+)?([A-Za-z_]\w*)\s*\(").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
        ],
    })
}

fn java_patterns() -> &'static LangPatterns {
    JAVA_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            // class/interface/enum declaration
            (
                Regex::new(r"^(\s*)(?:(?:public|private|protected|static|final|abstract|sealed)\s+)*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // method: <modifiers> <return_type> methodName(
            (
                Regex::new(r"^(\s{4,})(?:(?:public|private|protected|static|final|synchronized|abstract|native|default|override)\s+)*(?:[\w<>\[\]]+\s+)+([A-Za-z_]\w*)\s*\(").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Method,
            ),
        ],
    })
}

fn c_patterns() -> &'static LangPatterns {
    C_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            // C function: return_type name(... opening brace on same or next line
            // Heuristic: line starts non-whitespace, has word chars, then name(
            (
                Regex::new(r"^(\s*)(?:static\s+|extern\s+|inline\s+|const\s+)*(?:unsigned\s+|signed\s+|long\s+|short\s+)?(?:struct\s+)?[A-Za-z_]\w*\s*\*?\s*([A-Za-z_]\w*)\s*\([^;)]*\)\s*(?:\{|$)").unwrap(),
                2,
                SymbolKind::Function,
                SymbolKind::Function,
            ),
            // struct/union/enum tag
            (
                Regex::new(r"^(\s*)typedef\s+(?:struct|union|enum)\s+(?:[A-Za-z_]\w*)?\s*\{").unwrap(),
                0, // no name capture at definition — skip
                SymbolKind::Class,
                SymbolKind::Class,
            ),
        ],
    })
}

fn cpp_patterns() -> &'static LangPatterns {
    CPP_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            // class / struct at top level
            (
                Regex::new(r"^(\s*)(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+([A-Za-z_]\w*)(?:\s*:|[^;])").unwrap(),
                2,
                SymbolKind::Class,
                SymbolKind::Class,
            ),
            // function / method (same heuristic as C)
            (
                Regex::new(r"^(\s*)(?:(?:virtual|static|inline|explicit|friend|constexpr|override|const)\s+)*(?:[\w:*&<>]+\s+)+([A-Za-z_~]\w*)\s*\([^;]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:\{|:|$)").unwrap(),
                2,
                SymbolKind::Method,
                SymbolKind::Function,
            ),
        ],
    })
}

// ---------------------------------------------------------------------------
// Core parsing engine
// ---------------------------------------------------------------------------

pub fn parse_file(
    source: &[u8],
    path: &str,
    lang: Language,
    hash: [u8; 32],
) -> anyhow::Result<FileIndex> {
    let text = match std::str::from_utf8(source) {
        Ok(s) => s,
        Err(_) => {
            return Ok(FileIndex {
                path: path.to_string(),
                lang: lang.to_u8(),
                hash,
                symbols: Vec::new(),
                edges: Vec::new(),
            });
        }
    };

    let patterns = match lang {
        Language::Python => py_patterns(),
        Language::JavaScript => js_patterns(),
        Language::TypeScript => ts_patterns(),
        Language::Rust => rs_patterns(),
        Language::Go => go_patterns(),
        Language::Java => java_patterns(),
        Language::C => c_patterns(),
        Language::Cpp => cpp_patterns(),
    };

    let symbols = extract_symbols(text, patterns);

    Ok(FileIndex {
        path: path.to_string(),
        lang: lang.to_u8(),
        hash,
        symbols,
        edges: Vec::new(), // edge extraction is a future enhancement
    })
}

fn extract_symbols(text: &str, patterns: &LangPatterns) -> Vec<Symbol> {
    let mut symbols = Vec::new();

    for (line_num, line) in text.lines().enumerate() {
        let line_no = (line_num + 1) as u32; // 1-based
        let byte_offset = byte_offset_of_line(text, line_num) as u32;

        for (re, name_cap, kind_indented, kind_toplevel) in &patterns.defs {
            if let Some(caps) = re.captures(line) {
                // name_cap 0 means no useful capture (e.g. typedef stubs) → skip
                if *name_cap == 0 {
                    continue;
                }
                if let Some(name_match) = caps.get(*name_cap) {
                    let name = name_match.as_str().to_string();
                    // Skip names that look like keywords or are too short
                    if name.len() < 2 || is_keyword(&name) {
                        continue;
                    }
                    let ind = indent_of(line);
                    let kind = if ind == 0 { *kind_toplevel } else { *kind_indented };
                    symbols.push(Symbol {
                        name,
                        kind,
                        start_byte: byte_offset,
                        end_byte: byte_offset,
                        start_line: line_no,
                        end_line: line_no,
                    });
                    break; // first matching pattern wins for this line
                }
            }
        }
    }

    symbols
}

fn byte_offset_of_line(text: &str, target_line: usize) -> usize {
    let mut offset = 0;
    for (i, line) in text.split('\n').enumerate() {
        if i == target_line {
            return offset;
        }
        offset += line.len() + 1; // +1 for the \n
    }
    offset
}

fn is_keyword(name: &str) -> bool {
    matches!(
        name,
        "if" | "else" | "for" | "while" | "do" | "switch" | "case"
            | "return" | "break" | "continue" | "new" | "delete"
            | "try" | "catch" | "finally" | "throw" | "typeof"
            | "instanceof" | "in" | "of" | "this" | "self" | "super"
            | "true" | "false" | "null" | "None" | "True" | "False"
            | "and" | "or" | "not" | "is" | "as" | "with" | "pass"
            | "lambda" | "yield" | "from" | "import" | "raise" | "assert"
            | "global" | "nonlocal" | "del" | "print" | "exec"
            | "main" | "test" | "init" | "new"
    )
}
