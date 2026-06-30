use crate::detect::Language;
use crate::graph::{Edge, EdgeKind, FileIndex, Symbol, SymbolKind};
use regex::Regex;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Lazy-compiled regex patterns
// ---------------------------------------------------------------------------

struct LangPatterns {
    /// (regex, name_capture_group, kind_when_indented, kind_when_toplevel)
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

// Call-site extraction regexes (per language)
static PY_DEF_RE: OnceLock<Regex> = OnceLock::new();
static PY_CALL_RE: OnceLock<Regex> = OnceLock::new();
static JS_DEF_RE: OnceLock<Regex> = OnceLock::new();
static JS_CALL_RE: OnceLock<Regex> = OnceLock::new();
static RS_DEF_RE: OnceLock<Regex> = OnceLock::new();
static RS_CALL_RE: OnceLock<Regex> = OnceLock::new();
static GO_DEF_RE: OnceLock<Regex> = OnceLock::new();
static GO_CALL_RE: OnceLock<Regex> = OnceLock::new();

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

fn py_patterns() -> &'static LangPatterns {
    PY_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            (
                Regex::new(r"^(\s*)class\s+([A-Za-z_]\w*)").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
        ],
    })
}

fn js_patterns() -> &'static LangPatterns {
    JS_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$]\w*)").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$]\w*)\s*\(").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$]\w*)\s*=\s*(?:async\s+)?function").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$]\w*)\s*=\s*(?:async\s+)?\(?[^=]*\)?\s*=>").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
            (
                Regex::new(r"^(\s+)(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?([A-Za-z_$]\w*)\s*\([^)]*\)\s*\{").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
        ],
    })
}

fn ts_patterns() -> &'static LangPatterns {
    TS_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$]\w*)").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)(?:export\s+)?interface\s+([A-Za-z_$]\w*)").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)(?:export\s+)?type\s+([A-Za-z_$]\w*)\s*(?:<[^>]*>)?\s*=").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$]\w*)\s*[\(<]").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
            (
                Regex::new(r"^(\s*)(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$]\w*)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?(?:function|\()").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
            (
                Regex::new(r"^(\s+)(?:public|private|protected|static|abstract|readonly|async|override|\s)*\s+([A-Za-z_$]\w*)\s*\([^)]*\)").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
        ],
    })
}

fn rs_patterns() -> &'static LangPatterns {
    RS_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            (
                Regex::new(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?struct\s+([A-Za-z_]\w*)").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?enum\s+([A-Za-z_]\w*)").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?trait\s+([A-Za-z_]\w*)").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)impl(?:<[^>]*>)?\s+(?:[A-Za-z_]\w*(?:<[^>]*>)?\s+for\s+)?([A-Za-z_]\w*)").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
        ],
    })
}

fn go_patterns() -> &'static LangPatterns {
    GO_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            (
                Regex::new(r"^(\s*)type\s+([A-Za-z_]\w*)\s+(?:struct|interface)").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)func\s+(?:\([^)]+\)\s+)?([A-Za-z_]\w*)\s*\(").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
        ],
    })
}

fn java_patterns() -> &'static LangPatterns {
    JAVA_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            (
                Regex::new(r"^(\s*)(?:(?:public|private|protected|static|final|abstract|sealed)\s+)*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s{4,})(?:(?:public|private|protected|static|final|synchronized|abstract|native|default|override)\s+)*(?:[\w<>\[\]]+\s+)+([A-Za-z_]\w*)\s*\(").unwrap(),
                2, SymbolKind::Method, SymbolKind::Method,
            ),
        ],
    })
}

fn c_patterns() -> &'static LangPatterns {
    C_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            (
                Regex::new(r"^(\s*)(?:static\s+|extern\s+|inline\s+|const\s+)*(?:unsigned\s+|signed\s+|long\s+|short\s+)?(?:struct\s+)?[A-Za-z_]\w*\s*\*?\s*([A-Za-z_]\w*)\s*\([^;)]*\)\s*(?:\{|$)").unwrap(),
                2, SymbolKind::Function, SymbolKind::Function,
            ),
        ],
    })
}

fn cpp_patterns() -> &'static LangPatterns {
    CPP_PATS.get_or_init(|| LangPatterns {
        defs: vec![
            (
                Regex::new(r"^(\s*)(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+([A-Za-z_]\w*)(?:\s*:|[^;])").unwrap(),
                2, SymbolKind::Class, SymbolKind::Class,
            ),
            (
                Regex::new(r"^(\s*)(?:(?:virtual|static|inline|explicit|friend|constexpr|override|const)\s+)*(?:[\w:*&<>]+\s+)+([A-Za-z_~]\w*)\s*\([^;]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:\{|:|$)").unwrap(),
                2, SymbolKind::Method, SymbolKind::Function,
            ),
        ],
    })
}

// ---------------------------------------------------------------------------
// Public entry point
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

    let def_patterns = match lang {
        Language::Python => py_patterns(),
        Language::JavaScript => js_patterns(),
        Language::TypeScript => ts_patterns(),
        Language::Rust => rs_patterns(),
        Language::Go => go_patterns(),
        Language::Java => java_patterns(),
        Language::C => c_patterns(),
        Language::Cpp => cpp_patterns(),
    };

    let symbols = extract_symbols(text, def_patterns);
    let edges = extract_calls(text, lang);

    Ok(FileIndex {
        path: path.to_string(),
        lang: lang.to_u8(),
        hash,
        symbols,
        edges,
    })
}

// ---------------------------------------------------------------------------
// Symbol extraction (definition lines)
// ---------------------------------------------------------------------------

fn extract_symbols(text: &str, patterns: &LangPatterns) -> Vec<Symbol> {
    let mut symbols = Vec::new();

    for (line_num, line) in text.lines().enumerate() {
        let line_no = (line_num + 1) as u32;
        let byte_offset = byte_offset_of_line(text, line_num) as u32;

        for (re, name_cap, kind_indented, kind_toplevel) in &patterns.defs {
            if *name_cap == 0 { continue; }
            if let Some(caps) = re.captures(line) {
                if let Some(name_match) = caps.get(*name_cap) {
                    let name = name_match.as_str().to_string();
                    if name.len() < 2 || is_keyword(&name) { continue; }
                    let kind = if indent_of(line) == 0 { *kind_toplevel } else { *kind_indented };
                    symbols.push(Symbol {
                        name,
                        kind,
                        start_byte: byte_offset,
                        end_byte: byte_offset,
                        start_line: line_no,
                        end_line: line_no,
                    });
                    break;
                }
            }
        }
    }

    symbols
}

// ---------------------------------------------------------------------------
// Call-site extraction (edges)
// ---------------------------------------------------------------------------

fn extract_calls(text: &str, lang: Language) -> Vec<Edge> {
    match lang {
        Language::Python => extract_calls_python(text),
        Language::JavaScript | Language::TypeScript => extract_calls_js(text),
        Language::Rust => extract_calls_rust(text),
        Language::Go => extract_calls_go(text),
        _ => Vec::new(), // Java/C/C++ — future work
    }
}

/// Track indentation-based Python scopes and emit Calls edges.
fn extract_calls_python(text: &str) -> Vec<Edge> {
    let def_re = PY_DEF_RE.get_or_init(|| {
        Regex::new(r"^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(").unwrap()
    });
    // Match bare identifier calls: foo( or self.foo( — capture the function name.
    // Exclude strings, comments handled by skipping comment-only lines.
    let call_re = PY_CALL_RE.get_or_init(|| {
        Regex::new(r"(?:self\.|cls\.)?([A-Za-z_]\w*)\s*\(").unwrap()
    });

    let mut edges = Vec::new();
    // Stack of (function_name, def_indent)
    let mut scope_stack: Vec<(String, usize)> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        // Skip blank lines, comments, docstrings (heuristic)
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
            continue;
        }

        let indent = indent_of(line);

        // Pop scopes we've exited (current indent ≤ scope's def indent means we left it)
        while matches!(scope_stack.last(), Some((_, si)) if indent <= *si) {
            scope_stack.pop();
        }

        if let Some(caps) = def_re.captures(line) {
            let name = caps[2].to_string();
            scope_stack.push((name, indent));
        } else if let Some((func_name, _)) = scope_stack.last() {
            let func_name = func_name.clone();
            for caps in call_re.captures_iter(trimmed) {
                let called = &caps[1];
                if !is_keyword(called) && called.len() > 1 && called != &func_name {
                    edges.push(Edge {
                        from_name: func_name.clone(),
                        to_name: called.to_string(),
                        kind: EdgeKind::Calls,
                    });
                }
            }
        }
    }

    edges
}

/// Brace-counting JavaScript/TypeScript call extraction.
fn extract_calls_js(text: &str) -> Vec<Edge> {
    let def_re = JS_DEF_RE.get_or_init(|| {
        Regex::new(
            r"(?:function\s+([A-Za-z_$]\w*)|(?:const|let|var)\s+([A-Za-z_$]\w*)\s*=\s*(?:async\s+)?(?:function|\()|^\s*(?:async\s+)?([A-Za-z_$]\w*)\s*\([^)]*\)\s*\{)"
        ).unwrap()
    });
    let call_re = JS_CALL_RE.get_or_init(|| {
        Regex::new(r"(?:this\.)?([A-Za-z_$]\w*)\s*\(").unwrap()
    });

    let mut edges = Vec::new();
    // Use brace counting to track function scope
    let mut scope_stack: Vec<(String, i32)> = Vec::new(); // (name, brace_depth_at_entry)
    let mut brace_depth: i32 = 0;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("*") {
            continue;
        }

        // Count braces on this line (ignore strings — heuristic only)
        let open = trimmed.chars().filter(|&c| c == '{').count() as i32;
        let close = trimmed.chars().filter(|&c| c == '}').count() as i32;

        // Pop scopes whose entry depth >= current depth (before adding braces)
        while matches!(scope_stack.last(), Some((_, d)) if brace_depth <= *d && close > 0) {
            scope_stack.pop();
        }

        // Check for a function definition
        if let Some(caps) = def_re.captures(trimmed) {
            let name = caps.get(1)
                .or_else(|| caps.get(2))
                .or_else(|| caps.get(3))
                .map(|m| m.as_str().to_string());
            if let Some(n) = name {
                if n.len() > 1 && !is_keyword(&n) {
                    scope_stack.push((n, brace_depth));
                }
            }
        } else if let Some((func_name, _)) = scope_stack.last() {
            let func_name = func_name.clone();
            for caps in call_re.captures_iter(trimmed) {
                let called = &caps[1];
                if !is_keyword(called) && called.len() > 1 && called != &func_name {
                    edges.push(Edge {
                        from_name: func_name.clone(),
                        to_name: called.to_string(),
                        kind: EdgeKind::Calls,
                    });
                }
            }
        }

        brace_depth += open - close;
        if brace_depth < 0 { brace_depth = 0; }
    }

    edges
}

/// Brace-counting Rust call extraction.
fn extract_calls_rust(text: &str) -> Vec<Edge> {
    let def_re = RS_DEF_RE.get_or_init(|| {
        Regex::new(r"(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)").unwrap()
    });
    let call_re = RS_CALL_RE.get_or_init(|| {
        // Rust calls: foo(, foo::<T>(, Self::foo(, self.foo(
        Regex::new(r"(?:self\.|Self::)?([A-Za-z_]\w*)(?:::<[^>]*>)?\s*\(").unwrap()
    });

    let mut edges = Vec::new();
    let mut scope_stack: Vec<(String, i32)> = Vec::new();
    let mut brace_depth: i32 = 0;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("///") {
            continue;
        }

        let open = trimmed.chars().filter(|&c| c == '{').count() as i32;
        let close = trimmed.chars().filter(|&c| c == '}').count() as i32;

        while matches!(scope_stack.last(), Some((_, d)) if brace_depth <= *d && close > 0) {
            scope_stack.pop();
        }

        if let Some(caps) = def_re.captures(trimmed) {
            let name = caps[1].to_string();
            if name.len() > 1 && !is_keyword(&name) {
                scope_stack.push((name, brace_depth));
            }
        } else if let Some((func_name, _)) = scope_stack.last() {
            let func_name = func_name.clone();
            for caps in call_re.captures_iter(trimmed) {
                let called = &caps[1];
                if !is_keyword(called) && called.len() > 1 && called != &func_name {
                    edges.push(Edge {
                        from_name: func_name.clone(),
                        to_name: called.to_string(),
                        kind: EdgeKind::Calls,
                    });
                }
            }
        }

        brace_depth += open - close;
        if brace_depth < 0 { brace_depth = 0; }
    }

    edges
}

/// Go call extraction (brace-based).
fn extract_calls_go(text: &str) -> Vec<Edge> {
    let def_re = GO_DEF_RE.get_or_init(|| {
        Regex::new(r"^func\s+(?:\([^)]+\)\s+)?([A-Za-z_]\w*)\s*\(").unwrap()
    });
    let call_re = GO_CALL_RE.get_or_init(|| {
        Regex::new(r"(?<![.\w])([A-Za-z_]\w*)\s*\(").unwrap()
    });

    let mut edges = Vec::new();
    let mut scope_stack: Vec<(String, i32)> = Vec::new();
    let mut brace_depth: i32 = 0;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") { continue; }

        let open = trimmed.chars().filter(|&c| c == '{').count() as i32;
        let close = trimmed.chars().filter(|&c| c == '}').count() as i32;

        while matches!(scope_stack.last(), Some((_, d)) if brace_depth <= *d && close > 0) {
            scope_stack.pop();
        }

        if let Some(caps) = def_re.captures(trimmed) {
            let name = caps[1].to_string();
            if name.len() > 1 && !is_keyword(&name) {
                scope_stack.push((name, brace_depth));
            }
        } else if let Some((func_name, _)) = scope_stack.last() {
            let func_name = func_name.clone();
            for caps in call_re.captures_iter(trimmed) {
                let called = &caps[1];
                if !is_keyword(called) && called.len() > 1 && called != &func_name {
                    edges.push(Edge {
                        from_name: func_name.clone(),
                        to_name: called.to_string(),
                        kind: EdgeKind::Calls,
                    });
                }
            }
        }

        brace_depth += open - close;
        if brace_depth < 0 { brace_depth = 0; }
    }

    edges
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn byte_offset_of_line(text: &str, target_line: usize) -> usize {
    let mut offset = 0;
    for (i, line) in text.split('\n').enumerate() {
        if i == target_line { return offset; }
        offset += line.len() + 1;
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
            | "let" | "const" | "var" | "async" | "await"
            | "match" | "Ok" | "Err" | "Some" | "None"
            | "pub" | "fn" | "mut" | "use" | "mod" | "type"
            | "struct" | "enum" | "trait" | "impl" | "where"
            | "assert" | "panic" | "vec" | "map" | "format"
    )
}
