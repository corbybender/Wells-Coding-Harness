use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Python = 1,
    JavaScript = 2,
    TypeScript = 3,
    Go = 4,
    Rust = 5,
    Java = 6,
    C = 7,
    Cpp = 8,
}

impl Language {
    pub fn to_u8(&self) -> u8 {
        *self as u8
    }

    pub fn from_u8(n: u8) -> Option<Self> {
        match n {
            1 => Some(Language::Python),
            2 => Some(Language::JavaScript),
            3 => Some(Language::TypeScript),
            4 => Some(Language::Go),
            5 => Some(Language::Rust),
            6 => Some(Language::Java),
            7 => Some(Language::C),
            8 => Some(Language::Cpp),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Language::Python => "python",
            Language::JavaScript => "javascript",
            Language::TypeScript => "typescript",
            Language::Go => "go",
            Language::Rust => "rust",
            Language::Java => "java",
            Language::C => "c",
            Language::Cpp => "cpp",
        }
    }
}

pub fn detect_language(path: &Path) -> Option<Language> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    match ext.as_str() {
        "py" => Some(Language::Python),
        "js" | "mjs" | "cjs" => Some(Language::JavaScript),
        "ts" | "tsx" => Some(Language::TypeScript),
        "go" => Some(Language::Go),
        "rs" => Some(Language::Rust),
        "java" => Some(Language::Java),
        "c" | "h" => Some(Language::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "c++" => Some(Language::Cpp),
        _ => None,
    }
}

pub fn is_source_file(path: &Path) -> bool {
    detect_language(path).is_some()
}

pub fn should_skip_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".venv"
            | "venv"
            | "node_modules"
            | "target"
            | "__pycache__"
            | ".pytest_cache"
            | ".wells_index"
            | "build"
            | "dist"
            | ".tox"
            | ".eggs"
            | "*.egg-info"
    )
}
