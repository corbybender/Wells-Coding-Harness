use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Class = 1,
    Function = 2,
    Method = 3,
    Variable = 4,
    Module = 5,
}

impl fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolKind::Class => write!(f, "class"),
            SymbolKind::Function => write!(f, "function"),
            SymbolKind::Method => write!(f, "method"),
            SymbolKind::Variable => write!(f, "variable"),
            SymbolKind::Module => write!(f, "module"),
        }
    }
}

impl SymbolKind {
    pub fn from_u32(n: u32) -> Option<Self> {
        match n {
            1 => Some(SymbolKind::Class),
            2 => Some(SymbolKind::Function),
            3 => Some(SymbolKind::Method),
            4 => Some(SymbolKind::Variable),
            5 => Some(SymbolKind::Module),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Calls = 1,
    References = 2,
    Inherits = 3,
    Imports = 4,
}

impl fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeKind::Calls => write!(f, "calls"),
            EdgeKind::References => write!(f, "references"),
            EdgeKind::Inherits => write!(f, "inherits"),
            EdgeKind::Imports => write!(f, "imports"),
        }
    }
}

impl EdgeKind {
    pub fn from_u32(n: u32) -> Option<Self> {
        match n {
            1 => Some(EdgeKind::Calls),
            2 => Some(EdgeKind::References),
            3 => Some(EdgeKind::Inherits),
            4 => Some(EdgeKind::Imports),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub start_byte: u32,
    pub end_byte: u32,
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from_name: String,
    pub to_name: String,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone)]
pub struct FileIndex {
    pub path: String,
    pub lang: u8,
    pub hash: [u8; 32],
    pub symbols: Vec<Symbol>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone)]
pub struct SymbolMatch {
    pub file_path: String,
    pub name: String,
    pub kind: String,
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone)]
pub struct EdgeMatch {
    pub file_path: String,
    pub name: String,
    pub kind: String,
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone)]
pub struct IndexStats {
    pub files_indexed: usize,
    pub symbols_extracted: usize,
    pub edges_extracted: usize,
    pub total_files: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct RepoStats {
    pub total_files: usize,
    pub total_symbols: usize,
    pub total_edges: usize,
    pub last_indexed_at: i64,
}
