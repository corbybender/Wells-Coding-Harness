use crate::graph::{EdgeMatch, SymbolMatch};

pub struct QueryResult {
    pub symbol_matches: Vec<SymbolMatch>,
    pub edge_matches: Vec<EdgeMatch>,
}

pub fn format_symbol_match(m: &SymbolMatch) -> String {
    format!(
        "{}:{}:{} ({})",
        m.file_path, m.start_line, m.name, m.kind
    )
}

pub fn format_edge_match(m: &EdgeMatch) -> String {
    format!(
        "{}:{}:{} ({})",
        m.file_path, m.start_line, m.name, m.kind
    )
}
