use crate::detect::{detect_language, is_source_file, should_skip_dir};
use crate::graph::{EdgeMatch, IndexStats, SymbolMatch};
use crate::hash::blake3_file;
use crate::parse::parse_file;
use crate::store::Store;
use anyhow::anyhow;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

pub struct IndexEngine {
    workspace: PathBuf,
    store: Store,
}

impl IndexEngine {
    pub fn new(workspace: &str) -> anyhow::Result<Self> {
        let workspace_path = PathBuf::from(workspace);
        if !workspace_path.exists() {
            return Err(anyhow!("Workspace path does not exist: {}", workspace));
        }

        let index_dir = workspace_path.join(".wells_index");
        std::fs::create_dir_all(&index_dir)?;

        let db_path = index_dir.join("index.db");
        let mut store = Store::open(&db_path)?;
        store.create_schema()?;

        Ok(IndexEngine {
            workspace: workspace_path,
            store,
        })
    }

    pub fn index(&mut self) -> anyhow::Result<IndexStats> {
        let start = Instant::now();
        let mut files_indexed = 0;
        let mut symbols_extracted = 0;
        let mut edges_extracted = 0;

        // Scan workspace for source files
        for entry in WalkDir::new(&self.workspace)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| {
                // Skip hidden dirs and known excluded dirs
                !e.path()
                    .components()
                    .any(|c| {
                        if let std::path::Component::Normal(n) = c {
                            if let Some(name) = n.to_str() {
                                return should_skip_dir(name) || name.starts_with('.');
                            }
                        }
                        false
                    })
            })
        {
            let path = entry.path();
            if !is_source_file(path) {
                continue;
            }

            // Check if file needs re-indexing
            if let Ok(Some(stored_hash)) = self.store.get_file_hash(path.to_str().unwrap_or("")) {
                if let Ok(current_hash) = blake3_file(path) {
                    if stored_hash == current_hash.to_vec() {
                        continue; // Skip unchanged file
                    }
                }
            }

            // Index this file
            if let Ok(source) = std::fs::read(path) {
                if let Some(lang) = detect_language(path) {
                    if let Ok(hash) = blake3_file(path) {
                        let rel_path = path
                            .strip_prefix(&self.workspace)
                            .unwrap_or(path)
                            .to_string_lossy()
                            .to_string();

                        // Parse file (currently returns empty; tree-sitter parsing TODO)
                        if let Ok(file_index) = parse_file(&source, &rel_path, lang, hash) {
                            symbols_extracted += file_index.symbols.len();
                            edges_extracted += file_index.edges.len();

                            if let Ok(()) = self.store.upsert_file(&file_index) {
                                files_indexed += 1;
                            }
                        }
                    }
                }
            }
        }

        let duration = start.elapsed();

        Ok(IndexStats {
            files_indexed,
            symbols_extracted,
            edges_extracted,
            total_files: 0, // Will be filled by stats()
            duration_ms: duration.as_millis() as u64,
        })
    }

    pub fn find_symbol(&self, name: &str) -> anyhow::Result<Vec<SymbolMatch>> {
        self.store.find_symbol(name)
    }

    pub fn find_references(&self, symbol: &str) -> anyhow::Result<Vec<EdgeMatch>> {
        self.store.find_references(symbol)
    }

    pub fn find_callers(&self, symbol: &str) -> anyhow::Result<Vec<EdgeMatch>> {
        self.store.find_callers(symbol)
    }

    pub fn search_symbols(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SymbolMatch>> {
        self.store.search_symbols(query, limit)
    }

    pub fn list_in_file(&self, path: &str) -> anyhow::Result<Vec<SymbolMatch>> {
        self.store.list_in_file(path)
    }

    pub fn stats(&self) -> anyhow::Result<crate::graph::RepoStats> {
        self.store.stats()
    }

    pub fn clear(&mut self) -> anyhow::Result<()> {
        let index_dir = self.workspace.join(".wells_index");
        std::fs::remove_dir_all(&index_dir)?;
        std::fs::create_dir(&index_dir)?;
        Ok(())
    }
}
