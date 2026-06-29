use crate::graph::{EdgeMatch, FileIndex, RepoStats, SymbolMatch};
use anyhow::anyhow;
use rusqlite::{params, Connection, Result as SqlResult};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        // TODO: Handle LZ4 decompression if path.lz4 exists
        let conn = Connection::open(path)?;
        Ok(Store { conn })
    }

    pub fn create_schema(&self) -> anyhow::Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                hash BLOB NOT NULL,
                lang INTEGER NOT NULL,
                last_indexed INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS names (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                name_id INTEGER NOT NULL REFERENCES names(id),
                kind INTEGER NOT NULL,
                start_byte INTEGER NOT NULL,
                end_byte INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY,
                from_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
                to_name_id INTEGER NOT NULL REFERENCES names(id),
                kind INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(name_id);
            CREATE INDEX IF NOT EXISTS idx_edge_to ON edges(to_name_id);
            CREATE INDEX IF NOT EXISTS idx_edge_from ON edges(from_id);
            ",
        )?;
        Ok(())
    }

    pub fn get_file_hash(&self, path: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let result = self.conn.query_row(
            "SELECT hash FROM files WHERE path = ?1",
            params![path],
            |row| row.get::<_, Vec<u8>>(0),
        );

        match result {
            Ok(hash) => Ok(Some(hash)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(anyhow!(e)),
        }
    }

    pub fn upsert_file(&self, fi: &FileIndex) -> anyhow::Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs() as i64;

        // Delete old file entry if exists
        self.conn.execute(
            "DELETE FROM files WHERE path = ?1",
            params![&fi.path],
        )?;

        // Insert new file
        self.conn.execute(
            "INSERT INTO files (path, hash, lang, last_indexed) VALUES (?1, ?2, ?3, ?4)",
            params![&fi.path, fi.hash.to_vec(), fi.lang, now],
        )?;

        let file_id: i64 = self.conn.query_row(
            "SELECT id FROM files WHERE path = ?1",
            params![&fi.path],
            |row| row.get(0),
        )?;

        // Insert symbols and build name interning
        for symbol in &fi.symbols {
            // Get or insert name
            let name_id: i64 = self.conn.query_row(
                "INSERT OR IGNORE INTO names (name) VALUES (?1) RETURNING id;
                 SELECT id FROM names WHERE name = ?1",
                params![&symbol.name],
                |row| row.get(0),
            ).or_else(|_| {
                self.conn.query_row(
                    "SELECT id FROM names WHERE name = ?1",
                    params![&symbol.name],
                    |row| row.get(0),
                )
            })?;

            self.conn.execute(
                "INSERT INTO symbols (file_id, name_id, kind, start_byte, end_byte, start_line, end_line)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    file_id,
                    name_id,
                    symbol.kind as u32,
                    symbol.start_byte,
                    symbol.end_byte,
                    symbol.start_line,
                    symbol.end_line
                ],
            )?;
        }

        // Insert edges
        for edge in &fi.edges {
            // Get or insert from/to names
            let from_name_id: i64 = self.conn.query_row(
                "INSERT OR IGNORE INTO names (name) VALUES (?1) RETURNING id;
                 SELECT id FROM names WHERE name = ?1",
                params![&edge.from_name],
                |row| row.get(0),
            ).or_else(|_| {
                self.conn.query_row(
                    "SELECT id FROM names WHERE name = ?1",
                    params![&edge.from_name],
                    |row| row.get(0),
                )
            })?;

            let to_name_id: i64 = self.conn.query_row(
                "INSERT OR IGNORE INTO names (name) VALUES (?1) RETURNING id;
                 SELECT id FROM names WHERE name = ?1",
                params![&edge.to_name],
                |row| row.get(0),
            ).or_else(|_| {
                self.conn.query_row(
                    "SELECT id FROM names WHERE name = ?1",
                    params![&edge.to_name],
                    |row| row.get(0),
                )
            })?;

            self.conn.execute(
                "INSERT INTO edges (from_id, to_name_id, kind) VALUES
                 ((SELECT id FROM symbols WHERE name_id = ?1 AND file_id = ?2 LIMIT 1), ?3, ?4)",
                params![from_name_id, file_id, to_name_id, edge.kind as u32],
            )?;
        }

        Ok(())
    }

    pub fn find_symbol(&self, name: &str) -> anyhow::Result<Vec<SymbolMatch>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.path, n.name, s.kind, s.start_line, s.end_line
             FROM symbols s
             JOIN files f ON s.file_id = f.id
             JOIN names n ON s.name_id = n.id
             WHERE n.name = ?1",
        )?;

        let matches = stmt
            .query_map(params![name], |row| {
                Ok(SymbolMatch {
                    file_path: row.get(0)?,
                    name: row.get(1)?,
                    kind: format_kind(row.get(2)?),
                    start_line: row.get(3)?,
                    end_line: row.get(4)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(matches)
    }

    pub fn find_references(&self, symbol: &str) -> anyhow::Result<Vec<EdgeMatch>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.path, n.name, e.kind, s.start_line, s.end_line
             FROM edges e
             JOIN symbols s ON e.from_id = s.id
             JOIN files f ON s.file_id = f.id
             JOIN names n ON e.to_name_id = n.id
             WHERE n.name = ?1",
        )?;

        let matches = stmt
            .query_map(params![symbol], |row| {
                Ok(EdgeMatch {
                    file_path: row.get(0)?,
                    name: row.get(1)?,
                    kind: format_edge_kind(row.get(2)?),
                    start_line: row.get(3)?,
                    end_line: row.get(4)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(matches)
    }

    pub fn find_callers(&self, symbol: &str) -> anyhow::Result<Vec<EdgeMatch>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.path, n.name, e.kind, s.start_line, s.end_line
             FROM edges e
             JOIN symbols s ON e.from_id = s.id
             JOIN files f ON s.file_id = f.id
             JOIN names n ON s.name_id = n.id
             WHERE e.to_name_id = (SELECT id FROM names WHERE name = ?1)
             AND e.kind = 1",
        )?;

        let matches = stmt
            .query_map(params![symbol], |row| {
                Ok(EdgeMatch {
                    file_path: row.get(0)?,
                    name: row.get(1)?,
                    kind: format_edge_kind(row.get(2)?),
                    start_line: row.get(3)?,
                    end_line: row.get(4)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(matches)
    }

    pub fn search_symbols(&self, prefix: &str, limit: usize) -> anyhow::Result<Vec<SymbolMatch>> {
        let pattern = format!("{}%", prefix);
        let mut stmt = self.conn.prepare(
            "SELECT f.path, n.name, s.kind, s.start_line, s.end_line
             FROM symbols s
             JOIN files f ON s.file_id = f.id
             JOIN names n ON s.name_id = n.id
             WHERE n.name LIKE ?1
             LIMIT ?2",
        )?;

        let matches = stmt
            .query_map(params![pattern, limit], |row| {
                Ok(SymbolMatch {
                    file_path: row.get(0)?,
                    name: row.get(1)?,
                    kind: format_kind(row.get(2)?),
                    start_line: row.get(3)?,
                    end_line: row.get(4)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(matches)
    }

    pub fn list_in_file(&self, path: &str) -> anyhow::Result<Vec<SymbolMatch>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.path, n.name, s.kind, s.start_line, s.end_line
             FROM symbols s
             JOIN files f ON s.file_id = f.id
             JOIN names n ON s.name_id = n.id
             WHERE f.path = ?1",
        )?;

        let matches = stmt
            .query_map(params![path], |row| {
                Ok(SymbolMatch {
                    file_path: row.get(0)?,
                    name: row.get(1)?,
                    kind: format_kind(row.get(2)?),
                    start_line: row.get(3)?,
                    end_line: row.get(4)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(matches)
    }

    pub fn stats(&self) -> anyhow::Result<RepoStats> {
        let total_files: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?;

        let total_symbols: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM symbols", [], |row| row.get(0))?;

        let total_edges: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))?;

        let last_indexed_at: i64 = self
            .conn
            .query_row(
                "SELECT MAX(last_indexed) FROM files",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        Ok(RepoStats {
            total_files,
            total_symbols,
            total_edges,
            last_indexed_at,
        })
    }

    pub fn remove_deleted_files(&self, existing: &[String]) -> anyhow::Result<()> {
        for path in existing {
            self.conn.execute(
                "DELETE FROM files WHERE path = ?1",
                params![path],
            )?;
        }
        Ok(())
    }

    pub fn close(self) -> anyhow::Result<()> {
        self.conn.close().map_err(|(_, e)| anyhow!(e))
    }
}

fn format_kind(kind: u32) -> String {
    match kind {
        1 => "class".to_string(),
        2 => "function".to_string(),
        3 => "method".to_string(),
        4 => "variable".to_string(),
        5 => "module".to_string(),
        _ => "unknown".to_string(),
    }
}

fn format_edge_kind(kind: u32) -> String {
    match kind {
        1 => "calls".to_string(),
        2 => "references".to_string(),
        3 => "inherits".to_string(),
        4 => "imports".to_string(),
        _ => "unknown".to_string(),
    }
}
