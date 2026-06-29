use std::path::{Path, PathBuf};

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    // Define all grammars we want to support
    // Each grammar is (directory_name, source_files)
    let grammars = vec![
        ("python", vec!["parser.c"]),
        ("javascript", vec!["parser.c", "scanner.c"]),
        ("typescript/typescript", vec!["parser.c", "scanner.c"]),
        ("go", vec!["parser.c"]),
        ("rust", vec!["parser.c", "scanner.c"]),
        ("java", vec!["parser.c"]),
        ("c", vec!["parser.c"]),
        ("cpp", vec!["parser.c", "scanner.c"]),
    ];

    let grammars_base = PathBuf::from("grammars");

    for (lang_dir, source_files) in grammars {
        let grammar_src_dir = grammars_base.join(&lang_dir).join("src");

        // Check if grammar source directory exists
        if !grammar_src_dir.exists() {
            eprintln!(
                "[build-warning] Grammar source not found at {:?}",
                grammar_src_dir
            );
            eprintln!(
                "[build-warning] Skipping compilation for grammar: {}",
                lang_dir
            );
            eprintln!("[build-warning] See README.md for setup instructions");
            continue;
        }

        // Build the grammar
        let mut builder = cc::Build::new();

        // Add include directory
        builder.include(&grammar_src_dir);

        // Add all source files
        let mut has_all_files = true;
        for source_file in &source_files {
            let src_path = grammar_src_dir.join(source_file);
            if !src_path.exists() {
                eprintln!(
                    "[build-warning] Source file not found: {:?}",
                    src_path
                );
                has_all_files = false;
                break;
            }
            builder.file(&src_path);
        }

        if !has_all_files {
            eprintln!("[build-warning] Skipping grammar: {}", lang_dir);
            continue;
        }

        // Compile with optimizations
        builder
            .opt_level(3)
            .warnings(false) // tree-sitter generates warnings; suppress them
            .compile(&format!("tree-sitter-{}", lang_dir.replace('/', "-")));

        println!("[build-info] Compiled grammar: {}", lang_dir);
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=grammars");
}
