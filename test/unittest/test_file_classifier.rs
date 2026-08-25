use std::path::{Path, PathBuf};

use crate::core::git::diff::classifier::FileClassifier;
use crate::core::git::types::{
    ChangeType, ClassifiedSnapshot, FileCategory, StagedFile, StagedSnapshot,
};
use crate::infra::git::GitRunner;

//  helpers

/// RAII empty directory under `std::env::temp_dir()`, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!("autocommit-test-{name}"));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn git_in(runner: &GitRunner, args: &[&str]) -> String {
    runner
        .run(args, None)
        .await
        .unwrap_or_else(|err| panic!("git {:?} failed: {err}", args))
        .stdout_str()
        .trim()
        .to_owned()
}

/// Init a repo with a test identity so `git commit` works.
async fn init_repo(dir: &Path) -> GitRunner {
    let runner = GitRunner::new(Some(dir.to_path_buf()));
    git_in(&runner, &["init"]).await;
    git_in(&runner, &["config", "user.email", "test@example.com"]).await;
    git_in(&runner, &["config", "user.name", "Test"]).await;
    runner
}

/// Write `<dir>/<name>` (creating parent dirs) and stage it.
async fn stage_file(runner: &GitRunner, dir: &Path, name: &str, content: &[u8]) {
    let path = dir.join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, content).unwrap();
    git_in(runner, &["add", name]).await;
}

/// A fresh Unknown file for the snapshot; tests mutate fields as needed.
fn staged(path: &str, change_type: ChangeType) -> StagedFile {
    StagedFile {
        path: PathBuf::from(path),
        old_path: None,
        change_type,
        similarity: None,
        insertions: Some(1),
        deletions: Some(1),
        category: FileCategory::Unknown,
    }
}

fn snapshot_of(files: Vec<StagedFile>) -> StagedSnapshot {
    StagedSnapshot { files }
}

fn category_of(result: &ClassifiedSnapshot, path: &str) -> FileCategory {
    result
        .files
        .iter()
        .find(|file| file.path == PathBuf::from(path))
        .map(|file| file.category)
        .unwrap_or_else(|| panic!("{path} missing from result"))
}

//  Phase A: pure, no git involved

/// These tests run in a directory that is NOT a git repository on purpose:
/// Phase A must resolve without touching git, so if a change ever pushes
/// these files into Phase B, cat-file fails loudly with "not a repository".
#[tokio::test]
async fn phase_a_resolves_lock_and_generated_without_git() {
    let dir = TempDir::new("classifier_phase_a_no_git").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let result = classifier
        .classify(&snapshot_of(vec![
            staged("Cargo.lock", ChangeType::Modified),
            staged("src/gen/widget.rs", ChangeType::Added),
            staged("api/user.pb.go", ChangeType::Added),
        ]))
        .await
        .unwrap();

    assert_eq!(
        category_of(&result, "Cargo.lock"),
        FileCategory::DependencyLock
    );
    assert_eq!(
        category_of(&result, "src/gen/widget.rs"),
        FileCategory::Generated
    );
    assert_eq!(
        category_of(&result, "api/user.pb.go"),
        FileCategory::Generated
    );
}

#[tokio::test]
async fn phase_a_unknown_without_signal_still_needs_git() {
    // Sanity counterpart: an ordinary file in a non-repo dir MUST fail,
    // because it genuinely needs Phase B. Pins the "purity" boundary
    // from the other side.
    let dir = TempDir::new("classifier_phase_b_needs_git").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let err = classifier
        .classify(&snapshot_of(vec![staged("main.rs", ChangeType::Added)]))
        .await
        .unwrap_err();
    assert_eq!(
        err.code,
        crate::shared::exception::GitErrorCode::CommandFailed
    );
}

/// Stage 2 owns Submodule / Binary; 3.1 must never rewrite them, even
/// when the basename screams lock or codegen.
#[tokio::test]
async fn stage2_terminal_categories_are_never_rewritten() {
    let dir = TempDir::new("classifier_terminal_categories").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let mut binary = staged("Cargo.lock", ChangeType::Modified);
    binary.category = FileCategory::Binary;
    let mut submodule = staged("vendor/sub", ChangeType::Modified);
    submodule.category = FileCategory::Submodule;
    let mut binary_gen = staged("api/user.pb.go", ChangeType::Added);
    binary_gen.category = FileCategory::Binary;

    let result = classifier
        .classify(&snapshot_of(vec![binary, submodule, binary_gen]))
        .await
        .unwrap();

    assert_eq!(category_of(&result, "Cargo.lock"), FileCategory::Binary);
    assert_eq!(category_of(&result, "vendor/sub"), FileCategory::Submodule);
    assert_eq!(category_of(&result, "api/user.pb.go"), FileCategory::Binary);
}

/// Renames keep their generated/lock signal through the old path:
/// a codegen output renamed to a bland name must not become SemanticText.
#[tokio::test]
async fn rename_falls_back_to_old_path_signal() {
    let dir = TempDir::new("classifier_rename_old_path").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let mut gen_rename = staged("user.go", ChangeType::Renamed);
    gen_rename.old_path = Some(PathBuf::from("user.pb.go"));

    let mut lock_rename = staged("deps-resolved.txt", ChangeType::Renamed);
    lock_rename.old_path = Some(PathBuf::from("Cargo.lock"));

    // New path is checked first: a rename INTO a lockfile name is a lock.
    let mut into_lock = staged("Cargo.lock", ChangeType::Renamed);
    into_lock.old_path = Some(PathBuf::from("legacy.pb.go"));

    let result = classifier
        .classify(&snapshot_of(vec![gen_rename, lock_rename, into_lock]))
        .await
        .unwrap();

    assert_eq!(category_of(&result, "user.go"), FileCategory::Generated);
    assert_eq!(
        category_of(&result, "deps-resolved.txt"),
        FileCategory::DependencyLock
    );
    assert_eq!(
        category_of(&result, "Cargo.lock"),
        FileCategory::DependencyLock
    );
}

//  Phase B: blob header via real git

#[tokio::test]
async fn phase_b_generated_header_end_to_end() {
    let dir = TempDir::new("classifier_header_generated").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    stage_file(
        &runner,
        dir.path(),
        "user.pb.go",
        b"// Code generated by protoc-gen-go. DO NOT EDIT.\npackage api\n",
    )
    .await;

    let result = classifier
        .classify(&snapshot_of(vec![staged("user.pb.go", ChangeType::Added)]))
        .await
        .unwrap();

    assert_eq!(category_of(&result, "user.pb.go"), FileCategory::Generated);
}

/// Same probe path as above but the header only matches through the
/// comment-line fallback (`// Generated by sqlc`), not a bare marker.
#[tokio::test]
async fn phase_b_comment_fallback_end_to_end() {
    let dir = TempDir::new("classifier_comment_fallback").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    stage_file(
        &runner,
        dir.path(),
        "db.go",
        b"// Generated by sqlc 1.24.0\npackage db\n",
    )
    .await;

    let result = classifier
        .classify(&snapshot_of(vec![staged("db.go", ChangeType::Modified)]))
        .await
        .unwrap();

    assert_eq!(category_of(&result, "db.go"), FileCategory::Generated);
}

#[tokio::test]
async fn phase_b_plain_file_is_semantic_text() {
    let dir = TempDir::new("classifier_plain_semantic").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    stage_file(
        &runner,
        dir.path(),
        "main.rs",
        b"fn main() {\n    println!(\"hi\");\n}\n",
    )
    .await;

    let result = classifier
        .classify(&snapshot_of(vec![staged("main.rs", ChangeType::Added)]))
        .await
        .unwrap();

    assert_eq!(category_of(&result, "main.rs"), FileCategory::SemanticText);
}

/// Only the first 512 bytes are read: a marker past the head is
/// invisible and the file stays SemanticText.
#[tokio::test]
async fn phase_b_marker_beyond_head_is_not_seen() {
    let dir = TempDir::new("classifier_marker_beyond_head").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    let content = format!(
        "{}\n// Code generated by tool. DO NOT EDIT.\n",
        "x".repeat(600)
    );
    stage_file(&runner, dir.path(), "late.txt", content.as_bytes()).await;

    let result = classifier
        .classify(&snapshot_of(vec![staged("late.txt", ChangeType::Added)]))
        .await
        .unwrap();

    assert_eq!(category_of(&result, "late.txt"), FileCategory::SemanticText);
}

/// A staged empty blob resolves to Some(empty) → no marker → SemanticText.
#[tokio::test]
async fn phase_b_empty_blob_is_semantic_text() {
    let dir = TempDir::new("classifier_empty_blob").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    stage_file(&runner, dir.path(), "empty.txt", b"").await;

    let result = classifier
        .classify(&snapshot_of(vec![staged("empty.txt", ChangeType::Added)]))
        .await
        .unwrap();

    assert_eq!(
        category_of(&result, "empty.txt"),
        FileCategory::SemanticText
    );
}

/// A snapshot entry whose path is not in the index (cannot happen via
/// stage 2, but must not hang or panic): missing → SemanticText.
#[tokio::test]
async fn phase_b_missing_index_path_is_semantic_text() {
    let dir = TempDir::new("classifier_missing_path").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    let result = classifier
        .classify(&snapshot_of(vec![staged("ghost.txt", ChangeType::Added)]))
        .await
        .unwrap();

    assert_eq!(
        category_of(&result, "ghost.txt"),
        FileCategory::SemanticText
    );
}

//  Deleted branch

/// Deleted files classify from the HEAD blob: deleting a generated
/// artifact is still a Generated change.
#[tokio::test]
async fn deleted_file_classifies_from_head_blob() {
    let dir = TempDir::new("classifier_deleted_head").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    stage_file(
        &runner,
        dir.path(),
        "legacy_gen.go",
        b"// Code generated by protoc. DO NOT EDIT.\npackage legacy\n",
    )
    .await;
    git_in(&runner, &["commit", "-m", "add generated"]).await;
    git_in(&runner, &["rm", "legacy_gen.go"]).await;

    let result = classifier
        .classify(&snapshot_of(vec![staged(
            "legacy_gen.go",
            ChangeType::Deleted,
        )]))
        .await
        .unwrap();

    assert_eq!(
        category_of(&result, "legacy_gen.go"),
        FileCategory::Generated
    );
}

/// Unborn HEAD: `HEAD:path` resolves to nothing → SemanticText,
/// not an error. (Deletions can't occur before the first commit in
/// practice; this pins the defensive behavior.)
#[tokio::test]
async fn deleted_on_unborn_head_is_semantic_text() {
    let dir = TempDir::new("classifier_deleted_unborn").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    let result = classifier
        .classify(&snapshot_of(vec![staged(
            "anything.txt",
            ChangeType::Deleted,
        )]))
        .await
        .unwrap();

    assert_eq!(
        category_of(&result, "anything.txt"),
        FileCategory::SemanticText
    );
}

//  batch behavior & isolation

/// One --batch call serves a mixed probe list; results stay aligned.
#[tokio::test]
async fn mixed_batch_classifies_all_in_order() {
    let dir = TempDir::new("classifier_mixed_batch").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    stage_file(
        &runner,
        dir.path(),
        "gen_header.ts",
        b"/* @generated by tool */\nexport {};\n",
    )
    .await;
    stage_file(&runner, dir.path(), "plain.ts", b"export const x = 1;\n").await;

    let result = classifier
        .classify(&snapshot_of(vec![
            staged("gen_header.ts", ChangeType::Added),
            staged("plain.ts", ChangeType::Added),
            staged("ghost.ts", ChangeType::Added),
        ]))
        .await
        .unwrap();

    assert_eq!(
        category_of(&result, "gen_header.ts"),
        FileCategory::Generated
    );
    assert_eq!(category_of(&result, "plain.ts"), FileCategory::SemanticText);
    assert_eq!(category_of(&result, "ghost.ts"), FileCategory::SemanticText);
    // Order preserved — result[i] corresponds to input[i].
    assert_eq!(result.files[0].path, PathBuf::from("gen_header.ts"));
    assert_eq!(result.files[2].path, PathBuf::from("ghost.ts"));
}

/// A path with a newline cannot become a --batch line: that one file
/// degrades to SemanticText while the rest of the batch still classifies.
#[tokio::test]
async fn newline_path_is_isolated_as_semantic_text() {
    let dir = TempDir::new("classifier_newline_isolated").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    stage_file(
        &runner,
        dir.path(),
        "gen_header.py",
        b"# Code generated by tool. DO NOT EDIT.\n",
    )
    .await;

    let result = classifier
        .classify(&snapshot_of(vec![
            staged("weird\nname.txt", ChangeType::Added),
            staged("gen_header.py", ChangeType::Added),
        ]))
        .await
        .unwrap();

    assert_eq!(
        category_of(&result, "weird\nname.txt"),
        FileCategory::SemanticText
    );
    assert_eq!(
        category_of(&result, "gen_header.py"),
        FileCategory::Generated
    );
}

/// Classification touches only `category`: line counts, similarity and
/// rename metadata must survive into the ClassifiedSnapshot.
#[tokio::test]
async fn metadata_survives_classification() {
    let dir = TempDir::new("classifier_metadata_preserved").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    stage_file(&runner, dir.path(), "main.rs", b"fn main() {}\n").await;

    let mut rename = staged("user.go", ChangeType::Renamed);
    rename.old_path = Some(PathBuf::from("user.pb.go"));
    rename.similarity = Some(95);
    rename.insertions = Some(10);
    rename.deletions = Some(2);
    let mut text = staged("main.rs", ChangeType::Modified);
    text.insertions = Some(7);
    text.deletions = Some(3);

    let result = classifier
        .classify(&snapshot_of(vec![rename, text]))
        .await
        .unwrap();

    let rename = &result.files[0];
    assert_eq!(rename.category, FileCategory::Generated);
    assert_eq!(rename.old_path.as_deref(), Some(Path::new("user.pb.go")));
    assert_eq!(rename.similarity, Some(95));
    assert_eq!((rename.insertions, rename.deletions), (Some(10), Some(2)));

    let text = &result.files[1];
    assert_eq!(text.category, FileCategory::SemanticText);
    assert_eq!((text.insertions, text.deletions), (Some(7), Some(3)));
}

#[tokio::test]
async fn empty_snapshot_yields_empty_result() {
    let dir = TempDir::new("classifier_empty_snapshot").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let result = classifier.classify(&snapshot_of(vec![])).await.unwrap();
    assert!(result.files.is_empty());
}

//  Real-world layouts (Phase A, no git)

/// One classify() over a polyglot monorepo snapshot. Every entry here is
/// resolved by path / basename alone — if any of them ever falls through
/// to Phase B, cat-file fails because this directory is not a repo.
#[tokio::test]
async fn phase_a_polyglot_layouts() {
    let dir = TempDir::new("classifier_polyglot_phase_a").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let cases: &[(&str, FileCategory)] = &[
        // C / C++
        ("proto/user.pb.cc", FileCategory::Generated),
        ("proto/user.pb.h", FileCategory::Generated),
        (
            "cmake-build-debug/generated/config.h",
            FileCategory::Generated,
        ),
        ("third_party/vcpkg-lock.json", FileCategory::DependencyLock),
        ("conan.lock", FileCategory::DependencyLock),
        // Java / JVM
        (
            "target/generated-sources/protobuf/java/User.java",
            FileCategory::Generated,
        ),
        (
            "build/generated/source/apt/main/Mapper.java",
            FileCategory::Generated,
        ),
        ("gradle.lockfile", FileCategory::DependencyLock),
        ("MODULE.bazel.lock", FileCategory::DependencyLock),
        // C#
        ("MyApp/Form1.Designer.cs", FileCategory::Generated),
        (
            "obj/Debug/net8.0/MyApp.GlobalUsings.g.cs",
            FileCategory::Generated,
        ),
        ("src/packages.lock.json", FileCategory::DependencyLock),
        // Python
        ("pkg/user_pb2.py", FileCategory::Generated),
        ("pkg/user_pb2_grpc.py", FileCategory::Generated),
        ("poetry.lock", FileCategory::DependencyLock),
        ("uv.lock", FileCategory::DependencyLock),
        // JS / TS
        ("web/src/__generated__/graphql.ts", FileCategory::Generated),
        ("web/bundle.min.js", FileCategory::Generated),
        ("web/pnpm-lock.yaml", FileCategory::DependencyLock),
        // Go / Rust / Swift / Dart / PHP / Ruby
        ("api/user.pb.go", FileCategory::Generated),
        ("Cargo.lock", FileCategory::DependencyLock),
        ("Sources/API/user.pb.swift", FileCategory::Generated),
        ("lib/user.g.dart", FileCategory::Generated),
        ("composer.lock", FileCategory::DependencyLock),
        ("Gemfile.lock", FileCategory::DependencyLock),
        // Infra
        (".terraform.lock.hcl", FileCategory::DependencyLock),
        ("charts/app/Chart.lock", FileCategory::DependencyLock),
        // Whitespace in a VS / Xcode-style path
        ("My App/Form1.Designer.cs", FileCategory::Generated),
        ("App.xcodeproj/project.pbxproj", FileCategory::Generated),
    ];

    let files = cases
        .iter()
        .map(|(path, _)| staged(path, ChangeType::Added))
        .collect();
    let result = classifier.classify(&snapshot_of(files)).await.unwrap();

    for (path, expected) in cases {
        assert_eq!(category_of(&result, path), *expected, "{path}");
    }
}

/// Copied / TypeChanged go through the same Phase A path as Added.
#[tokio::test]
async fn copy_and_typechange_use_phase_a() {
    let dir = TempDir::new("classifier_copy_typechange").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let mut copied = staged("vendor/user.go", ChangeType::Copied);
    copied.old_path = Some(PathBuf::from("api/user.pb.go"));
    let typechange = staged("proto/user.pb.cc", ChangeType::TypeChanged);

    let result = classifier
        .classify(&snapshot_of(vec![copied, typechange]))
        .await
        .unwrap();

    assert_eq!(
        category_of(&result, "vendor/user.go"),
        FileCategory::Generated
    );
    assert_eq!(
        category_of(&result, "proto/user.pb.cc"),
        FileCategory::Generated
    );
}

//  Real-world tool banners (Phase B — bland names, so header is the only signal)

/// Typical codegen banners from C++ / Java / C# / Python / PHP / Swift,
/// none of which have a registered suffix or marker directory.
#[tokio::test]
async fn phase_b_polyglot_tool_banners() {
    let dir = TempDir::new("classifier_polyglot_banners").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    let cases: &[(&str, &[u8], FileCategory)] = &[
        (
            "User.pb.cc",
            // actually .pb.cc is Phase A — use a bland C++ name
            b"",
            FileCategory::SemanticText,
        ),
        (
            "config.h",
            b"/* Auto-generated by CMake. DO NOT EDIT. */\n#pragma once\n",
            FileCategory::Generated,
        ),
        (
            "UserOuterClass.java",
            b"// Generated by the protocol buffer compiler.  DO NOT EDIT!\npackage com.example;\n",
            FileCategory::Generated,
        ),
        (
            "PetApi.java",
            b"// AUTO-GENERATED FILE, DO NOT MODIFY.\npackage org.openapitools.client.api;\n",
            FileCategory::Generated,
        ),
        (
            "Models.cs",
            b"// <auto-generated>\n//     This code was generated by a tool.\n// </auto-generated>\n",
            FileCategory::Generated,
        ),
        (
            "schema.py",
            b"# Generated by the protocol buffer compiler.  DO NOT EDIT!\n# source: schema.proto\n",
            FileCategory::Generated,
        ),
        (
            "Types.kt",
            b"// Generated by the protocol buffer compiler. DO NOT EDIT!\npackage com.example\n",
            FileCategory::Generated,
        ),
        (
            "API.swift",
            b"// Code generated by Wire. DO NOT EDIT.\nimport Foundation\n",
            FileCategory::Generated,
        ),
        (
            "autoload.php",
            b"<?php\n/** This file is generated. Do not edit. */\n",
            FileCategory::Generated,
        ),
        (
            "schema.rb",
            b"# This file was automatically generated by graphql-client\n",
            FileCategory::Generated,
        ),
    ];

    // Drop the dummy C++ entry — keep the list honest.
    let cases: &[(&str, &[u8], FileCategory)] = &cases[1..];

    let mut files = Vec::new();
    for (path, content, _) in cases {
        stage_file(&runner, dir.path(), path, content).await;
        files.push(staged(path, ChangeType::Added));
    }

    let result = classifier.classify(&snapshot_of(files)).await.unwrap();
    for (path, _, expected) in cases {
        assert_eq!(category_of(&result, path), *expected, "{path}");
    }
}

/// Hand-written sources across languages must stay SemanticText even
/// when they live next to generated siblings.
#[tokio::test]
async fn hand_written_sources_are_semantic_text() {
    let dir = TempDir::new("classifier_handwritten").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    let sources: &[(&str, &[u8])] = &[
        (
            "src/main.c",
            b"#include \"app.h\"\nint main(void) { return 0; }\n",
        ),
        (
            "src/app.cpp",
            b"#include <vector>\nint run() { return 0; }\n",
        ),
        (
            "src/App.java",
            b"package com.example;\npublic class App {}\n",
        ),
        ("src/Program.cs", b"namespace MyApp { class Program {} }\n"),
        ("src/main.py", b"def main():\n    print(\"hi\")\n"),
        ("src/main.go", b"package main\nfunc main() {}\n"),
        ("src/lib.rs", b"pub fn answer() -> u8 { 42 }\n"),
        ("src/View.swift", b"import SwiftUI\nstruct View {}\n"),
        ("lib/widget.dart", b"class Widget {}\n"),
        ("src/index.ts", b"export const x = 1;\n"),
    ];

    let mut files = Vec::new();
    for (path, content) in sources {
        stage_file(&runner, dir.path(), path, content).await;
        files.push(staged(path, ChangeType::Modified));
    }

    let result = classifier.classify(&snapshot_of(files)).await.unwrap();
    for (path, _) in sources {
        assert_eq!(
            category_of(&result, path),
            FileCategory::SemanticText,
            "{path}"
        );
    }
}

/// A single commit mixing lockfiles, codegen, and hand-written sources
/// — the actual shape auto-commit sees in a monorepo.
#[tokio::test]
async fn mixed_monorepo_commit() {
    let dir = TempDir::new("classifier_monorepo").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    stage_file(
        &runner,
        dir.path(),
        "services/api/src/main.rs",
        b"fn main() {}\n",
    )
    .await;
    stage_file(
        &runner,
        dir.path(),
        "services/api/src/db.rs",
        b"// Code generated by sqlc. DO NOT EDIT.\n",
    )
    .await;

    let mut lock = staged("services/api/Cargo.lock", ChangeType::Modified);
    let proto = staged("proto/user.pb.cc", ChangeType::Added);
    let designer = staged("desktop/Form1.Designer.cs", ChangeType::Modified);
    let java = staged(
        "backend/target/generated-sources/User.java",
        ChangeType::Added,
    );
    let mut rust = staged("services/api/src/main.rs", ChangeType::Modified);
    rust.insertions = Some(12);
    let mut sqlc = staged("services/api/src/db.rs", ChangeType::Modified);
    let cpp = staged("native/src/app.cpp", ChangeType::Modified);
    // Hand-written C++ has to exist in the index so Phase B can read it.
    stage_file(
        &runner,
        dir.path(),
        "native/src/app.cpp",
        b"#include <iostream>\nint main() { return 0; }\n",
    )
    .await;

    let result = classifier
        .classify(&snapshot_of(vec![
            lock, proto, designer, java, rust, sqlc, cpp,
        ]))
        .await
        .unwrap();

    assert_eq!(
        category_of(&result, "services/api/Cargo.lock"),
        FileCategory::DependencyLock
    );
    assert_eq!(
        category_of(&result, "proto/user.pb.cc"),
        FileCategory::Generated
    );
    assert_eq!(
        category_of(&result, "desktop/Form1.Designer.cs"),
        FileCategory::Generated
    );
    assert_eq!(
        category_of(&result, "backend/target/generated-sources/User.java"),
        FileCategory::Generated
    );
    assert_eq!(
        category_of(&result, "services/api/src/main.rs"),
        FileCategory::SemanticText
    );
    assert_eq!(
        category_of(&result, "services/api/src/db.rs"),
        FileCategory::Generated
    );
    assert_eq!(
        category_of(&result, "native/src/app.cpp"),
        FileCategory::SemanticText
    );
}
