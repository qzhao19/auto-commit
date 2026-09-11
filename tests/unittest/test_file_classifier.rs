use std::path::{Path, PathBuf};

use crate::core::git::diff::FileClassifier;
use crate::core::git::types::{
    ChangeType, ClassifiedSnapshot, FileCategory, StagedFile, StagedSnapshot,
};
use crate::infra::git::GitRunner;
use crate::shared::exception::GitErrorCode;

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

//  Phase A: pure, no git involved (C-01, C-02)

/// C-01: runs in a directory that is NOT a git repository on purpose —
/// Phase A must resolve without touching git, so if a change ever
/// pushes these files into Phase B, cat-file fails loudly with
/// "not a repository".
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

/// C-02: sanity counterpart — an ordinary file in a non-repo dir MUST
/// fail, because it genuinely needs Phase B. Pins the "purity"
/// boundary from the other side.
#[tokio::test]
async fn phase_a_unknown_without_signal_still_needs_git() {
    let dir = TempDir::new("classifier_phase_b_needs_git").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let err = classifier
        .classify(&snapshot_of(vec![staged("main.rs", ChangeType::Added)]))
        .await
        .unwrap_err();
    assert_eq!(err.code, GitErrorCode::CommandFailed);
}

/// C-03: Stage 2 owns Submodule / Binary — classify() must never
/// rewrite ANY preset category, even when the basename screams lock
/// or codegen. Phase A's `continue` guard covers every non-Unknown.
#[tokio::test]
async fn stage2_terminal_categories_are_never_rewritten() {
    let dir = TempDir::new("classifier_terminal_categories").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let mut binary_lock = staged("Cargo.lock", ChangeType::Modified);
    binary_lock.category = FileCategory::Binary;
    let mut submodule = staged("vendor/sub", ChangeType::Modified);
    submodule.category = FileCategory::Submodule;
    let mut binary_gen = staged("api/user.pb.go", ChangeType::Added);
    binary_gen.category = FileCategory::Binary;
    // A Generated preset survives even on a lockfile basename.
    let mut generated_lock = staged("yarn.lock", ChangeType::Modified);
    generated_lock.category = FileCategory::Generated;

    let result = classifier
        .classify(&snapshot_of(vec![
            binary_lock,
            submodule,
            binary_gen,
            generated_lock,
        ]))
        .await
        .unwrap();

    assert_eq!(category_of(&result, "Cargo.lock"), FileCategory::Binary);
    assert_eq!(category_of(&result, "vendor/sub"), FileCategory::Submodule);
    assert_eq!(category_of(&result, "api/user.pb.go"), FileCategory::Binary);
    assert_eq!(category_of(&result, "yarn.lock"), FileCategory::Generated);
}

/// C-04: renames keep their generated/lock signal through the old
/// path; the new path is checked first.
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

//  Phase B: blob header via real git (C-05 .. C-12)

/// C-05: staged blob whose header hits a bare marker.
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

/// C-06: the header only matches through the comment-line fallback
/// (`// Generated by sqlc`), not a bare marker.
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

/// C-07: no signal at all — plain staged source.
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

/// C-08: only the first 512 bytes are read — a marker past the head
/// is invisible and the file stays SemanticText.
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

/// C-09: a staged empty blob resolves to Some(empty) → no marker →
/// SemanticText.
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

/// C-10: a snapshot entry whose path is not in the index (cannot
/// happen via stage 2, but must not hang or panic): missing →
/// SemanticText.
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

/// C-11: deleted files classify from the HEAD blob — deleting a
/// generated artifact is still a Generated change.
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

/// C-12: unborn HEAD — `HEAD:path` resolves to nothing →
/// SemanticText, not an error.
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

//  batch behavior & isolation (C-13 .. C-16)

/// C-13: one --batch call serves a mixed probe list; results stay
/// aligned with the input order.
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

/// C-14: a path with a newline cannot become a --batch line — that one
/// file degrades to SemanticText while the rest still classifies.
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

/// C-15: classification touches only `category` — line counts,
/// similarity and rename metadata survive into the snapshot.
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
    assert_eq!(rename.change_type, ChangeType::Renamed);
    assert_eq!(rename.old_path.as_deref(), Some(Path::new("user.pb.go")));
    assert_eq!(rename.similarity, Some(95));
    assert_eq!((rename.insertions, rename.deletions), (Some(10), Some(2)));

    let text = &result.files[1];
    assert_eq!(text.category, FileCategory::SemanticText);
    assert_eq!((text.insertions, text.deletions), (Some(7), Some(3)));
}

/// C-16: empty snapshot is a no-op.
#[tokio::test]
async fn empty_snapshot_yields_empty_result() {
    let dir = TempDir::new("classifier_empty_snapshot").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let result = classifier.classify(&snapshot_of(vec![])).await.unwrap();
    assert!(result.files.is_empty());
}

//  Real-world layouts (C-17 .. C-21)

/// C-17: one classify() over a polyglot monorepo snapshot. Every entry
/// resolves by path / basename alone — if any ever falls through to
/// Phase B, cat-file fails because this directory is not a repo.
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

/// C-18: Copied / TypeChanged go through the same Phase A path as
/// Added.
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

/// C-19: typical codegen banners from C++ / Java / C# / Python / PHP /
/// Swift — none of which have a registered suffix or marker directory.
#[tokio::test]
async fn phase_b_polyglot_tool_banners() {
    let dir = TempDir::new("classifier_polyglot_banners").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    let cases: &[(&str, &[u8])] = &[
        (
            "config.h",
            b"/* Auto-generated by CMake. DO NOT EDIT. */\n#pragma once\n",
        ),
        (
            "UserOuterClass.java",
            b"// Generated by the protocol buffer compiler.  DO NOT EDIT!\npackage com.example;\n",
        ),
        (
            "PetApi.java",
            b"// AUTO-GENERATED FILE, DO NOT MODIFY.\npackage org.openapitools.client.api;\n",
        ),
        (
            "Models.cs",
            b"// <auto-generated>\n//     This code was generated by a tool.\n// </auto-generated>\n",
        ),
        (
            "schema.py",
            b"# Generated by the protocol buffer compiler.  DO NOT EDIT!\n# source: schema.proto\n",
        ),
        (
            "Types.kt",
            b"// Generated by the protocol buffer compiler. DO NOT EDIT!\npackage com.example\n",
        ),
        (
            "API.swift",
            b"// Code generated by Wire. DO NOT EDIT.\nimport Foundation\n",
        ),
        (
            "autoload.php",
            b"<?php\n/** This file is generated. Do not edit. */\n",
        ),
        (
            "schema.rb",
            b"# This file was automatically generated by graphql-client\n",
        ),
    ];

    let mut files = Vec::new();
    for (path, content) in cases {
        stage_file(&runner, dir.path(), path, content).await;
        files.push(staged(path, ChangeType::Added));
    }

    let result = classifier.classify(&snapshot_of(files)).await.unwrap();
    for (path, _) in cases {
        assert_eq!(
            category_of(&result, path),
            FileCategory::Generated,
            "{path}"
        );
    }
}

/// C-20: hand-written sources across languages must stay SemanticText
/// even when they live next to generated siblings.
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

/// C-21: a single commit mixing lockfiles, codegen, and hand-written
/// sources — the actual shape auto-commit sees in a monorepo.
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
    // Hand-written C++ has to exist in the index so Phase B can read it.
    stage_file(
        &runner,
        dir.path(),
        "native/src/app.cpp",
        b"#include <iostream>\nint main() { return 0; }\n",
    )
    .await;

    let lock = staged("services/api/Cargo.lock", ChangeType::Modified);
    let proto = staged("proto/user.pb.cc", ChangeType::Added);
    let designer = staged("desktop/Form1.Designer.cs", ChangeType::Modified);
    let java = staged(
        "backend/target/generated-sources/User.java",
        ChangeType::Added,
    );
    let mut rust = staged("services/api/src/main.rs", ChangeType::Modified);
    rust.insertions = Some(12);
    let sqlc = staged("services/api/src/db.rs", ChangeType::Modified);
    let cpp = staged("native/src/app.cpp", ChangeType::Modified);

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

//  Priority ladder & realistic Rust portrait (C-22 .. C-24)

/// C-22: the full priority ladder in ONE classify() call —
/// Submodule → Binary → DependencyLock → Generated → SemanticText,
/// including both competition pairs (Binary beats a lock basename,
/// lock beats a generated-directory signal).
#[tokio::test]
async fn priority_ladder_one_classify_call_ranks_all_layers() {
    let dir = TempDir::new("classifier_priority_ladder").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    // Only the Phase B entries need real blobs in the index.
    stage_file(
        &runner,
        dir.path(),
        "src/db.rs",
        b"// Code generated by sqlc. DO NOT EDIT.\n",
    )
    .await;
    stage_file(&runner, dir.path(), "src/main.rs", b"fn main() {}\n").await;

    let mut submodule = staged("vendor/dep", ChangeType::Modified);
    submodule.category = FileCategory::Submodule;
    let mut binary = staged("Cargo.lock", ChangeType::Modified);
    binary.category = FileCategory::Binary;
    let lock_in_generated_dir = staged("generated/package-lock.json", ChangeType::Added);
    let generated_suffix = staged("api/user.pb.go", ChangeType::Added);
    let sqlc = staged("src/db.rs", ChangeType::Modified);
    let plain = staged("src/main.rs", ChangeType::Modified);

    let result = classifier
        .classify(&snapshot_of(vec![
            submodule,
            binary,
            lock_in_generated_dir,
            generated_suffix,
            sqlc,
            plain,
        ]))
        .await
        .unwrap();

    let ladder: Vec<FileCategory> = result.files.iter().map(|f| f.category).collect();
    assert_eq!(
        ladder,
        vec![
            FileCategory::Submodule,
            FileCategory::Binary,
            FileCategory::DependencyLock,
            FileCategory::Generated,
            FileCategory::Generated,
            FileCategory::SemanticText,
        ]
    );
}

/// C-23: a realistic Rust workspace commit — cargo + tonic/prost +
/// insta. Phase A (basename / dir / suffix) and Phase B (blob
/// banner) both occur in one snapshot.
#[tokio::test]
async fn rust_workspace_portrait_classifies_realistically() {
    let dir = TempDir::new("classifier_rust_portrait").unwrap();
    let runner = init_repo(dir.path()).await;
    let classifier = FileClassifier::new(&runner);

    stage_file(
        &runner,
        dir.path(),
        "Cargo.lock",
        b"# This file is automatically @generated by Cargo.\n# It is not intended for manual editing.\nversion = 3\n",
    )
    .await;
    stage_file(
        &runner,
        dir.path(),
        "Cargo.toml",
        b"[package]\nname = \"acme\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .await;
    // tonic-build output committed into a `gen/` directory (Phase A: dir).
    stage_file(
        &runner,
        dir.path(),
        "crates/rpc/src/gen/acme.v1.rs",
        b"// @generated by tonic-build\n#[allow(clippy::derive_partial_eq_without_eq)]\npub struct PingRequest {}\n",
    )
    .await;
    // protoc-gen-prost style committed artifact (Phase A: suffix).
    stage_file(
        &runner,
        dir.path(),
        "crates/rpc/src/acme.v1.pb.rs",
        b"// @generated by prost-build\npub struct PingReply {}\n",
    )
    .await;
    // Bland name — only the blob banner says generated (Phase B).
    stage_file(
        &runner,
        dir.path(),
        "crates/rpc/src/lib.rs",
        b"// @generated by prost-build. DO NOT EDIT.\npub mod gen;\n",
    )
    .await;
    // insta snapshot committed by a test run (Phase A: `.snap` suffix).
    stage_file(
        &runner,
        dir.path(),
        "crates/rpc/src/snapshots/rpc__ping.snap",
        b"---\nsource: crates/rpc/src/lib.rs\nexpression: ping()\n---\n\"pong\"\n",
    )
    .await;
    stage_file(
        &runner,
        dir.path(),
        "src/main.rs",
        b"fn main() {\n    println!(\"hello\");\n}\n",
    )
    .await;

    let cases: &[(&str, FileCategory)] = &[
        ("Cargo.lock", FileCategory::DependencyLock),
        ("Cargo.toml", FileCategory::SemanticText),
        ("crates/rpc/src/gen/acme.v1.rs", FileCategory::Generated),
        ("crates/rpc/src/acme.v1.pb.rs", FileCategory::Generated),
        ("crates/rpc/src/lib.rs", FileCategory::Generated),
        (
            "crates/rpc/src/snapshots/rpc__ping.snap",
            FileCategory::Generated,
        ),
        ("src/main.rs", FileCategory::SemanticText),
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

/// C-24: Deleted + Phase A name signal — the lock basename resolves
/// WITHOUT reading any HEAD blob, provable in a directory that is
/// not a repo: if the Deleted branch ever probed `HEAD:` here,
/// cat-file would fail and this test would fail with it.
#[tokio::test]
async fn deleted_lock_resolves_in_phase_a_without_head_blob() {
    let dir = TempDir::new("classifier_deleted_phase_a").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let classifier = FileClassifier::new(&runner);

    let result = classifier
        .classify(&snapshot_of(vec![
            staged("Cargo.lock", ChangeType::Deleted),
            staged("uv.lock", ChangeType::Deleted),
        ]))
        .await
        .unwrap();

    assert_eq!(
        category_of(&result, "Cargo.lock"),
        FileCategory::DependencyLock
    );
    assert_eq!(
        category_of(&result, "uv.lock"),
        FileCategory::DependencyLock
    );
}
