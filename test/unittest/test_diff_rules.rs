use std::path::{Path, PathBuf};

use crate::core::git::diff::{
    classify_by_header, classify_by_name, match_generated_header, match_generated_name,
    match_generated_path, match_lock_file,
};
use crate::core::git::types::FileCategory;

//  match_lock_file: registry

#[test]
fn match_lock_file_matches_registry_sample() {
    let samples = [
        // JS / TS
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lock",
        "deno.lock",
        // Rust
        "Cargo.lock",
        // Go — no `lock` in the name at all
        "go.sum",
        "go.work.sum",
        // Python
        "poetry.lock",
        "uv.lock",
        "conda-lock.yml",
        // Ruby
        "Gemfile.lock",
        "gems.locked",
        // PHP
        "composer.lock",
        // Dart
        "pubspec.lock",
        // .NET
        "packages.lock.json",
        // Apple — "resolved", not "lock"
        "Package.resolved",
        "Cartfile.resolved",
        // C / C++
        "vcpkg-lock.json",
        // JVM
        "gradle.lockfile",
        "MODULE.bazel.lock",
        // Infra — leading dot
        ".terraform.lock.hcl",
        "flake.lock",
        "Chart.lock",
        // Misc — no `lock` substring
        "cabal.project.freeze",
        "cpanfile.snapshot",
        "dub.selections.json",
        "renv.lock",
        "spack.lock",
    ];

    for name in samples {
        assert!(
            match_lock_file(Path::new(name)),
            "root-level {name} must match"
        );
        let nested = format!("packages/subdir/{name}");
        assert!(
            match_lock_file(Path::new(&nested)),
            "nested {nested} must match — only the basename is compared"
        );
    }
}

#[test]
fn match_lock_file_is_case_insensitive() {
    for name in [
        "CARGO.LOCK",
        "cargo.lock",
        "Yarn.LOCK",
        "GO.SUM",
        "Package.Resolved",
    ] {
        assert!(match_lock_file(Path::new(name)), "{name} must match");
    }
}

#[test]
fn match_lock_file_rejects_lookalikes() {
    let lookalikes = [
        "Cargo.toml",    // manifest, not the lock
        "go.mod",        // go.sum is the lock, go.mod the manifest
        "package.json",  // manifest, not the lock
        "my-cargo.lock", // prefix additions must not match
        "Cargo.lock.bak",
        "packages.lock.json.bak",
        "yarn.lock.txt",
        "lock", // bare word is not a known lockfile
    ];
    for name in lookalikes {
        assert!(!match_lock_file(Path::new(name)), "{name} must not match");
    }
}

#[test]
fn match_lock_file_rejects_paths_without_file_name() {
    // No basename to compare against the registry.
    assert!(!match_lock_file(Path::new("/")));
    assert!(!match_lock_file(Path::new("..")));
    assert!(!match_lock_file(Path::new(".")));
}

/// Non-UTF-8 basenames cannot be compared against the registry.
#[cfg(unix)]
#[test]
fn match_lock_file_non_utf8_name_is_false() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let path = PathBuf::from(OsString::from_vec(b"Cargo\xff.lock".to_vec()));

    assert!(!match_lock_file(&path));
    assert!(!match_generated_name(&path));
}

//  match_generated_path: marker directories

#[test]
fn match_generated_path_matches_marker_directories() {
    let paths = [
        "generated/foo.rs",
        "src/generated/foo.rs",
        "src/__generated__/a.ts",
        "src/_generated/a.py",
        "src/Gen/b.rs",
        "src/GENERATED/c.java",
        // Marker anywhere in the ancestor chain, not only the direct parent
        "a/b/gen/c.rs",
        "src/gensrc/a.go",
        // Multi-word marker components — root layouts, no leading slash needed
        "target/generated-sources/Foo.java",
        "build/generated-sources/Bar.java",
        "target/generated_sources/Baz.c",
        "target/generated-test-sources/T.java",
    ];
    for path in paths {
        assert!(match_generated_path(Path::new(path)), "{path} must match");
    }
}

#[test]
fn match_generated_path_rejects_near_misses() {
    let paths = [
        // No marker directory at all
        "src/foo.rs",
        // `generated` as a basename, not a directory component
        "generated.rs",
        "src/generated.txt",
        // A file *named* gen — the parent is `src`
        "src/gen",
        // Superset / decorated directory names must not match
        "src/generator/foo.rs",
        "src/gen2/foo.rs",
        "src/my-gen/foo.rs",
        "src/regenerate/foo.rs",
    ];
    for path in paths {
        assert!(
            !match_generated_path(Path::new(path)),
            "{path} must not match"
        );
    }
}

//  match_generated_name: suffix registry

#[test]
fn match_generated_name_matches_protobuf_family() {
    for name in [
        "user.pb.cc",
        "user.pb.h",
        "user.pb.go",
        "user.pb.rs",
        "user.pb.swift",
        "user.pb.ts",
        "user.pb.js",
        "user.pb.dart",
        "service.pbgrpc.dart",
        "user_pb.py",
        "user_pb.js",
        "user_pb2.py",
        "user_pb_grpc.py",
        "user_pb2_grpc.py",
    ] {
        assert!(match_generated_name(Path::new(name)), "{name} must match");
    }
}

#[test]
fn match_generated_name_matches_platform_conventions() {
    for name in [
        // .NET source generators & VS designers
        "AssemblyInfo.g.cs",
        "Form1.designer.cs",
        // Dart build_runner
        "user.g.dart",
        "user.freezed.dart",
        "user.mocks.dart",
        // buf / oapi-codegen style infixes
        "api.gen.go",
        "api.gen.ts",
        "api.gen.rs",
        "api.codegen.js",
        "api.codegen.ts",
        // explicit `generated` infixes
        "types.generated.go",
        "types.generated.jsx",
        "types.generated.tsx",
        "types_generated.go",
        "types_generated.java",
    ] {
        assert!(match_generated_name(Path::new(name)), "{name} must match");
    }
}

#[test]
fn match_generated_name_matches_bundler_output() {
    for name in [
        "bundle.min.js",
        "style.min.css",
        "app.min.mjs",
        "bundle.js.map",
        "app.mjs.map",
        "style.css.map",
        // Jest snapshot convention: `foo.test.tsx.snap`
        "button.test.tsx.snap",
        "project.pbxproj",
    ] {
        assert!(match_generated_name(Path::new(name)), "{name} must match");
    }
}

#[test]
fn match_generated_name_is_case_insensitive() {
    for name in [
        "USER.PB.GO",
        "Bundle.MIN.js",
        "TYPES_GENERATED.GO",
        "API.Codegen.TS",
    ] {
        assert!(match_generated_name(Path::new(name)), "{name} must match");
    }
}

/// Suffix match must be `ends_with`, not `contains` — the historical bug
/// classified `foo.snap.txt` and `grab_pb.py.bak` as generated.
#[test]
fn match_generated_name_requires_true_suffix() {
    for name in [
        "foo.snap.txt",
        "grab_pb.py.bak",
        "foo.pbxproj.orig",
        "foo.min.js.bak",
        "types.generated.go.md",
        // `.map` alone is not a marker — only `*.js.map` etc. are
        "styles.map",
        // `foo.snapshot` does not end with `.snap`
        "foo.snapshot",
        // Needs the dot: notpb.go is a hand-written file
        "notpb.go",
        // Ordinary sources
        "main.rs",
        "index.ts",
        "foo.go",
    ] {
        assert!(
            !match_generated_name(Path::new(name)),
            "{name} must not match"
        );
    }
}

//  match_generated_header: blob head markers

#[test]
fn match_generated_header_matches_markers() {
    let headers: &[&[u8]] = &[
        b"// Code generated by protoc-gen-go. DO NOT EDIT.\n",
        b"@generated by my-codegen\n",
        b"/* Auto-generated file. */",
        b"# This file was automatically generated by CMake.",
        b"// AUTO-GENERATED, do not commit by hand",
        b"/* autogenerated by gcc */",
        b"// Do Not Edit",
        b"// DO NOT MODIFY this section",
        b"// Generated by protoc v3.21",
        b"// Generated by the protocol buffer compiler.  DO NOT EDIT!",
        b"// this file is generated by `cargo build`",
        b"// this file was generated by the build system",
        // Marker anywhere in the head, not only on the first line
        b"#!/usr/bin/env python3\n# Code generated by flatc.\n",
    ];
    for header in headers {
        assert!(
            match_generated_header(header),
            "{:?} must match",
            String::from_utf8_lossy(header)
        );
    }
}

/// None of these contain a bare marker substring — the match must come
/// from a comment line that starts with "generated by" after the prefix.
#[test]
fn match_generated_header_matches_generated_by_comment_lines() {
    let headers: &[&[u8]] = &[
        b"// Generated by sqlc 1.24.0\n",
        b"# Generated by setuptools 69.0\n",
        b"-- Generated by sqlc 1.24.0\n",
        b"; generated by some-asm-tool\n",
        b"/* Generated by javadoc 21 */\n",
        b" * Generated by mockgen v1\n",
        b"// GENERATED BY AN UPPERCASE TOOL\n",
    ];
    for header in headers {
        assert!(
            match_generated_header(header),
            "{:?} must match via the comment fallback",
            String::from_utf8_lossy(header)
        );
    }
}

#[test]
fn match_generated_header_rejects_plain_content() {
    let headers: &[&[u8]] = &[
        // `generated` in code, not in a marker or comment line
        b"fn main() { println!(\"generated\"); }\n",
        // Prose about generation — no comment line starts with "generated by"
        b"// A plain comment about how files are generated by tools.\n",
        b"// not generated by hand\n",
        b"// generated\n",
        b"const x = \"generated by test\";\n",
        b"This document explains what gets generated by the pipeline.\n",
        // Empty / whitespace-only heads
        b"",
        b"   \n\t\n",
    ];
    for header in headers {
        assert!(
            !match_generated_header(header),
            "{:?} must not match",
            String::from_utf8_lossy(header)
        );
    }
}

/// Blob heads may be invalid UTF-8: lossy conversion must not panic,
/// and markers in the valid part must still be found.
#[test]
fn match_generated_header_tolerates_invalid_utf8() {
    assert!(match_generated_header(b"\xff\xfe// Code generated by x\n"));
    assert!(!match_generated_header(b"\xff\xfe\x00\x01 plain bytes"));
}

//  classify_by_name: Phase A composition

#[test]
fn classify_by_name_lock_beats_generated() {
    // Path says Generated (marker dir), basename says lock — lock wins:
    // DependencyLock is the higher-priority category.
    assert_eq!(
        classify_by_name(Path::new("generated/package-lock.json")),
        Some(FileCategory::DependencyLock)
    );
}

#[test]
fn classify_by_name_generated_via_path_or_suffix() {
    assert_eq!(
        classify_by_name(Path::new("gen/foo.pb.go")),
        Some(FileCategory::Generated)
    );
    assert_eq!(
        classify_by_name(Path::new("src/generated/foo.rs")),
        Some(FileCategory::Generated)
    );
    assert_eq!(
        classify_by_name(Path::new("src/types.generated.ts")),
        Some(FileCategory::Generated)
    );
}

#[test]
fn classify_by_name_returns_none_for_ordinary_files() {
    assert_eq!(classify_by_name(Path::new("src/main.rs")), None);
    assert_eq!(classify_by_name(Path::new("docs/readme.md")), None);
    // Precision regression: `.snap` must be a true suffix.
    assert_eq!(classify_by_name(Path::new("foo.snap.txt")), None);
}

//  classify_by_header: Phase B residue

#[test]
fn classify_by_header_maps_to_generated_or_none() {
    assert_eq!(
        classify_by_header(b"// Code generated by x. DO NOT EDIT.\n"),
        Some(FileCategory::Generated)
    );
    assert_eq!(classify_by_header(b"fn main() {}\n"), None);
    assert_eq!(classify_by_header(b""), None);
}

//  match_lock_file: remaining registry entries

/// Second half of the registry not covered by the first sample.
/// Together the two sample tests assert every entry in LOCK_FILES.
#[test]
fn match_lock_file_matches_remaining_registry() {
    let samples = [
        // Go legacy
        "Gopkg.lock",
        "glide.lock",
        // Python
        "Pipfile.lock",
        "pdm.lock",
        "pixi.lock",
        "conda-lock.yaml",
        // Elixir / Erlang
        "mix.lock",
        "rebar.lock",
        // .NET
        "paket.lock",
        // Apple
        "Podfile.lock",
        // C / C++
        "conan.lock",
        // Infra
        "helmfile.lock",
        // Misc
        "shard.lock",
        "stack.yaml.lock",
        "jsonnetfile.lock.json",
        "Brewfile.lock.json",
        "Berksfile.lock",
        "Policyfile.lock.json",
        "Puppetfile.lock",
    ];
    for name in samples {
        assert!(match_lock_file(Path::new(name)), "{name} must match");
    }
}

/// Precision > recall: a generic `*.lock` basename is NOT a dependency
/// lock of a known ecosystem and must stay SemanticText.
#[test]
fn match_lock_file_generic_lock_suffix_is_not_enough() {
    for name in [
        "requirements.lock",
        "thirdparty.lock",
        "vendor.lock",
        "deps.lock",
    ] {
        assert!(!match_lock_file(Path::new(name)), "{name} must not match");
    }
}

//  match_generated_path: edge shapes

#[test]
fn match_generated_path_handles_edge_shapes() {
    // Empty path: parent() is None.
    assert!(!match_generated_path(Path::new("")));
    // Root has no parent; `..` yields no Normal component.
    assert!(!match_generated_path(Path::new("/")));
    assert!(!match_generated_path(Path::new("..")));
    // CurDir components are skipped, Normal ones still match.
    assert!(match_generated_path(Path::new("./gen/foo.rs")));
    // Absolute paths work too — git hands us relative ones, but be robust.
    assert!(match_generated_path(Path::new("/workspace/gen/foo.rs")));
}

//  match_generated_name: real-world shapes

#[test]
fn match_generated_name_matches_nested_real_paths() {
    for path in [
        // protobuf output under a models dir
        "src/models/user.pb.go",
        // gRPC service stubs still end in a registered suffix
        "api/v1/health_grpc.pb.go",
        "grpc_out/service_grpc_pb2.py",
        // Jest snapshot stored under the convention directory
        "__snapshots__/button.test.tsx.snap",
        // Xcode project lives inside the .xcodeproj bundle
        "App.xcodeproj/project.pbxproj",
    ] {
        assert!(match_generated_name(Path::new(path)), "{path} must match");
    }
}

#[test]
fn match_generated_name_rejects_hand_written_declarations() {
    // `.d.ts` is a hand-written TypeScript declaration, not a codegen
    // artifact — no registered suffix may catch it.
    for name in ["index.d.ts", "types.d.ts", "global.d.ts", "env.d.ts"] {
        assert!(
            !match_generated_name(Path::new(name)),
            "{name} must not match"
        );
    }
}

//  match_generated_header: real tool output

#[test]
fn match_generated_header_matches_real_tool_output() {
    let headers: &[&[u8]] = &[
        // protoc (JavaScript): banner comment, marker on line 3
        b"/* eslint-disable */\n// @ts-nocheck\n/**\n * This file is a generated file. Do not edit.\n */\n",
        // sqlc (Go)
        b"// Code generated by sqlc. DO NOT EDIT.\npackage db\n",
        // protoc (Java) exact banner
        b"// Generated by the protocol buffer compiler.  DO NOT EDIT!\n// source: foo.proto\n",
        // TS / Dart style, inside a block comment
        b"/* @generated */\n",
        // Comment prefix with no space after it
        b"//Generated by buf v1\n",
        // Tab-indented comment
        b"\t# Generated by some-codegen\n",
        // CRLF line endings (Windows-authored generators)
        b"-- Generated by sqlc 1.24.0\r\n",
        // UTF-8 BOM before a marker — substring scan is BOM-agnostic
        b"\xef\xbb\xbf// Code generated by x\n",
    ];
    for header in headers {
        assert!(
            match_generated_header(header),
            "{:?} must match",
            String::from_utf8_lossy(header)
        );
    }
}

/// A leading UTF-8 BOM is not content — the line-anchored fallback must
/// still see the comment. Requires the BOM strip in
/// `match_generated_header`.
// #[test]
// fn match_generated_header_bom_does_not_break_comment_fallback() {
//     assert!(match_generated_header(b"\xef\xbb\xbf// Generated by z\n"));
// }

#[test]
fn match_generated_header_rejects_config_and_docs() {
    let headers: &[&[u8]] = &[
        // JSON / YAML with a `generated` key are configs, not codegen
        b"{\"generated\": false}\n",
        b"generated: false\n",
        // Docs describing generated files — fallback needs "generated by"
        b"# Generated file registry\n",
        b"# How generated files are laid out\n",
    ];
    for header in headers {
        assert!(
            !match_generated_header(header),
            "{:?} must not match",
            String::from_utf8_lossy(header)
        );
    }
}

//  classify_by_name: real-world composition

#[test]
fn classify_by_name_real_world_shapes() {
    // Case-insensitive lock at the composition level
    assert_eq!(
        classify_by_name(Path::new("CARGO.LOCK")),
        Some(FileCategory::DependencyLock)
    );
    // Lock nested under a directory that merely looks generated
    assert_eq!(
        classify_by_name(Path::new("lock/Cargo.lock")),
        Some(FileCategory::DependencyLock)
    );
    // A file literally named `gen` is not a generated directory
    assert_eq!(classify_by_name(Path::new("gen")), None);
    // `lock` is not a marker directory and Cargo.toml is not a lock
    assert_eq!(classify_by_name(Path::new("lock/Cargo.toml")), None);
    // Both signals agree → Generated
    assert_eq!(
        classify_by_name(Path::new("__generated__/foo.pb.go")),
        Some(FileCategory::Generated)
    );
    // Suffix-only → Generated
    assert_eq!(
        classify_by_name(Path::new("tests/foo.snap")),
        Some(FileCategory::Generated)
    );
}
