use std::path::{Path, PathBuf};

use crate::core::git::diff::{
    classify_by_header, classify_by_name, match_generated_header, match_generated_name,
    match_generated_path, match_lock_file,
};
use crate::core::git::types::FileCategory;

//  helpers

/// Fold a canonical registry name to the *other* casing so every
/// assertion sees a form different from the registry entry itself:
/// mixed-case names fold down, all-lowercase names fold up.
fn case_variant(name: &str) -> String {
    if name.bytes().any(|b| b.is_ascii_uppercase()) {
        name.to_ascii_lowercase()
    } else {
        name.to_ascii_uppercase()
    }
}

//  match_lock_file: full registry coverage (R-01)

/// R-01: mirror of `LOCK_FILES` in src/core/git/diff/rules.rs. The
/// section comments match the registry one-for-one — when the registry
/// gains or loses an entry, update the same group here. Each entry is
/// asserted in three forms: exact, case variant, nested deep path.
#[test]
fn match_lock_file_covers_full_registry() {
    let registry: &[&str] = &[
        // JS / TS
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lock",
        "deno.lock",
        // Rust
        "Cargo.lock",
        // Go
        "go.sum",
        "go.work.sum",
        "Gopkg.lock",
        "glide.lock",
        // Python
        "Pipfile.lock",
        "poetry.lock",
        "pdm.lock",
        "uv.lock",
        "conda-lock.yml",
        "conda-lock.yaml",
        "pixi.lock",
        // Ruby
        "Gemfile.lock",
        "gems.locked",
        // PHP
        "composer.lock",
        // Dart / Flutter
        "pubspec.lock",
        // Elixir / Erlang
        "mix.lock",
        "rebar.lock",
        // .NET
        "packages.lock.json",
        "paket.lock",
        // Apple / Swift
        "Package.resolved",
        "Podfile.lock",
        "Cartfile.resolved",
        // C / C++
        "conan.lock",
        "vcpkg-lock.json",
        // JVM
        "gradle.lockfile",
        "MODULE.bazel.lock",
        // Infra
        ".terraform.lock.hcl",
        "flake.lock",
        "Chart.lock",
        "helmfile.lock",
        // Misc package managers (high-signal names only)
        "shard.lock",
        "stack.yaml.lock",
        "cabal.project.freeze",
        "cpanfile.snapshot",
        "dub.selections.json",
        "renv.lock",
        "jsonnetfile.lock.json",
        "Brewfile.lock.json",
        "Berksfile.lock",
        "Policyfile.lock.json",
        "Puppetfile.lock",
        "spack.lock",
    ];

    for name in registry {
        assert!(
            match_lock_file(Path::new(name)),
            "{name}: exact basename must match"
        );

        let variant = case_variant(name);
        assert!(
            match_lock_file(Path::new(&variant)),
            "{name}: case variant {variant} must match"
        );

        let nested = format!("packages/deep/dir/{name}");
        assert!(
            match_lock_file(Path::new(&nested)),
            "{name}: nested {nested} must match — only the basename is compared"
        );
    }
}

/// R-02: manifests, decorated and truncated lookalikes must not match.
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

/// R-03: paths without a basename have nothing to compare against.
#[test]
fn matchers_reject_paths_without_file_name() {
    for path in ["/", "..", "."] {
        assert!(
            !match_lock_file(Path::new(path)),
            "{path}: lock match must be false"
        );
        assert!(
            !match_generated_name(Path::new(path)),
            "{path}: generated-name match must be false"
        );
    }
}

/// R-04: non-UTF-8 basenames cannot be compared at all — every matcher
/// must fail closed and the Phase A composition stays `None`.
#[cfg(unix)]
#[test]
fn non_utf8_name_fails_closed_across_all_matchers() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let path = PathBuf::from(OsString::from_vec(b"Cargo\xff.lock".to_vec()));

    assert!(!match_lock_file(&path));
    assert!(!match_generated_name(&path));
    assert!(!match_generated_path(&path));
    assert_eq!(classify_by_name(&path), None);
}

/// R-05: precision > recall — a generic `*.lock` basename is NOT a
/// dependency lock of a known ecosystem.
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

//  match_generated_path: marker directories (R-06 .. R-08)

/// R-06: all 8 marker directories, each in both casings, plus a deep
/// ancestor chain — the marker may sit at any depth.
#[test]
fn match_generated_path_covers_all_marker_directories() {
    let dirs = [
        "generated",
        "generated-sources",
        "generated_sources",
        "generated-test-sources",
        "__generated__",
        "_generated",
        "gensrc",
    ];

    for dir in dirs {
        let path = format!("src/{dir}/foo.rs");
        assert!(match_generated_path(Path::new(&path)), "{path} must match");

        let variant = case_variant(dir);
        let path = format!("src/{variant}/foo.rs");
        assert!(
            match_generated_path(Path::new(&path)),
            "{path} must match — directory match is case-insensitive"
        );
    }

    // Marker anywhere in the ancestor chain, not only the direct parent.
    assert!(match_generated_path(Path::new("a/b/generated/c.rs")));
}

/// R-07: `generated` as a basename, and superset / decorated directory
/// names must not match.
#[test]
fn match_generated_path_rejects_near_misses() {
    let paths = [
        // `generated` as a basename, not a directory component
        "generated.rs",
        "src/generated.txt",
        // A file *named* gen — the parent is `src`
        "src/gen",
        // Regression: a directory literally named `gen` is NOT a marker
        // (the name is too common in hand-written layouts). Any casing.
        "gen/foo.rs",
        "src/gen/foo.rs",
        "src/GEN/foo.rs",
        "src/Gen/bar.rs",
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
/// R-08: degenerate shapes and absolute paths.
#[test]
fn match_generated_path_handles_edge_shapes() {
    // Empty path / root / `..`: parent() is None or has no Normal component.
    assert!(!match_generated_path(Path::new("")));
    assert!(!match_generated_path(Path::new("/")));
    assert!(!match_generated_path(Path::new("..")));
    // CurDir components are skipped, Normal ones still match.
    assert!(match_generated_path(Path::new("./generated/foo.rs")));
    // git hands us relative paths, but absolute ones must work too.
    assert!(match_generated_path(Path::new(
        "/workspace/generated/foo.rs"
    )));
}

//  match_generated_name: suffix registry (R-09 .. R-15)

/// R-09: the full protobuf / gRPC family — all 14 registered suffixes.
#[test]
fn match_generated_name_covers_protobuf_family() {
    for suffix in [
        ".pb.cc",
        ".pb.h",
        ".pb.go",
        ".pb.rs",
        ".pb.swift",
        ".pb.ts",
        ".pb.js",
        ".pb.dart",
        ".pbgrpc.dart",
        "_pb.py",
        "_pb.js", // protoc JS actually emits `foo_pb.js`
        "_pb2.py",
        "_pb_grpc.py",
        "_pb2_grpc.py",
    ] {
        let name = format!("user{suffix}");
        assert!(match_generated_name(Path::new(&name)), "{name} must match");
    }
}

/// R-10: .NET / Dart / buf / oapi-codegen / `generated`-infix suffixes
/// — all 22 registered suffixes, fully enumerated.
#[test]
fn match_generated_name_covers_platform_conventions() {
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
        // explicit `.generated` infixes
        "types.generated.go",
        "types.generated.js",
        "types.generated.jsx",
        "types.generated.ts",
        "types.generated.tsx",
        // explicit `_generated` infixes
        "types_generated.go",
        "types_generated.h",
        "types_generated.java",
        "types_generated.js",
        "types_generated.py",
        "types_generated.rs",
        "types_generated.ts",
    ] {
        assert!(match_generated_name(Path::new(name)), "{name} must match");
    }
}

/// R-11: bundler output & sourcemaps — all 8 registered suffixes.
#[test]
fn match_generated_name_covers_bundler_output() {
    for name in [
        "style.min.css",
        "bundle.min.js",
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

/// R-12: the suffix match runs on the lowercased basename.
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

/// R-13: the match must be `ends_with`, not `contains` — the historical
/// bug classified `foo.snap.txt` and `grab_pb.py.bak` as generated.
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

/// R-14: suffixes nested inside realistic project layouts.
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

/// R-15: `.d.ts` files are hand-written TypeScript declarations — no
/// registered suffix may catch them.
#[test]
fn match_generated_name_rejects_hand_written_declarations() {
    for name in ["index.d.ts", "types.d.ts", "global.d.ts", "env.d.ts"] {
        assert!(
            !match_generated_name(Path::new(name)),
            "{name} must not match"
        );
    }
}

//  match_generated_header: blob-head markers (R-16 .. R-25, R-31)

/// R-16: the bare-marker substring scan (case-insensitive, anywhere in
/// the head) plus the edit-warning + generation-stem line branch.
#[test]
fn match_generated_header_matches_known_markers() {
    let headers: &[&[u8]] = &[
        b"// Code generated by protoc-gen-go. DO NOT EDIT.\n",
        b"@generated by my-codegen\n",
        b"/* Auto-generated file. */",
        b"# This file was automatically generated by CMake.\n",
        b"// AUTO-GENERATED, do not commit by hand\n",
        b"/* autogenerated by gcc */",
        // Edit warning + generation stem on the same line
        b"// GENERATED OUTPUT - DO NOT EDIT.\n",
        b"// Code generator output - do not modify.\n",
        b"// Generated by protoc v3.21\n",
        b"// Generated by the protocol buffer compiler.  DO NOT EDIT!\n",
        // Line phrases after a comment prefix
        b"// this file is generated by `cargo build`\n",
        b"// this file was generated by the build system\n",
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

/// R-17: the line-anchored fallback — a comment prefix followed by a
/// body starting with "generated by". None of these contain a bare
/// marker substring.
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

/// R-18: HTML comments are part of the registered prefix set —
/// hugo / jekyll generators emit them.
#[test]
fn match_generated_header_matches_html_comment_prefix() {
    assert!(match_generated_header(
        b"<!-- Generated by hugo 0.120 -->\n<body></body>\n"
    ));
}

/// R-19: `generated` wording in code or prose, without a marker or a
/// comment body starting with "generated by".
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

/// R-20: blob heads may be invalid UTF-8 — lossy conversion must not
/// panic, and markers in the valid part still hit.
#[test]
fn match_generated_header_tolerates_invalid_utf8() {
    assert!(match_generated_header(b"\xff\xfe// Code generated by x\n"));
    assert!(!match_generated_header(b"\xff\xfe\x00\x01 plain bytes"));
}

/// R-21: license prose is full of modify/edit wording but never a
/// generator declaration — this tiering exists to keep licenses in
/// the SemanticText layer.
#[test]
fn match_generated_header_rejects_license_headers() {
    let licenses: &[&[u8]] = &[
        b"// Licensed under the Apache License, Version 2.0 (the \"License\");\n// you may not use this file except in compliance with the License.\n// You may modify your copy or copies of the Program.\n",
        b"/* Permission is hereby granted, free of charge, to any person obtaining\n * a copy of this software... to use, copy, modify, merge, publish.\n * This file may not be edited except according to the terms above.\n */",
        b"// This program is distributed in the hope that it will be useful,\n// but WITHOUT ANY WARRANTY; without even the implied warranty of\n// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n",
    ];
    for header in licenses {
        assert!(
            !match_generated_header(header),
            "license prose must not be classified as generated: {:?}",
            String::from_utf8_lossy(header)
        );
    }
}

/// R-22: a bare edit warning describes hand-written files and
/// templates. The edit-warning branch is LINE-level — warning and
/// generation stem on different lines do not combine.
#[test]
fn match_generated_header_rejects_bare_and_split_edit_warnings() {
    let headers: &[&[u8]] = &[
        b"# DO NOT EDIT - copy to .env.local\n",
        b"// Do Not Edit",
        b"// DO NOT MODIFY this section\n",
        b"Please do not edit this file by hand.\n",
        // Warning and stem on different lines — each line fails alone
        b"// do not edit\n// regenerated weekly by cron\n",
        // Warning inside code or a string literal, not a comment
        b"throw new Error(\"do not edit generated files\");\n",
        b"let msg = \"do not modify the generated output\";\n",
        // Trailing comment — the warning does not open the line
        b"x = 1; // do not edit, generated by tool\n",
        // "regenerate" / "generative" wording is not a banner
        b"// TODO: regenerate fixtures; do not edit manually\n",
        b"// generative config \xE2\x80\x94 do not edit by hand\n",
    ];
    for header in headers {
        assert!(
            !match_generated_header(header),
            "{:?} must not match",
            String::from_utf8_lossy(header)
        );
    }
}

/// R-23: generation wording WITHOUT a comment prefix and without a
/// registered marker substring is prose, not codegen output.
#[test]
fn match_generated_header_rejects_bare_generation_lines() {
    let headers: &[&[u8]] = &[
        b"Generated by my-tool\n",
        b"This file was generated by the build system.\n",
    ];
    for header in headers {
        assert!(
            !match_generated_header(header),
            "{:?} must not match",
            String::from_utf8_lossy(header)
        );
    }
}

/// R-24: byte-exact banners from real generators — CRLF, BOM, no space
/// after the prefix, tab indentation.
#[test]
fn match_generated_header_matches_real_tool_output() {
    let headers: &[&[u8]] = &[
        // protoc (JavaScript): signal on line 4, inside a block comment
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
        // UTF-8 BOM before a marker — the substring scan is BOM-agnostic
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

/// R-25: configs and docs that merely talk about generation.
#[test]
fn match_generated_header_rejects_config_and_docs() {
    let headers: &[&[u8]] = &[
        // JSON / YAML with a `generated` key are configs, not codegen
        b"{\"generated\": false}\n",
        b"generated: false\n",
        // Docs describing generated files — the fallback needs a
        // comment body that *starts with* "generated by"
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

//  classify_by_name / classify_by_header: composition (R-26 .. R-30)

/// R-26: DependencyLock outranks Generated — the basename wins even
/// inside a marker directory.
#[test]
fn classify_by_name_lock_beats_generated() {
    assert_eq!(
        classify_by_name(Path::new("generated/package-lock.json")),
        Some(FileCategory::DependencyLock)
    );
}

/// R-27: all three Generated signals — marker dir, marker dir plus
/// suffix, suffix alone.
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
    // Marker dir alone
    assert_eq!(
        classify_by_name(Path::new("src/generated/foo.rs")),
        Some(FileCategory::Generated)
    );
    // Marker dir + suffix
    assert_eq!(
        classify_by_name(Path::new("generated-sources/api.pb.go")),
        Some(FileCategory::Generated)
    );
    // Regression: a bare `gen` parent contributes nothing — the suffix
    // alone decides here.
    assert_eq!(
        classify_by_name(Path::new("gen/foo.pb.rs")),
        Some(FileCategory::Generated)
    );
}

/// R-28: ordinary files fall through to None (→ Phase B probe).
#[test]
fn classify_by_name_returns_none_for_ordinary_files() {
    assert_eq!(classify_by_name(Path::new("src/main.rs")), None);
    assert_eq!(classify_by_name(Path::new("docs/readme.md")), None);
    // Precision regression: `.snap` must be a true suffix.
    assert_eq!(classify_by_name(Path::new("foo.snap.txt")), None);
}

/// R-29: realistic compositions at the Phase A level.
#[test]
fn classify_by_name_real_world_shapes() {
    let cases: &[(&str, Option<FileCategory>)] = &[
        // Case-insensitive lock at the composition level
        ("CARGO.LOCK", Some(FileCategory::DependencyLock)),
        // Lock nested under a directory that merely looks generated
        ("lock/Cargo.lock", Some(FileCategory::DependencyLock)),
        // A file literally named `gen` is not a generated directory
        ("gen", None),
        // Regression: a `gen` parent directory is not a marker either
        ("src/gen/widget.rs", None),
        // `lock` is not a marker directory and Cargo.toml is not a lock
        ("lock/Cargo.toml", None),
        // Both signals agree → Generated
        ("__generated__/foo.pb.go", Some(FileCategory::Generated)),
        // Suffix-only → Generated
        ("tests/foo.snap", Some(FileCategory::Generated)),
    ];
    for (path, expected) in cases {
        assert_eq!(classify_by_name(Path::new(path)), *expected, "{path}");
    }
}

/// R-30: Phase B residue mapping — header says Generated, else None
/// (the classifier turns None into SemanticText).
#[test]
fn classify_by_header_maps_to_generated_or_none() {
    assert_eq!(
        classify_by_header(b"// Code generated by x. DO NOT EDIT.\n"),
        Some(FileCategory::Generated)
    );
    assert_eq!(classify_by_header(b"fn main() {}\n"), None);
    assert_eq!(classify_by_header(b""), None);
}

/// R-31: KNOWN TRADE-OFF, pinned deliberately. The marker scan is a
/// plain substring match over the whole head, so prose containing
/// "code generated by" classifies as Generated. If the scan ever
/// becomes line- or comment-anchored, update this case explicitly.
#[test]
fn match_generated_header_marker_in_prose_is_accepted_trade_off() {
    assert!(match_generated_header(
        b"This section explains the code generated by the pipeline.\n"
    ));
}
