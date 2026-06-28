import { describe, test, expect } from "bun:test";
import { isValidStatusToken, toInt } from "../../src/lib/utils/git";

// ═══════════════════════════════════════════════════════════════════════════
// isValidStatusToken()
// ═══════════════════════════════════════════════════════════════════════════

describe("isValidStatusToken", () => {
  
  // ── Valid single-letter status codes ──
  
  test("accepts 'A' (added)", () => {
    expect(isValidStatusToken("A")).toBe(true);
  });

  test("accepts 'M' (modified)", () => {
    expect(isValidStatusToken("M")).toBe(true);
  });

  test("accepts 'D' (deleted)", () => {
    expect(isValidStatusToken("D")).toBe(true);
  });

  test("accepts 'T' (type-changed)", () => {
    expect(isValidStatusToken("T")).toBe(true);
  });

  // ── Valid rename/copy status codes with similarity score ──

  test("accepts 'R100' (rename with 100% similarity)", () => {
    expect(isValidStatusToken("R100")).toBe(true);
  });

  test("accepts 'R95' (rename with 95% similarity)", () => {
    expect(isValidStatusToken("R95")).toBe(true);
  });

  test("accepts 'R0' (rename with 0% similarity)", () => {
    expect(isValidStatusToken("R0")).toBe(true);
  });

  test("accepts 'C100' (copy with 100% similarity)", () => {
    expect(isValidStatusToken("C100")).toBe(true);
  });

  test("accepts 'C80' (copy with 80% similarity)", () => {
    expect(isValidStatusToken("C80")).toBe(true);
  });

  test("accepts 'C0' (copy with 0% similarity)", () => {
    expect(isValidStatusToken("C0")).toBe(true);
  });

  test("accepts 'R1' (single-digit similarity score)", () => {
    expect(isValidStatusToken("R1")).toBe(true);
  });

  test("accepts 'C9' (single-digit similarity score)", () => {
    expect(isValidStatusToken("C9")).toBe(true);
  });

  test("accepts 'R999' (multi-digit similarity score)", () => {
    expect(isValidStatusToken("R999")).toBe(true);
  });

  // ── Invalid: malformed status codes ──

  test("rejects 'R' without similarity score", () => {
    expect(isValidStatusToken("R")).toBe(false);
  });

  test("rejects 'C' without similarity score", () => {
    expect(isValidStatusToken("C")).toBe(false);
  });

  test("rejects 'X' (unknown git status code)", () => {
    expect(isValidStatusToken("X")).toBe(false);
  });

  test("rejects 'U' (unmerged status code)", () => {
    expect(isValidStatusToken("U")).toBe(false);
  });

  test("rejects '?' (untracked status code)", () => {
    expect(isValidStatusToken("?")).toBe(false);
  });

  test("rejects '!' (ignored status code)", () => {
    expect(isValidStatusToken("!")).toBe(false);
  });

  // ── Invalid: lowercase variants ──

  test("rejects 'a' (lowercase added)", () => {
    expect(isValidStatusToken("a")).toBe(false);
  });

  test("rejects 'm' (lowercase modified)", () => {
    expect(isValidStatusToken("m")).toBe(false);
  });

  test("rejects 'r100' (lowercase rename)", () => {
    expect(isValidStatusToken("r100")).toBe(false);
  });

  test("rejects 'c80' (lowercase copy)", () => {
    expect(isValidStatusToken("c80")).toBe(false);
  });

  // ── Invalid: extra characters or suffixes ──

  test("rejects 'A_anyway' (status with trailing text)", () => {
    expect(isValidStatusToken("A_anyway")).toBe(false);
  });

  test("rejects 'M123' (invalid: M does not take similarity score)", () => {
    expect(isValidStatusToken("M123")).toBe(false);
  });

  test("rejects 'A100' (invalid: A does not take similarity score)", () => {
    expect(isValidStatusToken("A100")).toBe(false);
  });

  test("rejects 'D50' (invalid: D does not take similarity score)", () => {
    expect(isValidStatusToken("D50")).toBe(false);
  });

  test("rejects 'T99' (invalid: T does not take similarity score)", () => {
    expect(isValidStatusToken("T99")).toBe(false);
  });

  test("rejects 'R100extra' (status with trailing text)", () => {
    expect(isValidStatusToken("R100extra")).toBe(false);
  });

  test("rejects 'C95_suffix' (status with trailing text)", () => {
    expect(isValidStatusToken("C95_suffix")).toBe(false);
  });

  // ── Invalid: whitespace ──

  test("rejects 'A ' (status with trailing space)", () => {
    expect(isValidStatusToken("A ")).toBe(false);
  });

  test("rejects ' M' (status with leading space)", () => {
    expect(isValidStatusToken(" M")).toBe(false);
  });

  test("rejects 'R 100' (status with internal space)", () => {
    expect(isValidStatusToken("R 100")).toBe(false);
  });

  test("rejects '\\tA' (status with leading tab)", () => {
    expect(isValidStatusToken("\tA")).toBe(false);
  });

  test("rejects 'R100\\n' (status with trailing newline)", () => {
    expect(isValidStatusToken("R100\n")).toBe(false);
  });

  // ── Invalid: empty or special inputs ──

  test("rejects empty string", () => {
    expect(isValidStatusToken("")).toBe(false);
  });

  test("rejects whitespace-only string", () => {
    expect(isValidStatusToken("   ")).toBe(false);
  });

  test("rejects tab character", () => {
    expect(isValidStatusToken("\t")).toBe(false);
  });

  test("rejects newline character", () => {
    expect(isValidStatusToken("\n")).toBe(false);
  });

  // ── Invalid: numeric-only or non-letter prefixes ──

  test("rejects '100' (number without status letter)", () => {
    expect(isValidStatusToken("100")).toBe(false);
  });

  test("rejects '0' (single digit without status letter)", () => {
    expect(isValidStatusToken("0")).toBe(false);
  });

  test("rejects 'R-100' (negative similarity score)", () => {
    expect(isValidStatusToken("R-100")).toBe(false);
  });

  test("rejects 'R+100' (similarity score with plus sign)", () => {
    expect(isValidStatusToken("R+100")).toBe(false);
  });

  // ── Invalid: multi-character prefixes ──

  test("rejects 'AM' (multiple status codes)", () => {
    expect(isValidStatusToken("AM")).toBe(false);
  });

  test("rejects 'RC100' (two letters with score)", () => {
    expect(isValidStatusToken("RC100")).toBe(false);
  });

  test("rejects 'ADD' (repeated letter)", () => {
    expect(isValidStatusToken("ADD")).toBe(false);
  });

  // ── Invalid: unicode and special characters ──

  test("rejects 'Å' (unicode lookalike)", () => {
    expect(isValidStatusToken("Å")).toBe(false);
  });

  test("rejects 'М' (cyrillic M lookalike)", () => {
    expect(isValidStatusToken("М")).toBe(false);
  });

  test("rejects 'A\\0' (null byte)", () => {
    expect(isValidStatusToken("A\0")).toBe(false);
  });

  // ── Invalid: decimal or fractional scores ──

  test("rejects 'R100.5' (decimal similarity score)", () => {
    expect(isValidStatusToken("R100.5")).toBe(false);
  });

  test("rejects 'C95.0' (decimal similarity score)", () => {
    expect(isValidStatusToken("C95.0")).toBe(false);
  });

  test("rejects 'R1e2' (scientific notation)", () => {
    expect(isValidStatusToken("R1e2")).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// toInt()
// ═══════════════════════════════════════════════════════════════════════════

describe("toInt", () => {
  
  // ── Valid: positive integers ──

  test("converts '0' to 0", () => {
    expect(toInt("0")).toBe(0);
  });

  test("converts '1' to 1", () => {
    expect(toInt("1")).toBe(1);
  });

  test("converts '42' to 42", () => {
    expect(toInt("42")).toBe(42);
  });

  test("converts '999' to 999", () => {
    expect(toInt("999")).toBe(999);
  });

  test("converts '123456789' to 123456789", () => {
    expect(toInt("123456789")).toBe(123456789);
  });

  test("converts large number '2147483647' (max signed 32-bit)", () => {
    expect(toInt("2147483647")).toBe(2147483647);
  });

  // ── Valid: negative integers ──

  test("converts '-1' to -1", () => {
    expect(toInt("-1")).toBe(-1);
  });

  test("converts '-42' to -42", () => {
    expect(toInt("-42")).toBe(-42);
  });

  test("converts '-999' to -999", () => {
    expect(toInt("-999")).toBe(-999);
  });

  test("converts '-2147483648' (min signed 32-bit)", () => {
    expect(toInt("-2147483648")).toBe(-2147483648);
  });

  // ── Valid: leading zeros ──

  test("converts '007' to 7 (leading zeros ignored)", () => {
    expect(toInt("007")).toBe(7);
  });

  test("converts '0000' to 0 (all zeros)", () => {
    expect(toInt("0000")).toBe(0);
  });

  test("converts '00123' to 123", () => {
    expect(toInt("00123")).toBe(123);
  });

  // ── Valid: whitespace handling (parseInt trims by default) ──

  test("converts ' 42' (leading space) to 42", () => {
    expect(toInt(" 42")).toBe(42);
  });

  test("converts '42 ' (trailing space) to 42", () => {
    expect(toInt("42 ")).toBe(42);
  });

  test("converts '  123  ' (leading and trailing spaces) to 123", () => {
    expect(toInt("  123  ")).toBe(123);
  });

  test("converts '\\t42' (leading tab) to 42", () => {
    expect(toInt("\t42")).toBe(42);
  });

  test("converts '42\\n' (trailing newline) to 42", () => {
    expect(toInt("42\n")).toBe(42);
  });

  test("converts '\\n\\t 99 \\t\\n' (mixed whitespace) to 99", () => {
    expect(toInt("\n\t 99 \t\n")).toBe(99);
  });

  // ── Invalid: non-numeric strings ──

  test("returns null for empty string", () => {
    expect(toInt("")).toBe(null);
  });

  test("returns null for 'abc'", () => {
    expect(toInt("abc")).toBe(null);
  });

  test("returns null for 'hello'", () => {
    expect(toInt("hello")).toBe(null);
  });

  test("returns null for 'NaN'", () => {
    expect(toInt("NaN")).toBe(null);
  });

  test("returns null for 'undefined'", () => {
    expect(toInt("undefined")).toBe(null);
  });

  test("returns null for 'null'", () => {
    expect(toInt("null")).toBe(null);
  });

  // ── Invalid: whitespace-only strings ──

  test("returns null for single space ' '", () => {
    expect(toInt(" ")).toBe(null);
  });

  test("returns null for multiple spaces '    '", () => {
    expect(toInt("    ")).toBe(null);
  });

  test("returns null for tab '\\t'", () => {
    expect(toInt("\t")).toBe(null);
  });

  test("returns null for newline '\\n'", () => {
    expect(toInt("\n")).toBe(null);
  });

  test("returns null for mixed whitespace '\\t\\n  '", () => {
    expect(toInt("\t\n  ")).toBe(null);
  });

  // ── Invalid: decimal numbers ──

  test("converts '3.14' to 3 (parseInt truncates decimal)", () => {
    expect(toInt("3.14")).toBe(3);
  });

  test("converts '99.99' to 99", () => {
    expect(toInt("99.99")).toBe(99);
  });

  test("converts '-5.5' to -5", () => {
    expect(toInt("-5.5")).toBe(-5);
  });

  test("converts '0.1' to 0", () => {
    expect(toInt("0.1")).toBe(0);
  });

  // ── Invalid: special numeric values ──

  test("returns null for 'Infinity'", () => {
    expect(toInt("Infinity")).toBe(null);
  });

  test("returns null for '-Infinity'", () => {
    expect(toInt("-Infinity")).toBe(null);
  });

  // ── Invalid: scientific notation ──

  test("converts '1e2' to 1 (parseInt stops at 'e')", () => {
    expect(toInt("1e2")).toBe(1);
  });

  test("converts '5e10' to 5", () => {
    expect(toInt("5e10")).toBe(5);
  });

  test("converts '3.5e2' to 3", () => {
    expect(toInt("3.5e2")).toBe(3);
  });

  // ── Invalid: mixed alphanumeric ──

  test("converts '42abc' to 42 (parseInt stops at first non-digit)", () => {
    expect(toInt("42abc")).toBe(42);
  });

  test("converts '123xyz' to 123", () => {
    expect(toInt("123xyz")).toBe(123);
  });

  test("returns null for 'abc123' (leading letters)", () => {
    expect(toInt("abc123")).toBe(null);
  });

  test("returns null for 'x42' (leading letter)", () => {
    expect(toInt("x42")).toBe(null);
  });

  // ── Invalid: special characters ──

  test("returns null for '$100' (leading dollar sign)", () => {
    expect(toInt("$100")).toBe(null);
  });

  test("returns null for '#42' (leading hash)", () => {
    expect(toInt("#42")).toBe(null);
  });

  test("returns null for '@99' (leading at sign)", () => {
    expect(toInt("@99")).toBe(null);
  });

  test("converts '42%' to 42 (percent sign after number)", () => {
    expect(toInt("42%")).toBe(42);
  });

  test("returns null for '%42' (leading percent sign)", () => {
    expect(toInt("%42")).toBe(null);
  });

  // ── Invalid: plus sign ──

  test("converts '+100' to 100 (plus sign is valid for parseInt)", () => {
    expect(toInt("+100")).toBe(100);
  });

  test("converts '+0' to 0", () => {
    expect(toInt("+0")).toBe(0);
  });

  test("returns null for '+ 100' (space after plus)", () => {
    expect(toInt("+ 100")).toBe(null);
  });

  // ── Invalid: multiple signs ──

  test("returns null for '--42' (double minus)", () => {
    expect(toInt("--42")).toBe(null);
  });

  test("returns null for '++42' (double plus)", () => {
    expect(toInt("++42")).toBe(null);
  });

  test("returns null for '+-42' (mixed signs)", () => {
    expect(toInt("+-42")).toBe(null);
  });

  // ── Edge case: git binary marker ──

  test("returns null for '-' (git binary marker)", () => {
    expect(toInt("-")).toBe(null);
  });

  test("returns null for '--' (double dash)", () => {
    expect(toInt("--")).toBe(null);
  });

  // ── Edge case: very large numbers ──

  test("converts extremely large number string", () => {
    const result = toInt("999999999999999");
    expect(result).toBe(999999999999999);
  });

  test("converts number beyond safe integer range (may lose precision)", () => {
    // JavaScript safe integer range: -(2^53 - 1) to 2^53 - 1
    const result = toInt("99999999999999999999");
    expect(typeof result).toBe("number");
    expect(result).not.toBe(null);
  });

  // ── Edge case: Unicode digits (parseInt accepts them) ──

  test("converts '٤٢' (Arabic-Indic digits) correctly if parseInt supports them", () => {
    // parseInt behavior may vary; this documents expected behavior
    const result = toInt("٤٢");
    // Most environments return null as parseInt doesn't parse non-ASCII digits
    expect(result).toBe(null);
  });

  // ── Edge case: hex notation (parseInt with radix 10 should stop at 'x') ──

  test("converts '0x10' to 0 (parseInt base-10 stops at 'x')", () => {
    expect(toInt("0x10")).toBe(0);
  });

  test("converts '0xFF' to 0", () => {
    expect(toInt("0xFF")).toBe(0);
  });

  // ── Edge case: octal notation ──

  test("converts '0777' to 777 (leading zero treated as decimal in radix 10)", () => {
    expect(toInt("0777")).toBe(777);
  });

  test("converts '0o10' to 0 (parseInt base-10 stops at 'o')", () => {
    expect(toInt("0o10")).toBe(0);
  });

  // ── Edge case: binary notation ──

  test("converts '0b101' to 0 (parseInt base-10 stops at 'b')", () => {
    expect(toInt("0b101")).toBe(0);
  });
});
