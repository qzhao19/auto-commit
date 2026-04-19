
export function assertIsObject(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(name + " must be a table");
  }
  return value as Record<string, unknown>;
}

export function assertIsString(value: unknown, name: string): string {
  if (typeof value !== "string") {
    throw new Error(name + " must be a string");
  }
  return value;
}

export function assertIsInt(value: string, name: string): number {
  const normalized = value.trim();
  if (!Number.isInteger(Number(normalized))) {
    throw new Error(name + " must be an integer");
  }

  const n = Number(normalized);
  if (!Number.isSafeInteger(n)) {
    throw new Error(name + " is out of safe integer range");
  }
  return n;
}

export function assertIsTomlInt(value: unknown, name: string): number {
  if (typeof value === "number") {
    if (!Number.isInteger(value)) {
      throw new Error(name + " must be an integer");
    }
    if (!Number.isSafeInteger(value)) {
      throw new Error(name + " is out of safe integer range");
    }
    return value;
  }

  if (typeof value === "string") {
    return assertIsInt(value, name);
  }

  throw new Error(name + " must be an integer");
}

export function assertIsFloat(value: string, name: string): number {
  const normalized = value.trim();
  if (normalized === "") {
    throw new Error(name + " must be a number");
  }

  const n = Number(normalized);
  if (!Number.isFinite(n)) {
    throw new Error(name + " must be a finite number");
  }

  return n;
}

export function assertIsTomlFloat(value: unknown, name: string): number {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error(name + " must be a finite number");
    }
    return value;
  }

  if (typeof value === "string") {
    return assertIsFloat(value, name);
  }

  throw new Error(name + " must be a number");
}

export function assertIsBool(value: string, name: string): boolean {
  const normalized = value.trim().toLowerCase();
  if (normalized === "true" || normalized === "1") return true;
  if (normalized === "false" || normalized === "0") return false;
  throw new Error(name + " must be true/false or 1/0");
}

export function assertIsTomlBool(value: unknown, name: string): boolean {
  if (typeof value === "boolean") {
    return value;
  }

  if (typeof value === "string") {
    return assertIsBool(value, name);
  }
  throw new Error(name + " must be a boolean");
}

export function assertOnlyKnownKeys(
  obj: Record<string, unknown>,
  knownKeys: readonly string[],
  scope: string,
): void {
  for (const key of Object.keys(obj)) {
    if (!knownKeys.includes(key)) {
      throw new Error("Unknown key " + scope + "." + key);
    }
  }
}

