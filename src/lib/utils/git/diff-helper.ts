export function isValidStatusToken(token: string): boolean {
  return /^([AMDT]|[RC]\d+)$/.test(token);
}

export function toInt(s: string): number | null {
  const num = parseInt(s, 10);
  return Number.isNaN(num) ? null : num;
}