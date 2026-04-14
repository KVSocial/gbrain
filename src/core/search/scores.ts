import type { SearchResult } from '../types.ts';

export function safeScore(value: unknown, fallback = 0): number {
  const score = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(score) ? score : fallback;
}

export function compareSearchScoreDesc(a: Pick<SearchResult, 'score'>, b: Pick<SearchResult, 'score'>): number {
  return safeScore(b.score) - safeScore(a.score);
}

export function formatSearchScore(value: unknown): string {
  return safeScore(value).toFixed(4);
}
