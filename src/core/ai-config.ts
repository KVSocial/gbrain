import { loadConfig } from './config.ts';

export type EmbeddingProvider = 'openai' | 'openrouter';
export type ExpansionProvider = 'anthropic' | 'openrouter';

export const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';
export const DEFAULT_EMBEDDING_DIMENSIONS = 1536;
export const DEFAULT_OPENAI_EMBEDDING_MODEL = 'text-embedding-3-large';
export const DEFAULT_OPENROUTER_EMBEDDING_MODEL = 'openai/text-embedding-3-large';
export const DEFAULT_ANTHROPIC_EXPANSION_MODEL = 'claude-haiku-4-5-20251001';
export const DEFAULT_OPENROUTER_EXPANSION_MODEL = 'openai/gpt-4o-mini';

export interface EmbeddingProviderConfig {
  provider: EmbeddingProvider;
  apiKey: string | null;
  model: string;
  dimensions: number;
}

export interface ExpansionProviderConfig {
  provider: ExpansionProvider;
  apiKey: string | null;
  model: string;
}

export function resolveEmbeddingConfig(): EmbeddingProviderConfig {
  const config = loadConfig();
  const openrouterKey = process.env.OPENROUTER_API_KEY || config?.openrouter_api_key || null;
  const openaiKey = process.env.OPENAI_API_KEY || config?.openai_api_key || null;
  const explicitProvider = parseEmbeddingProvider(
    process.env.GBRAIN_EMBEDDING_PROVIDER || config?.embedding_provider,
  );
  const provider = explicitProvider || (openrouterKey ? 'openrouter' : 'openai');

  return {
    provider,
    apiKey: provider === 'openrouter' ? openrouterKey : openaiKey,
    model: process.env.GBRAIN_EMBEDDING_MODEL
      || config?.embedding_model
      || (provider === 'openrouter'
        ? DEFAULT_OPENROUTER_EMBEDDING_MODEL
        : DEFAULT_OPENAI_EMBEDDING_MODEL),
    dimensions: resolveEmbeddingDimensions(config?.embedding_dimensions),
  };
}

export function resolveEmbeddingDimensions(configValue?: number | string): number {
  const raw = process.env.GBRAIN_EMBEDDING_DIMENSIONS ?? configValue;
  if (raw === undefined || raw === null || raw === '') return DEFAULT_EMBEDDING_DIMENSIONS;

  const parsed = typeof raw === 'number' ? raw : Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(
      `GBRAIN_EMBEDDING_DIMENSIONS must be a positive integer; got ${JSON.stringify(raw)}`,
    );
  }
  return parsed;
}

export function resolveExpansionConfig(): ExpansionProviderConfig {
  const config = loadConfig();
  const openrouterKey = process.env.OPENROUTER_API_KEY || config?.openrouter_api_key || null;
  const anthropicKey = process.env.ANTHROPIC_API_KEY || config?.anthropic_api_key || null;
  const explicitProvider = parseExpansionProvider(
    process.env.GBRAIN_EXPANSION_PROVIDER || config?.expansion_provider,
  );
  const provider = explicitProvider || (openrouterKey ? 'openrouter' : 'anthropic');

  return {
    provider,
    apiKey: provider === 'openrouter' ? openrouterKey : anthropicKey,
    model: process.env.GBRAIN_EXPANSION_MODEL
      || config?.expansion_model
      || (provider === 'openrouter'
        ? DEFAULT_OPENROUTER_EXPANSION_MODEL
        : DEFAULT_ANTHROPIC_EXPANSION_MODEL),
  };
}

export function canGenerateEmbeddings(): boolean {
  return Boolean(resolveEmbeddingConfig().apiKey);
}

export function canExpandQueries(): boolean {
  return Boolean(resolveExpansionConfig().apiKey);
}

export function openRouterHeaders(apiKey: string): Record<string, string> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  };
  const referer = process.env.OPENROUTER_HTTP_REFERER;
  const title = process.env.OPENROUTER_APP_TITLE || 'GBrain';
  if (referer) headers['HTTP-Referer'] = referer;
  if (title) headers['X-OpenRouter-Title'] = title;
  return headers;
}

function parseEmbeddingProvider(value: string | undefined): EmbeddingProvider | undefined {
  if (value === 'openai' || value === 'openrouter') return value;
  return undefined;
}

function parseExpansionProvider(value: string | undefined): ExpansionProvider | undefined {
  if (value === 'anthropic' || value === 'openrouter') return value;
  return undefined;
}
