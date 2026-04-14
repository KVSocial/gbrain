/**
 * Embedding Service
 * Ported from production Ruby implementation (embedding_service.rb, 190 LOC)
 *
 * OpenAI or OpenRouter embedding provider at 1536 dimensions.
 * Retry with exponential backoff (4s base, 120s cap, 5 retries).
 * 8000 character input truncation.
 */

import OpenAI from 'openai';
import {
  OPENROUTER_BASE_URL,
  type EmbeddingProviderConfig,
  openRouterHeaders,
  resolveEmbeddingConfig,
} from './ai-config.ts';
import {
  EmbeddingDimensionError,
  buildOpenRouterEmbeddingBody,
  normalizeEmbedding,
} from './embedding-utils.ts';

const MAX_CHARS = 8000;
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 4000;
const MAX_DELAY_MS = 120000;
const BATCH_SIZE = 100;

const openAiClients = new Map<string, OpenAI>();

class NonRetryableEmbeddingError extends Error {}

function getOpenAiClient(config: EmbeddingProviderConfig): OpenAI {
  if (!config.apiKey) {
    throw new Error('OPENAI_API_KEY is required for OpenAI embeddings');
  }
  const key = `${config.provider}:${config.apiKey}`;
  let client = openAiClients.get(key);
  if (!client) {
    client = new OpenAI({ apiKey: config.apiKey });
    openAiClients.set(key, client);
  }
  return client;
}

export async function embed(text: string): Promise<Float32Array> {
  const truncated = text.slice(0, MAX_CHARS);
  const result = await embedBatch([truncated]);
  return result[0];
}

export async function embedBatch(texts: string[]): Promise<Float32Array[]> {
  const config = resolveEmbeddingConfig();
  if (!config.apiKey) {
    const envName = config.provider === 'openrouter' ? 'OPENROUTER_API_KEY' : 'OPENAI_API_KEY';
    throw new Error(`${envName} is required for ${config.provider} embeddings`);
  }

  const truncated = texts.map(t => t.slice(0, MAX_CHARS));
  const results: Float32Array[] = [];

  // Process in batches of BATCH_SIZE
  for (let i = 0; i < truncated.length; i += BATCH_SIZE) {
    const batch = truncated.slice(i, i + BATCH_SIZE);
    const batchResults = await embedBatchWithRetry(config, batch);
    results.push(...batchResults);
  }

  return results;
}

async function embedBatchWithRetry(config: EmbeddingProviderConfig, texts: string[]): Promise<Float32Array[]> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      return config.provider === 'openrouter'
        ? await embedOpenRouterBatch(config, texts)
        : await embedOpenAiBatch(config, texts);
    } catch (e: unknown) {
      if (e instanceof NonRetryableEmbeddingError || e instanceof EmbeddingDimensionError) throw e;
      if (attempt === MAX_RETRIES - 1) throw e;

      // Check for rate limit with Retry-After header
      let delay = exponentialDelay(attempt);

      if (e instanceof OpenAI.APIError && e.status === 429) {
        const retryAfter = e.headers?.['retry-after'];
        if (retryAfter) {
          const parsed = parseInt(retryAfter, 10);
          if (!isNaN(parsed)) {
            delay = parsed * 1000;
          }
        }
      }

      await sleep(delay);
    }
  }

  // Should not reach here
  throw new Error('Embedding failed after all retries');
}

async function embedOpenAiBatch(config: EmbeddingProviderConfig, texts: string[]): Promise<Float32Array[]> {
  const response = await getOpenAiClient(config).embeddings.create({
    model: config.model,
    input: texts,
    dimensions: config.dimensions,
  });

  // Sort by index to maintain order
  const sorted = response.data.sort((a, b) => a.index - b.index);
  return sorted.map(d => normalizeEmbedding(config.model, config.dimensions, d.embedding));
}

async function embedOpenRouterBatch(config: EmbeddingProviderConfig, texts: string[]): Promise<Float32Array[]> {
  if (!config.apiKey) throw new Error('OPENROUTER_API_KEY is required for OpenRouter embeddings');

  const response = await fetch(`${OPENROUTER_BASE_URL}/embeddings`, {
    method: 'POST',
    headers: openRouterHeaders(config.apiKey),
    body: JSON.stringify(buildOpenRouterEmbeddingBody(config.model, texts)),
  });

  if (!response.ok) {
    const message = `OpenRouter embeddings failed (${response.status}): ${await response.text()}`;
    if (response.status !== 429 && response.status < 500) {
      throw new NonRetryableEmbeddingError(message);
    }
    throw new Error(message);
  }

  const body = await response.json() as {
    data?: { index?: number; embedding?: number[] }[];
  };
  const data = body.data || [];
  const sorted = data.sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  return sorted.map(d => normalizeEmbedding(config.model, config.dimensions, d.embedding || []));
}

function exponentialDelay(attempt: number): number {
  const delay = BASE_DELAY_MS * Math.pow(2, attempt);
  return Math.min(delay, MAX_DELAY_MS);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export {
  DEFAULT_OPENAI_EMBEDDING_MODEL as EMBEDDING_MODEL,
  DEFAULT_EMBEDDING_DIMENSIONS as EMBEDDING_DIMENSIONS,
} from './ai-config.ts';
