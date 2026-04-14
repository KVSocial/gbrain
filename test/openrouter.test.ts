import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import {
  DEFAULT_EMBEDDING_DIMENSIONS,
  OPENROUTER_BASE_URL,
  resolveEmbeddingConfig,
  resolveEmbeddingDimensions,
  resolveExpansionConfig,
} from '../src/core/ai-config.ts';
import { schemaWithEmbeddingDimensions } from '../src/core/schema-dimensions.ts';
import { buildOpenRouterEmbeddingBody, normalizeEmbedding } from '../src/core/embedding-utils.ts';
import { expandQuery, parseExpansionJson } from '../src/core/search/expansion.ts';

const ENV_KEYS = [
  'OPENROUTER_API_KEY',
  'OPENAI_API_KEY',
  'ANTHROPIC_API_KEY',
  'GBRAIN_EMBEDDING_PROVIDER',
  'GBRAIN_EMBEDDING_MODEL',
  'GBRAIN_EMBEDDING_DIMENSIONS',
  'GBRAIN_EXPANSION_PROVIDER',
  'GBRAIN_EXPANSION_MODEL',
  'OPENROUTER_HTTP_REFERER',
  'OPENROUTER_APP_TITLE',
] as const;

const savedEnv = new Map<string, string | undefined>();
const originalFetch = globalThis.fetch;

beforeEach(() => {
  for (const key of ENV_KEYS) {
    savedEnv.set(key, process.env[key]);
    delete process.env[key];
  }
});

afterEach(() => {
  for (const key of ENV_KEYS) {
    const value = savedEnv.get(key);
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
  savedEnv.clear();
  globalThis.fetch = originalFetch;
});

describe('OpenRouter provider config', () => {
  test('uses one OpenRouter key with independent embedding and expansion models', () => {
    process.env.OPENROUTER_API_KEY = 'sk-or-test';
    process.env.GBRAIN_EMBEDDING_MODEL = 'openai/text-embedding-3-small';
    process.env.GBRAIN_EMBEDDING_DIMENSIONS = '4096';
    process.env.GBRAIN_EXPANSION_MODEL = 'openai/gpt-4o-mini';

    const embedding = resolveEmbeddingConfig();
    const expansion = resolveExpansionConfig();

    expect(embedding.provider).toBe('openrouter');
    expect(embedding.apiKey).toBe('sk-or-test');
    expect(embedding.model).toBe('openai/text-embedding-3-small');
    expect(embedding.dimensions).toBe(4096);
    expect(expansion.provider).toBe('openrouter');
    expect(expansion.apiKey).toBe('sk-or-test');
    expect(expansion.model).toBe('openai/gpt-4o-mini');
  });

  test('defaults embedding dimensions to 1536 when unset', () => {
    expect(resolveEmbeddingDimensions()).toBe(DEFAULT_EMBEDDING_DIMENSIONS);
  });

  test('rejects invalid embedding dimensions', () => {
    process.env.GBRAIN_EMBEDDING_DIMENSIONS = 'not-a-number';
    expect(() => resolveEmbeddingDimensions()).toThrow('positive integer');
  });
});

describe('OpenRouter schema dimensions', () => {
  test('renders new brain schema with configured embedding dimensions', () => {
    const sql = schemaWithEmbeddingDimensions(`
      CREATE TABLE content_chunks (embedding vector(1536));
      INSERT INTO config (key, value) VALUES ('embedding_dimensions', '1536');
      CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON content_chunks USING hnsw (embedding vector_cosine_ops);
    `, 4096);

    expect(sql).toContain('embedding vector(4096)');
    expect(sql).toContain("('embedding_dimensions', '4096')");
    expect(sql).not.toContain('USING hnsw');
  });

  test('preserves HNSW index for default dimensions', () => {
    const sql = schemaWithEmbeddingDimensions(
      'CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON content_chunks USING hnsw (embedding vector_cosine_ops);',
      1536,
    );

    expect(sql).toContain('USING hnsw');
  });
});

describe('OpenRouter embeddings', () => {
  test('builds embedding requests with the configured model', () => {
    const body = buildOpenRouterEmbeddingBody('openai/text-embedding-3-small', ['alpha', 'beta']);

    expect(body.model).toBe('openai/text-embedding-3-small');
    expect(body.input).toEqual(['alpha', 'beta']);
    expect(body.encoding_format).toBe('float');
  });

  test('fails clearly when an embedding model returns incompatible dimensions', () => {
    expect(() => normalizeEmbedding('qwen/qwen3-embedding-0.6b', 1536, [0.1, 0.2, 0.3]))
      .toThrow('expects 1536');
  });
});

describe('OpenRouter query expansion', () => {
  test('sends expansion requests to OpenRouter with the configured model', async () => {
    process.env.OPENROUTER_API_KEY = 'sk-or-test';
    process.env.GBRAIN_EXPANSION_PROVIDER = 'openrouter';
    process.env.GBRAIN_EXPANSION_MODEL = 'openai/gpt-4o-mini';

    let requestBody: any;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toBe(`${OPENROUTER_BASE_URL}/chat/completions`);
      requestBody = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({
        choices: [
          { message: { content: '{"alternative_queries":["fintech founders","startup finance leaders"]}' } },
        ],
      }), { status: 200 });
    }) as typeof fetch;

    const expanded = await expandQuery('who works in financial technology');

    expect(requestBody.model).toBe('openai/gpt-4o-mini');
    expect(requestBody.response_format).toEqual({ type: 'json_object' });
    expect(expanded).toEqual([
      'who works in financial technology',
      'fintech founders',
      'startup finance leaders',
    ]);
  });

  test('parses fenced JSON and ignores extra alternatives', () => {
    expect(parseExpansionJson('```json\n{"alternative_queries":["a","b","c"]}\n```')).toEqual(['a', 'b']);
  });
});
