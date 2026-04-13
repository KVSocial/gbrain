import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import {
  OPENROUTER_BASE_URL,
  resolveEmbeddingConfig,
  resolveExpansionConfig,
} from '../src/core/ai-config.ts';
import { buildOpenRouterEmbeddingBody, normalizeEmbedding } from '../src/core/embedding-utils.ts';
import { expandQuery, parseExpansionJson } from '../src/core/search/expansion.ts';

const ENV_KEYS = [
  'OPENROUTER_API_KEY',
  'OPENAI_API_KEY',
  'ANTHROPIC_API_KEY',
  'GBRAIN_EMBEDDING_PROVIDER',
  'GBRAIN_EMBEDDING_MODEL',
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
    process.env.GBRAIN_EXPANSION_MODEL = 'openai/gpt-4o-mini';

    const embedding = resolveEmbeddingConfig();
    const expansion = resolveExpansionConfig();

    expect(embedding.provider).toBe('openrouter');
    expect(embedding.apiKey).toBe('sk-or-test');
    expect(embedding.model).toBe('openai/text-embedding-3-small');
    expect(expansion.provider).toBe('openrouter');
    expect(expansion.apiKey).toBe('sk-or-test');
    expect(expansion.model).toBe('openai/gpt-4o-mini');
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
