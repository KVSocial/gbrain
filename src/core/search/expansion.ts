/**
 * Multi-Query Expansion via Anthropic or OpenRouter chat models
 * Ported from production Ruby implementation (query_expansion_service.rb, 69 LOC)
 *
 * Skip queries < 3 words.
 * Generate 2 alternative phrasings via tool use.
 * Return original + alternatives (max 3 total).
 */

import Anthropic from '@anthropic-ai/sdk';
import {
  OPENROUTER_BASE_URL,
  type ExpansionProviderConfig,
  canExpandQueries,
  openRouterHeaders,
  resolveExpansionConfig,
} from '../ai-config.ts';

const MAX_QUERIES = 3;
const MIN_WORDS = 3;

const anthropicClients = new Map<string, Anthropic>();

function getAnthropicClient(config: ExpansionProviderConfig): Anthropic {
  if (!config.apiKey) {
    throw new Error('ANTHROPIC_API_KEY is required for Anthropic query expansion');
  }
  let client = anthropicClients.get(config.apiKey);
  if (!client) {
    client = new Anthropic({ apiKey: config.apiKey });
    anthropicClients.set(config.apiKey, client);
  }
  return client;
}

export async function expandQuery(query: string): Promise<string[]> {
  // CJK text is not space-delimited — count characters instead of whitespace-separated tokens
  const hasCJK = /[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/.test(query);
  const wordCount = hasCJK ? query.replace(/\s/g, '').length : (query.match(/\S+/g) || []).length;
  if (wordCount < MIN_WORDS) return [query];
  if (!canExpandQueries()) return [query];

  try {
    const config = resolveExpansionConfig();
    const alternatives = config.provider === 'openrouter'
      ? await callOpenRouterForExpansion(config, query)
      : await callAnthropicForExpansion(config, query);
    const all = [query, ...alternatives];
    // Deduplicate
    const unique = [...new Set(all.map(q => q.toLowerCase().trim()))];
    return unique.slice(0, MAX_QUERIES).map(q =>
      all.find(orig => orig.toLowerCase().trim() === q) || q,
    );
  } catch {
    return [query];
  }
}

async function callAnthropicForExpansion(config: ExpansionProviderConfig, query: string): Promise<string[]> {
  const response = await getAnthropicClient(config).messages.create({
    model: config.model,
    max_tokens: 300,
    tools: [
      {
        name: 'expand_query',
        description: 'Generate alternative phrasings of a search query to improve recall',
        input_schema: {
          type: 'object' as const,
          properties: {
            alternative_queries: {
              type: 'array',
              items: { type: 'string' },
              description: '2 alternative phrasings of the original query, each approaching the topic from a different angle',
            },
          },
          required: ['alternative_queries'],
        },
      },
    ],
    tool_choice: { type: 'tool', name: 'expand_query' },
    messages: [
      {
        role: 'user',
        content: `Generate 2 alternative search queries that would find relevant results for this question. Each alternative should approach the topic from a different angle or use different terminology.

Original query: "${query}"`,
      },
    ],
  });

  // Extract tool use result
  for (const block of response.content) {
    if (block.type === 'tool_use' && block.name === 'expand_query') {
      const input = block.input as { alternative_queries?: unknown };
      const alts = input.alternative_queries;
      if (Array.isArray(alts)) {
        return alts.map(String).slice(0, 2);
      }
    }
  }

  return [];
}

async function callOpenRouterForExpansion(config: ExpansionProviderConfig, query: string): Promise<string[]> {
  if (!config.apiKey) throw new Error('OPENROUTER_API_KEY is required for OpenRouter query expansion');

  const response = await fetch(`${OPENROUTER_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: openRouterHeaders(config.apiKey),
    body: JSON.stringify({
      model: config.model,
      max_tokens: 300,
      temperature: 0.2,
      response_format: { type: 'json_object' },
      messages: [
        {
          role: 'system',
          content: 'Return only valid JSON matching {"alternative_queries":["...","..."]}.',
        },
        {
          role: 'user',
          content: `Generate 2 alternative search queries that would find relevant results for this question. Each alternative should approach the topic from a different angle or use different terminology.

Original query: "${query}"`,
        },
      ],
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenRouter query expansion failed (${response.status}): ${await response.text()}`);
  }

  const body = await response.json() as {
    choices?: { message?: { content?: string | null } }[];
  };
  const content = body.choices?.[0]?.message?.content;
  return parseExpansionJson(content || '');
}

export function parseExpansionJson(content: string): string[] {
  const parsed = parseJsonObject(content);
  const alternatives = parsed?.alternative_queries;
  if (!Array.isArray(alternatives)) return [];

  return alternatives
    .map(value => String(value).trim())
    .filter(Boolean)
    .slice(0, 2);
}

function parseJsonObject(content: string): { alternative_queries?: unknown } | null {
  try {
    return JSON.parse(content) as { alternative_queries?: unknown };
  } catch {
    const start = content.indexOf('{');
    const end = content.lastIndexOf('}');
    if (start === -1 || end <= start) return null;
    try {
      return JSON.parse(content.slice(start, end + 1)) as { alternative_queries?: unknown };
    } catch {
      return null;
    }
  }
}
