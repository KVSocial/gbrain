import { readFileSync, writeFileSync, mkdirSync, chmodSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';
import type { EngineConfig } from './types.ts';
import type { StorageConfig } from './storage.ts';

// Lazy-evaluated to avoid calling homedir() at module scope (breaks in serverless/bundled environments)
function getConfigDir() { return join(homedir(), '.gbrain'); }
function getConfigPath() { return join(getConfigDir(), 'config.json'); }

export interface GBrainConfig {
  engine: 'postgres' | 'pglite';
  database_url?: string;
  database_path?: string;
  openrouter_api_key?: string;
  openai_api_key?: string;
  anthropic_api_key?: string;
  embedding_provider?: 'openai' | 'openrouter';
  embedding_model?: string;
  expansion_provider?: 'anthropic' | 'openrouter';
  expansion_model?: string;
  storage?: StorageConfig;
}

/**
 * Load config with credential precedence: env vars > config file.
 * Plugin config is handled by the plugin runtime injecting env vars.
 */
export function loadConfig(): GBrainConfig | null {
  let fileConfig: GBrainConfig | null = null;
  try {
    const raw = readFileSync(getConfigPath(), 'utf-8');
    fileConfig = JSON.parse(raw) as GBrainConfig;
  } catch { /* no config file */ }

  // Try env vars
  const dbUrl = process.env.GBRAIN_DATABASE_URL || process.env.DATABASE_URL;

  if (!fileConfig && !dbUrl) return null;

  // Infer engine type if not explicitly set
  const inferredEngine: 'postgres' | 'pglite' = fileConfig?.engine
    || (fileConfig?.database_path ? 'pglite' : 'postgres');

  // Merge: env vars override config file
  const merged = {
    ...fileConfig,
    engine: inferredEngine,
    ...(dbUrl ? { database_url: dbUrl } : {}),
    ...(process.env.OPENROUTER_API_KEY ? { openrouter_api_key: process.env.OPENROUTER_API_KEY } : {}),
    ...(process.env.OPENAI_API_KEY ? { openai_api_key: process.env.OPENAI_API_KEY } : {}),
    ...(process.env.ANTHROPIC_API_KEY ? { anthropic_api_key: process.env.ANTHROPIC_API_KEY } : {}),
    ...(process.env.GBRAIN_EMBEDDING_PROVIDER ? { embedding_provider: process.env.GBRAIN_EMBEDDING_PROVIDER } : {}),
    ...(process.env.GBRAIN_EMBEDDING_MODEL ? { embedding_model: process.env.GBRAIN_EMBEDDING_MODEL } : {}),
    ...(process.env.GBRAIN_EXPANSION_PROVIDER ? { expansion_provider: process.env.GBRAIN_EXPANSION_PROVIDER } : {}),
    ...(process.env.GBRAIN_EXPANSION_MODEL ? { expansion_model: process.env.GBRAIN_EXPANSION_MODEL } : {}),
  };
  return merged as GBrainConfig;
}

export function saveConfig(config: GBrainConfig): void {
  mkdirSync(getConfigDir(), { recursive: true });
  writeFileSync(getConfigPath(), JSON.stringify(config, null, 2) + '\n', { mode: 0o600 });
  try {
    chmodSync(getConfigPath(), 0o600);
  } catch {
    // chmod may fail on some platforms
  }
}

export function toEngineConfig(config: GBrainConfig): EngineConfig {
  return {
    engine: config.engine,
    database_url: config.database_url,
    database_path: config.database_path,
  };
}

export function configDir(): string {
  return join(homedir(), '.gbrain');
}

export function configPath(): string {
  return join(configDir(), 'config.json');
}
