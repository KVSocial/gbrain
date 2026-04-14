import { resolveEmbeddingDimensions } from './ai-config.ts';

export const HNSW_MAX_VECTOR_DIMENSIONS = 2000;

export function schemaWithEmbeddingDimensions(schemaSql: string, dimensions = resolveEmbeddingDimensions()): string {
  let rendered = schemaSql
    .replaceAll('vector(1536)', `vector(${dimensions})`)
    .replaceAll("('embedding_dimensions', '1536')", `('embedding_dimensions', '${dimensions}')`);

  if (dimensions > HNSW_MAX_VECTOR_DIMENSIONS) {
    rendered = rendered.replaceAll(
      'CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON content_chunks USING hnsw (embedding vector_cosine_ops);',
      `-- HNSW index skipped: pgvector hnsw supports vector dimensions up to ${HNSW_MAX_VECTOR_DIMENSIONS}.`,
    );
  }

  return rendered;
}
