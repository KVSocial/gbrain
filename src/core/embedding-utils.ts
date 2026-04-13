export interface OpenRouterEmbeddingRequestBody {
  model: string;
  input: string[];
  encoding_format: 'float';
}

export class EmbeddingDimensionError extends Error {}

export function buildOpenRouterEmbeddingBody(model: string, texts: string[]): OpenRouterEmbeddingRequestBody {
  return {
    model,
    input: texts,
    encoding_format: 'float',
  };
}

export function normalizeEmbedding(
  model: string,
  expectedDimensions: number,
  embedding: number[] | Float32Array,
): Float32Array {
  const vector = embedding instanceof Float32Array ? embedding : new Float32Array(embedding);
  if (vector.length !== expectedDimensions) {
    throw new EmbeddingDimensionError(
      `Embedding model ${model} returned ${vector.length} dimensions, `
      + `but this brain expects ${expectedDimensions}. Choose a ${expectedDimensions}-dimension `
      + 'embedding model, or rebuild the embedding schema before switching models.',
    );
  }
  return vector;
}
