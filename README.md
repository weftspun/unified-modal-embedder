# unified-modal-embedder

Fused multimodal content embedder for V-Sekai assets. Reduces each modality to one
vector, concatenates, and Matryoshka-truncates to **768-d** — the single content
embedding the [`residual-fsq-recommender`](https://github.com/weftspun/residual-fsq-recommender)
consumes for **both** its semantic-ID tokenization (`project_in: 768 → 4`) and its
per-token aux (`768 → 4 × 192`), so the ID and aux describe an item in the *same*
space.

## Modality slots

| slot | encoder | notes |
|---|---|---|
| image / text | **Qwen3-VL-Embedding-2B** (frozen, Matryoshka) | Elixir/Bumblebee port on `weftspun/bumblebee@qwen3-vl-embedding` (vision tower done) |
| mesh (3D) | **TRELLIS.2 SLAT** → [`trellis2-mesh-vae-slang`](https://github.com/weftspun/trellis2-mesh-vae-slang) | geometry-faithful; not mean-pooled |
| audio | deferred | — |
| phenotype | deferred | — |

Each slot reduces to one vector; the fused concat is Matryoshka-truncated to 768-d.
Frozen encoders — this repo owns **fusion + truncation**, not encoder training.

## Upstream decisions

Design lives in [`weftspun/multimodal-semantic-ids`](https://github.com/weftspun/multimodal-semantic-ids):
`multimodal-foss-encoder-stack`, `mesh-only-scope-postpone-qwen3vl-embedder`,
`multimodal-residual-fsq-semantic-ids`.

## License

MIT © 2026 K. S. Ernest (iFire) Lee
