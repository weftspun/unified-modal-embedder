# RFD 1 Few-shot mesh classification without unfreezing the TRELLIS.2 encoder

| | |
|---|---|
| **State** | ideation |
| **Authors** | K. S. Ernest (iFire) Lee |
| **Repo** | unified-modal-embedder |
| **Related** | `weftspun/multimodal-semantic-ids` (`mesh-only-scope-postpone-qwen3vl-embedder`, `multimodal-residual-fsq-semantic-ids`), `weftspun/trellis2-mesh-vae-slang`, `weftspun/residual-fsq-recommender` |

## Summary

A "Custom SetFit" recipe was originally proposed for few-shot classification
of 3D voxel/mesh assets: run TRELLIS.2's VAE encoder, generate anchor-positive/
anchor-negative pairs from a handful of labels, contrastively fine-tune the
encoder (or a projection on top of it) with a Siamese setup, then fit a
lightweight `LogisticRegression` head on the resulting embeddings.

That recipe fine-tunes the encoder itself, which this repo's architecture
already forecloses: encoders are frozen, and this repo's only job is fusion +
Matryoshka truncation into the 768-d space `residual-fsq-recommender` consumes
for both semantic-ID tokenization and per-token aux (README.md). Fine-tuning
the TRELLIS.2 SLAT encoder for a classification task would fork the mesh
embedding space away from that shared space, breaking the "ID and aux
describe an item in the same space" invariant for every other consumer of the
recommender.

A literature review (deep-research pass, 2026-07-21 — see Research findings
below) was run to pick the best-performing strategy among methods that keep
the encoder frozen. The result changes the recommendation from what an
earlier draft of this RFD called "SetFit-lite" (a small MLP head trained with
a contrastive loss) to a **cache/prototype-based adapter** in the
Tip-Adapter/Proto-Adapter family, which is both the strongest published
performer under a frozen-backbone constraint and cheaper to train than a
gradient-optimized contrastive head.

## Background

`unified-modal-embedder`'s stated scope (README.md):

- Each modality slot (image/text, mesh, audio-deferred, phenotype-deferred)
  reduces to one vector via a **frozen** encoder.
- Mesh uses TRELLIS.2 SLAT → `trellis2-mesh-vae-slang`, explicitly
  "geometry-faithful; not mean-pooled" — i.e. the reduction from SLAT's
  sparse per-active-voxel tokens to a single vector already uses a
  non-trivial pooling strategy chosen to preserve geometry-critical
  structure, not naive averaging.
- This repo's job is fusion + truncation only. Encoder training happens
  upstream, if at all, and is explicitly out of scope here
  ("not encoder training").

Separately, there's a recurring need for **few-shot labeling/classification**
of mesh assets (e.g. style tags, category tags, moderation flags) where only
a handful of labeled examples exist per class — the motivating case for
reaching for SetFit in the first place.

## Research findings

Ran a deep-research pass on: *given a frozen encoder, which downstream-head
strategy gets the highest few-shot classification accuracy* — comparing
plain linear probing, CLIP-Adapter, Tip-Adapter/Tip-Adapter-F,
prototypical-network-style embedding adaptation (FEAT), and SetFit-style
contrastive head training. 106 sub-agents, 23 primary sources fetched, 25
claims adversarially verified (18 confirmed / 7 refuted).

**Headline result:** among strictly frozen-backbone methods on the standard
CLIP + ImageNet few-shot benchmark, **Proto-Adapter-F** reports the highest
accuracy at 2/4/8/16-shot (66.17% at 16-shot), with **Tip-Adapter-F** a close
second (65.51%) and narrowly ahead at 1-shot. Both clearly beat CLIP-Adapter
(63.59%), CoOp (62.95%), and plain linear probing (56.13%).
[MDPI/Sensors 2024](https://www.mdpi.com/1424-8220/24/11/3624),
[Tip-Adapter, ECCV 2022](https://arxiv.org/pdf/2207.09519).

**Why this beats a contrastive MLP head:** Tip-Adapter/Proto-Adapter don't
backprop a projection at all — they build a key-value cache (or class
prototypes) directly from the few-shot embeddings and use similarity-based
retrieval as the classifier, with an optional lightweight fine-tuning pass
(`-F` variants) on top. Tip-Adapter-F reaches its accuracy in ~20 epochs vs.
~200 for CLIP-Adapter, and training-free Tip-Adapter alone is already
competitive with gradient-trained adapters — cheaper and less failure-prone
than fitting a contrastive projection from scratch on a handful of examples
per class.
[Tip-Adapter](https://arxiv.org/pdf/2207.09519).

**SetFit is disqualified as a comparator, confirming the original concern:**
SetFit's own reported gains (62.3% @ 8-shot text classification, on par with
T-Few 3B) come from contrastively fine-tuning the encoder itself before
training the head — not from a frozen-backbone downstream strategy. It isn't
a valid frozen-encoder baseline at all, which is exactly why the original
proposal's stage 2 was incompatible with this repo's architecture.
[SetFit paper](https://arxiv.org/abs/2209.11055).

**Plain linear probing is weakest in extreme low-shot regimes** (1–2 shot),
by large margins (e.g. CLIP-Adapter beats it by 53.6pp on OxfordPets
1-shot), though an improved 2024 formulation (LP++) closes much of that gap
at far lower compute cost — worth keeping as a cheap fallback baseline.
[CLIP-Adapter](https://arxiv.org/pdf/2110.04544),
[LP++, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_LP_A_Surprisingly_Strong_Linear_Probe_for_Few-Shot_CLIP_CVPR_2024_paper.pdf).

**Direct precedent for the 3D case:** Tip-Adapter-style caching has already
been applied to *frozen* 3D point-cloud embeddings — PointCLIP and
PointCLIP V2 use exactly this mechanism (multi-view rendering into a frozen
CLIP-style encoder, then a cache/inter-view adapter) for few-shot 3D
classification.
[PointCLIP](https://arxiv.org/pdf/2112.02413),
[PointCLIP V2](https://www.researchgate.net/publication/365633539_PointCLIP_V2_Adapting_CLIP_for_Powerful_3D_Open-world_Learning).
No evidence surfaced of this specifically on voxel/SLAT-style latents like
TRELLIS.2's, but the point-cloud precedent is close enough in kind
(frozen 3D shape encoder + cache-based few-shot adapter) to de-risk trying
it here.

**Caveats:** all the strongest numbers come from one benchmark family
(CLIP + ImageNet/OxfordPets); results may not transfer cleanly to a
different frozen backbone (TRELLIS.2 SLAT) or to mesh/voxel data. Several
specific numeric claims failed adversarial verification and were dropped
(see refuted claims in the research transcript) — treat exact percentages
as directional, not authoritative for this repo's actual data.

## Proposal

Build the few-shot classifier as a cache/prototype adapter sitting entirely
downstream of this repo's existing frozen fusion output — never touching
TRELLIS.2 or the fused 768-d vector that `residual-fsq-recommender` depends
on.

1. **Data prep** (unchanged from the original plan) — run the *frozen*
   TRELLIS.2 SLAT encoder + this repo's existing fusion/truncation path to
   get the standard 768-d embedding per asset.

2. **Cache construction (training-free baseline)** — build a key-value
   cache directly from the few-shot labeled 768-d embeddings (Tip-Adapter
   style) or per-class prototypes (Proto-Adapter style: mean + normalize
   embeddings per class). Classify a new asset by similarity-weighted
   retrieval against the cache. No backprop, no fine-tuning — try this
   first as the baseline, per the research above.

3. **Optional fine-tuning pass (`-F` variant)** — if the training-free
   cache underperforms, lightly fine-tune the cache weights (not the 768-d
   embedding, not TRELLIS.2) for a small number of epochs, matching
   Tip-Adapter-F/Proto-Adapter-F. This is still strictly downstream of the
   frozen fusion output.

4. **Inference** — a small wrapper class: raw asset → existing frozen
   fusion pipeline (unchanged, reused as-is) → cache/prototype lookup →
   class probabilities. This wrapper is a *consumer* of
   `unified-modal-embedder`'s output, structurally identical to how
   `residual-fsq-recommender` consumes it — it should not live inside this
   repo.

| Component | Text SetFit (standard) | Original 3D proposal | This RFD (cache/prototype adapter) |
|---|---|---|---|
| Base representation | `sentence-transformers` | TRELLIS.2 VAE encoder | This repo's frozen 768-d fused embedding |
| What gets fine-tuned | Full encoder | Encoder (or its projection) | Nothing (training-free) or a small cache (optional `-F` pass) |
| Data format | Tokenized text | 3D tensors / voxel grids | Precomputed 768-d vectors (already produced today) |
| Mechanism | Contrastive Siamese + `CosineSimilarityLoss` | `nn.CosineEmbeddingLoss` on pairs | Key-value cache / class-prototype retrieval |
| Classifier | Integrated `LogisticRegression` | scikit-learn `LogisticRegression` | Similarity-weighted cache lookup (+ optional fine-tuned residual) |
| Lives in | — | — | New downstream repo/tool, not `unified-modal-embedder` |

## Alternatives considered

- **Fine-tune TRELLIS.2 directly (original proposal).** Rejected: breaks
  the frozen-encoder invariant and forks the mesh embedding space away from
  what the recommender consumes. Would require re-running fusion/truncation
  and likely re-deriving semantic IDs for every asset, for the sake of one
  downstream classifier.
- **Contrastive MLP head on frozen embeddings ("SetFit-lite," earlier draft
  of this RFD).** Superseded by the research findings above: a
  gradient-trained contrastive projection is both more expensive to train
  (needs ~10x more epochs than Tip-Adapter-F) and not shown to outperform
  the cache/prototype approach on the closest available benchmark. Not
  ruled out entirely — worth a quick comparison once real labeled data
  exists — but no longer the default recommendation.
- **Fine-tune only a projection *inside* the mesh slot, before fusion.**
  Rejected — it still mutates the per-modality vector that feeds the
  shared 768-d space, so every other consumer of that space is affected by
  a change made for one classification task.
- **Plain linear probing (logreg/SVM directly on the frozen 768-d
  vector).** Cheapest option and the literature's own weakest baseline in
  low-shot regimes, but simple enough to be worth running once as a floor
  to compare the adapter against.

## Drawbacks

- The adapter's ceiling is bounded by whatever the frozen 768-d vector
  already encodes; if a classification task needs geometric detail that
  TRELLIS.2's "not mean-pooled" pooling discarded, no downstream adapter
  recovers it, regardless of which strategy (cache, prototype, or
  contrastive head) is used.
- All performance evidence above comes from CLIP-on-images/point-clouds
  benchmarks, not from TRELLIS.2 SLAT latents on mesh/voxel data — this is
  an extrapolation, not a proven result for this specific encoder. Should
  be validated empirically once real few-shot labeled mesh data exists.

## Unresolved questions

- Where does the adapter wrapper actually live? Own repo (mirroring how
  `residual-fsq-recommender` is separate from this repo), or a
  `tools/`-style subfolder here? Leaning toward a separate repo given this
  repo's stated scope is fusion + truncation only.
- Is there an actual near-term labeling task motivating this (asset tags,
  moderation, style categories), or is this speculative? Affects whether
  it's worth building now vs. deferring like audio/phenotype slots.
- Once real labeled mesh data exists: run training-free cache, `-F`
  fine-tuned cache, plain linear probe, and the contrastive-MLP-head
  alternative side by side on the same held-out set before committing —
  the literature comparison above is evidence for *where to start*, not a
  substitute for measuring on TRELLIS.2 embeddings directly.
