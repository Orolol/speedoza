# Survey vLLM / llama.cpp — réponses existantes à nos puits du cycle verify (2026-06-12)

Lecture croisée de vLLM (`~/workspace/vllm`, source local) et llama.cpp
(`~/workspace/llama.cpp`, clone shallow du 2026-06-12), ciblée sur la carte
des sinks @60 ms (DAILY 2026-06-12). Pointeurs fichier:ligne vérifiés dans
ces checkouts.

## 1. Bucket chunk GEMM (20 ms, ~36% BW à N=5) — DEUX références complètes

### Marlin (vLLM) — le blueprint du kernel v3

Le kernel de référence "W4/W8 batch ≤ 16 à pleine BW poids". Différences
structurelles vs notre m16n8k64 (1 tuile m16/CTA, split-K intra-CTA 16
warps, double-buffer 2 étages) :

- **Pipeline cp.async 4 étages** (`pipe_stages=4`, marlin.cuh:31) au lieu
  de notre double-buffer — chaque étage est une vraie tuile (16 KB-class),
  pas 512 B/warp.
- **K strié SUR LES CTAs** (striped partitioning, marlin_template.h:271-281)
  + **réduction globale 2 phases** via locks/atomics en L2
  (marlin_template.h:457-476, 1467-1549) — au lieu de notre split-K
  intra-CTA. Donne de l'occupation même aux shapes à M faible (nos
  down/o/q à 320-384 tuiles).
- **Activations (la matrice étroite) résidentes** : stagées une fois,
  double-buffer en REGISTRES dans le K-loop (marlin_template.h:768).
- Tuiles small-batch : threads=256, thread_k=128, thread_n=128,
  M-tile 16 (marlin.cu:128-142). Layout SMEM XOR anti-bank-conflict
  (marlin_template.h:720-737).

### llama.cpp — MMVQ valide notre approche, MMQ-FP4 Blackwell existe

- `MMVQ_MAX_BATCH_SIZE = 8` (mmvq.cuh:3) : leur GEMV quantifié multi-N va
  jusqu'à N=8, **templaté par N à la compile** (mmvq.cu:475, dispatch
  889-968) — même conclusion d'architecture que notre chunk kernel.
  Accumulateurs `tmp[N][rows]` en REGISTRES (mmvq.cu:566), K-loop externe,
  activations jamais relues (pointeur base en registre, mmvq.cu:569-583),
  1-2 lignes/CTA, 2-4 warps.
- **MMQ a un chemin FP4 Blackwell** : `MMQ_ITER_K_FP4 = 512` (mmq.cuh:14 —
  512 K par itération vs nos 64 !), `MMQ_MMA_TILE_X_K_FP4 = 72` (mmq.cuh:220),
  padding K%8==4 anti-bank-conflict. Une implémentation FP4 Blackwell
  open-source lisible de bout en bout.
- Bascules : MMVQ si N≤8, MMQ au-dessus, cuBLAS en dernier recours
  (ggml-cuda.cu:2579-2669).

### vLLM NVFP4 cutlass SM120 (notre classe de GPU)

`nvfp4_scaled_mm_sm120_kernels.cu:51-61` : M≤256 → TileShape 128×128×128,
cluster (1,1,1). C'est la même classe de config que ce que cuBLASLt nous
donne — cohérent avec "cuBLASLt n'est pas inefficace, la shape l'est".
Pas de Marlin-NVFP4 ; Machete = SM90 wgmma/TMA, hors SM120a.

**Plan v3 chunk kernel** (dé-risqué par ces deux références) : garder
l'atome m16n8k64 block-scale, adopter (a) pipeline 4 étages avec étages
plus profonds (256-512 K/warp/étage), (b) K strié inter-CTA + réduction
globale 2 phases pour les shapes à M faible, (c) B en registres sur tout
le K-loop, (d) layout XOR. Cible : sortir des ~36% vers 70-85% BW
(Marlin-class) = −6 à −10 ms/cycle.

## 2. DeltaNet verify 5 tokens (6 ms) — vLLM passe par le RÉCURRENT, pas le chunked

- vLLM : le chunked FLA (chunk_size **64**, fla/ops/utils.py:31) ne sert
  QUE le prefill ; le decode **y compris les blocs spec-decode** passe par
  `fused_recurrent_gated_delta_rule_packed_decode` (fla/ops/fused_recurrent.py:27-150)
  avec un flag `IS_SPEC_DECODING` — **un seul kernel** qui boucle les k+1
  tokens en interne. Dispatch : v1/attention/backends/gdn_attn.py:316-334.
- Notre essai `QWEN36_DELTANET_SEQ_SHORT_CHUNK` (NEG) lançait la
  SÉQUENCE de kernels par token — le différenciateur vLLM est la **fusion
  des 5 tokens dans un seul lancement récurrent** (pas de round-trip
  d'état, pas de 5× launch). Lever re-ouvert avec ce design : un kernel
  `deltanet_recurrent_block(tokens≤8)` — cible −3 à −4 ms sur les 6.

## 3. Frontières de cycle (1.9 ms) — vLLM décide sur device

- Rejection/acceptance = kernel Triton device-side
  (v1/worker/gpu/spec_decode/rejection_sampler_utils.py:156-300) : la
  comparaison target-vs-draft n'implique AUCUN sync host dans le step.
- Le readback des tokens est une copie async sur stream séparé,
  **recouverte par le forward suivant** (gpu_model_runner.py:240-318) ;
  le host ne parse qu'après coup.
- Le draft MTP (k tokens) est UN forward parallèle masqué (1+K positions,
  dmtp_linear.py:358-424) — pas une récursion séquentielle comme la
  nôtre. NB : c'est un compromis acceptance/coût différent (leur tête
  full-attention multi-position vs notre chaîne récurrente) — pas
  transposable tel quel, mais l'acceptance device-side + readback
  recouvert l'est.

## 4. lm_head — personne ne fait mieux que notre deux-étages

vLLM : logits pleins + argmax torch. llama.cpp : tête quantifiée sans
garde d'exactitude. Notre FP8-scan + top-8 rescore + garde est en avance ;
le levier FP4-scan (réouvert par l'architecture top-8) reste à nous.

## Ordre d'attaque révisé (cycle @128 ≈ 60.4 ms avec marge v2)

1. Kernel deltanet récurrent fusionné ≤8 tokens (−3-4 ms, design vLLM,
   effort moyen).
2. Sonde FP4 lm_head + scan FP4 si elle passe (−3.5 ms, petit/moyen).
3. Acceptance device + readback recouvert (−1-1.5 ms, engine).
4. Chunk GEMM v3 Marlin-class (−6-10 ms, le gros chantier, maintenant
   avec deux implémentations de référence à cribler).
