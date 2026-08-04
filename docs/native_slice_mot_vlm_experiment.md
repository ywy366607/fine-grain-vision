# Native Slice-MoT VLM: First Realistic Integration Experiment

Status: core architecture implemented on `feat/native-slice-mot-vlm`; training
and efficacy evaluation have not started.

This is the first experiment in this repository that moves beyond synthetic
classification and reconstruction into a jointly trained vision-language model.
It asks a narrow question:

> Can a persistent, non-patchified visual field communicate through transient
> Slice states with a language stream, when both modalities and their interface
> are trained together from random initialization on a realistic amount of
> public multimodal data?

The experiment is deliberately small enough for a single 16 GB GPU, but large
enough that a failed 16-sample micro-overfit is not treated as architectural
evidence. The primary run must consume at least as many supervised target tokens
as the model has parameters before an efficacy conclusion is allowed.

## 1. What This Experiment Changes

Previous VLM probes in this repository attach a new visual frontend to an
already trained or frozen language model. Those experiments are useful interface
tests, but they confound visual representation quality with compatibility with a
pre-existing language residual stream.

This experiment removes that confound:

- the visual stream is randomly initialized;
- the language stream is randomly initialized;
- the cross-modal attention interface is randomly initialized;
- all components train jointly from the first step;
- no frozen model, adapter, LoRA, teacher feature, or hidden API is used.

It is not a production model or an OmniDocBench submission. It is a mechanism
test at a scale where language modeling, visual recognition, and cross-modal
binding must all emerge from data.

## 2. Architecture

The model is called **SliceMoT-VLM-6L**.

It maintains two persistent streams at every depth:

```text
visual point field: X_l in R^(N x C_p)
language states:    H_l in R^(T x C_m)
```

The full-resolution point field is never replaced by a permanent visual token
sequence. At each layer, transient Slice states are read from the point field,
communicate with language through a Mixture-of-Transformers (MoT) block, and are
written back to every point:

```text
X_l -- local visual update -- SliceRead -- visual MoT expert -- Deslice -- X_(l+1)
                                      <-> global attention <->
H_l ------------------------------- language MoT expert ---------------- H_(l+1)
```

There is no serial `vision encoder -> projector -> LLM` boundary. Vision and
language advance together through all six layers.

### 2.1 Point field

Each input pixel owns one state vector:

```text
X_0 = Linear([R, G, B, y, x]) + DWConv3x3(Linear([R, G, B, y, x]))
```

The point width is `C_p = 256`. No patchify operation or spatial downsampling is
used. Large images are processed by tiled scans; image size changes compute
linearly rather than changing the architecture.

Before each Slice read, the point field receives one residual local update:

```text
X_local = X_l + Pointwise(DWConv3x3(RMSNorm(X_l)))
```

This local path is required to represent strokes and neighborhoods before global
pooling. It is not an auxiliary image encoder.

### 2.2 Transient Slice read

Every layer owns `M = 256` learned Slice queries and its own complete read/write
module. Assignment projections, value/back projections, local convolutions, and
point FFNs are not shared across depth. Layer-specific parameters remove the
requirement that one routing function serve incompatible feature depths.

```text
logits[n, m] = <W_point norm(X_local[n]), q_l[m]> / sqrt(C_p)
A[n, m] = softmax_m(logits[n, m])
mass[m] = sum_n A[n, m]
S_l[m] = sum_n A[n, m] W_value X_local[n] / max(mass[m], eps)
```

`S_l` has width `C_m = 512`. Assignment is data-dependent, but query identity is
stable within a layer.

The registered primary model uses:

- fixed softmax temperature `1.0`;
- no Gumbel sampling;
- no top-k routing;
- no adaptive temperature;
- no Stiefel projection;
- no learned residual gate.

These mechanisms are excluded from the primary test. They may be studied only
after the primary causal gate passes.

The `N x M` assignment is evaluated in tiles. The implementation must make two
streaming passes, one for mass-normalized read and one for Deslice, without
materializing the full assignment matrix.

The implemented scan aggregates in point width before the `C_p -> C_m`
projection and applies the reverse projection before Deslice. This preserves the
Transolver-3 matrix-order optimization: only `M` Slice states, rather than `N`
point states, cross the wider projection. CUDA FP16/BF16 configurations with
`C_p <= 128` may use the verified fused Triton assignment kernel. The registered
`C_p = 256` configuration uses PyTorch GEMM/softmax because it is faster on the
reference RTX 5060 Ti; Triton is not selected merely for implementation novelty.

Training checkpoints each complete per-layer visual read, MoT exchange, and
write-back unit. Diagnostics stay as detached device tensors so the hot path
does not introduce per-layer CPU/GPU synchronization.

### 2.3 Modality-specific MoT block

The experiment follows Mixture-of-Transformers rather than a shared Transformer.
Every non-embedding Transformer parameter is modality-specific:

```text
visual expert: Q_v, K_v, V_v, O_v, Norm_v, FFN_v
text expert:   Q_t, K_t, V_t, O_t, Norm_t, FFN_t
```

For modality `m`:

```text
Q_m = Wq_m RMSNorm_attn_m(Z_m)
K_m = Wk_m RMSNorm_attn_m(Z_m)
V_m = Wv_m RMSNorm_attn_m(Z_m)
```

Keys and values are restored to serialized multimodal order:

```text
K = concat_in_sequence_order(K_text, K_visual)
V = concat_in_sequence_order(V_text, V_visual)
```

Both modalities attend globally, but use their own queries and output matrices:

```text
A_m = softmax(Q_m K^T / sqrt(d_head) + causal_mask) V
Z_bar_m = Z_m + Wo_m A_m
Z_next_m = Z_bar_m + FFN_m(RMSNorm_ffn_m(Z_bar_m))
```

Routing is deterministic from token modality. There is no learned expert router
and no load-balancing loss. The shared object is the attention communication
graph, not QKV, output, normalization, or FFN parameters.

### 2.4 Deslice and persistent visual state

The updated visual states from the MoT block are projected back to point width
and scattered with the same assignment:

```text
message[n] = sum_m A[n, m] W_back S_next[m]
X_(l+1) = X_local + message + PointMLP(RMSNorm(X_local + message))
X_(l+1) = X_(l+1) + LocalConv_l(RMSNorm(X_(l+1)))
```

The transient `S_next` is then discarded. Only the complete point field and
language states persist to the next layer.

## 3. Autoregressive Sequence and Teacher Forcing

The primary experiment uses an ordinary serialized causal sequence:

```text
[BOS, task or document prefix, visual Slice positions, target text]
```

Examples:

```text
[BOS, <ocr>, <image_start>, S_1 ... S_M, <image_end>, transcription]
[BOS, <document>, <image_start>, S_1 ... S_M, <image_end>, structured text]
[BOS, plain text continuation]
```

The whole sequence uses one causal mask. No special answer-to-vision mask is
introduced.

During training, ground-truth target tokens are shifted right in the standard
teacher-forcing formulation:

```text
model input at target position t: prefix + visual states + target[:t]
prediction:                       target[t]
```

Future target tokens are inaccessible. Labels for prompt and visual positions
are `-100`; only continuation/transcription positions contribute to language
loss. This matches the causal training semantics of native autoregressive VLMs.

Because task/prefix tokens precede the visual positions, visual queries may read
the task context. Visual states cannot read the later target. Target states may
read all earlier visual positions and earlier target tokens.

The first pretraining phase does not use chat roles or instruction-response
templates. It trains text continuation, image transcription, and document
continuation. Chat-style supervised fine-tuning is outside the primary gate.

## 4. Registered Model Size

```yaml
name: SliceMoT-VLM-6L
layers: 6
point_width: 256
mot_width: 512
visual_slices: 256
attention_heads: 8
visual_ffn_width: 1536
text_ffn_width: 1536
tokenizer: UTF-8 byte
vocabulary_size: approximately 260
position_encoding:
  text: 1D RoPE
  visual: Slice centroid 2D RoPE
estimated_total_parameters: 47.84M
```

The text embedding and output head are tied. Visual and text experts are not
parameter-tied. Layers are not recurrently tied.

Six layers are sufficient for the mechanism test: they provide six complete
visual read, cross-modal communication, and write-back rounds without turning
the experiment into a depth-scaling study.

## 5. Public Pretraining Data

The minimum run contains **64 million loss-bearing UTF-8 byte tokens**. This is
approximately `1.4x` the estimated total parameter count.

Only positions that contribute to cross-entropy count toward this threshold.
Image positions, masked prompts, padding, repeated epochs, and evaluation data do
not count as extra supervised tokens.

| Source | Role | Registered target-token budget |
|---|---|---:|
| FineWeb-Edu and FineWeb2 | English and multilingual text continuation | 24M |
| RenderedText | line and character OCR | 16M |
| NVIDIA OCR-Synthetic-Multilingual-v1 | multilingual line/page OCR | 16M |
| DoclingMatix | structured document continuation | 6M |
| olmOCR-mix-1025 | natural documents, equations, HTML tables | 2M |
| **Total** | | **64M** |

Sources:

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- [RenderedText](https://huggingface.co/datasets/wendlerc/RenderedText)
- [NVIDIA OCR-Synthetic-Multilingual-v1](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1)
- [DoclingMatix](https://huggingface.co/datasets/HuggingFaceM4/DoclingMatix)
- [olmOCR-mix-1025](https://huggingface.co/datasets/allenai/olmOCR-mix-1025)

Data must be streamed or downloaded as bounded shards. The experiment must not
silently replace unavailable sources with generated samples. Every retained
example records source, split, sample ID, image dimensions, target byte count,
and license lineage.

Validation is split by source identity and, where available, document, font, and
render family. OmniDocBench and its derived pages are excluded from training.

## 6. Training Recipe

All parameters train jointly from random initialization from step one.

```yaml
optimizer: AdamW
betas: [0.9, 0.95]
weight_decay: 0.1
peak_learning_rate: 0.0003
schedule: 2% warmup, cosine decay
precision: BF16
gradient_clip_norm: 1.0
effective_batch: at least 32768 loss-bearing tokens
sequence_length: 512 initially, 1024 for the final document portion
activation_checkpointing: true
assignment_scan: tiled, checkpointed
```

The data mixture may be shuffled, but its registered token totals may not be
changed after inspecting validation outcomes. Resolution and sequence length may
increase according to a fixed schedule for memory efficiency; model parameters,
losses, and modality routes remain unchanged.

The primary run uses only next-byte cross-entropy. It excludes:

- contrastive or CLIP loss;
- image reconstruction loss;
- teacher-feature or optimal-transport alignment;
- KL regularization;
- LoRA or frozen components;
- GRPO/RL;
- sample retries or post-hoc routing;
- benchmark-specific format repair.

## 7. Controls

### 7.1 Required causal controls

Every validation checkpoint records:

- matched image and target NLL;
- cyclically shuffled image NLL;
- blank image NLL;
- first-target-byte NLL and full-target NLL;
- matched, shuffled, and blank greedy generations;
- gradient norm from target logits to RGB and Slice assignment parameters;
- per-layer assignment entropy, effective Slice mass, and visual-state change.

Training CE alone is never evidence that the image is used.

### 7.2 Patch-MoT baseline

A parameter-matched Patch-MoT control is required for an architectural claim.
It uses the same:

- six MoT layers;
- modality experts;
- tokenizer and target sequences;
- data order and 64M-token budget;
- optimizer, schedule, width, and number of visual communication states.

Only `SliceRead + Deslice + persistent point field` is replaced with learned
non-overlapping patch tokens. This control distinguishes failure of native
multimodal training from failure specific to the Slice visual mechanism.

The Slice run is executed first. The Patch control is mandatory before claiming
that Slice is better or worse, but it need not block an early finding that the
Slice model has or has not established visual causality.

## 8. Preregistered Gates

Intermediate smoke tests may catch implementation errors, but no architecture
conclusion is allowed before 64M supervised tokens.

### Gate A: native multimodal learning

At 64M target tokens, all conditions must hold on held-out OCR data:

1. matched character accuracy is at least 90%;
2. matched character accuracy exceeds shuffled by at least 50 percentage points;
3. at least 90% of samples have lower matched target NLL than shuffled target NLL;
4. median `NLL(shuffled) - NLL(matched)` is at least 0.5 nat per target token;
5. RGB and assignment gradient norms are finite and nonzero;
6. text-only validation loss does not diverge.

Passing Gate A establishes only that a native Slice-MoT VLM can learn a causal
visual-language channel. It does not establish fine-detail superiority.

### Gate B: fine-detail value

After Gate A passes, compare Slice-MoT with Patch-MoT on held-out thin-detail and
patch-collision examples at matched compute and token budget.

The Slice claim requires a positive paired confidence interval for character
accuracy or target NLL on the collision subset, while ordinary OCR and text
continuation do not materially regress.

### Gate C: real document transfer

Only after Gates A and B may the model receive chat-style OCR/document SFT and be
evaluated on OmniDocBench. The benchmark adapter must be identical for Slice and
Patch models and must not repair model outputs differently by architecture.

## 9. Decision Rules

- **Gate A fails, Patch also fails:** the training/data/model scale is
  insufficient; no Slice conclusion follows.
- **Gate A fails, Patch passes:** reject the registered Slice read/write package
  at this scale. Diagnose acquisition and Deslice before changing temperature or
  adding losses.
- **Gate A passes, Gate B fails:** persistent Slice is trainable but has not shown
  conditional fine-detail value over Patch.
- **Gates A and B pass:** proceed to document SFT and OmniDocBench.
- **Only training CE improves:** stop; this is compatible with a language-only
  shortcut.

No best-of-many hyperparameter selection is permitted. A changed temperature,
Slice count, gate, regularizer, or loss starts a new registered experiment.

## 10. References

- Liang et al., [Mixture-of-Transformers: A Sparse and Scalable Architecture for
  Multi-Modal Foundation Models](https://arxiv.org/abs/2411.04996), TMLR 2025.
- [Official Mixture-of-Transformers implementation](https://github.com/facebookresearch/Mixture-of-Transformers).
- Qwen Team, [Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5), 2026.
- [Hugging Face Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
- Wu et al., [Transolver++](https://github.com/thuml/Transolver_plus), ICML 2025.
