# LEADERBOARD EXPERIMENTS
## TeamIOTA - Pranav Birari, Vedant Patil

---

## Experiment 1 — Head-Only Training (Frozen Backbone) vs End-to-End Backbone Fine-Tuning

**WandB Run:** [exp8-megadesc-arcface-emb512-hidden1024-margin06-batch48](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/n125ot5y)  
**Notebook:** [Experiment_1.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_1/Experiment_1.ipynb)  
**Kaggle Submission ID:** 50743503  
**Public Leaderboard Score:** 0.755

---

### Research Question

Does training only the ArcFace projection head on frozen MegaDescriptor-L-384 features
achieve competitive identity-balanced mAP compared to the baseline's end-to-end backbone
fine-tuning (last 8 transformer blocks unfrozen)?

---

### Hypothesis

Freezing the MegaDescriptor backbone and training only the projection head will produce
lower or comparable mAP relative to end-to-end fine-tuning, because the frozen backbone
cannot adapt its feature representations to the jaguar domain. However, the approach may
still generalise better given the small dataset size (1895 images, 31 identities), where
full fine-tuning risks overfitting.

---

### Defined Intervention

| Component | Baseline | This Experiment (Experiment 1) |
|---|---|---|
| Backbone | MegaDescriptor-L-384, last 8 blocks unfrozen | MegaDescriptor-L-384, fully frozen ← primary change |
| Embedding head | EmbeddingProjection(1536 → 768 → 512) | EmbeddingProjection(1536 → 1024 → 512) |
| ArcFace margin | 0.5 | 0.6 |
| ArcFace scale | 64.0 | 64.0 |
| Dropout | 0.15 | 0.3 |
| Optimizer | AdamW, differential LR (backbone=1e-5, head=5e-5) | AdamW, single LR=1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-7) | ReduceLROnPlateau |
| Batch size | 24 | 48 |
| Augmentation | RandomHFlip, ColorJitter, RandomAffine | None |
| Epochs | 50 | 50 |
| Patience | 12 | 10 |
| Data split | 80/20 stratified, seed=42 | 80/20 stratified, seed=42 |
| Evaluation | Identity-balanced mAP (cosine similarity) | Identity-balanced mAP (cosine similarity) |

---

### Evaluation Protocol

Metric: Identity-balanced Mean Average Precision (mAP). Fixed 20% stratified validation
split (seed=42) with all 31 identities in both sets. Cosine similarity on L2-normalised
embeddings. Per-epoch tracking of train loss, val loss, val acc, val mAP, and lr logged
to WandB. Comparison point: provided baseline notebook.

---

### Results

| Configuration | Val mAP | Best Epoch | Batch Size | Augmentation | Backbone |
|---|---|---|---|---|---|
| Baseline notebook (end-to-end, 8 blocks unfrozen) | 0.7720 | 49 | 24 | Yes | Unfrozen (8 blocks) |
| Experiment 1 (head-only, frozen backbone) | 0.7927 | 45 | 48 | None | Fully frozen |

**WandB logged:** num_parameters, identity distributions, per-epoch train_loss / val_loss /
val_acc / val_mAP / lr, baseline MDS visualisation, fine-tuned MDS visualisation, model
artifact, submission artifact.

---

### Interpretation

Experiment 1 achieves a validation mAP of 0.793 using a fully frozen MegaDescriptor
backbone, training only the projection head with ArcFace loss. This shows that
MegaDescriptor's pre-trained representations already contain strong jaguar identity cues
that can be re-organised via a lightweight head without any backbone gradient updates.

A few things complicate interpreting the result cleanly. Several hyperparameters changed
alongside backbone freezing: a larger hidden dimension (1024 vs 768), higher ArcFace
margin (0.6 vs 0.5), higher dropout (0.3 vs 0.15), a different scheduler
(ReduceLROnPlateau vs CosineAnnealingLR), a larger batch (48 vs 24), and no data
augmentation. So any mAP difference relative to the baseline cannot be cleanly attributed
to backbone freezing alone.

That said, freezing the backbone is plausible here for a more fundamental reason.
MegaDescriptor-L-384 is pre-trained specifically on wildlife re-identification data, making
its feature space domain-relevant out of the box. With only 1895 training images across 31
identities, full backbone fine-tuning carries a real risk of overfitting or catastrophic
forgetting, especially for identities with very few samples (min: 13 images). Freezing the
backbone acts as a strong regulariser that constrains the model to use already-learned
visual features.

---

## Experiment 2 — Backbone Comparison: MobileNetV3-Large vs MegaDescriptor-L-384

**WandB Run:** [mobilenetv3-arcface-backbone-comparison](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/4t8ha3uk)  
**Notebook:** [experiment_2.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_2/experiment_2.ipynb)  
**Kaggle Submission ID:** 50744938  
**Public Leaderboard Score:** 0.719

---

### Research Question

Does replacing MegaDescriptor-L-384 (307M parameters, wildlife domain-pretrained,
1536-dim features) with MobileNetV3-Large (4.2M parameters, ImageNet-pretrained,
1280-dim features) achieve competitive identity-balanced mAP under an identical
ArcFace head and training protocol?

---

### Hypothesis

MobileNetV3-Large will underperform MegaDescriptor-L-384 for two reasons. First, domain
gap: MobileNetV3 is pretrained on ImageNet (general objects), while MegaDescriptor is
specifically trained on wildlife re-identification datasets. Second, capacity: at 73x
fewer parameters, MobileNetV3 has limited representational capacity for fine-grained
identity discrimination.

If the mAP gap is small (< 0.05), it suggests ArcFace metric learning can partially
compensate for generic pretraining on this small dataset. If large (> 0.10), it quantifies
the practical value of wildlife-specific domain pretraining.

---

### Defined Intervention

| Component | Baseline (assignment default config) | This Experiment (Experiment 2) |
|---|---|---|
| Backbone | MegaDescriptor-L-384, 307M params, 1536-dim output, frozen | MobileNetV3-Large, 4.2M params, 1280-dim output, frozen (primary change) |
| Input size | 384×384 | 224×224 |
| Backbone pretraining | Wildlife re-ID (domain-specific) | ImageNet (general) |
| Embedding head | EmbeddingProjection(1536 → 512 → 256) | EmbeddingProjection(1280 → 512 → 256) |
| Embedding dim | 256 | 256 |
| Hidden dim | 512 | 512 |
| ArcFace margin | 0.5 | 0.5 |
| ArcFace scale | 64.0 | 64.0 |
| Dropout | 0.3 | 0.3 |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 | AdamW, lr=1e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) | ReduceLROnPlateau (factor=0.5, patience=5) |
| Batch size | 32 | 32 |
| Epochs / patience | 50 / 10 | 50 / 10 |
| Data split | 80/20 stratified, seed=42 | 80/20 stratified, seed=42 |
| Evaluation | Identity-balanced mAP (cosine similarity) | Identity-balanced mAP (cosine similarity) |
| Total trainable params | ~796K (head only) | 796,672 (head only) |

The input feature dimension differs (1536 vs 1280) only because it is dictated by the
backbone. The rest of the head architecture (hidden dim 512, embedding dim 256) is
identical.

---

### Evaluation Protocol

Metric: Identity-balanced Mean Average Precision (mAP). Fixed 20% stratified validation
split (seed=42) with all 31 identities in both sets. Cosine similarity on L2-normalised
256-dim embeddings. Per-epoch tracking of train loss, val loss, val acc, val mAP, and lr
logged to WandB. Efficiency metrics: total trainable parameters, backbone parameter count.
Comparison point: assignment default config, MegaDescriptor-L-384 + ArcFace (val mAP = 0.741).

---

### Results

| Backbone | Val mAP | Best Epoch | Trainable Params | Backbone Params | Input | Pretraining |
|---|---|---|---|---|---|---|
| MegaDescriptor-L-384 (baseline notebook) | 0.7720 | 49 | ~84M trainable | 307M (8 blocks unfrozen) | 384×384 | Wildlife re-ID |
| MobileNetV3-Large (Exp 2) | 0.7795 | 50 | 796,672 | 4.2M (frozen) | 224×224 | ImageNet |

Delta vs baseline notebook: +0.0075 mAP. MobileNetV3-Large marginally exceeds the
MegaDescriptor baseline by 0.75 percentage points while using a backbone that is 73.1x
smaller (4.2M vs 307M parameters) and with ~105x fewer trainable parameters (796K vs ~84M).

WandB logs baseline_mAP: 0.741, which refers to the raw frozen MegaDescriptor (no
training). Against that reference, the delta is +0.0385.


---

### Interpretation

MobileNetV3-Large with ArcFace achieves 0.7795 val mAP, marginally outperforming the
MegaDescriptor-L-384 baseline notebook (0.7720, +0.0075 delta). The result is notable
not because of the mAP gap, which is small and may not be statistically significant with
a single seed, but because of the efficiency gain: MobileNetV3 achieves a comparable
result with a backbone that is 73.1x smaller, trains only the projection head (796K vs
~84M trainable parameters), and processes 224×224 instead of 384×384 inputs.

The gap is small partly because the baseline notebook benefits from end-to-end backbone
fine-tuning (last 8 blocks unfrozen), domain-specific pretraining, and data augmentation,
all of which compensate for its higher capacity. MobileNetV3 is frozen and trained
head-only with no augmentation, yet matches this performance. On a dataset this small
(1,516 training samples), ArcFace metric learning seems to be the dominant driver of
performance rather than backbone capacity or domain pretraining.

This comparison is not fully controlled: the baseline notebook uses backbone fine-tuning,
CosineAnnealingLR, differential LR, augmentation, and a larger batch size, while
Experiment 2 uses a frozen backbone, ReduceLROnPlateau, single LR, no augmentation, and
batch size 32. The +0.0075 mAP delta cannot be attributed to the backbone alone, and a
single seed limits statistical confidence.

---

## Experiment 3 — Loss Function Comparison

**WandB Run:** [loss-comparison-q3](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/f2q9hs8r)  
**Notebook:** [experiment_3.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_3/experiment-3-loss-comparison-v2.ipynb)  
**Kaggle Submission ID:** 50686692  
**Public Leaderboard Score:** 0.772

---

### Research Question

Which loss function, when paired with an identical frozen MegaDescriptor-L-384 backbone,
projection head, and training protocol, yields the highest identity-balanced mAP for
jaguar re-identification?

### Hypothesis

ArcFace, the baseline loss, is not necessarily the optimal objective for this dataset.
Angular margin losses that introduce sub-center structure (SubCenterArcFace) or that
down-weight easy examples (FocalLoss) may provide better gradient signal on a small,
class-imbalanced dataset (31 identities, 1,516 training samples).

---

### Controlled Variables (identical across all 4 runs)

| Parameter | Value |
|---|---|
| Backbone | MegaDescriptor-L-384 (frozen, pre-computed embeddings) |
| Projection head | Linear(1536 → 512 → 256), BatchNorm, ReLU, Dropout(0.3) |
| Embedding dim | 256 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Epochs / patience | 50 / 10 |
| Batch size | 32 |
| Val split | 0.2 (stratified, all 31 identities in both sets) |
| Seed | 42 |
| Evaluation metric | Identity-balanced mAP on fixed validation split |

### Intervention

Four projection heads were trained sequentially using pre-computed backbone embeddings.
Only the loss function and its associated hyperparameters changed between runs:

| Loss Function | Type | Key Hyperparameters |
|---|---|---|
| ArcFace | Angular margin | margin=0.5, scale=64 |
| CosFace | Cosine margin | margin=0.35, scale=64 |
| SubCenterArcFace | Sub-center angular margin | margin=0.5, scale=64, K=3 |
| FocalLoss | Weighted classification | gamma=2.0 |

---

### Results

| Loss Function | Best Val mAP | Best Epoch |
|---|---|---|
| **FocalLoss** | **0.8020** | **50** |
| ArcFace | 0.7970 | 44 |
| CosFace | 0.7810 | 43 |
| SubCenterArcFace | 0.6684 | 24 |
| Baseline notebook (MegaDescriptor + ArcFace, trained) | 0.7720 | 49 |

Best loss: FocalLoss — mAP 0.8020 (+0.030 vs trained baseline). The best model
(FocalLoss) was used for the competition submission. All runs, convergence curves,
bar chart, and identity distribution plots are logged to WandB run loss-comparison-q3.

---

### Interpretation

FocalLoss achieved the highest identity-balanced mAP (0.8020), outperforming ArcFace
(0.7970) and CosFace (0.7810) by small but consistent margins. SubCenterArcFace
significantly underperformed (0.6684), collapsing early at epoch 24.

The jaguar dataset has severe class imbalanc, identities range from 13 to 183 images.
ArcFace and CosFace apply uniform angular penalties, which can over-penalise
well-separated easy examples while giving insufficient gradient to hard, rare identities.
FocalLoss with Gamma=2.0 down-weights easy classifications and focuses training on
hard to distinguish identities, which directly addresses the long-tail problem here.

SubCenterArcFace (K=3 sub-centers per class) was designed for large datasets with noisy
labels and intra-class appearance variation. On a 31 identity dataset with only 1,516
training samples, 3 sub-centers per class likely caused over-parameterisation of the
class boundaries, destabilising training and causing early stopping at epoch 24.

The small gap between ArcFace and CosFace (+0.016 mAP ArcFace over CosFace) likely
reflects the fact that ArcFace's additive angular margin produces slightly tighter
hyperspherical clustering than CosFace's additive cosine margin at scale=64.

---

## Experiment 4 — Backbone Comparison for Jaguar Re-Identification

**W&B Run:** [backbone-comparison](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/c7km1stn)  
**Notebook:** [experiment_4.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_4/experiment-4.ipynb)  
**Kaggle Submission ID:** 50704896  
**Public Leaderboard Score:** 0.835

---

### Research Question

Which frozen backbone feature extractor, when paired with an identical ArcFace projection
head and training protocol, yields the highest identity-balanced mAP for jaguar
re-identification?

---

### Hypothesis

Domain-specialised or large-scale self-supervised backbones (e.g. MegaDescriptor, DINOv2)
will outperform general-purpose CNN backbones (e.g. ResNet-50) on jaguar re-ID, because
their pre-training encourages richer, more discriminative visual representations.

---

### Intervention

| Variable | Value |
|---|---|
| Changed | Backbone feature extractor (5 options below) |
| Fixed | Projection head Linear(dim→512→256) + BatchNorm + ReLU + Dropout(0.3) |
| Fixed | Loss: ArcFace (margin=0.5, scale=64) |
| Fixed | Optimiser: AdamW (lr=1e-4, weight_decay=1e-4) |
| Fixed | Scheduler: ReduceLROnPlateau (factor=0.5, patience=10) |
| Fixed | Epochs: 50, Batch size: 32, Seed: 42 |
| Fixed | Output embedding dim: 256 for all backbones |
| Fixed | Augmentations: RandomHorizontalFlip, ColorJitter, RandomRotation |
| Fixed | Validation split: 20% stratified (all 31 identities in both sets) |
| Backbone freeze | All backbones fully frozen — no gradient updates to backbone weights |

---

### Backbones Compared

| Backbone | Type | Pre-trained On | Embedding Dim |
|---|---|---|---|
| MegaDescriptor-L-384 | ViT-L / Swin | Wildlife Re-ID datasets | 1536 |
| DINOv2-ViT-L-14 | ViT-L | LVD-142M (self-supervised) | 1024 |
| EfficientNet-B4 | CNN | ImageNet | 1792 |
| ConvNeXt-V2-Base | CNN-Transformer | ImageNet-22K | 1024 |
| ResNet-50 | CNN | ImageNet | 2048 |

---

### Evaluation Protocol

Metric: Identity-balanced mAP on the fixed validation split. Best model selection:
checkpoint with lowest validation loss during training. Embedding extraction: frozen
backbone → projection head → L2-normalised 256-dim embeddings. Similarity: cosine
similarity matrix over validation embeddings.

---

### Results

| Backbone | Best Val mAP | Best Epoch |
|---|---|---|
| MegaDescriptor-L-384 | 0.7720 | 49 |
| DINOv2-ViT-L-14 | **0.8447** | 50 |
| EfficientNet-B4 | 0.8266 | 50 |
| ConvNeXt-V2-Base | 0.8273 | 49 |
| ResNet-50 | 0.7671 | 50 |

Best backbone: DINOv2-ViT-L-14 (mAP = 0.8447).  
Baseline (MegaDescriptor, Experiment 1): mAP = 0.741.  
Improvement over baseline: +0.10 mAP with DINOv2-ViT-L-14.

The submission CSV was generated using the DINOv2-ViT-L-14 checkpoint from epoch 50.

---

### Interpretation

DINOv2-ViT-L-14 wins despite not being wildlife-specialised. Its self-supervised training
on 142M diverse images produces extremely rich patch-level features, which transfer well
to fine-grained identity discrimination in jaguars.

MegaDescriptor underperforms expectations relative to DINOv2. Although it was pre-trained
specifically on wildlife re-ID datasets, its frozen features may be overfitted to the
pre-training domain's visual characteristics. The ArcFace projection head cannot overcome
this when the backbone is frozen.

ConvNeXt-V2-Base and EfficientNet-B4 are competitive (0.8273 and 0.8266) despite being
CNN-based and ImageNet-pretrained, suggesting the projection head successfully adapts
their features, though both remain behind DINOv2. ResNet-50 lags behind at 0.7671,
which is not surprising given that it is an older 
architecture with fewer parameters and was pretrained on a much simpler ImageNet
classification task compared to the other backbones tested here, but still beats
the MegaDescriptor baseline, suggesting the ArcFace head is doing real work regardless
of backbone.

---

## Experiment 5 — Ensemble Diversity Analysis for Jaguar Re-Identification

**W&B Run:** [ensemble-diversity](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/373tle4k)  
**Notebook:** [experiment_5.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_5/experiment-5-ensemble.ipynb)  
**Kaggle Submission ID:** 50725321  
**Public Leaderboard Score:** 0.846

---

### Research Question

Does a trained ensemble fusion model, where a single ArcFace head simultaneously learns
from the concatenated embeddings of multiple diverse frozen backbones, outperform any
individual backbone model? And do the complementarity and compute tradeoffs justify the
added complexity?

---

### Key Design Decision: Fusion Happens During Training

Unlike post-hoc averaging (combining already-trained models at inference), this experiment
trains a fusion head that sees all backbone signals simultaneously. The projection network
learns which backbone features are most discriminative for each jaguar identity, rather
than naively averaging post-hoc predictions.

Backbone 1 (MegaDescriptor)   [1536-d] --+
Backbone 2 (DINOv2)           [1024-d] --+
                                         +--> concat [5376-d] --> FusionHead --> [256-d] --> ArcFace
Backbone 3 (EfficientNet-B4)  [1792-d] --+
Backbone 4 (ConvNeXt-V2-Base) [1024-d] --+

---

### Intervention

| Variable | Value |
|---|---|
| Changed | Single backbone head vs. fused multi-backbone head |
| Fixed | Projection head architecture: Linear(dim→512→256) + BatchNorm + ReLU + Dropout(0.3) |
| Fixed | Loss: ArcFace (margin=0.5, scale=64) |
| Fixed | Optimiser: AdamW (lr=1e-4, weight_decay=1e-4) |
| Fixed | Scheduler: ReduceLROnPlateau (factor=0.5, patience=10) |
| Fixed | Epochs: 50, Batch size: 32, Seed: 42 |
| Fixed | Output embedding dim: 256 for all models |
| Fixed | Validation split: 20% stratified (all 31 identities in both sets) |
| All backbones | Fully frozen, no gradient updates to backbone weights |

---

### Ensemble Composition — Chosen for Diversity

| Backbone | Architecture | Pre-trained On | Embed Dim | Diversity Rationale |
|---|---|---|---|---|
| MegaDescriptor-L-384 | ViT-L / Swin | Wildlife Re-ID datasets | 1536 | Domain-specific wildlife identity features |
| DINOv2-ViT-L-14 | ViT-L | LVD-142M (self-supervised) | 1024 | No label bias; rich semantic representations |
| EfficientNet-B4 | CNN | ImageNet | 1792 | Local texture and edge features; CNN inductive bias |
| ConvNeXt-V2-Base | CNN-Transformer | ImageNet-22K | 1024 | Hybrid: bridges CNN locality and global attention |
| Fusion (concat) | n/a | n/a | 5376 → 256 | Joint projection over all backbone signals |

---

### Evaluation Protocol

Metric: Identity-balanced mAP on the fixed validation split. Comparison: 4 individual
ArcFace heads (one per backbone) vs. 1 fusion ArcFace head. Diversity evidence:
error-overlap matrix, per-identity AP gain, representational similarity (cosine sim
between embedding spaces). Compute tradeoff: head parameter count and inference latency
vs. mAP gain.

---

### Results

| Method | Val mAP | Head Params | Backbones Needed |
|---|---|---|---|
| MegaDescriptor-L-384 | 0.7748 | 927,744 | 1 |
| DINOv2-ViT-L-14 | 0.8257 | 665,600 | 1 |
| EfficientNet-B4 | 0.8245 | 1,058,816 | 1 |
| ConvNeXt-V2-Base | 0.8276 (best individual) | 665,600 | 1 |
| **FusionEnsemble** | **0.8358** | **2,893,824** | **4** |

Fusion best epoch: 48. Gain over best individual (ConvNeXt-V2-Base): +0.0082 mAP.
Per-identity summary (Fusion vs ConvNeXt-V2-Base): improved 13/31 identities, hurt
10/31 identities, mean per-identity gain: +0.0052.

The submission CSV was generated using the FusionEnsemble checkpoint.

---

### Diversity Evidence

The four backbones make partially non-overlapping errors. MegaDescriptor
and DINOv2 capture complementary identity cues, providing
the theoretical basis for fusion. 13 of 31 identities benefitted from fusion (mean gain
+0.0052). 10 identities were slightly hurt, suggesting the fusion head does not fully
exploit all complementary signals, likely because the projection network capacity is kept
small (256-d output). Lower cosine similarity between backbone embedding spaces indicates
higher diversity, the CNN-based (EfficientNet, ConvNeXt) vs. ViT-based (MegaDescriptor,
DINOv2) split confirms architectural diversity that the fusion head can leverage.

---

### Compute Tradeoff

| Metric | Best Individual | FusionEnsemble |
|---|---|---|
| Backbone count | 1 | 4 |
| Head params | ~528K | ~2.9M (5.5x) |
| Val mAP | 0.8276 | 0.8358 |
| Inference cost | 1x | ~4x (4 backbone forward passes) |

The +0.0082 mAP gain comes at the cost of running 4 full backbone forward passes at
inference and a 5.5x larger head.

---

### Interpretation

Fusion beats any individual model, confirming that architecturally diverse frozen backbones
capture complementary identity cues that a jointly-trained projection head can exploit.
The gains are small (+0.0082 mAP), partly because all backbones are frozen so the fusion
head must generalise with a fixed 5376-d input and limited parameters, and partly because
DINOv2 alone is already near-optimal for this dataset (Experiment 4), leaving little
complementary room for the other backbones.

ConvNeXt-V2-Base outperforms DINOv2 individually in this experiment, whereas Experiment 4
had DINOv2 winning. This reflects the stochastic nature of the 20% val split; across
experiments the gap is within noise. The 10/31 identities that were hurt by fusion are
likely cases where one backbone's strong signal gets diluted by weaker signals from others.

---

## Experiment 6 — Test-Time Augmentation & Query Expansion for Jaguar Re-Identification

**W&B Run:** [tta-aqe-q6](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/p0l6cobd)  
**Notebook:** [experiment_6.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_6/experiment-6-tta-aqe-v3.ipynb)  
**Kaggle Submission ID:** 50742056  
**Public Leaderboard Score:** 0.760

---

### Research Question

Can Test-Time Augmentation (TTA) and Average Query Expansion (AQE), applied to a frozen
model at inference time only with no additional GPU training, improve identity-balanced
mAP? And are their gains additive or super-additive when combined?

---

### Key Design Decision: Inference-Only Improvements

Unlike the previous experiments that modified training, this experiment keeps the model
checkpoint completely fixed and improves retrieval quality purely through two mechanisms.
TTA averages embeddings across N augmented views of each image before retrieval. AQE
refines each query embedding by blending it toward its top-k retrieved neighbours:

q_expanded = L2_norm( q + α · mean( top-k neighbours(q) ) )

This decouples inference quality from training cost, making TTA and AQE applicable to
any previously trained checkpoint.

---

### Intervention

| Variable | Value |
|---|---|
| Changed | Number of TTA views (1, 4, 8) and AQE config (k=0/5/10, α=0.0/1.0) |
| Fixed | Backbone: MegaDescriptor-L-384 (frozen) |
| Fixed | Projection head: Linear(1536→512→256) + BatchNorm + ReLU + Dropout(0.3) |
| Fixed | Loss: ArcFace (margin=0.5, scale=64) |
| Fixed | Optimiser: AdamW (lr=1e-4, weight_decay=1e-4) |
| Fixed | Scheduler: ReduceLROnPlateau (factor=0.5, patience=5) |
| Fixed | Epochs: 50, Batch size: 32, Seed: 42 |
| Fixed | Validation split: 20% stratified (all 31 identities in both sets) |

---

### Conditions Compared

| Condition | TTA Views | Query Expansion | Expected Benefit |
|---|---|---|---|
| Baseline | 1 | None | Reference single-pass inference |
| TTA-4 | 4 | None | Reduces single-view noise |
| TTA-8 | 8 | None | More robust, higher cost |
| AQE-k5 | 1 | top-5, α=1.0 | Pulls query toward match cluster |
| AQE-k10 | 1 | top-10, α=1.0 | Wider neighbourhood expansion |
| TTA-4 + AQE-k5 | 4 | top-5, α=1.0 | Combined — test if additive |
| TTA-8 + AQE-k10 | 8 | top-10, α=1.0 | Maximum quality |

---

### TTA Augmentation Views (ordered least to most aggressive)

| View | Augmentation |
|---|---|
| 1 | Original: resize + normalize only |
| 2 | Horizontal flip |
| 3 | Colour jitter (brightness, contrast, saturation 0.2) |
| 4 | Random resized crop (scale 0.85–1.0) |
| 5 | Flip + colour jitter |
| 6 | Flip + crop |
| 7 | Aggressive colour jitter (0.4, hue 0.1) |
| 8 | Crop + aggressive colour jitter |

TTA-4 always uses views 1–4 as a strict subset of TTA-8, keeping the ablation clean
and comparable.

---

### Results

| Condition | Val mAP | Gain vs Baseline | Inference Cost (×) |
|---|---|---|---|
| Baseline (TTA-1, no AQE) | 0.7748 | — | 1× |
| TTA-4 | 0.7755 | +0.0007 | 4× |
| TTA-8 | 0.7696 | −0.0052 | 8× |
| AQE-k5 (TTA-1) | 0.7713 | −0.0036 | ~1× |
| AQE-k10 (TTA-1) | 0.7638 | −0.0110 | ~1× |
| **TTA-4 + AQE-k5** | **0.7760** | **+0.0012** | **~4×** |
| TTA-8 + AQE-k10 | 0.7424 | −0.0325 | ~8× |

Best checkpoint: Epoch 50, val mAP 0.7748, val loss 4.6852.  
Best inference condition: TTA-4 + AQE-k5.  
Submission generated with: TTA-4 + AQE-k5 applied to test set.

---

### Additivity Analysis

| Metric | Value |
|---|---|
| Baseline | 0.7748 |
| Best TTA alone (TTA-4) | 0.7755 (+0.0007) |
| Best AQE alone (AQE-k5, TTA-1) | 0.7713 (−0.0036) |
| Best combined (TTA-4 + AQE-k5) | 0.7760 (+0.0012) |
| Additive prediction | 0.7720 |
| Interaction effect | +0.0040 (super-additive) |

TTA and AQE are super-additive, their combination outperforms the sum of their
individual gains. AQE alone hurts (−0.0036) because single-view embeddings contain
noise that corrupts the neighbourhood. TTA first stabilises the embedding, allowing
AQE to pull the query toward a cleaner match cluster.

---

### Interpretation

AQE on its own hurts performance (−0.0036). A single-pass embedding for a jaguar image
carries noise from pose variation, partial occlusion, and lighting differences. Blending
a noisy query toward its nearest neighbours does not fix these issues, it just propagates
the noise further into the refined embedding.  

TTA alone produces a small but consistent improvement (+0.0007). Averaging embeddings
across 4 augmented views of the same image reduces the instability introduced by any
single crop, resulting in a more representative embedding for that identity.  

The most important finding is that TTA and AQE together outperform both individually
(+0.0012 combined vs an additive prediction of −0.0029). TTA stabilises the query
embedding first, which then allows AQE to correctly pull it toward the true identity
cluster. Neither technique is sufficient on its own for this dataset, but the combination
works well because they address different problems sequentially.  

Increasing TTA views to 8 or widening the AQE neighbourhood to k=10 did not improve
results further. Beyond a certain point, more augmentation and wider neighbourhood
expansion begin to wash out the fine-grained coat pattern details that distinguish
individual jaguars. The optimal configuration for this dataset was TTA-4 combined with
AQE-k5.  

---

## Experiment 7 — PK Sampling & Batch-Hard Triplet Loss for Jaguar Re-Identification

**W&B Run:** [pk-triplet-q7](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/qc72pxg8)  
**Notebook:** [experiment_7.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_7/experiment-7-pk-triplet.ipynb)  
**Kaggle Submission ID:** 51007885   
**Public Leaderboard Score:** 0.747  

---

### Research Question

Does replacing random mini-batch construction with PK sampling — which guarantees
P identities × K images per batch, forcing every gradient update to confront hard
positives and meaningful negatives — improve identity-balanced mAP? And does adding
a batch-hard triplet loss on top of ArcFace provide further geometric improvement
by explicitly penalising embedding-space violations?

---

### Why PK Sampling Should Help (Prior Hypothesis)

With a random batch of 32 images drawn from 31 identities, most of the negative pairs
in any given batch are easy to distinguish. The model quickly learns to separate jaguars
that look very different from each other but gets very little training signal on the
pairs that are actually hard to tell apart. PK sampling addresses this by guaranteeing
that each batch contains exactly K images from each of P selected identities. This
forces the model to see multiple images of the same jaguar in every update, which means
it has to learn to handle the natural variation within an identity such as different
poses, lighting conditions, and partial occlusions rather than just the easy cases.

On top of that, batch-hard triplet loss goes one step further by identifying the hardest
pair within each batch. For each anchor image, it finds the same-identity image that the
model currently finds most dissimilar (hardest positive) and the different-identity image
that the model finds most similar (hardest negative). The loss then penalises the model
specifically for these difficult cases:  

loss = mean( max(0, d(a, p_hard) - d(a, n_hard) + margin) )  

The idea is that focusing the gradient on the most confusing pairs should push the
embedding space into a cleaner structure where same-identity images cluster together
and different-identity images stay further apart.

---

### Intervention

| Variable | Value |
|---|---|
| Changed | Batch strategy: random vs. PK (P=8 identities, K=4 images, batch=32) |
| Changed | Loss function: ArcFace only vs. ArcFace + λ·BatchHardTriplet |
| Fixed | Backbone: MegaDescriptor-L-384 (frozen, pre-computed embeddings) |
| Fixed | Projection head: Linear(1536→512→256) + BatchNorm + ReLU + Dropout(0.3) |
| Fixed | ArcFace: margin=0.5, scale=64 |
| Fixed | Triplet margin: 0.3 |
| Fixed | Optimiser: AdamW (lr=1e-4, weight_decay=1e-4) |
| Fixed | Scheduler: ReduceLROnPlateau (factor=0.5, patience=5) |
| Fixed | Epochs: 50, Batch size: 32 (P×K = 8×4), Seed: 42 |
| Fixed | Validation split: 20% stratified (all 31 identities in both sets) |

---

### Conditions Compared

| Condition | Batch Strategy | Loss | Research Purpose |
|---|---|---|---|
| A — RandomArcFace | Random, batch=32 | ArcFace only | Baseline — matches prior experiments |
| B — PKArcFace | PK (P=8, K=4) | ArcFace only | Isolates contribution of hard sampling |
| C — PKArcFaceTriplet0.5 | PK (P=8, K=4) | ArcFace + 0.5×Triplet | Combined loss, moderate geometry weight |
| D — PKArcFaceTriplet1.0 | PK (P=8, K=4) | ArcFace + 1.0×Triplet | Combined loss, equal geometry weight |

All four models are trained from the same random seed with identical hyperparameters.
The only variable is batch strategy × loss function.

---

### PK Sampler Details

Each batch is constructed by randomly picking P=8 identities from the training set
without replacement, then sampling K=4 images per identity, with replacement only if
an identity has fewer than 4 images available. This gives a batch size of 32 (P×K)
where every batch is guaranteed to contain multiple images of the same jaguar. With
31 identities in the dataset, this setup produces only 3 batches per epoch (96 training
samples), compared to approximately 47 batches per epoch under random sampling. All 31
identities have at least 4 images, so replacement was never needed.

---

### Results

| Condition | Best Val mAP | Δ Baseline | Best Epoch |
|---|---|---|---|
| **A — RandomArcFace** | **0.7793** | n/a | 50 |
| B — PKArcFace | 0.3719 | −0.4074 | 50 |
| C — PKArcFaceTriplet0.5 | 0.3727 | −0.4066 | 50 |
| D — PKArcFaceTriplet1.0 | 0.3744 | −0.4049 | 50 |

Best condition: A (Random + ArcFace only). Total gain over baseline: 0.0000.
Submission generated with: Condition A.

#### Gain Decomposition

| Effect | Value |
|---|---|
| PK sampling alone (B vs A) | −0.4074 |
| Best triplet on top of PK (D vs B) | +0.0025 |
| Total best vs baseline (best of B/C/D vs A) | −0.4049 |

---

### Triplet Violation Rate Analysis

The triplet violation rate tracks what fraction of anchors in a batch have an active
triplet loss (i.e. loss > 0). If the model is learning well, this rate should go down
over time as same-identity images get closer together and different-identity images move
further apart. At epoch 1, the violation rate is approximately 1.0 across all PK+Triplet
conditions, meaning every single triplet in the batch is violated. This rate does drop
over training, which shows that PK sampling does push the embeddings in the right
direction geometrically. However, this improvement in embedding structure does not carry
over into better retrieval mAP on the validation set.

---

### Interpretation

PK sampling produces a very large drop in mAP (−0.4074), which was not expected. The
root cause is a mismatch between the sampling strategy and the dataset size. With only
31 identities and P=8, K=4, the sampler produces just 3 batches per epoch instead of
the ~47 batches seen under random sampling. This means the model receives far fewer
weight updates per epoch and simply does not have enough gradient steps to converge
within 50 epochs. This is a practical limitation specific to small datasets rather than
a fundamental problem with PK sampling as a method.  

Adding the batch-hard triplet loss on top of PK sampling does recover a small amount of
performance (+0.0025 at λ=1.0). The higher the weight given to the triplet term, the
more it helps, but the gains are too small to matter when the baseline is already
degraded by 0.4074 mAP.  

Random batching with ArcFace alone remains the best configuration. It gives the model
approximately 47 batches of gradient updates per epoch across a diverse mix of negatives,
which is simply more training signal than PK sampling can provide at this dataset scale.  

---

## Experiment 10 — Swin Transformer + ArcFace (Backbone Comparison vs MegaDescriptor)

**W&B Run:** [swin-arcface-local](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/ch27ksiv)  
**Notebook:** [experiment_10.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_10/experiment_10.ipynb)  
**Kaggle Submission ID:** 50750223  
**Public Leaderboard Score:** 0.789

---

### Research Question

Does replacing MegaDescriptor-L-384 with a Swin Transformer backbone, trained under an identical ArcFace protocol, improve
identity-balanced mAP beyond the established baseline of 0.7720 for jaguar
re-identification?

---

### Hypotheses

**H₁:** The Swin Transformer's hierarchical shifted-window attention captures richer
multi-scale features than MegaDescriptor, which should result in higher mAP when
trained with ArcFace under the same setup.

**H₂:** Even though Swin uses a lower input resolution (224×224 vs 384×384),
its stronger general-purpose vision pre-training may be enough to outperform
MegaDescriptor on this 31-identity closed-set problem.

---

### Intervention

The only change from Experiment 1 is the backbone:

| Variable | Experiment 1 (Baseline) | Experiment 10 |
|---|---|---|
| Backbone | MegaDescriptor-L-384 (wildlife re-ID pre-trained) | swin_large_patch4_window7_224 (ImageNet pre-trained) |
| Input resolution | 384×384 | 224×224 |
| Projection head | Linear(1536→512→256) + BN + ReLU + Dropout(0.3) | same |
| Loss | ArcFace (margin=0.5, scale=64) | same |
| Optimiser | AdamW (lr=1e-4, wd=1e-4) | same |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) | same |
| Epochs | 50, Batch=32, Seed=42 | same |
| Val split | 20% stratified, all 31 identities | same |  

Both backbones output 1536-dimensional features before the projection head, so the
downstream training conditions are identical.

---

### Results

#### Backbone Comparison: Swin Transformer vs MegaDescriptor

| Backbone | Input Size | Val mAP (Identity-Balanced) |
|---|---|---|
| MegaDescriptor-L-384 (Exp 1 Baseline) | 384×384 | 0.7720 |
| swin_large_patch4_window7_224 (Exp 10) | 224×224 | **0.81448** |
| Delta | | **+0.04248** |  

Swin Transformer surpasses the MegaDescriptor baseline by +0.042 mAP (+5.5% relative
improvement), confirming both H₁ and H₂.

---

### Qualitative Analysis

Before fine-tuning, the Swin embeddings show some identity grouping but with a lot of
overlap between different jaguars, which is expected since ImageNet features have not
been adapted to tell apart individual animals. After ArcFace fine-tuning, the embeddings
form much tighter clusters per identity with clearer separation between different jaguars,
showing that the ArcFace loss successfully reshapes the embedding space.  

Before fine-tuning, retrieval often returns the wrong jaguar even if it looks visually
similar. After fine-tuning, the top-5 retrieved images are mostly correct identity
matches, which confirms that ArcFace is pulling out clearly discriminative features
from the Swin backbone.

---

### Interpretation

Swin Transformer outperforms MegaDescriptor on this dataset (+0.042 mAP) even though
MegaDescriptor was specifically pre-trained on wildlife re-identification data. This
suggests that for a small closed-set problem with only 31 identities, a strong
general-purpose vision backbone can be just as good, or even better, than a
domain-specific one.  

One thing worth noting is that the two backbones use different input resolutions.
Swin runs at 224×224 while MegaDescriptor runs at 384×384. The fact that Swin still
wins despite processing smaller images points to its shifted-window attention mechanism
making up for the lower resolution. To properly isolate the effect of the backbone
architecture alone, a resolution-matched comparison using SwinV2 at 384×384 would
be needed.  

Since both experiments use the exact same projection head, loss function, and optimiser,
the +0.042 mAP improvement can be attributed directly to the backbone swap and nothing
else.  

---

## Experiment 11 — K-Reciprocal Re-Ranking Parameter Search

**W&B Run:** [k-reciprocal-rerank-q8](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/6nkun9ww)  
**Notebook:** [experiment_11.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_11/experiment-11-krerank.ipynb)  
**Kaggle Submission ID:** 51007018  
**Public Leaderboard Score:** 0.762  

---

### Research Question

Does k-reciprocal re-ranking (Zhong et al., CVPR 2017), a post-processing step applied
to the pre-computed embedding similarity matrix, improve identity-balanced mAP beyond
the no-reranking baseline of 0.7748? And what parameter configuration (k₁, k₂, λ) works
best for the jaguar dataset?

---

### Background: K-Reciprocal Re-Ranking

Standard retrieval ranks gallery images by how similar they are to a query using cosine
similarity. K-reciprocal re-ranking improves this by checking whether the similarity is
mutual. If image g is in the top-k₁ nearest neighbours of query q, the algorithm checks
whether q is also in the top-k₁ nearest neighbours of g. Images that appear in each
other's neighbour lists are far more likely to be genuine identity matches than images
that only appear in one direction.  

The algorithm works in the following steps: it finds the k₁-reciprocal neighbourhood for
each query, expands it by merging overlapping neighbour sets to reduce noise, encodes
this neighbourhood as a Gaussian-weighted vector, applies a local smoothing step using
k₂ neighbours, computes a Jaccard distance based on neighbourhood overlap, and finally
blends this with the original cosine distance using the weighting parameter λ:  

d_final = (1−λ)·d_Jaccard + λ·d_original  

Importantly, re-ranking is applied entirely after training on the pre-computed embedding
matrix using only NumPy on CPU. The model itself is never retrained.

---

### Intervention

No changes are made to the model. The intervention is entirely at the retrieval stage:

| Variable | Value |
|---|---|
| Base model | MegaDescriptor-L-384 + ArcFace projection head (best checkpoint, epoch 50, val mAP = 0.7748) |
| Re-ranking algorithm | Zhong et al. CVPR 2017 k-reciprocal re-ranking |
| k₁ sweep | {10, 15, 20, 25, 30} — size of k-reciprocal neighbourhood |
| k₂ sweep | {3, 6, 9} — local query expansion neighbours |
| λ sweep | {0.1, 0.3, 0.5} — blend weight (0 = pure Jaccard, 1 = pure original cosine) |
| Total configurations | 45 (5 × 3 × 3) |
| Val split | 20% stratified (all 31 identities, Seed=42), identical to all prior experiments |
| Baseline (no re-ranking) | Cosine distance, single-pass retrieval: mAP = 0.7748 |  

---

### Results

#### Best Configuration

| Configuration | Val mAP | Gain over Baseline |
|---|---|---|
| Baseline (no re-ranking) | 0.7748 | n/a |
| **Best: k₁=15, k₂=3, λ=0.1** | **0.7892** | **+0.0144** |
| Zhong defaults (k₁=20, k₂=6, λ=0.3) | 0.7768 | +0.0020 |  

#### Full 45-Configuration Sweep (Selected Results)

| k₁ | k₂ | λ | Val mAP | Gain |
|---|---|---|---|---|
| 15 | 3 | 0.1 | **0.7892** | **+0.0144** |
| 20 | 3 | 0.1 | 0.7885 | +0.0137 |
| 10 | 3 | 0.1 | 0.7881 | +0.0133 |
| 25 | 3 | 0.1 | 0.7851 | +0.0103 |
| 10 | 6 | 0.1 | 0.7821 | +0.0072 |
| 30 | 3 | 0.1 | 0.7835 | +0.0086 |
| 20 | 6 | 0.3 | 0.7768 | +0.0020 |
| 15 | 9 | 0.3 | 0.7753 | +0.0005 |
| 15 | 9 | 0.1 | 0.7720 | −0.0028 |
| 20 | 9 | 0.1 | 0.7709 | −0.0039 |
| 25 | 9 | 0.1 | 0.7708 | −0.0040 |  

#### Compute Cost

| Condition | Time |
|---|---|
| Baseline (no re-ranking) | 0s |
| Fastest config (k₁=10, k₂=3) | 0.18s |
| Best config (k₁=15, k₂=3, λ=0.1) | 0.27s |
| Zhong defaults (k₁=20, k₂=6, λ=0.3) | 0.38s |
| Slowest config (k₁=30, k₂=9) | 0.65s |
| Max gain per second | +0.0144 in 0.27s |  

All 45 configurations run on CPU using pure NumPy, making re-ranking essentially
free to add relative to the cost of model training.

---

### Parameter Sensitivity Analysis

Three clear patterns emerge.    

**λ (blend weight):** A lower value is better. λ=0.1 consistently outperforms λ=0.3
and λ=0.5 across all parameter combinations. This means the Jaccard distance based on
neighbourhood overlap is a more reliable signal than the original cosine distance for
this dataset, so it should be weighted more heavily.  

**k₂ (local smoothing):** A smaller value is better. k₂=3 consistently outperforms
k₂=6 and k₂=9. Using more neighbours for the smoothing step blurs out the fine identity
structure that the re-ranking is trying to recover, which is a particular problem on a
dataset this small.  

**k₁ (reciprocal neighbourhood size):** A moderate value works best. k₁=15 gives the
best results. Too small (k₁=10) means there is not enough mutual-neighbour evidence to
work with. Too large (k₁=25 or 30) pulls in false positives that corrupt the ranking.
With only ~12 validation images per identity, k₁=15 is the right balance.   

The worst configurations are those with k₂=9 and λ=0.1, which actually hurt performance
below the baseline. Over-smoothed neighbourhood vectors combined with heavy Jaccard
weighting break the ranking entirely for small identity clusters.  

---

### Interpretation

K-reciprocal re-ranking gives a +0.0144 mAP improvement over the 0.7748 baseline at
no retraining cost and in under a second of compute. The mutual nearest-neighbour
structure captures identity information that cosine similarity alone misses.  

The default parameters from Zhong et al. (k₁=20, k₂=6, λ=0.3) only give +0.0020 on
this dataset, far below the best configuration's +0.0144. This is not surprising because
those defaults were tuned on large-scale person re-ID datasets like Market-1501 and
DukeMTMC with hundreds of identities and thousands of images. The jaguar dataset, with
only 31 identities and around 12 validation images per identity, needs smaller and more
conservative parameter values to avoid over-smoothing.  

The improvement is likely most useful for the harder identity pairs identified in
Experiment 9. For an identity like Pyte, which gets confused with Ti 14.3% of the time,
re-ranking can check whether Pyte images mutually appear in each other's neighbour lists
while Ti images do not, and correct the ranking accordingly.  

---

## Experiment 12 — Eight-Loss Metric Learning Comparison on DINOv2 ViT-L14

**W&B Run:** [loss-comparison-dinov2](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/nymfiyh5)  
**Notebook:** [experiment_12.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_12/experiment-12-loss-dinov2-v3.ipynb)  
**Kaggle Submission ID:** 50750836  
**Public Leaderboard Score:** 0.803  
**Cross-reference:** Experiment 4 (backbone comparison, identified DINOv2 ViT-L14 as
best-performing backbone)

---

### Research Question

Do loss functions that directly optimise the embedding space (Circle Loss, Multi-Similarity,
Proxy-NCA, Contrastive) outperform classification-margin losses (ArcFace, CosFace,
SubCenterArcFace, FocalLoss) on jaguar re-identification when using DINOv2 ViT-L14 as
the frozen backbone?

---

### Background: Two Loss Families

**Classification-Margin Losses** treat re-ID as a standard classification problem during
training. A classification head assigns logits to each of the 31 jaguar identities and a
margin term pushes same-identity embeddings closer together while separating different
identities. The classification head is discarded at inference and only the embedding is
used for retrieval. The key advantage is that every training step sees all 31 class
prototypes, which provides dense and stable gradient signal even in small batches.

**Metric Learning Losses** skip the classification head entirely and directly optimise
how similar or dissimilar pairs of embeddings should be. They apply larger gradient
updates to harder pairs. The downside on a small dataset is that hard pairs
can be sparse in any given batch, leaving much of the gradient signal unused. The one
exception here is Proxy-NCA, which sidesteps pair mining by learning one representative
proxy vector per identity instead.

| Loss | Family | Key Mechanism |
|---|---|---|
| ArcFace | Classification-margin | Additive angular margin on cosine logits |
| CosFace | Classification-margin | Additive cosine margin (large margin softmax) |
| SubCenterArcFace | Classification-margin | K=3 sub-centres per identity; tolerates intra-class noise |
| FocalLoss | Classification-margin | Down-weights easy examples via (1−p)^γ=2 with ArcFace head |
| Circle Loss | Metric learning | Per-score adaptive weighting; hard pairs get larger gradients |
| Multi-Similarity | Metric learning | Three-signal pair mining: self, neg-relative, pos-relative |
| Proxy-NCA | Metric learning | Learnable proxy per identity; no pair mining needed |
| Contrastive | Metric learning | Positive/negative pair loss with semi-hard negative mining |

---

### Experimental Design

The backbone is DINOv2 ViT-L14, fully frozen, producing 1024-dim embeddings at its
native 518px resolution. All embeddings are extracted and cached once before any
training begins. All 8 loss conditions train only the projection head from the same
random seed on the same cached embeddings, so any difference in mAP is purely down
to the choice of loss function.

| Controlled Variable | Value (Identical Across All 8 Conditions) |
|---|---|
| Backbone | DINOv2 ViT-L14, frozen, 1024-dim output |
| Projection head | Linear(1024→512)→BN→ReLU→Dropout(0.3)→Linear(512→256)→BN |
| Embedding dim | 256 |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Epochs / Patience | 50 epochs, early stopping patience=10 |
| Batch size | 32 |
| Seed | 42 |
| Val split | 20% stratified (all 31 identities) |

Loss-specific hyperparameters:

| Loss | Hyperparameters |
|---|---|
| ArcFace | margin=0.5, scale=64.0 |
| CosFace | margin=0.35, scale=64.0 |
| SubCenterArcFace | K=3 sub-centres, margin=0.5, scale=64.0 |
| FocalLoss | γ=2.0 (with ArcFace head) |
| Circle Loss | m=0.25, γ=64 |
| Multi-Similarity | α=2.0, β=50.0, base=0.5, ε=0.1 |
| Proxy-NCA | temperature=0.1 |
| Contrastive | margin=1.0, semi-hard negative mining |  

---

### Results

All 8 conditions ran for 50 epochs. Final ranked results from notebook output:  

| Rank | Loss | Family | Val mAP | Best Epoch |
|---|---|---|---|---|
| **1** | **ProxyNCA** | **Metric** | **0.8270** | **50** |
| 2 | ArcFace | Classification-margin | 0.8267 | 48 |
| 3 | FocalLoss | Classification-margin | 0.8259 | 50 |
| 4 | CosFace | Classification-margin | 0.8172 | 50 |
| 5 | SubCenterArcFace | Classification-margin | 0.7652 | 50 |
| 6 | CircleLoss | Metric | 0.7349 | 50 |
| 7 | MultiSimilarity | Metric | 0.7096 | 50 |
| 8 | Contrastive | Metric | 0.6146 | 50 |  

Kaggle submission used: ProxyNCA (val mAP = 0.8270).

#### Family-Level Summary

| Loss Family | Mean Val mAP | Best in Family |
|---|---|---|
| Classification-margin (4 losses) | 0.8087 | FocalLoss / ArcFace |
| Metric learning (4 losses) | 0.7215 | ProxyNCA (0.8270) |  

Although metric learning losses average lower as a group (0.7215 vs 0.8087),
ProxyNCA individually tops the overall ranking at 0.8270, just 0.0003 above ArcFace.
The other three metric learning losses (CircleLoss, MultiSimilarity, Contrastive) are
far behind in the 0.61 to 0.73 range, which drags the metric family average down.  

---

### Interpretation

ProxyNCA edges out the top spot at 0.8270, but the difference between it, ArcFace
(0.8267), and FocalLoss (0.8259) is so small that all three should be treated as
equally strong for this dataset. A gap of 0.0003 to 0.0008 mAP is well within the
range of training noise across different random seeds.  

The reason ProxyNCA works well compared to the other metric learning losses is
straightforward. CircleLoss, MultiSimilarity, and Contrastive all need to find 
hard pairs within each batch to generate useful gradient signal. On a dataset
with only 31 identities and small batch sizes, truly hard pairs are often not present
in a given batch, so these losses frequently get very little useful training signal.
ProxyNCA avoids this problem entirely by learning one fixed representative vector per
identity and training each embedding to be closest to its own identity's proxy. This
behaves much more like a classification loss, which is why it converges similarly to
ArcFace on this small dataset.  

SubCenterArcFace performs noticeably worse than standard ArcFace (0.7652 vs 0.8267).
The idea behind sub-centres is to handle cases where the same identity looks very
different across images, by allowing each class to have multiple cluster centres.
However, as shown in Experiment 9, the hard cases in this dataset are mostly jaguars
being confused with other jaguars rather than within-identity appearance variation.
Using 3 sub-centres per class with only around 12 validation images per identity ends
up complicating the decision boundaries without solving the actual problem.  

FocalLoss performs on par with ArcFace, which makes sense. By down-weighting the
gradient contribution of easy, well-classified examples, it focuses training on the
harder identities in the same way that ArcFace's angular margin naturally pushes
attention toward difficult cases.  

---

## Experiment 13 — ArcFace Hyperparameter Sweep on DINOv2 Embeddings

**WandB Run:** [arcface-sweep-dinov2](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/p91q08ms)
**Notebook:** [experiment-13-arcface-sweep.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_13/experiment-13-arcface-sweep.ipynb)
**Kaggle Submission ID:** 50767203
**Public Leaderboard Score:** 0.864 **TOP#2 Submission**

---

### Research Question

Is the default ArcFace configuration (margin=0.5, scale=64, embdim=256) optimal for jaguar re-ID on frozen DINOv2-ViT-L/14 embeddings, or does a 3-stage grid sweep of margin, scale, embedding dimension, hidden dimension, and dropout push mAP beyond the Exp 4 baseline of 0.8447?

---

### Hypothesis

ArcFace was originally designed for face recognition with millions of people and hundreds of photos per person. This problem is very different, with only 31 jaguars and some identities having as few as 13 training photos. The default settings (margin=0.5, scale=64, embdim=256) are therefore unlikely to be ideal here.

Three specific changes were expected to help. First, a margin around 0.6 should outperform the default 0.5, because jaguar coat patterns are visually more distinct than human faces, so slightly wider angular boundaries should help the model separate identities more cleanly. Second, a lower scale (48) should outperform the default (64), because DINOv2 already produces well-separated features for 31 jaguars and a high scale is unnecessary and may destabilise training. Third, a larger embedding dimension (512) should outperform the default (256), because DINOv2 outputs 1024-dimensional features and compressing them down to 256 likely discards useful identity information.

Overall, the expectation was to beat the Exp 4 baseline of mAP = 0.8447 by at least +0.01 through a systematic 3-stage sweep of these parameters.

---

### Defined Intervention

| Component | Exp 4 Baseline | Experiment 13 (Best Found) |
|---|---|---|
| Backbone | DINOv2-ViT-L/14, fully frozen | DINOv2-ViT-L/14, fully frozen |
| ArcFace margin | 0.5 | 0.6 (swept: {0.2, 0.3, 0.4, 0.5, 0.6, 0.7}) |
| ArcFace scale | 64.0 | 48.0 (swept: {48, 64, 80, 128}) |
| embdim | 256 | 512 (swept: {128, 256, 512, 1024}) |
| hiddendim | 512 | 1024 (swept: {512, 1024}) |
| Dropout | 0.3 | 0.3 (swept: {0.1, 0.3, 0.5}) |
| Optimizer | AdamW | AdamW (unchanged) |
| lr / weight_decay | 1e-4 / 1e-4 | 1e-4 / 1e-4 (unchanged) |
| Batch size | 32 | 32 (unchanged) |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau (unchanged) |
| Epochs / Patience | 50 / 10 | 50 / 10 (unchanged) |
| Data split | 80/20 stratified, seed=42 | 80/20 stratified, seed=42 (unchanged) |
| Evaluation | Identity-balanced mAP (cosine) | Identity-balanced mAP (cosine) (unchanged) |

Sweep structure: 3 sequential stages, 34 configs total, all on cached embeddings.

| Stage | Swept Parameters | Fixed | Configs |
|---|---|---|---|
| Stage 1 | margin x scale | embdim=256, hidden=512, dropout=0.3 | 6 x 4 = 24 |
| Stage 2 | embdim | Best margin/scale from S1, hidden=512, dropout=0.3 | 4 |
| Stage 3 | hiddendim x dropout | Best margin/scale/embdim from S1+S2 | 2 x 3 = 6 |

---

### Evaluation Protocol

Metric: Identity-balanced Mean Average Precision (mAP). A fixed 20% stratified validation split (seed=42) was used, giving 1516 train and 379 val samples across all 31 identities in both sets. Similarity was computed as cosine similarity on L2-normalised embeddings. Per-stage tracking included best val mAP per config, gain vs Exp 4 baseline, and best epoch, all logged to WandB. The comparison point was the Exp 4 best config (mAP = 0.8447, margin=0.5, scale=64, embdim=256, hidden=512, dropout=0.3).

---

### Results

#### Stage 1: Margin x Scale Grid (best per margin)

| margin | Best scale | Val mAP | Gain vs Exp 4 |
|---|---|---|---|
| 0.2 | 48 | 0.7687 | -0.0760 |
| 0.3 | 48 | 0.8080 | -0.0367 |
| 0.4 | 48 | 0.8219 | -0.0228 |
| 0.5 | 48 | 0.8308 | -0.0139 |
| **0.6** | **48** | **0.8353** | **-0.0094** |
| 0.7 | 48 | 0.8309 | -0.0138 |

Stage 1 best: margin=0.6, scale=48, mAP = 0.8353

#### Stage 2: Embedding Dimension (margin=0.6, scale=48)

| embdim | Val mAP | Gain vs Exp 4 |
|---|---|---|
| 128 | 0.8379 | -0.0068 |
| 256 | 0.8353 | -0.0094 |
| **512** | **0.8555** | **+0.0108** |
| 1024 | 0.8517 | +0.0070 |

Stage 2 best: embdim=512, mAP = 0.8555

#### Stage 3: Hidden Dim x Dropout (margin=0.6, scale=48, embdim=512)

| hiddendim | dropout | Val mAP | Gain vs Exp 4 |
|---|---|---|---|
| 512 | 0.1 | 0.8570 | +0.0123 |
| 512 | 0.3 | 0.8555 | +0.0108 |
| 512 | 0.5 | 0.8296 | -0.0151 |
| 1024 | 0.1 | 0.8555 | +0.0108 |
| **1024** | **0.3** | **0.8631** | **+0.0184** |
| 1024 | 0.5 | 0.8535 | +0.0088 |

Stage 3 best / Overall best: hiddendim=1024, dropout=0.3, mAP = 0.8631

#### Final Comparison

| Configuration | Val mAP | Gain |
|---|---|---|
| Exp 4 baseline | 0.8447 | - |
| Exp 13 best (margin=0.6, scale=48, embdim=512, hidden=1024, dropout=0.3) | **0.8631** | **+0.0184** |

**WandB logged:** per-config val mAP, gain vs baseline, best epoch, stage1 heatmap, stage2 embdim plot, stage3 results table, best model artifact, submission artifact.

---

### Interpretation

The 3-stage sequential ArcFace sweep on frozen DINOv2-ViT-L/14 embeddings yields a net gain of +0.0184 mAP over the Exp 4 baseline, reaching 0.8631. The two largest individual contributions came from increasing embdim from 256 to 512 (+0.0108), which recovered discriminative information discarded by the previous compression bottleneck, and increasing hiddendim from 512 to 1024 (+0.0076 marginal gain). Lowering scale from 64 to 48 was also consistently beneficial across all margins, confirming that DINOv2's well-separated frozen features do not require a high temperature for stable ArcFace training. The best overall configuration (margin=0.6, scale=48, embdim=512, hiddendim=1024, dropout=0.3) becomes the new baseline for all subsequent experiments.  

Stage 1 alone (margin=0.6, scale=48) actually underperforms the Exp 4 baseline (0.8353 vs 0.8447) because embdim was still fixed at 256. This indicates that margin and scale do not operate independently, their effect depends on how much capacity the embedding layer has. The Exp 4 baseline at margin=0.5, scale=64 happened to work reasonably well only because those values partially made up for the small embedding size. The real gain only shows up in Stage 2, once embdim is increased to 512. One limitation of the sequential sweep design is that earlier parameters are not re-evaluated after later ones are updated, so running a full grid search across all five parameters at once might arrive at a slightly different optimal combination.  

These parameter choices make sense given the nature of the problem. DINOv2 backbones pre-trained on wildlife data already do a good job of separating embeddings for 31 jaguar identities, so there is no need for an aggressive scale value. Keeping scale at 48 prevents the gradient updates from becoming too large during training, which matters a lot when the dataset is this small. A margin of 0.6 strikes a reasonable balance: jaguar coat patterns are distinctive enough that a bit more angular separation than the default helps, but pushing it to 0.7 would likely cause the model to overfit, especially for identities with only 13 training images. For the embedding dimension, 512 works well because it retains enough of DINOv2's 1024-dimensional output to preserve useful identity information, while still being compact enough to give ArcFace a meaningful classification challenge.   

---

## Experiment 14 — Seed Stability Analysis: DINOv2-ViT-L/14 + Optimised ArcFace Head

**W&B Run:** [seed-stability-dinov2](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/ji25wzxy)  
**Notebook:** [experiment_14.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_14/experiment-14-seed-stability.ipynb)  
**Kaggle Submission ID:** 50928578  
**Public Leaderboard Score:** 0.868   **TOP#1 Submission**

---

### Research Question

Is the best configuration from Experiment 13 (val mAP = 0.8631, seed=42) a 
strong result, or could it have been a lucky random initialisation? Does the improvement
over the Exp 4 baseline (0.8447) hold consistently across different random seeds?

---

### Configuration Under Test

*(Best configuration from Experiment 13)*

| Parameter | Value | Source |
|---|---|---|
| Backbone | DINOv2-ViT-L/14 (frozen) | Experiment 4 best backbone |
| margin | 0.6 | Exp 13 Stage 1 best |
| scale | 48.0 | Exp 13 Stage 1 best |
| embdim | 512 | Exp 13 Stage 2 best |
| hiddendim | 1024 | Exp 13 Stage 3 best |
| dropout | 0.3 | Exp 13 Stage 3 best |
| lr | 1e-4 | Fixed (same as Exp 4) |
| weight_decay | 1e-4 | Fixed (same as Exp 4) |
| batch_size | 32 | Fixed (same as Exp 4) |
| epochs | 50, patience 10 | Fixed (same as Exp 4) |

---

### Protocol

Seven seeds were tested: 42, 0, 1, 7, 123, 777, and 2024. The validation split was
fixed at seed=42 for every run, meaning all 7 runs are evaluated on exactly the same
379 validation images. This is important because it means the mAP values are directly
comparable across seeds without any variation coming from which images ended up in the
validation set.

What changes between runs is the model weight initialisation, the order in which
training batches are assembled, and the dropout mask. Everything else, including the
cached backbone embeddings and all hyperparameters, stays the same.

Baselines used for comparison: Exp 4 (MegaDescriptor baseline) at mAP = 0.8447, and
Exp 13 (ArcFace sweep, seed=42) at mAP = 0.8631.

---

### Results

#### Per-Seed Val mAP

| Seed | Val mAP | Best Epoch | Δ vs Exp 4 |
|---|---|---|---|
| 42 *(Exp 13 seed)* | 0.8672 | 31 | +0.0225 |
| 0 | **0.8818** | 35 | **+0.0371** |
| 1 | 0.8580 | 36 | +0.0133 |
| 7 | 0.8637 | 32 | +0.0190 |
| 123 | 0.8707 | 39 | +0.0260 |
| 777 | 0.8607 | 40 | +0.0160 |
| 2024 | 0.8714 | 41 | +0.0267 |

#### Aggregate Statistics

| Metric | Value |
|---|---|
| Mean mAP | **0.8676** |
| Std mAP | **0.0074** |
| 95% CI (approx ±2σ) | [0.8529, 0.8824] |
| Min mAP | 0.8580 |
| Max mAP | 0.8818 |
| Range | 0.0239 |
| Seeds beating Exp 4 baseline (0.8447) | **7 / 7** |
| Mean gain over Exp 4 | **+0.0229** |
| Min gain over Exp 4 | +0.0133 |
| Stability verdict | **STABLE** (std = 0.0074 < threshold 0.010) |

Best seed for submission: seed=0 → mAP = 0.8818

---

### Note on Exp 13 vs Exp 14 seed=42 Discrepancy

Exp 13 reported mAP = 0.8631 for seed=42, while this experiment gets 0.8672 for the
same seed, a difference of +0.0041. This is expected because Exp 13 ran all three
stages sequentially in a single session, whereas Exp 14 independently re-extracts and
re-caches the DINOv2 embeddings from scratch. The difference is small and sits
comfortably within the natural seed-level variance of 0.0074, so it does not change
any conclusions about stability.

---

### Analysis and Interpretation

The main finding is that the improvement from Experiment 13 is not a one-off result.
Every single seed beats the Exp 4 baseline, with the worst case being seed=1 at 0.8580,
which is still a clear improvement of +0.0133. This confirms that the better
configuration found in Experiment 13 is clearly better and not just a consequence of
a particularly favourable random initialisation.

The standard deviation across seeds is 0.0074, which is low. As a rule of thumb, a
standard deviation below 0.010 is taken as a sign of a stable configuration, and this
result clears that bar. The mean mAP of 0.8676 with a 95% confidence
interval of [0.8529, 0.8824] gives a reliable picture of what the model can be expected
to achieve on average.

Seed=0 gives the highest individual result at 0.8818 and was used for the Kaggle
submission. Since it is the best out of 7 runs it is naturally going to be above the
mean, but the tight confidence interval suggests the submitted result is not far from
what the model clearly achieves on average.

Training converged consistently across all seeds, with best epochs ranging from 31 to
41. No seed ran all the way to epoch 50 without improving, which confirms the training
schedule is well matched to this configuration.

The overall mean gain of +0.0229 over Exp 4 can be attributed to the ArcFace
hyperparameter tuning done in Experiment 13, and this experiment confirms that gain
is stable and repeatable across different random conditions.

---
