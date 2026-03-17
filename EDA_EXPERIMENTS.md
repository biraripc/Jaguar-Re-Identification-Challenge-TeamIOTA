# EDA EXPERIMENTS

---

## Experiment 8 — Background Reliance Analysis for Jaguar Re-Identification

**W&B Run:** [background-reliance](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/o39g7kn5)  
**Notebook:** [Experiment_8.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_8/experiment-8-background-reliance.ipynb)  

---

### Research Question

Does the MegaDescriptor-L-384 + ArcFace model identify jaguars by looking at coat
pattern features (spots, rosettes, body shape), or does it rely on background context
(camera-trap location, habitat) that happens to be correlated with individual identity
in this dataset?

---

### Hypothesis

**H₁ (coat-driven):** Foreground-only mAP is close to the original mAP, and
background-only mAP is close to random (1/31 = 0.032). The model uses visual coat
features to identify individuals.

**H₂ (background-shortcut):** Background-only mAP stays high even after the jaguar
is removed. The model is mostly exploiting location and habitat patterns.

**H₃ (mixed):** Both foreground and background contribute, with both conditions
scoring above random but below the original. The model uses a combination of both.

---

### Intervention

**Masking method:** Gradient saliency maps are computed from the trained ArcFace model
using input × ∂loss/∂input, smoothed with a Gaussian filter (σ=5.0), and thresholded
at the top-50% most salient pixels to produce a binary foreground/background mask.

Three versions of each validation image are created:

| Condition | Definition | Interpretation |
|---|---|---|
| **Original** | Unmodified image | Full model performance |
| **Foreground-only** | Background pixels replaced with ImageNet channel mean (0.485, 0.456, 0.406) | Coat/body features only |
| **Background-only** | Foreground pixels replaced with ImageNet channel mean | Location/habitat cues only |

The mask is based on the model's own gradient signal rather than a separate segmentation
model, so the foreground/background split reflects what the model actually pays attention
to during inference.

**Controlled variables** (identical to all prior experiments):

| Variable | Value |
|---|---|
| Backbone | MegaDescriptor-L-384 (frozen) |
| Projection head | Linear(1536→512→256) + BN + ReLU + Dropout(0.3) |
| Loss | ArcFace (margin=0.5, scale=64) |
| Optimiser | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Epochs | 50, Batch=32, Seed=42 |
| Val split | 20% stratified (all 31 identities in both sets) |

---

### Results

#### mAP by Masking Condition

| Condition | Val mAP | Δ Original | Interpretation |
|---|---|---|---|
| **Original** | **0.7755** | n/a | Baseline, full model |
| Foreground-only | 0.3570 | −0.4184 | Coat/body features alone |
| Background-only | 0.2565 | −0.5190 | Location/habitat cues alone |
| Random baseline (1/31) | 0.0323 | n/a | Theoretical chance level |

#### Background Reliance Score (BRS) and Foreground Retention Ratio (FRR)

Defined per-identity as:

BRS_i = AP_bg_i / AP_original_i (1.0 = fully background-identified)  
FRR_i = AP_fg_i / AP_original_i (1.0 = fully foreground-retained)

| Metric | Value |
|---|---|
| Mean BRS (↓ better) | 0.4563 |
| Identities with BRS > 0.30 | 15 / 31 |
| Mean FRR (↑ better) | 0.4604 |
| Identities with FRR > 0.70 | 5 / 31 |

#### Within- vs Cross-Identity Cosine Similarity Gaps

| Condition | Pos mean − Neg mean (gap) |
|---|---|
| Original | 0.6261 |
| Foreground-only | 0.2138 |
| Background-only | 0.0988 |

The near-zero gap under background-only (0.0988 vs 0.6261) shows that the background
alone carries very little discriminative signal in the embedding space. However, the
background-only mAP of 0.2565 is still 8 times above the random baseline, confirming
it does contain some usable location information.

---

### Interpretation

The model is clearly relying on background context to some degree, but it is not
exclusively doing so. Background-only mAP of 0.2565 is 8 times above random (0.0323),
which confirms the model has learned to use camera-trap location and habitat information.
Foreground-only mAP (0.3570) is higher than background-only (0.2565), which shows the
model leans more on coat features, but the margin between the two is not large.  

The full model (mAP = 0.7755) performs far better than either masked version on its own,
which means the two sources of information are working together rather than one
dominating the other. Removing either the foreground or the background causes a large
drop in performance, which confirms both are contributing to the final result.  

15 out of 31 identities have a BRS above 0.30, meaning nearly half of the jaguars in
the dataset are being partly identified through background cues rather than coat
patterns. This is likely the case for identities like Marcela, who has 183 images all
concentrated at one camera-trap site. Only 5 out of 31 identities have an FRR above
0.70, meaning only a small number of jaguars are being identified reliably from coat
patterns alone.  

The near-zero similarity gap under background-only conditions shows that when the
background is removed, same-identity images from different locations become almost
as dissimilar as images from completely different jaguars.  

One limitation of this experiment is that the saliency mask uses a fixed threshold of
the top 50% most salient pixels, which is an arbitrary choice. Using a different
threshold would produce a different mask and potentially different numbers.  

---

## Experiment 9 — Intra-Identity Embedding Variance Analysis

**W&B Run:** [intra-identity-variance](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/2ay7d8ao)  
**Notebook:** [Experiment_9.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_9/experiment-9-intra-identity-variance.ipynb)  
**Cross-reference:** LEADERBOARDEXPERIMENTS.md — Experiment 1 (ArcFace baseline model used here)

---

### Research Question

When the model fails to correctly retrieve a jaguar identity, is it because there are
too few training images for that identity, or because that jaguar looks noticeably
different across its images? And does ArcFace fine-tuning bring the embeddings of
difficult, visually inconsistent identities closer together more than it does for
easier ones?

---

### Hypotheses

**H₁:** Identities whose images are spread far apart in embedding space (high variance)
will have lower per-identity AP. The Spearman correlation between variance and AP is
expected to be strongly negative.

**H₂:** Fine-tuning will compress the embedding clusters of high-variance identities
more than those that are already tight before training.

**H₃ (outlier):** Some identities may have a tight, compact cluster but still have
low AP. This would mean the model has learned to group their images together but in
the wrong part of the embedding space, confusing them with a different jaguar.

---

### Intervention

Two sets of L2-normalised validation embeddings are compared:

| Condition | Embedding Source | Interpretation |
|---|---|---|
| **Raw** | Frozen MegaDescriptor-L-384, no fine-tuning | What the pre-trained backbone sees |
| **Fine-tuned** | Best ArcFace projection head (epoch 48) | What the trained model sees |

Intra-identity variance is measured as the mean pairwise cosine distance between all
images of the same jaguar in the validation set:

variance_i = mean over all pairs (a,b) in identity i of (1 - cos_sim(emb_a, emb_b))

A higher variance means the images of that jaguar are spread further apart in embedding
space, which typically means the model sees them as visually inconsistent.

**Controlled variables** (identical to all prior experiments):

| Variable | Value |
|---|---|
| Backbone | MegaDescriptor-L-384 (frozen) |
| Projection head | Linear(1536→512→256) + BN + ReLU + Dropout(0.3) |
| Loss | ArcFace (margin=0.5, scale=64) |
| Optimiser | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Epochs | 50, Batch=32, Seed=42 |
| Val split | 20% stratified (all 31 identities in both sets) |

Best checkpoint: epoch 48, val mAP = 0.7755

---

### Results

#### Overall mAP and Variance

| Condition | mAP | Mean Intra-ID Variance | Min Var | Max Var |
|---|---|---|---|---|
| **Raw MegaDescriptor** | 0.3531 | 0.6521 | 0.2374 | 0.9030 |
| **Fine-tuned (ArcFace)** | 0.7755 | 0.4673 | 0.1275 | 0.9716 |
| **Delta** | **+0.4224** | **−0.1847** | n/a | n/a |

Fine-tuning reduces mean intra-identity variance by 0.1847 while increasing mAP by
0.4224. Four identities out of 31 actually became slightly more spread out after
fine-tuning, with Ipepo being the most notable case (variance increased by 0.1267).
The largest cluster compression was seen in Abril, whose variance dropped by 0.4076.  

#### Spearman Correlation: Variance vs Per-Identity AP

| Condition | Spearman ρ | p-value |
|---|---|---|
| Raw MegaDescriptor | −0.8762 | 0.0000 |
| Fine-tuned (ArcFace) | **−0.9046** | 0.0000 |

Both the raw and fine-tuned embeddings show a strong, statistically significant negative
correlation between variance and AP. Identities with more spread-out embeddings
consistently score lower. Fine-tuning actually strengthens this relationship slightly
(ρ from −0.88 to −0.91), meaning the fine-tuned model's performance is even more
tightly linked to how compact an identity's cluster is.  

#### Quadrant Analysis — Four Failure Modes

Identities are split into four groups using the median variance and median AP as
thresholds:

| Quadrant | Definition | Interpretation |
|---|---|---|
| **Easy** | Low variance, High AP | Compact, distinctive cluster; model handles these well |
| **Hard-scattered** | High variance, Low AP | Visually inconsistent identity; model fails due to appearance spread |
| **Surprising-hard** | Low variance, Low AP | Tight cluster but in the wrong place; confused with another identity |
| **Surprising-easy** | High variance, High AP | Scattered but still distinctive enough to retrieve correctly |  

#### Surprising-Hard Identities: Tight but Wrong Clusters

Three identities fall into the surprising-hard group (low variance, below-median AP):

| Identity | Intra-ID Var (ft) | Per-ID AP (ft) | Most confused with | Confusion rate |
|---|---|---|---|---|
| Bagua | 0.4273 | 0.8641* | Madalena | 8.3% |
| Overa | 0.3569 | 0.8732* | Saseka | 8.3% |
| Pyte | 0.3748 | 0.7496 | Ti | 14.3% |  

*These AP values are relatively high in absolute terms but fall below the median
across all identities, which is why they land in the surprising-hard group.

---

### Interpretation

The main finding is that how spread out an identity's images are in embedding space is
the strongest predictor of whether the model will retrieve that identity correctly
(Spearman ρ = −0.91 after fine-tuning, p < 0.0001). This means retrieval failures
are primarily caused by visual inconsistency within an identity across its images,
not simply by having fewer training examples.  

Fine-tuning brings embedding clusters closer together on average, which explains most
of the +0.4224 mAP gain. However, it does not fix every difficult case. Ipepo is the
clearest example of a failure: its cluster actually expands after fine-tuning and its
AP drops to 0.0119. With only 3 validation images, there is simply not enough data
for the model to form a stable cluster.  

The surprising-hard identities (Bagua, Pyte, Overa) are the most interesting cases
because they do not fit the usual pattern. Their clusters are compact after fine-tuning,
yet their AP is still below average. This points to a situation where the model has
grouped their images together correctly but placed that cluster too close to a different
jaguar's cluster in embedding space. Pyte is retrieved as Ti 14.3% of the time despite
having a tight cluster, which suggests the two jaguars look genuinely similar rather
than that Pyte's images are scattered.  

---

## Experiment 15 — Class Imbalance vs Per-Identity AP Correlation

**W&B Run:** [class-imbalance-ap-correlation](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/v8prnm5o)  
**Notebook:** [Experiment_15.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_15/experiment-15-class-imbalance-ap.ipynb)

---

### Research Question

Does the number of training images per identity predict how well the model retrieves
that identity, or does how visually distinctive a jaguar looks matter more than how
many photos it has?

---

### Hypothesis

The Spearman correlation between training image count and per-identity AP was expected
to be positive but moderate (ρ = 0.3 to 0.6), meaning more training data would
partially explain better retrieval performance. However, the visual distinctiveness
of each jaguar's coat pattern was expected to be a confounding factor, meaning some
jaguars with very few training images might actually outperform jaguars with many
more images if their coat pattern is easier to distinguish.

---

### Configuration

All settings are identical to the best configuration from Experiment 13 (seed=42),
ensuring the results are directly comparable with Experiments 13 and 14.

| Parameter | Value |
|---|---|
| Backbone | DINOv2-ViT-L/14 (frozen) |
| margin | 0.6 |
| scale | 48.0 |
| embdim | 512 |
| hiddendim | 1024 |
| dropout | 0.3 |
| lr / weight_decay | 1e-4 / 1e-4 |
| batch_size | 32 |
| epochs | 50, patience 5 |
| Val split | 20%, fixed seed=42 → 379 val images |
| Random seed | 42 |

---

### Protocol

1. Train ArcFace head on DINOv2-ViT-L/14 cached embeddings using the Exp 13 best config.
2. Reload best checkpoint (highest val mAP epoch).
3. Compute per-identity AP for all 31 identities on the fixed val set using identity-balanced mAP logic.
4. Cross-reference training counts from `train.csv` (image count per identity in the 80% train split).
5. Spearman rank correlation test between training count and per-identity AP (n = 31).
6. Quadrant analysis: split identities by median train count (36) and median AP (0.9926) into four groups.
7. Outlier visualisation: display val images from the top Efficient and Ambiguous identities to visually verify whether coat-pattern distinctiveness explains the outliers.

---

### Dataset Statistics

| Metric | Value |
|---|---|
| Total images | 1895 |
| Unique identities | 31 |
| Training split | 1516 images (80%) |
| Validation split | 379 images (20%) |
| Min identity count (total) | 13 (Ipepo) |
| Max identity count (total) | 183 (Marcela) |
| Mean identity count (total) | 61.1 |
| Train count range (after split) | 10 to 147 |

---

### Results

#### Model Performance

| Metric | Value |
|---|---|
| Best val mAP | **0.8676** |
| Best epoch | 27 |

*(Matches Exp 14 seed=42 result, confirming reproducibility across independent runs.)*

#### Spearman Correlation: Training Count vs Per-Identity AP

| Metric | Value |
|---|---|
| Spearman ρ | 0.2828 |
| p-value | 0.1232 |
| n | 31 identities |
| Significant (p < 0.05)? | No |
| Interpretation | Visual distinctiveness, not data quantity, drives per-identity AP |

The pre-experiment hypothesis (ρ = 0.3 to 0.6, moderate correlation) was not
supported. The actual correlation is weaker (ρ = 0.28) and statistically
non-significant (p = 0.12), meaning the number of training images does not reliably
predict how well the model retrieves a given identity.

#### Quadrant Analysis

Split by: median training count = 36, median AP = 0.9926

| Quadrant | Definition | n | Interpretation |
|---|---|---|---|
| **Efficient** | Low train count, High AP | 6 | Visually distinctive coat pattern; model generalises from few examples |
| **Data-hungry** | High train count, High AP | 10 | Good performance, likely aided by both data quantity and distinctive appearance |
| **Ambiguous** | High train count, Low AP | 5 | Visually ambiguous; more data did not resolve confusion |
| **Difficult** | Low train count, Low AP | 10 | Ambiguous AND data-scarce; hardest cases in the dataset |

#### Top 3 Efficient Identities (Low Training Count, High AP)

| Identity | AP | Train N | Val N |
|---|---|---|---|
| Abril | 1.0000 | 17 | 4 |
| Akaloi | 1.0000 | 15 | 4 |
| Alira | 1.0000 | 18 | 5 |

These three identities achieve perfect AP with only 15 to 18 training images. This
strongly suggests their coat patterns are visually distinctive enough that the frozen
DINOv2 backbone already separates them well in embedding space without needing many
examples to learn from.

#### Top 3 Ambiguous Identities (High Training Count, Low AP)

| Identity | AP | Train N | Val N |
|---|---|---|---|
| Pixana | 0.6934 | 40 | 10 |
| Marcela | 0.9453 | 147 | 36 |
| Ousado | 0.9503 | 143 | 36 |

Marcela has the most training images of any identity (147) but still only achieves
AP = 0.9453. Pixana has 40 training images and achieves only AP = 0.6934. For these
identities, collecting more data has not solved the retrieval problem because the
underlying issue is that their coat patterns are hard to distinguish from other jaguars.

#### Worst 5 Performing Identities

| Identity | AP | Train N | Val N | Quadrant |
|---|---|---|---|---|
| Ipepo | 0.0922 | 10 | 3 | Difficult |
| Pollyanna | 0.3418 | 13 | 3 | Difficult |
| Bernard | 0.3429 | 10 | 3 | Difficult |
| Bororo | 0.5879 | 18 | 4 | Difficult |
| Patricia | 0.6427 | 15 | 4 | Difficult |

#### Full Per-Identity AP Table (sorted descending)

| Identity | AP | Train N | Val N | Quadrant |
|---|---|---|---|---|
| Abril | 1.0000 | 17 | 4 | Efficient |
| Akaloi | 1.0000 | 15 | 4 | Efficient |
| Alira | 1.0000 | 18 | 5 | Efficient |
| Jaju | 1.0000 | 83 | 21 | Data-hungry |
| Kamaikua | 1.0000 | 84 | 21 | Data-hungry |
| Estella | 1.0000 | 13 | 3 | Efficient |
| Guaraci | 1.0000 | 25 | 6 | Efficient |
| Katniss | 1.0000 | 50 | 13 | Data-hungry |
| Overa | 1.0000 | 50 | 12 | Data-hungry |
| Lua | 1.0000 | 96 | 24 | Data-hungry |
| Kwang | 1.0000 | 90 | 23 | Data-hungry |
| Oxum | 1.0000 | 14 | 3 | Efficient |
| Saseka | 0.9997 | 63 | 16 | Data-hungry |
| Tomas | 0.9986 | 50 | 13 | Data-hungry |
| Ti | 0.9926 | 69 | 17 | Data-hungry |
| Bagua | 0.9926 | 48 | 12 | Data-hungry |
| Medrosa | 0.9921 | 136 | 34 | Ambiguous |
| Benita | 0.9857 | 69 | 17 | Ambiguous |
| Ariely | 0.9830 | 27 | 7 | Difficult |
| Apeiara | 0.9667 | 16 | 4 | Difficult |
| Ousado | 0.9503 | 143 | 36 | Ambiguous |
| Marcela | 0.9453 | 147 | 36 | Ambiguous |
| Solar | 0.8158 | 36 | 9 | Difficult |
| Madalena | 0.7927 | 22 | 5 | Difficult |
| Pyte | 0.7803 | 29 | 7 | Difficult |
| Pixana | 0.6934 | 40 | 10 | Ambiguous |
| Patricia | 0.6427 | 15 | 4 | Difficult |
| Bororo | 0.5879 | 18 | 4 | Difficult |
| Bernard | 0.3429 | 10 | 3 | Difficult |
| Pollyanna | 0.3418 | 13 | 3 | Difficult |
| Ipepo | 0.0922 | 10 | 3 | Difficult |

---

### Analysis and Interpretation

The clearest takeaway from this experiment is that having more training images does
not reliably translate into better retrieval. The Spearman correlation between training
count and per-identity AP is only 0.28 with a p-value of 0.12, which means the result
is not statistically significant. What matters far more is how visually distinctive
the jaguar's coat pattern is.  

Six identities (Abril, Akaloi, Alira, Estella, Guaraci, Oxum) achieve perfect AP with
only 13 to 25 training images. On the other end, Ipepo achieves AP = 0.0922 with 10
images. The difference is not primarily about the number of training examples but about
how separable each jaguar is in DINOv2 embedding space.  

The Ambiguous quadrant is particularly worth noting. Marcela and Ousado are the two
most data-rich identities in the entire dataset with 147 and 143 training images
respectively, yet both fall short of the median AP threshold. Simply collecting more
images of these jaguars is unlikely to solve the problem since the bottleneck is visual
ambiguity, not data scarcity.  

The Difficult quadrant contains 10 identities and is the biggest drag on overall
performance. Identities like Ipepo (AP = 0.09), Pollyanna (0.34), and Bernard (0.34)
are directly responsible for the gap between the current mean mAP of 0.8676 and
a higher ceiling. Any method that specifically improves retrieval for these hard
identities will have the biggest impact on the  mAP score.   

The Efficient quadrant provides useful evidence that the frozen DINOv2 backbone is
already a strong feature extractor for visually distinctive jaguars. The fact that
it can achieve perfect retrieval with as few as 13 training examples confirms that the
backbone captures rich enough visual features to separate clearly distinct coat
patterns without needing large amounts of training data.  

---

## Experiment 16 — Hard Pair Confusion Analysis

**W&B Run:** [hard-pair-confusion-analysis](https://wandb.ai/pranav-birari-university-of-potsdam/jaguar-reid-iota/runs/mab3uu2z)  
**Notebook:** [experiment-16.ipynb](https://github.com/biraripc/Jaguar-Re-Identification-Challenge/blob/main/Experiment_16/experiment-16-hard-pair-confusion.ipynb)

---

### Research Question

Which jaguar identity pairs does the model most frequently mix up, and is the confusion
caused by those jaguars looking genuinely similar, or by them being photographed at the
same camera-trap locations?

---

### Hypothesis

Errors are expected to be concentrated on a small number of specific identity pairs
rather than spread evenly across all 31×31 combinations. Those pairs likely share
either similar coat patterns (similar rosette density or coat colouring) or the same
camera-trap background. Based on the findings from Experiment 15, identities with very
few training images (Ipepo, Bernard, Pollyanna) were expected to appear most frequently
in the confusion table.

---

### Defined Intervention

| Component | Exp 13 Best Config | Experiment 16 |
|---|---|---|
| Backbone | DINOv2-ViT-L/14, frozen | unchanged |
| ArcFace margin | 0.6 | unchanged |
| ArcFace scale | 48.0 | unchanged |
| embdim | 512 | unchanged |
| hiddendim | 1024 | unchanged |
| Dropout | 0.3 | unchanged |
| lr / weight_decay | 1e-4 / 1e-4 | unchanged |
| Batch size | 32 | unchanged |
| Epochs / Patience | 50 / 5 | unchanged |
| Data split | 80/20 stratified, seed=42 | unchanged |
| What's new | n/a | 31×31 confusion matrix, heatmap, top-5 pair image galleries, per-identity confusion rate, cross-reference with Exp 15 training counts |

No changes are made to the model. The Exp 13 checkpoint is reused and the confusion
analysis is applied directly on top of the validation retrieval results.

---

### Evaluation Protocol

Val mAP is first computed to confirm the checkpoint matches the Exp 13 numbers. For
each of the 379 validation queries, all other validation images are ranked by cosine
similarity on L2-normalised embeddings and the identity of the top-1 retrieved image
is recorded. This gives a 31×31 count matrix showing how often each identity gets
confused with every other. The top-5 most confused pairs are identified and 3 images
from each pair are visually inspected to judge whether the confusion is more likely
caused by coat similarity or shared background. Per-identity confusion rate is computed
as the number of wrong top-1 retrievals divided by the total number of validation
queries for that identity, then compared against the training counts from Experiment 15.
All results are logged to W&B including the confusion heatmap, pair galleries, and the
full per-identity table.

Baseline: Exp 13 (mAP = 0.8631), Exp 15 per-identity AP results.

---

### Results

| Metric | Value |
|---|---|
| Val mAP | **0.8604** |
| Best epoch | 43 |
| Top-1 accuracy | **96.0%** (364 / 379 queries correct) |
| Distinct pairs with at least one confusion | 15 |
| Total confusion events | 15 |

#### Top-5 Confused Pairs

| Rank | Identity A | Identity B | Count | Confusion rate (A) |
|---|---|---|---|---|
| 1 | Bernard | Overa | 1 | 0.33 |
| 2 | Bororo | Marcela | 1 | 0.25 |
| 3 | Bororo | Medrosa | 1 | 0.25 |
| 4 | Ipepo | Ariely | 1 | 0.33 |
| 5 | Ipepo | Ousado | 1 | 0.33 |

#### Per-Identity Confusion Rate (worst and best)

| | Identity | Confusion rate | Train images | Val count |
|---|---|---|---|---|
| **Worst** | Ipepo | 1.00 | 10 | 3 |
| | Bororo | 0.50 | 18 | 4 |
| | Bernard | 0.33 | 10 | 3 |
| **Best** | Saseka | 0.00 | 63 | 16 |
| | Ti | 0.00 | 69 | 17 |
| | Tomas | 0.00 | 50 | 13 |

**W&B artifacts:** confusion heatmap, Rank 1 to 5 pair galleries, confusion rate bar
and scatter plots, per-identity table, best checkpoint saved as artifact.

---

### Interpretation

The model gets the top-1 retrieval correct 96% of the time, which is a strong overall
result. However, the 4% of errors are not randomly distributed. All 15 error events
fall on 15 different identity pairs, with each pair having exactly one confusion,
which confirms that mistakes are concentrated on specific hard cases rather than
occurring broadly across the dataset.  

Every identity in the worst-performing group (Ipepo, Bororo, Bernard) has 18 or fewer
training images. Ipepo is the most extreme case, with a 100% confusion rate meaning
all 3 of its validation images get retrieved as the wrong jaguar, either Ariely or
Ousado. This is consistent with the near-zero AP of 0.092 reported in Experiment 15.
Bororo is a different kind of failure: it gets confused with two different identities
(Marcela and Medrosa), which suggests its embedding does not sit near any single
identity cluster but somewhere in between, without a well-defined position in the
embedding space.  

Looking at the gallery images for the top confused pairs, Bernard–Overa and
Ipepo–Ariely appear to be cases of clear visual similarity, with similar coat
texture. For Bororo–Marcela and Bororo–Medrosa, it is harder to
tell without GPS data and both coat similarity and shared camera-trap location are
plausible explanations.  

---