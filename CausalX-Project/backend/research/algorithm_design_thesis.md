# Algorithm Design for Thesis: Explainable-by-Design Deepfake Detection (CausalX)

## 1) Problem formulation
Given an input video \(V\), the goal is to determine whether it is **real** or **fake** by detecting **causal inconsistencies** between speech audio and lip motion, while also producing interpretable evidence (timestamps and face/mouth bounding boxes).

The system combines:
1. **CFN (Causal Fusion Network):** predicts frame-level fake probability from multimodal causal features.
2. **CVI (Causal Visualization Interface):** converts model outputs into explainable artifacts (highlight timestamps, causal segments, spatial bounding boxes).

---

## 2) Core idea (causal hypothesis)
Natural talking-head videos exhibit a consistent causal relation:

\[
\text{Audio speech dynamics} \rightarrow \text{Lip motion dynamics}
\]

Deepfake synthesis often breaks this relation due to generation artifacts, temporal lag, or poor lip-sync. CausalX models this breach using:
- **Audio-visual mismatch** (local lip-audio correlation breakdown), and
- **Physical instability cues** (landmark jitter statistics).

A frame is suspicious if these indicators cause high CFN output probability.

---

## 3) End-to-end inference algorithm (for thesis pseudocode)

### Algorithm 1: CausalX Inference Pipeline

```text
Input:
  V                : input video
  τ_prob           : frame fake-probability threshold (e.g., 0.6)
  τ_ratio          : video-level suspicious-frame ratio threshold (e.g., 0.3)
  τ_causal         : causal-break threshold on AV mismatch (e.g., 0.6)
  W_smooth         : temporal smoothing window for probabilities
  T_chunk          : chunk duration in seconds
  T_max (optional) : maximum processed duration

Output:
  y_video          : video label (0=real, 1=fake)
  s_video          : overall video fake score
  H                : highlight timestamps
  S                : causal segments [start, end]
  F                : per-frame records (probability, mismatch, bbox)

Procedure CausalX_Infer(V):
1.  (fps, duration) ← GetVideoMeta(V)
2.  total_duration ← min(duration, T_max) if T_max exists else duration
3.  Initialize empty list F

4.  for chunk_start from 0 to total_duration step T_chunk do
5.      frames ← ExtractFrameLevelFeatures(V, chunk_start, T_chunk, fps)
6.      if frames is empty: continue

7.      mismatch ← ComputeAVMismatch(frames)   # 1 - local corr(lip, audio)

8.      if embedding mode enabled then
9.          v_emb ← VisualEmbedding(mean lip signal in chunk)
10.         a_emb ← AudioEmbedding(chunk waveform)
11.     end if

12.     for each frame i in frames do
13.         av_features ← [lip_aperture_i, mismatch_i, optional embeddings]
14.         phys_features ← [jitter_i, jitter_std_i]
15.         (optional) standardize av_features and phys_features

16.         p_i ← CFN(av_features, phys_features)   # sigmoid probability
17.         if p_i ≥ τ_prob OR mismatch_i ≥ τ_causal then
18.             bbox_i ← MouthBBoxFromLandmarks(frame_i) else FaceBBox(frame_i)
19.         else
20.             bbox_i ← null
21.         end if

22.         append {timestamp_i, p_i, mismatch_i, bbox_i} to F
23.     end for
24. end for

25. F ← MovingAverageSmooth(F, key="fake_prob", window=W_smooth)
26. Mark causal_break_i = 1 if mismatch_i ≥ τ_causal else 0
27. S ← BuildContiguousSegments(timestamps where causal_break_i=1)

28. suspicious ← {i | F[i].prob_smooth ≥ τ_prob}
29. r_fake ← |suspicious| / |F|
30. y_video ← 1 if r_fake ≥ τ_ratio else 0
31. H ← timestamps(suspicious) if y_video=1 else []
32. s_video ← mean(F.prob_smooth)

33. return y_video, s_video, H, S, F
```

---

## 4) CFN neural network architecture

### 4.1 Base CFN (current default)
Two-branch MLP with learnable causal fusion:

- **AV branch input (3D):**
  - lip aperture (frame-level mouth opening signal)
  - AV mismatch score
  - reserved scalar (0.0 in base mode)
- **Physical branch input (2D):**
  - jitter
  - jitter standard deviation

Branch networks:
- AV: `Linear(3→8) → ReLU → Linear(8→4)`
- Physical: `Linear(2→8) → ReLU → Linear(8→4)`

Learnable fusion:
\[
z = \alpha \cdot z_{AV} + \beta \cdot z_{Phys}
\]

Classifier:
- `Linear(4→1) → Sigmoid`
- Output: frame fake probability \(p \in [0,1]\)

### 4.2 Embedding-aware CFN-V2 (optional)
When enabled, AV input expands (e.g., adding TCN visual embedding and Wav2Vec2 audio embedding summaries):
- AV branch: `Linear(av_dim→16) → ReLU → Linear(16→8)`
- Physical branch: `Linear(phys_dim→8) → ReLU → Linear(8→8)`
- Fusion: same \(\alpha, \beta\) weighted sum
- Classifier: `Linear(8→1) → Sigmoid`

---

## 5) Training algorithm (for thesis section)

### Algorithm 2: CFN Training

```text
Input:
  D = {(x_av, x_phys, y)}
  epochs, batch_size, lr, weight_decay, patience, val_split

Output:
  θ*            : best model parameters
  scaler (opt.) : feature standardizers

Procedure Train_CFN(D):
1.  Split D into train/validation using stratified split
2.  (optional) Fit standard scalers on train AV and physical features
3.  Compute class-balanced sample weights for imbalanced labels
4.  Initialize CFN (or CFN-V2), BCELoss, AdamW optimizer
5.  Initialize ReduceLROnPlateau scheduler on validation AUC

6.  best_auc ← -∞ ; no_improve ← 0
7.  for epoch = 1..epochs do
8.      Train one epoch on mini-batches
9.      Evaluate on validation set: loss, accuracy, ROC-AUC
10.     Sweep thresholds to estimate PR-AUC and best F1 threshold
11.     Update LR scheduler using validation AUC

12.     if val_auc > best_auc then
13.         best_auc ← val_auc
14.         Save model checkpoint (and scaler if used)
15.         no_improve ← 0
16.     else
17.         no_improve ← no_improve + 1
18.     end if

19.     if no_improve ≥ patience then break   # early stopping
20. end for

21. return best checkpoint
```

---

## 6) Suggested thesis flow-chart blocks
You may draw this as a standard flow chart:

1. **Input video**
2. **Chunking + multimodal extraction** (face landmarks, lip aperture, audio RMS)
3. **Causal feature computation** (AV mismatch + jitter metrics)
4. **CFN inference (frame-level probability)**
5. **Post-processing** (smoothing, causal-break tagging)
6. **Temporal grouping** (causal segments)
7. **Video-level decision** (ratio rule)
8. **Explainable output** (timestamps + bbox overlays + confidence)

---

## 7) Why this design is explainable-by-design
- **Causal basis:** decision is tied to interpretable causal signal (lip-audio coherence).
- **Transparent fusion:** AV vs physical contributions are explicitly weighted by learnable \(\alpha,\beta\).
- **Human-auditable outputs:** each frame stores probability, mismatch score, and localized region.
- **Temporal reasoning:** suspicious frames are aggregated into contiguous causal segments, not only a single binary label.

This directly supports thesis claims on interpretability and trustworthiness in practical deepfake detection.
