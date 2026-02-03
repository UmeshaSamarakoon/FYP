# AV deepfake detection with causal links — technique inventory for CausalX

## Scope and intent
This document compiles a **technique inventory** for **audio-visual (AV) deepfake detection with causal links**, spanning the full pipeline: data extraction → preprocessing → multi-modal fusion → causal explainability → decision. The focus is on candidate techniques to inform the **CausalX** project design decisions.

> **Note on Semantic Scholar access:** I attempted to query the Semantic Scholar API from the environment, but outbound access was blocked by a proxy (HTTP 403). As a result, the inventory below is derived from well-known AV deepfake detection literature and standard multimodal/causal ML practices rather than direct API pulls. This inventory is still structured to mirror the typical taxonomy of Semantic Scholar results.

---

## 1. Data extraction techniques (input capture & signal isolation)

### 1.1 Video extraction
- **Face detection**: MTCNN, RetinaFace, BlazeFace, S3FD; used to localize face regions for per-frame processing.
- **Face tracking**: SORT/DeepSORT, KLT, or landmark tracking to preserve identity across frames.
- **Landmark detection**: dlib 68-point, MediaPipe Face Mesh, HRNet-based landmark detectors.
- **ROI cropping**: lip region, lower face, or full-face crops depending on speech-driven coherence tasks.

### 1.2 Audio extraction
- **Audio stream demux**: ffmpeg for waveform extraction.
- **Voice activity detection (VAD)**: WebRTC VAD or pyannote for removing silence/non-speech.
- **Speaker diarization**: if multi-speaker sources are present (helps with speaker-specific modeling).

### 1.3 Temporal alignment
- **A/V synchronization**: timestamp-based alignment; resampling to fixed FPS and audio sample rates.
- **Lip-sync windows**: sliding temporal windows (e.g., 0.2–1.0s) to capture phoneme-to-viseme correspondence.

---

## 2. Preprocessing techniques (signal conditioning & representation)

### 2.1 Video preprocessing
- **Face alignment**: similarity transform via landmarks (eyes/nose), stabilizes motion.
- **Frame normalization**: resize to 224×224 or 112×112; per-channel mean/variance normalization.
- **Optical flow**: flow fields (e.g., RAFT, Farnebäck) for subtle motion artifacts.
- **Temporal smoothing**: remove jitter/irregular motion artifacts.
- **Artifacts emphasis**: error-level analysis (ELA), frequency-domain bandpass to highlight GAN artifacts.

### 2.2 Audio preprocessing
- **Spectrograms**: log-mel spectrograms, MFCCs, STFT magnitude.
- **Self-supervised embeddings**: wav2vec 2.0, HuBERT, WavLM for robust speech representations.
- **Noise normalization**: spectral gating, pre-emphasis filtering.

### 2.3 Cross-modal preprocessing
- **Phoneme/viseme alignment**: align phonetic units to visual lip movements.
- **Temporal resampling**: audio feature downsampling to match video FPS.
- **Masking**: missing-modality masking for robustness and causal ablation tests.

---

## 3. Multi-modal fusion techniques (audio + video integration)

### 3.1 Early fusion
- **Feature concatenation**: combine audio and video embeddings before joint encoder.
- **Co-embedding networks**: shared backbone for audio/video with shared projection head.

### 3.2 Late fusion
- **Decision-level fusion**: audio and video classifiers fused by weighted averaging or stacking.
- **Ensemble stacking**: meta-classifier on modality-specific logits.

### 3.3 Mid-level / cross-modal fusion
- **Cross-attention**: video queries audio keys/values (or bidirectional), capturing alignment.
- **Co-attention**: mutual attention between modalities to learn correlated features.
- **FiLM conditioning**: audio-conditioned modulation of visual feature maps.
- **Gated fusion**: learned gates to weigh modality contributions.
- **Bilinear pooling**: multiplicative interactions (e.g., MCB, MUTAN).
- **Graph fusion**: graph-based nodes for audio/video tokens with message passing.

### 3.4 Temporal fusion
- **Transformer encoders**: modality-specific tokens merged across time.
- **Temporal convolution**: TCN or 1D conv over fused sequences.
- **LSTM/GRU**: sequential fusion of fused features.

### 3.5 Causal fusion (preferred for CausalX)
- **SCM-guided fusion**: explicit causal links between audio/video cues (e.g., audio → sync → decision; video → artifacts → decision).
- **Causal attention**: attention weights constrained by causal graph edges to favor causal pathways.
- **Interventional fusion**: train with modality interventions (swap, mask, mismatch) to enforce causal consistency.
- **Invariant fusion**: IRM-style objectives that keep fusion features stable across confounding shifts (codec, noise, resolution).

---

## 4. Causal explainability techniques (discovering/validating causal links)

### 4.1 Causal structure modeling
- **Structural causal models (SCM)**: explicit nodes for audio, video, sync coherence, artifacts, identity, etc.
- **Causal graphs**: directed graph modeling dependencies between modalities and artifact cues.
- **Causal discovery**: PC, GES, NOTEARS on extracted features to infer relationships.

### 4.2 Causal intervention & counterfactuals
- **Do-operator ablation**: intervention by removing or swapping modalities to observe effect.
- **Counterfactual lip-sync**: replace audio with mismatched speech; check prediction shift.
- **Causal feature masking**: remove suspected confounders (background noise, compression).

### 4.3 Causal attribution & explainability
- **Causal SHAP**: SHAP with structural constraints.
- **Path-specific effects**: quantify effects of audio on decision through sync-related nodes.
- **Mediation analysis**: quantify how audio impacts prediction via visual mismatch.

### 4.4 Robustness & confounder control
- **Invariant Risk Minimization (IRM)**: enforce causal feature reliance across domains.
- **Domain generalization**: train across datasets to reduce spurious cues.
- **Adversarial debiasing**: remove confounding signals (compression, resolution).

---

## 5. Decision-stage techniques (classification & calibration)

### 5.1 Classifiers
- **Binary classifier**: deepfake vs real; softmax or sigmoid output.
- **Temporal aggregation**: majority vote, attention pooling, or max pooling across frames.
- **Confidence calibration**: temperature scaling or isotonic regression.

### 5.2 Explanation-aware decision
- **Causal consistency checks**: require decision to match learned causal alignment score.
- **Reject option**: abstain when causal evidence is weak or contradictory.

---

## 6. Candidate techniques for CausalX (end-to-end inventory)

### 6.1 Data extraction & preprocessing
- Face detection: **RetinaFace / MTCNN / MediaPipe**
- Landmarks & alignment: **MediaPipe Face Mesh / dlib 68 / HRNet**
- Lip ROI extraction: **mouth landmarks + stabilized crop**
- Audio extraction: **ffmpeg + VAD (WebRTC)**
- Audio features: **log-mel, MFCC, wav2vec 2.0 embeddings**
- Temporal alignment: **fixed window lip-sync segments (0.5–1s)**

### 6.2 Fusion models (causal fusion emphasis)
- **SCM-guided causal fusion** with explicit sync and artifact nodes
- **Causal attention** constrained by a causal graph
- **Interventional fusion** (swap/mask modalities during training)
- **Invariant fusion** (IRM-style objectives to reduce confounders)

### 6.3 Causal modeling & explainability
- **SCM-based decision node**: audio → sync → decision, video → artifacts → decision
- **Causal discovery** (PC, GES) on engineered features
- **Intervention tests**: swap audio or shuffle temporal order
- **Mediation analysis**: quantify if sync mismatch mediates fake prediction
- **IRM / domain generalization** to reduce compression confounds

### 6.4 Decision logic
- **Hybrid score** = detection probability × causal-consistency score
- **Abstain** if causal evidence conflicts with classifier score

---

## 7. Practical shortlist for CausalX

**Recommended starting pipeline**
1. **Extraction**: RetinaFace + lip ROI + ffmpeg audio.
2. **Preprocessing**: log-mel + wav2vec 2.0, aligned to 25 FPS video.
3. **Fusion**: **causal fusion** (SCM-guided causal attention + interventional training).
4. **Causal module**: SCM with explicit sync node; use do-interventions via modality swapping.
5. **Decision**: hybrid prediction + causal consistency score.

---

## 8. Next steps (when Semantic Scholar access is restored)
1. Query **Semantic Scholar** for: "audio-visual deepfake detection", "lip-sync detection", "multimodal deepfake detection".
2. Map each paper to the technique categories above.
3. Update this document with citations and direct evidence from top papers.
4. Add a **benchmark notebook** to evaluate candidate techniques per phase and record results.

### 8.1 Proposed notebook experiment plan
Create a notebook (e.g., `backend/notebooks/technique_benchmark.ipynb`) that:
- **Phase A (Extraction)**: compare face detectors/landmarkers (RetinaFace, MTCNN, MediaPipe) using detection rate and landmark stability metrics.
- **Phase B (Preprocessing)**: compare audio features (log-mel, MFCC, wav2vec 2.0) and video features (raw frames vs optical flow) using validation accuracy or AUC.
- **Phase C (Fusion)**: compare causal fusion variants (SCM-guided, causal attention, interventional) against baseline fusion (cross-attention/gated) and track robustness under modality swaps.
- **Phase D (Causal explainability)**: quantify mediation/path-specific effects and verify intervention sensitivity (do-audio, do-video).
- **Decision tracking**: log metrics and artifacts in a single table (CSV/JSON) with per-technique scores and confidence intervals.

---

## 9. Known AV deepfake detection anchors (non-exhaustive)
The following commonly-cited works can serve as anchors when updating with citations:
- **SyncNet / lip-sync detection** (audio-video sync mismatch)
- **Wav2Lip** (audio-driven lip generation; good for counterfactual experiments)
- **AVSR / lipreading models** (LipNet, AV-HuBERT) that provide aligned AV features
- **Deepfake detection benchmarks**: DFDC, FakeAVCeleb, FakeAVSpeaker

---

## 10. Deliverable summary for CausalX
This inventory lists candidate techniques from extraction through causal decision. It provides a **pipeline-ready shortlist** and enumerates **causal explainability methods** (SCM, mediation, interventions, IRM) that can be integrated into CausalX for robust, explainable AV deepfake detection.
