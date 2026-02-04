# CausalX — Video Presentation Slides (20 min)

> **Format**: Each numbered section below is one slide.
> **Timing**: ~1.5–2 minutes per slide.

---

## 1) Title Slide
- **Project Title**: CausalX — Causal Audio‑Visual Deepfake Detection
- **Student Name**: [Your Name]
- **UoW / IIT ID**: [ID]
- **Supervisor**: [Supervisor Name]
- **Date**: [Presentation Date]

---

## 2) Agenda (Point Form)
- Problem background & motivation
- Research problem & research gap
- Stakeholders (Onion diagram)
- Requirements (functional + non‑functional) & implementation status
- System design (OOAD + design goals)
- Architecture & UI wireframes
- Updated schedule (Gantt) & progress since PPRS
- Conclusion & references

---

## 3) Problem Background
- Deepfake videos are increasingly realistic, causing **misinformation**, **fraud**, and **reputational harm**.
- Existing detection tools often lack **explainability**, making results hard to trust in forensic settings.
- The key real‑world challenge: **audio‑visual inconsistencies** (e.g., lip‑sync) are hard to detect robustly across varying codecs and noise.
- **Visual Aid**: Insert chart showing growth of deepfake incidents or dataset size trends.

---

## 4) Research Problem & Research Gap
- **Research Problem**: Build a system that detects AV deepfakes **and explains why** by identifying causal audio‑visual mismatches.
- **Existing Work (examples)**:
  - Lip‑sync detection models (e.g., SyncNet‑style approaches)
  - AV representation models (e.g., AV‑HuBERT)
  - Deepfake benchmarks (e.g., DFDC, FakeAVCeleb)
- **Gap**: Most methods **output a label only** and do not provide **causal evidence** (e.g., mismatch timestamps or causal segments).
- **Goal**: Combine **fusion detection** with **causal explanations** (mismatch segments + bounding boxes).

---

## 5) Project Stakeholders (Onion Diagram)
- **Core**: End users (investigators, journalists, platform moderators)
- **Supporting**: ML engineers, data engineers
- **Wider**: Legal teams, social media platforms, policy makers
- **Visual**: Insert Onion Diagram with roles + brief descriptions

---

## 6) Formal Requirements (Functional)
- **FR1**: Upload a video for analysis (implemented)
- **FR2**: Run AV deepfake inference and return label (implemented)
- **FR3**: Provide confidence score and highlight timestamps (implemented)
- **FR4**: Visualize causal breach segments + bounding boxes (implemented)
- **FR5**: Show metrics summary to the user (implemented)
- **FR6**: Async processing for long videos (implemented in API)

---

## 7) Formal Requirements (Non‑Functional)
- **Performance**: Process typical video within acceptable latency (target < X minutes)
- **Explainability**: Provide interpretable causal evidence (implemented)
- **Scalability**: API supports queued jobs (implemented)
- **Usability**: Simple upload + clear results UI (implemented)
- **Security**: Validate file type; isolate uploads (partial — needs hardening)

---

## 8) System Design (One Slide)
- **Design Goals**:
  - High‑precision detection
  - Explainable outputs (causal segments + timestamps)
  - Modular pipeline (preprocessing → model → inference → UI)
- **OOAD Methodology**:
  - Classes/Modules: Preprocessing, CFN Model, Inference Service, API Controller, UI components
- **Visual**: Insert UML class diagram (high‑level modules)

---

## 9) Overall System Architecture
- **Frontend (React/Vite)**
  - Video upload UI
  - Result visualization (breach segments, bounding boxes)
- **Backend (FastAPI)**
  - `/analyze` & `/analyze/async` endpoints
  - Background worker for queued jobs
- **ML Pipeline**
  - Preprocessing + feature extraction
  - CFN model inference
  - Smoothing & causal segment extraction
- **Visuals to include**:
  - High‑level architecture diagram
  - Low‑level design (sequence or pipeline diagram)
  - Wireframes (upload + results screen)

---

## 10) Updated Time Schedule (Gantt)
- Insert **Gantt chart** from Chapter 8 (IPD thesis template)
- **Highlight changes**:
  - Dataset preprocessing took longer than expected (reason: data quality issues)
  - Model tuning improved accuracy (added time for experiments)
  - Frontend integration completed earlier than planned
- **Explain why actuals differ** and mitigation steps

---

## 11) Progress Since PPRS (One Slide)
- ✅ Implemented FastAPI inference service with async support
- ✅ Implemented CFN inference pipeline (smoothing, causal segments)
- ✅ Built React UI for upload + results visualization
- ✅ Added bounding box overlays + breach indicators
- ⏳ Remaining work: finalize evaluation, improve security checks, polish documentation

---

## 12) Conclusion (One Slide)
- **CausalX delivers** AV deepfake detection **with explainable evidence**
- Working end‑to‑end prototype from upload → inference → UI visualization
- Next steps: extend evaluation dataset, optimize latency, finalize thesis write‑up

---

## 13) References (Alphabetical)
- AV‑HuBERT (Audio‑Visual Speech Representation)
- DFDC (DeepFake Detection Challenge dataset)
- FakeAVCeleb (Deepfake AV dataset)
- SyncNet (Audio‑visual synchronization)
- Wav2Lip (Lip‑sync generation)

---

# Speaker Notes (Optional)
- Keep each slide focused on 3–5 bullet points.
- Use visuals wherever possible (architecture diagram, onion diagram, Gantt chart, UI screenshots).
- Aim for **~20 minutes total**.
